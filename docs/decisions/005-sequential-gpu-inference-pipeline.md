# ADR-005: Sequential GPU Inference Pipeline — YOLO Detect-Then-Crop + MediaPipe Pose Landmark

**Status:** Accepted  
**Date:** 2026-04-07  
**Deciders:** CourtVision core team  
**Replaces:** Frame-scheduling-spec Worker A / Worker B two-channel architecture

---

## Context

CourtVision's Day 3 baseline runs a single YOLOv8n TFLite model (5-class: `ball`, `made`, `person`, `rim`, `shoot`) on the GPU delegate for ball and shot event detection. The Day 6 target adds biomechanical pose analysis — specifically elbow flexion angle, wrist height relative to shoulder, and knee flexion at release — to power shot quality feedback.

The original frame-scheduling-spec proposed a two-channel architecture:

- **Worker A** — YOLO inference on every frame (GPU delegate, single-threaded executor)
- **Worker B** — MediaPipe Pose Landmarker full pipeline (CPU, every 3rd frame, ~10 FPS)

This design was acceptable as a placeholder but left three problems unresolved:

1. MediaPipe's internal person detector re-runs on every frame where tracking is lost, duplicating work that YOLO already performs.
2. Worker B's CPU-only path is estimated at 80–180 ms on the Galaxy A54 (Exynos 1380), threatening the Day 8 sub-100ms total pipeline gate.
3. Two independent channels require inter-worker coordination and produce misaligned timestamps — YOLO detections and pose landmarks may not correspond to the same frame, complicating FSM logic.

This ADR records the decision to collapse both workers into a single sequential GPU pipeline.

---

## Decision

**Run YOLO and the MediaPipe pose landmark model sequentially on the GPU delegate, on the same single-threaded coroutine dispatcher, per selected frame.**

The pipeline per frame is:

```
CameraX RGBA_8888 frame
    │
    ▼
[YOLO 5-class, 640×640, FP16, GPU delegate]
    │  NMS (Kotlin-side, pre-allocated buffers)
    │
    ├─── ball bbox  ──► Kalman tracker ──► FSM tick
    ├─── rim bbox   ──► FSM tick
    ├─── made/shoot ──► FSM tick
    │
    └─── person bbox ──► squarePadCrop(margin=1.25) ──► 256×256 Bitmap
                                                              │
                                                              ▼
                                          [Pose Landmark model, 256×256, GPU delegate]
                                                              │
                                               33 WorldLandmarks (meters, hip-origin)
                                                              │
                                                              └──► angle computation ──► FSM tick
```

Both TFLite `Interpreter` instances run on the same single-threaded dispatcher (`consumerDispatcher = Dispatchers.Default.limitedParallelism(1)`). Each `GpuDelegate` instance creates its own EGLContext by default; whether TFLite reuses the thread's ambient EGL context across two separate delegate instances is implementation-defined and was validated on S22+ (see Verification Gate §1).

---

## Alternatives Considered

### Alt A: Original Worker A / Worker B two-channel architecture (rejected)

MediaPipe's `PoseLandmarker` task runs its internal person detector on every frame after tracking loss. On the A54, the full two-stage CPU pipeline is 80–180 ms, exceeding the 50 ms per-frame pose budget. The two-channel design also requires inter-worker synchronization and produces temporally misaligned data.

**Rejected** because of latency and FSM complexity.

### Alt B: YOLO pose model on the person crop (rejected)

YOLOv8n-pose is a unified detection + keypoint model. Feeding it a pre-cropped person bbox still runs the full detection head on the crop — the detection work is wasted. The model also produces only 17 COCO keypoints, losing the foot landmarks (indices 27–32) needed for jump height estimation and the finger/thumb landmarks needed for wrist-above-shoulder detection in FSM release gating.

**Rejected** because of wasted compute and insufficient keypoint coverage.

### Alt C: MediaPipe `PoseLandmarker` task API with YOLO-provided ROI hint (rejected)

The `PoseLandmarker` Tasks API does not expose a public method to inject an external ROI — the internal person detector always runs. Bypassing it requires extracting the raw `pose_landmarks_detector.tflite` (same file name for lite, full, and heavy variants) model from the `.task` bundle and loading it directly into a standard `Interpreter`. This is the chosen approach.

### Alt D: Pose on CPU, YOLO on GPU (original Worker B fallback)

Would preserve the simpler single-interpreter GPU setup but caps pose FPS at ~5–10 on the A54 and does not converge on the Day 8 gate.

**Rejected** — kept only as the fallback path if sequential GPU verification fails on the A54.

---

## Implementation Details

### 1. Model loading

Extract `pose_landmarks_detector.tflite` from the `.task` archive at build time. Load as a standard TFLite `Interpreter` alongside the YOLO interpreter.

```kotlin
// Both initialized on the same HandlerThread / single-threaded dispatcher at app start
val yoloDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
val yoloInterpreter = Interpreter(
    loadModelFile(context, "yolov8n_fp16.tflite"),
    Interpreter.Options().addDelegate(yoloDelegate)
)

val poseDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
val poseInterpreter = Interpreter(
    loadModelFile(context, "pose_landmarks_detector.tflite"),
    Interpreter.Options().addDelegate(poseDelegate)
)
```

Both must be created and `AllocateTensors()` called on the same single-threaded dispatcher thread, not from mixed call sites.

Current implementation behavior:
- YOLO interpreter is initialized on the pipeline dispatcher via mode-switch path.
- Pose interpreter is initialized on-demand in `processImage()` via `initializePoseInterpreterIfNeeded()` on `consumerDispatcher`.

Eager pose initialization at startup remains an optimization candidate, not a Day 5 gate requirement.

### 2. Thread safety contract

The TFLite GPU delegate requires that `ModifyGraphWithDelegate()` and all subsequent `Invoke()` calls occur on the same thread within the same EGLContext. The current `consumerDispatcher = Dispatchers.Default.limitedParallelism(1)` satisfies the single-thread constraint. However, the official `GpuDelegate` Javadoc only specifies single-delegate behavior: "If an EGLContext does not exist, the delegate will internally create one." The spec is silent on what happens when a second `GpuDelegate` is created on a thread that already has a delegate-bound EGLContext. In practice, OpenGL ES ties contexts to threads, so the second delegate likely detects and reuses the existing context — but this is not guaranteed by the public API and must be verified empirically per device family (see Verification Gate §1).

```kotlin
// Correct — both .run() calls on same dispatcher, sequential
suspend fun analyzeFrame(frame: Bitmap) = withContext(consumerDispatcher) {
    // Stage 1: YOLO
    copyToInputTensor(frame, yoloInputBuffer)          // pre-allocated, no-alloc rule
    yoloInterpreter.run(yoloInputBuffer, yoloOutputs)
    val detections = decodeAndNms(yoloOutputs)         // Kotlin-side NMS

    // Stage 2: Pose (only when person detected and in SETUP or FLIGHT state)
    val personBox = detections.firstOrNull { it.classId == CLASS_PERSON } ?: return@withContext
    val crop = squarePadCrop(frame, personBox, marginFactor = 1.25f)
    val resized = Bitmap.createScaledBitmap(crop, 256, 256, false)
    copyToInputTensor(resized, poseInputBuffer)        // pre-allocated
    poseInterpreter.run(poseInputBuffer, poseOutputs)
    val landmarks = decodeLandmarks(poseOutputs)       // WorldLandmarks, meters
}
```

**What breaks the contract (do not do):**
- Initializing either interpreter on the main thread and calling `.run()` on `consumerDispatcher`
- Using two different dispatchers for the two `.run()` calls
- Calling `.run()` concurrently from any second coroutine

### 3. Person crop preprocessing

The pose landmark model expects a square, padded, normalized crop. Feeding a raw YOLO bounding box without margin degrades landmark accuracy, particularly for limbs near the box edge.

```kotlin
fun squarePadCrop(
    src: Bitmap,
    box: BoundingBox,       // normalized [0, 1]
    marginFactor: Float = 1.25f
): Bitmap {
    val W = src.width.toFloat()
    val H = src.height.toFloat()

    val cx = (box.left + box.right) / 2f * W
    val cy = (box.top + box.bottom) / 2f * H
    val halfSide = maxOf(
        (box.right - box.left) * marginFactor * W,
        (box.bottom - box.top) * marginFactor * H
    ) / 2f

    val left   = (cx - halfSide).coerceAtLeast(0f).toInt()
    val top    = (cy - halfSide).coerceAtLeast(0f).toInt()
    val right  = (cx + halfSide).coerceAtMost(W).toInt()
    val bottom = (cy + halfSide).coerceAtMost(H).toInt()

    val cropped = Bitmap.createBitmap(src, left, top, right - left, bottom - top)
    return Bitmap.createScaledBitmap(cropped, 256, 256, false)
}
```

Recommended margin: `1.25f` baseline. Increase to `1.35f` if YOLO bounding boxes are observed to clip limbs at low camera angles.

### 4. WorldLandmarks interpretation

The pose landmark model outputs two parallel lists per inference:

| Output | Coordinate space | Units | Origin |
|---|---|---|---|
| `Landmarks` | Normalized image [0, 1] | — | Top-left of crop |
| `WorldLandmarks` | 3D metric | meters | Hip midpoint |

Use **`WorldLandmarks`** for all biomechanical computations. The axis convention is:
- `x` — lateral (positive = subject's right)  
- `y` — vertical (positive = up)  
- `z` — depth (positive = behind subject, away from camera; negative = toward camera)

**Reliability by axis and use case:**

| Metric | Axis | Reliability | Notes |
|---|---|---|---|
| Elbow flexion angle at release | X, Y | ✅ Good | Ratio-based; robust to scale drift |
| Wrist height vs shoulder | Y | ✅ Good | Vertical axis is most accurate |
| Knee flexion (load at takeoff) | X, Y | ✅ Good | Same — relative joint angles work |
| Jump apex (hip Y delta) | Y | ⚠️ Noisy | Requires Kalman smoothing across frames |
| Wrist-to-camera depth | Z | ❌ Unreliable | Monocular ill-posed; defer to ARCore Phase 2 |

World coordinates are derived by fitting a statistical human body model (GHUM) to the 2D projection — not from camera geometry. Absolute accuracy is ~36–45 mm MAE per joint. Bone length variance across frames is significant (~370 mm²); the Kalman filter on top of the landmark stream is essential to suppress per-frame jitter.

**Landmark indices for CourtVision:**

| Index | Joint | Use |
|---|---|---|
| 11 | Left shoulder | Wrist-above-shoulder gate (FSM release detection) |
| 12 | Right shoulder | Wrist-above-shoulder gate (FSM release detection) |
| 13 | Left elbow | Elbow flexion |
| 14 | Right elbow | Elbow flexion |
| 15 | Left wrist | Release point |
| 16 | Right wrist | Release point |
| 23 | Left hip | Hip origin anchor |
| 24 | Right hip | Hip origin anchor |
| 25 | Left knee | Jump load |
| 26 | Right knee | Jump load |
| 27 | Left ankle | Foot tracking |
| 28 | Right ankle | Foot tracking |

### 5. No-alloc-in-analyze rule extension

The existing CONTEXT.md no-alloc rule now extends to both models. Pre-allocate all input/output buffers at initialization time:

```kotlin
// At init (once, on inference thread)
val yoloInputBuffer  = ByteBuffer.allocateDirect(1 * 3 * H_in * W_in * 4).order(ByteOrder.nativeOrder())
val poseInputBuffer  = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4).order(ByteOrder.nativeOrder())
// Output buffers sized to model output shapes
```

`Bitmap.createScaledBitmap()` in `squarePadCrop` **does allocate** — this is acceptable at ~1 ms and outside the hot tensor path. If profiling shows GC pressure, replace with a pre-allocated `Canvas`-based rescale.

---

## Consequences

### Positive

- **Latency improvement**: Estimated sequential GPU total ~31–45 ms on S22+ class, vs. ~80–180 ms CPU pose on A54. Fits Day 8 sub-100ms gate on primary reference device.
- **FSM simplification**: Ball detections and pose landmarks are now co-temporal — the FSM receives both on the same frame tick, eliminating timestamp alignment logic.
- **No mediapipe person detector duplication**: YOLO's `person` bbox replaces MediaPipe's most expensive stage (body detector at ~224×224).
- **Keypoint coverage**: 33 world-coordinate landmarks vs. 17 COCO keypoints from YOLO-pose, including foot indices 27–32 for jump height estimation.
- **Single inference thread**: Both interpreters run sequentially on the same dispatcher; EGLContext sharing behavior is verified on S22+, with cross-device verification pending (see Verification Gate §1).

### Negative / Risks

- **Pose FPS coupled to YOLO FPS**: In the old two-channel design, pose could be gated to run every 3rd frame independently. Now both models run together. Mitigate by skipping pose when FSM state is `IDLE` or `MADE` (no biomechanics needed).
- **Cold start latency**: eager startup init for both interpreters is not fully implemented yet. Current pose init is on-demand; this may shift some startup cost to first pose-use frame.
- **Peak GPU memory**: YOLO FP16 (~6 MB model) + `pose_landmarks_detector.tflite` (lite variant, ~3–4 MB) + tensor buffers must fit within the ~400 MB RAM budget. Verify on A54 with Android Profiler before shipping.
- **`.task` bundle extraction**: `pose_landmarks_detector.tflite` must be extracted from the MediaPipe `.task` archive and bundled separately in `assets/`. This is a build-time step, not a runtime concern, but must be documented in the build README.
- **Pose accuracy when person bbox is partial**: If the shooter is near the frame edge and YOLO's `person` box clips limbs, landmark accuracy degrades. Monitor via `visibility` field on each landmark; gate angle computation on `visibility > 0.6`.

### Deferred (Phase 2)

- **QAI Hub NPU compilation via QNN delegate**: The Hexagon NPU on Qualcomm SoCs is accessed through the QNN TFLite delegate (`com.qualcomm.qti:qnn-tflite-delegate` AAR + `libQnnTFLiteDelegate.so`), **not** through NNAPI. NNAPI on Qualcomm devices routes to CPU/GPU via the Android HAL — it does not reach the Hexagon NPU. The existing `InferenceMode.NNAPI` in the codebase is therefore a lateral CPU/GPU path, not a stepping stone to NPU acceleration.

  **Phase 2 delegate matrix:**

  | Config | YOLO delegate | Pose delegate | New dependency | Rationale |
  |--------|--------------|---------------|----------------|-----------|
  | A (Phase 0, this ADR) | GPU | GPU | None | Default spike path |
  | B (recommended) | QNN → NPU | GPU | QAI Hub INT8 YOLO + QNN AAR | Offloads heavy model (~25–35ms GPU → sub-5ms NPU on SD 8 Gen 2+), frees GPU for pose |
  | C (alternative) | GPU | QNN → NPU | QAI Hub INT8 pose + QNN AAR | If pose is the bottleneck instead |
  | D (fallback) | GPU | CPU (4 threads) | None | If sequential GPU fails latency gate |

  Config B is the highest-priority optimization: moving the larger model to dedicated hardware frees ~80% of the GPU frame budget. Requires w8a8 post-training quantization of the YOLO model via QAI Hub, plus accuracy validation that 5-class mAP does not regress. The QNN delegate performs on-device Hexagon graph compilation at model load time — expect increased cold-start latency (~1–3s additional).

- **ARCore depth intrinsics**: Replace monocular Z estimate with camera-calibrated back-projection for accurate wrist-to-camera distance. Required for full perspective correction of the shot arc parabola fit.
- **Physics-informed post-processing**: Bone-length-constrained Kalman smoothing (94.3% variance reduction vs. raw BlazePose world coordinates) for more stable joint angle time series.
- **Rational curve fit** for arc: Once ARCore Z is available, evaluate whether perspective-corrected parabola or rational conic section better models the observed trajectory.

---

## Verification Gate (Day 5 Spike)

Current Day 5 verification status:

1. **Thread safety & EGL context**
   - **S22+ (`SM-S906U1`)**: PASS on `consumerDispatcher` (no runtime `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread` observed).
   - **Pixel 6 / A54**: deferred (device unavailable in Day 5 run).
2. **Combined latency p95**
   - **S22+ (`SM-S906U1`)**: PASS_STRONG, `pose_inference_p95_ms=10.774` in sequential YOLO-warm mode.
   - **Pixel 6 / A54**: deferred.
3. **Landmark correctness**
   - **S22+**: PASS (manual plausibility gate >= 8/10).
4. **Peak GPU memory**
   - Deferred to Day 6+ profiling passes on target devices.
5. **No-alloc compliance**
   - Day 5 accepted with known allocation caveats; Day 6 gate remains open for full no-alloc verification.

Evidence artifacts:
- `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_summary.txt`
- `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_validation.csv`

If p95 latency on the A54 exceeds 80 ms, fall back to **Config D** (pose on CPU, 4 threads, every 3rd frame) and document as ADR-005-fallback. Note: `NnApiDelegate` is not a useful intermediate fallback — on Qualcomm SoCs it does not access the Hexagon NPU. NPU acceleration requires the QNN TFLite delegate (Phase 2, see Deferred section).

---

## Scheduling Definitions (Issue #3 Q4 / Q5)

The original frame scheduling spec answered five questions for the Worker A/Worker B design.
Q1–Q3 are fully addressed by the sequential pipeline architecture above. Q4 and Q5 are restated
here under the sequential model.

### Q4 — "Combined FPS" definition

Under the sequential pipeline, YOLO and Pose run co-temporally in one pass per selected frame on
`consumerDispatcher`. **"Combined FPS" = pipeline completions/sec** — i.e., frames where the full
YOLO pass (and Pose, if triggered) completes and results are emitted to the FSM.

Pose is conditional (skipped when FSM state is `IDLE` or `MADE`), so pipeline throughput is
YOLO-bound. The Day 8 primary gate metric remains pipeline completions/sec, consistent with the
original spec's "Worker A FPS" definition.

Pose FPS is reported separately in `PipelineStats` as a secondary diagnostic, not as a gate.

### Q5 — Thermal throttling definition

Inherited unchanged from the original spec:

> Thermal throttling is declared when pipeline FPS drops more than 20% below the rolling
> 30-second average for a sustained period of ≥ 30 seconds.

"Worker A FPS" in the original spec maps directly to "pipeline FPS" here — same dispatcher,
same frame tick. Reporting requirements are unchanged: compute rolling 30s FPS window from the
benchmark CSV; flag windows where FPS < `baseline_fps × 0.80`; pass criterion = zero throttle
events in the 10-minute measured window.

---

## References

- Google AI Edge — GPU acceleration delegate docs: `https://ai.google.dev/edge/litert/android/gpu`
- TFLite GPU delegate README (official): `tensorflow/lite/delegates/gpu/README.md`
- GitHub Issue #25657 — GPU delegate cross-thread deadlock confirmation
- GitHub Issue #46132 — GPU delegate interpreter initialization ~2.5 s
- MediaPipe Pose Landmarker Android guide: `https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/android`
- BlazePose GHUM Holistic paper: arXiv 2206.11678
- Accuracy evaluation of BlazePose world coordinates: PMC11644880
- Physics-informed pose post-processing (bone length stabilization): arXiv 2512.06783
- CourtVision CONTEXT.md — no-alloc-in-analyze rule
- ADR-001: YOLOv8n TFLite FP16 export
- ADR-002: Kotlin-side NMS
- ADR-003: CameraX RGBA_8888 path
