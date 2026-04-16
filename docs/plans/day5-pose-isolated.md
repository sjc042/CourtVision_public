# Plan: Day 5 — Pose Landmark Model: Isolated Validation

**Status:** ✅ COMPLETE (Day 5 gates passed on S22+; Day 6 follow-ups deferred)

## Context

Days 1–4 are complete. The pipeline is:
`CameraX → YOLO (5-class TFLite) → Kalman tracker → [StateFlow] → UI overlay`

Day 5 validates the pose landmark model **in isolation** before combining it with YOLO in Day 6. The goal is to confirm model loading, output correctness, and GPU latency on both target devices — and to verify the critical EGL context constraint (two `GpuDelegate` instances on one thread).

**Key architecture constraint (ADR-005):** Day 5 does NOT use the MediaPipe `PoseLandmarker` task API. Instead, `pose_landmarks_detector.tflite` is extracted from the MediaPipe `.task` bundle at build time and loaded as a raw TFLite `Interpreter` with `GpuDelegate`. This eliminates the MediaPipe internal person detector — YOLO's `person` bbox will replace it in Day 6.

> Full architecture decision: [ADR-005 — Sequential GPU Inference Pipeline](../decisions/005-sequential-gpu-inference-pipeline.md)

---

## Branch

`spike/day5-pose-isolated` (branch from `main`)

### Completion Snapshot (2026-04-14, S22+ `SM-S906U1`)
- Gate §1 EGL/thread: PASS (no `TfLiteGpuDelegate ... must run on the same thread` runtime log)
- Gate §2 visual plausibility: PASS (manual check >= 8/10)
- Gate §3 latency: PASS_STRONG (`pose_inference_p95_ms=10.774`)
- Artifacts: `benchmarks/phase0/pose_validation/inference_20260414_004536/`

---

## Prerequisites

- Day 4 complete: Kalman tracker integrated, `inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)` established ✅
- MediaPipe `.task` bundle downloaded (see Step 0)
- Test clips: at least 10 video frames with a clearly visible person for ground truth validation

---

## Design Decisions

### Standalone TFLite Interpreter — Not PoseLandmarker Task API

The MediaPipe `PoseLandmarker` task API bundles a person detector + pose landmark model and manages them internally. Its person detector re-runs on every frame after tracking loss — work that YOLO already performs. There is no public API to inject an external ROI into `PoseLandmarker`.

The chosen approach: extract the raw `pose_landmarks_detector.tflite` from the `.task` archive and load it directly as a standard `Interpreter`. This is the same approach documented in ADR-005 Alt C rationale.

### Isolated First — No Live Camera

Day 5 feeds the pose model with **static pre-cropped test frames**, not a live camera feed. Live integration is Day 6.

> ⚠️ **Revised:** Latency is no longer fully isolated — validation piggybacks on the live camera
> frame loop (YOLO runs on every frame before pose). Measured `pose_inference_ms` reflects a
> GPU-warm sequential pipeline, which is closer to the Day 6 production scenario. See Step 3.5.

### Two `GpuDelegate` Instances on One Thread

The existing `yoloInterpreter` uses a `GpuDelegate`. Day 5 adds a second `GpuDelegate` for `poseInterpreter` on the same single-threaded `consumerDispatcher`. Whether TFLite reuses the thread's ambient EGLContext for the second delegate is implementation-defined and must be confirmed empirically (see Step 5 and Verification Gate).

### Validation Runner: Integrated into `consumerDispatcher` (replaces standalone runner)

> **Old (implemented but revised):** `CameraViewModel.runDay5PoseIsolatedValidation()` launched a
> separate `poseValidationDispatcher = Dispatchers.Default.limitedParallelism(1)` and constructed
> a new `PoseLandmarkInterpreter` (third `GpuDelegate`) inside the runner. Images read via raw
> `File` I/O from `/sdcard/Download/pose_isolate`.
>
> **New:** Validation is integrated into `FrameProcessor.processImage()` on `consumerDispatcher`.
> One validation image is processed per camera frame, piggyback after YOLO. No new dispatcher or
> interpreter is created. `FrameProcessor.poseInterpreter` (initialized dormant in Day 5) is reused.

Required because:
1. A third `GpuDelegate` on a second dispatcher invalidated ADR-005 Gate §1 — no longer sequential on one thread
2. Raw `File` I/O to `/sdcard/Download/` fails silently on `targetSdk=34` — manifest declares only `CAMERA`

### Image Input: System Media Picker (replaces `adb push` + `File` I/O)

> **Old:** `adb push .../pose_isolate /sdcard/Download/` + `File("/sdcard/Download/pose_isolate").exists()`
>
> **New:** System media picker (SAF/ContentResolver) invoked by the validation button. No storage
> permissions needed — `ContentResolver` URI access is granted by the picker implicitly.

Bitmap decode (`ContentResolver` → `BitmapFactory.decodeStream`) happens on `Dispatchers.IO` before
bitmaps are submitted to `FrameProcessor`. All cropping and preprocessing remain on `consumerDispatcher`
to simulate the real pipeline.

### Pipeline Shutdown After Validation

> **Old:** Not specified — system kept running after validation batch.
>
> **New:** When the last validation image is dequeued inside `processImage()`,
> `poseValidationComplete = true`. `submitImage()` checks this flag and drops all subsequent
> camera frames (`image.close(); return`). The `consumerJob` idles naturally (no new channel
> input). UI emits a completion notification.

### Tensor Contract - Flattened Outputs (Netron-verified)

Day 5 uses the extracted `pose_landmarks_detector.tflite` directly via TFLite `Interpreter`, so tensor contracts must match the raw model, not the MediaPipe Tasks wrapper.

Verified contract for the current model file:
- Input: `input_1` -> `float32[1,256,256,3]` RGB in `[0,1]`
- Output: `Identity` -> `float32[1,195]` (39 pose landmarks x 5 fields)
- Output: `Identity_1` -> `float32[1,1]` (global pose presence)
- Output: `Identity_4` -> `float32[1,117]` (39 world landmarks x 3 fields)
- Optional outputs ignored for Day 5: `Identity_2` segmentation mask, `Identity_3` heatmap

Interpretation rules used in Day 5:
- `Identity` landmark coordinates are ROI-local pixel scale (roughly `[0,255]`), not normalized `[0,1]`
- `visibility` and `presence` from `Identity` are logits; apply sigmoid before thresholding
- CourtVision logic uses the first 33 landmarks as the canonical subset; keep all 39 for contract verification


---

## Files

### Create
- `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt`
- `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarks.kt` (data classes)
- `app/src/test/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreterTest.kt`
- `app/src/main/assets/pose_landmarks_detector.tflite` (extracted at build time — see Step 0)
- `docs/decisions/006-pose-landmark-egl-verification.md` (brief ADR recording Day 5 EGL result — only needed if result is non-obvious)

### Modify
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` — add `poseInterpreter` field, initialized on `inferenceDispatcher` (idle for Day 5; wired up in Day 6)
- `CONTEXT.md` — record EGL context sharing result after Verification Gate §1
- `TASKS.md` — mark tasks complete as work progresses
- Bind only `Identity`, `Identity_1`, and `Identity_4`; ignore segmentation mask and heatmap in Day 5
- Netron tensor identifiers (e.g., `310`, `315`) are not guaranteed Java output-map indices; verify with `getOutputTensor(i).shape()`
- `visibility` and `presence` must be sigmoid-transformed before applying thresholds

---

## Step-by-Step Implementation

### Step 0 — Extract `pose_landmarks_detector.tflite` from MediaPipe `.task` bundle (build-time, one-off)
**Status:** ✅Complete.
The MediaPipe `.task` file is a ZIP archive. Extract the model:

```bash
# Download the .task file (if not already present)
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

cp pose_landmarker_lite.task pose_landmarker_lite.zip
unzip pose_landmarker_lite.zip -d pose_task_extracted/
# The .tflite model is at: pose_task_extracted/pose_landmarks_detector.tflite
cp pose_task_extracted/pose_landmarks_detector.tflite \
   app/src/main/assets/pose_landmarks_detector.tflite
```

- File size: ~4 MB (lite variant) — verify it falls within the `assets/` budget
- Add this extraction step to `docs/dev-workflow.md` or build README so it is reproducible

---

### Step 1 — `PoseLandmarks.kt`: Data Classes
**Status:** ✅ Complete.

```kotlin
package com.courtvision.spike.pipeline

/** Single 3D landmark in world space (meters, Y-up, origin = hip midpoint). */
data class WorldLandmark(
    val x: Float,           // lateral (positive = subject's right)
    val y: Float,           // vertical (positive = up)
    val z: Float,           // depth (positive = behind subject; monocular — unreliable)
    val visibility: Float,  // [0, 1] — gate angle computation on > 0.6
    val presence: Float
)

/** Full 33-landmark pose result from one inference. */
data class PoseResult(
    val worldLandmarks: List<WorldLandmark>,   // size 33
    val imageLandmarks: List<WorldLandmark>,   // size 33 — normalized [0,1] to crop; for debug overlay only
    val inferenceMs: Long
)

/** Landmark indices used by CourtVision. */
object PoseLandmarkIndex {
    const val LEFT_SHOULDER  = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW     = 13
    const val RIGHT_ELBOW    = 14
    const val LEFT_WRIST     = 15
    const val RIGHT_WRIST    = 16
    const val LEFT_HIP       = 23
    const val RIGHT_HIP      = 24
    const val LEFT_KNEE      = 25
    const val RIGHT_KNEE     = 26
    const val LEFT_ANKLE     = 27
    const val RIGHT_ANKLE    = 28
}
```

---

### Step 2 — `PoseLandmarkInterpreter.kt`: Standalone TFLite Interpreter
**Status:** ✅ Complete.

```kotlin
package com.courtvision.spike.pipeline

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PoseLandmarkInterpreter(context: Context) : AutoCloseable {

    private val delegate = GpuDelegate()
    private val interpreter: Interpreter

    // Pre-allocated input: [1, 256, 256, 3] float32
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4).order(ByteOrder.nativeOrder())

    // Output shapes — verify against model with Netron before finalising
    // pose_landmarks_detector outputs: landmarks [1, 33, 5], world_landmarks [1, 33, 5]
    private val landmarksOutput    = Array(1) { Array(33) { FloatArray(5) } }
    private val worldLandmarksOutput = Array(1) { Array(33) { FloatArray(5) } }

    init {
        // MUST be called on inferenceDispatcher — not on Main or calling coroutine
        val opts = Interpreter.Options().addDelegate(delegate)
        interpreter = Interpreter(loadModelFile(context, "pose_landmarks_detector.tflite"), opts)
        interpreter.allocateTensors()
    }

    /**
     * Run pose inference on a pre-cropped 256×256 RGB bitmap.
     * Must be called on the same thread/dispatcher as [init].
     */
    fun infer(crop256: Bitmap): PoseResult {
        val startMs = System.currentTimeMillis()

        // Fill input buffer — normalized [0, 1] float32
        inputBuffer.rewind()
        val pixels = IntArray(256 * 256)
        crop256.getPixels(pixels, 0, 256, 0, 0, 256, 256)
        for (px in pixels) {
            inputBuffer.putFloat(((px shr 16) and 0xFF) / 255f)  // R
            inputBuffer.putFloat(((px shr 8)  and 0xFF) / 255f)  // G
            inputBuffer.putFloat((px          and 0xFF) / 255f)  // B
        }

        val outputs = mapOf(
            0 to landmarksOutput,
            1 to worldLandmarksOutput
            // Note: verify output tensor indices with Netron — lite model may differ
        )
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        val inferenceMs = System.currentTimeMillis() - startMs
        return PoseResult(
            worldLandmarks = decodeWorldLandmarks(worldLandmarksOutput[0]),
            imageLandmarks  = decodeWorldLandmarks(landmarksOutput[0]),
            inferenceMs = inferenceMs
        )
    }

    private fun decodeWorldLandmarks(raw: Array<FloatArray>): List<WorldLandmark> =
        raw.map { lm -> WorldLandmark(lm[0], lm[1], lm[2], lm[3], lm[4]) }

    override fun close() {
        interpreter.close()
        delegate.close()
    }
}
```

**Critical implementation notes:**
- `init` block must execute on `inferenceDispatcher` — wrap construction in `withContext(inferenceDispatcher)` at call site
- Bind only `Identity`, `Identity_1`, and `Identity_4`; ignore segmentation mask and heatmap in Day 5
- Netron tensor identifiers (e.g., `310`, `315`) are not guaranteed Java output-map indices; verify with `getOutputTensor(i).shape()`
- `visibility` and `presence` must be sigmoid-transformed before applying thresholds

---

### Step 3 — `FrameProcessor.kt`: Add Pose Interpreter Field (Dormant)
**Status:** ✅ Complete.

Add `poseInterpreter` as an initialized field so Day 6 integration is a one-line wiring change, not a restructure. Keep it dormant for Day 5 (no calls from the live pipeline outside validation):

```kotlin
private var poseInterpreter: PoseLandmarkInterpreter? = null
```

Current implementation initializes on-demand inside `processImage()` on `consumerDispatcher`:

```kotlin
private fun initializePoseInterpreterIfNeeded() {
    if (poseInterpreter != null) return
    val poseBuffer = poseModelBufferProvider?.invoke() ?: return
    poseInterpreter = PoseLandmarkInterpreter(poseBuffer)
}
```
### Step 3.5 — Validation Runner: Media Picker + `consumerDispatcher` Integration
**Status:** ✅ Complete.

> ⚠️ **Revised from original plan.** See Design Decisions above for why the standalone runner
> and `/sdcard/Download/` path were replaced.

**`FrameProcessorGateway` additions:**
```kotlin
fun submitPoseValidationBatch(bitmaps: List<Bitmap>)
val poseValidationResults: Flow<PoseFrameResult>
```

**`FrameProcessor` additions:**
```kotlin
// Validation-only queue — not part of live YOLO pipeline
private val poseValidationQueue    = ConcurrentLinkedQueue<Bitmap>()
private val poseValidationActive   = AtomicBoolean(false)
private val poseValidationComplete = AtomicBoolean(false)
private val poseValidationResultChannel = Channel<PoseFrameResult>(capacity = Channel.UNLIMITED)

// Pre-allocated, initialized alongside poseInterpreter on consumerDispatcher
private var poseValTensorImage: TensorImage? = null       // reused via .load(bitmap)
private var poseValImageProcessor: ImageProcessor? = null // ResizeOp(256,256) + NormalizeOp(0f,255f)
```

**Button behaviour (CameraScreen → CameraViewModel):**
- Button opens system media picker (multi-select images)
- On result: decode URIs → `List<Bitmap>` on `Dispatchers.IO` via `ContentResolver`
- Call `frameProcessor.submitPoseValidationBatch(bitmaps)`

**Per-frame processing (inside `FrameProcessor.processImage()`, after YOLO block):**
```
if poseValidationActive && poseValidationQueue.isNotEmpty():
  t_crop_start
    val crop = poseValidationQueue.poll()           // pop next validation bitmap
  pose_crop_ms = elapsed                            // crop step (identity for pre-cropped input)

  t_pre_start
    poseValTensorImage.load(crop)                   // no new TensorImage allocation
    val poseTensor = poseValImageProcessor.process(poseValTensorImage)
  pose_preprocess_ms = elapsed

  t_inf_start
    poseInterpreter.infer(poseTensor.buffer)
  pose_inference_ms = elapsed

  t_post_start
    decode landmarks, sigmoid, take(33), count visible
  pose_postprocess_ms = elapsed

  crop.recycle()
  emit PoseFrameResult(...) to poseValidationResultChannel

  if poseValidationQueue.isEmpty():
    poseValidationComplete = true                   // gates submitImage() to drop new frames
```

**Async output writer (ViewModel, `Dispatchers.IO`):**
Collects `poseValidationResults` flow:
- Write overlay PNG via `MediaStore` insert
- Append CSV row
- On `isLast == true`: write summary file + show completion UI notification

---

### Step 4 — Ground Truth Validation (Manual)
**Status:** ✅ Complete (manual plausibility gate passed).

Before benchmarking, confirm the model produces plausible output:

1. Select 10 frames from each video clip in 'C:\Users\Ciel Sun\Desktop\Dev\videos'(14 clips) containing a clearly visible standing person (uniform/evenly spaced sampling)
2. Use trained detector to detect person extract bounding box crop; resize to 256×256; export cropped images to 'C:\Users\Ciel Sun\Desktop\Dev\images\pose_isolate', naming each file with original clip filename + frame number + person index (if multiple persons in frame).
3. Trigger the Step 3.5 runner and run `poseInterpreter.infer(crop)` for each sample
4. Visually verify (overlaying image landmarks on the crop):
   - Save visualized images preferrably to 'C:\Users\Ciel Sun\Desktop\Dev\images\pose_isolate\inference_{datetimestamp}', or to phone storage (location TBD).
   - Convert ROI-local `xPx/yPx` to overlay coordinates (or normalize by `/255f` first)
   - Wrist (15/16) above elbow (13/14) when arms are raised
   - Use sigmoid-transformed `visibility`/`presence`; check `visibility > 0.6` for clearly visible joints
5. Fail condition: if more than 2/10 clips show landmark index misalignment (left/right swap, implausible joint ordering), re-check output index mapping and 39-to-33 slicing


---

### Step 5 — EGL Context Verification (ADR-005 Gate §1)
**Status:** ✅ Complete (PASS on S22+).

This is the most critical Day 5 test — result must be recorded in `CONTEXT.md`.

**Setup:** `FrameProcessor` now holds both `yoloInterpreter` (existing `GpuDelegate`) and `poseInterpreter` (new `GpuDelegate`), both initialized/invoked on `consumerDispatcher`.

**Test:** Run both interpreters sequentially on `consumerDispatcher` for 100 frames. Watch logcat for:
```
TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.
```

**Expected outcomes:**

| Result | Action |
|--------|--------|
| No EGL errors on S22+ | Record "EGL context shared — verified on S22+ (Adreno/Xclipse)" in CONTEXT.md. Day 6 can proceed on S22+. Pixel 6 and A54 EGL results remain unknown — flag as Day 6 risk. |
| EGL errors on S22+ | Sequential GPU architecture is fundamentally broken. Fall back to Config D (pose CPU, 4 threads). Revisit Day 6 plan — no point testing lower-end devices. |

> **Pixel 6 and A54 deferred:** Both devices unavailable. EGL behaviour on Mali-G68 (A54) and Tensor G2 (Pixel 6) may differ from S22+. Validate before Phase 2 — options: borrow device, Firebase Test Lab, or BrowserStack cloud testing.

**Observed result (2026-04-14):**
- PASS on S22+ (`SM-S906U1`)
- No runtime `TfLiteGpuDelegate ... must run on the same thread` error observed

---

### Step 6 — Latency Benchmark
**Status:** ✅ Complete (PASS_STRONG on S22+).

> ⚠️ **Latency characterisation change:** Validation runs YOLO on every camera frame before pose.
> `pose_inference_ms` reflects GPU-warm sequential pipeline timing, **not** isolated pose-only
> timing. Gate criteria apply to the `pose_inference_ms` column only, not `frame_total_ms`.
> Label benchmark artifacts with `mode=sequential_yolo_warm`.

Run sequential YOLO+pose inference via the Step 3.5 runner on pre-cropped bitmaps (≥100 images).

**Per-frame CSV columns:**
```
timestamp, file,
yolo_preprocess_ms, yolo_inference_ms, yolo_nms_ms,
pose_crop_ms, pose_preprocess_ms, pose_inference_ms, pose_postprocess_ms,
frame_total_ms, overlay_write_ms,
pose_presence, visible_joints_33, decoded_landmarks_39, yolo_detections,
mode, device, gpu_mode, status, error
```

**Summary file header** (add to top of `day5_pose_summary.txt`):
```
device=${Build.MODEL}
gpu_mode=GPU|CPU
input_size=256
mode=sequential_yolo_warm
ram_mb=...
thermal_status=...
```

Artifacts committed under:
- `/benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_summary.txt`
- `/benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_validation.csv`

**Gate criteria:**

| Device | p95 target | Fallback trigger |
|--------|-----------|-----------------|
| Galaxy S22+ (Adreno 730 or Xclipse 920) | ≤ 50ms | > 70ms → Config D regardless of other devices |
| Pixel 6 (Tensor G2 GPU) | ≤ 70ms | **Deferred** — not available; validate before Phase 2 |
| Galaxy A54 (Exynos 1380 GPU) | ≤ 80ms | **Deferred** — not available; validate before Phase 2 |

> S22+ target is set tighter (≤ 50ms) to provide headroom against lower-end devices — if S22+ is in the 50–70ms range, the Pixel 6 and A54 gaps are unknown and should be flagged as a Day 8 risk.
> Fallback trigger: if S22+ p95 > 70ms → Config D (CPU) is the safe path regardless of other devices.

**Observed result (2026-04-14):**
- `samples=100`, `success=100`, `failure=0`
- `pose_inference_p95_ms=10.774` (`PASS_STRONG`)

---

### Step 7 — Unit Tests: `PoseLandmarkInterpreterTest`
**Status:** ✅ Complete for Day 5 scope; JUnit 5 migration remains deferred to Day 6.

> ⚠️ **JUnit 4 used in current implementation.** `PoseLandmarkInterpreterTest` uses
> `org.junit.Assert.*` and `org.junit.Test`. CONTEXT.md specifies JUnit 5. Align before Phase 2:
> replace with `org.junit.jupiter.api.Test` and `assertThrows<T> { }`.

**Pulled forward from Day 6:** picker-based validation now accepts arbitrary-aspect photos, so
Day 5 cannot assume square inputs. `squarePadCrop()` is introduced now with a bbox-free signature.
Day 6 will crop YOLO `person` bbox upstream and call the same function before pose infer.

**Add now with `squarePadCrop`:**
```kotlin
@Test fun squarePadCrop_squareInput_returnsSameInstance() { ... }
@Test fun squarePadCrop_tallInput_returnsSquareAndCenteredContent() { ... }
@Test fun squarePadCrop_wideInput_returnsSquareAndCenteredContent() { ... }
```

**Written and passing in Day 5 (core + follow-ups):**
- `decodeImageLandmarks_195float_produces39landmarks`
- `decodeWorldLandmarks_117float_produces39landmarks`
- `imageLandmarks33_isFirst33ofFull39`
- `sigmoid_positiveLogit_producesValueAboveHalf`
- `sigmoid_negativeLogit_producesValueBelowHalf`
- `resolveOutputIndices_outOfOrderShapes_resolvesCorrectly`
- `resolveOutputIndices_missingTensor_throws`
- `validateInputTensor_wrongShape_throws`
- `resolveRuntimeConfig_cpuFallback_usesFourThreads`
- `worldLandmarkVisibility_readsFromImageLandmarks33`
- `worldLandmarkVisibility_throwsWhenIndexOutOfRange`
- `squarePadCrop_squareInput_returnsSameInstance` (`BitmapOpsTest`)
- `squarePadCrop_tallInput_returnsSquareAndCenteredContent` (`BitmapOpsTest`)
- `squarePadCrop_wideInput_returnsSquareAndCenteredContent` (`BitmapOpsTest`)

Run with:
```
./gradlew :app:testDebugUnitTest --tests "*.PoseLandmarkInterpreterTest"
```

---

## Noise / Output Tensor Verification Checklist

- [x] Input tensor shape: `[1, 256, 256, 3]` float32
- [x] Output tensor `Identity` shape: `[1, 195]` (39 x 5 image landmarks)
- [x] Output tensor `Identity_1` shape: `[1, 1]` (global pose presence)
- [x] Output tensor `Identity_4` shape: `[1, 117]` (39 x 3 world landmarks)
- [x] Optional outputs present but ignored in Day 5:
  - `Identity_2`: `[1,256,256,1]` segmentation mask
  - `Identity_3`: `[1,64,64,39]` heatmap
- [x] Output map in code binds only image/presence/world tensors
- [x] `Identity` x/y/z interpreted as ROI-local pixel scale, not normalized `[0,1]`
- [x] Sigmoid applied to landmark `visibility` and `presence` logits before thresholding
- [x] 39-landmark decode verified, then first 33 landmarks used for CourtVision indices


If runtime output indices differ from assumed indices, update the output map constants after verifying with `interpreter.getOutputTensor(i).shape()`.

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §1 EGL context | No `GpuDelegate must run on same thread` errors on S22+ | Record failure; fall back to Config D |
| §2 Landmark correctness | ≥ 8/10 clips show plausible landmark positions | Re-verify output index mapping and 39-to-33 slicing |
| §3 Pose p95 latency (S22+) | ≤ 50ms (strong pass) / ≤ 70ms (marginal) | > 70ms → Config D; 50–70ms → flag Pixel 6 and A54 risk |
| §3 Pose p95 latency (Pixel 6) | ≤ 70ms — **deferred; validate before Phase 2** | > 70ms → Config D |
| §3 Pose p95 latency (A54) | ≤ 80ms — **deferred; validate before Phase 2** | > 80ms → Config D |
| §4 No-alloc (Day 6 gate) | Acceptable for Day 5 isolation; must be fixed before Day 6 | Replace `getPixels` loop with pre-allocated Canvas rescale |

Observed on 2026-04-14 (S22+): §1 PASS, §2 PASS, §3 PASS_STRONG (`pose_inference_p95_ms=10.774`).
Results are recorded in `CONTEXT.md`.

---

## Known Deviations and Day-6 Follow-ups

### Blocker B1 — `PoseWorldLandmark` missing `visibility` field (Resolved)

> **Plan schema (Step 1):** `WorldLandmark` included `visibility: Float` and `presence: Float`.
> **Implemented:** `PoseWorldLandmark` has only `x`, `y`, `z` — correct per raw tensor `Identity_4`
> `[1,117]` = 39×3 (world tensor carries no visibility), but creates an API trap: Day 7 angle
> computation iterates `worldLandmarks33` and has no visibility field to gate on.

**Implemented in Day 5:** Added to `PoseLandmarks.kt`:
```kotlin
/**
 * Returns sigmoid visibility for world landmark [index] from imageLandmarks33.
 * Always gate world-coordinate angle computations on this value > 0.6f per tdd.md §4.2.
 */
fun PoseResult.worldLandmarkVisibility(index: Int): Float = imageLandmarks33[index].visibility
```

### Blocker B2 — `PoseLandmarkInterpreter` has no CPU fallback path (Resolved)

ADR-005 Config D (pose CPU, 4 threads) requires constructing the interpreter without `GpuDelegate`.
Implemented with `useGpu` parameter and CPU(4-thread) runtime config:
```kotlin
class PoseLandmarkInterpreter(
    modelBuffer: MappedByteBuffer,
    useGpu: Boolean = true          // false → CPU only, 4 threads (Config D)
)
```

### Deviation D1 — `squarePadCrop()` pulled forward into Day 5

The revised runner uses gallery-selected images and no longer guarantees square 256×256 inputs.
To avoid landmark distortion from anisotropic resize, Day 5 now runs `squarePadCrop()` in
`processPoseValidationIfNeeded()` before `infer()`. Function signature stays bbox-agnostic;
Day 6 live callers will perform YOLO bbox crop first, then pad-to-square.

### Note — Bitmap lifecycle for Day 6 (`processImage()`)

Current `processImage()` recycles the source bitmap in a `finally` block immediately after YOLO
preprocessing (`FrameProcessor.kt`). For Day 6, `squarePadCrop` needs the original bitmap AFTER
YOLO inference returns a person bbox.

Required changes in Day 6:
1. `bitmap.recycle()` — deferred to after `squarePadCrop()` (or immediately if no person detected)
2. `personCrop.recycle()` — after pose `ImageProcessor.process()` materialises the float tensor

`squarePadCrop` must also recycle its intermediate crop:
```kotlin
val cropped = Bitmap.createBitmap(src, left, top, right - left, bottom - top)
val scaled  = Bitmap.createScaledBitmap(cropped, 256, 256, false)
if (cropped !== scaled) cropped.recycle()   // guard: createScaledBitmap may return same object
return scaled
```
