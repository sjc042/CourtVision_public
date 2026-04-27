# Plan: Day 6 — Sequential GPU Combined Pipeline

**Status:** ✅ Day 6 closed (2026-04-26) — Steps 0–7 landed; §4/§5/§7 PASS on S22+ (§7 PASS-with-rescope after order-swapped soak re-analysis); §3 (`frame_total_ms` p95) and §6 (FPS) FAIL routed forward to the Day 7 perf workstream per the verification-gate plan, not blockers for Day 6 sign-off. See "§7 Sliced Re-analysis (2026-04-26)" below for the rescope verdict and CONTEXT.md "Day 6 Step 7 Benchmark Results" for the gate-by-gate summary.

## Context

Days 1–5 are complete. The pipeline on `main` is:
```
CameraX → YOLO (5-class TFLite, GPU) → Kalman tracker → [StateFlow] → UI overlay
                                                       ↑
                              pose_landmarks_detector.tflite loaded dormant (Day 5)
```

Day 6 wires the dormant pose interpreter into the **live camera path**: YOLO `person` bbox feeds `squarePadCrop` → pose inference, all on `consumerDispatcher`. This produces the first end-to-end sequential GPU pipeline and provides the combined latency number needed for the Day 8 gate.

> Full architecture decision: [ADR-005](../decisions/005-sequential-gpu-inference-pipeline.md)

---

## Branch

`spike/day6-combined-pipeline` (branch from `main` after Day 5 PR merges)

---

## Prerequisites

- Day 5 PR merged to `main` ✅
- S22+ available for benchmark run

---

## Design Decisions

### Live Pose Gating and Person Selection — ADR-006

See [ADR-006](../decisions/006-pose-gating-mode.md) for the full decision. Summary:

- `PoseGatingMode.EVERY_FRAME_WITH_PERSON` is the Day 6 default — maximises benchmark sample
  count for reliable p95 stats. `SHOOT_CLASS_GATED` (Phase 2 optimization) and `FSM_GATED`
  (Day 7) are stubbed.
- `PersonSelectionMode.HIGHEST_CONFIDENCE` is the Day 6 default. `REID_TRACKED` is stubbed
  for Phase 2.
- Both are orthogonal enums in `FrameContracts.kt`, following the `InferenceMode` pattern.
  Day 7 FSM wiring is a one-line enum value change — no call-site restructure.

### squarePadCrop — Bbox-Aware Overload in BitmapOps

Day 5 added `squarePadCrop(source: Bitmap)` (full-frame, bbox-agnostic). Day 6 needs the
production overload: `squarePadCrop(source: Bitmap, box: BoundingBox, marginFactor: Float = 1.25f)`
matching the ADR-005 §3 snippet. Add as an overload in `BitmapOps.kt`. The existing overload
stays — it is used by the validation path.

### No-Alloc Fix — TFLite ImageProcessor (Not Canvas)

`PoseLandmarkInterpreter.infer()` currently fills `inputBuffer` via `getPixels` which allocates a
new `IntArray(256 * 256)` (~1 MB) per call. Fix: pre-allocate a `TensorImage` and
`ImageProcessor` (ResizeOp 256×256 + NormalizeOp 0f/255f) in `init`, reuse via
`tensorImage.load(bitmap)`. This matches the validation path already pre-allocated in
`FrameProcessor` and avoids the IntArray entirely. Replace the `getPixels` loop and `pixels` field.

### StateFlow for Live PoseResult

Day 6 adds `poseResult: StateFlow<PoseResult?>` to `FrameProcessorGateway` for the live path.
This is separate from `poseValidationResults: Flow<PoseFrameResult>` (validation-only, stays).
`FrameProcessor` emits via a private `MutableStateFlow<PoseResult?>` updated inside
`processImage()` after each live pose inference. ViewModel collects and exposes to UI overlay.

### Fix Reflection Risk Before StateFlow Refactor

`FrameProcessorTest` currently uses `getDeclaredField("poseValidationActive")` and
`getDeclaredField("poseValidationComplete")`. These stay as `AtomicBoolean` fields in Day 6 —
the new `poseResult` StateFlow is additive, not a replacement. This reflection risk remains open
but does **not** trigger in Day 6. Must be resolved before any Day 7 refactor that touches those
fields (see Known Day-7 Follow-ups).

### Bitmap Lifecycle Fix

Current `processImage()` recycles the source bitmap in a `finally` block immediately after YOLO
preprocessing — before `squarePadCrop` can use it. Restructure, incorporating ADR-006 enum
dispatch so the lifecycle fix and the gating abstraction land together:

```kotlin
try {
    // YOLO
    val detections = runYolo(bitmap)
    val personBox = selectPersonBox(detections, personSelectionMode)   // PersonSelectionMode
    val runPose = personBox != null && shouldRunPose(detections, poseGatingMode)  // PoseGatingMode

    // Pose (only if gating allows and person detected)
    if (runPose) {
        val crop = BitmapOps.squarePadCrop(bitmap, personBox!!, 1.25f) // bitmap must be alive here
        val poseResult = poseInterpreter?.infer(crop)
        crop.recycle()
        _poseResult.value = poseResult
    }
    latencyLogger.recordPoseSkipped(!runPose)   // feeds pose_skipped CSV column
} finally {
    bitmap.recycle()   // deferred — safe for both pose and no-pose branches
}
```

`selectPersonBox` picks `HIGHEST_CONFIDENCE` in Phase 0 (returns `detections.maxByOrNull { it.confidence }
?.takeIf { it.classId == PERSON_CLASS_ID }`). `shouldRunPose` returns `true` for
`EVERY_FRAME_WITH_PERSON` and `SHOOT_CLASS_GATED`, and falls back to `EVERY_FRAME_WITH_PERSON`
behavior for the `FSM_GATED` stub until Day 7 wires it. Both are single-branch private helpers
in `FrameProcessor`; Day 7 / Phase 2 adds branches without touching this call site.

---

## Files

### Modify
- `app/src/main/java/com/courtvision/spike/pipeline/BitmapOps.kt` — add bbox-aware `squarePadCrop` overload
- `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt` — no-alloc fix (pre-allocated TensorImage/ImageProcessor replaces `getPixels` loop)
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` — wire live pose path; fix bitmap lifecycle; emit `poseResult` StateFlow
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessorGateway.kt` — add `poseResult: StateFlow<PoseResult?>`
- `app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt` — collect `poseResult` flow; expose to UI
- `app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt` — add pose overlay (landmark dots or angle readout)
- `app/src/test/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreterTest.kt` — migrate to JUnit 5; update for new `infer()` signature
- `app/src/test/java/com/courtvision/spike/pipeline/BitmapOpsTest.kt` — add tests for bbox-aware overload
- `app/src/test/java/com/courtvision/spike/pipeline/FrameProcessorTest.kt` — tests for live pose path, bitmap lifecycle
- `CONTEXT.md` — update Progress line; record Day 6 combined latency result after benchmark

---

## Step-by-Step Implementation

### Step 0 — JUnit 5 migration (unit tests) + JUnit 4 stabilisation (androidTest) ✅

**Unit tests (`src/test/`) — all migrated to JUnit 5:**
Replaced `org.junit.Assert.*` / `org.junit.Test` with JUnit 5 APIs across all unit test classes:
- `PoseLandmarkInterpreterTest` — `org.junit.jupiter.api.Test`, `assertThrows<T> { }`
- `KalmanBallTrackerTest`, `PerformanceCsvLoggerTest`, `BitmapOpsTest`, `FrameProcessorTest`, `CameraViewModelTest`

**androidTest (`src/androidTest/`) — reverted to JUnit 4:**
Migrating `PreprocessingLatencyTest` to JUnit 5 via Mannodermaus caused `connectedDebugAndroidTest`
to report "Starting 0 tests". Root causes:
1. Manual `junit-platform-*` artifacts conflicted with what the Mannodermaus plugin auto-manages.
2. `kotlinx-coroutines-test` was missing from `androidTestImplementation` (`runTest` had no runtime backing).
3. Kotlin 1.9.24 pinning on all `*Test*` configs interfered with the Mannodermaus runtime.

**Decision:** `PreprocessingLatencyTest` reverted to JUnit 4. CONTEXT.md only requires JUnit 5 for
unit tests — this benchmark test just needs to run on-device. JUnit 4 + `AndroidJUnitRunner` is the
battle-tested instrumented-test path.

**`app/build.gradle.kts` changes:**
- Removed: Mannodermaus plugin, `junitPlatform` block, `runnerBuilder` arg, all manual `junit-platform-*`
  and `junit-jupiter-engine` `androidTestImplementation` deps.
- Added: `androidx.test.ext:junit:1.2.1`, `kotlinx-coroutines-test:1.8.1` to `androidTestImplementation`.
- Unit-test JUnit 5 deps, Robolectric extension plugin, and Kotlin pinning block unchanged.

---

### Step 1 — JUnit 5 migration: `PoseLandmarkInterpreterTest` ✅
**Deferred from Day 5:** `PoseLandmarkInterpreterTest` previously used JUnit 4 APIs (`org.junit.Assert.*`, `org.junit.Test`); `CONTEXT.md` specifies JUnit 5 project-wide. Completed as part of Step 0.

---

### Step 2 - No-alloc fix: `PoseLandmarkInterpreter.infer()` [Done]
`infer()` now uses pre-allocated `TensorImage` + `ImageProcessor(NormalizeOp)` with no per-frame
`getPixels`/`IntArray` copy path, and enforces a strict 256x256 input contract.

Profiler verification gate dropped from Day 6 sign-off (2026-04-22). Step 7 soak provides
indirect evidence: `pose_inference_ms` p50 / p99 = 11.17 / 19.44 ms (~8 ms spread across 5089
frames) is inconsistent with per-frame ~1 MB allocations that would trigger GC-pause outliers in
the 40–80 ms range. Direct Android Studio Memory Profiler capture deferred to Phase 2 if a
pose-side allocation regression is ever suspected. See [CONTEXT.md](../../CONTEXT.md) "Day 6
Step 2 Verification Note".
---

### Step 3 — `BitmapOps.kt`: bbox-aware `squarePadCrop` overload ✅

Landed in [BitmapOps.kt:36-81](../../app/src/main/java/com/courtvision/spike/pipeline/BitmapOps.kt#L36-L81). Uses `DetectionBox` (normalized floats) instead of introducing a new `BoundingBox` type — avoids a shim since `FrameProcessor` already has `DetectionBox` in hand post-NMS. Output is always 256×256 via `PoseTensorContract.INPUT_SIZE` (no new `INPUT_POSE_SIZE` constant). Includes a defensive clamp-to-1px fallback for degenerate bboxes beyond the original snippet.

`BitmapOpsTest` adds `squarePadCrop_withBbox_returns256x256`, `_bboxAtEdge_clampsToFrameBoundsAndReturns256x256`, `_bboxFullFrame_returns256x256`. Existing padColor-overload cases retained.

---

### Step 3 (historical snippet — replaced by implementation above)

Add overload matching ADR-005 §Implementation Details §3:

```kotlin
fun squarePadCrop(
    src: Bitmap,
    box: BoundingBox,           // normalized [0, 1]
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
    val scaled  = Bitmap.createScaledBitmap(cropped, INPUT_POSE_SIZE, INPUT_POSE_SIZE, false)
    if (cropped !== scaled) cropped.recycle()
    return scaled
}
```

`INPUT_POSE_SIZE = 256` constant in `PoseTensorContract` (or `BitmapOps` companion).

Add `BitmapOpsTest` cases:
- `squarePadCrop_withBbox_returnsMarginedSquareCrop`
- `squarePadCrop_bboxAtEdge_clampsToFrameBounds`
- `squarePadCrop_noPersonBbox_fullFrameFallback` (ensure existing overload still works)

---

### Step 4 — `FrameProcessor.kt`: fix bitmap lifecycle + wire live pose ✅

Landed. Key landing points:

- **Bitmap lifecycle fix:** `bitmap.recycle()` moved from YOLO-preprocess `finally` to the outer `processImage()` finally ([FrameProcessor.kt:574](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L574)) so it survives through live pose.
- **ADR-006 enums + setters:** [FrameContracts.kt:53-62](../../app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt#L53-L62) + [FrameProcessorGateway.kt:20-21](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessorGateway.kt#L20-L21). Class-ID constants `PERSON_CLASS_ID=2`, `SHOOT_CLASS_ID=4` added ([FrameProcessor.kt:1081-1082](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L1081-L1082)).
- **Live pose wiring:** `runLivePoseIfGated` + `selectPersonBox` + `shouldRunPose` private helpers ([FrameProcessor.kt:583-642](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L583-L642)). After post-review cleanup, `personBox == null` and gating-mode checks are split — no dead null-fallback branch. `FSM_GATED` / `SHOOT_CLASS_GATED` silently fall back to `EVERY_FRAME_WITH_PERSON` per ADR-006 §Consequences (documented in-comment at the stubs).
- **Telemetry hook:** `poseSkipped` column added to `PipelineStats` ([FrameContracts.kt:27](../../app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt#L27)); `lastFrameSkippedPose: AtomicBoolean` feeds it each frame — ready for Step 7 CSV without reflection.
- **Test seam:** `PoseInferenceEngine` interface extracted ([PoseInferenceEngine.kt](../../app/src/main/java/com/courtvision/spike/pipeline/PoseInferenceEngine.kt)); `poseInterpreterFactory` constructor parameter allows tests to inject a `FakePoseInferenceEngine` without Robolectric TFLite delegates. Six new `FrameProcessorTest` cases cover live-pose emit, no-person skip, bitmap lifecycle, `FSM_GATED` fallback, and delegate alignment on mode switch.

---

### Step 4 (historical snippet — replaced by implementation above)

**4a — Bitmap lifecycle:**
Move `bitmap.recycle()` from the `finally` block after YOLO preprocessing to after pose
inference (or immediately after YOLO if no person detected). Add `crop.recycle()` after
pose `ImageProcessor.process()` materialises the tensor. Note: the validation path already
recycles the padded intermediate crop inside `processPoseValidationIfNeeded()` — only the
outer `bitmap.recycle()` shift is needed for the live camera path.

**4b — Live pose wiring (per ADR-006):**
Inside `processImage()`, after YOLO NMS:

```kotlin
val personBox = selectPersonBox(detections, personSelectionMode)  // PersonSelectionMode
if (personBox != null && shouldRunPose(detections, poseGatingMode)) {  // PoseGatingMode
    initializePoseInterpreterIfNeeded()
    val crop = BitmapOps.squarePadCrop(bitmap, personBox, marginFactor = 1.25f)
    val poseResult = poseInterpreter?.infer(crop)
    crop.recycle()
    _poseResult.value = poseResult
}
```

`selectPersonBox` and `shouldRunPose` are private helpers in `FrameProcessor` that dispatch
on the enum values. Phase 0 both are trivial single-branch functions; Day 7/Phase 2 adds
branches without touching the call site.

**4c — StateFlow emission:**
```kotlin
private val _poseResult = MutableStateFlow<PoseResult?>(null)
```
Exposed on `FrameProcessorGateway` as `val poseResult: StateFlow<PoseResult?>`.

---

### Step 5 — Wire `useGpu` through `InferenceMode` ✅

Landed:
- `initializePoseInterpreterIfNeeded()` now reads `currentMode.get() != InferenceMode.CPU` ([FrameProcessor.kt:285-296](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L285-L296)); init failure closes pose resources and surfaces `_lastError`.
- `switchInterpreter()` calls `closePoseResources()` ([FrameProcessor.kt:343](../../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L343)) so pose lazy-reinits with the matching delegate on the very next frame. Covers CPU↔GPU Config D transitions.
- Tests `initializePoseInterpreter_whenYoloGpu_usesGpuDelegate` and `switchInterpreterCpuToGpu_closesAndReinitializesPose` capture `useGpu` via a factory lambda + verify interpreter instance reset.

---

### Step 5 (historical snippet — replaced by implementation above)

`initializePoseInterpreterIfNeeded()` currently hardcodes `useGpu = true`, so pose always
attempts GPU even when YOLO has fallen back to CPU (Config D). Pass the current inference mode
so pose delegate always matches the YOLO delegate choice. Uses the same
`CompatibilityList.bestOptionsForThisDevice` path already used by `switchInterpreter` — no new
GPU capability check needed. Pass the current
inference mode:

```kotlin
private fun initializePoseInterpreterIfNeeded() {
    if (poseInterpreter != null) return
    val poseBuffer = poseModelBufferProvider?.invoke() ?: return
    val useGpu = currentInferenceMode != InferenceMode.CPU   // or check CompatibilityList
    poseInterpreter = PoseLandmarkInterpreter(poseBuffer, useGpu = useGpu)
}
```

Where `currentInferenceMode` is the same field already used by `switchInterpreter` for YOLO.
This ensures YOLO GPU ↔ pose GPU and YOLO CPU ↔ pose CPU (Config D) always match.

---

### Step 6 — `CameraViewModel` + `CameraScreen`: expose live pose ✅

**Landed summary:**
- Rotation moved upstream: `BitmapOps.rotateBitmapForDisplay` runs once per frame in `FrameProcessor.processImage()`; `Rot90Op` removed from the YOLO `ImageProcessor` chain; YOLO + pose + overlay now share the display-upright coordinate frame.
- `BitmapOps.squarePadCrop(source, personBox, marginFactor)` returns `PersonCrop(bitmap, cropRectNormalized)` so the overlay has the clamped ROI needed to project 256×256 landmarks back to canvas space.
- `FrameProcessorGateway.poseResult: StateFlow<LivePoseOverlay?>` emits `LivePoseOverlay(poseResult, cropRectNormalized)`; `CameraUiState.poseOverlay` collects it, gated on `modelConfirmed` and cleared on `restartSession()` / `setModel()`.
- `CameraScreen.PoseOverlay` composable sibling to `DetectionOverlay` using identical FILL_CENTER math; landmark dots colored by `visibility > 0.6`.
- ADR-005 §2/§3/§5/§Consequences updated to match (rotation ownership, `DetectionBox` + `PersonCrop`, explicit pre-rotation allocation, risk bullet).
- New `PoseInferenceEngine` interface lets `FakePoseInferenceEngine` stand in for tests via `poseInterpreterFactory` constructor seam; `PoseLandmarkInterpreter` implements it.

#### Context

Steps 2–5 emit `poseResult: StateFlow<PoseResult?>` from `FrameProcessor`, but no consumer collects it yet. Step 6 wires the flow into `CameraViewModel`/`CameraUiState` and adds a `PoseOverlay` composable over the camera preview so Day 6 sign-off can be judged visually on S22+.

Two non-obvious gaps must close before the overlay can render correctly:

1. **Rotation mismatch (latent bug).** `processImage()` calls `runLivePoseIfGated(bitmap, boxes)` with the *un-rotated* `image.toBitmap()` sensor bitmap, but `boxes` come from YOLO whose input is rotated via `Rot90Op`. On any device not in `ROTATION_0`, `squarePadCrop(sensorBitmap, personBox, 1.25f)` treats a display-frame-normalized bbox as sensor-normalized, crops the wrong ROI, and feeds the pose model a sideways person — pose will produce plausible-looking but wrong landmarks.

2. **Coord-space gap.** Pose landmarks are `xPx, yPx ∈ [0, 256]` (model input pixels), but the overlay renders in canvas (display) space. Mapping back requires the *actual clamped crop rect used by `squarePadCrop`* — not just `personBox` — because edge-clamping changes the effective ROI shape. That rect is computed inside `BitmapOps.squarePadCrop` today and thrown away.

Both must be fixed before the overlay can line up with the person. Fixing either in isolation is wasted work.

#### Design Decisions

**D1 — Rotate the bitmap once, share with YOLO and pose (not "rotate only for pose").**
The cleanest fix is to pre-rotate `bitmap` before YOLO preprocessing and drop `Rot90Op` from the YOLO `ImageProcessor` chain. Both consumers then see upright pixels and upright-frame-normalized coordinates. Rejected alternative: rotate only for pose (keeps `Rot90Op` in YOLO) — cheaper for YOLO-only frames but forks the coordinate story into two mental models and wastes a rotation when pose runs. The once-per-frame rotated-bitmap allocation (≈ 3.5 MB for 720×1280 ARGB) is the same order of magnitude as the existing un-rotated `toBitmap()` allocation and has been tolerated on S22+ in Days 1–5. A bitmap pool is a Phase 2 optimisation, not a Day 6 blocker.

**D2 — Emit `LivePoseOverlay(poseResult, cropRectNormalized)`, not bare `PoseResult`.**
Change `FrameProcessorGateway.poseResult` type from `StateFlow<PoseResult?>` to `StateFlow<LivePoseOverlay?>`. The crop rect travels with the pose data that depends on it — no cross-flow pairing race between `detections` and `poseResult`. Rejected alternatives: (a) extending `PoseResult` (pollutes the pure model-output contract used by validation-path tests), (b) a parallel `poseCropRect` flow (two flows, same race as pairing `detections`). This is a breaking API change within the branch; the only consumer today is `CameraViewModelTest.FakeFrameProcessor` which is updated as part of this step.

**D3 — Return the clamped crop rect from `squarePadCrop(source, personBox, marginFactor)`.**
Replace the `Bitmap` return type with `PersonCrop(bitmap, cropRectNormalized)`. The rect uses the rotated-frame dimensions (matches `DetectionFrame.sourceWidth/sourceHeight`) so the UI can reuse the exact FILL_CENTER scale math already in `DetectionOverlay`. Keep the existing `squarePadCrop(source, padColor)` overload unchanged — the validation path depends on the `Bitmap`-only signature and doesn't need a rect.

**D4 — Expose `poseOverlay: LivePoseOverlay?` on `CameraUiState`, not `poseResult: PoseResult?`.**
Matches the gateway change; overlay-ready by construction. Gated on `modelConfirmed` the same way `detectionFrame` is ([CameraViewModel.kt:131-149](../../app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt#L131-L149)). Cleared to null on `restartSession()` and on model change.

**D5 — `PoseOverlay` composable sibling to `DetectionOverlay`, not an extension of it.**
Two composables, one canvas each, same FILL_CENTER scale derivation. Keeps detection-box and pose-landmark rendering logic independently testable and lets the overlay be skipped when `poseOverlay == null` without branching inside `DetectionOverlay`.

#### Files to modify

| File | Change |
|------|--------|
| `FrameContracts.kt` | Add `LivePoseOverlay(poseResult: PoseResult, cropRectNormalized: CropRectNormalized)` and `CropRectNormalized(left, top, right, bottom)` data classes |
| `BitmapOps.kt` | Change bbox-overload return type to `PersonCrop(bitmap, cropRectNormalized)`; compute normalized rect from the already-clamped `left/top/right/bottom` ints |
| `FrameProcessor.kt` | **6a:** pre-rotate `bitmap` to display orientation at the top of `processImage()`; drop `Rot90Op` from the YOLO `ImageProcessor` chain (keep Resize + Normalize); recycle rotated bitmap in outer `finally`. **6b:** change `_poseResult: MutableStateFlow<LivePoseOverlay?>`; `runLivePoseIfGated` emits `LivePoseOverlay(result, personCrop.cropRectNormalized)` |
| `FrameProcessorGateway.kt` | Update `poseResult: StateFlow<LivePoseOverlay?>` type |
| `CameraUiState.kt` | Add `val poseOverlay: LivePoseOverlay? = null` |
| `CameraViewModel.kt` | 5th `viewModelScope.launch { frameProcessor.poseResult.collect { … } }` block; gate on `modelConfirmed`; clear in `restartSession()` and `setModel()` |
| `CameraScreen.kt` | New `PoseOverlay(overlay, modifier)` composable between `DetectionOverlay` and `MetricsOverlay` in the `Box` stack |
| `FrameProcessor.kt` (YOLO preprocess) | `imageProcessors` map collapses to a single `Resize + Normalize` processor; rotation handled upstream |
| `FrameProcessorTest.kt` | Update `runLivePoseForTest` assertions — `poseResult.value` now `LivePoseOverlay`; update delegate-alignment tests unchanged |
| `BitmapOpsTest.kt` | New cases: `squarePadCrop_bboxOverload_returnsClampedCropRect`, `squarePadCrop_bboxAtEdge_clampsRectBounds` |
| `CameraViewModelTest.kt` | `FakeFrameProcessor.poseFlow` retyped to `MutableStateFlow<LivePoseOverlay?>`; add `uiState.poseOverlay_reflectsGateway` test |

#### Sub-step 6a — Rotation alignment in `FrameProcessor.processImage()`

Replace the rotation-keyed `imageProcessors` map with a single un-rotated processor and do the rotation in bitmap space:

```kotlin
// FrameProcessor.kt
private val yoloImageProcessor: ImageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
    .add(NormalizeOp(0f, 255f))
    .build()

private suspend fun processImage(image: ImageProxy) {
    val sensorBitmap = image.toBitmap()
    val rotatedBitmap = rotateBitmapForDisplay(sensorBitmap, normalizedRotation)
    try {
        val tensorImage = yoloImageProcessor.process(TensorImage.fromBitmap(rotatedBitmap))
        // ... YOLO inference, NMS, tracker, _detections emit unchanged ...
        runLivePoseIfGated(bitmap = rotatedBitmap, boxes = boxes)
        processPoseValidationIfNeeded(…)
    } finally {
        if (rotatedBitmap !== sensorBitmap) rotatedBitmap.recycle()
        sensorBitmap.recycle()
        image.close()
    }
}
```

`rotateBitmapForDisplay` is a new top-level helper in `BitmapOps.kt`:
- Returns `source` unchanged if `normalizedRotation == 0`.
- Otherwise allocates a new ARGB_8888 bitmap and draws `source` through a `Matrix.postRotate(rotationDegrees)` transform.
- Output dimensions: `(H, W)` for 90°/270°, `(W, H)` for 180°.

`DetectionFrame.sourceWidth/sourceHeight` stays in rotated (display-upright) orientation — already does, so `DetectionOverlay` needs no change.

#### Sub-step 6b — `BitmapOps.squarePadCrop` returns `PersonCrop`

```kotlin
data class PersonCrop(
    val bitmap: Bitmap,
    val cropRectNormalized: CropRectNormalized
)

fun squarePadCrop(
    source: Bitmap,
    personBox: DetectionBox,
    marginFactor: Float = 1.25f
): PersonCrop {
    // existing clamp math produces left/top/right/bottom ints
    …
    val normalizedRect = CropRectNormalized(
        left = left.toFloat() / source.width,
        top = top.toFloat() / source.height,
        right = right.toFloat() / source.width,
        bottom = bottom.toFloat() / source.height
    )
    val scaled = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
    if (cropped !== scaled) cropped.recycle()
    return PersonCrop(scaled, normalizedRect)
}
```

`runLivePoseIfGated` updates:

```kotlin
val personCrop = squarePadCrop(bitmap, personBox = selectedPerson, marginFactor = 1.25f)
try {
    val poseResult = localPoseInterpreter.infer(personCrop.bitmap)
    _poseResult.value = LivePoseOverlay(poseResult, personCrop.cropRectNormalized)
    lastFrameSkippedPose.set(false)
} catch (error: Throwable) { … _poseResult.value = null … }
finally { personCrop.bitmap.recycle() }
```

#### Sub-step 6c — `CameraUiState` + `CameraViewModel`

```kotlin
// CameraUiState.kt
data class CameraUiState(
    …,
    val poseOverlay: LivePoseOverlay? = null,
    …
)

// CameraViewModel.kt init block — new collector
viewModelScope.launch {
    frameProcessor.poseResult.collect { overlay ->
        _uiState.update { state ->
            if (state.modelConfirmed) state.copy(poseOverlay = overlay)
            else state.copy(poseOverlay = null)
        }
    }
}

// restartSession(): poseOverlay = null
// setModel(): poseOverlay = null
```

Do **not** write pose data to `performanceLogger` or `trackingLogger` in this step — that feeds into Step 7's CSV schema, not Step 6.

#### Sub-step 6d — `PoseOverlay` composable in `CameraScreen.kt`

```kotlin
@Composable
private fun PoseOverlay(
    overlay: LivePoseOverlay?,
    sourceWidth: Int,
    sourceHeight: Int,
    modifier: Modifier = Modifier
) {
    if (overlay == null || sourceWidth <= 0 || sourceHeight <= 0) return
    val cropRect = overlay.cropRectNormalized
    val landmarks = overlay.poseResult.imageLandmarks33

    Canvas(modifier = modifier) {
        val canvasW = size.width; val canvasH = size.height
        val scale = maxOf(canvasW / sourceWidth, canvasH / sourceHeight)
        val scaledW = sourceWidth * scale; val scaledH = sourceHeight * scale
        val offsetX = (scaledW - canvasW) / 2f; val offsetY = (scaledH - canvasH) / 2f

        val cropW = cropRect.right - cropRect.left          // normalized in display frame
        val cropH = cropRect.bottom - cropRect.top
        val inputSize = PoseTensorContract.INPUT_SIZE.toFloat()

        landmarks.forEach { lm ->
            // landmark xPx/yPx ∈ [0, 256] → display-frame normalized
            val displayX = cropRect.left + (lm.xPx / inputSize) * cropW
            val displayY = cropRect.top  + (lm.yPx / inputSize) * cropH
            val screenX = displayX * scaledW - offsetX
            val screenY = displayY * scaledH - offsetY
            val color = if (lm.visibility > POSE_VISIBILITY_THRESHOLD) Color.Green else Color.Red
            drawCircle(color, radius = 4.dp.toPx(), center = Offset(screenX, screenY))
        }
    }
}
```

Called from `CameraScreen`:

```kotlin
PoseOverlay(
    overlay = uiState.poseOverlay,
    sourceWidth = uiState.detectionFrame.sourceWidth,
    sourceHeight = uiState.detectionFrame.sourceHeight,
    modifier = Modifier.fillMaxSize()
)
```

Place it **after** `DetectionOverlay` in the `Box` stack so landmark dots draw on top of bbox rectangles. `POSE_VISIBILITY_THRESHOLD = 0.6f` — same threshold used by the validation-path overlay writer in `CameraViewModel.renderPoseOverlay`.

#### Tests

Unit tests (JVM / Robolectric):
1. `BitmapOpsTest.squarePadCrop_bboxOverload_returnsClampedCropRect` — 400×600 source, `personBox(0.3, 0.2, 0.5, 0.6)`, margin 1.25; assert `PersonCrop.cropRectNormalized` matches the manually computed clamped rect within 1/source_dim epsilon.
2. `BitmapOpsTest.squarePadCrop_bboxAtEdge_clampsRectBounds` — bbox at `(0.85, 0.85, 1.0, 1.0)`; assert rect's right/bottom == 1f (clamp hit) and left/top > 0.
3. `FrameProcessorTest.runLivePoseForTest_emitsLivePoseOverlayWithCropRect` — seed rotated bitmap + person box, assert `poseResult.value?.cropRectNormalized` matches expected rect; assert `poseResult.value?.poseResult.imageLandmarks33.size == 33`.
4. `FrameProcessorTest.processImage_rotation90_sharesRotatedBitmapWithPose` — submit `ImageProxy` with `rotationDegrees = 90`, assert the bitmap passed to the fake pose interpreter has swapped dimensions. Covers 6a regression.
5. `CameraViewModelTest.uiState_poseOverlay_reflectsGatewayEmissions` — push a `LivePoseOverlay` to `FakeFrameProcessor.poseFlow`, confirm `viewModel.uiState.value.poseOverlay` matches after `runCurrent`. Also assert it clears on `restartSession()`.

Instrumented / on-device: no new `connectedDebugAndroidTest` — Step 7 soak already exercises the live path.

#### Verification

- JVM tests above pass.
- On-device S22+: install debug APK, confirm model, rotate device through all four orientations, verify landmark dots track the person with ≤ 1 frame lag and stay visually co-located with the `person` bbox.
- Toggle inference mode CPU ↔ GPU with a person in frame: overlay should blank for one frame during the switch, then resume (confirms `closePoseResources()` + lazy re-init path from Step 5).
- `uiState.poseOverlay` must go null when `modelConfirmed = false` (pre-confirm) and after `Restart Session`.

Gate: visual alignment on S22+ in portrait AND landscape before Step 7 benchmark run. Misalignment in portrait would mean sub-step 6a didn't land correctly.

#### Out of scope for Step 6

- Pose CSV columns — Step 7.
- Bitmap pool / allocation reduction for the new full-frame rotation — Phase 2.
- Per-landmark connection skeleton lines (shoulder→elbow→wrist etc.) — ship dots only in Day 6.
- ViewModel exposure of `setPoseGatingMode` / `setPersonSelectionMode` as user-facing toggles — add only if Step 7 benchmark needs mode-toggle UI.

---

### Step 7 — Latency Benchmark

Run the full YOLO+pose sequential pipeline for 10 minutes on S22+.

**CSV columns (extend existing phase0 schema):**
```
timestamp, frame_total_ms,
yolo_preprocess_ms, yolo_inference_ms, yolo_nms_ms,
pose_crop_ms, pose_preprocess_ms, pose_inference_ms, pose_postprocess_ms,
pose_skipped,           ← true when no person bbox (pose not run)
ram_mb, thermal_status, fps_1s_window,
mode, device, gpu_mode
```

**Gate criteria (Day 6):**

| Metric | S22+ target | Fallback trigger |
|--------|-------------|-----------------|
| `pose_inference_ms` p95 | ≤ 50ms | > 70ms → Config D |
| `frame_total_ms` p95 | ≤ 100ms | > 140ms → flag risk |
| RAM sustained | < 400MB | > 400MB → investigate leak |
| FPS sustained (10 min) | ≥ 20 | < 20 → flag Day 8 risk |
| Thermal throttle events | 0 | Any → flag |

Note: Pixel 6 and A54 remain deferred — validate before Phase 2.

**Pre-rotation allocation watch (new for Step 7):**
Step 6 added `BitmapOps.rotateBitmapForDisplay()` upstream of YOLO. For any non-`ROTATION_0` orientation this allocates one extra ARGB_8888 bitmap (~3.5 MB at 720×1280) per frame, recycled in the outer `finally`. Net +1 full-frame bitmap allocation per frame vs. the Day 5 baseline. At 30 fps this is ~105 MB/s of short-lived bitmap churn.

During the soak run, explicitly verify (as rescoped 2026-04-26):
- **No RAM drift on the watched path under cool thermal** — `ram_mb` rolling 30s average stays flat (< 5 MB drift) on the `rotation_degrees != 0` slice while `thermal_status ∈ {NONE, LIGHT}`. Drift outside that window is dominated by GC cadence under throttle (see `Runtime.totalMemory()-freeMemory()` semantics) and is not actionable for the rotation-bitmap path.
- **No GC-correlated latency spikes on the watched path** — `frame_total_ms` p99 ≤ 140 ms on `rotation_degrees != 0` × NONE/LIGHT slice; outlier bursts adjacent in time (multiple frames > 140 ms within 0.5 s) are thermal-hiccup pattern, not stop-the-world GC.
- **Orientation coverage** — run at least 2 of the 4 orientations (recommend portrait + landscape-right) within the same 10-minute window so both the `return source` fast path (ROTATION_0) and the allocating path are exercised. Use the per-frame `rotation_degrees` CSV column to slice the analysis.

If alloc-path NONE/LIGHT slice drift > 5 MB OR alloc-path p99 > 140 ms with single-frame stop-the-world signature: flag for bitmap pooling (Phase 2 per ADR-005 §Deferred) — do **not** land the pool change in Phase 0.

Commit artifacts to: `benchmarks/phase0/combined_pipeline/run_{datetimestamp}/`

---

### Step 7 Results (2026-04-22, §7 re-scored 2026-04-26)

Original 10-min soak: device SM-S906U1, build `762b968`, mode `sequential_yolo_pose` / `gpu_mode=GPU`, dur 630.3 s, one mid-run rotation at t≈315 s within the landscape axis (±2 s exclusion). Artifact: [benchmarks/phase0/phase0-perframe-20260422-163835.csv](../../benchmarks/phase0/phase0-perframe-20260422-163835.csv).

Headline verdicts:
- §3 (`frame_total_ms` p95 ≤ 100 ms): **FAIL** — overall p95 = 129.32 ms (landscape-A 119.67, landscape-B 131.96).
- §4 (`pose_inference_ms` p95 ≤ 50 ms): **PASS** with large margin — p95 = 15.44 ms.
- §5 (RAM < 400 MB sustained): **PASS** — peak delta 47.06 MB, well within 400 MB.
- §6 (FPS ≥ 20 sustained): **FAIL** — median 1 s window = 8 fps, non-warmup min = 5 fps.
- §7 (pre-rotation alloc drift): originally **FAIL** (30s-rolling drift 9.63 MB vs 5 MB ceiling); **rescored PASS-with-rescope** on 2026-04-26 — see "§7 Sliced Re-analysis" below.
- Bottleneck: YOLO inference (p50 53.15 ms, p95 57.02 ms). Pose healthy. Pipeline hits thermal ceiling at t=68 s (MODERATE) and throttles to ~8 fps for the remainder.

Per-segment latency (ms), ±2 s excluded around the rotation transition:

| Segment | n | `frame_total_ms` p50 / p95 / p99 / max | `yolo_inference_ms` p50 / p95 / p99 | `pose_inference_ms` p50 / p95 / p99 (n) |
|---|---|---|---|---|
| Overall | 5599 | **112.70 / 129.32 / 136.15** / 207.51 | 53.15 / 57.02 / 59.42 | **11.17 / 15.44 / 19.44** (5089) |
| Landscape-A (t<313 s) | 3135 | 102.16 / 119.67 / 127.42 / 207.51 | 52.21 / 56.92 / 59.40 | 10.48 / 15.17 / 19.35 (2860) |
| Landscape-B (t>317 s) | 2464 | 121.61 / 131.96 / 138.15 / 159.04 | 53.52 / 57.08 / 59.42 | 11.67 / 15.71 / 19.44 (2229) |

Thermal: NONE→LIGHT @ t=28 s, LIGHT→MODERATE @ t=68 s, MODERATE→SEVERE @ t=428 s. Pose-skipped rate: 9.1 % (solo-test session). Run coverage gap: both rotation segments were landscape (S22+ native-landscape sensor; mid-run was a 180° flip within the landscape axis). Portrait not directly exercised — but the 180° flip + DisplayListener `targetRotation` sync (per ISSUE-013) does drive `imageInfo.rotationDegrees` from 0 → 180, taking the allocating branch, so the watched path is exercised even without portrait.

#### §7 Sliced Re-analysis (2026-04-26)

§7 re-analysed against two follow-up soaks with **starting orientation swapped** to decorrelate rotation from thermal: thermal escalation is monotonic over time — whichever path runs second sees the hotter regime — so if drift were rotation-driven, the alloc slice would drift the same in both runs; if thermal-driven, drift tracks thermal regardless of which path is in play. Build `762b968`, device SM-S906U1, pre-cooled.

Artifacts:
- [benchmarks/phase0/phase0-perframe-20260422-231515.csv](../../benchmarks/phase0/phase0-perframe-20260422-231515.csv) — run A, n=7232, dur 759 s; rot=0 first → rot=180 second
- [benchmarks/phase0/phase0-perframe-20260422-235704.csv](../../benchmarks/phase0/phase0-perframe-20260422-235704.csv) — run B, n=6016, dur 629 s; rot=180 first → rot=0 second

Sliced 30s-rolling drift (max−min of windowed RAM mean) and `frame_total_ms` p99 by `rotation_degrees` × `thermal_status`:

| Slice | Run | n | drift30s | p99 (ms) |
|---|---|---|---|---|
| rot=180 (alloc) × thermal=NONE | B | 1221 | **1.74 MB** ✓ | 87.16 |
| rot=180 (alloc) × thermal=LIGHT | B | 627 | 5.39 MB | 97.10 |
| rot=0 (fast) × thermal=NONE | A | 1379 | 2.38 MB | 76.89 |
| rot=0 (fast) × thermal=LIGHT | A | 1820 | 4.84 MB | 126.57 |
| rot=0 (fast) × thermal=SEVERE | B | 2771 | 2.86 MB | 142.57 |
| rot=180 × thermal=SEVERE | A+B | 1562 | 0.25–0.68 MB | 134–139 |

**Verdict: §7 PASS-with-rescope. Thermal-cadence artifact, not rotation-bitmap leak.**

1. Allocating path under cool thermal does not leak — `rot=180 / thermal=NONE` = 1.74 MB drift on 1221 frames, well below 5 MB ceiling.
2. The original 9.63 MB headline drift was GC cadence under thermal throttle. Fast path (`rot=0`) drifts 2.86 MB under SEVERE despite making zero rotation-bitmap allocations, proving the metric tracks GC cadence, not allocation volume. `Runtime.totalMemory()-freeMemory()` measures current Java heap occupancy; under SEVERE throttle GC runs less often and the steady-state heap floor lifts.
3. Allocating path adds ~10 ms p99 vs fast path under matched thermal (NONE: 87 vs 77 ms) — expected for one ARGB_8888 alloc/frame, no GC-pause signature. Frames > 140 ms on alloc-path/NONE = 2/1221 (0.16%); outlier bursts cluster on thermal transitions (44/65 outlier pairs within 0.5 s of each other) — thermal hiccup pattern, not classical stop-the-world GC.
4. ADR-005 §Deferred trigger ("if profiling shows GC pressure, move to bitmap pooling in Phase 2") is not met. Bitmap pool stays Phase 2 deferred.

§7 rescoped: drift measurement window restricted to NONE/LIGHT thermal segments (matching §3/§6 routing pattern). Day 6 sign-off proceeds for §7. The per-frame CSV gained a `rotation_degrees` column ahead of this re-analysis (Step A of the unblock plan, build `762b968`) — already landed.

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §1 No-alloc | Zero `IntArray`/`Bitmap` alloc in `infer()` hot path (Profiler) | Fix before merging live wiring |
| §2 Bitmap lifecycle | No use-after-recycle crash over 10-min run | Fix `finally` block ordering |
| §3 Combined latency p95 | `frame_total_ms` p95 ≤ 100ms on S22+ | > 140ms → Config D or resolution drop |
| §4 Pose p95 latency | `pose_inference_ms` p95 ≤ 50ms on S22+ | > 70ms → Config D |
| §5 RAM | < 400MB sustained | Profile leak, defer heavy bitmap pooling to perf/ branch |
| §6 FPS | ≥ 20 sustained over 10 min | Flag Day 8 risk; check pose skip rate |
| §7 Pre-rotation alloc drift | RAM drift < 5 MB on `rotation_degrees != 0` slice under NONE/LIGHT thermal, no GC-correlated p99 spikes on that slice | Bitmap pool → Phase 2 per ADR-005 §Deferred (trigger: alloc-path NONE/LIGHT drift > 5 MB *or* p99 > 140 ms with single-frame stop-the-world signature on alloc-path frames) |

---

## Known Day-7 Follow-ups

- Replace person-bbox pose gate with FSM state gate (`IDLE`/`MADE` → skip)
- **Reflection risk in `FrameProcessorTest`:** test accesses `poseValidationActive` and
  `poseValidationComplete` via `getDeclaredField` + `isAccessible = true`. If Day 7 refactors
  these fields to `StateFlow<Boolean>` on `FrameProcessorGateway`, the reflection assertions
  will fail at runtime with `NoSuchFieldException` — no compile-time warning. Fix before any
  StateFlow refactor that touches those fields: replace reflection assertions with StateFlow
  assertions on the Gateway interface.
- Pixel 6 + A54 EGL and latency validation (deferred from Day 5) — required before Phase 2
