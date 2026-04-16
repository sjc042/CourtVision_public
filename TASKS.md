# Tasks — Phase 0 Spike

Last updated: 2026-04-14

## Active branch
`spike/day5-pose-isolated` (branch from `main`)

> Days 1–4 complete. See archived plans: `docs/plans/day1-2-plan.md`, `docs/plans/day3+weekend-plan.md`, `docs/plans/day4_kalman-filter-tracker.md`.

---

## Active 🔲 Day 5: Pose Landmark Model — Isolated Validation

> Branch: `spike/day5-pose-isolated`
> Plan: `docs/phase0-spike-plan.md` (Day 5 section)
> Architecture: `docs/decisions/005-sequential-gpu-inference-pipeline.md`

### Completed ✅
1. ✅ Extract `pose_landmarks_detector.tflite` from MediaPipe `.task` bundle (build-time step; document in build README) — **lite variant** (`pose_landmarker_lite.task`, ~4 MB FP16)
2. ✅ `PoseLandmarks.kt` — data classes (`PoseImageLandmark`, `PoseWorldLandmark`, `PoseResult`) and `PoseTensorContract` constants
3. ✅ `PoseLandmarkInterpreter.kt` — shape-based output index resolution, sigmoid decode, TFLite Support Library preprocessing (ResizeOp + NormalizeOp), stage latency timing (pre/inf/post/total)
4. ✅ `FrameProcessor.kt` — dormant `poseInterpreter` field initialized on `consumerDispatcher` via `initializePoseInterpreterIfNeeded()`
5. ✅ `PoseLandmarkInterpreterTest.kt` — 7 JVM unit tests: decode (195→39×5, 117→39×3), take(33) subsetting, sigmoid direction, shape-based index resolution + out-of-order, resolution failure, input validation

### Required before validation run 🔲
6. ✅ Migrate validation runner to `consumerDispatcher` integration
   - **Old:** `CameraViewModel.runDay5PoseIsolatedValidation()` on separate `poseValidationDispatcher`; new `PoseLandmarkInterpreter` constructed inside runner; reads via raw `File` I/O from `/sdcard/Download/pose_isolate`
   - **New:** `FrameProcessor.submitPoseValidationBatch(bitmaps)` enqueues into `poseValidationQueue`; `processImage()` pops one bitmap per frame after YOLO; results emitted to `poseValidationResultChannel`
   - See revised Step 3.5 in `docs/plans/day5-pose-isolated.md`
7. ✅ Replace image input with system media picker
   - **Old:** Raw `File` I/O to `/sdcard/Download/` — broken on `targetSdk=34`; manifest declares only `CAMERA`
   - **New:** SAF/ContentResolver multi-select picker; bitmaps decoded on `Dispatchers.IO`; submitted to `FrameProcessor`
8. ✅ Add `worldLandmarkVisibility(index: Int)` accessor to `PoseResult` — Blocker B1, required before Day 6 angle-gating code
9. ✅ Add `useGpu: Boolean = true` parameter to `PoseLandmarkInterpreter` constructor — Blocker B2, required for ADR-005 Config D CPU fallback
10. ✅ Pull forward `squarePadCrop()` + tests from Day 6
    - **Why now:** Picker-based validation feeds arbitrary-aspect photos; without letter-box padding the model skews landmarks on tall/wide inputs.
    - **Implementation:** Added bbox-agnostic `squarePadCrop(source)` for Day 5 full-frame validation; Day 6 will crop YOLO person bbox upstream, then reuse the same function.

### Validation gates ✅
11. ✅ Verify two `GpuDelegate` interpreters (YOLO + pose) coexist on `consumerDispatcher` — ADR-005 Verification Gate §1
    - Result: PASS on S22+ (`SM-S906U1`) with no runtime `TfLiteGpuDelegate ... must run on the same thread` log observed during validation run
12. ✅ Benchmark sequential YOLO+pose inference on S22+
    - **Old:** Isolated pose latency (YOLO not running)
    - **New:** YOLO-warm sequential; gate applies to `pose_inference_ms` p95 column (≤50ms strong pass, ≤70ms marginal, >70ms → Config D)
    - Result: `pose_inference_p95_ms=10.774` (**PASS_STRONG**) from `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_summary.txt`
    - Artifacts committed under: `benchmarks/phase0/pose_validation/inference_20260414_004536/`
13. ✅ Verify decoded landmarks visually on selected test frames (≥8/10 plausible positions — Gate §2)
    - Result: PASS (manual review confirmed plausibility threshold met)
14. ✅ Record EGL context sharing result (shared vs. separate) in `CONTEXT.md`

### Deferred to Day 6 🔲
15. 🔲 Align `PoseLandmarkInterpreterTest` to JUnit 5 (`org.junit.jupiter.api.Test`, `assertThrows<T> { }`) — currently uses JUnit 4 APIs; CONTEXT.md specifies JUnit 5
16. 🔲 Update `processImage()` bitmap lifecycle: defer `bitmap.recycle()` to after `squarePadCrop()`; add `personCrop.recycle()` after pose `ImageProcessor.process()`
    - **Note:** Validation path now recycles the padded intermediate crop in `processPoseValidationIfNeeded()`. Day 6 live camera path still needs the deferred outer `bitmap.recycle()` shift.
17. ⚠️ **Risk** — `FrameProcessorTest` accesses `poseValidationActive`/`poseValidationComplete` via reflection (`getDeclaredField` + `isAccessible = true`). If Day 6 refactors these fields to `StateFlow<Boolean>` on `FrameProcessorGateway` (required for ViewModel observation), the reflection-based assertions will fail at runtime with `NoSuchFieldException` — no compile-time warning. Fix: replace with StateFlow assertions on the Gateway interface before the StateFlow refactor lands.
18. 🔲 Wire pose interpreter GPU/CPU mode through `InferenceMode` selection — `initializePoseInterpreterIfNeeded()` currently hardcodes `useGpu = true`; Day 6 must pass the selected mode (GPU or Config D CPU fallback) so pose delegate matches the YOLO delegate choice. Requires Blocker B2 (`useGpu` parameter, already done) and the same `CompatibilityList.bestOptionsForThisDevice` path used by `switchInterpreter`.
19. 🔲 Replace `getPixels` pixel-copy loop in `PoseLandmarkInterpreter.infer()` with a pre-allocated `Canvas` rescale — **§4 No-alloc gate**
    - **Current:** `crop256.getPixels(pixels, ...)` allocates a new `IntArray(256 * 256)` on every inference call
    - **Fix:** pre-allocate a `Canvas`-backed `Bitmap` (ARGB_8888, 256×256) in `init`; use `Canvas.drawBitmap(src, null, dstRect, null)` + `ByteBuffer.rewind()` to fill the existing `inputBuffer` in-place; alternatively keep TFLite Support Library `ImageProcessor` path if benchmarks confirm it avoids the allocation
    - **Gate:** no `IntArray` or `Bitmap` allocation on the hot path inside `infer()` (verify with Android Studio Profiler before merging Day 6 live wiring)

**Device:** Samsung Galaxy S22+ (only available device). Pixel 6 and A54 deferred — validate before Phase 2.
**Gate:** `pose_inference_ms` p95 ≤ 50ms on S22+ (strong pass); 50–70ms = marginal (flag A54 risk); > 70ms → Config D fallback.
**Fallback:** Config D — pose on CPU, 4 threads (see ADR-005 §Deferred). Requires Blocker B2 (`useGpu=false`) first.

---

## Out of scope
- Shot detection FSM (Day 7)
