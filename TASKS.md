# Tasks — Phase 0 Spike

Last updated: 2026-04-26

## Active branch
`spike/day6-combined-pipeline` (branch from `main`)

> Days 1–5 complete. See archived plans: `docs/plans/day1-2-plan.md`, `docs/plans/day3+weekend-plan.md`, `docs/plans/day4_kalman-filter-tracker.md`, `docs/plans/day5-pose-isolated.md`.

---

## Active 🔲 Day 6: Sequential GPU Combined Pipeline

> Branch: `spike/day6-combined-pipeline`
> Plan: `docs/plans/day6-combined-pipeline.md`
> Architecture: `docs/decisions/005-sequential-gpu-inference-pipeline.md`

### Pre-requisites from Day 5 ✅
1. ✅ Migrate `PoseLandmarkInterpreterTest` to JUnit 5 (`org.junit.jupiter.api.Test`, `assertThrows<T> { }`) — currently JUnit 4; CONTEXT.md specifies JUnit 5 *(Day 5 item 15)*
2. ✅ Fix `PoseLandmarkInterpreter.infer()` no-alloc: pre-allocated `TensorImage`/`ImageProcessor` hot path now in place; `infer()` enforces 256x256 caller contract.
   - Profiler gate dropped from Day 6 sign-off (2026-04-22). Step 7 soak provides indirect verification: `pose_inference_ms` p50/p99 = 11.17/19.44 ms spread is inconsistent with per-frame ~1 MB allocations. Direct profiler capture deferred to Phase 2 if pose-side allocation regression ever suspected. See CONTEXT.md "Day 6 Step 2 Verification Note".
3. ✅ Fix bitmap lifecycle in `processImage()`: `bitmap.recycle()` moved to outer `finally` ([FrameProcessor.kt:574](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L574)); live path `crop.recycle()` in `runLivePoseIfGated` finally ([line 612](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L612)).
4. ✅ Wire pose `useGpu` through `InferenceMode` — `initializePoseInterpreterIfNeeded()` reads `currentMode.get() != InferenceMode.CPU` ([FrameProcessor.kt:288](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L288)); `switchInterpreter` forces pose re-init on mode change ([line 343](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L343)).

### Core implementation ✅
5. ✅ `BitmapOps.kt` — bbox-aware `squarePadCrop(src, box, marginFactor=1.25f)` overload added ([BitmapOps.kt:36-81](app/src/main/java/com/courtvision/spike/pipeline/BitmapOps.kt#L36-L81)); output always 256×256 per `PoseTensorContract.INPUT_SIZE`, with defensive clamp-to-1px for degenerate bboxes. `BitmapOpsTest` covers withBbox/bboxAtEdge/bboxFullFrame cases.
5a. ✅ `FrameContracts.kt` — `PoseGatingMode` + `PersonSelectionMode` enums added ([FrameContracts.kt:53-62](app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt#L53-L62)); `PERSON_CLASS_ID = 2` and `SHOOT_CLASS_ID = 4` constants in `FrameProcessor.Companion` ([lines 1081-1082](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L1081-L1082)); gateway setters added ([FrameProcessorGateway.kt:20-21](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessorGateway.kt#L20-L21)).
6. ✅ `FrameProcessor.kt` — live pose wired via `runLivePoseIfGated` ([FrameProcessor.kt:583-614](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L583-L614)); `selectPersonBox` and `shouldRunPose` dispatch ADR-006 enums; `personBox == null` and gating-mode checks now split (single-responsibility). `FSM_GATED` / `SHOOT_CLASS_GATED` silently fall back to `EVERY_FRAME_WITH_PERSON` per ADR-006 §Consequences.
7. ✅ `FrameProcessorGateway.kt` — `val poseResult: StateFlow<PoseResult?>` added ([line 11](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessorGateway.kt#L11)); `_poseResult: MutableStateFlow<PoseResult?>` in `FrameProcessor` ([line 80](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L80)).
8. ✅ `CameraViewModel.kt` — 5th collector block gates `poseOverlay` on `modelConfirmed` ([CameraViewModel.kt:151-161](app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt#L151-L161)); cleared on `setModel()` ([line 286](app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt#L286)) and `restartSession()` ([line 311](app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt#L311)). `CameraUiState.poseOverlay: LivePoseOverlay?` added ([CameraUiState.kt:32](app/src/main/java/com/courtvision/spike/camera/CameraUiState.kt#L32)).
9. ✅ `CameraScreen.kt` — `PoseOverlay` composable between `DetectionOverlay` and `MetricsOverlay` ([CameraScreen.kt:122-127](app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt#L122-L127)); landmark dots colored by `visibility > POSE_VISIBILITY_THRESHOLD = 0.6f` ([line 839](app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt#L839)); uses FILL_CENTER + `cropRectNormalized` to map 256×256 pose coords back to canvas space. Gateway contract upgraded from `PoseResult?` to `LivePoseOverlay?` ([FrameProcessorGateway.kt:11](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessorGateway.kt#L11)); rotation moved upstream via `rotateBitmapForDisplay` ([BitmapOps.kt:37-53](app/src/main/java/com/courtvision/spike/pipeline/BitmapOps.kt#L37-L53)), `Rot90Op` dropped from YOLO chain.

### Tests 🔲
10. ✅ `FrameProcessorTest` — live pose path tests added: `processImage_liveYoloWithPerson_runsPoseAndEmitsResult`, `processImage_liveYoloNoPerson_skipsPoseAndLeavesPoseResultNull`, `processImage_bitmapRecycledInOuterFinally_noCrash`, `setPoseGatingMode_FSM_GATED_currentlyFallsBackToEveryFrame`, `initializePoseInterpreter_whenYoloGpu_usesGpuDelegate`, `switchInterpreterCpuToGpu_closesAndReinitializesPose` — use `FakePoseInferenceEngine` + `poseInterpreterFactory` injection seam.
11. ✅ `poseValidationActive`/`poseValidationComplete` reflection-based assertions preserved unchanged ([FrameProcessorTest.kt:282-344](app/src/test/java/com/courtvision/spike/pipeline/FrameProcessorTest.kt#L282-L344)); no StateFlow refactor of those fields in Day 6. Risk still tracked in item 17 for Day 7.

### Benchmark 🔲
12. ✅ Ran 10-minute soak test on S22+ with full sequential YOLO+pose live pipeline (2026-04-22). Artifact: [benchmarks/phase0/phase0-perframe-20260422-163835.csv](benchmarks/phase0/phase0-perframe-20260422-163835.csv). Coverage: both rotation segments were landscape (S22+ native-landscape sensor); 180° flip within the landscape axis exercises the allocating rotation path via DisplayListener `targetRotation` sync (per ISSUE-013). Headline `ram_mb` 30s-rolling drift = 9.63 MB; rescoped 2026-04-26 via order-swapped soaks ([231515.csv](benchmarks/phase0/phase0-perframe-20260422-231515.csv) + [235704.csv](benchmarks/phase0/phase0-perframe-20260422-235704.csv)) — verdict thermal-cadence artifact, not rotation-bitmap leak.
13. ✅ Recorded combined latency result in [CONTEXT.md](CONTEXT.md) → "Day 6 Step 7 Benchmark Results". Gate outcomes: §3 FAIL (p95=129 ms) → Day 7 perf, §4 PASS (p95=15 ms), §5 PASS, §6 FAIL (median 8 fps) → Day 7 perf, §7 PASS-with-rescope (alloc-path × NONE/LIGHT drift = 1.74 MB; bitmap pool stays Phase 2 per ADR-005 §Deferred). Full slice table + methodology in [docs/plans/day6-combined-pipeline.md](docs/plans/day6-combined-pipeline.md) "§7 Sliced Re-analysis (2026-04-26)".

**Gate (S22+):** `pose_inference_ms` p95 ≤ 50ms · `frame_total_ms` p95 ≤ 100ms · RAM < 400MB · FPS ≥ 20 sustained · zero thermal throttle events
**Fallback:** Config D — pose CPU 4 threads if `pose_inference_ms` p95 > 70ms (item 4 above must be done first)

---

### Carried to Day 6 🔲
15. ✅ `PoseLandmarkInterpreterTest` JUnit 5 migration completed in Day 6 Step 0 (see plan doc).
16. ✅ Live-path `bitmap.recycle()` deferred to outer `finally` ([FrameProcessor.kt:574](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L574)); `crop.recycle()` in `runLivePoseIfGated` finally ([line 612](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L612)). Validation-path recycle in `processPoseValidationIfNeeded()` unchanged.
17. ⚠️ **Risk (still open for Day 7)** — `FrameProcessorTest` accesses `poseValidationActive`/`poseValidationComplete` via reflection. Day 6 left these fields as `AtomicBoolean` (additive `poseResult` StateFlow only), so reflection still resolves. Any Day 7 refactor of these fields to `StateFlow<Boolean>` must migrate reflection assertions first.
18. ✅ Pose `useGpu` now derived from `currentMode.get() != InferenceMode.CPU` ([FrameProcessor.kt:288](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L288)); `switchInterpreter` calls `closePoseResources()` so pose lazy-reinits with matched delegate on the next frame ([line 343](app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L343)).
19. ✅ Replace `getPixels` pixel-copy loop in `PoseLandmarkInterpreter.infer()` with pre-allocated TFLite Support path (`TensorImage` + `ImageProcessor(NormalizeOp)`), no per-frame pixel-array copy in code path.
    - **Gate:** Profiler capture gate dropped from Day 6 sign-off (2026-04-22) in favour of Step 7 soak indirect verification (pose p50/p99 spread = 8 ms, rules out GC-triggering per-frame allocations). See CONTEXT.md "Day 6 Step 2 Verification Note".

**Device:** Samsung Galaxy S22+ (only available device). Pixel 6 and A54 deferred — validate before Phase 2.
**Gate:** `pose_inference_ms` p95 ≤ 50ms on S22+ (strong pass); 50–70ms = marginal (flag A54 risk); > 70ms → Config D fallback.
**Fallback:** Config D — pose on CPU, 4 threads (see ADR-005 §Deferred). Requires Blocker B2 (`useGpu=false`) first.

---

## Out of scope
- Shot detection FSM (Day 7)
- Pixel 6 + A54 EGL and latency validation (deferred from Day 5 — validate before Phase 2)
