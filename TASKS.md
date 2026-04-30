# Tasks — Phase 0 Spike

Last updated: 2026-04-27

## Next planned branches
`perf/day6-2-trivial-perf-wins` → `perf/day6-1-qnn-npu-pipeline` → `perf/day6-3-gpu-serialization-cpu-fallback`

> Days 1–5 complete. Archived plans: `docs/plans/day1-2-plan.md`, `docs/plans/day3+weekend-plan.md`, `docs/plans/day4_kalman-filter-tracker.md`, `docs/plans/day5-pose-isolated.md`.

---

## Day 6 Closed ✅

Day 6 sequential combined pipeline is complete and closed. The detailed execution record remains in `docs/plans/day6-combined-pipeline.md`.

Outcome summary:
- Pose path passed and is not the bottleneck (`pose_inference_ms` p95 = 15 ms).
- Combined latency and FPS gates failed, so follow-on perf work moved to Day 6.x.
- Baseline results driving the next queue:
  - YOLO p50 = 53 ms
  - `frame_total_ms` p95 = 129 ms
  - FPS median = 8
  - thermal `MODERATE` at t=68 s

---

## Active 🔲 Day 6.x: Post-Day-6 Perf Follow-up

Purpose: close the perf gaps identified after Day 6 combined-pipeline validation.

Source gap analysis: `research-reports/Claude-qualcomm-demo-gap-analysis.md`

### Execution order

1. `perf/day6-2-trivial-perf-wins`
   - Plan: `docs/plans/day6-2-trivial-perf-wins.md`
   - Gaps: 3, 4, 5
   - Why first:
     - no new dependencies
     - smallest risk
     - establishes `buildSustainedSpeedGpuDelegate()`
     - improves the GPU baseline before QNN benchmarking
   - Unblocks:
     - shared GPU helper used by Day 6.1 QNN GPU sub-delegate path
     - serialization upgrade point used by Day 6.3

2. `perf/day6-1-qnn-npu-pipeline`
   - Plan: `docs/plans/day6-1-qnn-npu-pipeline.md`
   - Gaps: 1
   - Why second:
     - primary bottleneck fix
     - adds `QNN_NPU`
     - adds `nativeLibraryDir` / `modelCacheDir` plumbing
     - adds stable model-token logic that Day 6.3 can reuse
   - Unblocks:
     - real NPU path for the S22+ target device
     - cache-dir and model-token reuse for Day 6.3 GPU shader serialization

3. `perf/day6-3-gpu-serialization-cpu-fallback`
   - Plan: `docs/plans/day6-3-gpu-serialization-cpu-fallback.md`
   - Gaps: 2, 7
   - Why last:
     - upgrades the helper introduced in Day 6.2
     - benefits from cache-dir/model-token support introduced in Day 6.1
     - cleans up the legacy NNAPI path after QNN exists as the true NPU option
   - Unblocks:
     - faster repeat GPU cold starts
     - explicit XNNPack CPU fallback
     - NNAPI compatibility cleanup without duplicating earlier plumbing work

---

## Dependency / Overlap Notes

Concrete overlap driving the order:
- shared delegate construction in `GpuDelegateProbe.kt`
- shared runtime switching in `FrameProcessor.kt`
- shared pose GPU delegate creation in `PoseLandmarkInterpreter.kt`
- cache-dir / model-token plumbing in `FrameProcessor` and `CameraViewModel`

Sequencing rules:
- Day 6.3 assumes Day 6.2’s shared GPU helper exists.
- Day 6.3 should preferably reuse Day 6.1’s `modelCacheDir` and model-token plumbing.
- Day 6.1 should land before Day 6.3 to avoid duplicate cache-plumbing work.

Branching rule:
- Track the three efforts as separate serial work items, not parallel implementation branches.

---

## Per-Plan Queue

### Day 6.2 — Trivial no-dep wins

- Branch: `perf/day6-2-trivial-perf-wins`
- Plan: `docs/plans/day6-2-trivial-perf-wins.md`
- Gaps addressed: Gap 3 (`INFERENCE_PREFERENCE_SUSTAINED_SPEED`), Gap 4 (pre-allocated `TensorImage` for YOLO), Gap 5 (`setAllowBufferHandleOutput(true)`)
- Prerequisites: Day 6 baseline available; no new dependency coordination required
- Expected outcome: cleaner GPU hot path, reduced alloc pressure, and improved sustained GPU baseline before QNN testing
- Validation gate summary: GPU path still works; YOLO p50 stays near the Day 6 baseline; thermal trajectory is at least not worse than Day 6

### Day 6.1 — QNN NPU pipeline

- Branch: `perf/day6-1-qnn-npu-pipeline`
- Plan: `docs/plans/day6-1-qnn-npu-pipeline.md`
- Gaps addressed: Gap 1 (QNN NPU delegate)
- Prerequisites: Day 6.2 merged or available; S22+ available; QNN dependencies added; `nativeLibraryDir` and `modelCacheDir` wiring ready — **Step 0 prerequisite met (INT8 model ready 2026-04-28)**
- Expected outcome: YOLO moves from GPU to Hexagon HTP and becomes the primary latency reduction path
- Validation gate summary: QNN probe succeeds on S22+; `QNN_NPU` mode switches cleanly; CSV records `gpu_mode=QNN_NPU`; benchmark determines whether INT8 is needed later

### Day 6.3 — GPU shader serialization + CPU fallback cleanup

- Branch: `perf/day6-3-gpu-serialization-cpu-fallback`
- Plan: `docs/plans/day6-3-gpu-serialization-cpu-fallback.md`
- Gaps addressed: Gap 2 (GPU shader serialization), Gap 7 (NNAPI → XNNPack CPU fallback)
- Prerequisites: Day 6.2 helper exists; Day 6.1 cache-dir/model-token plumbing preferably available
- Expected outcome: faster second GPU cold start, explicit CPU fallback behavior, and NNAPI downgraded to a compatibility alias
- Validation gate summary: second GPU cold start is materially faster than first; CPU path explicitly uses XNNPack; `NNAPI` selection routes to GPU with warning/compat behavior

> Detailed implementation steps stay in the plan docs and are not duplicated here.

## Carried Day 7 Risk

- `FrameProcessorTest` still accesses `poseValidationActive` and `poseValidationComplete` via reflection.
- Any Day 7 refactor that replaces those fields with `StateFlow<Boolean>` must migrate those assertions first, or the test will fail at runtime.

---

## NOTE:

- Update gap source file on gap closing
- Update day6-x plan files on steps completion / gaps closing.

## Validation Order

1. Validate Day 6.2 GPU baseline before starting Day 6.1.
2. Validate QNN behavior and benchmark results after Day 6.1 before starting Day 6.3.
3. Validate Day 6.3 cold-start improvement and NNAPI compatibility alias last.

Acceptance checkpoints:
- After Day 6.2:
  - GPU path still works
  - no meaningful YOLO p50 regression
  - thermal trajectory is at least not worse than Day 6 baseline
- After Day 6.1:
  - QNN probe succeeds on S22+
  - `QNN_NPU` mode switches cleanly
  - `gpu_mode=QNN_NPU` appears in CSV
  - benchmark determines whether INT8 is needed later
- After Day 6.3:
  - second GPU cold start is materially faster than first
  - CPU path explicitly uses XNNPack
  - `NNAPI` selection routes to GPU with warning/compat behavior

---

## Out of Scope / Deferred

- Gap 6 (rotate-after-resize) remains deferred
- INT8 path confirmed: split-output w8a8 model validated 2026-04-28 (run `spike_qai_yolo11n_640_5-class_04-28-2026_int8_14`, mAP50=0.9143); FP16 GPU asset remains active for non-QNN delegates
- full removal of `InferenceMode.NNAPI` remains a later cleanup, not part of this queue
- Shot detection FSM remains Day 7 work, not part of Day 6.x perf follow-up
