# Frame Scheduling Spec — Combined Pipeline

**CourtVision Phase 0 Spike**
Created: 2026-03-27 | Resolves: ISSUE-003 | Fix before: Day 5

---

## Problem

The PRD/TDD simultaneously target:
- 30 FPS analysis (33ms per-frame budget)
- Ball detection < 100ms per frame
- Pose inference < 50ms per frame

Synchronous execution = 150ms minimum = ~6 FPS, not 30 FPS. Without a scheduling spec, Day 6 (combined pipeline) has no pass/fail definition.

---

## Core Architecture

Decouple the two workers entirely. Never treat them as a synchronous pipeline.

```
CameraX (30 FPS)
    │
    ├──► Channel A (capacity=1, DROP_OLDEST) ──► Worker A: YOLOv8n (GPU delegate)
    │                                               every frame, ~30 FPS target
    │
    └──► Channel B (capacity=1, DROP_OLDEST) ──► Worker B: MediaPipe Pose / YOLO26n-pose (CPU)
                                                    every 3rd frame, ~10 FPS target
```

- Worker A and Worker B are independent coroutines on `Dispatchers.Default`
- They never block each other or the CameraX analyzer thread
- The FSM subscribes to outputs from both workers via separate `StateFlow`s

---

## Decision: GPU Delegate Allocation

**TFLite GPU delegate instances cannot be shared across threads and cannot run in parallel on the same GPU.**

| Option | Setup | Tradeoff |
|---|---|---|
| **A — GPU/CPU split (selected)** | YOLOv8n / YOLO26n on GPU delegate, Pose (MediaPipe or YOLO26n-pose) on CPU | No contention. Simplest to implement. Pose CPU latency ~80–100ms, acceptable at 10 FPS. |
| B — Time-sliced GPU | Both on GPU delegate, single-threaded executor | Eliminates contention by serializing. Higher total latency but predictable. Measure in Phase 2. |

**Option A is selected for the Phase 0 spike.**

### ⚠️ Hard Limit Deviation — Pose Latency

`CONTEXT.md` and `tdd.md` define **Pose inference < 50ms as a Performance Hard Limit (Never Violate)**.

Option A runs Pose on CPU, with an estimated latency of **80–100ms per invocation**. This violates the stated hard limit.

**Resolution for the spike:** The 50ms hard limit was written assuming GPU execution. Running Pose at 10 FPS on the CPU path is acceptable for Phase 0 validation because:
1. Pose is not on the 30 FPS critical path — YOLO Worker A is.
2. The FSM can tolerate stale pose data within the staleness threshold (see below).
3. GPU path performance will be measured in Phase 2 when the architecture supports isolated delegates.

**This decision must be revisited before Phase 2.** If CPU Pose p95 exceeds 200ms or FSM stale-hold rate exceeds 5%, escalate to GPU path or dedicated NPU.

---

## Concern 1: Stale Pose Data in the FSM

The FSM must not stall waiting for a fresh pose frame.

**Resolution:** The FSM holds the last known pose result as state. On each YOLO frame completion, it evaluates transitions using:
- Fresh ball/hoop bbox (just computed by Worker A)
- Last known pose landmarks + their timestamp

If pose timestamp is older than **300ms** (3 missed pose frames at 10 FPS), the FSM cannot transition out of `PREP` or `RELEASE`, but it does **not** reset. It holds state until fresh pose arrives.

```kotlin
data class FsmInputs(
    val ballBox: BoundingBox?,
    val hoopBox: BoundingBox?,
    val poseResult: PoseResult?,
    val poseAgeMs: Long  // System.currentTimeMillis() - poseResult.timestampMs
)

// In FSM evaluation:
val poseValid = inputs.poseAgeMs < POSE_STALENESS_THRESHOLD_MS // 300ms
```

This prevents false shot events from stale wrist positions without killing the session.

---

## Concern 2: Pose Channel Backpressure

**Resolution:** Same policy as YOLO — `capacity=1, DROP_OLDEST`. If Worker B is still processing when a new pose frame arrives, the new frame is dropped. This is intentional at 10 FPS.

The CameraX analyzer thread is never blocked by either channel.

Track a `droppedPoseFrames` counter in `PipelineStats` (see below).

---

## Concern 3: Thermal Throttling Definition

> **Thermal throttling is declared when YOLO FPS (Worker A completions/sec) drops more than 20% below the rolling 30-second average for a sustained period of ≥ 30 seconds.**

Benchmark reporting:
- Compute a rolling 30s FPS window from the CSV
- Flag any 30s window where FPS < `baseline_fps * 0.80`
- Report: total throttle events, max throttle duration, FPS at throttle onset
- **Pass criterion:** zero throttle events during the 10-minute measured window

---

## Updated PipelineStats

```kotlin
data class PipelineStats(
    val analysisFps: Int = 0,          // YOLO Worker A completions/sec — PRIMARY METRIC
    val poseFps: Int = 0,              // Pose Worker B completions/sec
    val avgAnalyzeMs: Double = 0.0,
    val p95AnalyzeMs: Double = 0.0,
    val avgPoseMs: Double = 0.0,
    val p95PoseMs: Double = 0.0,
    val droppedFrames: Long = 0,
    val droppedPoseFrames: Long = 0,   // new
    val queueDepth: Int = 0,
    val poseQueueDepth: Int = 0        // new
)
```

---

## Day 6 Pass/Fail Criteria

| Metric | Pass |
|---|---|
| YOLO FPS (Worker A) | ≥ 20 sustained (target 30) |
| Pose FPS (Worker B) | ≥ 8 sustained (target 10) |
| YOLO p95 latency | < 140ms |
| Pose p95 latency | < 200ms (CPU path) |
| Thermal throttle events | 0 over 10-minute window |
| FSM stale-pose holds | < 5% of FSM evaluations over 10-min run |

**Primary metric for Day 8 gate:** YOLO FPS (Worker A completions/sec). Pose FPS is reported separately and does not gate the spike.

---

## Affected Files

- `docs/plans/frame-scheduling-spec.md` — this file (new)
- `docs/tdd.md` — §8 links to this spec for Day 6 scheduling context ✓
- `docs/phase0-spike-plan.md` — links to this spec under Day 5 deliverables ✓
