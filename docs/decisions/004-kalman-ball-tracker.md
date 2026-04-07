# ADR-004: Kalman Ball Tracker for Day 4

**Status:** Accepted
**Date:** 2026-04-02
**Context:** Phase 0 - Technical Spike (Day 4)

---

## Decision

Add a pure Kotlin Kalman tracker between YOLO detections and UI/state emission:

- State model: constant velocity `[cx, cy, vx, vy]`
- Coordinate space: normalized `[0,1]` (same as detection boxes)
- Runtime policy: predict-only on missing measurements; reset after configurable miss streak
- Covariance update: Joseph form in the implemented filter update
- Logging: dual CSV streams
  - Existing benchmark CSV stays 1 Hz and includes tracker snapshot columns
  - New tracking CSV logs per-frame trajectory values

## Context

Day 4 requires stable ball trajectory and velocity estimation for downstream shot-state logic. Raw detector centroids are noisy and occasionally missing for short spans (occlusion, motion blur, edge-of-frame cases).

The tracker had to be integrated into the current single-module spike without adding new dependencies or violating the existing CameraX consumer threading constraints.

## Rationale

- Constant-velocity model is enough for Day 4 validation and cheaper to tune than constant-acceleration.
- Reusing normalized coordinates avoids transform overhead and mismatch risk.
- Pure Kotlin implementation avoids dependency overhead and keeps control over fixed-size math operations.
- Runtime-configurable miss threshold (slider) allows rapid on-device tuning.
- Dual CSV avoids breaking Day 1-3 benchmark cadence while still enabling frame-level trajectory analysis.
- The original Day 4 design allowed a simpler covariance update for spike speed, but the implementation was upgraded to Joseph form during Day 4 to better preserve covariance symmetry and non-negativity under noisy `dt` and dropout conditions.

### Joseph Form vs Simple Covariance Update

- The simpler covariance update (`P = (I - K H) P`) was initially acceptable for a short Phase 0 spike because it is cheaper to write, easier to inspect, and likely stable enough for a 10-minute validation session with a single tracked ball.
- That simpler form is more prone to losing covariance symmetry or drifting into small negative diagonal values because of floating-point error, especially with repeated predict/update cycles, variable `dt`, and measurement dropouts.
- Joseph form (`P = (I - K H) P (I - K H)^T + K R K^T`) adds a small amount of matrix work but is still cheap at this fixed 4-state size, so the runtime cost is negligible relative to detector inference.
- Given that tradeoff, Joseph form is the better implementation choice even for the spike, while the simpler form remains useful only as the baseline design that the Day 4 implementation evolved from.

## Consequences

- `DetectionFrame` now carries tracker output (`trackedBall`, `missStreak`).
- `FrameProcessor` now computes per-frame `dt` from camera timestamps and tracks highest-confidence `ball` detection.
- Overlay includes tracker centroid + velocity vector for visual tuning.
- Benchmark CSV includes tracker snapshot columns: `tracking_active`, `track_cx`, `track_cy`, `track_vx`, `track_vy`, `miss_streak`.
- A second per-frame tracking CSV is produced for trajectory plotting and Day 4 validation.

## Phase 2 Follow-ups

- Evaluate multi-ball tracking strategy if detections include multiple valid balls.
- Revisit process/measurement noise values using labeled clips.
- Revisit the diagonal-only process-noise (`Q`) matrix; it is a spike simplification and may need cross-axis or motion-model-coupled terms for production tracking.
- Couple tracker quality and shot FSM thresholds with pose/rim context in combined pipeline days.
