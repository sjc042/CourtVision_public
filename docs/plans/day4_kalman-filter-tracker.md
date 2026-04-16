# Plan: Day 4 — Kalman Ball Tracker

**Status:** ✅ COMPLETED (2026-04-02) — all 10 tasks done.

## Context

Days 1–3 are complete. The pipeline is:
`CameraX → YOLO (5-class TFLite) → [bounding boxes] → StateFlow → UI overlay`

Day 4's job is to insert a **Kalman filter** between YOLO output and the StateFlow emission — smoothing the ball centroid across frames, filling gaps when YOLO misses a frame, and producing a velocity estimate. This is a hard prerequisite for the Day 7 shot FSM (arc geometry, ball-leaves-hand detection).

Day 4 also has a **pre-task block** (export custom-trained model + validate on device) that must happen in parallel or before — but the Kalman implementation is independent of which model is loaded.

---

## Branch

`spike/day4-kalman-tracker` (branch from `main`)

---

## Pre-Day-4 Checklist (user runs these), Non-Blocking

These are TASKS.md blockers left over from the weekend training track:
1. ✅ Export best yolov8n (640) custom model to TFLite FP16 → drop into `assets/`
2. 🔲 Complete remaining training matrix (yolov8s, yolov11n, yolov11s at 3 resolutions) — still pending
3. 🔲 Export best model(s) from remaining training to TFLite FP16 → drop into `assets/`
4. ✅ Run Day 3 benchmark loop with custom-trained model; validate detection quality on-device
5. 🔲 Update TASKS.md training matrix status once remaining training completes

The Kalman code can be written while these are in progress — it consumes `DetectionBox` output regardless of model.

---

## Design Decisions

### Motion Model: Constant Velocity, 4-State Vector
State: `[cx, cy, vx, vy]` (ball centroid + velocity, all in normalized [0,1] image coords)
- Constant velocity is sufficient for spike — basketball arcs deviate slowly enough that a 30fps tracker handles it
- Constant acceleration would require longer tuning time with no Day 4 measurable benefit
- `dt` computed from actual camera frame timestamps (nanoseconds → seconds), not fixed 1/30s

### Coordinate Space: Normalized [0,1]
Kalman state lives in the same normalized space as `DetectionBox.left/top/right/bottom`. No coordinate transform needed at integration point.

### Dependency: Pure Kotlin — No New Library
4×4 Kalman math is implemented inline with `FloatArray`. Avoids a new Gradle dependency (EJML, Commons Math) in a single-module spike where adding deps has risk. Matrices are fixed-size; no dynamic allocation needed.

### Miss-Frame Policy
- If no ball detection in a frame: run **predict-only** (no measurement update). Tracker extrapolates position.
- After **N consecutive missed frames**: reset tracker state (`isTracked = false`). This prevents runaway extrapolation during dribble / out-of-frame periods.
- When a new ball detection arrives after a reset: reinitialize tracker from that measurement.
- **N is user-configurable at runtime** (see Step 4a). Default: `10` frames. At observed FPS of 9–20, this equals ~500ms–1.1s of extrapolation before reset — enough to survive brief occlusions, short enough to avoid ghost-tracking.

### Multi-Ball: Take Highest-Confidence `ball` Detection
YOLO may return multiple `ball` boxes. Day 4 picks the highest-confidence one. Multi-target tracking is Phase 2+.

### No New ADR for Rim Smoothing (Yet)
TDD §4.3 mentions "temporal smoothing for rim ROI." Rim smoothing is deferred to Day 6 (combined pipeline) — it requires pose context and is not a Day 4 deliverable.

---

## Files

### Create
- `app/src/main/java/com/courtvision/spike/pipeline/KalmanBallTracker.kt`
- `docs/decisions/004-kalman-ball-tracker.md`

### Modify
- `app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt`
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
- `app/src/main/java/com/courtvision/spike/camera/CameraUiState.kt`
- `app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt`
- `app/src/main/java/com/courtvision/spike/pipeline/PerformanceCsvLogger.kt`
- `TASKS.md`
- `CONTEXT.md`

---

## Step-by-Step Implementation

### Step 1 — `FrameContracts.kt`: Add `TrackedBall`, extend `DetectionFrame`

Add `TrackedBall` data class:
```kotlin
data class TrackedBall(
    val centroidX: Float,       // normalized [0,1]
    val centroidY: Float,       // normalized [0,1]
    val velocityX: Float,       // units/sec in normalized space
    val velocityY: Float,
    val isTracked: Boolean,     // false = track lost or never initialized
    val rawBox: DetectionBox?   // the YOLO box this frame, null if no detection
)
```

Extend `DetectionFrame` with one optional field:
```kotlin
data class DetectionFrame(
    ...
    val trackedBall: TrackedBall? = null   // null until Day 4 tracker integrated
)
```
Backward compatible (default null) — no call site changes needed.

---

### Step 2 — `KalmanBallTracker.kt`: Pure Kotlin Kalman Implementation

Class structure:
```kotlin
class KalmanBallTracker(
    private val processNoise: Float = 1e-4f,      // Q diagonal — tune on device
    private val measurementNoise: Float = 1e-2f,  // R diagonal — tune on device
    maxMissFrames: Int = 10
) {
    var maxMissFrames: Int = maxMissFrames         // var — mutable at runtime
        set(value) { field = value.coerceAtLeast(1) }
    val missFrames: Int get() = missCount          // read-only for external observation
    ...
}
```

Internal state:
```kotlin
private var state = FloatArray(4)      // [cx, cy, vx, vy]
private var P = identity4() * 1f      // 4×4 covariance, initialized large
private var missCount = 0
private var initialized = false
```

Key methods:
```kotlin
fun predict(dtSec: Float): TrackedBall   // F·x, F·P·Fᵀ + Q
fun update(measCx: Float, measCy: Float) // Kalman gain, state + P update
fun reset()                              // called after maxMissFrames exceeded
```

`predict()` is called every frame. If a ball box exists, `update()` is called immediately after. Return a `TrackedBall` from `predict()` reflecting post-update state.

Matrix math: implement `mat4x4Mul`, `mat4x2Mul`, `mat2x2Inv`, `mat4x4Add`, `mat4x4Sub` as private top-level functions using `FloatArray(16)` row-major storage. All fixed-size — no heap allocation in steady state.

---

### Step 3 — `FrameProcessor.kt`: Integrate Tracker

Member variable (initialized in `FrameProcessor` constructor or `init`):
```kotlin
private val ballTracker = KalmanBallTracker()
private var lastFrameTimestampNs = 0L
```

Inside `processImage()`, after NMS (after `val boxes = ...`), before `_detections.value = ...`:

```kotlin
val ballBox = boxes
    .filter { it.classId == BALL_CLASS_ID }
    .maxByOrNull { it.confidence }

val dtSec = if (lastFrameTimestampNs == 0L) (1f / 30f)
            else (image.imageInfo.timestamp - lastFrameTimestampNs) / 1_000_000_000f
lastFrameTimestampNs = image.imageInfo.timestamp

val trackedBall = if (ballBox != null) {
    val cx = (ballBox.left + ballBox.right) / 2f
    val cy = (ballBox.top + ballBox.bottom) / 2f
    ballTracker.predict(dtSec)
    ballTracker.update(cx, cy)
} else {
    ballTracker.predict(dtSec)   // extrapolate; tracker handles miss count internally
}

_detections.value = DetectionFrame(
    timestampNs = image.imageInfo.timestamp,
    sourceWidth = rotatedWidth,
    sourceHeight = rotatedHeight,
    rotationDegrees = 0,
    boxes = boxes,
    trackedBall = trackedBall
)
```

`BALL_CLASS_ID = 0` (index of `"ball"` in `CUSTOM_CLASS_NAMES`).

---

### Step 4 — `CameraUiState.kt`: Expose Tracker State

Add to the UI state data class that currently holds `detectionBoxes`:
```kotlin
val trackedBall: TrackedBall? = null,
val trackerMaxMissFrames: Int = 10
```

In `CameraViewModel.kt`, inside the `detections.collect {}` block, propagate `frame.trackedBall` into `_uiState`. Add a setter method:
```kotlin
fun setTrackerMaxMissFrames(n: Int) {
    frameProcessor.setTrackerMaxMissFrames(n)
    _uiState.update { it.copy(trackerMaxMissFrames = n) }
}
```

In `FrameProcessor.kt`, expose pass-through:
```kotlin
fun setTrackerMaxMissFrames(n: Int) { ballTracker.maxMissFrames = n }
```

---

### Step 4a — `CameraScreen.kt`: Miss-Frame Slider

Add a slider to the existing debug controls panel (alongside GPU/CPU toggle and model selector), locked while `isTracked` is true to prevent mid-track changes:

```kotlin
// Miss-frame threshold slider — range 1..30
Slider(
    value = uiState.trackerMaxMissFrames.toFloat(),
    onValueChange = { viewModel.setTrackerMaxMissFrames(it.roundToInt()) },
    valueRange = 1f..30f,
    steps = 28,
    enabled = uiState.trackedBall?.isTracked != true
)
Text("Miss reset: ${uiState.trackerMaxMissFrames} frames")
```

Range 1–30 covers ~50ms–3.3s at 10fps and ~50ms–1.5s at 20fps — enough latitude for tuning during the 10-shot evaluation.

---

### Step 5 — `CameraScreen.kt`: Tracker Overlay

In the bounding box `Canvas` drawing block, after drawing YOLO boxes, draw tracker state:
- If `trackedBall?.isTracked == true`: draw a small filled circle at `(centroidX * canvasWidth, centroidY * canvasHeight)` in a distinct color (e.g. cyan, to differentiate from the raw yellow YOLO box)
- Draw a short line from centroid in velocity direction: `endX = cx + vx * 0.5f * canvasWidth` (scale factor tunable)
- Label: small `"T"` or `"track"` text near circle, only when `isTracked`

This makes tracker lag / drift visually obvious during the 10-shot evaluation.

---

### Step 6 — `PerformanceCsvLogger.kt`: Add Tracking Columns

Add to CSV header:
```
...,tracking_active,track_cx,track_cy,track_vx,track_vy,miss_streak
```

Add corresponding fields to `PipelineStats` (or pass `TrackedBall?` directly to logger). Log per-second aggregate: `isTracked` (latest frame), centroid and velocity (latest), and a miss-streak counter surfaced from `KalmanBallTracker` via a public `val missFrames: Int`.

---

### Step 7 — `ADR-004`: Document Kalman Design ✅

Write `docs/decisions/004-kalman-ball-tracker.md` covering:
> Note: TASKS.md item 8 incorrectly references this as `003-kalman-ball-tracker.md` (ADR-003). The actual file is `004-kalman-ball-tracker.md` (ADR-004). ADR-003 is `003-rotation-source-of-truth.md`.
- Motion model choice (constant velocity, why not acceleration)
- Coordinate space choice (normalized [0,1])
- Dependency decision (pure Kotlin vs EJML)
- Miss-frame reset threshold (default 10 frames, configurable 1–30 via UI slider, rationale for default given observed 9–20 FPS)
- Initial noise parameters and tuning plan
- What changes when moving to Phase 2 (multi-ball, pose coupling)

---

### Step 8 — Update `TASKS.md` and `CONTEXT.md` ✅

Marked Day 4 active in TASKS.md. Updated CONTEXT.md progress line to: *"Day 4 (Kalman tracker) in progress - implementation underway (core tracking + dual CSV logging integrated)."*

---

## Noise Parameter Tuning (on-device)

Initial values (conservative):
- `processNoise Q = 1e-4` (trust model between frames)
- `measurementNoise R = 1e-2` (moderate trust in YOLO centroid)

Tuning signal: watch tracker overlay on device. If tracker lags raw box → lower R. If tracker is jittery → raise R or lower Q. Log `track_vx/vy` to CSV to verify trajectory smoothness offline.

---

## Verification

1. **Build and install** on S22+. Open camera, point at a basketball.
2. **Visual check:** cyan tracker circle visible, stays on ball through moderate motion. Circle stays briefly visible when ball exits frame (extrapolation). Disappears after N missed frames (test by adjusting slider).
3. **Shot sequence test:** shoot 10 times with phone on tripod. Export CSV from device. Plot `track_cx` vs time — should show smooth arcs with no single-frame spikes.
4. **Latency check:** confirm `avg_analyze_ms` in `PipelineStats` hasn't regressed significantly from Day 3 baseline (Kalman adds ~0.1ms per frame — negligible).
5. **Miss reset test:** cover ball with hand. Tracker should reset after N missed frames (confirm by watching slider value vs tracker disappearance timing). Uncover ball — tracker re-locks within 1–2 frames. Try slider at N=5 and N=20 to confirm the setting takes effect live.

Tests to run (user runs locally):
```
./gradlew :app:testDebugUnitTest --tests "*.KalmanBallTrackerTest"
```
Write `KalmanBallTrackerTest` in `app/src/test/` covering:
- Predict-only convergence (no measurement): position drifts, covariance grows
- Update reduces uncertainty (P shrinks on diagonal)
- Miss-frame counter triggers reset at exactly `maxMissFrames`
- Centroid smoothing: step-change measurement → tracker lags, not jumps

---

## Post-Day-4 Addition — NNAPI Delegate Mode (2026-04-01)

Not in original Day 4 scope. Added as an extension to the GPU/CPU delegate switching infrastructure built in Day 3.

1. Added `InferenceMode.NNAPI` with runtime switching and CPU fallback on delegate init failure
2. Added `NnApiDelegateProbe` — probed synchronously in `CameraViewModel.init`
3. Added NNAPI probe visibility in debug overlay and mode selector gating
4. Added unit tests covering NNAPI probe state and CSV serialization of `delegate_mode=NNAPI`
5. Phase 2 note: move both GPU and NNAPI delegate probes to `Dispatchers.Default`

> **Architecture note:** `InferenceMode.NNAPI` routes through Android's NNAPI HAL and does **not** access the Hexagon NPU on Qualcomm devices. NPU acceleration requires the QNN TFLite delegate — a Phase 2 dependency. See [ADR-005 Deferred section](../decisions/005-sequential-gpu-inference-pipeline.md).

---

## Completion Record

All 10 Day 4 tasks completed and committed by 2026-04-02. Branch: `spike/day4-kalman-tracker`.
On-device validation passed: 10-shot sequence CSV confirmed smooth trajectories with no single-frame spikes.
