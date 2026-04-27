# ADR-006: Pose Gating Mode and Person Selection Mode

**Status:** Accepted
**Date:** 2026-04-15
**Deciders:** CourtVision core team
**Extends:** ADR-005 — Sequential GPU Inference Pipeline

---

## Context

ADR-005 specifies that pose inference should be skipped when the FSM is in `IDLE` or `MADE`
states. It does not define *how* the gating condition is represented in code, nor does it address
how to select a single `person` bbox when YOLO returns multiple detections.

Day 6 wires the live pose path. Two questions must be answered before implementation:

1. **When should pose run?** ADR-005 gives the Day 7 answer (FSM state), but Day 6 runs before
   the FSM exists. A hardcoded `person bbox detected` gate would need a call-site refactor in
   Day 7. A configurable enum lets Day 7 be a one-line change.

2. **Which person bbox to use?** YOLO may return multiple `person` detections. The selection
   policy is a separate concern from the gating policy and should be independently configurable.

---

## Decision

Introduce two orthogonal enums in `FrameContracts.kt`:

### `PoseGatingMode`

```kotlin
enum class PoseGatingMode {
    /** Run pose on every frame where a person bbox is detected.
     *  Phase 0 default — maximises benchmark sample count for reliable p95. */
    EVERY_FRAME_WITH_PERSON,

    /** Run pose only on frames where YOLO also returns a "shoot" class (ID 4) detection.
     *  Phase 2 optimization — reduces GPU load during non-shooting periods.
     *  ⚠️ See Consequences before enabling. */
    SHOOT_CLASS_GATED,

    /** Skip pose in IDLE and MADE FSM states; run in PREP, RELEASE, FLIGHT.
     *  Stub — Day 7 wires this. Selecting this mode before Day 7 is a no-op equivalent
     *  to EVERY_FRAME_WITH_PERSON. */
    FSM_GATED,
}
```

### `PersonSelectionMode`

```kotlin
enum class PersonSelectionMode {
    /** Use the person bbox with the highest YOLO confidence score.
     *  Phase 0 default — adequate for single-shooter sessions. */
    HIGHEST_CONFIDENCE,

    /** Track the primary user across frames via ReID embedding similarity.
     *  Stub — Phase 2. Selecting this mode before Phase 2 falls back to HIGHEST_CONFIDENCE. */
    REID_TRACKED,
}
```

**Phase 0 defaults:** `EVERY_FRAME_WITH_PERSON` + `HIGHEST_CONFIDENCE`.

**Mirror the `InferenceMode` pattern** — enum in `FrameContracts.kt`, `AtomicReference` state
in `FrameProcessor`, setter exposed on `FrameProcessorGateway`:
```kotlin
fun setPoseGatingMode(mode: PoseGatingMode)
fun setPersonSelectionMode(mode: PersonSelectionMode)
```

Add named class ID constants alongside `BALL_CLASS_ID = 0` in `FrameProcessor.Companion`:
```kotlin
private const val PERSON_CLASS_ID = 2
private const val SHOOT_CLASS_ID  = 4
```

---

## Alternatives Considered

### Combined enum (gating + selection merged) — rejected

A single `PoseConfig` enum combining both axes forces cartesian product entries
(`EVERY_FRAME_HIGHEST_CONF`, `EVERY_FRAME_REID`, `SHOOT_GATED_HIGHEST_CONF`, …).
The axes are orthogonal — keeping them separate allows free composition and matches the
`InferenceMode` precedent already established in the codebase.

### Hardcoded `firstOrNull { classId == PERSON_CLASS_ID }` with TODO comment — rejected

The call site would require a structural refactor in Day 7 (extracting the gating predicate,
adding FSM state parameter). An enum abstraction costs one extra field and two setters but
makes Day 7 a single-line enum value change with no structural change to `processImage()`.

### `SHOOT_CLASS_GATED` as Phase 0 default — rejected

Using the `shoot` class as the gating signal is architecturally appealing (player is provably
mid-shot) but creates a benchmark coverage problem: `shoot` detections are sparse, typically
5–15 per minute during a validation session vs. ~300+ person-bbox detections. A p95 latency
computed from 15 samples is not statistically reliable. The Day 8 gate requires a trustworthy
combined latency number, so Mode 1 (`EVERY_FRAME_WITH_PERSON`) is the correct Phase 0 default.
`SHOOT_CLASS_GATED` is better understood as a Phase 2 GPU-load optimization once the baseline
is established — not a validation strategy.

---

## Consequences

### Positive

- **Day 7 FSM wiring is a one-line change:** replace `EVERY_FRAME_WITH_PERSON` with `FSM_GATED`
  in `FrameProcessor`; no call-site restructure.
- **Phase 2 `SHOOT_CLASS_GATED` is already named and stubbed:** can be enabled and benchmarked
  without touching the gating abstraction.
- **`PersonSelectionMode.REID_TRACKED` stub:** ReID integration in Phase 2 drops in behind the
  existing setter without changing `processImage()` structure.
- **Benchmark CSV already tags `pose_skipped`:** both gating modes produce the same skip flag,
  so the benchmark schema is mode-agnostic.

### Negative / Risks

- Two extra enums and two extra setters to carry through `FrameProcessorGateway`. Minor.
- **`SHOOT_CLASS_GATED` risk:** YOLO `shoot` class precision is unvalidated. A missed `shoot`
  detection at the release point = no pose inference on that frame = missing biomechanical data
  at the most critical moment. Do not enable in production without measuring `shoot` class
  recall on real shooting footage first.
- **`FSM_GATED` stub silent fallback:** if Day 7 is delayed and `FSM_GATED` is set prematurely,
  the fallback to `EVERY_FRAME_WITH_PERSON` behaviour will silently produce correct-looking but
  ungated output. Document this in the stub implementation comment.

### Deferred (Phase 2)

- `SHOOT_CLASS_GATED` production enablement — validate `shoot` class recall first
- `REID_TRACKED` implementation — requires cross-frame embedding similarity model
- Per-mode benchmark comparison (Mode 1 vs. Mode 2 GPU load reduction %)
