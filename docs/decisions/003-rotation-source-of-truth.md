# ADR-003: Orientation-Driven CameraX Rotation Control

**Status:** Accepted
**Date:** 2026-04-02
**Context:** Phase 0 - Technical Spike (Post-Day 4 rotation investigation)

---

## Decision

Use `OrientationEventListener` as the source of truth for CameraX `targetRotation` updates in `CameraScreen`, and treat `DisplayListener` as non-authoritative for rotation correctness.

Rotation health is guarded by pipeline telemetry:
- Track expected target rotation vs frame metadata rotation.
- If mismatch persists for >= 2s with no frame drops, re-apply `targetRotation` (reconcile).
- If mismatch persists for >= 5s, rebind `ImageAnalysis` only (guarded recovery path).

Verbose rotation diagnostics remain behind a debug flag.

## Context

During GPU mode testing, back-to-back device rotations were occasionally missed when using display-driven rotation callbacks as the primary signal. CPU mode was stable, but GPU mode showed cases where UI rotation state and frame metadata could diverge long enough to affect inference orientation.

Instrumentation confirmed:
- `droppedFrames` stayed at 0 during problematic windows, so this was not channel backpressure loss.
- Display callback sequencing could lag device orientation transitions.
- Orientation-driven updates produced stable frame rotation transitions in both GPU and CPU tests.

## Rationale

- `OrientationEventListener` reflects physical orientation transitions directly and avoids relying on display callback timing as the correctness path.
- Mismatch telemetry plus reconcile/recovery provides runtime protection against metadata propagation stalls.
- Rebinding only `ImageAnalysis` limits blast radius compared with full camera session restart.
- Keeping verbose logs gated preserves debuggability without shipping high-volume telemetry by default.

## Trade-offs

- Adds rotation control complexity (state tracking + recovery logic).
- Orientation callbacks can be noisy near threshold angles, requiring snapped surface-rotation mapping.
- Recovery rebind introduces a rare but intentional maintenance operation during prolonged mismatch.

## Consequences

- Rotation correctness is now resilient across GPU/CPU delegate modes for the tested repro sequence.
- `DisplayListener` is no longer required for correctness and should not be reintroduced as primary rotation logic.
- Future refactors of camera binding or analyzer setup must preserve:
  - expected-vs-actual rotation telemetry,
  - reconcile threshold behavior,
  - guarded recovery rebind path.
