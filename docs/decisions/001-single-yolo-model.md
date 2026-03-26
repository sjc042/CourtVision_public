# ADR-001: Single Multi-Class YOLO Model for Ball + Hoop Detection

**Status:** Accepted
**Date:** 2026-03-25
**Context:** Phase 0 — Technical Spike

---

## Decision

Use a **single multi-class YOLOv8n detector** as baseline (2 classes: `basketball`, `hoop`) exported to TFLite FP16, rather than separate models or manual hoop anchoring. Later explore YOLO26 detector.

## Context

Early TDD drafts referenced SSD MobileNet for ball detection and manual hoop anchoring as alternatives. The Phase 0 spike plan needed a canonical detection strategy before implementation began.

Three options were evaluated:

1. **Single multi-class YOLO** — one model, one inference pass, detects both ball and hoop
2. **Separate models** — dedicated ball detector + dedicated hoop detector
3. **Ball detector + manual hoop anchor** — ML for ball only, user taps hoop location

## Rationale

- **One inference pass** reduces GPU contention and simplifies the pipeline
- **Hoop as spatial anchor** provides geometry context for make/miss classification without extra overhead
- **Simpler profiling** — single model means one latency number, one RAM footprint
- **Easier debugging** — no model synchronization or result merging logic
- **FP16 on GPU delegate** meets the <100ms latency target on mid-range devices

## Alternatives Rejected

- **Separate models:** doubles inference cost and integration complexity for marginal accuracy gain at this stage
- **Manual hoop anchor:** requires user interaction every session, poor UX, and breaks ground mode where hoop may not be visible initially

## Consequences

- Must train/source a single YOLOv8n model with both classes (Roboflow public dataset as starting point)
- Hoop detection confidence may be lower than a specialized model — mitigated by temporal smoothing and optional manual fallback
- If thermal or FPS targets are missed, input resolution can be reduced (640 → 416 → 320) as a tuning knob
