# ADR-001: Single Multi-Class YOLO Model

**Status:** Accepted (updated 2026-03-30 — expanded from 2 to 5 classes based on training data)
**Date:** 2026-03-25
**Context:** Phase 0 — Technical Spike

---

## Decision

Use a **single multi-class YOLOv8n detector** (5 classes: `ball`, `made`, `person`, `rim`, `shoot`) exported to TFLite FP16, rather than separate models or manual hoop anchoring. **Alternative:** YOLO26n (object detection) and YOLO26n-pose (combined detection + pose, replacing MediaPipe) — evaluate in Phase 2.

**Update (2026-03-30):** Original decision was 2 classes (`basketball`, `hoop`). Expanded to 5 classes after sourcing a combined Roboflow dataset (15,856 images) that includes shot outcome (`made`), player (`person`), and shooting action (`shoot`) labels. The `made` class provides a direct detector signal for shot outcome, potentially simplifying the FSM.

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

- Must train/source a single YOLOv8n model with all 5 classes (Roboflow combined dataset: 15,856 images)
- Rim detection confidence may be lower than a specialized model — mitigated by temporal smoothing and optional manual fallback
- `made` class may enable direct shot outcome detection, reducing reliance on trajectory-based make/miss logic
- `person` class available for player tracking — useful for pose estimation alignment in Day 5+
- `shoot` class provides shooting action context — potential noise if bboxes enclose distant ball+person pairs (see dataset refinement notes)
- If thermal or FPS targets are missed, input resolution can be reduced (640 → 480 → 320) as a tuning knob
