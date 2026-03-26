# ❗ Issues Log

Tracking findings, risks, and issues discovered during development.

---

## Phase 0 Day 1 — 2026-03-25

**Reviewed Docs**

- [Project Overview](project-overview.md)
- [Phase 0 Spike Plan](phase0-spike-plan.md)
- [PRD](prd.md)
- [TDD](tdd.md)
- [User Stories](user-stories.md)

**Findings (Highest Risk First)**

1. **Conflicting core detection architecture**: Spike says single YOLOv8n (ball+hoop), TDD says SSD MobileNet ball detector, and TDD open questions still lean manual hoop anchoring. This will cause rework unless one approach is declared canonical.
2. **MVP scope is contradictory and too wide**: PRD/User Stories mark many P1 features as MVP (shot type, paywall, full analytics), while Spike says only shot feasibility first and defer most features. "MVP" needs one explicit definition.
3. **No measurable protocol for `>90% shot accuracy` gate**: there is no labeled validation set, annotation plan, or pass/fail rubric (precision/recall/F1 by mode). Current "stable" wording is not testable.
4. **Performance budgets are internally inconsistent**: target 30fps plus `<100ms` ball + `<50ms` pose per frame is not feasible if run every frame in one loop. Need explicit frame-skipping/async scheduling budget.
5. **Device strategy is inconsistent**: Spike baseline devices differ from TDD test matrix. You need one official gate matrix (min/mid/high + ARCore coverage).
6. **Execution planning gap**: user stories have empty story points, no owners, no sprint capacity, and no dated milestones beyond the 8-day spike.
7. **ARCore dependency risk is acknowledged but not scheduled as a hard gate**: fallback behavior exists in notes, but no acceptance criteria for non-ARCore devices in roadmap.
8. **Data retention conflicts with analytics goals**: auto-delete clips/session data vs long-term trend and coaching workflows is not reconciled.
9. **Privacy/compliance plan missing**: camera/video handling, retention consent, deletion controls, and policy requirements are not defined.

**What to fix first**

1. Lock one detection strategy and update PRD/TDD/Spike/User Stories to match.
2. Define MVP v1 in one table: `Must ship` / `Deferred`.
3. Add a formal Phase 0 evaluation protocol (dataset, metrics, thresholds, test conditions).
4. Publish one device-and-mode gate matrix with explicit pass criteria.
5. Add story points + owner + target sprint for all P0/P1 items.
6. Add a privacy/data-retention spec before implementation starts.
