# CourtVision — Project Documentation

> Local documentation hub for the CourtVision Android app.
> All project planning, design, and tracking lives here as markdown files.

---

## Core Documents

| Doc | Description |
| --- | --- |
| [🏀 Project Overview](project-overview.md) | High-level summary, goals, quick stats |
| [📄 PRD](prd.md) | Product Requirements Document — features, scope, constraints |
| [🔧 TDD](tdd.md) | Technical Design Document — architecture, ML pipeline, data models |
| [🗂️ User Stories](user-stories.md) | Epics, user stories, acceptance criteria, delivery plan |

## Development

| Doc | Description |
| --- | --- |
| [💻 Dev Workflow](dev-workflow.md) | 6-step AI-delegated development loop |
| [📋 Coding Style Guide](coding-style-guide.md) | Language conventions (Kotlin, Python, JS/TS) |

## Phase 0 — Technical Spike

| Doc | Description |
| --- | --- |
| [🚀 Phase 0 Spike Plan](phase0-spike-plan.md) | 8-day spike plan, gate criteria, evaluation protocol |
| [Day 1-2 Plan](plans/day1-2-plan.md) | CameraX pipeline setup details |

## Tracking

| Doc | Description |
| --- | --- |
| [❗ Issues Log](issues-log.md) | Findings, risks, and issues by date |

## Architecture Decisions

| ADR | Description |
| --- | --- |
| [ADR-001](decisions/001-single-yolo-model.md) | Single multi-class YOLO model for ball + hoop detection |
| [ADR-002](decisions/002-support-lib-preprocessing.md) | Use TFLite Support Library preprocessing (`ImageProcessor`) |
| [ADR-003](decisions/003-rotation-source-of-truth.md) | Orientation-driven CameraX rotation control with stall recovery |
| [ADR-004](decisions/004-kalman-ball-tracker.md) | Kalman ball tracker for Day 4 |

---

## How to Use

- **AI agents** (Claude, Codex, Gemini): reference these docs for project context
- **Obsidian**: open `docs/` as a vault for linked navigation
- **CONTEXT.md** at repo root is a lean AI session briefing — it links here for details
