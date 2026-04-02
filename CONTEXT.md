# CourtVision — AI Session Context

> Paste this file in full at the start of every Codex or Gemini session.
> Last updated: March 2026 — Phase 0 (Technical Spike)

---

## Project Summary

CourtVision is an **Android-native basketball performance tracking app** using on-device AI,
computer vision, and AR-assisted court mapping to deliver real-time shot analytics,
biomechanical metrics, and player development insights. Comparable product: HomeCourt (iOS).

- Platform: Android API 26+
- Stage: Planning / Pre-Development (Phase 0 spike active)
- Target launch: 12 months from kickoff
- Revenue model: Freemium — free tier + premium analytics

---

## Active Phase

## ⚠️ Phase 0 Spike Override — Single Module

The module structure listed below is the **target architecture for Phase 2+**. It does not exist yet.

For all Phase 0 work:
- The repo is a **single `:app` module**
- Do NOT create new Gradle modules
- Do NOT add Hilt or Room
- All code lives under `app/src/main/java/com/courtvision/spike/`
- Existing packages: `camera/`, `pipeline/`

---

**Phase 0 — Technical Spike (8-day plan)**

Goal: Validate that real-time shot detection is achievable on Android using on-device ML
BEFORE any product architecture is locked in.

Pipeline under test:
```
CameraX → YOLO (5-class) → Kalman tracker → shot state machine → metrics logger
```

> Full spike plan: [docs/phase0-spike-plan.md](docs/phase0-spike-plan.md)

**Progress:** Days 1–3 complete. Weekend training track complete (YOLOv8n trained at 640/480/320). Day 4 (Kalman tracker) in progress — plan ready, implementation starting.

### Day 8 Gate Criteria (all must pass)
- [ ] Ball detection latency < 100ms (p95 < 140ms)
- [ ] Pose inference latency < 50ms (p95 < 70ms)
- [ ] Combined RAM < 400MB sustained
- [ ] End-to-end FPS ≥ 20 sustained for 10 min (target 30)
- [ ] Shot detection stable across Tripod and Ground modes
- Benchmark results committed to `/benchmarks/phase0/`

### Phase 0 Out of Scope
UI design · user accounts · cloud backend · AR court mapping · freemium/paywall ·
analytics pipeline · session history storage

---

## Performance Hard Limits (Never Violate)

| Metric                  | Target                         |
|-------------------------|--------------------------------|
| Ball detection latency  | < 100ms per frame              |
| Pose inference latency  | < 50ms per frame               |
| Shot detection delay    | < 200ms from release to event  |
| Heatmap render time     | < 500ms for up to 500 shots    |
| App cold start          | < 3 seconds to camera live view|
| Battery usage           | < 15% per hour active session  |
| RAM footprint           | < 400MB active session         |

---

## Tech Stack

| Component         | Technology                         | Version / Notes                               |
|-------------------|------------------------------------|-----------------------------------------------|
| Language          | Kotlin                             | 1.9+                                          |
| UI Framework      | Jetpack Compose                    | Latest stable                                 |
| Architecture      | MVVM + Clean Architecture          | Android Architecture Components               |
| DI Framework      | Hilt (Dagger)                      | 2.x                                           |
| Camera            | CameraX                            | Jetpack — API 21+                             |
| Pose Estimation   | MediaPipe Pose Landmarker (alt: YOLO26n-pose) | 0.10.x — 33 landmarks, 30fps           |
| Object Detection  | TFLite YOLOv8n FP16 (alt: YOLO26n) | 5-class model: `ball`, `made`, `person`, `rim`, `shoot` |
| AR / Spatial      | ARCore                             | Ground plane + anchors (Phase 3)              |
| Computer Vision   | OpenCV Android                     | 4.x — corner/line detection, homography       |
| Local Database    | Room                               | Session, shot, metric entities                |
| Charts            | MPAndroidChart                     | 3.1.x — heatmap, trend lines                  |
| Networking        | Retrofit + OkHttp                  | Optional — cloud sync tier only               |
| Testing           | JUnit 5 + MockK + Espresso         | Unit, integration, UI tests                   |
| Linting / Style   | ktlint + detekt                    | Enforced in CI                                |

---

## Architecture

### Pattern: MVVM + Clean Architecture (strictly layered)

```
UI Layer          → Jetpack Compose screens, ViewModels, UI state (StateFlow)
Domain Layer      → Use cases, business logic, shot detection algorithms
                   NO Android imports allowed in this layer
Data Layer        → Room DB, file storage, optional cloud sync Repository
ML / CV Layer     → CameraX feed, MediaPipe, TFLite detector, ARCore
```

**Layer rules (enforce these in every review):**
- UI → ViewModel → UseCase → Repository — no layer skipping
- No Repository calls directly from Fragments or Composables
- No Android framework imports (Context, etc.) inside domain layer classes
- State exposed as `StateFlow`, consumed with `collectAsStateWithLifecycle`
- `LiveData` is not used — `StateFlow` only

### Module Structure

```
:app                  — Entry point, Hilt DI setup, navigation graph
:feature:capture      — Camera session, capture mode switching, live overlay
:feature:analytics    — Heatmap, session review, shot timeline
:feature:history      — Session list, drill history, progress charts
:core:ml              — MediaPipe wrapper, TFLite YOLOv8n multi-class detector
:core:ar              — ARCore ground plane, homography, court mapper
:core:data            — Room entities, DAOs, Repository interfaces
:core:domain          — Use cases, models, ShotMetrics data classes
:core:ui              — Shared Compose components, theme, design tokens
```

---

## ML Pipeline Architecture

> Full details: [docs/tdd.md](docs/tdd.md) §4

### Single Multi-Class YOLO Detector (Canonical Decision)
- Model: YOLOv8n exported to TFLite FP16 (alt: YOLO26n)
- Classes (nc=5): `ball`, `made`, `person`, `rim`, `shoot` — one inference pass
- Input: 640×640 (fallback to 480 or 320 if thermal or FPS targets missed)
- Preprocessing: TFLite Support Library `ImageProcessor` (`ResizeOp` BILINEAR + `NormalizeOp`) — ~19ms p50, resolution-independent
- Runtime: GPU delegate primary, CPU fallback required
- Supports both standard YOLO (external NMS) and end-to-end YOLO (NMS built-in)
- Post-process: Kalman tracking on ball centroid + temporal smoothing for rim ROI
- Note: `made` class provides a direct detector signal for shot outcome — may simplify FSM FLIGHT→OUTCOME transition

### Shot Detection State Machine (FSM)
```
IDLE → PREP (knee bend + ball held)
     → RELEASE (wrist above shoulder, ball leaves hand)
     → FLIGHT (ball ascending arc)
     → OUTCOME (make/miss via hoop intersection or trajectory)
     → IDLE
```

### Thread Safety Rules (critical — AI often gets this wrong)
- All inference runs on `Dispatchers.Default` or a dedicated `ExecutorService` — never Main
- `TFLiteInterpreter` is NOT thread-safe — never share instances across coroutines
- `PoseLandmarker` is NOT thread-safe — same rule
- No object allocation inside `ImageAnalysis.Analyzer.analyze()` — pre-allocate

---

## Kotlin Coding Conventions

> Full style guide: [docs/coding-style-guide.md](docs/coding-style-guide.md)

- **Naming:** `camelCase` for functions/variables, `PascalCase` for classes
- **Immutability:** Prefer `val` over `var` — use `var` only when mutation is required
- **Coroutines:** Always scoped — `viewModelScope` in ViewModels, `lifecycleScope` in UI
  - Never use `GlobalScope`
  - Never block with `.runBlocking` on Main thread
- **Null safety:** Prefer `?.let`, `?: return`, or `requireNotNull` over `!!`
- **StateFlow:** Expose as `StateFlow<UiState>`, not `MutableStateFlow` to callers
- **Hilt scopes:**
  - `@Singleton` — TFLite interpreter, Room DB, Repository implementations
  - `@ViewModelScoped` — Use cases
  - Never manually construct Hilt-managed dependencies

---

## AI Coder Task Split

| Task                                                    | Use          |
|---------------------------------------------------------|--------------|
| Kotlin logic — ViewModel, UseCase, Repository           | Codex        |
| CameraX inference loop, Kalman filter                   | Codex        |
| Shot state machine (FSM)                                | Codex        |
| Room DAOs, Hilt module wiring                           | Codex        |
| TFLite / MediaPipe integration logic                    | Codex        |
| Jetpack Compose UI, screen layouts                      | Gemini       |
| Debugging with Logcat output (paste directly)           | Gemini       |
| Debugging with UI screenshot (paste directly)           | Gemini       |
| Performance analysis with benchmark output              | Gemini       |

**Prompt structure for every session:**
1. Paste this entire CONTEXT.md
2. Paste the specific file(s) to be modified
3. State the exact task with class/function names
4. State constraints (no new deps, follow existing error pattern, etc.)

---

## Branch Naming Convention

| Prefix       | Use case                                          |
|--------------|---------------------------------------------------|
| `spike/`     | Phase 0 experiments — throwaway-safe              |
| `feature/`   | New capabilities (Phase 2+)                       |
| `fix/`       | Bug fixes                                         |
| `perf/`      | Latency, RAM, battery improvements               |
| `refactor/`  | Architecture cleanup, no behaviour change         |
| `chore/`     | CI, tooling, dependency updates                   |

---

## Phase Roadmap

| Phase      | Focus                                                         | Gate                    |
|------------|---------------------------------------------------------------|-------------------------|
| 0 (now)    | CameraX → detector → tracker → shot FSM                      | —                       |
| 2          | Pose integration, release angle, jump height, FSM refinement  | Spike green             |
| 3          | AR court mapping, homography, shot heatmap                    | ARCore validation       |
| 4          | Session history, coaching insights, drill mode                | Phase 2 stable          |
| 5          | Freemium paywall, export, share cards                         | Phase 3 stable          |

---

## Project Documents (Local)

- [📁 All Docs Index](docs/README.md)
- [🏀 Project Overview](docs/project-overview.md)
- [📄 PRD](docs/prd.md)
- [🔧 TDD](docs/tdd.md)
- [🗂️ User Stories & Features](docs/user-stories.md)
- [🚀 Phase 0 Spike Plan](docs/phase0-spike-plan.md)
- [💻 Dev Workflow](docs/dev-workflow.md)
- [📋 Coding Style Guide](docs/coding-style-guide.md)
- [❗ Issues Log](docs/issues-log.md)
- [Architecture Decisions](docs/decisions/)
