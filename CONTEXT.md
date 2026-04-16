# CourtVision â€” AI Session Context

> Paste this file in full at the start of every Codex or Gemini session.
> Last updated: March 2026 â€” Phase 0 (Technical Spike)

---

## Project Summary

CourtVision is an **Android-native basketball performance tracking app** using on-device AI,
computer vision, and AR-assisted court mapping to deliver real-time shot analytics,
biomechanical metrics, and player development insights. Comparable product: HomeCourt (iOS).

- Platform: Android API 26+
- Stage: Planning / Pre-Development (Phase 0 spike active)
- Target launch: 12 months from kickoff
- Revenue model: Freemium â€” free tier + premium analytics
- **MVP scope:** see `docs/prd.md` Section 3.2 â this is the canonical definition.

---

## Active Phase

## âš ï¸ Phase 0 Spike Override â€” Single Module

The module structure listed below is the **target architecture for Phase 2+**. It does not exist yet.

For all Phase 0 work:
- The repo is a **single `:app` module**
- Do NOT create new Gradle modules
- Do NOT add Hilt or Room
- All code lives under `app/src/main/java/com/courtvision/spike/`
- Existing packages: `camera/`, `pipeline/`

---

**Phase 0 â€” Technical Spike (8-day plan)**

Goal: Validate that real-time shot detection is achievable on Android using on-device ML
BEFORE any product architecture is locked in.

Pipeline under test:
```
CameraX â†’ YOLO (5-class) â†’ Kalman tracker â†’ shot state machine â†’ metrics logger
```

> Full spike plan: [docs/phase0-spike-plan.md](docs/phase0-spike-plan.md)

**Progress:** Days 1â€“4 complete. Day 5 pose isolated validation gates passed on S22+ (SM-S906U1): EGL/thread check passed, visual plausibility gate passed, and pose p95 inference at 10.774ms (PASS_STRONG).

### Day 8 Gate Criteria (all must pass)
- [ ] Ball detection latency < 100ms (p95 < 140ms)
- [ ] Pose inference latency < 50ms (p95 < 70ms)
- [ ] Combined RAM < 400MB sustained
- [ ] End-to-end FPS â‰¥ 20 sustained for 10 min (target 30)
- [ ] Shot detection stable across Tripod and Ground modes
- Benchmark results committed to `/benchmarks/phase0/`

### Day 5 Validation Snapshot (2026-04-14)
- Device: `SM-S906U1`
- Mode: `sequential_yolo_warm`, `gpu_mode=GPU`
- Gate Â§1 (EGL/thread): PASS (no `TfLiteGpuDelegate ... must run on the same thread` runtime log observed)
- Gate Â§2 (visual plausibility): PASS (manual check >= 8/10)
- Gate Â§3 (latency): PASS_STRONG (`pose_inference_p95_ms=10.774`)
- Artifacts:
  - `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_summary.txt`
  - `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_validation.csv`

### Phase 0 Out of Scope
UI design Â· user accounts Â· cloud backend Â· AR court mapping Â· freemium/paywall Â·
analytics pipeline Â· session history storage

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
| Camera            | CameraX                            | Jetpack â€” API 21+                             |
| Pose Estimation   | `pose_landmarks_detector.tflite` (standalone TFLite `Interpreter`, GPU delegate) â€” see ADR-005 | **Lite variant** extracted from `pose_landmarker_lite.task`; 33 WorldLandmarks; not via PoseLandmarker task API |
| Object Detection  | TFLite YOLOv8n FP16 (alt: YOLO26n) | 5-class model: `ball`, `made`, `person`, `rim`, `shoot` |
| AR / Spatial      | ARCore                             | Ground plane + anchors (Phase 3)              |
| Computer Vision   | OpenCV Android                     | 4.x â€” corner/line detection, homography       |
| Local Database    | Room                               | Session, shot, metric entities                |
| Charts            | MPAndroidChart                     | 3.1.x â€” heatmap, trend lines                  |
| Networking        | Retrofit + OkHttp                  | Optional â€” cloud sync tier only               |
| Testing           | JUnit 5 + MockK + Espresso         | Unit, integration, UI tests                   |
| Linting / Style   | ktlint + detekt                    | Enforced in CI                                |

---

## Architecture

### Pattern: MVVM + Clean Architecture (strictly layered)

```
UI Layer          â†’ Jetpack Compose screens, ViewModels, UI state (StateFlow)
Domain Layer      â†’ Use cases, business logic, shot detection algorithms
                   NO Android imports allowed in this layer
Data Layer        â†’ Room DB, file storage, optional cloud sync Repository
ML / CV Layer     â†’ CameraX feed, TFLite YOLO detector, pose_landmarks_detector.tflite (sequential GPU, ADR-005), ARCore
```

**Layer rules (enforce these in every review):**
- UI â†’ ViewModel â†’ UseCase â†’ Repository â€” no layer skipping
- No Repository calls directly from Fragments or Composables
- No Android framework imports (Context, etc.) inside domain layer classes
- State exposed as `StateFlow`, consumed with `collectAsStateWithLifecycle`
- `LiveData` is not used â€” `StateFlow` only

### Module Structure

```
:app                  â€” Entry point, Hilt DI setup, navigation graph
:feature:capture      â€” Camera session, capture mode switching, live overlay
:feature:analytics    â€” Heatmap, session review, shot timeline
:feature:history      â€” Session list, drill history, progress charts
:core:ml              â€” TFLite YOLOv8n multi-class detector, pose_landmarks_detector.tflite interpreter (ADR-005)
:core:ar              â€” ARCore ground plane, homography, court mapper
:core:data            â€” Room entities, DAOs, Repository interfaces
:core:domain          â€” Use cases, models, ShotMetrics data classes
:core:ui              â€” Shared Compose components, theme, design tokens
```

---

## ML Pipeline Architecture

> Full details: [docs/tdd.md](docs/tdd.md) Â§4

### Single Multi-Class YOLO Detector (Canonical Decision)
- Model: YOLOv8n exported to TFLite FP16 (alt: YOLO26n)
- Classes (nc=5): `ball`, `made`, `person`, `rim`, `shoot` â€” one inference pass
- Input: 640Ã—640 (fallback to 480 or 320 if thermal or FPS targets missed)
- Preprocessing: TFLite Support Library `ImageProcessor` (`ResizeOp` BILINEAR + `NormalizeOp`) â€” ~19ms p50, resolution-independent
- Runtime: GPU delegate primary, CPU fallback required
- Supports both standard YOLO (external NMS) and end-to-end YOLO (NMS built-in)
- Post-process: Kalman tracking on ball centroid + temporal smoothing for rim ROI
- Note: `made` class provides a direct detector signal for shot outcome â€” may simplify FSM FLIGHTâ†’OUTCOME transition

### Shot Detection State Machine (FSM)
```
IDLE â†’ PREP (knee bend + ball held)
     â†’ RELEASE (wrist above shoulder, ball leaves hand)
     â†’ FLIGHT (ball ascending arc)
     â†’ OUTCOME (make/miss via hoop intersection or trajectory)
     â†’ IDLE
```

### Thread Safety Rules (critical â€” AI often gets this wrong)
- All inference runs on `Dispatchers.Default` or a dedicated `ExecutorService` â€” never Main
- `TFLiteInterpreter` is NOT thread-safe â€” never share instances across coroutines
- `pose_landmarks_detector.tflite` `Interpreter` is NOT thread-safe â€” same rule (loaded directly, not via PoseLandmarker task API; see ADR-005)
- No object allocation inside `ImageAnalysis.Analyzer.analyze()` â€” pre-allocate

---

## Kotlin Coding Conventions

> Full style guide: [docs/coding-style-guide.md](docs/coding-style-guide.md)

- **Naming:** `camelCase` for functions/variables, `PascalCase` for classes
- **Immutability:** Prefer `val` over `var` â€” use `var` only when mutation is required
- **Coroutines:** Always scoped â€” `viewModelScope` in ViewModels, `lifecycleScope` in UI
  - Never use `GlobalScope`
  - Never block with `.runBlocking` on Main thread
- **Null safety:** Prefer `?.let`, `?: return`, or `requireNotNull` over `!!`
- **StateFlow:** Expose as `StateFlow<UiState>`, not `MutableStateFlow` to callers
- **Hilt scopes:**
  - `@Singleton` â€” TFLite interpreter, Room DB, Repository implementations
  - `@ViewModelScoped` â€” Use cases
  - Never manually construct Hilt-managed dependencies

---

## AI Coder Task Split

| Task                                                    | Use          |
|---------------------------------------------------------|--------------|
| Kotlin logic â€” ViewModel, UseCase, Repository           | Codex        |
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
| `spike/`     | Phase 0 experiments â€” throwaway-safe              |
| `feature/`   | New capabilities (Phase 2+)                       |
| `fix/`       | Bug fixes                                         |
| `perf/`      | Latency, RAM, battery improvements               |
| `refactor/`  | Architecture cleanup, no behaviour change         |
| `chore/`     | CI, tooling, dependency updates                   |

---

## Phase Roadmap

| Phase      | Focus                                                         | Gate                    |
|------------|---------------------------------------------------------------|-------------------------|
| 0 (now)    | CameraX â†’ detector â†’ tracker â†’ shot FSM                      | â€”                       |
| 2          | Pose integration, release angle, jump height, FSM refinement  | Spike green             |
| 3          | AR court mapping, homography, shot heatmap                    | ARCore validation       |
| 4          | Session history, coaching insights, drill mode                | Phase 2 stable          |
| 5          | Freemium paywall, export, share cards                         | Phase 3 stable          |

---

## Project Documents (Local)

- [ðŸ“ All Docs Index](docs/README.md)
- [ðŸ€ Project Overview](docs/project-overview.md)
- [ðŸ“„ PRD](docs/prd.md)
- [ðŸ”§ TDD](docs/tdd.md)
- [ðŸ—‚ï¸ User Stories & Features](docs/user-stories.md)
- [ðŸš€ Phase 0 Spike Plan](docs/phase0-spike-plan.md)
- [ðŸ’» Dev Workflow](docs/dev-workflow.md)
- [ðŸ“‹ Coding Style Guide](docs/coding-style-guide.md)
- [â— Issues Log](docs/issues-log.md)
- [Architecture Decisions](docs/decisions/)

