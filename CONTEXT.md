# CourtVision — AI Session Context

> Paste this file in full at the start of every Codex or Gemini session.
> Last updated: April 2026 — Phase 0 (Technical Spike, Days 1–5 complete)

---

## Project Summary

CourtVision is an **Android-native basketball performance tracking app** using on-device AI,
computer vision, and AR-assisted court mapping to deliver real-time shot analytics,
biomechanical metrics, and player development insights. Comparable product: HomeCourt (iOS).

- Platform: Android API 26+
- Stage: Planning / Pre-Development (Phase 0 spike active)
- Target launch: 12 months from kickoff
- Revenue model: Freemium — free tier + premium analytics
- **MVP scope:** see `docs/prd.md` Section 3.2 — this is the canonical definition.

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

**Progress:** Days 1–5 complete. Day 5 gates passed on S22+ (SM-S906U1): EGL/thread PASS, visual plausibility PASS, pose p95 10.774ms (PASS_STRONG). ADR-005 (sequential GPU pipeline) accepted 2026-04-07; ADR-006 (pose gating + person selection modes) accepted 2026-04-15.

### Day 8 Gate Criteria (all must pass)
- [ ] Ball detection latency < 100ms (p95 < 140ms)
- [ ] Pose inference latency < 50ms (p95 < 70ms)
- [ ] Combined RAM < 400MB sustained
- [ ] End-to-end FPS ≥ 20 sustained for 10 min (target 30)
- [ ] Shot detection stable across Tripod and Ground modes
- Benchmark results committed to `/benchmarks/phase0/`

### Day 5 Validation Snapshot (2026-04-14)
- Device: `SM-S906U1`
- Mode: `sequential_yolo_warm`, `gpu_mode=GPU`
- Gate §1 (EGL/thread): PASS (no `TfLiteGpuDelegate ... must run on the same thread` runtime log observed)
- Gate §2 (visual plausibility): PASS (manual check >= 8/10)
- Gate §3 (latency): PASS_STRONG (`pose_inference_p95_ms=10.774`)
- Artifacts:
  - `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_summary.txt`
  - `benchmarks/phase0/pose_validation/inference_20260414_004536/day5_pose_validation.csv`

### Day 6 Step 2 Verification Note (2026-04-16, updated 2026-04-22)
- `PoseLandmarkInterpreter.infer()` is on the no-alloc code path (`TensorImage` + `ImageProcessor(NormalizeOp)`), and now requires 256x256 input.
- Android Studio Memory Profiler capture dropped from Day 6 sign-off (2026-04-22). Indirect verification via the Step 7 soak: `pose_inference_ms` p50 / p99 = 11.17 / 19.44 ms (spread ~8 ms) across 5089 non-skipped pose frames — inconsistent with per-frame ~1 MB allocations that would trigger GC-pause outliers in the 40–80 ms range. Direct profiler capture deferred to Phase 2 if a pose-side allocation regression is ever suspected.

### Day 6 Step 7 Benchmark Results (2026-04-22; §7 re-scored 2026-04-26)

S22+ 10-min soak, build `762b968`, mode `sequential_yolo_pose`/`gpu_mode=GPU`. Gate verdicts:

- §3 (`frame_total_ms` p95 ≤ 100 ms): **FAIL** (overall p95 = 129 ms) → Day 7 / Phase 2 perf workstream
- §4 (`pose_inference_ms` p95 ≤ 50 ms): **PASS** (p95 = 15 ms)
- §5 (RAM < 400 MB sustained): **PASS** (peak delta 47 MB)
- §6 (FPS ≥ 20 sustained): **FAIL** (median 8 fps) → Day 7 / Phase 2 perf workstream
- §7 (pre-rotation alloc drift): originally FAIL (9.63 MB drift) → re-analysed via order-swapped soaks, rescoped to alloc-path × NONE/LIGHT thermal → **PASS-with-rescope** (watched-path drift = 1.74 MB; original drift was thermal-cadence artifact, not rotation-bitmap leak)

Bottleneck: YOLO inference (p50 53 ms) + thermal ceiling at t=68 s (MODERATE) throttling to ~8 fps. Pose inference healthy. Full latency tables, thermal trajectory, sliced §7 re-analysis, and artifacts: see [docs/plans/day6-combined-pipeline.md](docs/plans/day6-combined-pipeline.md) "Step 7 Results".

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
| Pose Estimation   | `pose_landmarks_detector.tflite` (standalone TFLite `Interpreter`, GPU delegate) — see ADR-005, ADR-006 | **Lite variant** extracted from `pose_landmarker_lite.task`; 33 WorldLandmarks; not via PoseLandmarker task API; gating per ADR-006 (Phase 0 default: EVERY_FRAME_WITH_PERSON + HIGHEST_CONFIDENCE) |
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
ML / CV Layer     → CameraX feed, TFLite YOLO detector, pose_landmarks_detector.tflite (sequential GPU, ADR-005), ARCore
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
:core:ml              — TFLite YOLOv8n multi-class detector, pose_landmarks_detector.tflite interpreter (ADR-005)
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
- `pose_landmarks_detector.tflite` `Interpreter` is NOT thread-safe — same rule (loaded directly, not via PoseLandmarker task API; see ADR-005)
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

