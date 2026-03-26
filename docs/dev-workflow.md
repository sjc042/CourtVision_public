# 💻 Dev Workflow — AI-Delegated Android Development

> CourtVision-specific development workflow for a solo developer delegating to Codex and Gemini. Consolidated 6-step loop designed around the unfamiliar stack, hard ML performance gates, and GitHub Actions CI/CD.

---

## Overview

| Field | Details |
| --- | --- |
| **Stack** | Kotlin 1.9+ · Jetpack Compose · MVVM + Clean Architecture · Hilt · Room |
| **AI Coders** | Codex (logic/pipeline) · Gemini (UI/multimodal debug) |
| **CI/CD** | GitHub + GitHub Actions |
| **Active Phase** | Phase 0 — Technical Spike |
| **Performance Gates** | Ball <100ms · Pose <50ms · RAM <400MB · FPS ≥20 |

---

## The 6-Step Loop

### 1. Scope → Branch

Write the AI brief before touching code. For CourtVision, specify:

- The active phase and day (e.g. "Phase 0, Day 3")
- The exact component to modify (`CameraXInferenceLoop`, `ShotStateMachine`, `PoseLandmarkerWrapper`)
- Which performance targets apply
- What adjacent code to leave untouched

Branch naming:

```
git checkout main && git pull
git checkout -b spike/day3-yolo-tflite-integration
# or: feature/shot-state-machine
# or: perf/reduce-pose-latency
# or: fix/kalman-tracker-drift
# or: chore/ci-add-detekt
```

Use `spike/` for Phase 0 — signals throwaway-safe code that may not survive the Day 8 gate.

---

### 2. Delegate to AI

Split work by AI strength:

**Codex** → Kotlin logic, CameraX inference loop, Kalman filter, shot state machine, Room DAOs, Hilt module wiring, ViewModels, Repositories, UseCases.

**Gemini** → Jetpack Compose UI, XML layouts, multimodal debugging (paste screenshots or Logcat output directly).

Prompt structure every session:

1. **Context block** — paste `CONTEXT.md` + relevant existing files
2. **Task block** — precise goal with class/function names
3. **Constraint block** — `StateFlow` not `LiveData`, no new deps, follow existing error handling pattern

---

### 3. Review Gate

This is the most critical step — the entire stack is unfamiliar, so this is where architectural drift gets caught. Review runs in two passes before you touch the code yourself.

#### Pass 1 — Cross-AI peer review

The AI that did **not** implement the code reviews it first. This catches issues the implementing AI is blind to by design.

| Implemented by | Reviewed by | How |
| --- | --- | --- |
| Codex | Gemini | Paste code + CONTEXT.md into Gemini. Ask it to review against CourtVision's architecture rules. |
| Gemini | Codex | Paste code + CONTEXT.md into Codex. Ask it to review for correctness, thread safety, and Kotlin conventions. |

Prompt to use for cross-AI review:

```
You are reviewing code written by another AI for the CourtVision Android app.
Here is the project context: [paste CONTEXT.md]
Here is the code to review: [paste generated code]

Review specifically for:
1. MVVM + Clean Architecture layer violations
2. Coroutine scoping errors (GlobalScope, wrong dispatcher)
3. ML thread safety (TFLiteInterpreter / PoseLandmarker shared across coroutines)
4. Hilt injection scope mistakes
5. Kotlin anti-patterns (!! operators, var where val works, blocking calls on Main)
6. Any allocations inside the hot inference path

Respond with: issues found (critical / warning / suggestion), and a corrected snippet for each critical issue.
```

If the reviewer AI flags **critical** issues → re-prompt the implementing AI with the specific feedback before proceeding. Do not move to Pass 2 until critical issues are resolved.

#### Pass 2 — Your human review

With AI peer feedback in hand, your review is now a targeted sign-off rather than a full audit.

**Kotlin / Compose**

- Coroutines scoped to `viewModelScope` or `lifecycleScope`, never `GlobalScope`
- No blocking calls on Main thread
- State via `StateFlow` + `collectAsStateWithLifecycle`

**MVVM + Clean layer boundaries**

- UI → ViewModel → UseCase → Repository — no skipping layers
- No Android imports in domain layer classes
- No Repository calls from Fragments directly

**ML thread safety**

- Inference on `Dispatchers.Default` or a dedicated `ExecutorService`, never Main
- `TFLiteInterpreter` and `PoseLandmarker` are not thread-safe — check for shared instances across coroutines

**Hilt**

- Scope correct: `@Singleton` for interpreter, `@ViewModelScoped` for use cases
- No manually constructed dependencies that Hilt should own

**Performance**

- No new allocations inside `ImageAnalysis.Analyzer.analyze()`
- No latency-adding operations added to the hot inference path

If the output fails your review → re-prompt the implementing AI with the specific issue, not a full rewrite request.

---

### 4. Validate Locally

```
./gradlew ktlintCheck
./gradlew detekt
./gradlew test
./gradlew assembleDebug
```

For any change touching the ML pipeline, run an **on-device benchmark** before pushing — not after CI. Commit results to `/benchmarks/phase0/` as required by the Phase 0 protocol.

PRD hard targets to validate on-device:

| Metric | Target |
| --- | --- |
| Ball detection latency | <100ms (p95 <140ms) |
| Pose inference latency | <50ms (p95 <70ms) |
| End-to-end FPS | ≥20 sustained (target 30) |
| RAM footprint | <400MB sustained |
| Battery usage | <15%/hr |

---

### 5. PR → GitHub Actions CI

**GitHub Actions workflow** (`.github/workflows/android.yml`):

```yaml
name: Android CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      - uses: actions/cache@v4
        with:
          path: |
            ~/.gradle/caches
            ~/.gradle/wrapper
          key: gradle-${{ hashFiles('**/*.gradle*') }}
      - run: ./gradlew ktlintCheck
      - run: ./gradlew detekt
      - run: ./gradlew test
      - run: ./gradlew assembleDebug
      - uses: actions/upload-artifact@v4
        with:
          name: debug-apk
          path: app/build/outputs/apk/debug/*.apk
```

**PR description template:**

```
## What
<one-line summary>

## Why
<user story or issue reference>

## How to test
1. Step one
2. Step two

## AI-generated
Codex: <what was generated and what you modified>
Gemini: <what was generated and what you modified>

## Benchmark log
<attach CSV for any ML pipeline changes>

## Screenshots / recordings
<attach if UI changed>
```

Tagging AI-generated sections creates an audit trail across sessions with two different AI coders.

---

### 6. Squash Merge → Ship → Monitor

**Merge:** Squash merge collapses AI iteration commits into one clean commit on `main`. After merge:

```
git checkout main && git pull
git branch -d spike/day3-yolo-tflite-integration
```

**Deploy track (Play Console):**

- Internal testing → smoke test on physical device
- Alpha → after internal passes, promote via CI/CD (`upload-google-play` action)
- Production → manual promotion after alpha validation

**Post-deploy monitoring:**

- Firebase Crashlytics → crash-free rate for 24–48hrs post-release
- Play Console → Android Vitals → ANR rate and crash rate per release
- Logcat on a production build → run through the changed feature manually

**Phase gate check:** Before starting the next phase, verify all Day 8 gate criteria are green and `/benchmarks/phase0/` is committed and complete.

---

## CONTEXT.md — Anchor File

Keep a `CONTEXT.md` at the repo root and paste it at the start of every Codex and Gemini session. Prevents architectural drift across sessions.

It should contain:

- Architecture rules (MVVM + Clean, layer boundaries)
- Active phase and what's in/out of scope
- Performance hard limits (the 6 PRD targets)
- Hilt module structure
- Naming conventions and package structure
- Key files and their responsibilities

---

## AI Coder Split Reference

| Task type | AI to use |
| --- | --- |
| Kotlin logic, ViewModel, UseCase, Repository | Codex |
| CameraX inference loop, Kalman filter | Codex |
| Shot state machine (IDLE → PREP → RELEASE → FLIGHT → OUTCOME) | Codex |
| Room DAOs, Hilt modules | Codex |
| Jetpack Compose UI, XML layouts | Gemini |
| Debugging with Logcat output or screenshot | Gemini |
| TFLite / MediaPipe integration | Codex (logic) + Gemini (debug) |

---

## Branch Naming Convention

| Prefix | Use case |
| --- | --- |
| `spike/` | Phase 0 experiments — throwaway safe |
| `feature/` | New capabilities (Phase 2+) |
| `fix/` | Bug fixes |
| `perf/` | Latency, RAM, battery improvements |
| `refactor/` | Architecture cleanup, no behaviour change |
| `chore/` | CI, tooling, dependency updates |
