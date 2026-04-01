# CourtVision — Codex Session

Read before doing anything else:
- ./CONTEXT.md
- ./TASKS.md

## Your role
- Kotlin logic: ViewModel, UseCase, Repository
- CameraX inference loop, Kalman filter
- Shot state machine (IDLE → PREP → RELEASE → FLIGHT → OUTCOME)
- Room DAOs, Hilt module wiring
- TFLite / MediaPipe integration logic

## Rules
- No new dependencies without flagging
- No layer skipping: UI → ViewModel → UseCase → Repository
- Inference always on Dispatchers.Default, never Main
- TFLiteInterpreter and PoseLandmarker are not thread-safe — never share across coroutines
- No allocations inside ImageAnalysis.Analyzer.analyze()

## Build & Test

**Do NOT run Gradle, build commands, or tests yourself.** Your sandbox environment lacks the correct JDK, writable Gradle home, and Android SDK paths — builds will fail with environment errors, not code errors.

Instead:
- Write the code and tests
- Tell the user exactly which test commands to run (class names, flags)
- The user will run them locally and paste back results
- Do NOT create workaround directories (.gradle-user, .gradle-user-home, etc.) in the repo

## When commiting changes or pushing to remote
- Review ./docs/dev-workflow.md before proceeding. Instruct user to validate.
