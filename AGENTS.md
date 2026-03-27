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