# Day 1-2 Review Summary
- **Status:** COMPLETED ✅
- **Date:** 2026-03-26
- **Outcome:** CameraX 720p pipeline is stable. Real-time metrics and CSV logging are operational. GPU probe confirms hardware acceleration readiness. Ready for Day 3 YOLO integration.

---

# Day 1-2 Plan — CameraX Pipeline

## Summary

Build a minimal Android app in `CourtVision_Android` that proves a stable camera pipeline before ML integration.
By end of Day 2, you should have: CameraX preview, 720p analysis loop, live FPS/latency overlay, dropped-frame tracking, and a GPU-delegate readiness probe (no YOLO inference yet).

## Implementation Changes

### 1. Project bootstrap (Day 1 morning)

- Create a new **Kotlin + Compose** Android app (minSdk 26, target latest stable SDK).
- Add dependencies: `camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`, `lifecycle-viewmodel-compose`, `kotlinx-coroutines-android`, `tensorflow-lite`, `tensorflow-lite-gpu`.
- Add `CAMERA` permission in manifest and runtime permission flow in Compose.
- Keep architecture minimal: single app module, simple MVVM, no Hilt/Room/Retrofit yet.

### 2. Core spike interfaces (Day 1 midday)

- Define internal contracts now so Day 3 YOLO plugs in without refactor:
  - `data class FramePacket(timestampNs, width, height, rotationDegrees, format)`
  - `data class PipelineStats(analysisFps, avgAnalyzeMs, p95AnalyzeMs, droppedFrames, queueDepth)`
  - `interface FrameConsumer { suspend fun consume(frame: FramePacket) }`
- ViewModel owns `StateFlow<PipelineStats>`; UI only reads state.

### 3. Camera + analysis loop (Day 1 afternoon)

- Build CameraX with `Preview + ImageAnalysis`.
- Use backpressure `KEEP_ONLY_LATEST`.
- Target analysis resolution `1280x720`.
- Analyzer measures per-frame processing time and increments dropped-frame counter.
- Overlay in UI shows: camera running state, FPS, avg/p95 analyze ms, dropped frames.

### 4. Inference scaffolding (Day 2 morning)

- Add async pipeline with a bounded channel (`capacity=1`, drop oldest).
- Analyzer enqueues frame metadata quickly; worker coroutine processes frames.
- Start with simulated inference delay modes (`0ms`, `10ms`, `20ms`) to stress-test throughput.
- Keep frame handoff zero-copy for now (metadata only); avoid bitmap conversion until Day 3.

### 5. GPU delegate readiness + logging (Day 2 afternoon)

- Add a startup probe that attempts to create/close `GpuDelegate` and logs `GPU_AVAILABLE` or `CPU_FALLBACK`.
- Add CSV performance logger (1-second aggregates): timestamp, fps, avg/p95 ms, dropped frames, queue depth, gpu status.
- Run a 10-minute soak test on your primary device and save logs under app files for baseline comparison.

## Suggested File Targets

- `app/src/main/java/.../camera/CameraScreen.kt`
- `app/src/main/java/.../camera/CameraViewModel.kt`
- `app/src/main/java/.../pipeline/FrameProcessor.kt`

## Test Plan (Day 1-2 acceptance)

### 1. Functional smoke tests

- First launch asks camera permission and recovers correctly after deny/allow.
- Camera preview starts within 3 seconds and survives app background/foreground.
- Landscape orientation works; no analyzer crash loops.

### 2. Performance tests

- `0ms` simulated inference: sustained analysis FPS near camera feed baseline (target ~30 on Pixel 6 class devices).
- `20ms` simulated inference: sustained FPS remains stable, no runaway queue growth, dropped frames bounded.
- 10-minute soak: no OOM/crash; stats remain stable; thermal notes recorded manually.

### 3. Gate for Day 1-2 complete

- Stable CameraX preview + analysis at 720p.
- Real-time metrics visible in-app and exported to CSV.
- GPU probe implemented and status logged.
- Ready to replace simulated worker with real YOLO TFLite in Day 3.

## Assumptions and Defaults

- You selected `CourtVision_Android` as the target repo.
- You selected speed-first spike style (minimal MVVM now, full Clean/Hilt later).
- Day 1-2 excludes pose, ARCore, Room, networking, and final UI polish.
- Device baseline is a physical Android phone (Pixel 6/A54 class preferred), not emulator.
