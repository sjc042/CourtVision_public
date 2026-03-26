# CourtVision Spike (Day 1-2)

Minimal Android spike app for CameraX pipeline validation before YOLO integration.

## What is implemented

- Kotlin + Compose single-module app (minSdk 26)
- CameraX Preview + ImageAnalysis
- Backpressure strategy: `KEEP_ONLY_LATEST`
- Analysis target: `1280x720`
- Async processing pipeline:
  - bounded channel capacity = 1
  - drop oldest on overflow
  - simulated processing delays (0/10/20ms)
- Runtime overlay:
  - camera status
  - GPU probe status
  - analysis FPS
  - avg/p95 analyze time
  - dropped frame count
  - queue depth
  - CSV output path
- GPU delegate compatibility probe (`GPU_SUPPORTED`, `GPU_UNSUPPORTED`, `GPU_INIT_FAILED`)
- CSV performance logging (1-second aggregates) at:
  - `sdcard/Android/data/com.courtvision.spike/files/benchmarks/phase0-day1-day2-<timestamp>.csv`
  - fallback: `files/benchmarks/phase0-day1-day2-<timestamp>.csv` if external files dir is unavailable

## Open in Android Studio

1. Open folder: `CourtVision_Android`
2. Let Gradle sync download dependencies.
3. Run on a physical Android device (camera required).

## 10-minute soak test

1. Launch app and grant camera permission.
2. Let pipeline run for 10 minutes.
3. Toggle delay mode `0ms`, then `20ms`.
4. Pull CSV from app files and inspect FPS, p95, and dropped frames.
