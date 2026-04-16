# CourtVision Spike (Phase 0)

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

## Model Assets Setup

`pose_landmarks_detector.tflite` is not checked into the repo. Extract it from the MediaPipe
`.task` bundle once per machine before building:

```bash
# 1. Download the lite variant (~4 MB FP16)
curl -L -o pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# 2. Extract the raw TFLite model
cp pose_landmarker_lite.task pose_landmarker_lite.zip
unzip pose_landmarker_lite.zip -d pose_task_extracted/
cp pose_task_extracted/pose_landmarks_detector.tflite \
   app/src/main/assets/pose_landmarks_detector.tflite
```

Expected file size: ~4 MB. If the build fails with a missing asset error, re-run this step.
See [ADR-005](docs/decisions/005-sequential-gpu-inference-pipeline.md) for why the raw `.tflite`
is used instead of the `PoseLandmarker` Task API.

## Open in Android Studio

1. Open folder: `CourtVision_Android`
2. Let Gradle sync download dependencies.
3. Run on a physical Android device (camera required).

## 10-minute soak test

1. Launch app and grant camera permission.
2. Let pipeline run for 10 minutes.
3. Toggle delay mode `0ms`, then `20ms`.
4. Pull CSV from app files and inspect FPS, p95, and dropped frames.
