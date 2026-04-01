# Tasks — Phase 0 Day 3


## Active branch
spike/day3-yolo-tflite-integration

## Today's goal (Friday — Day 3 main focus)
3. Plug single multi-class TFLite FP16 model (YOLOv8n or YOLO26n; 2 classes: basketball + hoop; 640×640 input) into the CameraX inference loop.
4. implement UI toggle for CPU vs GPU inference
5. Measure per-frame latency, FPS, RAM, and thermal behavior over 10 minutes, log using CPU or GPU
NOTE: use pretrained model weights for tasks 3–5 (custom-trained model not required).

## Weekend track (non-blocking, does not gate Day 3 output)
1. Collect and clean basketball/hoop dataset from roboflow
2. Train detector YOLOv8n and YOLO26n on basketball/hoop dataset; measure Precision/Recall at IoU ≥ 0.5 per docs/tdd.md metrics and docs/phase0-spike-plan.md.

## In scope
- TFLite interpreter setup in FrameProcessor.kt
- GPU delegate wiring (probe already done in Day 1-2)
- Latency logging to existing CSV logger

## Out of scope
- Kalman tracker (Day 4)
- Pose estimation (Day 5)

## Files to touch
- pipeline/FrameProcessor.kt
- pipeline/SpikeImageAnalyzer.kt
- camera/SpikeOverlayView.kt (or equivalent) — CPU/GPU toggle UI

## Acceptance criteria
- Ball detection latency p50 < 100ms on Samsung Galaxy S22+
- Ball detection latency p95 < 140ms on Samsung Galaxy S22+
- No crash in 10-minute soak test
- Benchmark CSV + summary table committed to /benchmarks/phase0/