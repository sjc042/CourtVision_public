### Context override: this is only and example of what a tasks file looks like

# Tasks — Phase 0 Day 3 


## Active branch
spike/day3-yolo-tflite-integration

## Today's goal
Plug YOLOv8n TFLite model into the CameraX inference loop.
Measure per-frame latency, FPS, RAM over 10 minutes.

## In scope
- TFLite interpreter setup in FrameProcessor.kt
- GPU delegate wiring (probe already done in Day 1-2)
- Latency logging to existing CSV logger

## Out of scope
- Kalman tracker (Day 4)
- Pose estimation (Day 5)
- Any UI changes

## Files to touch
- pipeline/FrameProcessor.kt
- pipeline/SpikeImageAnalyzer.kt

## Acceptance criteria
- Ball detection latency p50 < 100ms on primary device
- No crash in 10-minute soak test