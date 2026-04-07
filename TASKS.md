# Tasks 🔲 Phase 0 Spike

Last updated: 2026-04-02

## Active branch
'spike/day4-kalman-tracker' (branch from `main`)

---

## Completed 🔲 Day 3 (2026-03-28)

All Day 3 goals achieved and committed. See `docs/plans/day3+weekend-plan.md` Part 1.

1. ✅ Export COCO-pretrained YOLOv8n `.pt` weights to TFLite FP16, plug into CameraX inference loop
2. ✅ Implement UI toggle for CPU vs GPU inference
3. ✅ Add pre-session model selector UI; lock after first inference frame
4. ✅ Render live bounding boxes on overlay with post-NMS class whitelist
5. ✅ Measure per-frame inference latency, FPS, RAM, thermal over 10 min; log `model_used` to CSV
6. ✅ Support end-to-end YOLO models (NMS built-in) alongside standard YOLO (external NMS)
7. ✅ Switch to TFLite Support Library preprocessing (`ImageProcessor` + `TensorImage`) 🔲 ~3Ã— faster than manual Bitmap path
8. ✅ On-device preprocessing and parsing latency benchmarks

## Completed ✅ Weekend Track (2026-03-29 ✅ 2026-03-30)

1. ✅ Collect basketball datasets from Roboflow (3 sources combined)
2. ✅ Clean duplicates (exact match dedup across and within splits)
3. ✅ Train YOLOv8n at 640/480/320 🔲 50 epochs each

### Dataset

| Property | Value |
|----------|-------|
| Source | Combined: `MathieuLec` + `OwnProjects` + `test-datset` |
| Format | YOLOv8 (`.txt` annotations) |
| Classes (nc=5) | `ball`, `made`, `person`, `rim`, `shoot` |
| Train | 13,578 images |
| Valid | 866 images |
| Test | 1,412 images |
| **Total** | **15,856 images** |
| Location | `D:\Basketball Datasets\BasketBall.v1i.MathieuLec+OwnProjects+test-datset.yolo26` |

### Training Results (50 epochs each)
| Model | Input Size | mAP50 | mAP50-95 | Notes |
|-------|-----------|-------|----------|-------|
| yolov8n | 640x640 | 0.94006 | 0.73543 | |
| yolov8n | 480x480 | 0.93476 | 0.72150 | |
| yolov8n | 320x320 | 0.89546 | 0.66649 | |
| yolo11n | 640x640 | 0.93473 | 0.74061 | |
| yolo11n | 480x480 | 0.93375 | 0.73111 | |
| yolo11n | 320x320 | 0.90842 | 0.67137 | |
| yolov8s | 640×640 | 0.94244 | 0.76115 | |

### Remaining Training

| Model | Input Size | Status |
|-------|-----------|--------|
| yolov8s | 640 / 480 / 320 | Pending |
| yolov11s | 640 / 480 / 320 | Pending |

---

## Active Post-Weekend / Day 4 Prep (user)

1. ✅ Benchmarked manual vs Support Library preprocessing; switched to Support Library (~3Ã— faster, resolution-independent ~19ms p50)
2. 🔲 Complete remaining training matrix (yolov8s, yolov11n, yolov11s at 3 resolutions)
3. 🔲 Export best model(s) to TFLite FP16, drop into Android `assets/`
4. 🔲 Run Day 3 benchmark loop with custom-trained model 🔲 validate detection quality on-device
5. 🔲 Update planning docs to reflect 5-class model, dataset details, and training results (in progress)

## Active 🔲 Day 4: Kalman Tracker

> Branch: 'spike/day4-kalman-tracker'
> Plan: `./docs/plans/day4_kalman-filter-tracker.md`

1. ✅ Add `TrackedBall` data class + extend `DetectionFrame` in `FrameContracts.kt`
2. ✅ Implement `KalmanBallTracker.kt` - pure Kotlin, constant-velocity 4-state, `maxMissFrames` configurable at runtime
3. ✅ Integrate tracker into `FrameProcessor.kt` - predict+update after each NMS pass, real `dt` from frame timestamps
4. ✅ Propagate `trackedBall` through `CameraUiState` + `CameraViewModel`; add `setTrackerMaxMissFrames()` setter
5. ✅ Add miss-frame slider (range 1-30, default 10) to debug controls in `CameraScreen.kt`
6. ✅ Draw cyan tracker circle + velocity vector on canvas overlay in `CameraScreen.kt`
7. ✅ Extend CSV logging with `tracking_active, track_cx, track_cy, track_vx, track_vy, miss_streak`
8. ✅ Write `docs/decisions/003-kalman-ball-tracker.md` (ADR-003)
9. ✅ Write `KalmanBallTrackerTest` unit tests
10. ✅ On-device validation: 10-shot sequence, confirm smooth trajectory in exported CSV

## Out of scope (still)
- Pose estimation (Day 5)
- Shot detection FSM (Day 7)

## Added - NNAPI Delegate Mode (2026-04-01)

1. Added `InferenceMode.NNAPI` with runtime switching and CPU fallback on delegate init failure
2. Added NNAPI probe visibility in debug overlay and mode selector gating
3. Added unit tests covering NNAPI probe state and CSV serialization of `delegate_mode=NNAPI`
4. Phase 0 choice: GPU/NNAPI probes run synchronously in `CameraViewModel.init`
5. Phase 2 note: move both delegate probes to `Dispatchers.Default`
