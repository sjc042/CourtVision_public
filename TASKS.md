# Tasks — Phase 0 Spike

Last updated: 2026-03-30

## Active branch
spike/day3-yolo-tflite-integration

---

## Completed — Day 3 (2026-03-28)

All Day 3 goals achieved and committed. See `docs/plans/day3+weekend-plan.md` Part 1.

1. ✅ Export COCO-pretrained YOLOv8n `.pt` weights to TFLite FP16, plug into CameraX inference loop
2. ✅ Implement UI toggle for CPU vs GPU inference
3. ✅ Add pre-session model selector UI; lock after first inference frame
4. ✅ Render live bounding boxes on overlay with post-NMS class whitelist
5. ✅ Measure per-frame inference latency, FPS, RAM, thermal over 10 min; log `model_used` to CSV
6. ✅ Support end-to-end YOLO models (NMS built-in) alongside standard YOLO (external NMS)
7. ✅ Switch to TFLite Support Library preprocessing (`ImageProcessor` + `TensorImage`) — ~3× faster than manual Bitmap path
8. ✅ On-device preprocessing and parsing latency benchmarks

## Completed — Weekend Track (2026-03-29 – 2026-03-30)

1. ✅ Collect basketball datasets from Roboflow (3 sources combined)
2. ✅ Clean duplicates (exact match dedup across and within splits)
3. ✅ Train YOLOv8n at 640/480/320 — 50 epochs each

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

### Training Results (YOLOv8n, 50 epochs each)

| Model | Input Size | mAP50 | mAP50-95 | Notes |
|-------|-----------|-------|----------|-------|
| yolov8n | 640x640 | 0.94006 | 0.73543 | |
| yolov8n | 480x480 | 0.93476 | 0.72150 | |
| yolov8n | 320x320 | 0.89546 | 0.66649 | |

### Remaining Training

| Model | Input Size | Status |
|-------|-----------|--------|
| yolov8s | 640 / 480 / 320 | Pending |
| yolov11n | 640 / 480 / 320 | Pending |
| yolov11s | 640 / 480 / 320 | Pending |

---

## Active — Post-Weekend / Day 4 Prep

1. ✅ Benchmarked manual vs Support Library preprocessing; switched to Support Library (~3× faster, resolution-independent ~19ms p50)
2. 🔲 Complete remaining training matrix (yolov8s, yolov11n, yolov11s at 3 resolutions)
2. 🔲 Export best model(s) to TFLite FP16, drop into Android `assets/`
3. 🔲 Run Day 3 benchmark loop with custom-trained model — validate detection quality on-device
4. 🔲 Update planning docs to reflect 5-class model, dataset details, and training results (in progress)

## Next — Day 4: Kalman Tracker

- Add Kalman filter tracker on ball bounding box output
- Track ball position and velocity across frames
- Measure tracking stability and false positive rate
- Output: smooth ball trajectory over a 10-shot sequence

## Out of scope (still)
- Pose estimation (Day 5)
- Shot detection FSM (Day 7)
