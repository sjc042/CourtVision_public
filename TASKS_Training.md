## Model Training Track (Day 3 Weekend) 🔲 In Progress (non-blocking)

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
| yolov11s | 640×640| 0.94717 | 0.76909 | |

### Remaining Training

| Model | Input Size | Status |
|-------|-----------|--------|
| yolov8s | 480 / 320 | Pending |
| yolov11s | 640 / 480 / 320 | Pending |

### Non-Blocking Follow-Up Tasks (user)

1. 🔲 Complete remaining training matrix (yolov8s 480/320, yolov11s 640/480/320)
2. 🔲 Export best model(s) from remaining training to TFLite FP16 → drop into Android `assets/`
3. 🔲 Update training matrix status above once runs complete

---