# Day 3 & Weekend Plan — YOLO TFLite Integration

**Status:** Day 3 ✅ COMPLETED (2026-03-28) | Weekend ✅ COMPLETED (2026-03-30)

---

## Part 1 — Day 3 (Friday): YOLO Integration & Benchmark ✅

### Summary

Validate that the YOLO model architecture runs end-to-end on-device at target speed.
Export COCO-pretrained YOLOv8n (or YOLO26n) weights to TFLite FP16, plug into the CameraX
pipeline, add a runtime CPU/GPU toggle, and run a 10-minute benchmark on the Samsung Galaxy S22+.
Detection outputs are not meaningful with COCO weights — Day 3 is a pure latency/throughput test.
Custom basketball+hoop weights are produced by the weekend track (Part 2).

> Branch: `spike/day3-yolo-tflite-integration`

### Prerequisites

- Day 1-2 complete: CameraX 720p pipeline stable, GPU delegate probe passing ✅
- ultralytics Python package installed: `pip install ultralytics`

### Implementation Steps

#### 0. Model export (Python/CLI — run before Android work) ✅

Export COCO-pretrained weights to TFLite FP16 on a desktop/Colab environment:

```bash
# YOLOv8n
yolo export model=yolov8n.pt format=tflite half=True imgsz=640

# YOLO26n (if weights available — substitute equivalent ultralytics CLI)
yolo export model=yolo26n.pt format=tflite half=True imgsz=640
```

- Place the output `.tflite` file(s) in `app/src/main/assets/`
- These are COCO-pretrained weights — Day 3 uses them as a speed proxy only, not for real detection

#### 1. TFLite interpreter setup — `FrameProcessor.kt` ✅

- Load the `.tflite` model from `assets/` using `Interpreter(loadModelFile(...))`
- Build `Interpreter.Options`: attach `GpuDelegate` when GPU mode is selected, use CPU otherwise
- ~~Pre-allocate input buffer: `ByteBuffer` shaped `[1, 640, 640, 3]` FP16 — allocate once, reuse per frame~~ Replaced by TFLite Support Library `ImageProcessor` (see §2 deviation note)
- Pre-allocate output buffers for bounding boxes + confidence scores — no allocation inside `analyze()`
- Run inference on `Dispatchers.Default` — never on Main thread
- `TFLiteInterpreter` is NOT thread-safe: one instance per coroutine/executor, never shared

#### 2. Frame preprocessing — `SpikeImageAnalyzer.kt` ✅

- Convert `ImageProxy` (YUV_420_888) → RGB bitmap → scale to 640×640
- Normalize pixel values to `[0, 1]` range and write into the pre-allocated FP16 `ByteBuffer`
- Pass the filled buffer to `FrameProcessor` via the existing `FrameConsumer` interface
- **Implementation deviation:** Initially switched to manual Bitmap preprocessing (commit `d8e3595`), then replaced with TFLite Support Library `ImageProcessor` (`ResizeOp` BILINEAR + `NormalizeOp`) after benchmarking showed ~3× speedup (p50 ~19ms vs ~52ms manual, resolution-independent)

#### 3. Post-processing ✅

- Parse raw output: `[1, num_boxes, 6]` (x, y, w, h, conf_ball, conf_hoop) or model-specific layout
- Apply confidence threshold (start at 0.4) and NMS
- Emit `DetectionResult(ballBox: BoundingBox?, hoopBox: BoundingBox?, inferenceMs: Long)` via `StateFlow`
- **Implementation deviation:** Added support for end-to-end YOLO models with built-in NMS (commit `d8e3595`)

#### 4. CPU/GPU toggle + model selector UI + live stats — `camera/CameraScreen.kt` ✅

- Add a toggle button to the existing live overlay (button label: "GPU" / "CPU")
- Add a model selector UI that lists discovered `*_float16.tflite` / `*_float32.tflite` assets
- Default model selection: `yolov8n_saved_model/yolov8n_float16.tflite`
- Disable model selector after the first inference frame to prevent mid-session model switching
- On toggle: close current `Interpreter` + `GpuDelegate`, reinitialise with new options
- Reinitialisation must happen off Main thread; show a brief loading indicator during switch
- Persist the selected mode in `PipelineStats` so it is logged to CSV
- Display live stats on the overlay, updated each frame from `PipelineStats`:
  - Inference latency (ms) — last frame value
  - FPS — `analysisFps` from `PipelineStats`
  - Active delegate mode (CPU / GPU)
  - Active model short name (for example `yolov8n-fp16`)

#### 5. Live bounding box overlay — `camera/CameraScreen.kt` ✅

Qualitative sanity check: render detection boxes live on the camera feed to confirm the pipeline
is producing plausible output before running the formal benchmark.

- After NMS, filter to a class whitelist before passing to the overlay — drop all other boxes:
  - `sports ball` (class 32) — covers basketballs; primary signal
  - `person` (class 0) — optional but useful; confirms subjects are detected, relevant for Day 5 pose work
  - All other COCO classes: discard (noise on a basketball court)
- Draw each surviving box with its class label and confidence score
- COCO has no hoop class — no hoop boxes expected; this is fine for Day 3
- Pass/fail bar is low: a box should appear on the ball and roughly track it as you move it
- No accuracy measurement needed here — video-based Precision/Recall testing is post-weekend (Part 2, Step 5)

#### 6. Benchmark run — 10-minute soak test ✅

- Record per-frame: `inferenceMs`, `analysisFps`, `ramMb`, `delegateMode` (CPU/GPU), `modelUsed` (asset-relative model path)
- Thermal throttle detection: flag any 30-second window where FPS drops >20% below rolling average
  (see [frame-scheduling-spec.md](frame-scheduling-spec.md) §Concern 3 for definition)
- Run once on GPU delegate, once on CPU — log both
- Commit CSV + summary table to `/benchmarks/phase0/day3-{gpu|cpu}-s22plus.csv`
- Ensure CSV contains a populated `model_used` column for every row

### Acceptance Criteria

- Model inference latency p50 < 100ms on Samsung Galaxy S22+
- Model inference latency p95 < 140ms on Samsung Galaxy S22+
- No crash during 10-minute soak test
- Model selector is available before inference starts and disabled after first inference frame
- Benchmark CSV + summary table committed to `/benchmarks/phase0/`
- Benchmark CSV includes non-empty `model_used` values

### Code Review Notes (post-implementation)

**CameraViewModel.kt — init ordering dependency:**
`analyzer` (line 44-46) captures `_uiState` via lambda; moved after `_uiState` declaration so init order is safe. The lambda is only invoked at analysis time, not construction — no bug, but the ordering is load-bearing. Don't reorder these fields.

**FrameProcessorTest.kt — fragile test dependencies:**
1. `FakeImageProxy` pulls Android/CameraX framework types (`Rect`, `Image`, `ImageInfo`, `TagBundle`) into `src/test/` (local JVM). Tests pass only because Robolectric or CameraX stubs are on the classpath — implicit dependency, will break if test config changes.
2. `FakeImageProxy.imageInfo` implements `populateExifData(ExifData.Builder)` — `ExifData` is a CameraX **internal** class (`androidx.camera.core.impl.utils`). Brittle across CameraX version bumps.

**Recommendation:** Acceptable for spike. If tests become flaky on CameraX upgrade, replace `FakeImageProxy` with MockK mock or move `submitImage_*` tests to `src/androidTest/`.

### Out of Scope

- Kalman tracker (Day 4)
- Pose estimation (Day 5)
- Custom-trained model weights (weekend track — see Part 2 below)

---

## Part 2 — Weekend Track: Dataset Collection & Model Training ✅

### Summary

Non-blocking parallel track completed over the weekend. Produced custom-trained YOLOv8n
TFLite FP16 models (5 classes: `ball`, `made`, `person`, `rim`, `shoot`) ready to swap into the Day 3
inference loop for Day 4+ testing.

> Run in a separate environment (local GPU machine), not the Android repo.

### Step 1 — Dataset Collection (Roboflow) ✅

- Combined 3 Roboflow datasets: `MathieuLec` + `OwnProjects` + `test-datset`
- Downloaded in YOLOv8 format
- **Actual:** 15,856 total images (13,578 train / 866 valid / 1,412 test)
- **Classes (nc=5):** `ball`, `made`, `person`, `rim`, `shoot`
- Location: `D:\Basketball Datasets\BasketBall.v1i.MathieuLec+OwnProjects+test-datset.yolo26`

### Step 2 — Dataset Cleaning ✅

- Cleaned duplicates inside each individual dataset using corresponding `duplicates_report_*.csv`
- Combined all datasets, re-ran duplicate detection on merged train/valid sets
- Exact match dedup only (`type="exact"`, `hamming_distance=0`)
- Final class indices in `data.yaml`: `0=ball`, `1=made`, `2=person`, `3=rim`, `4=shoot`

### Step 3 — Training ✅ (YOLOv8n) / 🔲 (remaining)
#### Training Results (50 epochs each)
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

Settings: Mixed precision, no dropout, default YOLO augmentation, cosine LR (50 epochs sufficient for convergence).

### Step 4 — Export to TFLite FP16 ✅ (YOLOv8n)

```bash
yolo export model=runs/detect/train/weights/best.pt format=tflite half=True imgsz=640
```

- Verified exported `.tflite` loads correctly
- Input shape: `[1, imgsz, imgsz, 3]`

### Step 5 — Detection Quality Measurement

Evaluated via training validation metrics (mAP50, mAP50-95). Formal held-out test split evaluation TBD.

| Metric | How measured |
| --- | --- |
| Precision | TP / (TP + FP) at IoU ≥ 0.5 |
| Recall | TP / (TP + FN) at IoU ≥ 0.5 |
| mAP@0.5 | Mean Average Precision across all 5 classes |

Per `docs/phase0-spike-plan.md` §Formal Evaluation Protocol — record results per class
(`ball`, `made`, `person`, `rim`, `shoot`) separately.

### Deliverable

- Custom-trained YOLOv8n `.tflite` models at 640/480/320 — ready to drop into Android `assets/`
- Training results logged in `training/training_log.csv`

### Out of Scope

- On-device latency benchmarking (Day 3 uses COCO-pretrained weights for that)
- Fine-tuning on own recorded footage (post-spike, per TDD §10 open questions)

> Note: the `.tflite` models produced here are the ones that make Day 3's inference results
> meaningful — once swapped in, the same benchmark loop from Day 3 tests real basketball+hoop detection.


## Phase 0 Post-Weekend — 2026-03-30

### Preprocessing benchmark: manual vs TFLite Support Library

**Finding:** Manual `fillInputTensorFromBitmap` (nearest-neighbor resize + per-pixel normalize into `ByteBuffer`) is ~3× slower than the TFLite Support Library `ImageProcessor` (`ResizeOp` BILINEAR + `NormalizeOp`).

**Benchmark results (p50, Samsung Galaxy S22+, 2 runs averaged):**

| Resolution | Manual | Support Library |
|---|---|---|
| 640×480 | ~51ms | ~19ms |
| 1280×720 | ~52ms | ~19ms |
| 1920×1080 | ~58ms | ~19ms |

**Key observations:**
- Support Library is resolution-independent (~19ms regardless of input size)
- Manual path scales with source resolution and has wider p95 variance
- Support Library uses bilinear interpolation (better quality than manual nearest-neighbor)

**Decision:** Switched `FrameProcessor` to use `ImageProcessor` + `TensorImage`. Added `tensorflow-lite-support:0.4.4` dependency. Removed manual `fillInputTensorFromBitmap`, `inputTensorBuffer`, and `reusablePixels`.