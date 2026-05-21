# CourtVision — On-Device Basketball Tracking (Android)

A technical spike for an Android-native basketball performance tracker that runs
real-time shot detection **fully on-device** — no cloud, no backend. This repository
covers **Phase 0**: proving that a CameraX → object-detection → tracking → pose
pipeline can hit real-time latency budgets on a mid-range Android phone before any
product architecture is committed.

> **Status:** Phase 0 technical spike (Days 1–6.3 complete). Validated on a
> Samsung Galaxy S22+ (`SM-S906U1`).

---

## What this demonstrates

- **Real-time CV pipeline on Android** — CameraX `ImageAnalysis` feeding a custom
  5-class YOLO detector, a Kalman ball tracker, and a MediaPipe pose model, with a
  bounded drop-oldest frame queue for backpressure.
- **NPU acceleration** — INT8 YOLO delegated to the Qualcomm Hexagon NPU (QNN/HTP):
  **357/357 nodes on HTP, 0% CPU fallback**.
- **Disciplined performance engineering** — per-frame CSV instrumentation, 10-minute
  soak tests, thermal-state slicing, and allocation-drift analysis, all under fixed
  latency budgets.
- **Architecture rigor** — seven Architecture Decision Records, a TDD, per-day
  implementation plans, and a dated issues log.

## Headline results (S22+, 10-min soak)

| Metric                          | Result                          |
|---------------------------------|---------------------------------|
| YOLO inference (NPU, INT8)       | p50 **3.46 ms** / p95 3.69 ms   |
| Pose inference (GPU)             | p95 **10.77 ms**                |
| Frame total                     | p95 **81.4 ms**                 |
| HTP node delegation             | 357/357 (0% fallback)           |
| Sustained throughput            | ~14 FPS median                  |

CPU-side preprocessing (~38.9 ms p50) is the dominant remaining cost — the documented
target for the next perf workstream. Full tables, thermal trajectories, and raw CSVs
are in [`benchmarks/phase0/`](benchmarks/phase0/) and the per-day plans under
[`docs/plans/`](docs/plans/).

---

## Tech stack

Kotlin · Jetpack Compose · CameraX · TensorFlow Lite (GPU + QNN/NNAPI delegates) ·
custom-trained YOLO · MediaPipe pose · Kotlin coroutines · JUnit + MockK.

## Pipeline

```
CameraX → YOLO (5-class) → Kalman ball tracker → shot state machine → metrics logger
                        ↘ MediaPipe pose (gated) ↗
```

See [`docs/tdd.md`](docs/tdd.md) and the ADRs in [`docs/decisions/`](docs/decisions/)
for the design rationale.

---

## Model Assets Setup

Model weights are **not committed** to this repository. Two models are required in
`app/src/main/assets/` before the app will build:

### 1. Pose model (MediaPipe — publicly available)

`pose_landmarks_detector.tflite` is extracted from the MediaPipe
`pose_landmarker_lite` bundle:

```bash
# Download the lite variant (~4 MB FP16)
curl -L -o pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# Extract the raw TFLite model from the .task bundle (it is a zip)
cp pose_landmarker_lite.task pose_landmarker_lite.zip
unzip pose_landmarker_lite.zip -d pose_task_extracted/
cp pose_task_extracted/pose_landmarks_detector.tflite \
   app/src/main/assets/pose_landmarks_detector.tflite
```

See [ADR-005](docs/decisions/005-sequential-gpu-inference-pipeline.md) for why the raw
`.tflite` is used directly instead of the `PoseLandmarker` Task API.

### 2. Object detection model (custom-trained — not distributed)

The 5-class detector is a project-trained YOLOv11n exported to INT8 TFLite. The
trained weights are **not redistributed**. Its contract, so the code is fully
readable without it:

- **Classes (nc=5):** `ball`, `made`, `person`, `rim`, `shoot`
- **Input:** 640×640, INT8 quantized
- **Runtime:** QNN/HTP NPU delegate primary, GPU and XNNPACK CPU fallback
- **Output:** split-output decoder — see
  [ADR-007](docs/decisions/007-tflite-npu-split-output-contract.md)

To build and run, drop a compatible 5-class TFLite model into
`app/src/main/assets/` and update the asset filename referenced in the pipeline
code, or substitute any TFLite YOLO with a matching class layout.

---

## Build & Run

1. Open the project folder in Android Studio and let Gradle sync.
2. Provision the model assets (above).
3. Run on a **physical Android device** (camera required; emulator not supported).

---

## Repository layout

| Path | Contents |
|------|----------|
| `app/src/main/`     | Kotlin + Compose app, CV pipeline (`camera/`, `pipeline/`) |
| `app/src/test/`     | Unit tests (JUnit + MockK) |
| `docs/decisions/`   | Architecture Decision Records (ADR 001–007) |
| `docs/plans/`       | Per-day implementation plans |
| `docs/tdd.md`       | Technical design document |
| `benchmarks/phase0/`| Raw per-frame benchmark CSVs and validation summaries |

---

## License & attribution

This project's source is released under the [MIT License](LICENSE).

The pose model is provided by Google's
[MediaPipe](https://github.com/google-ai-edge/mediapipe) project (Apache License 2.0)
and is downloaded at setup time — it is not redistributed here.
