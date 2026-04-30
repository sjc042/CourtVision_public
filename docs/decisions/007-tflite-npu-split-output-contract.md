# ADR-007: TFLite NPU Output Contract — Split Detection Head for INT8 Models

**Status:** Accepted  
**Date:** 2026-04-28  
**Deciders:** CourtVision core team  
**Relates to:** [ADR-005](005-sequential-gpu-inference-pipeline.md) Config B (QNN NPU path, §Deferred)

---

## Context

The Phase 0 GPU pipeline (ADR-005 Config A) uses a single `(1, 9, 8400)` FP32/FP16 output tensor — 4 box coordinate channels followed by 5 class score channels, all in one buffer. This layout is straightforward to decode because both numeric ranges (image-space pixels and 0–1 probabilities) are preserved at floating-point precision.

Day 6.1 introduces the QNN NPU path, which requires an INT8 QDQ TFLite model (QAI Hub Route B). During validation, the first QAI Hub export retained the combined `(1, 9, 8400)` shape. All 5 validation metrics collapsed to zero despite box channels remaining numerically alive.

Root-cause investigation (`research-reports/Codex-quantized-yolo-zero-map-findings.md`):

| Tensor | Optimized ONNX (FP) | Quantized ONNX (INT8) |
|--------|--------------------|-----------------------|
| Box min / max | 2.2 / 637.9 | 2.5 / 636.5 |
| Class min / max | 0.0 / 0.881 | **0.0 / 0.0** |

The INT8 quantizer calibrated a shared per-tensor scale to the box magnitude range (~2.62). Class probability values (0–1) are smaller than one quantization step at that scale and round to exactly zero after dequantization. The result is zero mAP across all classes regardless of calibration data quality.

---

## Decision

**For all QNN_NPU / INT8 TFLite model assets, the YOLO detection head must be exported with split output tensors:**

| Tensor | Name | Shape | Quantization |
|--------|------|-------|-------------|
| Box coordinates | `output_0` | `(1, 4, 8400)` | scale≈2.62, calibrated to image-space pixels (0–640) |
| Class scores | `output_1` | `(1, 5, 8400)` | scale=0.00390625 (≈1/256), calibrated to 0–1 probability range |

Both tensors use `dtype=int8, zero_point=-128`.

The combined `(1, 9, 8400)` format is **explicitly rejected** for any INT8 export targeting the Hexagon HTP backend.

The FP16 GPU asset (`yolo11n_640_5-class_04-01-2026_float16.tflite`) retains the combined `(1, 9, 8400)` layout. This decision applies only to quantized INT8 models.

---

## Validated Asset

```
spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite
  Input  : image     (1, 3, 640, 640)  int8  scale=0.003921568859368563  zp=-128  [NCHW]
  output_0            (1, 4, 8400)      int8  scale=2.621687412261963     zp=-128  [boxes]
  output_1            (1, 5, 8400)      int8  scale=0.00390625            zp=-128  [scores]

Validation run: spike_qai_yolo11n_640_5-class_04-28-2026_int8_14
  mAP50(B): 0.9143  (borderline/accepted — PTQ ceiling on Route B; see Day 6.1 plan §0c)
  conf=0.1, iou=0.35, max_det=20
```

---

## Consequences

### Android decoder change

`FrameProcessor.kt` must branch on `InferenceMode.QNN_NPU` to read two output tensors instead of one:

```kotlin
// Dequantization: float_val = (int8_val.toInt() + 128) * scale  (zp = -128 for both)
// output_0: boxes — (val + 128) * 2.621687  → image-space cx, cy, w, h
// output_1: scores — (val + 128) * 0.00390625 → apply sigmoid → class probabilities
```

NMS parameters matched to validation: `conf=0.1, iou=0.35, max_det=20`.  
Implementation tracked in [Day 6.1 plan Step 5d](../plans/day6-1-qnn-npu-pipeline.md).

### Asset validation requirement

Before any new INT8 model is placed in `app/src/main/assets/`, verify split heads are present:

```python
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="<model>.tflite")
interp.allocate_tensors()
outputs = interp.get_output_details()
assert len(outputs) == 2, "Expected split output heads (boxes + scores)"
assert outputs[0]["shape"].tolist() == [1, 4, 8400]
assert outputs[1]["shape"].tolist() == [1, 5, 8400]
```

### GPU path unaffected

`InferenceMode.CPU / GPU / NNAPI` continue to use the combined FP16 decoder — no changes to those paths.

---

## Alternatives Considered

| Option | Rejected reason |
|--------|----------------|
| Keep combined `(1, 9, 8400)` and use per-channel quantization | QAI Hub PTQ does not expose per-channel output quantization for this op; would require QAT |
| Exclude class head from quantization (partial INT8) | Requires custom graph surgery; not supported in the QAI Hub standard pipeline |
| Use higher bit-width (INT16 for class scores) | HTP runtime support for mixed INT8/INT16 graphs is device-specific and unvalidated on SM8450 |
| Raise confidence threshold to compensate for zero-class scores | Cannot compensate for numerically zero scores regardless of threshold |

---

## References

- [`research-reports/Codex-quantized-yolo-zero-map-findings.md`](../../research-reports/Codex-quantized-yolo-zero-map-findings.md) — root-cause analysis
- [`docs/plans/day6-1-qnn-npu-pipeline.md`](../plans/day6-1-qnn-npu-pipeline.md) — DD-10, Step 5d
- [ADR-005 §Deferred Config B](005-sequential-gpu-inference-pipeline.md) — QNN NPU pipeline context
- Qualcomm AIMET quantization docs: https://quic.github.io/aimet-pages/releases/latest/tutorials/quantsim.html (per-tensor vs per-channel encoding tradeoffs)
