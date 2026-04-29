# Plan: Day 6.1 — QNN NPU Pipeline (Model Conversion → Quantization → Deploy)

**Status:** 🔲 TODO — Not started
**Version:** v2 (supersedes v1; flips FP16-first → INT8-only path)
**Created:** 2026-04-27
**Bridges:** Day 6 (sequential GPU combined pipeline, §3/§6 FAIL) → Day 7 (shot FSM)
**Gap addressed:** Gap 1 — QNN NPU delegate for YOLO inference on Hexagon 780 HTP (SM8450, Samsung S22+)

---

## Why this plan changed (v1 → v2)

v1 proposed a "FP16 fast path first; INT8 as upgrade" strategy using the existing Ultralytics FP16 TFLite asset with `setHtpPrecision(HTP_PRECISION_FP16)`. **This contradicts the QNN HTP backend's hard constraints** documented in three places already aligned on this project:

1. **`Claude_nnapi_qnn_report` §7.1**: *"The HTP backend only supports quantised models. Float32 models must be quantised to INT8 (or INT16 for select ops) before compilation. A float model submitted to a QNN compile job will either fail or be routed away from the NPU. This is why the current CourtVision FP16 Ultralytics export is a Phase 0 artifact — it is intentionally incompatible with the NPU path until calibration is performed."*
2. **`Claude-qualcomm-demo-gap-analysis.md` Gap 1**: Reference to QAI-Hub-compiled INT8 model, not stock FP16.
3. **Qualcomm HuggingFace YOLO performance tables**: TFLite runtime offers only `float` and `w8a8` precision tiers on HTP; FP16-as-NPU-target is not a published path.

The Qualcomm demo `TFLiteHelpers.java` that v1 cited as precedent runs **QAI-Hub-compiled INT8/QDQ models**. Its `setHtpPrecision(FP16)` flag controls internal accumulator precision *within an already-quantized graph* — it is not an FP16-graph executor switch.

**v2 reorders the work**: Step 0 (INT8 model production) is now **blocking** rather than deferred. All Android wiring (Steps 1–9 of v1) is preserved — that work was correct. Only the model-side strategy and verification gates change.

---

## Context

Day 6 10-min soak (build `762b968`, SM-S906U1, 2026-04-22):

| Metric | Result | Gate | Verdict |
|--------|--------|------|---------|
| YOLO inference p50 | 53 ms | — | Primary bottleneck |
| `frame_total_ms` p95 | 129 ms | ≤ 100 ms | **FAIL** |
| FPS median | 8 fps | ≥ 20 fps | **FAIL** |
| Thermal ceiling | MODERATE at t=68 s → sustained throttle | — | Root cause |

Pose is healthy (p95 = 15 ms). Moving YOLO from GPU (~53 ms p50) to Hexagon HTP is the primary lever. ADR-005 §Deferred names this **Config B — highest-priority Phase 2 optimization.** Published QAI Hub w8a8 benchmarks for YOLOv8n / YOLO11n on SM8450-class hardware show 1–5 ms YOLO inference on Hexagon HTP — an order of magnitude headroom.

**ADR-005:** [`docs/decisions/005-sequential-gpu-inference-pipeline.md`](../decisions/005-sequential-gpu-inference-pipeline.md)
**Gap analysis:** [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md)
**NNAPI/QNN report:** [`research-reports/Claude_nnapi_qnn_reportv2.pdf`](../../research-reports/Claude_nnapi_qnn_reportv2.pdf)
**Day 6 benchmark:** [`docs/plans/day6-combined-pipeline.md`](./day6-combined-pipeline.md) "Step 7 Results"

---

## Branch

`perf/day6-x-post-day6-perf-follow-up` (shared workstream branch from `main` after Day 6 PR merges; execute Day 6.2 → Day 6.1 → Day 6.3 in order)

---

## Prerequisites

- Day 6 PR merged to `main` ✅
- S22+ (SM-S906U1) available for 10-min benchmark run
- Python env with `ultralytics` installed
- Access to YOLO training `dataset.yaml` and a 500–1000-image calibration subset (per cookbook §8) — OR a Qualcomm AI Hub account (`https://app.aihub.qualcomm.com`, free tier sufficient)
- Held-out validation split for INT8 mAP regression check

---

## Design Decisions

### DD-1 — INT8 QDQ TFLite is the only Day 6.1 path

The QNN HTP backend requires quantized graphs. Two production routes:

- **Route A (preferred for Day 6.1) — Ultralytics INT8 TFLite export.** Uses your existing training pipeline, no external account needed. Output is a w8a8 QDQ TFLite consumable by `qnn-litert-delegate`. On-device Hexagon compilation happens at first launch (1–3 s stall, cached thereafter).
- **Route B (production-grade) — QAI Hub w8a8 compile job.** Higher-quality calibration tooling, server-side, produces TFLite or DLC artifact. Use for production cut; Route A is sufficient for Day 6.1 validation.

**FP16 path is explicitly out of scope.** It cannot pass §5 gate because HTP will not execute an FP16 graph. The existing FP16 asset remains the GPU-mode default; it is not deleted.

### DD-2 — Package: `com.qualcomm.qti:qnn-litert-delegate:2.40.0`

ADR-005 referred to the older name `qnn-tflite-delegate`. The 2025 Qualcomm AI Hub demo uses `qnn-litert-delegate:2.40.0` (LiteRT = rebranded TFLite) — the actively maintained package. Confirmed: resolves from `mavenCentral()` — **no changes to `settings.gradle.kts` required.**

### DD-3 — Fallback cascade: QNN_NPU → GPU → CPU

If QNN init fails or the delegate reports excessive op fallback, `switchInterpreter()` falls back to GPU (which uses the FP16 asset); if GPU fails, to CPU. Cascade is internal — the external caller (ViewModel) sets `QNN_NPU` and degradation is transparent. `_lastError` records each fallback for debug display.

**New in v2:** the QNN branch loads a *different model asset* than the GPU branch (INT8 vs FP16). `FrameProcessor` selects the asset based on `InferenceMode` at switch time.

### DD-4 — Runtime API guard: `Build.VERSION.SDK_INT >= Build.VERSION_CODES.S`

`minSdk = 26`; Hexagon 780 HTP (SM8450) ships on API 31 devices. Guard QNN init with the API 31 check; fall back to GPU below it. `minSdk` bump to 31 is a product decision — deferred.

### DD-5 — Inject `nativeLibraryDir` + `modelCacheDir` into `FrameProcessor`

`FrameProcessor` has no `Context`. Add `nativeLibraryDir: String? = null` and `modelCacheDir: String? = null` as nullable constructor params. `null` disables caching (recompiles on every cold start but fully functional). `CameraViewModel` passes `application.applicationInfo.nativeLibraryDir` and `application.cacheDir.absolutePath`. Preserves the pipeline/Android boundary from ADR-005.

### DD-6 — `QnnProbeResult` follows `GpuProbeResult` typed pattern

`NnApiDelegateProbe` returns a raw `String` (an existing inconsistency in the codebase). `QnnProbeResult` follows the typed `GpuProbeResult` + `GpuStatus` enum pattern — cleaner for UI binding and test assertions.

### DD-7 — Calibration data sourcing

Existing `calibration_image_sample_data_20x128x128x3_float32.npy` in assets is **128×128 — wrong shape for YOLO's 640×640 input. Cannot be used.** Path per route:
- **Route A**: Ultralytics export reads training `dataset.yaml` automatically; uses `fraction=0.2` for first-pass (≈ 100 images per cookbook §8 quick-iteration recommendation), expanded to 500–1000 for production cut.
- **Route B**: QAI Hub handles calibration server-side from a representative subset uploaded as `calibration_data`.

### DD-8 — Hard fail-fast probe before 10-min soak

The QNN delegate can silently route ops to its CPU sub-backend if quantization is incomplete or operator support is missing. Before the soak, run a one-frame smoke test that inspects partition logs (logcat tag `QnnDelegate`). If > 5% of nodes fall back, abort and investigate. Cheap; saves a 10-minute wasted run.

### DD-9 — Asset path routing per `InferenceMode`

`FrameProcessor` gains a `modelAssetForMode(mode)` selector:
- `CPU`, `GPU`, `NNAPI` → `yolo11n_640_5-class_04-01-2026_float16.tflite` (existing)
- `QNN_NPU` → `yolo11n_640_5-class_<DATE>_int8.tflite` (new, produced in Step 0)

`modelBufferProvider` becomes mode-aware via a thin closure in `CameraViewModel.loadModelBuffer(mode)`. Both assets ship in the APK; size budget impact: +3 MB (INT8 ≈ 3 MB vs FP16 ≈ 6 MB; both kept).

---

## Files

### New files
- `app/src/main/java/com/courtvision/spike/pipeline/QnnDelegateProbe.kt`
- `app/src/main/assets/yolo11n_640_5-class_<EXPORT_DATE>_int8.tflite` (produced offline in Step 0)

### Modify
| File | Change |
|------|--------|
| `app/build.gradle.kts` | Add `qnn-runtime:2.40.0`, `qnn-litert-delegate:2.40.0` ⚠ new deps |
| `app/src/main/AndroidManifest.xml` | Add `libcdsprpc.so` optional native lib declaration |
| `app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt` | Add `InferenceMode.QNN_NPU`, `QnnStatus` enum, `QnnProbeResult` data class |
| `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` | Add `qnnDelegate` field + close; `QNN_NPU` branch in `switchInterpreter()`; `nativeLibraryDir`/`modelCacheDir` constructor params; mode-aware model asset selection |
| `app/src/main/java/com/courtvision/spike/camera/CameraUiState.kt` | Add `qnnProbeResult: QnnProbeResult`, `qnnAvailable: Boolean` |
| `app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt` | Add `QnnDelegateProbe.probe()` in `init`; pass new params to `FrameProcessor`; `qnnProbeResult` in `TestOverrides`; update `gpuModeForSummary` `when`; mode-aware `loadModelBuffer` |
| `app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt` | Add "NPU" mode button; QNN probe status line in debug overlay |
| `app/src/test/java/com/courtvision/spike/pipeline/FrameProcessorTest.kt` | QNN cascade fallback tests |
| `CONTEXT.md` | Progress line + Day 6.1 benchmark results section after soak |
| `docs/decisions/005-sequential-gpu-inference-pipeline.md` | Mark Config B "In progress (Day 6.1)" in §Deferred; record INT8-only decision |

---

## Step-by-Step Implementation

---

### Step 0 — Offline INT8 model production (REQUIRED, blocking)

**Status:** 🔲 TODO

This step is no longer optional. The Android wiring in Steps 1–9 is meaningless without an HTP-compatible model asset.

#### Step 0a — Produce INT8 QDQ TFLite (Route A, default)

```bash
pip install ultralytics
yolo export model=yolo11n_640_5-class_04-01-2026.pt \
    format=tflite \
    int8=True \
    data=<path/to/dataset.yaml> \
    fraction=0.2 \
    imgsz=640 \
    nms=False
# Output: yolo11n_640_5-class_04-01-2026_saved_model/yolo11n_640_5-class_04-01-2026_int8.tflite
```

`fraction=0.2` ≈ 100-image quick-iteration calibration per cookbook §8. If accuracy gate (Step 0c) fails, expand to `fraction=1.0` with a curated 500–1000-image subset.

`nms=False` keeps NMS in Kotlin code, matching current pipeline. Critical: do not enable `nms=True` here.

Place the output at:
```
app/src/main/assets/yolo11n_640_5-class_<DATE>_int8.tflite
```

#### Step 0b — Verify graph shape and quantization (sanity check)

```python
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="yolo11n_640_5-class_<DATE>_int8.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("Input dtype:", inp["dtype"], "shape:", inp["shape"])   # expect int8 or uint8, [1,640,640,3]
print("Output dtype:", out["dtype"], "shape:", out["shape"])  # expect int8 with quant params
print("Input quant:", inp["quantization_parameters"])
print("Output quant:", out["quantization_parameters"])
```

Pass condition: input dtype is `int8` or `uint8`, both input and output have non-empty `scale` and `zero_point` arrays. If dtype is float32, the export silently fell back — re-run with explicit `int8=True` and verify the calibration set is reachable.

#### Step 0c — INT8 accuracy gate (mAP regression check)

Run mAP eval on held-out validation split:
```bash
yolo val model=yolo11n_640_5-class_<DATE>_int8.tflite data=<dataset.yaml>
```

**Pass:** mAP50 ≥ 0.92 (≥ 98% of FP16 baseline 0.94).
**Borderline (0.88–0.92):** expand calibration to `fraction=1.0` and re-export.
**Fail (< 0.88):** switch to Route B (QAI Hub) which has better calibration tooling, or escalate to a curated calibration subset of 500–1000 images stratified by class.

Block Steps 1–11 on this gate. Burning 10 min of soak time on a model that loses 10% mAP is wasted work.

#### Step 0d — Route B (QAI Hub) — fallback if Route A fails accuracy gate

1. Export ONNX: `yolo export model=yolo11n_640_5-class_04-01-2026.pt format=onnx imgsz=640 nms=False`
2. `https://app.aihub.qualcomm.com` → submit ONNX
3. Device: SM8450 · Precision: w8a8 PTQ · Runtime: TFLite · `input_specs=dict(image=(1,3,640,640))`
4. Upload calibration data (500–1000 representative images preprocessed to 640×640 NCHW float32)
5. Download compiled `.tflite` → place in `app/src/main/assets/`
6. Re-run Step 0c accuracy gate

---

### Step 1 — `build.gradle.kts`: add QNN dependencies

**Status:** 🔲 TODO

Add to `dependencies` block. **⚠ New dependencies — flagged per CLAUDE.md:**

```kotlin
implementation("com.qualcomm.qti:qnn-runtime:2.40.0")
implementation("com.qualcomm.qti:qnn-litert-delegate:2.40.0")
```

**NDK note:** if the build fails with an NDK-related error after adding the QNN AAR (which ships native `.so` files), add inside the `android {}` block:
```kotlin
ndkVersion = "27.3.13750724"
```

---

### Step 2 — `AndroidManifest.xml`: declare `libcdsprpc.so`

**Status:** 🔲 TODO

Add inside `<application>`, adjacent to the existing `libOpenCL.so` declaration:
```xml
<uses-native-library
    android:name="libcdsprpc.so"
    android:required="false" />
```

`required="false"` ensures the app installs on all devices; the delegate fallback handles non-Qualcomm hardware.

---

### Step 3 — `FrameContracts.kt`: `QNN_NPU` mode + probe types

**Status:** 🔲 TODO

**3a — Extend `InferenceMode`:**
```kotlin
enum class InferenceMode {
    CPU,
    GPU,
    NNAPI,
    QNN_NPU    // Hexagon HTP via QNN LiteRT delegate (INT8 only)
}
```

**3b — Add probe types** (mirrors `GpuStatus` / `GpuProbeResult` exactly):
```kotlin
enum class QnnStatus {
    QNN_SUPPORTED,
    QNN_UNSUPPORTED,
    QNN_INIT_FAILED
}

data class QnnProbeResult(
    val status: QnnStatus = QnnStatus.QNN_UNSUPPORTED,
    val htpFp16Supported: Boolean = false,        // accumulator precision capability
    val htpQuantizedSupported: Boolean = false,   // ← required for our INT8 path
    val reason: String? = null,
    val deviceModel: String = "",
    val apiLevel: Int = 0
)
```

**Note:** `htpFp16Supported` is reported for completeness (matches QNN SDK capability enum) but is **not** the gate for `qnnAvailable` — `htpQuantizedSupported` is. The CameraViewModel logic in Step 7 treats `qnnAvailable = htpQuantizedSupported`.

**Exhaustiveness check:** `InferenceMode.QNN_NPU` is used in `when` statements in `switchInterpreter()`, `appendPerFramePerfRow()` `gpuModeForSummary` in `CameraViewModel`, and `InferenceModeSelector` in `CameraScreen.kt`. Add `QNN_NPU` branches to each before compile.

---

### Step 4 — New `QnnDelegateProbe.kt`

**Status:** 🔲 TODO

```kotlin
package com.courtvision.spike.pipeline

import android.os.Build
import com.qualcomm.qti.QnnDelegate

object QnnDelegateProbe {
    fun probe(): QnnProbeResult {
        val model = Build.MODEL ?: "unknown"
        val apiLevel = Build.VERSION.SDK_INT

        if (apiLevel < Build.VERSION_CODES.S) {
            return QnnProbeResult(
                status = QnnStatus.QNN_UNSUPPORTED,
                reason = "QNN requires API 31+ (S); device is API $apiLevel",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }

        return try {
            val htpFp16 = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
            val htpQuant = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)
            QnnProbeResult(
                // Quantized capability is the gate for our INT8 path; FP16 capability is informational
                status = if (htpQuant) QnnStatus.QNN_SUPPORTED else QnnStatus.QNN_UNSUPPORTED,
                htpFp16Supported = htpFp16,
                htpQuantizedSupported = htpQuant,
                reason = if (!htpQuant) "HTP quantized runtime not available on this device" else null,
                deviceModel = model,
                apiLevel = apiLevel
            )
        } catch (error: Throwable) {
            QnnProbeResult(
                status = QnnStatus.QNN_INIT_FAILED,
                reason = "${error::class.java.simpleName}: ${error.message ?: "unknown error"}",
                deviceModel = model,
                apiLevel = apiLevel
            )
        }
    }
}
```

`QnnDelegate.checkCapability()` is static — safe on any thread including Main. `Throwable` catch handles absent `libQnnTFLiteDelegate.so` on non-Qualcomm devices and `UnsatisfiedLinkError` from JVM unit tests.

---

### Step 5 — `FrameProcessor.kt`: QNN branch in `switchInterpreter()`

**Status:** 🔲 TODO

#### 5a — Constructor + `qnnDelegate` field + mode-aware model loading

```kotlin
class FrameProcessor(
    ...
    private val nativeLibraryDir: String? = null,
    private val modelCacheDir: String? = null,
    ...
)

private var qnnDelegate: QnnDelegate? = null
```

The `modelBufferProvider` signature changes from `() -> ByteBuffer` to `(InferenceMode) -> ByteBuffer` so that the QNN_NPU branch can load the INT8 asset while other branches load FP16. Update the type alias and the corresponding `CameraViewModel.loadModelBuffer` (Step 7).

Add to `closeInterpreterResources()`:
```kotlin
qnnDelegate?.close()
qnnDelegate = null
```

#### 5b — `InferenceMode.QNN_NPU` branch

```kotlin
InferenceMode.QNN_NPU -> {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) {
        _lastError.value = "QNN_NPU requires API 31+; falling back to GPU"
        return switchInterpreter(InferenceMode.GPU)
    }
    val localQnnDelegate = tryCreateQnnDelegate()
    if (localQnnDelegate == null) {
        return switchInterpreter(InferenceMode.GPU)
    }
    options.addDelegate(localQnnDelegate)
    qnnDelegate = localQnnDelegate
    // GPU as sub-delegate for any quantized ops QNN's HTP backend cannot handle.
    // CPU/XNNPACK is the implicit fallback for ops neither HTP nor GPU support.
    val compat = CompatibilityList()
    if (compat.isDelegateSupportedOnThisDevice) {
        localGpuDelegate = GpuDelegate(compat.bestOptionsForThisDevice)
            .also { options.addDelegate(it) }
    }
}
```

#### 5c — Private helpers

```kotlin
private fun tryCreateQnnDelegate(): QnnDelegate? {
    return try {
        val opts = QnnDelegate.Options()
        if (nativeLibraryDir != null) opts.setSkelLibraryDir(nativeLibraryDir)
        opts.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN)
        if (modelCacheDir != null) {
            opts.setCacheDir(modelCacheDir)
            opts.setModelToken(computeModelMd5())
        }
        opts.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
        opts.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
        opts.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
        // Note: HTP_PRECISION_FP16 controls accumulator precision within the
        // already-quantized graph; it does NOT make HTP execute FP16 graphs.
        // The graph itself must be INT8 QDQ (produced in Step 0).
        opts.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
        QnnDelegate(opts)
    } catch (error: Throwable) {
        _lastError.value = "QNN delegate init failed: ${error.message ?: "unknown error"}"
        null
    }
}

private fun computeModelMd5(): String {
    return try {
        val buffer = modelBufferProvider?.invoke(InferenceMode.QNN_NPU) ?: return "model_unknown"
        val bytes = ByteArray(buffer.remaining())
        buffer.duplicate().get(bytes)  // duplicate to avoid consuming the original
        java.security.MessageDigest.getInstance("MD5")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }
    } catch (_: Throwable) {
        "model_unknown"
    }
}
```

---

### Step 6 — `CameraUiState.kt`: QNN probe fields

**Status:** 🔲 TODO

```kotlin
// Add alongside gpuProbeResult:
val qnnProbeResult: QnnProbeResult = QnnProbeResult(),
val qnnAvailable: Boolean = false,
```

---

### Step 7 — `CameraViewModel.kt`: probe wiring + `FrameProcessor` params

**Status:** 🔲 TODO

**`init` block** (after existing GPU/NNAPI probe calls):
```kotlin
val qnnResult = overrides?.qnnProbeResult ?: QnnDelegateProbe.probe()
_uiState.update {
    it.copy(
        qnnProbeResult = qnnResult,
        // Gated on quantized capability — our path requires INT8 graphs
        qnnAvailable = qnnResult.status == QnnStatus.QNN_SUPPORTED &&
                       qnnResult.htpQuantizedSupported
    )
}
```

**Mode-aware model loading:**
```kotlin
private fun loadModelBuffer(mode: InferenceMode): ByteBuffer {
    val assetName = when (mode) {
        InferenceMode.QNN_NPU -> "yolo11n_640_5-class_<DATE>_int8.tflite"
        else                  -> "yolo11n_640_5-class_04-01-2026_float16.tflite"
    }
    return loadAsset(assetName)
}
```

**`FrameProcessor` construction** — add two params + mode-aware model loader:
```kotlin
FrameProcessor(
    scope = viewModelScope,
    modelBufferProvider = ::loadModelBuffer,    // now takes InferenceMode
    poseModelBufferProvider = ::loadPoseModelBuffer,
    nativeLibraryDir = application.applicationInfo.nativeLibraryDir,
    modelCacheDir = application.cacheDir.absolutePath,
    thermalStatusProvider = { ... },
    perFrameLogger = perFramePerfLogger,
    perFrameMode = PER_FRAME_MODE,
    perFrameDevice = Build.MODEL
)
```

**`TestOverrides`** — add:
```kotlin
val qnnProbeResult: QnnProbeResult? = null,
```

**`gpuModeForSummary` in `startDay5PoseIsolatedValidation()`** — update:
```kotlin
val gpuModeForSummary = when (selectedMode) {
    InferenceMode.CPU      -> "CPU"
    InferenceMode.GPU      -> "GPU"
    InferenceMode.NNAPI    -> "NNAPI"
    InferenceMode.QNN_NPU  -> "QNN_NPU"
}
```

---

### Step 8 — `CameraScreen.kt`: NPU mode button

**Status:** 🔲 TODO

Add `qnnAvailable: Boolean` param to `InferenceModeSelector`. Add NPU button:
```kotlin
ModeButton(
    label = "NPU",
    selected = selectedMode == InferenceMode.QNN_NPU,
    enabled = qnnAvailable
) { onModeSelect(InferenceMode.QNN_NPU) }
```

Add QNN status line to debug overlay:
```kotlin
Text("QNN: ${uiState.qnnProbeResult.status} HTP_QUANT=${uiState.qnnProbeResult.htpQuantizedSupported}")
```

---

### Step 9 — Tests

**Status:** 🔲 TODO

Add to `FrameProcessorTest.kt`:

- **`switchInterpreter_qnnNpu_fallsBackToGpu_onApiBelow31()`** — set `Build.VERSION.SDK_INT = 29` via Robolectric shadow; assert `delegateMode == InferenceMode.GPU`; assert `lastError` contains "API 31+".
- **`switchInterpreter_qnnNpu_fallsBackToGpu_whenQnnDelegateFails()`** — on Robolectric JVM, `QnnDelegate()` throws `UnsatisfiedLinkError` (no `.so`). Test must catch `Throwable` (not `Exception`) to exercise the cascade. Assert fallback to GPU; assert `lastError` non-null.
- **`switchInterpreter_qnnNpu_loadsInt8Model_notFp16()`** — inject a fake `modelBufferProvider` that records the `InferenceMode` arg; switch to `QNN_NPU`; assert provider was called with `QNN_NPU` (not `GPU`).
- **`closeInterpreterResources_closesQnnDelegate()`** — inject fake delegate via `@VisibleForTesting` seam; call `resetInterpreter()`; assert `close()` called.
- **`CameraViewModelTest` — `qnnProbeSupportedAndQuantized_setsQnnAvailableTrue()`** — `TestOverrides(qnnProbeResult = QnnProbeResult(status = QnnStatus.QNN_SUPPORTED, htpQuantizedSupported = true))`; assert `uiState.qnnAvailable == true`.
- **`CameraViewModelTest` — `qnnProbeSupportedButFp16Only_setsQnnAvailableFalse()`** — `TestOverrides(qnnProbeResult = QnnProbeResult(status = QnnStatus.QNN_SUPPORTED, htpFp16Supported = true, htpQuantizedSupported = false))`; assert `uiState.qnnAvailable == false`. Guards against the v1 misconception.
- **`appendPerFramePerfRow_gpuMode_isQNN_NPU()`** — force `currentMode = QNN_NPU`; assert `gpu_mode` column = `"QNN_NPU"` in captured CSV row.

---

### Step 10 — Fail-fast smoke probe (DD-8)

**Status:** 🔲 TODO

Before running the 10-min soak, run a one-frame inspection to confirm HTP actually owns the YOLO graph.

1. Install APK, launch, switch to NPU mode
2. Capture logcat:
   ```bash
   adb logcat -s QnnDelegate:V QNN:V
   ```
3. Run for ≈ 30 seconds (covers cold compile + first 100 frames)
4. Parse logcat for delegate partition output. Expected pattern (exact string varies by SDK version):
   ```
   QnnDelegate: <N> nodes delegated to HTP, <M> nodes on fallback (CPU/GPU)
   ```

**Pass:** `M / (N + M) < 0.05` (≥ 95% of nodes on HTP).
**Borderline (0.05–0.20):** acceptable for Day 6.1 but capture the fallback op list for a follow-up; common offenders are NMS-adjacent ops or unsupported activations, addressable by re-exporting with op-substitution flags.
**Fail (> 0.20):** abort soak. Most likely causes:
- Step 0b dtype check missed a float32 graph
- Calibration was incomplete (some ops emitted as float fallback)
- Op not supported on HTP for this Hexagon generation

If failed, re-export Step 0 with explicit verification rather than running the soak with a half-CPU graph.

---

### Step 11 — 10-min soak benchmark (user-run)

**Status:** 🔲 TODO

**Pre-conditions:** Steps 0–10 complete, Step 10 smoke probe PASS, S22+ pre-cooled (≥ 30 min rest).

1. Launch app → select **NPU** mode
2. **First NPU switch:** expect 1–3 s stall (Hexagon graph compilation, one-time). Subsequent launches are fast (cache hit).
3. Verify overlay renders (detection boxes + pose landmarks) — confirms QNN mode did not silently degrade to CPU
4. Verify `lastError` display is null or informational
5. Run 10 minutes portrait, then pull:
   ```
   adb pull /sdcard/Android/data/com.courtvision.spike/files/benchmarks/phase0/combined_pipeline/
   ```
6. Save to `benchmarks/phase0/combined_pipeline/run_YYYYMMDD_HHMMSS_qnn_npu/`

**Gate analysis:**
```python
import pandas as pd
df = pd.read_csv("phase0-perframe-YYYYMMDD-HHMMSS.csv")
qnn = df[df["gpu_mode"] == "QNN_NPU"]
print("YOLO p50:", qnn["yolo_inference_ms"].quantile(0.50))
print("YOLO p95:", qnn["yolo_inference_ms"].quantile(0.95))
print("frame_total p95:", qnn["frame_total_ms"].quantile(0.95))
print("FPS median:", qnn["fps_1s_window"].median())
print("Thermal MODERATE onset s:", (qnn[qnn["thermal_status"]=="MODERATE"]["timestampMs"].min()
      - qnn["timestampMs"].min()) / 1000)
```

---

### Step 12 — `CONTEXT.md` + `ADR-005` update

**Status:** 🔲 TODO

After benchmark:
- Update `CONTEXT.md` "Progress" line; add "Day 6.1 Benchmark Results" section
- Update `ADR-005` §Deferred: mark Config B "In progress (Day 6.1)"; record **INT8-only** decision and the FP16-on-HTP misconception flagged-and-corrected

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §0a INT8 export | `int8.tflite` produced; size ≈ 3 MB | Verify `dataset.yaml` accessible; check Ultralytics version |
| §0b Graph dtype | input/output dtype int8/uint8 with non-empty quant params | Re-export; calibration likely silently skipped |
| §0c Accuracy | mAP50 ≥ 0.92 | Expand `fraction` to 1.0; if still fails, switch to Route B (QAI Hub) |
| §1 Build | `./gradlew :app:assembleDebug` succeeds | Fix NDK/dep issue; add `ndkVersion` if needed |
| §2 Unit tests | All green incl. new INT8 model-load test | Fix before APK install |
| §3 QNN probe on S22+ | Overlay shows `QNN: QNN_SUPPORTED HTP_QUANT=true` | `adb logcat \| grep QNN`; verify `libcdsprpc.so` declared |
| §4 NPU mode switch | No fallback error; `gpu_mode=QNN_NPU` throughout CSV | GPU fallback still functional; capture QNN logcat tag |
| §5 Smoke probe (DD-8) | ≥ 95% of YOLO nodes on HTP | Re-do Step 0 — graph likely contains float fallback ops |
| §6 YOLO p50 | ≤ 10 ms (revised from v1's 20 ms — INT8 on HTP target) | 10–25 ms → check thermal throttle / op partial fallback; > 25 ms → investigate |
| §7 Combined p95 | `frame_total_ms` p95 ≤ 100 ms | Check if pose is new bottleneck (`pose_inference_ms` p95) |
| §8 FPS sustained | ≥ 20 fps median over 10 min | Note thermal MODERATE onset time vs Day 6 baseline t=68 s — should be later given lower YOLO load |
| §9 GPU mode regression | GPU p50 ≈ 53 ms ± 5 ms after switching back | Day 6.1 makes no changes to the GPU branch — should be trivially true |
| §10 Cache warmup | 2nd cold start NPU switch < 500 ms | Verify `modelCacheDir` non-null; check logcat for QNN cache hit |

**Note on §6 target revision:** v1 set 20 ms because that was the FP16-on-HTP expectation (which doesn't exist as a real path). With INT8 on HTP the published Qualcomm benchmarks for YOLO11n show 1–5 ms on SM8450-class hardware, so 10 ms is conservative and leaves headroom for app-side overhead.

---

## Deferred Items

- **QAI Hub w8a8 production cut (Route B)** — only invoked if Route A accuracy gate fails; otherwise deferred to production hardening
- **QNN for pose (Config C)** — pose p95 = 15 ms; not a bottleneck; Phase 2
- **Other chipsets** — GPU fallback cascade handles non-Qualcomm devices automatically
- **MediaTek Dimensity NPU path** — separate INT8 TFLite + Neuron delegate; out of scope
- **`minSdk` bump to 31** — product decision; PRD review required
- **`InferenceMode.NNAPI` removal** — addressed in Day 6.3
- **QNN context-binary ahead-of-time compile** — eliminates 1–3s first-launch stall but device-specific; defer until production hardening

---

## References

- [`docs/decisions/005-sequential-gpu-inference-pipeline.md`](../decisions/005-sequential-gpu-inference-pipeline.md) — Config B, QNN deferred section
- [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md) — Gap 1 with line citations
- [`research-reports/Claude_nnapi_qnn_reportv2.pdf`](../../research-reports/Claude_nnapi_qnn_reportv2.pdf) — §7.1 INT8 requirement, §8 calibration scale
- [`research-reports/ChatGPT-deep-research-report_NPU-model-deployment-cookbook.md`](../../research-reports/ChatGPT-deep-research-report_NPU-model-deployment-cookbook.md) — Ultralytics INT8 export recipe, fallback strategies
- [`docs/plans/day6-combined-pipeline.md`](./day6-combined-pipeline.md) — Day 6 baseline
- Qualcomm demo wiring reference: `D:\ai-hub-apps\apps\_shared\android\tflite_helpers\TFLiteHelpers.java` lines 247–333
- Qualcomm demo dependency versions: `D:\ai-hub-apps\apps\object_detection_android\build.gradle` lines 46–57
- Qualcomm YOLO11 perf table: `https://huggingface.co/qualcomm/YOLOv11-Detection`
- QAI Hub: `https://app.aihub.qualcomm.com`

---

## Changelog

**v2 (2026-04-27)** — Flipped FP16-first → INT8-only path. Step 0 promoted to blocking. Added DD-8 fail-fast smoke probe, DD-9 mode-aware asset routing. Tightened §6 YOLO p50 target from 20 ms → 10 ms based on published HTP w8a8 benchmarks. Added test cases guarding against the FP16-HTP misconception. Probe gate now keys on `htpQuantizedSupported`, not `htpFp16Supported`.

**v1 (2026-04-27, superseded)** — Initial plan with FP16 fast path. Conflicted with QNN HTP backend constraints documented in `Claude_nnapi_qnn_report` §7.1 and Qualcomm published benchmark tables. See "Why this plan changed" at top.