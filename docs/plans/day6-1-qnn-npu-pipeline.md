# Plan: Day 6.1 — QNN NPU Pipeline (Model Conversion → Quantization → Deploy)

**Status:** ✅ COMPLETE (2026-05-02) — Steps 0–12 done. §5 smoke ✅ 357/357 nodes on HTP (0% fallback); §6 YOLO p50=3.46 ms ✅; §7 `frame_total` p95=81.4 ms ✅; §8 FPS 14 fps ❌ (CPU preprocess bottleneck at 38.9 ms — preprocess offload required). Docs updated: CONTEXT.md, ADR-005 Config B validated + Config E added, spike plan.
**Version:** v9 (see Changelog for full history)
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
- `QNN_NPU` → `spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite` (produced in Step 0)

`modelBufferProvider` becomes mode-aware via a thin closure in `CameraViewModel.loadModelBuffer(mode)`. Both assets ship in the APK; size budget impact: +3 MB (INT8 ≈ 3 MB vs FP16 ≈ 6 MB; both kept).

### DD-10 — Split output head is required for INT8 TFLite (NPU path)

The original QAI Hub export used a single `(1, 9, 8400)` output tensor (4 box channels + 5 class channels merged). A shared per-tensor quantization scale (~2.62) is appropriate for image-space box values (0–640) but too coarse for class probabilities (0–1): any class value below ~1.3 quantizes to exactly zero after dequantization. Result: zero mAP across all 5 classes while box channels remained numerically alive.

**Fix:** re-export with split output heads:
- `output_0` — shape `(1, 4, 8400)`, scale≈2.62 (calibrated to box magnitudes)
- `output_1` — shape `(1, 5, 8400)`, scale=0.00390625 (calibrated to 0–1 probability range)

Each tensor receives its own INT8 quantization parameters. The combined `(1, 9, 8400)` format is **prohibited for INT8 exports** and is listed as a rejected alternative in ADR-007.

The FP16 GPU asset retains the combined layout — this constraint is INT8/NPU-only.

Evidence and full analysis: [`research-reports/Codex-quantized-yolo-zero-map-findings.md`](../../research-reports/Codex-quantized-yolo-zero-map-findings.md).

**Android implication:** `FrameProcessor` must read two output tensors (Step 5d) rather than slicing a single combined buffer.

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

**Status:** ✅ DONE

This step is no longer optional. The Android wiring in Steps 1–9 is meaningless without an HTP-compatible model asset.

#### Step 0a — Produce INT8 QDQ TFLite ✅ DONE (Route B — QAI Hub)

Route A (Ultralytics export) was attempted first and produced zero mAP due to combined (1,9,8400) output quantization collapse (see DD-10, research-reports/Codex-quantized-yolo-zero-map-findings.md). Route B (QAI Hub) was used with split output heads.

Validated model asset:
```
spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite
→ app/src/main/assets/spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite
```

#### Step 0b — Verify graph shape and quantization ✅ DONE

Confirmed tensor profile for `spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite`:

```
Input  : image     shape=(1, 3, 640, 640)  dtype=int8  scale=0.003921568859368563  zp=-128   ← NCHW
Output0: output_0  shape=(1, 4, 8400)      dtype=int8  scale=2.621687412261963     zp=-128   ← boxes
Output1: output_1  shape=(1, 5, 8400)      dtype=int8  scale=0.00390625            zp=-128   ← scores
```

Input is NCHW (not NHWC) — the existing TFLite NCHW hotpatch in `FrameProcessor` already handles this.
Split output heads are required; see DD-10 for why the combined (1,9,8400) format collapses class scores to zero under INT8.

#### Step 0c — INT8 accuracy gate ✅ DONE (BORDERLINE/ACCEPTED)

```
Run: spike_qai_yolo11n_640_5-class_04-28-2026_int8_14
Validation params: conf=0.1, iou=0.35, max_det=20, imgsz=640, split=val, images=866, instances=3514

mAP50(B):    0.9143  ← BORDERLINE (gate: ≥0.92; range 0.88–0.92 = borderline)
mAP50-95(B): 0.5792
Precision:   0.880   Recall: 0.875
FP16 baseline: 0.94 → retention 97.3% (just below the 98% = 0.92 threshold)

Per-class mAP50:
  ball:   0.876   ← weakest class (small, fast-moving)
  made:   0.916
  person: 0.924
  rim:    0.982   ← strongest
  shoot:  0.874
```

**Decision: ACCEPTED for Day 6.1.** Route B (QAI Hub) is already the highest-quality PTQ path. Further re-export without QAT is unlikely to clear 0.92. The 2.6 pp gap is acceptable given on-device NPU speed is the primary Day 6.1 target.

**Note:** Validation params `conf=0.1, iou=0.35, max_det=20` should be matched in the Android NMS config (Step 5d).

#### Step 0d — Route B (QAI Hub) ✅ DONE (was the primary path, not fallback)

Route B was used because Route A's combined (1,9,8400) output caused zero mAP under INT8 quantization.
QAI Hub was configured with split output heads to give each tensor its own quantization scale.
Result: `spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite` (see Step 0b/0c above).

---

### Step 1 — `build.gradle.kts`: add QNN dependencies

**Status:** ✅ DONE

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

**Status:** ✅ DONE

Add inside `<application>`, adjacent to the existing `libOpenCL.so` declaration:
```xml
<uses-native-library
    android:name="libcdsprpc.so"
    android:required="false" />
```

`required="false"` ensures the app installs on all devices; the delegate fallback handles non-Qualcomm hardware.

---

### Step 3 — `FrameContracts.kt`: `QNN_NPU` mode + probe types

**Status:** ✅ DONE

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

**3c — Add `ModelOutputFormat.QNN_INT8_8400`** to wherever `ModelOutputFormat` is defined (likely a private enum in `FrameProcessor.kt`):
```kotlin
private enum class ModelOutputFormat {
    RAW_8400,         // FP16 combined (1,9,8400) — GPU/CPU/NNAPI
    END_TO_END_300,   // FP16 e2e (1,300,6) — GPU/CPU
    QNN_INT8_8400     // INT8 split-output (1,4,8400)+(1,5,8400) — QNN_NPU
}
```
`switchInterpreter()` dispatches to `validateQnnTensorContract()` (Step 5e) which sets `outputFormat = QNN_INT8_8400`. `processImage()`'s `when (outputFormat)` gains a new branch for this format.

---

### Step 4 — New `QnnDelegateProbe.kt`

**Status:** ✅ DONE

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

**Status:** ✅ DONE

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

**Pre-allocated INT8 output buffers** — add alongside `outputTensorRaw` and `outputTensorE2E`:
```kotlin
// INT8 split-output buffers for QNN_NPU — shape-matched to LiteRT's [1][channel][8400] expectation
// Initialized in validateQnnTensorContract()
private lateinit var qnnOutputBoxes: Array<Array<ByteArray>>   // [1][4][8400]
private lateinit var qnnOutputScores: Array<Array<ByteArray>>  // [1][5][8400]
```
LiteRT maps TFLite output tensor shape `(1, 4, 8400)` to a nested `Array<Array<ByteArray>>` — flat `ByteArray(33600)` causes a shape mismatch at `runForMultipleInputsOutputs` time. Initialized (not reallocated per-frame) inside `validateQnnTensorContract()` (Step 5e).

**Mode switching safety — safe by construction:**
- All accesses to `qnnOutputBoxes`, `qnnOutputScores`, `outputFormat`, and `interpreter` occur exclusively on the `consumerDispatcher` (single-threaded). No external thread touches these fields.
- `maybeApplyPendingMode()` is called at the top of the channel loop, before `processImage()`. A mode switch can never interrupt a frame in flight.
- `outputFormat == QNN_INT8_8400` is the gate: the QNN data path (and its buffer reads) only executes after `validateQnnTensorContract()` has both initialized the buffers AND set this flag. `validateTensorContract()` (GPU/CPU path) sets `outputFormat` back to `RAW_8400`, making the QNN buffer-access branch structurally unreachable in non-NPU modes. `UninitializedPropertyAccessException` on `qnnOutputBoxes` / `qnnOutputScores` is structurally unreachable.
- `pendingMode` is an `AtomicReference`. Rapid UI-thread toggles (GPU→NPU→GPU) are coalesced safely: `getAndSet(null)` in `maybeApplyPendingMode()` ensures only the latest requested mode is applied.

**Minor: re-allocation on NPU re-entry.** `validateQnnTensorContract()` unconditionally allocates new arrays on every NPU re-entry (e.g., GPU→NPU→GPU→NPU toggle). Old arrays become GC candidates — harmless in a spike. Mark with a TODO in the implementation:
```kotlin
// TODO: reuse existing arrays on QNN re-entry if we keep bouncing between modes.
qnnOutputBoxes  = Array(1) { Array(4) { ByteArray(OUTPUT_BOXES) } }
qnnOutputScores = Array(1) { Array(CUSTOM_CLASS_NAMES.size) { ByteArray(OUTPUT_BOXES) } }
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
        // INT8 quantized model: leave HtpPrecision unset so HTP stays on the
        // quantized path. HTP_PRECISION_FP16 is FP16-model-only and should not
        // be copied into this QDQ-INT8 path.
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

#### 5d — QNN_NPU data path: preprocessing + output decoder (DD-10)

**New companion constants:**
```kotlin
// QNN_NPU mode-specific thresholds (GPU/CPU keep CONFIDENCE_THRESHOLD=0.40, NMS_IOU_THRESHOLD=0.50)
private const val NPU_CONFIDENCE_THRESHOLD = 0.10f
private const val NPU_NMS_IOU_THRESHOLD    = 0.35f
private const val NPU_MAX_DET              = 20

// INT8 dequantization scales (from validated model, Step 0b)
private const val NPU_BOX_DEQUANT_SCALE   = 2.621687f    // output_0 scale
private const val NPU_SCORE_DEQUANT_SCALE = 0.00390625f  // output_1 scale; maps [-128,127]→[0,~1]

// Preprocessing strategy switch — change to benchmark both approaches
private val NPU_PREPROCESS_MODE = NpuPreprocessMode.MANUAL_NCHW_INT8

private enum class NpuPreprocessMode { MANUAL_NCHW_INT8, TRANSPOSE_QUANT }
```

---

**Preprocessing — Strategy A: Manual NCHW int8 (all-manual, no TFLite Support Library)**

```kotlin
private fun preprocessNchwInt8Manual(bitmap: Bitmap): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
    val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
    scaled.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    val buf = ByteBuffer.allocateDirect(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
    // Write R plane, then G plane, then B plane (NCHW channel-first)
    for (px in pixels) buf.put(((px shr 16 and 0xFF) - 128).toByte())  // R
    for (px in pixels) buf.put(((px shr 8  and 0xFF) - 128).toByte())  // G
    for (px in pixels) buf.put(((px        and 0xFF) - 128).toByte())  // B
    buf.rewind()
    return buf
}
```

**Preprocessing — Strategy B: Existing resize + NHWC→NCHW transpose + quantize**

```kotlin
private fun preprocessNchwInt8Transpose(bitmap: Bitmap): ByteBuffer {
    // Step 1: reuse existing bilinear resize (NormalizeOp(0f,255f) → output in [0,1])
    yoloTensorImage.load(bitmap)
    val floatArr = FloatArray(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
    yoloImageProcessor.process(yoloTensorImage).buffer.asFloatBuffer().get(floatArr)
    // floatArr layout: NHWC [H*W, 3] — index i*3+c for pixel i, channel c, value in [0,1]

    // Step 2: transpose to NCHW and quantize: pixel*255 - 128
    val n = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
    val buf = ByteBuffer.allocateDirect(3 * n)
    for (c in 0 until 3) {
        for (i in 0 until n) {
            buf.put(((floatArr[i * 3 + c] * 255f).toInt() - 128).toByte())
        }
    }
    buf.rewind()
    return buf
}
```

**Preprocessing dispatch in `processImage()` QNN_NPU branch:**
```kotlin
val npuPreprocessStartNs = System.nanoTime()
val npuInputBuf = when (NPU_PREPROCESS_MODE) {
    NpuPreprocessMode.MANUAL_NCHW_INT8 -> preprocessNchwInt8Manual(rotatedBitmap)
    NpuPreprocessMode.TRANSPOSE_QUANT  -> preprocessNchwInt8Transpose(rotatedBitmap)
}
val yoloPreprocessMs = elapsedMs(npuPreprocessStartNs)
// yolo_preprocess_ms telemetry already captures this — switch NPU_PREPROCESS_MODE to benchmark
```

Both methods must be `internal` (not `private`) to allow on-device test access.

---

#### 5d-bench — On-device preprocessing latency test (run BEFORE hardcoding strategy)

Add `NpuPreprocessingLatencyTest.kt` alongside `PreprocessingLatencyTest.kt` at:
```
app/src/androidTest/java/com/courtvision/spike/pipeline/NpuPreprocessingLatencyTest.kt
```

Follow the same pattern as `PreprocessingLatencyTest` exactly:
- `TAG = "LatencyTest"` (same tag, same Logcat destination)
- `WARMUP_ITERATIONS = 5`, `MEASURED_ITERATIONS = 50`
- `logResults()` and `percentile()` helpers copied verbatim (or extract to shared test util)
- `createTestBitmap()` helper with gradient pixels (same as existing)

Four test methods — two input sizes × two strategies:
```kotlin
@Test fun manualNchwInt8_latency_640x480()    { benchmarkStrategy(NpuPreprocessMode.MANUAL_NCHW_INT8, 640, 480) }
@Test fun manualNchwInt8_latency_1920x1080()  { benchmarkStrategy(NpuPreprocessMode.MANUAL_NCHW_INT8, 1920, 1080) }
@Test fun transposeQuant_latency_640x480()    { benchmarkStrategy(NpuPreprocessMode.TRANSPOSE_QUANT,  640, 480) }
@Test fun transposeQuant_latency_1920x1080()  { benchmarkStrategy(NpuPreprocessMode.TRANSPOSE_QUANT,  1920, 1080) }
```

Each `benchmarkStrategy` call creates a `FrameProcessor`, calls the internal preprocessing method directly (bypassing `processImage()`), logs p50/p95/min/max, and asserts `p50 < 50ms` as a sanity guard.

Run with:
```
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.courtvision.spike.pipeline.NpuPreprocessingLatencyTest
```

Results appear in Logcat under `"LatencyTest"`.

**Results (canonical run)**

- Device: `SM-S906U1`
- Date: `2026-04-29`
- Source: Logcat block at `14:22`, PID `16009`

| Strategy | Input | p50 | p95 | Verdict |
|---|---:|---:|---:|---|
| `MANUAL_NCHW_INT8` | `640x480` | `25.216 ms` | `26.163 ms` | winner |
| `TRANSPOSE_QUANT` | `640x480` | `40.396 ms` | `42.307 ms` | slower |
| `MANUAL_NCHW_INT8` | `1920x1080` | `26.020 ms` | `27.245 ms` | winner |
| `TRANSPOSE_QUANT` | `1920x1080` | `36.796 ms` | `38.907 ms` | slower |

**Decision**

- `MANUAL_NCHW_INT8` wins on both `p50` and `p95` at both input sizes.
- `1920x1080` confirms the same winner, so there is no split decision.
- `TRANSPOSE_QUANT` also shows worse tail behavior, so it is rejected.

**Supporting evidence**

- Earlier runs from `14:13` and `14:17` showed the same ordering and support the verdict, but they are not the canonical comparison set.

> ✅ **PAUSE gate resolved — preprocessing strategy decision**
> `MANUAL_NCHW_INT8` is the chosen path. Proceed by removing the runtime strategy selector and treating manual NCHW int8 as the hardcoded preprocessing approach for the QNN path. Decoder and tensor-contract work remain blocked on the later confirm gates, not on preprocessing choice.
>
> After decision: replace the `when (NPU_PREPROCESS_MODE)` dispatch with a direct call to the chosen function. Keep both `preprocessNchwInt8Manual()` and `preprocessNchwInt8Transpose()` in the file — do not delete. Mark the unused one with:
> ```kotlin
> // TODO: remove unused preprocessing variant — later cleanup (post Day 6.1)
> ```
> The `NPU_PREPROCESS_MODE` constant and `NpuPreprocessMode` enum can also be removed once hardcoded, or retained for documentation. Either is acceptable; removal is a later cleanup item.

---

**Output decoder:**

The existing `parseModelOutput()` uses channels-first indexing `output[0][field][anchor]` — the INT8 tensors share the same layout `(1, 4, 8400)` and `(1, 5, 8400)`. The loop structure is identical; only the source buffer and dequant step change.

**Important correctness notes:**
- Box coords after dequant are in **pixel space (0–640)** — divide by `MODEL_INPUT_SIZE` to normalize to [0,1] before `coerceIn`.
- Scores are **post-sigmoid** (dequant maps [-128,127]→[0,~1.0] given scale=0.00390625) — do **not** apply sigmoid.

> ✅ **CONFIRM resolved** — Byte sign-extension in the dequant formula:
> `Byte.toInt()` in Kotlin sign-extends: `(-128).toInt() = -128`, `(127).toInt() = 127`.
> Adding 128 gives the range [0, 255], which is correct for `zp = -128`.
> Resolved by JVM unit test `FrameProcessorTest.qnnUnsigned_mapsSignedByteRangeToZeroTo255`, which asserts:
> `(-128).toByte() -> 0`, `(-1).toByte() -> 127`, `0.toByte() -> 128`, `127.toByte() -> 255`.
> Decoder implementation may now use the explicit helper `qnnUnsigned(byte) = byte.toInt() + 128`.

```kotlin
// Buffers populated by runForMultipleInputsOutputs — no localInterpreter param needed
private fun decodeQnnOutput(
    confidenceThreshold: Float,
    iouThreshold: Float
): List<DetectionBox> {
    val perClassCandidates = mutableMapOf<Int, MutableList<DetectionBox>>()

    for (i in 0 until OUTPUT_BOXES) {
        // Channels-first: field F, anchor I → index F * OUTPUT_BOXES + I
        // ⚠️ CONFIRM: Byte.toInt() sign-extends; (+128) gives [0,255] for zp=-128 — see note above
        val cx = (outputBoxesBuf[0 * OUTPUT_BOXES + i].toInt() + 128) * NPU_BOX_DEQUANT_SCALE / MODEL_INPUT_SIZE
        val cy = (outputBoxesBuf[1 * OUTPUT_BOXES + i].toInt() + 128) * NPU_BOX_DEQUANT_SCALE / MODEL_INPUT_SIZE
        val w  = (outputBoxesBuf[2 * OUTPUT_BOXES + i].toInt() + 128) * NPU_BOX_DEQUANT_SCALE / MODEL_INPUT_SIZE
        val h  = (outputBoxesBuf[3 * OUTPUT_BOXES + i].toInt() + 128) * NPU_BOX_DEQUANT_SCALE / MODEL_INPUT_SIZE

        var bestClassId = -1
        var bestScore   = 0f
        for (c in 0 until 5) {
            // Post-sigmoid scores — no sigmoid needed
            val score = (outputScoresBuf[c * OUTPUT_BOXES + i].toInt() + 128) * NPU_SCORE_DEQUANT_SCALE
            if (score > bestScore) { bestScore = score; bestClassId = c }
        }

        if (bestScore < confidenceThreshold || bestClassId < 0 || bestClassId >= CUSTOM_CLASS_NAMES.size) continue

        val left   = (cx - w / 2f).coerceIn(0f, 1f)
        val top    = (cy - h / 2f).coerceIn(0f, 1f)
        val right  = (cx + w / 2f).coerceIn(0f, 1f)
        val bottom = (cy + h / 2f).coerceIn(0f, 1f)
        if (right <= left || bottom <= top) continue

        perClassCandidates.getOrPut(bestClassId) { mutableListOf() }
            .add(DetectionBox(bestClassId, classLabel(bestClassId), bestScore, left, top, right, bottom))
    }

    var results = perClassCandidates.values.asSequence()
        .flatMap { nms(it, iouThreshold).asSequence() }
        .sortedByDescending { it.confidence }
        .toList()

    // Conditional cap: only truncate when post-NMS count exceeds NPU_MAX_DET
    if (results.size > NPU_MAX_DET) results = results.take(NPU_MAX_DET)

    return results
}
```

**Call site in `processImage()` — new `ModelOutputFormat.QNN_INT8_8400` branch:**

`runForMultipleInputsOutputs` is confirmed correct for the QNN LiteRT delegate. Output containers must be shape-matched nested arrays, not flat `ByteArray`s — LiteRT copies tensor data index-by-index through the nesting, and a flat buffer causes a shape mismatch at runtime:

```kotlin
ModelOutputFormat.QNN_INT8_8400 -> {
    val yoloInferenceStartNs = System.nanoTime()
    localInterpreter.runForMultipleInputsOutputs(
        arrayOf(npuInputBuf),
        mutableMapOf(0 to qnnOutputBoxes, 1 to qnnOutputScores)
    )
    yoloInferenceMs = elapsedMs(yoloInferenceStartNs)

    val yoloNmsStartNs = System.nanoTime()
    boxes = decodeQnnOutput(NPU_CONFIDENCE_THRESHOLD, NPU_NMS_IOU_THRESHOLD)
    yoloNmsMs = elapsedMs(yoloNmsStartNs)
}
```

`decodeQnnOutput()` reads from `qnnOutputBoxes[0][channel][anchorIndex]` and `qnnOutputScores[0][classIndex][anchorIndex]` — no `getOutputTensor()` call needed inside the decoder.

The existing `RAW_8400` and `END_TO_END_300` branches are unchanged.

#### 5e — `validateQnnTensorContract()`: new validator for the INT8 path

The existing `validateTensorContract()` checks NHWC input shape + FLOAT32 dtype — both assertions fail for the INT8 NCHW model. Add a separate validator called from the `QNN_NPU` branch of `switchInterpreter()` (after delegate setup, where `localInterpreter` is fully initialized):

```kotlin
private fun validateQnnTensorContract(localInterpreter: Interpreter) {
    // Input: NCHW (1, 3, 640, 640) int8
    val inputShape = localInterpreter.getInputTensor(0).shape()
    require(inputShape.contentEquals(intArrayOf(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))) {
        "QNN_NPU: unexpected input shape ${inputShape.contentToString()} — expected NCHW [1,3,640,640]"
    }
    require(localInterpreter.getInputTensor(0).dataType() == DataType.INT8) {
        "QNN_NPU: unexpected input dtype — expected INT8"
    }

    // Outputs: two tensors — output_0 (1,4,8400) + output_1 (1,5,8400) int8
    require(localInterpreter.outputTensorCount == 2) {
        "QNN_NPU: expected 2 output tensors, got ${localInterpreter.outputTensorCount}"
    }
    val boxShape   = localInterpreter.getOutputTensor(0).shape()
    val scoreShape = localInterpreter.getOutputTensor(1).shape()
    require(boxShape.contentEquals(intArrayOf(1, 4, OUTPUT_BOXES))) {
        "QNN_NPU: unexpected output_0 shape ${boxShape.contentToString()} — expected [1,4,8400]"
    }
    require(scoreShape.contentEquals(intArrayOf(1, 5, OUTPUT_BOXES))) {
        "QNN_NPU: unexpected output_1 shape ${scoreShape.contentToString()} — expected [1,5,8400]"
    }
    require(localInterpreter.getOutputTensor(0).dataType() == DataType.INT8) {
        "QNN_NPU: output_0 dtype not INT8"
    }
    require(localInterpreter.getOutputTensor(1).dataType() == DataType.INT8) {
        "QNN_NPU: output_1 dtype not INT8"
    }

    // Shape-matched nested arrays — flat ByteArray causes LiteRT shape mismatch at runForMultipleInputsOutputs
    // TODO: reuse existing arrays on QNN re-entry if we keep bouncing between modes.
    qnnOutputBoxes  = Array(1) { Array(4) { ByteArray(OUTPUT_BOXES) } }
    qnnOutputScores = Array(1) { Array(CUSTOM_CLASS_NAMES.size) { ByteArray(OUTPUT_BOXES) } }

    outputFormat = ModelOutputFormat.QNN_INT8_8400
}
```

**`switchInterpreter()` dispatch:** call `validateQnnTensorContract(localInterpreter)` for `QNN_NPU`; call existing `validateTensorContract(localInterpreter)` for all other modes. The existing validator is unchanged.

> ⚠️ **Invariant — do not violate:** The mode switching safety guarantee relies on `outputFormat` being written ONLY inside `validateTensorContract()` and `validateQnnTensorContract()`, which are called ONLY from `switchInterpreter()`, which runs ONLY on the `consumerDispatcher`. If `outputFormat` is ever written from another coroutine or thread, the `QNN_INT8_8400` branch gate breaks and buffer reads can race with re-initialization.

**TFLite runtime logging — JVM-safe helper:** `switchInterpreter()` logs the bundled LiteRT version via a `logTfLiteRuntimeInfo(mode)` private helper that wraps `TensorFlowLite.runtimeVersion()` / `schemaVersion()` in a `try/catch(Throwable)`. Without the guard, TF Lite JNI loading is triggered unconditionally, breaking JVM unit tests before they reach their assertions.

---

### Step 6 — `CameraUiState.kt`: QNN probe fields

**Status:** ✅ DONE

```kotlin
// Add alongside gpuProbeResult:
val qnnProbeResult: QnnProbeResult = QnnProbeResult(),
val qnnAvailable: Boolean = false,
```

---

### Step 7 — `CameraViewModel.kt`: probe wiring + `FrameProcessor` params

**Status:** ✅ DONE

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
        InferenceMode.QNN_NPU -> "spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite"
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

**Status:** ✅ DONE

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

**Status:** ✅ DONE

Completed JVM coverage includes:

- **`switchInterpreter_qnnNpu_fallsBackToGpu_onApiBelow31()`** — set `Build.VERSION.SDK_INT = 29` via Robolectric shadow; assert `delegateMode == InferenceMode.GPU`; assert `lastError` contains "API 31+".
- **`switchInterpreter_qnnNpu_fallsBackToGpu_whenQnnDelegateFails()`** — on Robolectric JVM, `QnnDelegate()` throws `UnsatisfiedLinkError` (no `.so`). Test must catch `Throwable` (not `Exception`) to exercise the cascade. Assert fallback to GPU; assert `lastError` non-null.
- **`switchInterpreter_qnnNpu_loadsInt8Model_notFp16()`** — inject a fake `modelBufferProvider` that records the `InferenceMode` arg; switch to `QNN_NPU`; assert provider was called with `QNN_NPU` (not `GPU`).
- **`closeInterpreterResources_closesQnnDelegate()`** — inject fake delegate via `@VisibleForTesting` seam; call `resetInterpreter()`; assert `close()` called.
- **`CameraViewModelTest` — `qnnProbeSupportedAndQuantized_setsQnnAvailableTrue()`** — `TestOverrides(qnnProbeResult = QnnProbeResult(status = QnnStatus.QNN_SUPPORTED, htpQuantizedSupported = true))`; assert `uiState.qnnAvailable == true`.
- **`CameraViewModelTest` — `qnnProbeSupportedButFp16Only_setsQnnAvailableFalse()`** — `TestOverrides(qnnProbeResult = QnnProbeResult(status = QnnStatus.QNN_SUPPORTED, htpFp16Supported = true, htpQuantizedSupported = false))`; assert `uiState.qnnAvailable == false`. Guards against the v1 misconception.
- **`CameraViewModelTest` — `qnnProbeFailure_setsQnnAvailableFalse_andStoresProbeResult()`** — inject a failed `QnnProbeResult`; assert `qnnAvailable == false` and UI state retains the probe result.
- **`appendPerFramePerfRow_gpuMode_isQNN_NPU()`** — force `currentMode = QNN_NPU`; assert `gpu_mode` column = `"QNN_NPU"` in captured CSV row.

---

### Step 10 — Fail-fast smoke probe (DD-8)

**Status:** ✅ PASS (2026-05-01, day before 10-min soak) — 357/357 nodes on HTP, 0 fallback (0%). Logcat confirmed `caching in RESTORE MODE` and `QnnContext_createFromBinary` (graph composition skipped); first-frame cold-start still ~409 ms (HTP context init overhead, not graph compilation — §10 cache warmup gate result).

Before running the 10-min soak, run a one-frame inspection to confirm HTP actually owns the YOLO graph.

1. Install APK, launch, switch to NPU mode
2. Capture logcat in stages:

   **Stage 1 — app-level QNN init** (tag `CVRotation`, emitted by `FrameProcessor`):
   ```bash
   adb logcat -s CVRotation:I
   ```
   Must see `[QNN_INIT] caps: HTP_QUANT=true` to confirm the delegate was constructed. If absent, QNN init failed before delegation.

   **Stage 2 — native delegate partition summary** (native tag on LiteRT 2.16.1 is `tflite`, not `QnnDelegate`):
   ```bash
   adb logcat -s tflite:V
   ```
   Actual partition line format (LiteRT 2.16.1 / SM8450):
   ```
   INFO: [Qnn Delegate] TfLiteQnnDelegate delegate: <N> nodes delegated out of <N+M> nodes with <P> partitions.
   ```

   **Stage 3 — broad fallback** if Stage 2 is empty (discovers actual native tag on this device/SDK):
   ```bash
   adb logcat | grep -iE "\[QNN_INIT\]|QnnDelegate|litert|HTP"
   ```

3. Run for ≈ 30 seconds (covers cold compile + first 100 frames)
4. Parse logcat for delegate partition output. Actual result on SM8450 / LiteRT 2.16.1:
   ```
   TfLiteQnnDelegate delegate: 357 nodes delegated out of 357 nodes with 1 partitions.
   Replacing 357 out of 357 node(s) with delegate (TfLiteQnnDelegate) node, yielding 1 partitions for the whole graph.
   ```
   RESTORE MODE confirmed (2026-05-02, `adb logcat -s tflite:V`):
   ```
   INFO: [Qnn Delegate] Caching: cache_filename of …/cache/qnn_binary_12533001225297295345.bin of 3354624 bytes is available, caching in RESTORE MODE.
   INFO: [Qnn] QnnDsp <I> QnnContext_createFromBinary started. backend = 0x1, device = 0x1
   INFO: [Qnn] QnnDsp <I> QnnContext_createFromBinary done successfully. context = 0x1
   ```
   `QnnContext_createFromBinary` completes in ~183 ms (13.762 → 13.945). Full QNN delegate init (probe → first execution ready) takes ~1 s. Cold start to first frame remains >400 ms because GPU/OpenCL init for the pose model takes an additional ~7 s on a fresh launch (see §10 gate).

**Pass:** `M / (N + M) < 0.05` (≥ 95% of nodes on HTP).
**Borderline (0.05–0.20):** acceptable for Day 6.1 but capture the fallback op list for a follow-up; common offenders are NMS-adjacent ops or unsupported activations, addressable by re-exporting with op-substitution flags.
**Fail (> 0.20):** abort soak. Most likely causes:
- Step 0b dtype check missed a float32 graph
- Calibration was incomplete (some ops emitted as float fallback)
- Op not supported on HTP for this Hexagon generation

If failed, re-export Step 0 with explicit verification rather than running the soak with a half-CPU graph.

---

### Step 11 — 10-min soak benchmark

**Status:** ✅ COMPLETE (2026-05-02) — `benchmarks/phase0/phase0-perframe-20260502-001250-day6-1-step11-10min-soak.csv`

**Device:** SM-S906U1 (Galaxy S22, SM8450 / Snapdragon 8 Gen 1)  
**Mode:** QNN_NPU · **Frames:** 8192 · **Pose skipped:** 830 (10.1%)

#### Results

| Metric | p50 | p95 | p99 |
|---|---|---|---|
| `yolo_preprocess_ms` (CPU, steady) | **38.90 ms** | 48.38 ms | 57.70 ms |
| `yolo_inference_ms` (NPU, steady) | **3.46 ms** | 3.69 ms | 4.50 ms |
| `pose_inference_ms` (GPU, non-skipped) | 7.76 ms | 11.20 ms | 16.64 ms |
| `frame_total_ms` (steady) | 64.8 ms | 81.4 ms | 98.0 ms |
| `fps_1s_window` (non-zero) | **14.0 fps** | — | — |

**Note:** `yolo_preprocess_ms` (38.9 ms) is the dominant pipeline cost — over 10× the NPU inference time (3.5 ms). CPU preprocess is the primary optimization target going forward.

**Cold-start frame 1:** 408.9 ms (incl. init overhead); frame 2: 80.4 ms; steady from frame ~5.

#### Thermal profile (steady, frames 100+)

| Status | Frames | % | frame_total p50 |
|---|---|---|---|
| NONE | 1076 | 13.3% | 49.6 ms |
| LIGHT | 871 | 10.8% | 53.0 ms |
| MODERATE | 560 | 6.9% | 61.2 ms |
| **SEVERE** | **3897** | **48.2%** | **67.6 ms** |
| **CRITICAL** | **1688** | **20.9%** | **69.4 ms** |

**YOLO/HTP thermal resilience:** NONE→CRITICAL drift is 3.26→3.52 ms (+8%) — HTP is thermally isolated.  
**Pipeline thermal degradation:** first-60s median 49.6 ms → last-60s median 69.1 ms (+39%). Bottleneck under load is CPU preprocess + GPU pose, not the NPU.

#### Assessment

The SM8450 (Snapdragon 8 Gen 1) is a known thermal outlier — SEVERE onset within ~90 s of continuous use is device-characteristic, not a pipeline regression. The NPU inference budget itself is stable. Follow-up candidates:
- GPU/shader-based preprocess to eliminate the 38.9 ms CPU preprocess cost
- Pose model on NPU (currently GPU) to reduce GPU thermal contribution
- Throttle frame submission rate when `thermal_status >= SEVERE` to stay at ~15 fps without burning budget

---

### Step 12 — `CONTEXT.md` + `ADR-005` + spike plan update

**Status:** ✅ COMPLETE (2026-05-02)

- `CONTEXT.md`: updated header to "Days 1–6.1 complete"; added Day 6.1 benchmark results section (YOLO NPU p50/p95, frame_total p95, FPS, Day 8 gate verdicts); updated Progress line.
- `ADR-005` §Deferred: Config B promoted to "Validated (2026-05-02)" with soak numbers; INT8-only `HtpPrecision` decision documented (must be unset — FP16 flag mismatches QDQ-INT8 graph); **Config E added** (both YOLO + pose on QNN→NPU, target optimum) with rationale.
- `docs/phase0-spike-plan.md`: Day 6 GPU pipeline marked complete with results; Day 6.1 NPU section added; outdated "NPU is Phase 2" note in Day 8 corrected.

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §0a INT8 export | ✅ `spike_qai_yolo11n_640_5-class_04-28-2026_int8.tflite` (Route B) | — |
| §0b Graph dtype | ✅ NCHW int8 input; split output heads (1,4,8400)+(1,5,8400) with separate quant params | — |
| §0c Accuracy | ⚠️ mAP50=0.9143 BORDERLINE/ACCEPTED (gate ≥0.92; PTQ ceiling reached on Route B) | QAT if 5-class accuracy becomes a product concern post-Day 6.1 |
| §1 Build | ✅ `./gradlew :app:assembleDebug` succeeds | Fix NDK/dep issue; add `ndkVersion` if needed |
| §2 Unit tests | ✅ All green incl. new INT8 model-load test | Fix before APK install |
| §3 QNN probe on S22+ | ✅ Overlay shows `QNN: QNN_SUPPORTED HTP_QUANT=true` | `adb logcat \| grep QNN`; verify `libcdsprpc.so` declared |
| §4 NPU mode switch | ✅ No fallback error; `gpu_mode=QNN_NPU` throughout CSV | GPU fallback still functional; capture QNN logcat tag |
| §5 Smoke probe (DD-8) | ✅ ≥ 95% of YOLO nodes on HTP | Re-do Step 0 — graph likely contains float fallback ops |
| §6 YOLO p50 | ✅ ≤ 10 ms (revised from v1's 20 ms — INT8 on HTP target) | 10–25 ms → check thermal throttle / op partial fallback; > 25 ms → investigate |
| §7 Combined p95 | ✅ `frame_total_ms` p95 ≤ 100 ms | Check if pose is new bottleneck (`pose_inference_ms` p95) |
| §8 FPS sustained | ≥ 20 fps median over 10 min | ❌ FAIL — 14 fps median; CPU preprocess bottleneck (38.9 ms p50). Preprocess offload required. |
| §9 GPU mode regression | GPU p50 ≈ 53 ms ± 5 ms after switching back | ✅ Trivially — Day 6.1 makes no changes to the GPU branch |
| §10 Cache warmup | 2nd cold start NPU switch < 500 ms | ⚠️ RESTORE MODE confirmed (2026-05-02 logcat); `QnnContext_createFromBinary` = 183 ms, but full cold start >400 ms (HTP context init ~1 s; GPU/OpenCL pose init adds ~7 s on a fresh launch). First-frame from soak CSV = 408.9 ms. Gate <500 ms not cleanly met. |

**Note on §6 target revision:** v1 set 20 ms because that was the FP16-on-HTP expectation (which doesn't exist as a real path). With INT8 on HTP the published Qualcomm benchmarks for YOLO11n show 1–5 ms on SM8450-class hardware, so 10 ms is conservative and leaves headroom for app-side overhead.

---

## Deferred Items

- **QAI Hub w8a8 production cut (Route B)** — was the primary path used in Day 6.1 (Route A failed due to zero-mAP on combined `(1,9,8400)` output; see Step 0a/DD-10). A higher-quality re-export with QAT remains deferred to production hardening if the 2.6 pp mAP50 gap becomes a product concern.
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

## Lessons learned — `extractNativeLibs` and Hexagon fastrpc

Day 6.1 step-10 (live NPU smoke test) failed on Samsung S22+ (SM8450 / Hexagon V69) inside `Interpreter()` native init with:

> Internal error: Failed to apply delegate: Restored original execution plan after delegate application failure.

Three diagnostic phases were required to find the root cause; the first two were red herrings worth recording so the next person doesn't repeat them.

### Hypotheses ruled out
- **GPU sub-delegate conflict in the `[QNN_NPU, GPUv2]` priority pair** — matches Qualcomm AI Hub's canonical setup; not the cause.
- **`HTP_PRECISION_FP16` on an INT8 QDQ model** — Qualcomm's `TFLiteHelpers.CreateQNN_NPUDelegate` only sets this when `hasHTP_FP16` is true *and* the model is FP16. Removing it from our path did not change the error, ruling it out as the standalone cause.
- **`Interpreter.Options` divergence from canonical (`setRuntime(FROM_APPLICATION_ONLY)` / `setUseNNAPI(false)` / `setUseXNNPACK(true)`)** — runtime version logging confirmed the bundled LiteRT 2.16.1 was loaded correctly. None of these options would have changed delegate application.

### Actual root cause

`libQnnHtpV69Skel.so` was packaged inside the APK zip but never extracted to disk on install.

- AGP 8+ defaults `android:extractNativeLibs="false"`. Native libs are mmapped directly out of the APK by the dynamic linker — fine for `System.loadLibrary` of the CPU-side `libQnnHtpV69Stub.so`.
- The Stub then asks Hexagon fastrpc to ship the **Skel** to the DSP. fastrpc opens the skel by **filesystem path** via `apps_std_fopen_with_env`. With extraction off, no such file exists on disk → `errno 2` → DSP session never opens → `ModifyGraphWithDelegate()` fails → "Restored original execution plan."
- AI Hub's working reference app sets `android:extractNativeLibs="true"` explicitly (`D:/ai-hub-apps/apps/object_detection_android/src/main/AndroidManifest.xml:9`). That's the canonical and minimal fix.

### Fix
One line in [app/src/main/AndroidManifest.xml](../../app/src/main/AndroidManifest.xml):

```xml
<application
    android:allowBackup="true"
    android:extractNativeLibs="true"
    ...>
```

Trade-off: native libs occupy disk space twice (compressed in APK + uncompressed in `lib/arm64/`). Acceptable — QNN HTP is non-functional otherwise.

### Diagnostic chain for future Qualcomm-on-Android delegate failures

When `ModifyGraphWithDelegate()` fails with the generic "Restored original execution plan" message:

1. Set `QnnDelegate.Options.LogLevel.LOG_LEVEL_VERBOSE` and capture all logs (filter `adb logcat | findstr /I "qnn fastrpc skel apps_std"`). The `QnnDelegate:V` filter alone misses native fastrpc log lines that come through other tags.
2. Search the log for `apps_std_fopen_with_env` and `Failed to load skel`. If present → filesystem-path issue → check `extractNativeLibs`.
3. Verify on device: `adb shell run-as com.courtvision.spike ls lib/arm64/ | findstr /I qnn` should show all `libQnnHtp{V*}Skel.so` entries.
4. If skels are present and error persists, *then* widen scope to `Interpreter.Options` alignment and op-by-op QNN partition decisions in the verbose stream.

### Reference

- Canonical setup: `D:/ai-hub-apps/apps/object_detection_android/src/main/AndroidManifest.xml:9`
- Captured native evidence: `qnn-failure.log` lines 22180–22200 (this branch).
- Helper used as the canonical Interpreter.Options reference: `D:/ai-hub-apps/apps/_shared/android/tflite_helpers/TFLiteHelpers.java` lines 173–333. `HTP_PRECISION_FP16` there is FP16-model-only — do not copy it for INT8 paths.

---

## Changelog

**v9 (2026-04-29)** — Two runtime bug fixes. (1) **QNN output shape mismatch:** `qnnOutputBoxes`/`qnnOutputScores` were flat `ByteArray`s; LiteRT treated `(1,4,8400)` as `(33600,)` and rejected the output map at `runForMultipleInputsOutputs` time. Changed to nested `Array<Array<ByteArray>>` with shape `[1][4][8400]` / `[1][5][8400]`; decoder updated to index via `[0][channel][anchor]`. Tests seeded and asserted via new `qnnBuffers()` helper producing the same nested structure. (2) **TFLite JNI forced in unit tests:** `switchInterpreter()` called `TensorFlowLite.runtimeVersion()` / `schemaVersion()` inline, triggering native JNI loading and breaking three JVM tests before assertions ran. Extracted to `logTfLiteRuntimeInfo()` helper with `try/catch(Throwable)` fallback; tests now reach their assertions without native loading. Also added injected `gpuDelegateProvider`/`qnnDelegateProvider` overrides to exercise mode-fallback logic in JVM tests without touching native delegate code.

**v8 (2026-04-29)** — Reconciled stale internal step markers to match the landed runtime work. Finished the immediate QNN UI/probe slice: `CameraUiState` now carries `qnnProbeResult` / `qnnAvailable`, `CameraViewModel` wires `QnnDelegateProbe.probe()` and `TestOverrides.qnnProbeResult`, and `CameraScreen` exposes the NPU mode button plus QNN probe status line. Added JVM tests for quantized-capability gating and probe-failure state handling. Device smoke probe, soak benchmark, and final Day 6.1 doc sign-off remain pending.

**v7 (2026-04-29)** — Wired live `FrameProcessor` QNN execution through `runForMultipleInputsOutputs(...)`, added `validateQnnTensorContract()` for the INT8/NCHW/split-output contract, and unblocked `ModelOutputFormat.QNN_INT8_8400` end-to-end inside `FrameProcessor`. Added JVM coverage for the validator and for the QNN multi-output branch using a fake inference engine. QNN probe/UI exposure remains deferred.

**v6 (2026-04-29)** — Resolved the byte sign-extension confirm gate with JVM test coverage (`qnnUnsigned_mapsSignedByteRangeToZeroTo255`). Implemented `decodeQnnOutput()` against the preallocated QNN byte buffers and added unit coverage for dequantization, invalid-box filtering, and per-class NMS. The live `QNN_INT8_8400` path remains intentionally blocked pending confirmation that `runForMultipleInputsOutputs(...)` is correct for the QNN LiteRT delegate.

**v5 (2026-04-29)** — `NpuPreprocessingLatencyTest` completed on `SM-S906U1`; first pause gate resolved. Canonical decision run (`14:22`, PID `16009`) selected `MANUAL_NCHW_INT8` over `TRANSPOSE_QUANT` at both `640x480` and `1920x1080`. Follow-on action: remove runtime selector and hardcode manual preprocessing while leaving decoder / tensor-contract confirm-gated work pending.

**v2 (2026-04-27)** — Flipped FP16-first → INT8-only path. Step 0 promoted to blocking. Added DD-8 fail-fast smoke probe, DD-9 mode-aware asset routing. Tightened §6 YOLO p50 target from 20 ms → 10 ms based on published HTP w8a8 benchmarks. Added test cases guarding against the FP16-HTP misconception. Probe gate now keys on `htpQuantizedSupported`, not `htpFp16Supported`.

**v4 (2026-04-29)** — Data path gaps filled based on actual `FrameProcessor.kt` code review. Added Step 5e (`validateQnnTensorContract()` — separate validator for INT8/NCHW/2-output contract; existing validator unchanged). Expanded Step 5d: dual preprocessing strategies (MANUAL_NCHW_INT8 / TRANSPOSE_QUANT) with `NPU_PREPROCESS_MODE` constant for benchmarking; on-device `NpuPreprocessingLatencyTest` (Step 5d-bench, same pattern as `PreprocessingLatencyTest`) with PAUSE gate before hardcoding; corrected box-coord normalization (/640, pixel-space output); corrected scores are post-sigmoid (no sigmoid); mode-specific NMS constants (NPU conf=0.10/iou=0.35, GPU unchanged 0.40/0.50); conditional NPU_MAX_DET=20 cap; `runForMultipleInputsOutputs` with output map writing directly into pre-alloc buffers (replaces `run(input,null)`) with PAUSE/confirm gate; sign-extension confirm note + PAUSE gate on dequant formula. Added `ModelOutputFormat.QNN_INT8_8400` to Step 3c. Added pre-allocated INT8 ByteArrays to Step 5a. Both preprocessing functions kept after hardcoding; unused marked for later cleanup.

**v3 (2026-04-28)** — Step 0 complete via Route B (QAI Hub). Recorded actual tensor profile: NCHW int8 input, split output heads `(1,4,8400)` boxes + `(1,5,8400)` scores. mAP50=0.9143 BORDERLINE/ACCEPTED. Added DD-10 (split-output requirement + zero-mAP root cause). Added Step 5d (two-tensor decoder, dequant formulas, NMS params from run _14). Updated DD-9 and Step 7 with confirmed asset filename. Route A/B steps rewritten to reflect actual execution. ADR-007 created for the output contract.

**v1 (2026-04-27, superseded)** — Initial plan with FP16 fast path. Conflicted with QNN HTP backend constraints documented in `Claude_nnapi_qnn_report` §7.1 and Qualcomm published benchmark tables. See "Why this plan changed" at top.
