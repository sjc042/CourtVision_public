# Plan: Day 6.2 — Trivial Perf Wins (Gaps 3, 4, 5)

**Status:** 🟡 In progress — Steps 0–5 implemented; Step 6 soak pending  
**Created:** 2026-04-27  
**Gaps addressed:** Gap 3 (`INFERENCE_PREFERENCE_SUSTAINED_SPEED`), Gap 4 (pre-allocated `TensorImage` for YOLO), Gap 5 (`setAllowBufferHandleOutput`)  
**New dependencies:** None

---

## Context

Three one-to-three-line fixes identified by comparing the Qualcomm AI Hub demo against the CourtVision pipeline. None require new libraries. All target the delegate options or preprocessing hot path.

**Best applied before Day 6.1 (QNN):** Gap 3 improves the GPU thermal baseline independently of whether QNN is added, and the `buildSustainedSpeedGpuDelegate()` helper created here becomes a reuse point for both Day 6.1 (QNN sub-delegate) and Day 6.3 (serialization upgrade).

**Gap analysis:** [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md) Gaps 3, 4, 5  
**Day 6 baseline:** frame_total p95 = 129 ms, FPS = 8, thermal MODERATE onset t=68 s

---

## Branch

`perf/day6-x-post-day6-perf-follow-up` (shared workstream branch from `main`; execute Day 6.2 → Day 6.1 → Day 6.3 in order)

---

## Gap 3 — `INFERENCE_PREFERENCE_SUSTAINED_SPEED`

### Problem

`switchInterpreter()` creates the GPU delegate as:
```kotlin
GpuDelegate(CompatibilityList().bestOptionsForThisDevice)
```
`bestOptionsForThisDevice` returns device-default options which on many Adreno GPUs defaults to `INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER` — maximises single-inference throughput at the cost of thermal stability under sustained load. This is the likely driver-level cause of the thermal ceiling at t=68 s in the Day 6 soak.

`GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED` (value = 1) instructs the Adreno driver to hold a stable clock frequency across repeated calls instead of boosting then throttling. This enum is in the standard `tensorflow-lite-gpu` library already on the classpath — **no new dependency required.**

### Fix

Extract a shared helper in `GpuDelegateProbe.kt` (reused by `FrameProcessor` and `PoseLandmarkInterpreter`):

```kotlin
// GpuDelegateProbe.kt — add top-level function
fun buildSustainedSpeedGpuDelegate(): GpuDelegate =
    GpuDelegate(
        GpuDelegate.Options().apply {
            inferencePreference = GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
            isPrecisionLossAllowed = true  // allow FP16 internally on compatible Adreno
        }
    )
```

Apply in `FrameProcessor.kt` — GPU branch of `switchInterpreter()`:
```kotlin
// Replace:
localGpuDelegate = GpuDelegate(CompatibilityList().bestOptionsForThisDevice).also { options.addDelegate(it) }
// With:
localGpuDelegate = buildSustainedSpeedGpuDelegate().also { options.addDelegate(it) }
```

Apply in `PoseLandmarkInterpreter.kt` — `init` block (pose GPU delegate):
```kotlin
// Replace:
GpuDelegate(compatibility.bestOptionsForThisDevice).also { options.addDelegate(it) }
// With:
buildSustainedSpeedGpuDelegate().also { options.addDelegate(it) }
```

Drop the `CompatibilityList` instantiation from both callers. In `FrameProcessor`, the `isDelegateSupportedOnThisDevice` check remains in `switchInterpreter()`. In `PoseLandmarkInterpreter`, no guard is needed at this call site because `useGpu` already carries the runtime decision from `FrameProcessor`.

### Files changed
- `app/src/main/java/com/courtvision/spike/pipeline/GpuDelegateProbe.kt` — add `buildSustainedSpeedGpuDelegate()` top-level function
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` — GPU branch in `switchInterpreter()`
- `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt` — `init` block

---

## Gap 4 — Pre-allocated `TensorImage` for YOLO

### Problem

[`FrameProcessor.kt:491`](../app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt#L491):
```kotlin
val tensorImage = yoloImageProcessor.process(TensorImage.fromBitmap(rotatedBitmap))
```
`TensorImage.fromBitmap()` constructs a new `TensorImage` and its internal `TensorBuffer` on every frame. At 30fps this is ~30 heap allocations per second on the hot path — a violation of the no-alloc rule in CONTEXT.md §ML Pipeline Architecture.

`PoseLandmarkInterpreter` already follows the correct pattern:
```kotlin
// PoseLandmarkInterpreter.kt:33-36 — pre-allocated class members
private val preprocessor = ImageProcessor.Builder().add(NormalizeOp(0f, 255f)).build()
private val tensorImage   = TensorImage(DataType.FLOAT32)
// in infer():
tensorImage.load(cropBitmap)
preprocessor.process(tensorImage)
```

### Fix

Add a pre-allocated `TensorImage` field alongside the existing `yoloImageProcessor` in `FrameProcessor`:

```kotlin
// Class members — add alongside yoloImageProcessor:
private val yoloTensorImage = TensorImage(DataType.FLOAT32)
```

Update `processImage()` at line 491:
```kotlin
// Before:
val tensorImage = yoloImageProcessor.process(TensorImage.fromBitmap(rotatedBitmap))

// After:
yoloTensorImage.load(rotatedBitmap)
val tensorImage = yoloImageProcessor.process(yoloTensorImage)
```

`TensorImage.load(Bitmap)` reuses the backing buffer if the bitmap dimensions match; otherwise it reallocates once and reuses thereafter. Since `rotatedBitmap` is consistently the full camera frame at fixed resolution within a session, the buffer is allocated at most once per mode switch.

### Files changed
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` — one new field, one line changed in `processImage()`

---

## Gap 5 — `setAllowBufferHandleOutput(true)`

### Problem

`Interpreter.Options` default is `allowBufferHandleOutput = false`. With this default, TFLite automatically copies all output tensors from GPU memory to CPU DRAM after every inference call. For YOLO RAW_8400 output `[1, 9, 8400]` float32 ≈ 302 KB of DMA transfer per frame.

### Fix

Add to `Interpreter.Options` in `switchInterpreter()`:

```kotlin
val options = Interpreter.Options().apply {
    setNumThreads(4)
    setAllowBufferHandleOutput(true)   // defer GPU→CPU output copy until tensor is explicitly read
}
```

**Scope note:** CourtVision's CPU-side NMS reads the output tensor immediately after `run()`, so the copy still occurs at read time. The immediate benefit is establishing the correct default for Phase 2 (GPU-side NMS, chained GPU ops). Setting it now is a free one-liner with no risk.

### Files changed
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` — one line in `switchInterpreter()` options block

---

## Step-by-Step Implementation

---

### Step 0 — Add `buildSustainedSpeedGpuDelegate()` helper

**Status:** ✅ Done

Add the top-level function at the bottom of `GpuDelegateProbe.kt`. Import `GpuDelegate` and `GpuDelegate.Options` (already imported in that file). No new imports needed.

---

### Step 1 — Gap 3: apply to `FrameProcessor.kt` GPU branch

**Status:** ✅ Done

In `switchInterpreter()`, `InferenceMode.GPU` branch, replace the `GpuDelegate(CompatibilityList().bestOptionsForThisDevice)` call with `buildSustainedSpeedGpuDelegate()`. The surrounding `CompatibilityList().isDelegateSupportedOnThisDevice` guard remains unchanged.

---

### Step 2 — Gap 3: apply to `PoseLandmarkInterpreter.kt` `init`

**Status:** ✅ Done

Replace `GpuDelegate(compatibility.bestOptionsForThisDevice)` with `buildSustainedSpeedGpuDelegate()`. Import the helper. The `CompatibilityList` variable is still used for the `isDelegateSupportedOnThisDevice` guard — keep it.

---

### Step 3 — Gap 4: `yoloTensorImage` field + `load()`

**Status:** ✅ Done

Add `private val yoloTensorImage = TensorImage(DataType.FLOAT32)` as a class member. Update `processImage()` line 491.

---

### Step 4 — Gap 5: `setAllowBufferHandleOutput(true)`

**Status:** ✅ Done

Add the single line to the `Interpreter.Options` block in `switchInterpreter()`.

---

### Step 5 — Tests

**Status:** ✅ Done

- **Implemented test strategy:** the original plan named `switchInterpreter_gpu_usesSustainedSpeedPreference()`. Actual implementation uses `GpuDelegateProbeTest.sustainedSpeedGpuDelegateConfig_enablesSustainedSpeedAndPrecisionLoss()` as a JNI-safe substitute. It verifies the config seam values rather than reflecting into native delegate internals.
- **No regressions on existing GPU delegate tests** — re-run `FrameProcessorTest` after changes.

---

### Step 6 — Benchmark: GPU mode 10-min soak (user-run)

**Status:** 🔲 TODO — Pending user-run 10-minute GPU soak

Measure thermal improvement from Gap 3 independently of QNN.

1. Install APK with Day 6.2 changes; device pre-cooled
2. Select GPU mode (not NPU)
3. Run 10-min soak; pull CSV
4. Compare thermal trajectory vs Day 6 baseline:
   ```python
   import pandas as pd
   df = pd.read_csv("phase0-perframe-YYYYMMDD-HHMMSS.csv")
   gpu = df[df["gpu_mode"] == "GPU"]
   moderate = gpu[gpu["thermal_status"] == "MODERATE"]["timestampMs"]
   print("MODERATE onset s:", (moderate.min() - gpu["timestampMs"].min()) / 1000)
   print("FPS median:", gpu["fps_1s_window"].median())
   ```
5. Save artifact to `benchmarks/phase0/combined_pipeline/run_YYYYMMDD_HHMMSS_gpu_sustained/`

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §1 Build | `./gradlew :app:assembleDebug` green — no new deps to resolve | — |
| §2 Unit tests | All existing tests green; new Gap 3 test passes | Fix before APK |
| §3 No GPU regression | YOLO p50 ≈ 53 ms ± 5 ms | If slower: revert `SUSTAINED_SPEED` from GPU-only branch; keep for Day 6.1 QNN sub-delegate |
| §4 Thermal onset deferred | MODERATE onset at t > 68 s (Day 6 baseline) | Note result; if onset unchanged, thermal source may be non-GPU (SoC core, camera ISP); result still valid as a Day 6.1 baseline improvement |
| §5 No alloc regression | GC pause pattern stable vs Day 6 | Memory Profiler spot-check on `processImage()` hot path |

---

## Files Modified

| File | Change |
|------|--------|
| `app/src/main/java/com/courtvision/spike/pipeline/GpuDelegateProbe.kt` | Add `buildSustainedSpeedGpuDelegate()` top-level function |
| `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` | GPU branch in `switchInterpreter()`; `yoloTensorImage` field; `setAllowBufferHandleOutput(true)` |
| `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt` | GPU delegate in `init` |

---

## Deferred Items

- **Gap 2 (GPU shader serialization)** — Day 6.3; upgrades `buildSustainedSpeedGpuDelegate()` to `GpuDelegateFactory.Options` + `setSerializationParams()`
- **Gap 6 (rotate-after-resize)** — requires CameraX config change; conflicts with ADR-003/ADR-005 rotation handling; dedicated spike
- **Full `setAllowBufferHandleOutput` benefit** — realised in Phase 2 when GPU-side NMS or chained GPU ops are introduced

---

## References

- [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md) — Gaps 3, 4, 5 with line citations
- [`docs/plans/day6-combined-pipeline.md`](./day6-combined-pipeline.md) — Day 6 baseline (thermal onset t=68 s, MODERATE)
- Qualcomm demo GPU options reference: `D:\ai-hub-apps\apps\_shared\android\tflite_helpers\TFLiteHelpers.java` lines 349–375
- CONTEXT.md §ML Pipeline Architecture: "No object allocation inside `ImageAnalysis.Analyzer.analyze()` — pre-allocate"
- `PoseLandmarkInterpreter.kt` lines 33–36, 73–74 — existing correct pre-alloc pattern to mirror
