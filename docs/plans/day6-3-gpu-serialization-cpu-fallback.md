# Plan: Day 6.3 — GPU Shader Serialization + CPU XNNPack Fallback (Gaps 2, 7)

**Status:** 🔲 TODO — Not started  
**Created:** 2026-04-27  
**Bridges:** Day 6.2 (sustained-speed GPU baseline) → Day 7 / Phase 2 fallback cleanup  
**Gaps addressed:** Gap 2 (GPU shader serialization), Gap 7 (NNAPI → XNNPack CPU fallback)  
**New dependency:** `org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4`

---

## Context

Two low-effort fixes remain after the Day 6.2 trivial wins:

- **Gap 2** removes multi-second cold-start GPU shader compilation by caching compiled kernels to disk.
- **Gap 7** replaces the legacy NNAPI path with explicit XNNPack CPU fallback and deprecates NNAPI selection on Qualcomm.

These do not target the primary Day 6 steady-state bottleneck as directly as Day 6.1 QNN does, but they tighten the non-QNN paths in two important ways:

1. **Cold-start UX**: first GPU mode switch no longer pays the full OpenCL/Vulkan compile cost on every process start.
2. **Fallback quality**: the CPU path becomes explicit and modern (`XNNPACK`), while `NNAPI` stops pretending to be an NPU path on Snapdragon devices.

**Gap analysis:** [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md) Gaps 2 and 7  
**Day 6 baseline:** [`docs/plans/day6-combined-pipeline.md`](./day6-combined-pipeline.md) — YOLO p50 = 53 ms, `frame_total_ms` p95 = 129 ms, FPS median = 8, thermal MODERATE onset t=68 s  
**Dependency on Day 6.2:** `buildSustainedSpeedGpuDelegate()` helper exists and becomes the upgrade point for serialization

---

## Branch

`perf/day6-x-post-day6-perf-follow-up` (shared workstream branch from `main`; execute Day 6.2 → Day 6.1 → Day 6.3 in order)

---

## Prerequisites

- Day 6.2 merged or available to cherry-pick locally, because Day 6.3 upgrades the shared GPU delegate helper introduced there
- S22+ (SM-S906U1) available for cold-start timing verification
- User available to run build, tests, and install steps locally per repo constraints

If Day 6.1 has not landed yet, Day 6.3 must independently add `modelCacheDir: String? = null` plumbing where needed for serialization.

---

## Design Decisions

### DD-1 — Upgrade the shared GPU helper instead of forking GPU delegate creation

Day 6.2 introduces `buildSustainedSpeedGpuDelegate()` as the canonical place to build GPU delegates. Day 6.3 upgrades that helper to prefer `GpuDelegateFactory.Options` with serialization support, while keeping a fallback to plain `GpuDelegate.Options`.

Rationale: one construction path keeps GPU configuration coherent across YOLO and pose, and reduces the chance of Day 6.2 / Day 6.3 drift.

### DD-2 — Serialization is opportunistic, not mandatory

`setSerializationParams(cacheDir, modelToken)` should be used only when a non-null cache directory is available. If `cacheDir == null`, the delegate still works; it just recompiles shaders on each cold start.

Rationale: cache absence is a performance miss, not a functional failure.

### DD-3 — Use stable model tokens, not ad hoc strings

The serialization cache key should use the same model-identity strategy as Day 6.1:

- Prefer MD5 of model bytes if the helper already exists
- Accept model filename-derived token only as a fallback

Rationale: shader caches must invalidate cleanly when the model changes.

### DD-4 — Apply serialization to pose as well as YOLO

`PoseLandmarkInterpreter` creates its own GPU delegate and therefore pays its own cold-start compile cost. It should use the same upgraded helper and its own model token.

Rationale: a half-migrated GPU path still leaves cold-start stalls in the session.

### DD-5 — Keep `InferenceMode.NNAPI`, but make it a compatibility alias

Do not remove `InferenceMode.NNAPI` in this plan. Instead:

- keep the enum value
- keep UI/tests/CSV compatibility
- route the implementation to GPU with an explicit warning

Rationale: this is the lowest-risk cleanup. Removing the enum is a separate product-facing change.

### DD-6 — CPU branch must state intent explicitly

Even if XNNPack is enabled by default in current TFLite, the CPU branch should set:

```kotlin
options.setUseXNNPACK(true)
options.setUseNNAPI(false)
```

Rationale: fallback behavior should be obvious in code review and stable across TFLite default changes.

### DD-7 — Do not over-promise steady-state gains from serialization

Gap 2 improves startup latency, not sustained inference throughput. Day 6.3 should be evaluated primarily on:

- cold-start GPU switch time
- absence of GPU steady-state regressions
- CPU fallback correctness

Rationale: otherwise the benchmark narrative becomes misleading relative to the actual mechanism.

---

## Files

### Modify

| File | Change |
|------|--------|
| `app/build.gradle.kts` | Add `tensorflow-lite-gpu-delegate-plugin:0.4.4` ⚠ new dep |
| `app/src/main/java/com/courtvision/spike/pipeline/GpuDelegateProbe.kt` | Upgrade `buildSustainedSpeedGpuDelegate()` to `GpuDelegateFactory.Options` + serialization params |
| `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt` | Pass cache params/model token to GPU helper; update CPU branch to explicit XNNPack; route NNAPI branch to GPU |
| `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt` | Use upgraded GPU helper with serialization support |
| `app/src/test/java/com/courtvision/spike/pipeline/FrameProcessorTest.kt` | CPU/XNNPack + NNAPI-forwarding coverage |
| `CONTEXT.md` | Add Day 6.3 cold-start / fallback result note after user-run validation |
| `docs/decisions/005-sequential-gpu-inference-pipeline.md` | Mark NNAPI as legacy compatibility path if this plan lands |

### Conditional modify

| File | Change |
|------|--------|
| `app/src/main/java/com/courtvision/spike/camera/CameraViewModel.kt` | If Day 6.1 is not merged, inject `modelCacheDir` into `FrameProcessor` here |
| `app/src/main/java/com/courtvision/spike/pipeline/FrameContracts.kt` | No planned change, unless comments around NNAPI mode need clarification |

---

## Gap 2 — GPU Shader Serialization (`setSerializationParams`)

### Problem

Current GPU delegate creation uses the direct `GpuDelegate(...)` API, which recompiles OpenCL/Vulkan kernels on each cold start. For YOLO 640×640 this can cost roughly 500 ms to 4 s on first use, producing:

- slow first GPU mode switch
- slow first app launch into a GPU-backed pipeline
- noisy first-frame timing data

The Qualcomm demo uses `GpuDelegateFactory.Options.setSerializationParams(cacheDir, modelToken)` so the compiled kernels are written to disk and reused on later launches.

### Fix

Add the plugin dependency:

```kotlin
implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
```

Upgrade the Day 6.2 helper:

```kotlin
private fun buildSustainedSpeedGpuDelegate(
    cacheDir: String?,
    modelToken: String
): GpuDelegate {
    return try {
        val opts = GpuDelegateFactory.Options().apply {
            setInferencePreference(
                GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
            )
            setPrecisionLossAllowed(true)
            if (cacheDir != null) {
                setSerializationParams(cacheDir, modelToken)
            }
        }
        GpuDelegateFactory.create(opts) as GpuDelegate
    } catch (_: Throwable) {
        GpuDelegate(
            GpuDelegate.Options().apply {
                inferencePreference =
                    GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                isPrecisionLossAllowed = true
            }
        )
    }
}
```

### Notes

- The fallback branch is required because the plugin path may fail at runtime or be unavailable in test environments.
- Serialization should be applied to both YOLO and pose, because each delegate compiles its own shaders.
- A missing cache entry on first launch is expected behavior, not an error.

### Files changed

- `app/build.gradle.kts`
- `app/src/main/java/com/courtvision/spike/pipeline/GpuDelegateProbe.kt`
- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
- `app/src/main/java/com/courtvision/spike/pipeline/PoseLandmarkInterpreter.kt`

---

## Gap 7 — NNAPI → XNNPack CPU Fallback

### Problem

`InferenceMode.NNAPI` currently implies a hardware acceleration path that is misleading on Snapdragon:

- Android NNAPI is deprecated in Android 15
- on Qualcomm devices it may route to CPU/GPU through the Android HAL
- it is not the same as the Hexagon QNN path from Day 6.1

The CPU branch also does not explicitly document that XNNPack is the intended fallback engine.

### Fix

#### CPU branch

```kotlin
InferenceMode.CPU -> {
    options.setNumThreads(4)
    options.setUseXNNPACK(true)
    options.setUseNNAPI(false)
}
```

#### NNAPI branch

```kotlin
InferenceMode.NNAPI -> {
    _lastError.value = "NNAPI mode is deprecated; using GPU delegate instead"
    return switchInterpreter(InferenceMode.GPU)
}
```

### Notes

- This preserves backward compatibility for the enum, UI mode list, and any existing CSV/test expectations.
- If a future cleanup removes `InferenceMode.NNAPI`, that should be a separate PR with user-visible behavior review.
- If `NnApiDelegateProbe.kt` still exists for diagnostics, its code may remain for now even though runtime selection no longer uses the delegate.

### Files changed

- `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
- `app/build.gradle.kts` only if cleanup of unused imports/dependencies becomes safe

---

## Step-by-Step Implementation

---

### Step 0 — Add GPU delegate plugin dependency

**Status:** 🔲 TODO

Add to `app/build.gradle.kts`:

```kotlin
implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
```

This is a **new dependency** and must be flagged in the PR description per repo rules.

---

### Step 1 — Upgrade `buildSustainedSpeedGpuDelegate()` in `GpuDelegateProbe.kt`

**Status:** 🔲 TODO

Replace the Day 6.2 direct `GpuDelegate.Options` implementation with a layered version:

1. prefer `GpuDelegateFactory.Options`
2. set sustained-speed preference
3. enable precision loss
4. set serialization params when `cacheDir` is available
5. fall back to direct `GpuDelegate.Options` on failure

If the helper currently takes no arguments, change its signature to accept:

```kotlin
cacheDir: String?,
modelToken: String
```

---

### Step 2 — Update `FrameProcessor.kt` GPU caller

**Status:** 🔲 TODO

In the `InferenceMode.GPU` branch of `switchInterpreter()`:

- compute or reuse the YOLO model token
- pass `modelCacheDir` and token into `buildSustainedSpeedGpuDelegate(...)`
- keep existing `CompatibilityList().isDelegateSupportedOnThisDevice` guard behavior unchanged

If Day 6.1 has already landed, reuse its `computeModelMd5()` helper. If not, add the smallest local equivalent needed for token stability.

Example shape:

```kotlin
localGpuDelegate = buildSustainedSpeedGpuDelegate(
    cacheDir = modelCacheDir,
    modelToken = computeModelMd5()
).also { options.addDelegate(it) }
```

---

### Step 3 — Update `PoseLandmarkInterpreter.kt` GPU caller

**Status:** 🔲 TODO

Replace direct `GpuDelegate(...)` construction with the upgraded helper and pass:

- cache directory
- stable pose model token

If the class currently lacks cache-dir access, choose one of two paths:

1. inject `modelCacheDir: String?` into the constructor, or
2. provide a narrower cache-dir string from the caller

Keep the change minimal and consistent with existing constructor boundaries.

---

### Step 4 — Wire cache-dir plumbing if Day 6.1 is not merged

**Status:** 🔲 TODO

If `FrameProcessor` does not yet accept `modelCacheDir`, add:

```kotlin
private val modelCacheDir: String? = null
```

Then pass `application.cacheDir.absolutePath` from `CameraViewModel`.

This step is conditional. Skip it if Day 6.1 has already provided the plumbing.

---

### Step 5 — Update CPU branch to explicit XNNPack

**Status:** 🔲 TODO

In `FrameProcessor.kt`, update `InferenceMode.CPU`:

```kotlin
InferenceMode.CPU -> {
    options.setNumThreads(4)
    options.setUseXNNPACK(true)
    options.setUseNNAPI(false)
}
```

This is behavior clarification more than a large functional change, but it makes fallback intent explicit.

---

### Step 6 — Update NNAPI branch to forward to GPU

**Status:** 🔲 TODO

In `FrameProcessor.kt`, replace direct `NnApiDelegate()` runtime usage with:

```kotlin
InferenceMode.NNAPI -> {
    _lastError.value = "NNAPI mode is deprecated; using GPU delegate instead"
    return switchInterpreter(InferenceMode.GPU)
}
```

This preserves mode selection compatibility while preventing the runtime from taking the legacy path.

---

### Step 7 — Tests

**Status:** 🔲 TODO

Add or update tests in `FrameProcessorTest.kt`:

- **`switchInterpreter_cpu_setsXnnPack()`** — assert CPU branch explicitly enables XNNPack and disables NNAPI
- **`switchInterpreter_nnapi_delegatesToGpu()`** — selecting `NNAPI` routes into the GPU branch and sets the warning message
- **`buildSustainedSpeedGpuDelegate_usesFallbackWhenFactoryUnavailable()`** — plugin path failure still produces a delegate through the direct-options fallback
- **`switchInterpreter_gpu_passesSerializationParams_whenCacheDirProvided()`** — verify non-null cache dir/token are passed through the helper via a test seam

Because `GpuDelegate` and factory-backed delegates are JNI classes, prefer seam-based testing over brittle reflection into native objects.

---

### Step 8 — Cold-start verification benchmark (user-run)

**Status:** 🔲 TODO

Goal: measure cold-start improvement, not steady-state throughput.

1. Install APK with Day 6.3 changes
2. Clear app storage or install fresh APK
3. Launch app, switch to GPU mode, time first switch
4. Kill app process completely
5. Relaunch app, switch to GPU mode again, time second switch
6. Compare first vs second launch timings

Expected pattern:

- launch 1: cache miss, slower
- launch 2: cache hit, materially faster

If possible, confirm via logcat that the cache is being read/written.

---

### Step 9 — CPU / NNAPI behavior verification (user-run)

**Status:** 🔲 TODO

Manual checks:

1. Select `CPU` mode and confirm inference still runs
2. Select `NNAPI` mode and confirm UI remains functional
3. Pull CSV or inspect debug output to verify runtime path is GPU when `NNAPI` is selected
4. Confirm warning string is visible or logged for the NNAPI alias path

This verifies that backward compatibility is preserved while runtime behavior is corrected.

---

### Step 10 — Documentation updates after validation

**Status:** 🔲 TODO

After user-run validation:

- update `CONTEXT.md` with a short Day 6.3 result note
- update ADR-005 deferred/runtime notes if the NNAPI compatibility alias becomes the accepted direction

No `TASKS.md` references should be added to this plan.

---

## Verification Gate Summary

| Gate | Pass condition | Fail action |
|------|---------------|-------------|
| §1 Build | `./gradlew :app:assembleDebug` green with plugin dep added | Fix missing import or version mismatch for `GpuDelegateFactory` |
| §2 Unit tests | All green; CPU and NNAPI forwarding tests updated | Fix before APK install |
| §3 Serialization wiring | Non-null cache dir/token reach the helper when available | Verify constructor plumbing and model-token source |
| §4 Cold-start improvement | 2nd GPU mode switch < 500 ms and materially faster than 1st switch | Check cache-dir path, token stability, and logcat for cache misses |
| §5 No steady-state GPU regression | YOLO p50 within ± 5 ms of Day 6.2 baseline | Revert helper to direct `GpuDelegate.Options` if plugin path regresses runtime |
| §6 CPU fallback clarity | CPU mode runs with explicit XNNPack path | Inspect `switchInterpreter()` configuration and logs |
| §7 NNAPI compatibility alias | Selecting NNAPI results in GPU runtime path and warning message | Debug branch forwarding and any stale UI / CSV assumptions |

---

## Deferred Items

- **Remove `InferenceMode.NNAPI` entirely** — separate cleanup PR; product-facing behavior change
- **Unify GPU helper signatures across YOLO and pose** if Day 6.1 / Day 6.3 land out of order
- **Cache instrumentation in UI/debug overlay** — useful but not required for Day 6.3
- **Non-Qualcomm cold-start validation** — nice to have, not required for S22+ target validation
- **Gap 6 (rotate-after-resize)** — separate spike; still conflicts with existing rotation handling decisions

---

## References

- [`research-reports/Claude-qualcomm-demo-gap-analysis.md`](../../research-reports/Claude-qualcomm-demo-gap-analysis.md) — Gaps 2 and 7
- [`docs/plans/day6-combined-pipeline.md`](./day6-combined-pipeline.md) — Day 6 baseline
- [`docs/plans/day6-2-trivial-perf-wins.md`](./day6-2-trivial-perf-wins.md) — source of the shared GPU helper upgraded here
- Qualcomm demo GPU serialization reference: `D:\ai-hub-apps\apps\_shared\android\tflite_helpers\TFLiteHelpers.java` lines 349–375
- Qualcomm demo CPU/XNNPack reference: `D:\ai-hub-apps\apps\_shared\android\tflite_helpers\TFLiteHelpers.java` lines 179–185
- TFLite GPU delegate docs: `GpuDelegateFactory.Options.setSerializationParams(...)`
