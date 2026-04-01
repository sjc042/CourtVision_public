# ADR-002: TFLite Support Library for Preprocessing

**Status:** Accepted
**Date:** 2026-03-30
**Context:** Phase 0 — Technical Spike (Post-Day 3)

---

## Decision

Use the TFLite Support Library (`ImageProcessor` with `ResizeOp` + `NormalizeOp`) for frame preprocessing in `FrameProcessor`, replacing the manual `fillInputTensorFromBitmap` implementation. Added `tensorflow-lite-support:0.4.4` as a dependency.

## Context

Day 3 shipped with a manual preprocessing path: `Bitmap.getPixels()` into a reusable `IntArray`, nearest-neighbor resize via integer division, per-pixel `/255f` normalization, and direct `ByteBuffer.putFloat()` writes. This approach gave full control but was slow and scaled with source resolution.

A benchmark was added to `PreprocessingLatencyTest` comparing the manual path against the TFLite Support Library path on a Samsung Galaxy S22+.

## Benchmark Results (p50, 2 runs averaged)

| Resolution | Manual | Support Library | Speedup |
|---|---|---|---|
| 640×480 | ~51ms | ~19ms | 2.7× |
| 1280×720 | ~52ms | ~19ms | 2.7× |
| 1920×1080 | ~58ms | ~19ms | 3.0× |

Key observations:
- Support Library is **resolution-independent** (~19ms regardless of input size)
- Manual path scales with source resolution and has wider p95 variance (54–96ms vs ~20ms)
- Support Library uses bilinear interpolation (better quality than manual nearest-neighbor)

## Rationale

- **~3× faster preprocessing** frees latency budget for inference and post-processing
- **Resolution-independent** — no regression risk if camera resolution changes
- **Bilinear resize** produces better input quality than nearest-neighbor
- **Less code to maintain** — removes `fillInputTensorFromBitmap`, `inputTensorBuffer`, `reusablePixels`, and `INPUT_SIZE_BYTES`
- `ImageProcessor` is built once and reused per frame — no per-frame allocation

## Trade-offs

- Adds `tensorflow-lite-support:0.4.4` dependency (~1.5MB AAR) — acceptable for a spike; evaluate bundle impact in Phase 2
- Less granular control over pixel processing — acceptable since the standard resize+normalize pipeline matches YOLO's expected input
- Support Library internally allocates a `TensorImage` per frame — lightweight compared to the manual `IntArray` bulk copy

## Consequences

- `FrameProcessor.processImage()` now uses `TensorImage.fromBitmap()` → `imageProcessor.process()` → `tensorImage.buffer` → `Interpreter.run()`
- `PreprocessingLatencyTest` benchmarks the Support Library path only; manual path tests removed
- Thread safety unchanged — `ImageProcessor` is stateless, `TensorImage` is created per frame on the consumer dispatcher
