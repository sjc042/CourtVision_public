# 🚀 Phase 0 Immediate Next Steps — Technical Spike Plan

> Consolidated spike plan — final version. Synthesized from PRD requirements, multi-session AI review (Claude + ChatGPT), and agreed architectural decision: single multi-class detector for ball + hoop.

---

## 🎯 Objective

Validate that real-time shot detection is achievable on Android using on-device ML — **before any product architecture is locked in.** Everything else (UI, AR mapping, pose, freemium) waits until this spike is green.

---

## 🧠 Key Architectural Decision (Resolved)

**Single multi-class YOLO detector** (YOLOv8n, 5 classes: `ball`, `made`, `person`, `rim`, `shoot`; alt: YOLO26n).

- One inference pass instead of two models
- `rim` provides a static spatial anchor for shot geometry without extra overhead
- `made` class provides a direct detector signal for shot outcome
- TFLite FP16, GPU delegate; supports both standard and end-to-end YOLO models
- Simpler debugging and performance profiling

> See [ADR-001: Single Multi-Class YOLO Model](decisions/001-single-yolo-model.md)

Core spike pipeline:

```
CameraX → YOLO (5-class) → Kalman tracker → shot state machine → metrics logger
```

---

## 📱 Target Hardware Baseline

> See [TDD §9.3](../docs/tdd.md#93-device-testing-matrix) for the canonical device gate matrix.

---

## 📅 8-Day Spike Plan

### Day 1–2 — CameraX Pipeline ✅

- Barebones Android project: Kotlin + CameraX
- Camera → frame buffer → inference loop scaffolding
- Log raw FPS with no ML attached (baseline)
- GPU delegate enabled, 720p input, 30fps target
- **Output:** Stable camera pipeline at 30fps on target device

> See [Day 1-2 Detailed Plan](plans/day1-2-plan.md) — **COMPLETED 2026-03-26**

### Day 3 — Single Detector ✅

- Export YOLOv8n to TFLite FP16 (COCO-pretrained for latency proxy)
- Plug into CameraX inference loop with GPU delegate, UI toggle for CPU vs GPU
- Added model selector UI, live bounding box overlay, end-to-end YOLO support, Bitmap preprocessing
- Measure: per-frame latency, FPS, RAM, thermal behavior over 10-min session
- **PRD target:** Ball detection < 100ms
- **Output:** Latency benchmarks committed to `/benchmarks/phase0/`

> See [Day 3 & Weekend Plan](plans/day3+weekend-plan.md) — **Day 3 COMPLETED 2026-03-28**

**Weekend training track (parallel, 2026-03-29 – 2026-03-30):**
- Dataset: 15,856 images, 5 classes (`ball`, `made`, `person`, `rim`, `shoot`), dedup completed
- YOLOv8n trained at 640/480/320: mAP50 = 0.940 / 0.935 / 0.895
- Remaining: yolov8s, yolov11n, yolov11s training matrix

### Day 4 — Ball Tracker (Kalman Filter) ✅

- Add Kalman filter tracker on top of ball bounding box output
- Track ball position and velocity across frames — prerequisite for shot arc and geometry
- Measure tracking stability and false positive rate
- **Output:** Smooth ball trajectory over a 10-shot sequence

### Day 5 — Pose Landmark Model: Isolated Validation ✅

> Architecture: [ADR-005 — Sequential GPU Inference Pipeline](decisions/005-sequential-gpu-inference-pipeline.md)

- Extract `pose_landmark_lite.tflite` from MediaPipe `.task` bundle; load as standalone TFLite `Interpreter` with `GpuDelegate`
- **Do not** use the MediaPipe `PoseLandmarker` task API — ADR-005 replaces its internal person detector with YOLO's `person` bbox
- Feed manually cropped 256×256 person images (static test frames, not live camera yet)
- Verify 33 `WorldLandmarks` output: correct joint positions against video ground truth (10 clips minimum)
- Benchmark isolated pose inference: per-frame latency, FPS, RAM on Pixel 6 and Galaxy A54
- Verify two `GpuDelegate` interpreters co-exist on `Dispatchers.Default.limitedParallelism(1)` — ADR-005 Verification Gate §1 (EGL context)
- **PRD target:** Pose inference < 50ms per frame
- **Output:** Pose latency log + EGL context verification result recorded in CONTEXT.md

### Day 6 — Sequential GPU Combined Pipeline

> Architecture: [ADR-005 — Sequential GPU Inference Pipeline](decisions/005-sequential-gpu-inference-pipeline.md)

- Integrate YOLO + pose as sequential stages on a single inference dispatcher (ADR-005 pipeline diagram)
- YOLO `person` bbox → `squarePadCrop(margin=1.25)` → 256×256 → pose landmark model — all on GPU, same thread
- Skip pose when FSM state is `IDLE` or `MADE` (no biomechanics needed — reduces GPU load)
- Pre-allocate all input/output `ByteBuffer`s at init; verify zero GC events during `analyzeFrame`
- Measure: combined YOLO + pose p95 latency, GPU/CPU contention, thermal throttling, RAM over 10-min session
- **PRD targets:** Combined p95 ≤ 50ms on Pixel 6, ≤ 80ms on A54; RAM < 400MB; no sustained thermal throttle
- **Fallback:** If A54 p95 > 80ms, fall back to pose on CPU (4 threads, every 3rd frame) — see ADR-005 Config D
- **Output:** Combined performance report
- **Pose gating:** `PoseGatingMode.EVERY_FRAME_WITH_PERSON` (Day 6 default); `FSM_GATED` wired in Day 7 — see [ADR-006](decisions/006-pose-gating-mode.md)

> **Note:** The original `frame-scheduling-spec.md` (Worker A / Worker B two-channel design) is
> superseded by ADR-005. See `frame-scheduling-spec.md` header for details.

### Day 7 — Shot Detection Logic (State Machine)

- Implement state machine: IDLE → PREP → RELEASE → FLIGHT → OUTCOME
- Use ball trajectory (Kalman tracker) + hoop bounding box (spatial anchor) for make/miss
- Use pose landmarks for FSM transitions: wrist-above-shoulder gate for RELEASE detection, knee flexion for PREP
- Ground Mode fallback: trajectory-only logic when hoop is not in frame
- Console/log output is sufficient — no UI polish needed
- **Output:** Shot counter logging make/miss across both Ground and Tripod scenarios

### Day 8 — Decision Checkpoint

- [ ] Ball detection latency < 100ms (p95 < 140ms)
- [ ] Pose inference latency < 50ms (p95 < 70ms)
- [ ] Combined YOLO + pose p95 ≤ 80ms on A54
- [ ] Combined RAM < 400MB sustained
- [ ] End-to-end FPS ≥ 20 sustained for 10 min (target 30)
- [ ] Shot detection stable across Tripod and Ground modes
- [ ] Two `GpuDelegate` interpreters verified stable on Pixel 6 and A54

**If green** → proceed to full architecture build (Phase 2)

**If red** → apply Config D fallback (pose on CPU), revisit model quantization or input resolution (416 or 320), or descope pose from MVP. NPU acceleration via QNN delegate is a Phase 2 optimization (see ADR-005 Deferred section) — not available during the spike.

---

## ⚡ PRD Hard Targets (Gate Criteria)

| Metric | Target |
| --- | --- |
| Shot Detection Accuracy | > 90% |
| Pose Inference Latency | < 50ms |
| Ball Detection Latency | < 100ms |
| App Cold Start | < 3 seconds |
| Battery Usage | < 15%/hr |
| RAM Footprint | < 400MB |

---

## 🧪 Formal Phase 0 Evaluation Protocol

### 1) Evaluation Dataset (Fixed for Gate Decision)

| Split | Definition | Count |
| --- | --- | --- |
| Dev Set | Used during implementation tuning only (not for final gate) | 30 clips (~300 shot attempts) |
| Gate Set | Locked holdout used for Day 8 go/no-go decision | 40 clips (~400 shot attempts) |

- Include both capture modes: 50% Tripod, 50% Ground
- Include lighting diversity: indoor gym, outdoor daytime, outdoor dusk
- Include player diversity: at least 6 distinct shooters
- Label schema per shot: release frame, make/miss, mode, lighting, device used

### 2) Metrics To Compute

| Area | Metric | How Measured |
| --- | --- | --- |
| Detection | Ball/Hoop detection quality | Precision/Recall on labeled frames (IoU >= 0.5) |
| Shot Events | Shot detection F1 | FSM event vs labeled shot attempts |
| Outcome | Make/Miss accuracy | Correct outcome classification per attempt |
| Latency | Pose + Detector latency | p50/p95 ms on-device telemetry |
| Runtime | FPS, RAM, thermal, battery | 10-min continuous session logs |

### 3) Pass/Fail Thresholds (Day 8 Gate)
- End-to-end FPS >= 20 sustained for 10 minutes (target 30)
- Ball detection latency p50 < 20ms (p95 < 60ms)
- Pose inference latency p50 < 20ms (p95 < 50ms)
- Shot detection F1 >= 0.90 on Gate Set
- Make/Miss accuracy >= 0.90 on Gate Set
- RAM < 400MB sustained
- No severe thermal throttling event during 10-minute run

### 4) Test Conditions (Must Be Reproducible)

- Devices: Samsung Galaxy S22+ (primary), Pixel 6 or Pixel 7 (mid-range cross-check), Xiaomi Redmi Note 12 or Galaxy A32 (low-end stretch)
- Resolution/FPS: 720p input, 30fps target
- Runtime protocol: 2-min warm-up + 10-min measured window
- Per-condition runs: minimum 3 runs per device per mode
- Environment log: lux estimate, indoor/outdoor flag, temperature band, capture distance/angle
- Report format: one CSV + one summary table per day, committed to repo under `/benchmarks/phase0/`

## 🏗️ Parallel: Repo Setup (Days 1–3)

- [x] Create GitHub repo
- [x] Kotlin + Jetpack Compose + MVVM scaffold
- [ ] GitHub Actions CI (build + lint)
- [ ] ktlint / detekt code style
- [ ] Branch protection rules

---

## 🔍 ARCore — Separate Validation Track

ARCore is required for the heatmap/court-mapping tier and is not universally supported. Validate independently — not part of this spike.

- [ ] Audit ARCore device compatibility against target user hardware
- [ ] Confirm outdoor ground plane detection viability (direct sunlight)
- [ ] Flag device coverage gap before Phase 3 architecture is locked

---

## 🚫 Explicitly Out of Scope for This Spike

UI design, wireframes, user accounts, cloud backend, AR court mapping, freemium/paywall, analytics pipeline, session history storage.

---

## 📋 Phase Roadmap (Post-Spike)

| Phase | Focus | Gate |
| --- | --- | --- |
| **Spike (now)** | Camera → detector → tracker → shot logic | — |
| **Phase 2** | Pose integration, release angle, jump height, state machine refinement | Spike green |
| **Phase 3** | AR court mapping, homography, shot heatmap | ARCore validation |
| **Phase 4** | Session history, coaching insights, drill mode | Phase 2 stable |
| **Phase 5** | Freemium paywall, export, share cards | Phase 3 stable |
