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

| Tier | Device |
| --- | --- |
| **Primary testing device** | Samsung Galaxy S22+ |
| **Mid-range (cross-check)** | Pixel 6 or Samsung Galaxy A54 |
| **Low-end (stretch validation)** | Samsung Galaxy A32 |
| **High-end (ceiling check)** | Samsung Galaxy S23 |

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

### Day 4 — Ball Tracker (Kalman Filter)

- Add Kalman filter tracker on top of ball bounding box output
- Track ball position and velocity across frames — prerequisite for shot arc and geometry
- Measure tracking stability and false positive rate
- **Output:** Smooth ball trajectory over a 10-shot sequence

### Day 5 — Pose Estimation Isolated (MediaPipe Pose / YOLO26n-pose)

- Add MediaPipe Pose (or YOLO26n-pose as alternative) in a separate branch — not combined yet
- Same benchmarks: FPS, per-frame latency, RAM
- **PRD target:** Pose inference < 50ms
- **Output:** Pose latency log on same device
- **Scheduling spec:** [Frame Scheduling Spec](plans/frame-scheduling-spec.md) — must be complete before Day 5

### Day 6 — Combined Pipeline

- Run YOLO detector + Kalman tracker + Pose estimator (MediaPipe Pose or YOLO26n-pose) simultaneously
- Measure: GPU/CPU contention, thermal throttling, RAM over 10-min session
- **PRD targets:** RAM < 400MB, no sustained thermal throttle
- **Output:** Combined performance report

### Day 7 — Shot Detection Logic (State Machine)

- Implement state machine: IDLE → PREP → RELEASE → FLIGHT → OUTCOME
- Use ball trajectory (tracker) + hoop bounding box (spatial anchor) for make/miss
- Ground Mode fallback: trajectory-only logic when hoop is not in frame
- Console/log output is sufficient — no UI polish needed
- **Output:** Shot counter logging make/miss across both Ground and Tripod scenarios

### Day 8 — Decision Checkpoint

- [ ] Ball detection latency < 100ms
- [ ] Pose inference latency < 50ms
- [ ] Combined RAM < 400MB
- [ ] End-to-end FPS ≥ 20 (target 30, red flag below 20)
- [ ] Shot detection stable across both capture modes

**If green** → proceed to full architecture build (Phase 2)

**If red** → revisit model quantization, input resolution (try 416 or 320), delegate strategy, or descope pose from MVP

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

- Devices: Pixel 6 (primary), Galaxy A54 (mid-range cross-check), Galaxy A32 (low-end stretch)
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
