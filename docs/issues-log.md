# ❗ Issues Log

Tracking findings, risks, and issues discovered during development.
Issues are tracked on GitHub: [sjc042/Court-Vision Issues](https://github.com/sjc042/Court-Vision/issues)

---

## GitHub Issues Index

| GH# | Title | Priority | Milestone | Source | Status |
|-----|-------|----------|-----------|--------|--------|
| [#1](https://github.com/sjc042/Court-Vision/issues/1) | Fix double-counted drop counter in FrameProcessor | Critical | Day 3 | ISSUE-001 | ✅ Fixed (`4ab53f0`) |
| [#2](https://github.com/sjc042/Court-Vision/issues/2) | Add Phase 0 single-module override to CONTEXT.md | Critical | Day 3 | ISSUE-002 | ✅ Fixed (`de3490c`) |
| [#3](https://github.com/sjc042/Court-Vision/issues/3) | Define frame scheduling strategy for combined pipeline | Critical | Day 5 | ISSUE-003 + Day1 #4 | ✅ Spec written (`de3490c`) |
| [#4](https://github.com/sjc042/Court-Vision/issues/4) | Consolidate MVP scope to single canonical definition | High | Day 5 | ISSUE-004 + Day1 #2 | Open |
| [#5](https://github.com/sjc042/Court-Vision/issues/5) | Update TDD to reflect ADR-001 and remove stale references | High | Day 5 | ISSUE-005 + Day1 #1 | Open |
| [#6](https://github.com/sjc042/Court-Vision/issues/6) | Add acceptance criteria for US-13 through US-18 | High | Day 5 | ISSUE-006 + Day1 #6 | Open |
| [#7](https://github.com/sjc042/Court-Vision/issues/7) | Unify device test matrix across all docs | Medium | Day 5 | Day1 #5 | Open |
| [#8](https://github.com/sjc042/Court-Vision/issues/8) | Add FSM transition threshold table to TDD | Medium | Phase 2 | ISSUE-007 | Open |
| [#9](https://github.com/sjc042/Court-Vision/issues/9) | Add pipeline architecture diagram to TDD | Medium | Phase 2 | ISSUE-008 | Open |
| [#10](https://github.com/sjc042/Court-Vision/issues/10) | Define ARCore fallback UX and add US-09b | Medium | Phase 3 | ISSUE-009 + Day1 #7 | Open |
| [#11](https://github.com/sjc042/Court-Vision/issues/11) | Create privacy and data retention spec | Low | Pre-launch | ISSUE-010 + Day1 #8,#9 | Open |
| [#12](https://github.com/sjc042/Court-Vision/issues/12) | Incorrect bbox scale (FILL_CENTER letterboxing) | High | Day 4 | ISSUE-011 | ✅ Fixed |
| [#13](https://github.com/sjc042/Court-Vision/issues/13) | Incorrect class ID on first session start | Critical | Day 4 | ISSUE-012 | ✅ Fixed |
| — | Detection fails on 180° rotation (sensorLandscape) | High | Day 4 | ISSUE-013 | ✅ Fixed |

---

## Phase 0 Day 1 — 2026-03-25

**Reviewed Docs**

- [Project Overview](project-overview.md)
- [Phase 0 Spike Plan](phase0-spike-plan.md)
- [PRD](prd.md)
- [TDD](tdd.md)
- [User Stories](user-stories.md)

**Findings (Highest Risk First)**

1. **Conflicting core detection architecture**: Spike says single YOLOv8n (ball+hoop), TDD says SSD MobileNet ball detector, and TDD open questions still lean manual hoop anchoring. This will cause rework unless one approach is declared canonical.
2. **MVP scope is contradictory and too wide**: PRD/User Stories mark many P1 features as MVP (shot type, paywall, full analytics), while Spike says only shot feasibility first and defer most features. "MVP" needs one explicit definition.
3. **No measurable protocol for `>90% shot accuracy` gate**: there is no labeled validation set, annotation plan, or pass/fail rubric (precision/recall/F1 by mode). Current "stable" wording is not testable.
4. **Performance budgets are internally inconsistent**: target 30fps plus `<100ms` ball + `<50ms` pose per frame is not feasible if run every frame in one loop. Need explicit frame-skipping/async scheduling budget.
5. **Device strategy is inconsistent**: Spike baseline devices differ from TDD test matrix. You need one official gate matrix (min/mid/high + ARCore coverage).
6. **Execution planning gap**: user stories have empty story points, no owners, no sprint capacity, and no dated milestones beyond the 8-day spike.
7. **ARCore dependency risk is acknowledged but not scheduled as a hard gate**: fallback behavior exists in notes, but no acceptance criteria for non-ARCore devices in roadmap.
8. **Data retention conflicts with analytics goals**: auto-delete clips/session data vs long-term trend and coaching workflows is not reconciled.
9. **Privacy/compliance plan missing**: camera/video handling, retention consent, deletion controls, and policy requirements are not defined.

**What to fix first**

1. Lock one detection strategy and update PRD/TDD/Spike/User Stories to match.
2. Define MVP v1 in one table: `Must ship` / `Deferred`.
3. Add a formal Phase 0 evaluation protocol (dataset, metrics, thresholds, test conditions).
4. Publish one device-and-mode gate matrix with explicit pass criteria.
5. Add story points + owner + target sprint for all P0/P1 items.
6. Add a privacy/data-retention spec before implementation starts.


## Phase 0 Day 2 — 2026-03-26

**Reviewed Sources**
- All source files in `CourtVision_Android/app/src/main/java/`
- All docs in `docs/` including PRD, TDD, user stories, spike plan, ADR-001, dev workflow
- Cross-referenced with ChatGPT doc review (2026-03-26)

---

### ISSUE-001 | Drop counter double-counted in FrameProcessor

**Priority:** Critical
**Type:** Bug
**File:** `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
**Fix before:** Day 3

**Problem**

`droppedByOverflow` is incremented in two places simultaneously for the same overflow event:

1. In the `Channel` constructor via `onUndeliveredElement` callback
2. In the `submitFrame()` else branch when `trySend` fails

Both fire on the same failed send. Every dropped frame is counted twice, making the overlay and CSV metrics report double the actual drop rate.

**Relevant code**

```kotlin
// Location 1 — Channel constructor (FrameProcessor.kt ~line 32)
private val frameChannel = Channel<FramePacket>(
    capacity = 1,
    onBufferOverflow = BufferOverflow.DROP_OLDEST,
    onUndeliveredElement = { _ ->
        droppedByOverflow.incrementAndGet() // counts here
    }
)

// Location 2 — submitFrame() (FrameProcessor.kt ~line 56)
val result = frameChannel.trySend(frame)
if (result.isSuccess) {
    queueDepth.set(1)
} else {
    droppedByOverflow.incrementAndGet() // also counts here — double count
}
```

**Fix**

Remove the `droppedByOverflow.incrementAndGet()` from the `submitFrame()` else branch. Keep only the `onUndeliveredElement` callback as the single source of truth for overflow drops.

Note: `onUndeliveredElement` fires when `DROP_OLDEST` evicts an element that was already in the channel to make room. The `trySend` else branch fires when the send itself fails, which for a capacity-1 `DROP_OLDEST` channel should not happen in normal operation — the channel makes room before rejecting. Relying solely on `onUndeliveredElement` is correct.

**Corrected submitFrame()**

```kotlin
fun submitFrame(frame: FramePacket) {
    if (frameChannel.isClosedForSend) return
    val result = frameChannel.trySend(frame)
    if (result.isSuccess) {
        queueDepth.set(1)
    }
    // Drop counting handled exclusively by onUndeliveredElement
}
```

---

### ISSUE-002 | CONTEXT.md references multi-module structure that does not exist

**Priority:** Critical
**Type:** Documentation / AI tooling
**File:** `CONTEXT.md`
**Fix before:** Day 3

**Problem**

`CONTEXT.md` is the anchor file pasted at the start of every Codex and Gemini session. It lists a multi-module Gradle structure:

```
:app
:feature:capture
:feature:analytics
:feature:history
:core:ml
:core:ar
:core:data
:core:domain
:core:ui
```

The actual repo is a single-module app (`:app` only). An AI coder following `CONTEXT.md` literally will attempt to create modules, wire Hilt across module boundaries, and scaffold files in paths that do not exist. This will produce code that does not compile and will require manual cleanup before it can be tested.

**Fix**

Add a clearly marked Phase 0 override block directly below the document header, before any AI coder reads the module structure section. Do not delete the multi-module structure — it is the target architecture. Override it for the spike only.

**Insert after the "Active Phase" header in CONTEXT.md:**

```markdown
## ⚠️ Phase 0 Spike Override — Single Module

The module structure listed below under "Module Structure" is the **target architecture
for Phase 2+**. It does not exist yet.

For all Phase 0 work:
- The repo is a **single `:app` module** — no feature or core modules
- Do NOT create new Gradle modules
- Do NOT add Hilt — no DI framework in the spike
- Do NOT add Room — no database in the spike
- All code lives under `app/src/main/java/com/courtvision/spike/`
- Existing packages: `camera/`, `pipeline/`

When in doubt: keep it in `:app`. Module extraction happens at Phase 2 kickoff.
```

---

### ISSUE-003 | Frame scheduling strategy undefined — 30fps + 150ms combined latency is incoherent

**Priority:** Critical
**Type:** Architecture / Documentation
**Files:** `docs/tdd.md`, `docs/phase0-spike-plan.md`, `CONTEXT.md`
**Fix before:** Day 5 (before combined pipeline test on Day 6)

**Problem**

The PRD and TDD simultaneously target:
- 30fps analysis (33ms per frame budget)
- Ball detection < 100ms per frame
- Pose inference < 50ms per frame

If both models run synchronously on every frame, the minimum combined cost is 150ms — a hard ceiling of ~6fps, not 30fps. The Phase 0 spike plan acknowledges async scheduling is needed but provides no design. Day 6 (combined pipeline) has no specification to validate against, meaning there is no way to declare Day 6 a pass or fail.

**Fix**

Write a one-page frame scheduling spec before Day 5 begins. Add it as `docs/plans/frame-scheduling-spec.md` and reference it from the Day 6 plan. The spec must answer:

1. Do YOLO and MediaPipe share an analysis thread or run on separate executors?
2. Does pose run every frame, every Nth frame, or on a timer-gated schedule?
3. What is the frame-skip policy when inference is still in progress when the next frame arrives?
4. What defines "combined FPS" in the Day 8 gate — camera frames analysed per second, or inference completions per second?
5. At what point is the pipeline considered thermally throttled for gate purposes?

**Suggested starting point for the spec:**

```markdown
## Frame Scheduling Strategy (Draft — Phase 0)

Pipeline runs two parallel coroutine workers on Dispatchers.Default:
- Worker A: YOLO detector — every frame submitted via bounded channel (capacity=1, DROP_OLDEST)
- Worker B: MediaPipe pose — every 3rd frame (10fps target), gated by a frame counter

Frame budget:
- Camera feed: 30fps (33ms cadence)
- YOLO worker: 30fps target, 100ms budget, backpressure via DROP_OLDEST
- Pose worker: 10fps target, 50ms budget, sampled every 3rd frame

Gate definition:
- "Combined FPS" = YOLO inference completions per second (primary metric)
- Pose FPS reported separately
- Thermal throttle = sustained FPS drop >20% below baseline for >30s
```

Adjust the N-frame pose skip value based on Day 5 benchmark results before running Day 6.

---

### ISSUE-004 | MVP definition contradicts itself across PRD sections and user stories

**Priority:** High
**Type:** Documentation
**Files:** `docs/prd.md` (sections 3.1 and 3.2), `docs/user-stories.md` (delivery plan table)
**Fix before:** Day 5

**Problem**

Three documents define MVP scope differently and they do not agree:

| Feature | PRD 3.1 (feature table) | PRD 3.2 (MVP table) | User stories delivery plan |
|---|---|---|---|
| Shot type classification | MVP: Yes | Must Ship | Sprint 6 — implies late/deferred |
| Freemium paywall (US-24) | MVP: Yes | Deferred | P1, Sprint 6 |
| Cumulative heatmap (US-12) | — | Deferred | P2, no sprint |
| Vertical jump height | P0, MVP: Yes | Must Ship | — |
| Guided drills | P2, MVP: No | Deferred | P2, no sprint |

An AI coder reading PRD 3.1 and an AI coder reading PRD 3.2 will build different products.

**Fix**

1. Designate `docs/prd.md` Section 3.2 (the Must Ship / Deferred table) as the **single canonical MVP definition**.
2. Update PRD 3.1 to remove the `MVP?` column and replace it with a note: *"See Section 3.2 for MVP scope."*
3. Update `docs/user-stories.md` — add a `MVP?` column to the delivery plan table that references the 3.2 definition. Any story marked Deferred in 3.2 must be marked MVP: No in the delivery plan.
4. Add a one-line note at the top of `CONTEXT.md` under the project summary: *"MVP scope: see docs/prd.md Section 3.2 — this is the canonical definition."*

---

### ISSUE-005 | TDD contains pre-ADR-001 language and unresolved open questions

**Priority:** High
**Type:** Documentation
**File:** `docs/tdd.md`
**Fix before:** Day 5

**Problem**

ADR-001 was accepted on 2026-03-25 and locked the single multi-class YOLO decision. The TDD has not been updated to reflect this. Specific stale content:

- Section 3 tech stack table correctly lists YOLOv8n, but the surrounding prose in section 4 still implies the hoop detection strategy was under debate
- Section 10 (Open Technical Questions) still lists hoop detection as a question with an inline "Resolved:" note rather than a proper reference to ADR-001. Inline "Resolved:" comments in a planning table read as unresolved to an AI agent scanning the doc
- The phrase "optional manual rim-box fallback" appears in section 4.3 without clarifying it is a low-confidence edge case, not a primary path — this could cause an AI coder to implement it as a first-class feature

**Fix**

1. In Section 10, replace the hoop detection row with: *"Resolved — see ADR-001: Single Multi-Class YOLO Model (`docs/decisions/001-single-yolo-model.md`)"* and remove the inline resolution note.
2. In Section 4.3, add one sentence clarifying the manual fallback scope: *"Manual rim-box override is a low-confidence fallback only — it is not a primary path and is out of scope for Phase 0."*
3. Do a full-text search for "SSD MobileNet" and "manual hoop anchor" — remove or update any remaining references.

---

### ISSUE-006 | Acceptance criteria missing for US-13 through US-18

**Priority:** High
**Type:** Documentation
**File:** `docs/user-stories.md`
**Fix before:** Day 5

**Problem**

Six P1 user stories have story text and priority but no acceptance criteria. These stories cover core biomechanical metrics that feed directly into the ML pipeline design. Without AC, there is no testable definition of done and no way to validate the Day 8 gate for these features.

Stories affected:
- **US-13** — Release Speed
- **US-14** — Release Angle
- **US-15** — Knee Flexion Feedback
- **US-16** — Metrics in Both Modes
- **US-17** — Vertical Jump Height (P2, MVP: No — lower priority but still needs AC before Phase 2)
- **US-18** — Auto Session Save

**Fix**

Add acceptance criteria for each story. Draft AC below — review and adjust before using as sprint input:

**US-13 | Release Speed**
- [ ] Release speed computed per shot and visible in post-session review
- [ ] Displayed in mph and km/h (user-selectable or locale-based)
- [ ] Accuracy target: within ±2 mph of manual reference measurement on test clips
- [ ] Speed computed using ball centroid displacement + ARCore scale (tripod mode) or player height reference (ground mode)

**US-14 | Release Angle**
- [ ] Release angle computed from elbow-wrist landmark vector at detected release frame
- [ ] Displayed in degrees, rounded to one decimal place
- [ ] Feedback label shown: "Too flat" (<40°), "Optimal" (40°–55°), "Too steep" (>55°)
- [ ] Available in both capture modes
- [ ] Uses `pose_world_landmarks` (3D world space), not image-space coordinates

**US-15 | Knee Flexion Feedback**
- [ ] Knee angle measured at prep phase (0.2–0.5s before release, foot velocity ~0)
- [ ] Hip-knee-ankle angle computed and averaged across left and right
- [ ] Feedback label shown: "Deep bend" (<90°), "Optimal" (90°–120°), "Minimal bend" (>120°)
- [ ] Visible in post-session shot review per shot

**US-16 | Metrics in Both Modes**
- [ ] Release angle, knee flexion, and release speed all available in ground mode and tripod mode
- [ ] Ground mode shows estimation disclaimer for jump height only
- [ ] No metric is hidden or disabled based on capture mode alone

**US-18 | Auto Session Save**
- [ ] Session saved automatically on session end without user action
- [ ] Session record includes: date, duration, capture mode, total shots, makes, shooting %, avg release angle, avg knee angle
- [ ] Session visible in history list immediately after save
- [ ] Save succeeds even if app is backgrounded mid-session

---

### ISSUE-007 | Shot detection FSM missing measurable trigger thresholds

**Priority:** Medium
**Type:** Documentation / Architecture
**File:** `docs/tdd.md` (Section 4.4)
**Fix before:** Phase 2 kickoff

**Problem**

The FSM in TDD Section 4.4 defines states and transitions in prose but does not define the measurable trigger conditions needed to implement or test them. Without thresholds, two different AI coders will produce two different implementations and there is no spec to arbitrate between them.

Missing definitions:
- What pixel separation between ball bbox and wrist landmark counts as "ball leaves hand"?
- What confidence threshold gates a make classification?
- What temporal window is used for hoop intersection?
- What constitutes "ball ascending arc" — minimum frame count? Minimum Y displacement?
- What is the miss detection timeout — how long after FLIGHT before a non-intersection is declared a miss?

**Fix**

Expand TDD Section 4.4 with a threshold table. Add the following sub-section after the FSM state diagram:

```markdown
### 4.4.1 FSM Transition Thresholds (Phase 0 Starting Values — tune from benchmark data)

| Transition | Trigger condition | Starting threshold |
|---|---|---|
| IDLE → PREP | Knee angle delta | >15° flexion increase over 500ms window |
| PREP → RELEASE | Ball-wrist separation | Ball bbox centroid >40px from nearest wrist landmark (at 720p) |
| PREP → RELEASE | Wrist position | Shooting wrist Y-coordinate above shoulder Y-coordinate |
| RELEASE → FLIGHT | Ball trajectory direction | Ball centroid moving upward (negative Y delta) for ≥3 consecutive frames |
| FLIGHT → OUTCOME (make) | Hoop intersection | Ball bbox overlaps hoop bbox region for ≥2 frames within 2s of release, hoop confidence >0.6 |
| FLIGHT → OUTCOME (miss) | Timeout / floor | No intersection within 3s of release, OR ball detected at floor level (Y > 90% of frame height) |
| OUTCOME → IDLE | Cooldown | 1.5s after OUTCOME state entered |

All pixel thresholds assume 1280×720 analysis resolution. Scale proportionally if resolution changes.
```

---

### ISSUE-008 | Architecture pipeline diagram missing

**Priority:** Medium
**Type:** Documentation
**File:** `docs/tdd.md` or `docs/project-overview.md`
**Fix before:** Phase 2 kickoff

**Problem**

The TDD describes the pipeline architecture in text across multiple sections (2.1, 4.1–4.5) but there is no single visual diagram. A coder agent reading CONTEXT.md cannot quickly orient to how the layers connect. This becomes more important as the pipeline grows from Day 3 (YOLO) through Day 6 (combined).

**Fix**

Add a pipeline diagram to `docs/tdd.md` Section 2.1 showing the full data flow from camera to UI. The diagram should cover the Phase 0 spike pipeline and the target Phase 2+ pipeline side by side or as a before/after. Minimum content:

```
Phase 0 (spike):
CameraX (YUV 720p, 30fps)
  → ImageAnalysis.Analyzer
  → FrameProcessor (Channel, capacity=1, DROP_OLDEST)
  → YOLO TFLite FP16 (GPU delegate) [Day 3+]
  → Kalman tracker [Day 4+]
  → MediaPipe Pose [Day 5+]
  → Shot FSM [Day 7+]
  → PipelineStats (StateFlow)
  → CameraViewModel
  → CameraScreen (Compose overlay)
  → PerformanceCsvLogger
```

This can be added as a Mermaid flowchart in the markdown for easy diffing, or as an image generated separately.

---

### ISSUE-009 | ARCore fallback has no acceptance criteria

**Priority:** Medium
**Type:** Documentation
**File:** `docs/user-stories.md` (US-09), `docs/tdd.md` (Section 10)
**Fix before:** Phase 3 kickoff

**Problem**

US-09 (ARCore Floor Detection) only covers the happy path — ARCore initializes, plane detected, calibration proceeds. The TDD open questions table acknowledges a fallback exists ("use user height, court dimensions and orientation, and rim height as cues") but there are no acceptance criteria for:
- What features are disabled or hidden on non-ARCore devices
- What the user sees when ARCore is unavailable
- What the minimum viable experience is in tripod mode without ARCore

ARCore device coverage is not universal. If a meaningful percentage of target users are on non-ARCore devices and the fallback experience is undefined, this will generate support issues and negative reviews at launch.

**Fix**

Add a companion story US-09b to `docs/user-stories.md`:

```markdown
### US-09b | Non-ARCore Fallback Experience

> As a user whose device does not support ARCore, I want the app to gracefully
> disable spatial features and inform me clearly, so I can still use shot
> tracking and biomechanical metrics without a confusing error state.

**Priority:** P1
**Story Points:** 3
**MVP:** Yes

**Acceptance Criteria**
- [ ] App detects ARCore availability at launch via `ArCoreApk.checkAvailability()`
- [ ] If ARCore unavailable: heatmap, court mapping, and zone classification features
      are hidden from the UI (not shown as disabled/greyed — fully hidden)
- [ ] User sees a one-time informational banner: "Court heatmap requires ARCore.
      Shot tracking and form analysis are fully available."
- [ ] All biomechanical metrics, shot detection, and session history remain fully functional
- [ ] Non-ARCore state persisted in preferences — banner not shown on every launch
```

Also update TDD Section 10 to reference US-09b rather than the inline note.

---

### ISSUE-010 | No privacy and data retention spec

**Priority:** Low (required before Play Store submission)
**Type:** Documentation / Compliance
**File:** New file — `docs/privacy-spec.md`
**Fix before:** Play Store internal testing track submission

**Problem**

The issues log flagged this on Day 1. It remains unaddressed. Google Play requires a privacy policy for apps that access camera or microphone. The app currently:
- Accesses the camera continuously during sessions
- Writes video frame metadata to CSV on external storage
- Will store video clips in a future phase

There is no document defining what data is collected, where it lives, default retention policy, or deletion flow.

**Fix**

Create `docs/privacy-spec.md` with at minimum the following sections. This is not a legal privacy policy — it is an internal spec that will inform the public policy:

```markdown
# Privacy & Data Retention Spec

## Data collected
- Camera frames: processed in-memory only, never written to disk in Phase 0
- Performance metrics: written to CSV at external storage path (benchmarks only, no PII)
- Future phases: shot clips (+/- 3s per shot), session metadata, pose landmark sequences

## Data storage
- All data stored on-device only (no cloud in MVP)
- CSV benchmarks: `Android/data/com.courtvision.spike/files/benchmarks/`
- Future session clips: `Android/data/com.courtvision/files/sessions/{sessionId}/`

## Retention policy
- Benchmark CSVs: user-deletable, no auto-delete
- Session clips (future): auto-delete after 30 days (configurable: 7 / 14 / 30 / never)
- Session metadata (Room): retained until user deletes session

## Data leaving the device
- Phase 0–3: none
- Phase 4+ (optional cloud sync tier): session metadata and aggregated stats only,
  no raw video — user opt-in required

## Deletion flow
- User can delete individual sessions including all clips from session review screen
- User can delete all data from settings screen
- Uninstall removes all app-private storage automatically (Android platform guarantee)

## Play Store requirements
- Privacy policy URL required before production release
- DATA_SAFETY form: camera usage declared, no data shared with third parties (Phase 0–3)
```
---

### ISSUE-011 | Incorrect Bbox scale (FILL_CENTER letterboxing)

**Priority:** High (for spike testing visualization)
**Type:** Bug
**File:** `app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt`
**Fix before:** Day 4 (Kalman tracker)

**Problem**

Bounding boxes render with correct width but squished height. `DetectionOverlay` maps normalized [0,1] model coordinates to canvas pixels via simple multiplication (`normalized * canvasSize`), but ignores that `PreviewView.ScaleType.FILL_CENTER` crops the camera feed to fill the screen. The camera feed (9:16 in portrait after rotation) has a different aspect ratio than the screen (e.g. 9:20), so the Y-axis scaling is wrong.

**Root cause:** The normalized coords from the 640×640 model are correct (stretch cancels during normalization). The only issue is the canvas↔preview mapping — FILL_CENTER scales and center-crops, but the overlay assumed a 1:1 mapping from normalized space to screen space.

**Fix**

Replace simple `normalized * canvasSize` with FILL_CENTER-aware mapping in `DetectionOverlay`:
1. Compute effective source dimensions (swap W/H for 90°/270° rotation)
2. Compute `scale = max(canvasW/srcW, canvasH/srcH)`, then crop offsets
3. Map: `screenX = normalized * scaledW - offsetX`

**Status:** ✅ Fixed

---
### ISSUE-012 | Incorrect Class ID

**Priority:** Critical
**Type:** Bug
**File:** `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
**Fix before:** Day 4 (Kalman tracker)

**Problem**

On first session start, person boxes display as "ball" and random boxes appear with labels like "class32" or "class66" (gray/white, unmapped color).

**Root cause (dual):**
1. **COCO class count mismatch** (primary): `parseModelOutput` computes `numClasses = output[0].size - 4` dynamically. For COCO YOLOv8n, output shape is `[1, 84, 8400]` → `numClasses = 80`. Best class picked from all 80, but `CUSTOM_CLASS_NAMES` has 5 entries. COCO person (class 0) → "ball"; COCO sports ball (class 32) → "class32".
2. **Stale GPU buffers** (secondary): TFLite GPU delegate may leave uninitialized memory in output arrays on first frame. 
    - **Update**: Current pre-allocated buffer and default model is yolov8n with COCO class, when I first open the app, change model to 5-class YOLOv8n model, and use GPU to inference, 
       prediction boxes are not accurate and classes incorrect as if output by COCO YOLOv8n model; however, if loading via CPU for the first time, boxes are correct, 5 classes and accurate.


**Fix:**
1. **Implemented** Zero output buffers before each `interpreter.run()`
2. **Implemented** Filter `classId >= CUSTOM_CLASS_NAMES.size` in both `parseModelOutput` and `parseEndToEndOutput`

**Status:** ✅ Fixed

---

### ISSUE-013 | Detection fails on 180° phone rotation (sensorLandscape)

**Priority:** High
**Type:** Bug
**Files:** `app/src/main/java/com/courtvision/spike/camera/CameraScreen.kt`, `app/src/main/java/com/courtvision/spike/pipeline/FrameProcessor.kt`
**Fix before:** Day 4

**Problem**

When the phone is rotated 180° in `sensorLandscape` mode, YOLO detections fail (0 boxes returned) because the model receives an upside-down image. Returning to the original orientation restores correct detection.

**Root cause**

`ImageAnalysis.targetRotation` was never updated after initial camera bind. The activity is configured with `sensorLandscape` + `configChanges="orientation"`, so it does not recreate on 180° flips. CameraX defaults `targetRotation` to the display rotation at bind time and does not auto-track changes. As a result, `imageProxy.imageInfo.rotationDegrees` stays `0` even when the display rotates to `ROTATION_2` (180°), and no `Rot90Op` correction is applied before inference.

**Diagnosis**

Added `CV_Rotation` diagnostic logging at each pipeline step. Confirmed:
- Default orientation: `raw=0 normalized=0` → detections correct
- Rotated 180°: `raw=0 normalized=0` → model sees upside-down image, `boxCount=0`

**Fix**

Added `DisplayManager.DisplayListener` in `CameraPreview()` that updates `preview.targetRotation` and `analysis.targetRotation` whenever display rotation changes. Listener is unregistered in `DisposableEffect.onDispose` to avoid leaks.

**Known limitation — CameraX buffer lag (~1-2s)**

After a rotation change, there is a ~1-2 second delay before `rotationDegrees` updates in incoming frames. This is expected CameraX behavior:
- `DisplayListener.onDisplayChanged` fires ~100-300ms after rotation settles
- CameraX HAL capture pipeline buffers 3-8 frames with old `targetRotation` metadata
- Frames already queued retain the stale rotation value until drained

Net effect: a brief transition window where a few frames process with the old rotation. This does not affect steady-state correctness and is inherent to the CameraX architecture.

**Update (2026-04-02)**

Display-driven rotation callbacks were later found to be unreliable in GPU-mode rapid rotation scenarios. Canonical fix now uses `OrientationEventListener` as the source of truth for `targetRotation` updates, with runtime mismatch telemetry and guarded recovery:
- Reconcile `targetRotation` when expected vs frame rotation mismatch persists >=2s with `droppedFrames=0`
- Rebind `ImageAnalysis` only when mismatch persists >=5s

Decision captured in ADR-003: `docs/decisions/003-rotation-source-of-truth.md`.

**Status:** ✅ Fixed

---

