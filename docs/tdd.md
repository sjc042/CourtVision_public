# 🔧 TDD — Technical Design Document

**CourtVision — Android Basketball AI Tracker**

Version 1.0 | Planning Stage | March 2026

---

## 1. Document Overview

| Field | Detail |
| --- | --- |
| **Project** | CourtVision — Android Basketball AI Tracker |
| **Primary Stack** | Kotlin, CameraX, MediaPipe / YOLO26n-pose, TensorFlow Lite (YOLOv8n / YOLO26n), ARCore |
| **Architecture Pattern** | MVVM + Clean Architecture (Repository Pattern) |
| **Target Latency** | <50ms pose inference, <100ms object detection per frame |

---

## 2. System Architecture

### 2.1 High-Level Architecture

CourtVision follows a Clean Architecture / MVVM pattern with these layers:

- **Presentation Layer** — Jetpack Compose UI, ViewModels, UI state management.
- **Domain Layer** — Use cases, business logic, shot detection algorithms, metric computation.
- **Data Layer** — Room DB (local sessions), file storage (clips), optional cloud sync repository.
- **ML / CV Layer** — CameraX feed, MediaPipe Pose (alt: YOLO26n-pose), single multi-class TFLite detector (5 classes: `ball`, `made`, `person`, `rim`, `shoot`; alt: YOLO26n), ARCore spatial anchors.

### 2.2 Module Structure

```
:app                  — Main entry, DI setup (Hilt), navigation
:feature:capture      — Camera session, mode switching, live overlay
:feature:analytics    — Heatmap, session review, shot timeline
:feature:history      — Session list, drill history, progress charts
:core:ml              — MediaPipe / YOLO26n-pose wrapper, TFLite YOLOv8n / YOLO26n multi-class detector (ball, made, person, rim, shoot)
:core:ar              — ARCore ground plane, homography, court mapper
:core:data            — Room entities, DAOs, Repository interfaces
:core:domain          — Use cases, models, ShotMetrics data classes
:core:ui              — Shared Compose components, theme, design tokens
```

---

## 3. Technology Stack

| Component | Technology / Library | Version / Notes |
| --- | --- | --- |
| Language | Kotlin | 1.9+ |
| UI Framework | Jetpack Compose | Latest stable |
| Architecture | MVVM + Clean Architecture | Android Architecture Components |
| DI Framework | Hilt (Dagger) | 2.x |
| Camera | CameraX | Jetpack — API 21+ |
| Pose Estimation | MediaPipe Pose Landmarker (alt: YOLO26n-pose) | 0.10.x — 33 landmarks, 30fps |
| Object Detection | TensorFlow Lite (YOLOv8n FP16) (alt: YOLO26n) | 5-class model: `ball`, `made`, `person`, `rim`, `shoot` |
| AR / Spatial | ARCore | Google — ground plane + anchors |
| Computer Vision | OpenCV Android | 4.x — corner/line detection, homography |
| Local Database | Room | Jetpack — session, shot, metric entities |
| Charts | MPAndroidChart | 3.1.x — heatmap, trend lines |
| Networking | Retrofit + OkHttp | Optional — cloud sync tier only |
| Testing | JUnit 5, MockK, Espresso | Unit, integration, UI tests |

---

## 4. Computer Vision Pipeline

### 4.1 Camera & Frame Processing

CameraX ImageAnalysis use case provides YUV_420_888 frames at 30fps to a processing pipeline. All analysis runs on a background thread pool via Kotlin coroutines.

- **Resolution:** 1920x1080 (ground-placed mode and landscape tripod mode).
- **Frame rate target:** 30fps analysis; display preview at 60fps uncoupled.
- CameraX binds to lifecycle; session cleanup handled automatically.

### 4.2 Pose Estimation (MediaPipe Pose Landmarker / YOLO26n-pose)

MediaPipe Pose Landmarker detects 33 body landmarks per frame. Alternative: YOLO26n-pose, which unifies object detection and pose estimation in a single inference pass. Key landmarks used:

- **Landmarks 15/16 (wrists)** — release point and angle computation.
- **Landmarks 13/14 (elbows)** — elbow flexion at release.
- **Landmarks 23/24 (hips)** — jump height displacement.
- **Landmarks 25/26 (knees) + 27/28 (ankles)** — knee flexion / leg angle.

Landmark world coordinates (normalized to body scale) are used for angle calculations to be camera-distance independent.

### 4.3 Ball + Hoop Detection Model (Canonical)

CourtVision uses a **single multi-class YOLOv8n detector** exported to TensorFlow Lite FP16 (alt: YOLO26n):

- **Classes (nc=5):** `ball`, `made`, `person`, `rim`, `shoot`
- **Input:** 640x640 (fallback to 480/320 when thermal or FPS targets are missed)
- **Output:** bounding boxes + confidence for all classes in one inference pass
- **Runtime:** GPU delegate primary, CPU fallback path required
- **Model variants:** supports both standard YOLO (external NMS) and end-to-end YOLO (NMS built-in)
- **Rationale:** one pass reduces integration complexity, keeps `rim` as a spatial anchor, and simplifies profiling
- **Post-process:** Kalman tracking on ball centroid + temporal smoothing for rim ROI stability
- **Note:** `made` class provides a direct detector signal for shot outcome — may simplify or supplement hoop-intersection logic in the FSM FLIGHT→OUTCOME transition

> See [ADR-001: Single Multi-Class YOLO Model](decisions/001-single-yolo-model.md) for the full decision record.

### 4.4 Shot Detection State Machine

Shot events detected by a finite state machine (FSM) tracking combined pose + ball signals:

```
IDLE -> PREP (knee bend detected, ball held)
     -> RELEASE (wrist above shoulder, ball leaves hand)
     -> FLIGHT (ball ascending arc)
     -> OUTCOME (make/miss via net detection or trajectory)
     -> IDLE
```

- Ball 'leaves hand' = tracked ball bbox separates from wrist landmark by >N pixels.
- **Make detection:** ball trajectory intersects the detected hoop region (temporal window + confidence gating).
- **Miss detection:** ball trajectory does not intersect hoop region, or ball detected on floor.

### 4.5 Shot Type Classification

Shot type inferred from player movement in the pre-release phase:

- **Catch & Shoot** — player stationary >0.5s before ball contact.
- **Pull-Up Jumper** — player moving toward basket, stops, shoots.
- **Step-Back** — player moves backward before shot prep.
- **Post Move** — player's back to basket, pivot/turn detected.
- **Off Dribble** — continuous dribble motion detected immediately before shot.

Classification uses hip and foot landmark velocity vectors over a 1-2 second window preceding release.

---

## 5. AR Court Mapping & Heatmap

### 5.1 Ground Plane Detection (ARCore)

- ARCore PlaneDetectionMode.HORIZONTAL detects the court floor plane.
- App prompts user to slowly pan camera across the floor to initialize tracking.
- Once plane detected with sufficient confidence, user can tap to anchor court corners.
- ARCore world coordinates provide real-world scale (meters) for jump height and speed calibration.

### 5.2 Court Corner Anchoring & Homography

- User taps four court reference points on screen (e.g., three-point arc corners, key corners).
- Use corner and line detection; sample corner points to known court coordinates (US high school standard: 84ft x 50ft halved, 10-foot rim height).
- Computes homography matrix H using OpenCV findHomography() (RANSAC for robustness).
- Stores court homography while in session.
- Prompts recalibration if camera moves significantly.
- Each shot's release-point image coords are transformed: `courtPt = H * imgPt` (perspective transform).
- Result: normalized (x, y) on court plane, ready for heatmap overlay.

### 5.3 Heatmap Rendering

- Heatmap uses 2D Kernel Density Estimation (KDE) over accumulated court coordinates.
- Gaussian kernel with bandwidth auto-set by Silverman's rule.
- Density grid rendered onto standard half-court SVG overlay.
- Color: blue (low density) → yellow → red (high density).
- Separate layers for makes and misses; togglable in the UI.
- Minimum 5 shots required to render a meaningful heatmap.

---

## 6. Biomechanical Metric Computation

### 6.1 Release Ball Speed

- Ball centroid tracked over frames F_n-2 to F_n+2 around detected release frame.
- Pixel displacement converted to meters/second using ARCore ground plane scale and player height.
- Average over 3-frame window to reduce noise. Displayed as mph or km/h.

### 6.2 Release Time

- Time delta between Prep state and Release state timestamps.

### 6.3 Release Angle

- At the release frame: extract 3D world-space coordinates of Elbow (E) and Wrist (W) landmarks from MediaPipe's pose_world_landmarks.
- Forearm direction vector: `V = (Wx − Ex, Wy − Ey, Wz − Ez)`
- Release angle: `θ = atan2(Ey − Wy, √((Wx − Ex)² + (Wz − Ez)²))`
- Threshold feedback: θ < 40° = Too flat | 40°-55° = Optimal | θ > 55° = Too steep.

### 6.4 Leg Angle (Knee Flexion)

- Measure at 'prep phase' — 0.2-0.5s before release, when foot velocity ~0.
- Hip (H), Knee (K), Ankle (A) landmarks: `angle = acos(dot(KH, KA) / (|KH| * |KA|))`
- Average left and right knee angles.
- Feedback: <90 deg = deep bend | 90-120 deg = optimal | >120 deg = minimal bend.

### 6.5 Vertical Jump Height

- Using user height as reference for hip Y-coordinate (sum of joint lengths ≈ height).
- Track hip midpoint Y-coordinate (landmark 23/24 average) from takeoff to apex.
- Height in meters = `(apex_local_Y - takeoff_local_Y)` using user local height coordinates.
- Tripod mode is more accurate; ground mode shows output as estimation.

---

## 7. Data Models & Storage

### 7.1 Core Data Entities (Room)

```kotlin
Session: id, date, durationMs, captureMode, courtType, totalShots, madeShots, clips
Shot: id, sessionId, timestampMs, made, shotType, courtX, courtY,
      releaseSpeedMph, releaseAngleDeg, legAngleDeg, jumpHeightM
DrillResult: id, sessionId, drillType, targetZone, completionRate, avgReleaseSpeed
```

### 7.2 File Storage

- Session clips stored in app-private external storage: `/Movies/CourtVision/{sessionId}/`
- Clips auto-trimmed to +/- 3s around each shot event.
- User configurable auto-delete policy (7, 14, 30 days or never).
- Heatmap images cached as PNGs per session.

---

## 8. Performance Targets

> For Day 6 combined pipeline pass/fail criteria, see [Frame Scheduling Spec](plans/frame-scheduling-spec.md#day-6-passfail-criteria).

| Metric | Target |
| --- | --- |
| Pose inference latency | < 50ms per frame (GPU delegate) |
| Ball detection latency | < 100ms per frame |
| Shot detection delay | < 200ms from release to event fired |
| Heatmap render time | < 500ms for up to 500 shot datapoints |
| App startup (cold) | < 3 seconds to camera live view |
| Battery usage | < 15% per hour of active session |
| RAM footprint | < 400MB active session |

---

## 9. Testing Strategy

### 9.1 Unit Tests

- Shot detection FSM state transitions with mock landmark sequences.
- Homography computation with known test point sets.
- Biomechanical angle calculations with synthetic landmark data.
- Room DAO queries with in-memory database.

### 9.2 Integration Tests

- Full ML pipeline on pre-recorded video clips with known shot counts.
- ARCore ground plane detection on controlled test environment.
- End-to-end: video clip → shot events → DB storage → heatmap render.

### 9.3 Device Testing Matrix

| Device | API | Purpose |
| --- | --- | --- |
| Samsung Galaxy S22+ | API 33 | Primary testing device, Samsung One UI |
| Pixel 7 | API 33 | Stock Android baseline |
| Pixel 5a | API 31 | Mid-range performance baseline |
| Xiaomi Redmi Note 12 | API 32 | Low-RAM performance testing |

---

## 10. Open Technical Questions

| Question | Resolution |
| --- | --- |
| Hoop detection strategy: trained model vs. manual anchor? | **Resolved:** see [ADR-001](decisions/001-single-yolo-model.md) — single multi-class YOLO model detects hoop directly; optional manual rim-box fallback only when hoop confidence is persistently low |
| OpenCV dependency size (~40MB AAR) acceptable? | **Acceptable for MVP, explore alternatives in later versions** |
| ARCore availability fallback for tripod mode? | **Yes, use user height, court dimensions and orientation, and rim height as cues** |
| Ball model training: Roboflow public dataset vs. own data? | **Use public Roboflow dataset first; fine-tune on own data later if needed** |
| GPU delegate availability: define CPU fallback performance floor? | **TBD — needs benchmarking on Xiaomi Redmi Note 12** |
