# 📄 PRD — Product Requirements Document

**CourtVision — Android Basketball AI Tracker**

Version 1.0 | Planning / Pre-Dev Stage | March 2026

---

## 1. Product Overview

| Field | Details |
| --- | --- |
| **Project Name** | CourtVision |
| **Platform** | Android (API 26+) |
| **Stage** | Planning / Pre-Development |
| **Version** | 1.0 |
| **Date** | March 2026 |
| **Comparable Product** | HomeCourt (iOS) |

### 1.1 Vision Statement

CourtVision is an Android-native basketball performance tracking app that uses on-device AI, computer vision, and AR-assisted court mapping to deliver real-time shot analytics, biomechanical metrics, and player development insights — regardless of whether the device is placed on the ground or mounted on a tripod.

### 1.2 Problem Statement

- No HomeCourt equivalent exists on Android.
- Coaches and players lack affordable, mobile-first tools for shot tracking with spatial context (court heatmaps).
- Existing tools require expensive hardware or manual data entry.
- No solution combines AR court mapping + biomechanics (release angle, leg angle, jump height) in one app on Android.

### 1.3 Target Users

- Amateur and semi-professional basketball players seeking self-coaching tools.
- Youth coaches tracking player development across sessions.
- Personal trainers integrating shooting mechanics into athlete development.
- College players without access to professional analytics staff.

---

## 2. Goals & Success Metrics

### 2.1 Business Goals

- Launch MVP on Google Play within 12 months.
- Achieve 4.3+ star rating in first 90 days post-launch.
- Target 10,000 installs within 6 months of launch.
- Establish freemium revenue model with premium analytics tier.

### 2.2 Product Goals

- Detect and classify shots (made/missed) with >90% accuracy in well-lit outdoor/indoor environments.
- Map shot locations onto a 2D court heatmap using homography-based ground plane detection.
- Provide real-time biomechanical feedback: release speed, release angle, leg angle, jump height.
- Support two distinct capture modes: ground-placed and tripod-mounted.
- Process all CV/ML inference on-device (no cloud dependency for core features).

### 2.3 Non-Goals (v1.0)

- Multi-camera or multi-angle tracking.
- Team-level analytics or multiplayer sessions.
- Wearable or external sensor integration.
- Live streaming or real-time remote coaching.
- Web dashboard (mobile-only for v1.0).

---

## 3. Feature Requirements

### 3.1 Feature Priority Table

| Feature Area | Description | Priority | MVP? |
| --- | --- | --- | --- |
| Shot Detection & Counting | Single multi-class YOLO detector (basketball + hoop) drives release detection, make/miss classification, and shot counting per session | P0 | Yes |
| Capture Mode: Tripod | Stationary elevated wide-angle shot with three-point line in full view | P0 | Yes |
| Capture Mode: Ground | Stationary wide-angle; low angle; limited court context; biomechanics focus | P0 | Yes |
| Vertical Jump Height | Estimate jump height from hip landmark displacement over time | P0 | Yes |
| Court Heatmap | AR homography maps shot locations to standard court overlay in Tripod mode | P0 | Yes |
| Ground Plane Detection (Manual Calibration) | ARCore detects floor plane to establish court coordinate system in Tripod mode | P0 | Yes |
| Ground Plane Detection (Auto w/ drag calibration) | ARCore detects floor plane automatically in Tripod mode | P1 | No |
| Shot Type Classification | Identify catch-and-shoot, pull-up, step-back, post moves | P1 | Yes |
| Release Time | Estimate time taken to shoot the ball | P1 | Yes |
| Release Speed | Estimate ball velocity from frame-delta of detected ball | P1 | Yes |
| Release Angle | MediaPipe pose landmarks compute wrist/elbow angle at release | P1 | Yes |
| Leg Angle (Knee Bend) | Detect knee flexion angle using pose estimation at shot prep | P1 | Yes |
| Session History & Stats | Store session summaries locally; shooting % trends over time | P1 | Yes |
| Drill Mode | Guided shooting drills with targets (e.g., corner 3s only) | P2 | No |
| Share / Export | Export heatmap image and session stats as PDF or share card | P2 | No |
| Freemium Paywall | Core tracking free; advanced analytics, drill history behind paywall | P1 | Yes |

### 3.2 MVP v1 Scope (Must Ship vs Deferred)

| Must Ship (MVP v1) | Deferred (Post-MVP) |
| --- | --- |
| **Onboarding & setup** | Auto drag calibration |
| • Onboarding + mode selection | Advanced shot-type taxonomy |
| • Camera start in under 2 minutes | Production-grade vertical jump KPI |
| **Core tracking (AI/CV)** | Cumulative multi-session heatmap |
| • Single multi-class YOLO (ball + hoop) | Guided drills |
| • Real-time shot detection + make/miss counting | Export/share cards |
| • Post-session manual correction | Freemium paywall/subscription flows |
| **Capture modes** | Coach dashboard and multi-player workflows |
| • Ground + Tripod capture support | |
| **Biomechanics (pose-based)** | |
| • Release angle | |
| • Leg angle | |
| **Sessions & history** | |
| • Session save | |
| • Basic trend/history | |
| **AR + court mapping (Tripod mode)** | |
| • ARCore floor detection | |
| • Manual 4-point calibration | |
| • Heatmap views: All / Makes / Misses | |

### 3.3 Capture Mode Specifications

#### Tripod Mode

- Phone mounted at ~3-6 ft height, angled down at ~0-30 degrees.
- Partial half-court and full three-point line or key area visible in frame.
- ARCore ground plane detection establishes court floor reference.
- Manual on-screen court layout drag/rotate calibration on detected plane surface.
- Homography transformation maps 2D image coordinates to normalized court coordinates.
- Enables: shot location heatmap, court zone classification, spatial trends.
- All biomechanical metrics available (full body visible).

#### Ground Mode

- Phone placed on ground in landscape orientation against object such as water bottle or wall.
- Court context limited; heatmap unavailable or estimated.
- Primary focus: release mechanics, shot form, shot classification, make/miss.
- Pose estimation still active, biomechanical metrics fully available.

### 3.4 AR Court Mapping & Heatmap

- 3D calibration with one-time manual input of user's height.
- Use ARCore (Google) for ground plane detection and real-world scale estimation.
- User manually drag and rotate court overlay on screen.
- User manually anchors four court corners via on-screen tap (two-finger zoom-in).
- Compute homography matrix H mapping image pixels to court plane coordinates.
- Each detected shot location transforms through H to normalized court coords.
- Heatmap rendered as overlay on standard court diagram using density interpolation (KDE).
- Color gradient: blue (cold/low frequency) to red (hot/high frequency).
- User can view heatmap by: all shots, makes only, misses only, by session, cumulative.

### 3.5 Biomechanical Metrics

#### Release Speed

- Estimated via ball displacement (pixels) between frames, calibrated by known court dimensions and player height.
- Displayed in mph or m/s. Accuracy target: +/- 2 mph.

#### Release Time

- Estimated via time delta between Prep state timestamp and Release state timestamp.

#### Release Angle

- Computed from elbow-wrist vector at the frame of ball release (detected via pose landmarks).
- Angle relative to vertical axis. Optimal range: 45-55 degrees flagged as ideal.
- Available in both capture modes.

#### Leg Angle (Knee Flexion)

- Measured at shot preparation phase (before jump).
- Hip-knee-ankle angle computed from MediaPipe pose landmarks.
- Feedback: deep bend (<90 deg), optimal (90-120 deg), minimal bend (>120 deg).

#### Vertical Jump Height

- Estimated by tracking hip landmark Y-position over time in the jump arc.
- Calibrated against one-time manual input of player's height.
- Requires tripod mode for accuracy; ground mode flagged as approximate.

---

## 4. Technical & Platform Constraints

| Constraint | Detail |
| --- | --- |
| Minimum Android Version | API 26 (Android 8.0 Oreo) |
| ARCore Required | ARCore-supported devices only (for heatmap mode) |
| Camera | Rear camera, minimum 1080p @ 30fps |
| Processing | All CV/ML on-device via TFLite and MediaPipe |
| Internet | Required only for account sync and optional cloud backup |
| Storage | Session video clips stored locally; auto-delete after 30 days (configurable) |
| ML Framework | TensorFlow Lite (YOLOv8n multi-class: basketball + hoop), MediaPipe (pose) |
| AR Framework | ARCore (Google) |
| Target Latency | <50ms pose inference, <100ms ball detection per frame |

---

## 5. Assumptions & Open Questions

### 5.1 Assumptions

- Primary use is outdoor courts or well-lit indoor gyms.
- Standard high school half-court dimensions used as baseline for court mapping in MVP.
- User must set up tripod and manually align court for heatmap mode.
- Device must support ARCore for spatial features.

### 5.2 Open Questions

- Will we offer a cloud sync tier for multi-device session access?
- Paywall threshold: **Answer: historic trend, number of shots per month, cloud sync.**

---

## 6. Competitor Analysis

| Competitor | Notes |
| --- | --- |
| **HomeCourt (iOS)** | Gold standard — shot tracking, form analysis, drill mode. iOS only. |
| **Hoops (iOS)** | Silver standard, shot tracking, form analysis, iPhone only. |
| **Trace Basketball** | Team-level video analysis, coach-focused. Requires hardware. |
| **ShotTracker** | Sensor-based wearable system. Accurate but expensive. |
| **Hudl Technique** | Generic slow-motion analysis. No AI shot detection. |
| **CourtVision** | Android-native, AI + AR, biomechanics, no external hardware required. |
