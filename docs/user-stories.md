# 🗂️ User Stories & Features

**CourtVision — Android Basketball AI Tracker**

Version 1.0 | Planning Stage | March 2026

---

## Epic Overview

| Epic | Title | Stories | MVP Scope |
| --- | --- | --- | --- |
| EPIC-1 | Onboarding & Capture Setup | US-01 to US-04 | Yes |
| EPIC-2 | Shot Tracking & Detection | US-05 to US-08 | Yes |
| EPIC-3 | AR Court Mapping & Heatmap | US-09 to US-12 | Yes |
| EPIC-4 | Biomechanical Metrics | US-13 to US-17 | Partial |
| EPIC-5 | Session History & Analytics | US-18 to US-21 | Yes |
| EPIC-6 | Drills, Export & Premium | US-22 to US-25 | No |

---

## Story Point Reference

| Points | Size | Description |
| --- | --- | --- |
| 1 | XS | Trivial — UI copy, minor state toggle, known implementation |
| 2 | S | Small — simple logic, clear requirements, <half day |
| 3 | M | Medium — some complexity, ~1 day, single component |
| 5 | L | Large — multiple components, ~2-3 days, integration needed |
| 8 | XL | Very Large — significant complexity, split into sub-tasks if possible |
| 13 | XXL | Spike or major feature — requires research, >1 week, break down before sprint |

---

## P0/P1 Delivery Plan (Story Points + Owner + Target Sprint)

> **MVP?** column references `docs/prd.md` Section 3.2 (Must Ship / Deferred table) as the canonical definition.

| Story | Priority | Story Points | Owner | Target Sprint | MVP? |
| --- | --- | --- | --- | --- | --- |
| US-01 First-Time Onboarding | P0 | 3 | Product/UX + Android | Sprint 1 | Yes |
| US-02 Tripod Mode Setup | P0 | 2 | Android Capture | Sprint 1 | Yes |
| US-03 Ground Mode Setup | P0 | 3 | Android Capture | Sprint 1 | Yes |
| US-04 Tripod Court Calibration | P0 | 5 | AR/CV | Sprint 2 | Yes |
| US-05 Auto Shot Detection | P0 | 8 | ML/CV | Sprint 2 | Yes |
| US-09 ARCore Floor Detection | P0 | 8 | AR/CV | Sprint 3 | Yes |
| US-06 Shot Type Classification | P1 | 8 | ML/CV | Sprint 6 | No |
| US-07 Live Session Counter | P1 | 3 | Android Capture | Sprint 3 | Yes |
| US-08 Manual Shot Correction | P1 | 3 | Android + Data | Sprint 4 | Yes |
| US-10 Shot Heatmap View | P1 | 5 | AR/CV + Analytics | Sprint 5 | Yes |
| US-11 Heatmap Filters | P1 | 3 | Analytics/UI | Sprint 5 | Yes |
| US-13 Release Speed | P1 | 5 | ML/CV | Sprint 4 | Yes |
| US-14 Release Angle | P1 | 3 | ML/CV | Sprint 4 | Yes |
| US-15 Knee Flexion Feedback | P1 | 3 | ML/CV | Sprint 4 | Yes |
| US-16 Metrics in Both Modes | P1 | 2 | ML/CV + QA | Sprint 4 | Yes |
| US-18 Auto Session Save | P1 | 5 | Data/Android | Sprint 5 | Yes |
| US-19 Shooting Trend Chart | P1 | 3 | Analytics/UI | Sprint 6 | Yes |
| US-20 Shot-by-Shot Session Review | P1 | 5 | Android + Data | Sprint 6 | Yes |
| US-24 Freemium Upgrade Prompt | P1 | 5 | Monetization + Android | Sprint 6 | No |

## EPIC-1: Onboarding & Capture Setup

### US-01 | First-Time Onboarding

> *As a first-time user, I want to be guided through camera permissions, mounting modes, and court setup in under 2 minutes, so that I can start tracking shots immediately without reading documentation. I will put in my height as reference.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 3 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Camera permission flow completes in <5 taps
- [ ] Height input captured and stored on first launch
- [ ] Mode selection (Tripod / Ground) shown clearly
- [ ] Onboarding completable in under 2 minutes

---

### US-02 | Tripod Mode Setup

> *As a player using a tripod, I want to mount my phone on a tripod, open the app and select the tripod mode.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 2 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Tripod mode is selectable from home screen
- [ ] App provides mounting guidance (height, angle)
- [ ] Camera preview displays in tripod orientation

---

### US-03 | Ground Mode Setup

> *As a player using ground mode, I want to start a session with a single tap and use automatic hoop detection (without mandatory manual rim boxing), with heatmap features disabled and biomechanics-focused tracking, so I get relevant metrics for this setup.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 3 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Ground mode launchable in a single tap
- [ ] Automatic hoop detection initializes without mandatory manual rim boxing
- [ ] Heatmap features disabled/hidden in ground mode
- [ ] Biomechanical metrics fully active

---

### US-04 | Tripod Court Calibration

> *As a user in tripod mode, I want to tap four court reference points and confirm automatic hoop detection before my session starts, so the app can correctly map shot locations onto the court heatmap.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 5 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Four-point court tapping interface available
- [ ] Hoop detection is auto-derived from the multi-class detector, with optional manual override only if confidence is low
- [ ] Homography computed after calibration
- [ ] Recalibration prompt if camera moves

---

## EPIC-2: Shot Tracking & Detection

### US-05 | Auto Shot Detection

> *As a player, I want the app to automatically detect each shot attempt and count makes and misses in real time without me pressing any buttons, so that I can focus entirely on shooting without interruption.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 8 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Shot detection triggers without user input
- [ ] Make/miss classified per shot attempt
- [ ] Detection accuracy >90% in well-lit environments
- [ ] <200ms delay from release to event fired

---

### US-06 | Shot Type Classification

> *As a player, I want each detected shot to be classified by type — catch and shoot, pull-up, step-back, or off-dribble — automatically, so that I can review my shot diet and see which situations I am most and least efficient from.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 8 |
| MVP | No |

**Acceptance Criteria**

- [ ] Shot types: catch & shoot, pull-up, step-back, off-dribble, post move
- [ ] Classification visible in session review
- [ ] Per-type shooting % shown in analytics

---

### US-07 | Live Session Counter

> *As a player, I want to see a live counter of makes, misses, and shooting percentage on screen during my session, so that I stay motivated and can monitor my performance in real time.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Makes, misses, and % displayed live on camera overlay
- [ ] Counter updates within 200ms of shot detection
- [ ] Overlay is non-obstructive and dismissible

---

### US-08 | Manual Shot Correction

> *As a player reviewing footage, I want to manually correct a make or miss classification after the session by reviewing the shot clip, so that the stats accurately reflect my actual shooting.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Each shot has a reviewable clip (+/- 3s)
- [ ] Make/miss toggle available in review
- [ ] Stats recalculate after correction

---

## EPIC-3: AR Court Mapping & Heatmap

### US-09 | ARCore Floor Detection

> *As a player using tripod mode, I want the app to use ARCore to detect the court floor plane and establish a real-world coordinate system automatically, so that shot location data is spatially accurate without manual measurement.*

| Field | Value |
| --- | --- |
| Priority | P0 |
| Story Points | 8 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] ARCore initializes and detects horizontal plane within 30s of session start
- [ ] App provides pan guidance for plane initialization
- [ ] Real-world scale used for metric calculations

---

### US-10 | Shot Heatmap View

> *As a player, I want to view a color-coded heatmap of all my shot attempts overlaid on a standard court diagram after my session, so that I can immediately see which zones I shoot well from and which I avoid.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 5 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Heatmap renders after minimum 5 shots
- [ ] Color gradient: blue (low) → red (high density)
- [ ] Overlay on standard half-court diagram
- [ ] Renders in <500ms for up to 500 shots

---

### US-11 | Heatmap Filters

> *As a player, I want to filter the heatmap to show only makes, only misses, or all attempts with a single tap toggle, so that I can analyze where I am most and least efficient spatially.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

**Acceptance Criteria**

- [ ] Toggle buttons for: All / Makes / Misses
- [ ] Heatmap updates on toggle tap
- [ ] Current filter state clearly indicated

---

### US-12 | Cumulative Heatmap

> *As a player reviewing multiple sessions, I want to view a cumulative heatmap that aggregates shot data across multiple sessions, so that I can track long-term patterns in my shot selection.*

| Field | Value |
| --- | --- |
| Priority | P2 |
| Story Points | |
| MVP | No |

**Acceptance Criteria**

- [ ] Cumulative view selectable from analytics screen
- [ ] Date range filter available
- [ ] Session count shown alongside heatmap

---

## EPIC-4: Biomechanical Metrics

### US-13 | Release Speed

> *As a player, I want to see my release speed in mph or km/h for each shot at the end of a session, so that I can track whether my shot has enough velocity and whether it is consistent.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 5 |
| MVP | Yes |

---

### US-14 | Release Angle

> *As a player, I want to see my release angle — the ballistic launch angle of the ball relative to the horizontal ground plane — in degrees for each shot, with a clear indicator of whether it falls in the optimal 40–55 degree range, so that I can improve my shot arc with objective feedback.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

---

### US-15 | Knee Flexion Feedback

> *As a player or coach, I want to see my knee flexion angle at the shot preparation phase, with feedback labels such as deep bend, optimal, or minimal, so that I can understand whether I am generating power from my legs correctly.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

---

### US-16 | Metrics in Both Modes

> *As a player, I want all biomechanical metrics to be available in both tripod and ground capture modes, so that I am not limited to one setup when focusing on shooting mechanics.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 2 |
| MVP | Yes |

---

### US-17 | Vertical Jump Height

> *As a player, I want the app to estimate my vertical jump height in inches or centimeters per shot, so that I can track athletic development alongside shooting improvement over time.*

| Field | Value |
| --- | --- |
| Priority | P2 |
| Story Points | |
| MVP | No |

---

## EPIC-5: Session History & Analytics

### US-18 | Auto Session Save

> *As a player, I want every session to be automatically saved with a summary including date, total shots, shooting percentage, and top biomechanical averages, so that I have a complete record of all my workouts.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 5 |
| MVP | Yes |

---

### US-19 | Shooting Trend Chart

> *As a player, I want to view a timeline of my shooting percentage trend over the past 30 days on a chart, so that I can see whether my shooting is improving or declining over time.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 3 |
| MVP | Yes |

---

### US-20 | Shot-by-Shot Session Review

> *As a player, I want to tap any past session and review a shot-by-shot breakdown with the clip for each attempt, so that I can learn from past sessions and identify recurring mechanical errors.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 5 |
| MVP | Yes |

---

### US-21 | Per-Zone Shooting %

> *As a player, I want to see per-zone shooting percentage data alongside my heatmap, for example 3P left corner: 38%, so that I understand not just where I shoot from but how well I shoot from each area.*

| Field | Value |
| --- | --- |
| Priority | P2 |
| Story Points | |
| MVP | No |

---

## EPIC-6: Drills, Export & Premium (Post-MVP)

### US-22 | Guided Drill Mode

> *As a player, I want to select a guided drill mode such as Mikan drill or corner 3-point challenge, where the app tracks my completion and performance per drill, so that I have structured workouts with measurable goals.*

| Field | Value |
| --- | --- |
| Priority | P2 |
| Story Points | |
| MVP | No |

---

### US-23 | Export Session

> *As a player, I want to export my session summary and heatmap as a PDF or shareable image card, so that I can share progress with a coach or post it to social media.*

| Field | Value |
| --- | --- |
| Priority | P2 |
| Story Points | |
| MVP | No |

---

### US-24 | Freemium Upgrade Prompt

> *As a free tier user, I want to clearly understand which features are free and which require a premium subscription, with a non-intrusive upgrade prompt after my third session, so that I can make an informed decision about upgrading without feeling pressured.*

| Field | Value |
| --- | --- |
| Priority | P1 |
| Story Points | 5 |
| MVP | No |

---

### US-25 | Coach Dashboard

> *As a coach, I want to create a player profile and assign drills to a player who uses the app independently, then review their session results in my own coach dashboard view, so that I can remotely monitor multiple players without needing to be present at every session.*

| Field | Value |
| --- | --- |
| Priority | P3 |
| Story Points | |
| MVP | No |
