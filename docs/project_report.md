# Project Report: AI Rock Paper Scissors Game (Using Hand Gestures)

**Author:** Abid Ali
**Program:** AI & Machine Learning Diploma
**Project Type:** Real-Time Computer Vision Game
**Repository:** [github.com/abidcore/AI-Rock-Paper-Scissors](https://github.com/abidcore)

---

## 1. Introduction

Rock-Paper-Scissors is a simple yet universally understood game of chance and simultaneous decision-making, making it an excellent testbed for demonstrating applied computer vision and real-time interactive system design. The **AI Rock Paper Scissors Game** reimagines this classic game as a human-vs-AI experience: a player shows a hand gesture to a standard webcam, the system recognizes it as Rock, Paper, or Scissors using landmark-based hand tracking, and a computer opponent responds with a randomly generated move, with the winner determined automatically according to the standard rules.

This project was undertaken as part of an AI & Machine Learning Diploma program to demonstrate practical, end-to-end applied computer vision engineering combined with real-time game-state architecture — extending beyond simple detection into a fully interactive, rule-governed application with match-level progression, fairness safeguards, and a polished user interface.

---

## 2. Problem Statement

Introductory hand-gesture-game tutorials typically hard-code a single-shot classification inside a monolithic script, lack any real match structure (no persistent scoring, no best-of-N logic), do not stabilize noisy per-frame gesture reads, and provide no mechanism for handling ambiguous or undetected gestures. These implementations are not representative of a properly engineered interactive computer vision system.

**The problem this project addresses is threefold:**

1. **Technical:** Reliably classify a player's hand shape into Rock, Paper, or Scissors in real time using only a 2D webcam feed, without dedicated depth sensors, while resisting frame-to-frame landmark noise.
2. **Game Design:** Structure this classification into a fair, complete, replayable match experience — countdown timing, a bounded capture window, automatic winner determination, persistent scoring across rounds, and configurable match formats (Best of 3 / Best of 5).
3. **Engineering:** Package the system as a modular, maintainable software application with a clean separation between computer-vision detection, game rules, and presentation, rather than an ad-hoc script.

---

## 3. Objectives

1. Achieve real-time (near 30 FPS) single-hand detection and landmark tracking on a standard consumer webcam.
2. Design a geometrically robust gesture classifier that accurately distinguishes Rock, Paper, and Scissors hand shapes.
3. Implement a deterministic, unit-testable rules engine that correctly resolves round winners and detects match completion under both Best-of-3 and Best-of-5 formats.
4. Eliminate frame-to-frame gesture-detection flicker through a two-tier temporal stabilization strategy.
5. Provide a complete, timing-driven game flow: menu, countdown, capture, reveal, and game-over states.
6. Handle the "no gesture detected" edge case fairly and deterministically rather than allowing indefinite stalling.
7. Build the system using clean, modular, PEP8-compliant, object-oriented Python architecture.
8. Produce professional documentation suitable for GitHub publication and academic evaluation.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.12+** | Core application language |
| **OpenCV** | Webcam capture, image processing, real-time UI rendering, icon compositing |
| **MediaPipe Hands** | Pre-trained deep-learning model for 21-point hand landmark detection |
| **NumPy** | Efficient numerical operations on image and icon arrays |
| **random (standard library)** | Uniform random move generation for the computer AI opponent |
| **Git/GitHub** | Version control and project hosting |

As in landmark-based hand-tracking pipelines generally, MediaPipe's Hands solution internally combines a palm-detector model with a landmark-regression model to produce 21 keypoints per hand. This project consumes that output directly, focusing engineering effort on the downstream gesture-classification algorithm, the game's rules engine, temporal stabilization, and the interactive state machine that ties them together.

---

## 5. Workflow

The application operates as a continuous per-frame loop layered beneath a higher-level, timing-driven game state machine:

1. **Capture** — Read a frame from the webcam via OpenCV's `VideoCapture`.
2. **Preprocess** — Horizontally flip the frame for a natural mirror-view experience; convert BGR → RGB for MediaPipe.
3. **Detect** — Run the MediaPipe Hands model to obtain the player's 21 landmarks (if a hand is visible), plus a detection confidence score.
4. **Classify** — Apply the geometric Rock/Paper/Scissors classifier to the current frame's landmarks, producing a raw per-frame move (or `None` if ambiguous).
5. **Stabilize (live preview)** — Feed the raw classification into a sliding-window majority-vote stabilizer that drives the continuous on-screen "Detected: ..." bounding-box label.
6. **Game State Machine** — Advance through:
   - **MENU** — waits for the player to choose Best-of-3 or Best-of-5.
   - **COUNTDOWN** — a timed 3-2-1-SHOOT sequence before each round.
   - **CAPTURE** — a short, fixed-duration window during which every raw per-frame classification is collected into a dedicated buffer.
   - **REVEAL** — once the capture window closes, the buffer resolves to a single final move via majority vote; the computer's move is generated randomly; the round winner is determined and the scoreboard updated; both moves and the result are displayed for a fixed duration.
   - **GAME_OVER** — shown once one side reaches the required number of round wins for the selected match format.
7. **Render** — Draw the professional UI overlay for the current state: top status bar (FPS, webcam status), scoreboard panel, round-history strip, detection-confidence panel, and state-specific content (menu instructions, countdown numerals, capture progress bar, reveal icons/labels, or game-over summary).
8. **Loop** — Repeat until the user exits via `Q`, `ESC`, or closes the window; `R` returns to the menu from the game-over screen to start a new match.

---

## 6. System Architecture

The project follows a **layered, single-responsibility architecture**:

```
┌───────────────────────────────────────────────────────────────┐
│                           main.py                                │
│  RockPaperScissorsApp: camera lifecycle, timing-based game state │
│  machine (MENU→COUNTDOWN→CAPTURE→REVEAL→GAME_OVER), rendering,   │
│  keyboard input, error handling                                   │
└───────┬───────────────────┬───────────────────┬─────────────────┘
        │                   │                   │
┌───────▼────────┐  ┌───────▼─────────┐ ┌───────▼─────────┐
│ hand_tracker.py │  │gesture_detector  │ │  game_engine.py   │
│ MediaPipe wrap  │  │ .py              │ │  Move / GameMode /  │
│ → HandData       │  │ RPSGestureClass- │ │  RoundResult enums,   │
│                    │  │ ifier, Gesture-  │ │  GameEngine (scores,   │
│                    │  │ Stabilizer,       │ │  round history,        │
│                    │  │ RoundCaptureBuffer│ │  match completion)      │
└────────────────────┘  └──────────────────┘ └─────────────────────────┘
        │                                              │
┌───────▼────────┐                            ┌────────▼─────────┐
│  scoreboard.py │                            │      fps.py        │
│  Renders score  │                            │  Smoothed FPS      │
│  panel from      │                            │  counter            │
│  GameEngine state │                            └─────────────────────┘
└────────────────────┘
        │
┌───────▼────────┐          ┌──────────────────────┐
│   utils.py      │◄─────────┤    config/settings.py │
│  Drawing helpers,│          │  Centralized constants│
│  icon overlay,    │          │  (camera, timing,      │
│  optional sound    │          │   colors, UI layout)    │
└────────────────────┘          └────────────────────────┘
```

**Design principles applied:**

- **Single Responsibility Principle** — `hand_tracker.py` only detects; `gesture_detector.py` only classifies/stabilizes; `game_engine.py` only enforces rules and tracks match state; `scoreboard.py` only renders score information; `utils.py` only provides drawing/asset/sound primitives.
- **Zero cross-contamination between logic and rendering** — `game_engine.py` and the classification logic in `gesture_detector.py` have no dependency on OpenCV drawing calls or MediaPipe internals, making them independently unit-testable with synthetic data.
- **Configuration over hard-coding** — all magic numbers (timing durations, thresholds, colors, dimensions) live in `config/settings.py`.
- **Typed data contracts** — `HandData` and `RoundOutcome` are `dataclasses` providing clean, typed interfaces between pipeline stages.
- **Explicit state machine** — the `GameState` enum in `main.py` makes every valid application state and its rendering/transition logic explicit and easy to reason about, rather than relying on implicit boolean flags.
- **Resource safety** — `HandTracker` implements the context-manager protocol, and the main application guarantees camera/window cleanup via a `finally` block regardless of how the loop exits.

---

## 7. Implementation

### 7.1 Hand Detection Layer (`hand_tracker.py`)

Wraps MediaPipe's `Hands` solution (configured for a single hand, since only the player's gesture is relevant), converting raw normalized landmark output into both normalized and pixel-space coordinate representations, alongside a confidence score and padded bounding box — packaged into a typed `HandData` object.

### 7.2 Gesture Classification (`gesture_detector.py`)

`RPSGestureClassifier` first derives a 5-element finger-extension state (thumb, index, middle, ring, pinky) using the same dual heuristic proven effective for general finger counting: a normalized, orientation-independent distance comparison for the thumb, and a vertical tip-vs-joint comparison for the remaining four fingers. This state is then mapped onto the three recognized hand shapes:

- **Rock** — index, middle, ring, and pinky all curled (closed fist).
- **Scissors** — index and middle extended, ring and pinky curled.
- **Paper** — all four fingers extended (open hand).

Any other finger configuration yields `None`, signaling an ambiguous or unrecognized pose rather than forcing an incorrect classification.

Two complementary temporal mechanisms handle noise:

- **`GestureStabilizer`** — a continuous sliding-window majority vote that drives the always-visible "Detected: ..." live preview shown under the player's hand throughout the game.
- **`RoundCaptureBuffer`** — an independent, per-round sample collector that gathers every raw (non-stabilized) classification during the fixed-duration capture window and resolves them into the single decisive move used for that round's result, ignoring `None` samples unless every sample in the window was ambiguous.

### 7.3 Game Rules Engine (`game_engine.py`)

The `Move` enum encodes the three moves and their win relationships via a `beats()` method backed by an explicit set of winning `(winner, loser)` pairs — avoiding fragile chained if/elif logic. `GameMode` encodes Best-of-3 and Best-of-5 formats, each exposing a `required_wins` property. `GameEngine` tracks player/computer scores, round number, full round history, and match-winner detection; a `None` player move (no gesture captured) is treated as an automatic round loss, keeping the game fair. Attempting to play a round after the match has concluded raises a `RuntimeError`, guarding against invalid state transitions.

### 7.4 Scoreboard Presentation (`scoreboard.py`)

`Scoreboard` reads state from a `GameEngine` instance purely for display purposes — match format, round number, both scores, and a compact round-by-round win/loss/draw indicator strip — with zero involvement in score computation itself.

### 7.5 UI Rendering & Icon Assets (`utils.py` + `main.py`)

Custom helpers provide translucent rounded panels, drop-shadow text, confidence bars, and alpha-blended PNG icon overlay (`overlay_icon`) used to display the Rock/Paper/Scissors icon graphics on the reveal screen. An optional, dependency-free `play_beep()` helper uses the standard-library `winsound` module on Windows and silently no-ops elsewhere, ensuring sound is a pure enhancement that can never crash or block the game loop on unsupported platforms.

### 7.6 Game State Machine (`main.py`)

`RockPaperScissorsApp` owns a `GameState` enum (`MENU`, `COUNTDOWN`, `CAPTURE`, `REVEAL`, `GAME_OVER`) and drives transitions using wall-clock timestamps (`time.time()`) rather than frame counts, ensuring consistent pacing regardless of the machine's actual frame rate. Each state has a dedicated render method, keeping the rendering logic for each phase of the game cleanly isolated.

### 7.7 Error Handling

- Camera initialization retries a bounded number of times before raising a descriptive `CameraNotAvailableError`, caught at the top level in `main()` and reported via `stderr` with a non-zero exit code.
- Per-frame read failures set a "WEBCAM: LOST" UI indicator and trigger a retry loop rather than crashing.
- A top-level `try/except/finally` in `RockPaperScissorsApp.run()` guarantees resource cleanup (camera release, window destruction, MediaPipe model closure) even on unexpected exceptions or `Ctrl+C` interruption.

---

## 8. Results

The system reliably classifies Rock, Paper, and Scissors hand shapes under typical indoor lighting conditions, running at real-time frame rates on standard consumer webcams. The two-tier stabilization strategy (continuous live-preview smoothing plus independent round-capture majority voting) noticeably improves the reliability of the final per-round decision compared to trusting a single frame's classification. The rules engine was verified through deterministic simulation to correctly resolve round outcomes, correctly terminate matches the instant a side reaches the required win count (without playing unnecessary extra rounds), and correctly reject further rounds once a match has concluded. The "no gesture detected" fairness rule was confirmed to correctly forfeit ambiguous rounds to the computer rather than stalling the match indefinitely.

---

## 9. Advantages

- Fully real-time performance suitable for interactive gameplay.
- Robust, orientation-independent gesture classification logic shared in spirit with proven finger-counting heuristics.
- Deterministic, thoroughly testable rules engine completely decoupled from the video pipeline.
- Two-tier temporal stabilization produces a noticeably more reliable and less "flickery" experience than naive single-frame classification.
- Clean, modular, extensible codebase suitable for further research, feature additions, or product development.
- Comprehensive error handling for real-world deployment conditions (missing/disconnected camera).
- Professional-grade UI comparable to commercial computer-vision game demos, including icon-based move reveal and a round-history trajectory strip.

---

## 10. Limitations

- Relies on 2D RGB input; unusual hand orientations or partial occlusion can occasionally cause an ambiguous (`None`) classification.
- Performance and accuracy can degrade under poor lighting, motion blur, or a hand positioned at the very edge of the frame.
- The computer AI selects moves uniformly at random rather than adapting to the player's historical tendencies.
- Designed for a single player using one hand; it does not currently support two-player local gesture-vs-gesture matches.
- Sound effect support is limited to a simple Windows-native beep and produces no audio on macOS/Linux.

---

## 11. Future Scope

- Replace the purely random computer AI with a simple adaptive strategy that reacts to the player's move-history distribution.
- Add a two-hand or two-camera local multiplayer mode.
- Integrate a richer, cross-platform audio library for countdown ticks, win/lose stings, and background music.
- Export a full match history (round-by-round moves and results) to CSV/JSON for post-game analysis.
- Port the video pipeline to a browser-based front end (e.g. via Streamlit or Flask with WebRTC) for camera-less deployment demonstrations.
- Add an automated `pytest` suite with continuous integration covering `game_engine.py` and `gesture_detector.py` for regression protection.

---

## 12. Conclusion

The AI Rock Paper Scissors Game successfully demonstrates the integration of a modern, pre-trained landmark-detection model with a custom-engineered geometric gesture classifier, a deterministic rules engine, and a timing-driven interactive state machine — all assembled with professional software architecture. Beyond its immediate function as a game, the project showcases core competencies expected of an applied AI/ML engineer: modular system design, robust handling of real-world sensor noise and ambiguity, careful UX and fairness considerations, and thorough technical documentation — making it a strong, representative artifact for an AI & Machine Learning Diploma portfolio.
