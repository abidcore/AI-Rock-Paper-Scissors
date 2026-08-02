"""
src
---
Core package for the AI Rock Paper Scissors Game.

Modules:
    hand_tracker      -- MediaPipe-based hand detection and landmark extraction
    gesture_detector  -- Rock/Paper/Scissors classification and temporal smoothing
    game_engine       -- Pure game rules, scoring, and match-completion logic
    scoreboard        -- Presentation layer for the live score panel
    fps               -- Lightweight, smoothed frames-per-second counter
    utils             -- Reusable OpenCV drawing/UI helpers, icon overlay, sound
"""

from .fps import FPSCounter
from .game_engine import GameEngine, GameMode, Move, RoundOutcome, RoundResult
from .gesture_detector import GestureStabilizer, RoundCaptureBuffer, RPSGestureClassifier
from .hand_tracker import HandData, HandTracker
from .scoreboard import Scoreboard

__all__ = [
    "HandTracker",
    "HandData",
    "RPSGestureClassifier",
    "GestureStabilizer",
    "RoundCaptureBuffer",
    "GameEngine",
    "GameMode",
    "Move",
    "RoundOutcome",
    "RoundResult",
    "Scoreboard",
    "FPSCounter",
]

__version__ = "1.0.0"
