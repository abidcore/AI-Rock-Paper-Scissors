"""
config/settings.py
-------------------
Centralized, single-source-of-truth configuration for the AI Rock
Paper Scissors Game.

Every tunable constant -- camera parameters, MediaPipe thresholds,
game timing, colors, and UI layout -- lives here rather than being
scattered as "magic numbers" throughout the codebase. This makes the
application easy to tune, review, and extend without touching game
or rendering logic.
"""

import cv2

# --------------------------------------------------------------------------
# Camera / Capture Settings
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FLIP_CAMERA_HORIZONTALLY: bool = True
CAMERA_WARMUP_FRAMES: int = 5
CAMERA_RECONNECT_ATTEMPTS: int = 3

# --------------------------------------------------------------------------
# MediaPipe Hands Model Settings
# --------------------------------------------------------------------------
MAX_NUM_HANDS: int = 1                 # Only the player's single hand matters
MIN_DETECTION_CONFIDENCE: float = 0.75
MIN_TRACKING_CONFIDENCE: float = 0.70
BBOX_PADDING: int = 20

# --------------------------------------------------------------------------
# Gesture Classification Settings
# --------------------------------------------------------------------------
THUMB_EXTENSION_RATIO: float = 0.15    # Relative-distance threshold for thumb logic
GESTURE_STABILIZATION_BUFFER_SIZE: int = 7
GESTURE_STABILIZATION_MIN_AGREEMENT: int = 4

# --------------------------------------------------------------------------
# Game Flow / Timing Settings (all in seconds)
# --------------------------------------------------------------------------
COUNTDOWN_DURATION: float = 3.0        # "3, 2, 1" shown before each round
CAPTURE_WINDOW_DURATION: float = 0.9   # Window during which the player's move is sampled
REVEAL_DISPLAY_DURATION: float = 2.6   # How long the round result banner stays on screen
GAME_OVER_LINGER_DURATION: float = 0.0  # No forced delay -- waits for user key press

# --------------------------------------------------------------------------
# Game Rules Settings
# --------------------------------------------------------------------------
BEST_OF_3_ROUNDS: int = 3
BEST_OF_5_ROUNDS: int = 5
DEFAULT_GAME_MODE_ROUNDS: int = BEST_OF_3_ROUNDS

# --------------------------------------------------------------------------
# FPS Counter Settings
# --------------------------------------------------------------------------
FPS_SMOOTHING_FACTOR: float = 0.9

# --------------------------------------------------------------------------
# Sound Effects (optional, zero-dependency, Windows-native beep)
# --------------------------------------------------------------------------
SOUND_ENABLED: bool = True
BEEP_FREQ_COUNTDOWN_TICK: int = 600
BEEP_FREQ_COUNTDOWN_GO: int = 1000
BEEP_FREQ_ROUND_WIN: int = 1200
BEEP_FREQ_ROUND_LOSE: int = 300
BEEP_FREQ_ROUND_DRAW: int = 700
BEEP_DURATION_MS: int = 120

# --------------------------------------------------------------------------
# Window / Display Settings
# --------------------------------------------------------------------------
WINDOW_NAME: str = "AI Rock Paper Scissors | Abid Ali"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_XLARGE: float = 3.2
FONT_SCALE_LARGE: float = 1.4
FONT_SCALE_MEDIUM: float = 0.8
FONT_SCALE_SMALL: float = 0.6
FONT_THICKNESS: int = 2

# --------------------------------------------------------------------------
# Color Palette (BGR format, as used by OpenCV)
# --------------------------------------------------------------------------
COLOR_PRIMARY = (255, 130, 0)           # Vivid azure-blue accent
COLOR_SECONDARY = (0, 210, 255)         # Amber highlight
COLOR_SUCCESS = (80, 220, 100)          # Green -- win / connected
COLOR_DANGER = (60, 60, 235)            # Red -- lose / disconnected
COLOR_WARNING = (0, 165, 255)           # Orange -- draw / warnings
COLOR_TEXT_LIGHT = (245, 245, 245)
COLOR_TEXT_DARK = (25, 25, 25)
COLOR_PANEL_BG = (35, 35, 35)
COLOR_PLAYER = (255, 130, 0)
COLOR_COMPUTER = (60, 60, 235)
COLOR_ROCK = (140, 140, 140)
COLOR_PAPER = (0, 210, 255)
COLOR_SCISSORS = (180, 105, 255)

# --------------------------------------------------------------------------
# UI Layout Settings
# --------------------------------------------------------------------------
PANEL_OPACITY: float = 0.6
TOP_BAR_HEIGHT: int = 90
CORNER_RADIUS: int = 14
SCOREBOARD_PANEL_WIDTH: int = 300
SCOREBOARD_PANEL_HEIGHT: int = 130

# --------------------------------------------------------------------------
# Keyboard Shortcuts
# --------------------------------------------------------------------------
EXIT_KEYS = {ord("q"), ord("Q"), 27}
KEY_BEST_OF_3 = ord("3")
KEY_BEST_OF_5 = ord("5")
KEY_RESTART = ord("r")
KEY_RESTART_ALT = ord("R")
