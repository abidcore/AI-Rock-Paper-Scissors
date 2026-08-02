"""
gesture_detector.py
--------------------
Pure-logic module responsible for two related jobs:

    1. RPSGestureClassifier -- converts 21 MediaPipe hand landmarks
       into a `Move` (Rock, Paper, Scissors) or `None` if the hand
       shape doesn't clearly match any of the three recognized poses.

    2. GestureStabilizer -- smooths a noisy per-frame stream of
       classifications using a sliding-window majority vote, so the
       live "Detected: ROCK" preview shown to the player doesn't
       flicker between frames.

This module contains no OpenCV/UI code, so the classification logic
can be unit tested independently of the video pipeline.
"""

from collections import Counter, deque
from typing import Deque, List, Optional, Tuple

from config import settings
from .game_engine import Move

# MediaPipe hand landmark indices used by the classifier.
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

Landmark = Tuple[float, float, float]


class RPSGestureClassifier:
    """
    Classifies a hand's landmark configuration as Rock, Paper, Scissors,
    or an ambiguous/unrecognized pose (None).

    Classification proceeds in two steps:
        (a) Determine which of the five fingers are extended, reusing
            the same dual-heuristic approach proven effective for
            general finger counting: a normalized distance comparison
            for the orientation-independent thumb, and a vertical
            joint comparison for the remaining four fingers.
        (b) Map the resulting 5-boolean finger pattern onto one of the
            three recognized Rock-Paper-Scissors hand shapes.
    """

    def __init__(self, thumb_extension_ratio: float = settings.THUMB_EXTENSION_RATIO) -> None:
        self._thumb_ratio_threshold = thumb_extension_ratio

    @staticmethod
    def _distance(point_a: Landmark, point_b: Landmark) -> float:
        return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

    def _is_thumb_extended(self, landmarks: List[Landmark]) -> bool:
        tip_to_pinky = self._distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
        ip_to_pinky = self._distance(landmarks[THUMB_IP], landmarks[PINKY_MCP])
        palm_width = self._distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
        palm_width = palm_width if palm_width > 1e-6 else 1e-6
        relative_extension = (tip_to_pinky - ip_to_pinky) / palm_width
        return relative_extension > self._thumb_ratio_threshold

    @staticmethod
    def _is_finger_extended(landmarks: List[Landmark], tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        tip_y = landmarks[tip_idx][1]
        pip_y = landmarks[pip_idx][1]
        mcp_y = landmarks[mcp_idx][1]
        return tip_y < pip_y and tip_y < mcp_y

    def get_finger_states(self, landmarks: List[Landmark]) -> List[bool]:
        """Return [thumb, index, middle, ring, pinky] extended booleans."""
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}.")

        return [
            self._is_thumb_extended(landmarks),
            self._is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
            self._is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            self._is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
            self._is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]

    def classify(self, landmarks: List[Landmark]) -> Optional[Move]:
        """
        Args:
            landmarks: Exactly 21 (x, y, z) normalized landmark tuples.

        Returns:
            The recognized Move, or None if the hand shape does not
            clearly match Rock, Paper, or Scissors.
        """
        _, index, middle, ring, pinky = self.get_finger_states(landmarks)

        # ROCK: a closed fist -- all four fingers curled into the palm.
        # The thumb's state is ignored since it commonly rests either
        # tucked in or lightly across the fist in a natural rock pose.
        if not index and not middle and not ring and not pinky:
            return Move.ROCK

        # SCISSORS: exactly index and middle extended, ring and pinky curled.
        if index and middle and not ring and not pinky:
            return Move.SCISSORS

        # PAPER: an open hand -- all four fingers extended.
        if index and middle and ring and pinky:
            return Move.PAPER

        # Any other configuration (e.g. three fingers up, only one
        # finger up, etc.) does not correspond to a recognized move.
        return None


class GestureStabilizer:
    """
    Smooths a noisy per-frame stream of `Optional[Move]` classifications
    using a fixed-size sliding window and majority-vote resolution,
    preventing the live on-screen "Detected: ..." preview from flickering.
    """

    def __init__(
        self,
        buffer_size: int = settings.GESTURE_STABILIZATION_BUFFER_SIZE,
        min_agreement: int = settings.GESTURE_STABILIZATION_MIN_AGREEMENT,
    ) -> None:
        self._buffer_size = buffer_size
        self._min_agreement = min_agreement
        self._history: Deque[Optional[Move]] = deque(maxlen=buffer_size)
        self._stable_value: Optional[Move] = None

    def update(self, raw_value: Optional[Move]) -> Optional[Move]:
        """Feed the latest raw classification; get back the stable value."""
        self._history.append(raw_value)

        if len(self._history) < self._buffer_size:
            self._stable_value = raw_value
            return self._stable_value

        counts = Counter(self._history)
        most_common_value, frequency = counts.most_common(1)[0]

        if frequency >= self._min_agreement:
            self._stable_value = most_common_value

        return self._stable_value

    def reset(self) -> None:
        """Clear all accumulated history (e.g. when the hand disappears)."""
        self._history.clear()
        self._stable_value = None


class RoundCaptureBuffer:
    """
    Accumulates raw (non-stabilized) gesture classifications throughout
    a round's short capture window, then resolves them into a single
    final move via majority vote once the window closes.

    Kept deliberately separate from `GestureStabilizer`: the stabilizer
    drives the continuous live preview shown throughout the whole game,
    while this buffer captures a fresh, independent sample specifically
    for the decisive "reveal" moment of each round.
    """

    def __init__(self) -> None:
        self._samples: List[Optional[Move]] = []

    def add_sample(self, move: Optional[Move]) -> None:
        self._samples.append(move)

    def resolve(self) -> Optional[Move]:
        """
        Return the majority move among valid (non-None) samples, or
        None if no valid gesture was captured at all during the window.
        """
        valid_samples = [move for move in self._samples if move is not None]
        if not valid_samples:
            return None
        return Counter(valid_samples).most_common(1)[0][0]

    def clear(self) -> None:
        self._samples.clear()
