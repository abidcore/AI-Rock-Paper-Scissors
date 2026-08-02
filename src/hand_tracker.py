"""
hand_tracker.py
----------------
Encapsulates all interaction with the MediaPipe Hands solution.

This module is responsible ONLY for detecting a hand in a video frame
and extracting its landmark coordinates, handedness, and detection
confidence. It intentionally contains no gesture-classification or
UI-rendering logic, keeping the architecture modular and adhering to
the Single Responsibility Principle.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import settings


@dataclass
class HandData:
    """Structured, typed representation of a single detected hand."""

    landmarks_px: List[Tuple[int, int, float]]        # Pixel-space (x, y, z)
    landmarks_norm: List[Tuple[float, float, float]]  # Normalized (0-1) coords
    handedness_label: str                              # "Left" or "Right"
    handedness_score: float                             # Confidence 0.0 - 1.0
    bounding_box: Tuple[int, int, int, int]             # (x_min, y_min, x_max, y_max)


class HandTracker:
    """
    Thin, robust wrapper around MediaPipe's Hands solution.

    Responsibilities:
        * Initialize and configure the MediaPipe Hands model.
        * Run inference on incoming BGR frames exactly once per call.
        * Convert normalized landmarks into both normalized and
          pixel-space coordinates.
        * Optionally draw the landmark skeleton for visualization.
        * Provide clean, typed HandData objects to the rest of the app.
    """

    def __init__(
        self,
        max_num_hands: int = settings.MAX_NUM_HANDS,
        min_detection_confidence: float = settings.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = settings.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Failed to initialize the MediaPipe Hands model: {exc}"
            ) from exc

    def find_hands(
        self, frame_bgr: np.ndarray, draw: bool = True
    ) -> Tuple[np.ndarray, List[HandData]]:
        """
        Run hand detection on a single BGR frame and optionally draw the
        landmark skeleton directly onto that frame.

        Args:
            frame_bgr: The current webcam frame in BGR color space.
            draw: Whether to draw the MediaPipe landmark skeleton in-place.

        Returns:
            A tuple of (frame_bgr, list_of_HandData).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        detected_hands: List[HandData] = []

        if not results.multi_hand_landmarks:
            return frame_bgr, detected_hands

        height, width = frame_bgr.shape[:2]
        handedness_list = results.multi_handedness or []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if draw:
                self._mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

            landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            landmarks_px = [
                (int(lm.x * width), int(lm.y * height), lm.z)
                for lm in hand_landmarks.landmark
            ]

            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                label = classification.label
                score = classification.score
            else:
                label, score = "Unknown", 0.0

            xs = [p[0] for p in landmarks_px]
            ys = [p[1] for p in landmarks_px]
            bounding_box = (
                max(min(xs) - settings.BBOX_PADDING, 0),
                max(min(ys) - settings.BBOX_PADDING, 0),
                min(max(xs) + settings.BBOX_PADDING, width),
                min(max(ys) + settings.BBOX_PADDING, height),
            )

            detected_hands.append(
                HandData(
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    handedness_label=label,
                    handedness_score=score,
                    bounding_box=bounding_box,
                )
            )

        return frame_bgr, detected_hands

    def close(self) -> None:
        """Release the underlying MediaPipe model resources."""
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
