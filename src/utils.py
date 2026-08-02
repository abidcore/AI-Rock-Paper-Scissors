"""
utils.py
--------
Reusable OpenCV drawing/UI helper functions, PNG icon overlay support,
and optional zero-dependency sound effects shared across the
application. Isolating these here keeps main.py focused on the
high-level game-flow control rather than low-level pixel manipulation.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings

# --------------------------------------------------------------------------
# Basic drawing primitives
# --------------------------------------------------------------------------


def draw_translucent_rect(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = settings.PANEL_OPACITY,
) -> None:
    """Draw a semi-transparent filled rectangle directly onto `frame`."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_rounded_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = settings.PANEL_OPACITY,
    radius: int = settings.CORNER_RADIUS,
    border_color: Optional[Tuple[int, int, int]] = None,
    border_thickness: int = 2,
) -> None:
    """Draw a translucent panel with rounded corners for a polished look."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = frame.copy()

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    if border_color is not None:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), border_color, border_thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), border_color, border_thickness)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = settings.FONT_SCALE_MEDIUM,
    color: Tuple[int, int, int] = settings.COLOR_TEXT_LIGHT,
    thickness: int = settings.FONT_THICKNESS,
    font=settings.FONT,
    shadow: bool = True,
) -> None:
    """Draw text with an optional soft drop-shadow for readability over video."""
    if shadow:
        shadow_origin = (origin[0] + 2, origin[1] + 2)
        cv2.putText(frame, text, shadow_origin, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    center_y: int,
    scale: float = settings.FONT_SCALE_XLARGE,
    color: Tuple[int, int, int] = settings.COLOR_TEXT_LIGHT,
    thickness: int = 4,
    font=settings.FONT,
) -> None:
    """Draw text horizontally centered on the frame at a given y-coordinate."""
    height, width = frame.shape[:2]
    text_w, text_h = get_text_size(text, scale=scale, thickness=thickness, font=font)
    origin = ((width - text_w) // 2, center_y + text_h // 2)
    draw_text(frame, text, origin, scale=scale, color=color, thickness=thickness, font=font)


def get_text_size(
    text: str, scale: float = settings.FONT_SCALE_MEDIUM, thickness: int = settings.FONT_THICKNESS, font=settings.FONT
) -> Tuple[int, int]:
    """Return (width, height) in pixels that `text` will occupy when rendered."""
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    return width, height


def draw_confidence_bar(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    width: int,
    height: int,
    confidence: float,
    color: Tuple[int, int, int] = settings.COLOR_SUCCESS,
) -> None:
    """Draw a horizontal confidence/progress bar with a filled proportion."""
    x, y = top_left
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1)
    filled_width = int(width * max(0.0, min(confidence, 1.0)))
    if filled_width > 0:
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, cv2.FILLED)


def draw_bounding_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a labeled bounding box around a detected hand region."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label_w, label_h = get_text_size(label, scale=settings.FONT_SCALE_SMALL, thickness=1)
    label_bg_top_left = (x1, max(y1 - label_h - 12, 0))
    label_bg_bottom_right = (x1 + label_w + 14, y1)

    draw_translucent_rect(frame, label_bg_top_left, label_bg_bottom_right, color, alpha=0.75)
    draw_text(
        frame,
        label,
        (x1 + 7, max(y1 - 8, label_h)),
        scale=settings.FONT_SCALE_SMALL,
        color=settings.COLOR_TEXT_LIGHT,
        thickness=1,
        shadow=False,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Constrain `value` to the inclusive range [minimum, maximum]."""
    return max(minimum, min(value, maximum))


# --------------------------------------------------------------------------
# PNG icon loading and alpha-blended overlay
# --------------------------------------------------------------------------

_ASSETS_ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icons")


@lru_cache(maxsize=16)
def load_icon(filename: str) -> Optional[np.ndarray]:
    """
    Load a (possibly RGBA) PNG icon from the assets/icons directory,
    cached after first load. Returns None if the file cannot be found
    or read, allowing callers to fail gracefully instead of crashing.
    """
    path = os.path.join(_ASSETS_ICON_DIR, filename)
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return icon


def overlay_icon(
    frame: np.ndarray,
    icon: Optional[np.ndarray],
    top_left: Tuple[int, int],
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Alpha-blend an (optionally RGBA) icon image onto `frame` at the
    given top-left pixel position. Silently does nothing if `icon` is
    None, so missing/undownloaded icon assets never crash the app.

    Args:
        frame: Destination BGR frame (mutated in-place).
        icon: Source icon image, either BGR or BGRA.
        top_left: (x, y) pixel position for the icon's top-left corner.
        target_size: Optional (width, height) to resize the icon to.
    """
    if icon is None:
        return

    if target_size is not None:
        icon = cv2.resize(icon, target_size, interpolation=cv2.INTER_AREA)

    x, y = top_left
    icon_h, icon_w = icon.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    if x < 0 or y < 0 or x + icon_w > frame_w or y + icon_h > frame_h:
        return  # Skip drawing if it would fall outside frame bounds.

    region = frame[y : y + icon_h, x : x + icon_w]

    if icon.shape[2] == 4:
        icon_bgr = icon[:, :, :3]
        alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
        blended = (icon_bgr.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        frame[y : y + icon_h, x : x + icon_w] = blended
    else:
        frame[y : y + icon_h, x : x + icon_w] = icon[:, :, :3]


# --------------------------------------------------------------------------
# Optional, zero-dependency sound effects
# --------------------------------------------------------------------------


def play_beep(frequency: int, duration_ms: int = settings.BEEP_DURATION_MS) -> None:
    """
    Play a short beep tone if sound effects are enabled in settings.

    Uses the standard-library `winsound` module, which is only
    available on Windows. On any other platform (or if playback fails
    for any reason, e.g. a headless/CI environment) this function
    silently does nothing -- sound is an optional enhancement, never a
    requirement for gameplay, so its absence must never interrupt or
    crash the game loop.
    """
    if not settings.SOUND_ENABLED:
        return

    try:
        import winsound  # Windows-only standard library module

        winsound.Beep(frequency, duration_ms)
    except Exception:
        # Non-Windows platform, no audio device, or any other failure --
        # sound is purely cosmetic, so we fail silently and continue.
        pass
