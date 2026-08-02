"""
scoreboard.py
-------------
Presentation-layer module dedicated to rendering the live scoreboard
panel. Deliberately separated from `game_engine.py` (which owns the
actual scoring *logic*) so that rules and presentation can evolve, be
tested, and be reasoned about independently -- a change to the panel's
visual layout should never risk touching match-resolution logic, and
vice versa.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings
from . import utils
from .game_engine import GameEngine, RoundResult


class Scoreboard:
    """Renders the round counter, player score, and computer score."""

    def draw(self, frame: np.ndarray, engine: GameEngine) -> None:
        """
        Draw the scoreboard panel in the top-left area of the frame,
        just beneath the main top status bar.

        Args:
            frame: The current video frame (mutated in-place).
            engine: The active GameEngine instance to read state from.
        """
        panel_x1, panel_y1 = 20, settings.TOP_BAR_HEIGHT + 20
        panel_x2 = panel_x1 + settings.SCOREBOARD_PANEL_WIDTH
        panel_y2 = panel_y1 + settings.SCOREBOARD_PANEL_HEIGHT

        utils.draw_rounded_panel(
            frame,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            settings.COLOR_PANEL_BG,
            alpha=0.68,
            border_color=settings.COLOR_SECONDARY,
            border_thickness=2,
        )

        mode_label = f"Best of {engine.mode.value}  (First to {engine.required_wins})"
        utils.draw_text(
            frame,
            mode_label,
            (panel_x1 + 15, panel_y1 + 28),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_SECONDARY,
            thickness=1,
        )

        round_label = f"Round: {engine.round_number}"
        utils.draw_text(
            frame,
            round_label,
            (panel_x1 + 15, panel_y1 + 54),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )

        score_line = f"You: {engine.player_score}"
        utils.draw_text(
            frame,
            score_line,
            (panel_x1 + 15, panel_y1 + 90),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_PLAYER,
            thickness=2,
        )

        computer_line = f"CPU: {engine.computer_score}"
        cpu_text_w, _ = utils.get_text_size(computer_line, scale=settings.FONT_SCALE_MEDIUM, thickness=2)
        utils.draw_text(
            frame,
            computer_line,
            (panel_x2 - cpu_text_w - 15, panel_y1 + 90),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_COMPUTER,
            thickness=2,
        )

        utils.draw_text(
            frame,
            "VS",
            ((panel_x1 + panel_x2) // 2 - 16, panel_y1 + 90),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )

    def draw_round_history_strip(
        self, frame: np.ndarray, engine: GameEngine, origin: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Draw a compact horizontal strip of win/loss/draw indicator dots,
        one per completed round, giving an at-a-glance match trajectory.
        """
        if not engine.history:
            return

        x, y = origin if origin is not None else (20, settings.TOP_BAR_HEIGHT + settings.SCOREBOARD_PANEL_HEIGHT + 40)
        radius = 8
        spacing = 24

        color_map = {
            RoundResult.PLAYER_WIN: settings.COLOR_SUCCESS,
            RoundResult.COMPUTER_WIN: settings.COLOR_DANGER,
            RoundResult.DRAW: settings.COLOR_WARNING,
            RoundResult.NO_GESTURE: settings.COLOR_DANGER,
        }

        for idx, outcome in enumerate(engine.history):
            center = (x + radius + idx * spacing, y)
            color = color_map.get(outcome.result, settings.COLOR_TEXT_LIGHT)
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, radius, (15, 15, 15), 1, cv2.LINE_AA)
