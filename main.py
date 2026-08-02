#!/usr/bin/env python3
"""
AI Rock Paper Scissors Game (Using Hand Gestures)
===================================================
Main application entry point.

A real-time computer vision game where the player shows a Rock,
Paper, or Scissors hand gesture to the webcam and plays a full match
against a randomized computer AI opponent. The game features a
countdown before each round, gesture capture with majority-vote
resolution, automatic winner detection, a live scoreboard, Best-of-3
and Best-of-5 match modes, an FPS counter, detection confidence
display, webcam status indication, and full exception handling.

Author:  Abid Ali
Project: AI & Machine Learning Diploma Portfolio
License: MIT (see LICENSE file)

Usage:
    python main.py

Keyboard Shortcuts:
    3        -- Start a Best-of-3 match (from the main menu)
    5        -- Start a Best-of-5 match (from the main menu)
    R        -- Restart / return to menu (from the game-over screen)
    Q / ESC  -- Exit the application at any time
"""

import sys
import time
from enum import Enum, auto
from typing import Optional

import cv2

from config import settings
from src import (
    FPSCounter,
    GameEngine,
    GameMode,
    GestureStabilizer,
    HandTracker,
    Move,
    RoundCaptureBuffer,
    RoundResult,
    RPSGestureClassifier,
    Scoreboard,
)
from src import utils


class CameraNotAvailableError(Exception):
    """Raised when the webcam cannot be opened after all retry attempts."""


class GameState(Enum):
    """The finite set of high-level UI/flow states the game moves through."""

    MENU = auto()
    COUNTDOWN = auto()
    CAPTURE = auto()
    REVEAL = auto()
    GAME_OVER = auto()


_MOVE_ICON_FILENAMES = {
    Move.ROCK: "rock.png",
    Move.PAPER: "paper.png",
    Move.SCISSORS: "scissors.png",
}

_MOVE_COLORS = {
    Move.ROCK: settings.COLOR_ROCK,
    Move.PAPER: settings.COLOR_PAPER,
    Move.SCISSORS: settings.COLOR_SCISSORS,
}

_RESULT_COLORS = {
    RoundResult.PLAYER_WIN: settings.COLOR_SUCCESS,
    RoundResult.COMPUTER_WIN: settings.COLOR_DANGER,
    RoundResult.DRAW: settings.COLOR_WARNING,
    RoundResult.NO_GESTURE: settings.COLOR_DANGER,
}

_RESULT_MESSAGES = {
    RoundResult.PLAYER_WIN: "YOU WIN THIS ROUND!",
    RoundResult.COMPUTER_WIN: "COMPUTER WINS THIS ROUND!",
    RoundResult.DRAW: "DRAW!",
    RoundResult.NO_GESTURE: "NO GESTURE DETECTED -- ROUND LOST",
}


class RockPaperScissorsApp:
    """
    Top-level application controller.

    Owns the video capture device, the CV pipeline (hand tracking and
    gesture classification), the game engine, and the timing-based
    state machine that drives the on-screen game flow.
    """

    def __init__(self) -> None:
        self._capture = self._open_camera()
        self._hand_tracker = HandTracker()
        self._classifier = RPSGestureClassifier()
        self._live_stabilizer = GestureStabilizer()
        self._capture_buffer = RoundCaptureBuffer()
        self._fps_counter = FPSCounter()
        self._scoreboard = Scoreboard()

        self._engine = GameEngine(mode=GameMode.BEST_OF_3)
        self._state = GameState.MENU
        self._state_entered_at = time.time()

        self._webcam_connected = True
        self._current_round_outcome = None
        self._last_detection_confidence = 0.0

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _open_camera() -> cv2.VideoCapture:
        """Attempt to open the configured webcam with bounded retries."""
        last_error = None
        for _ in range(settings.CAMERA_RECONNECT_ATTEMPTS):
            try:
                capture = cv2.VideoCapture(settings.CAMERA_INDEX)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
                    for _ in range(settings.CAMERA_WARMUP_FRAMES):
                        capture.read()
                    return capture
                capture.release()
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
            time.sleep(0.5)

        raise CameraNotAvailableError(
            f"Could not access webcam at index {settings.CAMERA_INDEX} "
            f"after {settings.CAMERA_RECONNECT_ATTEMPTS} attempts. "
            f"Ensure no other application is using the camera and that "
            f"OS-level camera permissions are granted."
            + (f" Last error: {last_error}" if last_error else "")
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _enter_state(self, new_state: GameState) -> None:
        self._state = new_state
        self._state_entered_at = time.time()

    def _time_in_state(self) -> float:
        return time.time() - self._state_entered_at

    def _start_new_match(self, mode: GameMode) -> None:
        self._engine.reset(mode=mode)
        self._live_stabilizer.reset()
        self._capture_buffer.clear()
        self._current_round_outcome = None
        self._enter_state(GameState.COUNTDOWN)
        utils.play_beep(settings.BEEP_FREQ_COUNTDOWN_GO)

    # ------------------------------------------------------------------
    # Per-frame CV processing
    # ------------------------------------------------------------------
    def _process_hand(self, frame):
        """Run hand detection + gesture classification for the current frame."""
        frame, detected_hands = self._hand_tracker.find_hands(frame, draw=True)

        raw_move: Optional[Move] = None
        confidence = 0.0
        bounding_box = None

        if detected_hands:
            hand = detected_hands[0]
            raw_move = self._classifier.classify(hand.landmarks_norm)
            confidence = hand.handedness_score
            bounding_box = hand.bounding_box

        stable_move = self._live_stabilizer.update(raw_move)
        self._last_detection_confidence = confidence

        if self._state == GameState.CAPTURE:
            self._capture_buffer.add_sample(raw_move)

        if bounding_box is not None:
            label = stable_move.value if stable_move else "Detecting..."
            color = _MOVE_COLORS.get(stable_move, settings.COLOR_SECONDARY)
            utils.draw_bounding_box(frame, bounding_box, label, color)

        return frame, stable_move, confidence

    # ------------------------------------------------------------------
    # UI rendering -- shared chrome
    # ------------------------------------------------------------------
    def _draw_top_bar(self, frame) -> None:
        height, width = frame.shape[:2]
        utils.draw_translucent_rect(
            frame, (0, 0), (width, settings.TOP_BAR_HEIGHT), settings.COLOR_PANEL_BG, alpha=0.65
        )
        utils.draw_text(
            frame,
            "AI ROCK PAPER SCISSORS",
            (20, 35),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SECONDARY,
            thickness=2,
        )
        utils.draw_text(
            frame,
            "Hand Gesture Battle vs Computer AI",
            (20, 68),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )

        fps_text = f"FPS: {self._fps_counter.fps:5.1f}"
        text_w, _ = utils.get_text_size(fps_text, scale=settings.FONT_SCALE_MEDIUM, thickness=2)
        utils.draw_text(
            frame, fps_text, (width - text_w - 25, 35), scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SUCCESS, thickness=2,
        )

        status_text = "WEBCAM: LIVE" if self._webcam_connected else "WEBCAM: LOST"
        status_color = settings.COLOR_SUCCESS if self._webcam_connected else settings.COLOR_DANGER
        status_w, _ = utils.get_text_size(status_text, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_text(
            frame, status_text, (width - status_w - 25, 68), scale=settings.FONT_SCALE_SMALL,
            color=status_color, thickness=1,
        )
        cv2.circle(frame, (width - status_w - 40, 62), 5, status_color, cv2.FILLED)

    def _draw_confidence_panel(self, frame, confidence: float) -> None:
        height, width = frame.shape[:2]
        panel_w, panel_h = 240, 70
        x1, y1 = width - panel_w - 20, height - panel_h - 60
        x2, y2 = x1 + panel_w, y1 + panel_h

        utils.draw_rounded_panel(
            frame, (x1, y1), (x2, y2), settings.COLOR_PANEL_BG, alpha=0.65,
            border_color=settings.COLOR_SECONDARY, border_thickness=2,
        )
        utils.draw_text(
            frame, "Detection Confidence", (x1 + 12, y1 + 24), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )
        utils.draw_confidence_bar(
            frame, (x1 + 12, y1 + 36), width=panel_w - 24, height=12, confidence=confidence,
            color=settings.COLOR_SUCCESS,
        )
        utils.draw_text(
            frame, f"{confidence * 100:.0f}%", (x1 + 12, y1 + 62), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )

    def _draw_footer(self, frame, message: str) -> None:
        height, width = frame.shape[:2]
        text_w, text_h = utils.get_text_size(message, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_translucent_rect(
            frame, (0, height - text_h - 24), (width, height), settings.COLOR_PANEL_BG, alpha=0.55
        )
        utils.draw_text(
            frame, message, ((width - text_w) // 2, height - 15), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1, shadow=False,
        )

    # ------------------------------------------------------------------
    # UI rendering -- per state
    # ------------------------------------------------------------------
    def _render_menu(self, frame) -> None:
        height, width = frame.shape[:2]
        utils.draw_translucent_rect(frame, (0, settings.TOP_BAR_HEIGHT), (width, height), (20, 20, 20), alpha=0.55)

        utils.draw_centered_text(frame, "ROCK  \u2022  PAPER  \u2022  SCISSORS", height // 2 - 130,
                                  scale=1.3, color=settings.COLOR_SECONDARY, thickness=3)
        utils.draw_centered_text(frame, "Choose Your Match Format", height // 2 - 60,
                                  scale=settings.FONT_SCALE_MEDIUM, color=settings.COLOR_TEXT_LIGHT, thickness=2)
        utils.draw_centered_text(frame, "Press  [3]  for Best of 3", height // 2,
                                  scale=settings.FONT_SCALE_MEDIUM, color=settings.COLOR_SUCCESS, thickness=2)
        utils.draw_centered_text(frame, "Press  [5]  for Best of 5", height // 2 + 55,
                                  scale=settings.FONT_SCALE_MEDIUM, color=settings.COLOR_SUCCESS, thickness=2)
        utils.draw_centered_text(frame, "Show Rock, Paper, or Scissors to the camera when prompted!",
                                  height // 2 + 120, scale=settings.FONT_SCALE_SMALL,
                                  color=settings.COLOR_TEXT_LIGHT, thickness=1)

    def _render_countdown(self, frame) -> None:
        elapsed = self._time_in_state()
        remaining = max(settings.COUNTDOWN_DURATION - elapsed, 0.0)
        count_value = int(remaining) + 1

        height = frame.shape[0]
        utils.draw_translucent_rect(frame, (0, settings.TOP_BAR_HEIGHT), (frame.shape[1], height), (20, 20, 20), alpha=0.25)

        if remaining > 0:
            display_text = str(min(count_value, int(settings.COUNTDOWN_DURATION)))
            utils.draw_centered_text(frame, display_text, height // 2, scale=settings.FONT_SCALE_XLARGE,
                                      color=settings.COLOR_SECONDARY, thickness=6)
            utils.draw_centered_text(frame, "Get Ready...", height // 2 + 90, scale=settings.FONT_SCALE_MEDIUM,
                                      color=settings.COLOR_TEXT_LIGHT, thickness=2)
        else:
            utils.draw_centered_text(frame, "SHOOT!", height // 2, scale=settings.FONT_SCALE_XLARGE * 0.75,
                                      color=settings.COLOR_SUCCESS, thickness=6)

        if remaining <= 0:
            self._capture_buffer.clear()
            self._enter_state(GameState.CAPTURE)

    def _render_capture(self, frame) -> None:
        elapsed = self._time_in_state()
        height = frame.shape[0]

        utils.draw_translucent_rect(frame, (0, settings.TOP_BAR_HEIGHT), (frame.shape[1], height), (20, 20, 20), alpha=0.15)
        utils.draw_centered_text(frame, "Hold Your Gesture!", height - 130, scale=settings.FONT_SCALE_MEDIUM,
                                  color=settings.COLOR_WARNING, thickness=2)

        progress = utils.clamp(elapsed / settings.CAPTURE_WINDOW_DURATION, 0.0, 1.0)
        bar_w = 300
        bar_x = (frame.shape[1] - bar_w) // 2
        bar_y = height - 100
        utils.draw_confidence_bar(frame, (bar_x, bar_y), bar_w, 14, progress, color=settings.COLOR_WARNING)

        if elapsed >= settings.CAPTURE_WINDOW_DURATION:
            player_move = self._capture_buffer.resolve()
            computer_move = self._engine.generate_computer_move()
            outcome = self._engine.play_round(player_move, computer_move=computer_move)
            self._current_round_outcome = outcome

            beep_freq = {
                RoundResult.PLAYER_WIN: settings.BEEP_FREQ_ROUND_WIN,
                RoundResult.COMPUTER_WIN: settings.BEEP_FREQ_ROUND_LOSE,
                RoundResult.DRAW: settings.BEEP_FREQ_ROUND_DRAW,
                RoundResult.NO_GESTURE: settings.BEEP_FREQ_ROUND_LOSE,
            }[outcome.result]
            utils.play_beep(beep_freq)

            self._enter_state(GameState.REVEAL)

    def _render_reveal(self, frame) -> None:
        height, width = frame.shape[:2]
        outcome = self._current_round_outcome
        if outcome is None:
            self._enter_state(GameState.COUNTDOWN)
            return

        utils.draw_translucent_rect(frame, (0, settings.TOP_BAR_HEIGHT), (width, height), (15, 15, 15), alpha=0.65)

        player_label = outcome.player_move.value if outcome.player_move else "No Gesture"
        computer_label = outcome.computer_move.value

        icon_size = (140, 140)
        icon_y = height // 2 - 170
        left_icon_x = width // 2 - 260
        right_icon_x = width // 2 + 120

        if outcome.player_move is not None:
            player_icon = utils.load_icon(_MOVE_ICON_FILENAMES[outcome.player_move])
            utils.overlay_icon(frame, player_icon, (left_icon_x, icon_y), target_size=icon_size)

        computer_icon = utils.load_icon(_MOVE_ICON_FILENAMES[outcome.computer_move])
        utils.overlay_icon(frame, computer_icon, (right_icon_x, icon_y), target_size=icon_size)

        card_y = height // 2 - 10
        utils.draw_centered_text(frame, f"YOU: {player_label}", card_y - 50, scale=1.05,
                                  color=settings.COLOR_PLAYER, thickness=3)
        utils.draw_centered_text(frame, "VS", card_y, scale=0.9, color=settings.COLOR_TEXT_LIGHT, thickness=2)
        utils.draw_centered_text(frame, f"CPU: {computer_label}", card_y + 50, scale=1.05,
                                  color=settings.COLOR_COMPUTER, thickness=3)

        result_color = _RESULT_COLORS[outcome.result]
        result_message = _RESULT_MESSAGES[outcome.result]
        utils.draw_centered_text(frame, result_message, card_y + 130, scale=settings.FONT_SCALE_MEDIUM,
                                  color=result_color, thickness=2)

        if self._engine.is_match_over:
            self._enter_state(GameState.GAME_OVER)
        elif self._time_in_state() >= settings.REVEAL_DISPLAY_DURATION:
            self._enter_state(GameState.COUNTDOWN)

    def _render_game_over(self, frame) -> None:
        height, width = frame.shape[:2]
        utils.draw_translucent_rect(frame, (0, settings.TOP_BAR_HEIGHT), (width, height), (10, 10, 10), alpha=0.75)

        winner = self._engine.match_winner
        winner_color = settings.COLOR_SUCCESS if winner == "Player" else settings.COLOR_DANGER
        headline = "\U0001F3C6 YOU WON THE MATCH!" if winner == "Player" else "COMPUTER WON THE MATCH"

        utils.draw_centered_text(frame, headline, height // 2 - 100, scale=1.2, color=winner_color, thickness=4)
        utils.draw_centered_text(
            frame,
            f"Final Score  --  You: {self._engine.player_score}   CPU: {self._engine.computer_score}",
            height // 2 - 20,
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=2,
        )
        utils.draw_centered_text(
            frame, "Press [R] to Play Again  |  [Q] / [ESC] to Quit", height // 2 + 50,
            scale=settings.FONT_SCALE_MEDIUM, color=settings.COLOR_SECONDARY, thickness=2,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[INFO] AI Rock Paper Scissors Game starting...")
        print("[INFO] Press '3' or '5' at the menu to choose a match format.")
        print("[INFO] Press 'Q' or 'ESC' at any time to exit.")

        cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                success, frame = self._capture.read()

                if not success or frame is None:
                    self._webcam_connected = False
                    print("[WARNING] Failed to read frame from webcam. Retrying...")
                    time.sleep(0.5)
                    continue

                self._webcam_connected = True

                if settings.FLIP_CAMERA_HORIZONTALLY:
                    frame = cv2.flip(frame, 1)

                frame, _stable_move, confidence = self._process_hand(frame)

                if self._state == GameState.MENU:
                    self._render_menu(frame)
                elif self._state == GameState.COUNTDOWN:
                    self._render_countdown(frame)
                elif self._state == GameState.CAPTURE:
                    self._render_capture(frame)
                elif self._state == GameState.REVEAL:
                    self._render_reveal(frame)
                elif self._state == GameState.GAME_OVER:
                    self._render_game_over(frame)

                if self._state in (GameState.COUNTDOWN, GameState.CAPTURE):
                    self._scoreboard.draw(frame, self._engine)
                    self._draw_confidence_panel(frame, confidence)
                elif self._state in (GameState.REVEAL, GameState.GAME_OVER):
                    self._scoreboard.draw(frame, self._engine)

                self._fps_counter.tick()
                self._draw_top_bar(frame)

                footer_messages = {
                    GameState.MENU: "Press [3] Best of 3   |   Press [5] Best of 5   |   [Q]/[ESC] Exit",
                    GameState.COUNTDOWN: "Get your hand ready in front of the camera...",
                    GameState.CAPTURE: "Hold your Rock / Paper / Scissors gesture steady!",
                    GameState.REVEAL: "Next round starting shortly...",
                    GameState.GAME_OVER: "Press [R] to play again or [Q]/[ESC] to exit",
                }
                self._draw_footer(frame, footer_messages[self._state])

                cv2.imshow(settings.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in settings.EXIT_KEYS:
                    print("[INFO] Exit key pressed. Shutting down...")
                    break

                if self._state == GameState.MENU:
                    if key == settings.KEY_BEST_OF_3:
                        self._start_new_match(GameMode.BEST_OF_3)
                    elif key == settings.KEY_BEST_OF_5:
                        self._start_new_match(GameMode.BEST_OF_5)
                elif self._state == GameState.GAME_OVER:
                    if key in (settings.KEY_RESTART, settings.KEY_RESTART_ALT):
                        self._enter_state(GameState.MENU)

                if cv2.getWindowProperty(settings.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user. Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C). Shutting down...")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release all hardware and library resources deterministically."""
        self._hand_tracker.close()
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye!")


def main() -> int:
    """Application bootstrap with top-level error handling."""
    try:
        app = RockPaperScissorsApp()
        app.run()
        return 0
    except CameraNotAvailableError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"[FATAL] An unexpected error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
