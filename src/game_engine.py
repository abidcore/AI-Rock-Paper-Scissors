"""
game_engine.py
---------------
Pure game-logic module for the AI Rock Paper Scissors Game. Contains
no OpenCV, MediaPipe, or rendering code whatsoever, so the rules
engine can be exercised and unit tested completely independently of
the camera pipeline.

This module owns:
    * The `Move` enum (Rock / Paper / Scissors) and its win-condition rules.
    * The `GameMode` enum (Best of 3 / Best of 5).
    * The `RoundResult` enum describing the outcome of a single round.
    * The `GameEngine` class, which tracks scores, round history, and
      determines when the overall match has been won.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Move(Enum):
    """The three possible Rock-Paper-Scissors moves."""

    ROCK = "Rock"
    PAPER = "Paper"
    SCISSORS = "Scissors"

    def beats(self, other: "Move") -> bool:
        """
        Return True if `self` defeats `other` under standard
        Rock-Paper-Scissors rules:
            Rock crushes Scissors
            Scissors cuts Paper
            Paper covers Rock
        """
        winning_pairs = {
            (Move.ROCK, Move.SCISSORS),
            (Move.SCISSORS, Move.PAPER),
            (Move.PAPER, Move.ROCK),
        }
        return (self, other) in winning_pairs

    @classmethod
    def random_move(cls) -> "Move":
        """Return a uniformly random move -- used for the computer AI."""
        return random.choice(list(cls))


class GameMode(Enum):
    """Supported match formats."""

    BEST_OF_3 = 3
    BEST_OF_5 = 5

    @property
    def required_wins(self) -> int:
        """Number of round wins needed to clinch the match."""
        return (self.value // 2) + 1


class RoundResult(Enum):
    """Outcome classification for a single completed round."""

    PLAYER_WIN = "Player Wins Round"
    COMPUTER_WIN = "Computer Wins Round"
    DRAW = "Draw"
    NO_GESTURE = "No Gesture Detected -- Round Forfeited"


@dataclass
class RoundOutcome:
    """Immutable record describing exactly what happened in one round."""

    round_number: int
    player_move: Optional[Move]
    computer_move: Move
    result: RoundResult


@dataclass
class GameEngine:
    """
    Owns all Rock-Paper-Scissors match state: scores, round counter,
    round history, and match-completion detection.

    A `None` player move (i.e., no clear gesture was captured within
    the round's capture window) is treated as an automatic round loss
    for the player -- this keeps the game fair and prevents players
    from stalling indefinitely by not showing a gesture.
    """

    mode: GameMode = GameMode.BEST_OF_3
    player_score: int = field(default=0, init=False)
    computer_score: int = field(default=0, init=False)
    round_number: int = field(default=0, init=False)
    history: List[RoundOutcome] = field(default_factory=list, init=False)
    match_winner: Optional[str] = field(default=None, init=False)

    @property
    def required_wins(self) -> int:
        return self.mode.required_wins

    @property
    def is_match_over(self) -> bool:
        return self.match_winner is not None

    def generate_computer_move(self) -> Move:
        """Generate the computer's move for the upcoming round."""
        return Move.random_move()

    def _determine_result(self, player_move: Optional[Move], computer_move: Move) -> RoundResult:
        if player_move is None:
            return RoundResult.NO_GESTURE
        if player_move == computer_move:
            return RoundResult.DRAW
        if player_move.beats(computer_move):
            return RoundResult.PLAYER_WIN
        return RoundResult.COMPUTER_WIN

    def play_round(self, player_move: Optional[Move], computer_move: Optional[Move] = None) -> RoundOutcome:
        """
        Resolve one round of play, update scores/history, and check for
        match completion.

        Args:
            player_move: The move captured from the player's hand gesture,
                or None if no clear gesture was detected in time.
            computer_move: Optionally inject a specific computer move
                (primarily for deterministic unit testing). If omitted,
                a random move is generated.

        Returns:
            A RoundOutcome record describing what happened.
        """
        if self.is_match_over:
            raise RuntimeError("Cannot play another round -- the match has already concluded.")

        if computer_move is None:
            computer_move = self.generate_computer_move()

        result = self._determine_result(player_move, computer_move)
        self.round_number += 1

        if result == RoundResult.PLAYER_WIN:
            self.player_score += 1
        elif result in (RoundResult.COMPUTER_WIN, RoundResult.NO_GESTURE):
            self.computer_score += 1
        # RoundResult.DRAW leaves both scores unchanged.

        outcome = RoundOutcome(
            round_number=self.round_number,
            player_move=player_move,
            computer_move=computer_move,
            result=result,
        )
        self.history.append(outcome)

        if self.player_score >= self.required_wins:
            self.match_winner = "Player"
        elif self.computer_score >= self.required_wins:
            self.match_winner = "Computer"

        return outcome

    def reset(self, mode: Optional[GameMode] = None) -> None:
        """Reset the engine for a brand-new match, optionally changing mode."""
        if mode is not None:
            self.mode = mode
        self.player_score = 0
        self.computer_score = 0
        self.round_number = 0
        self.history.clear()
        self.match_winner = None
