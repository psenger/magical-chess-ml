"""GameState class and helper functions for Magical Chess ML.

This module provides the core game state representation for the Magical Chess
game simulator and RL environment. The GameState is immutable and uses
copy-on-modify semantics to support safe lookahead simulations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Constants (Task Group 2)
# =============================================================================

TIER_EMPTY = "Empty"
TIER_BRONZE = "Bronze"
TIER_SILVER = "Silver"
TIER_GOLD = "Gold"

MAX_PIECE_LEVEL = 127  # int8 max positive value


# =============================================================================
# Helper Functions (Task Group 2)
# =============================================================================

def get_tier(value: int) -> str:
    """Get the tier name for a given tile value.

    Args:
        value: The tile value (0-127)

    Returns:
        The tier name: "Empty", "Bronze", "Silver", or "Gold"

    Examples:
        >>> get_tier(0)
        'Empty'
        >>> get_tier(2)
        'Bronze'
        >>> get_tier(5)
        'Silver'
        >>> get_tier(10)
        'Gold'
    """
    if value == 0:
        return TIER_EMPTY
    elif 1 <= value <= 3:
        return TIER_BRONZE
    elif 4 <= value <= 6:
        return TIER_SILVER
    else:  # value >= 7
        return TIER_GOLD


def get_level(value: int) -> int:
    """Get the level number for a given tile value.

    Args:
        value: The tile value (0-127)

    Returns:
        The level number (same as the value)

    Examples:
        >>> get_level(0)
        0
        >>> get_level(5)
        5
    """
    return value


def _default_grid() -> np.ndarray:
    """Create the default empty 4x4 grid."""
    return np.zeros((4, 4), dtype=np.int8)


# =============================================================================
# GameState Class (Task Groups 1, 3, 4, 5, 6)
# =============================================================================

@dataclass(frozen=True)
class GameState:
    """Immutable representation of the Magical Chess game state.

    This class represents the complete state of the game board at a given
    point in time. It is immutable (frozen dataclass) to support safe
    lookahead simulations and MCTS algorithms.

    Attributes:
        grid: A 4x4 NumPy array of int8 values representing tile positions.
              0 = empty, 1-3 = Bronze tier, 4-6 = Silver tier, 7+ = Gold tier.
        hourglass_count: The current hourglass resource balance.
        score: The cumulative score earned across all moves.
        last_score_delta: Points earned in the most recent move.

    Examples:
        >>> state = GameState()
        >>> state.grid.shape
        (4, 4)
        >>> state.hourglass_count
        0

        >>> new_state = state.with_hourglass(100)
        >>> new_state.hourglass_count
        100
    """

    grid: np.ndarray = field(default_factory=_default_grid)
    hourglass_count: int = 0
    score: int = 0
    last_score_delta: int = 0

    def __post_init__(self) -> None:
        """Validate state after initialization."""
        self.validate()

    # =========================================================================
    # Validation Methods (Task Group 5)
    # =========================================================================

    def validate(self) -> None:
        """Validate the game state.

        Raises:
            ValueError: If the state is invalid with a descriptive message.
        """
        # Validate grid shape
        if self.grid.shape != (4, 4):
            raise ValueError(
                f"Invalid grid shape: expected (4, 4), got {self.grid.shape}"
            )

        # Validate grid values are in range
        if np.any(self.grid < 0) or np.any(self.grid > MAX_PIECE_LEVEL):
            raise ValueError(
                f"Grid values must be in range [0, {MAX_PIECE_LEVEL}]"
            )

        # Validate non-negative hourglass count
        if self.hourglass_count < 0:
            raise ValueError(
                f"hourglass_count must be non-negative, got {self.hourglass_count}"
            )

    def is_valid(self) -> bool:
        """Check if the state is valid without raising an exception.

        Returns:
            True if the state is valid, False otherwise.
        """
        try:
            self.validate()
            return True
        except ValueError:
            return False

    # =========================================================================
    # Equality and Hashing (Task Group 1)
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        """Compare two GameState instances for equality.

        Args:
            other: The other object to compare with.

        Returns:
            True if all properties are equal, False otherwise.
        """
        if not isinstance(other, GameState):
            return NotImplemented

        return (
            np.array_equal(self.grid, other.grid)
            and self.hourglass_count == other.hourglass_count
            and self.score == other.score
            and self.last_score_delta == other.last_score_delta
        )

    def __hash__(self) -> int:
        """Compute hash for the GameState.

        Returns:
            Hash value for use in sets and dict keys.
        """
        # Convert grid to tuple of tuples for hashability
        grid_tuple = tuple(tuple(row) for row in self.grid)
        return hash((grid_tuple, self.hourglass_count, self.score, self.last_score_delta))

    # =========================================================================
    # Clone and Copy-on-Modify Methods (Task Group 3)
    # =========================================================================

    def clone(self) -> GameState:
        """Create a deep copy of this GameState.

        Essential for MCTS and lookahead simulations to avoid side effects.

        Returns:
            A new GameState with copied values.
        """
        return GameState(
            grid=np.copy(self.grid),
            hourglass_count=self.hourglass_count,
            score=self.score,
            last_score_delta=self.last_score_delta,
        )

    def with_grid(self, new_grid: np.ndarray) -> GameState:
        """Return a new GameState with a different grid.

        Args:
            new_grid: The new grid array (will be deep copied).

        Returns:
            A new GameState with the specified grid.
        """
        return GameState(
            grid=np.copy(new_grid),
            hourglass_count=self.hourglass_count,
            score=self.score,
            last_score_delta=self.last_score_delta,
        )

    def with_score(self, score: int, delta: int) -> GameState:
        """Return a new GameState with updated score.

        Args:
            score: The new cumulative score.
            delta: The new last_score_delta.

        Returns:
            A new GameState with the specified score values.
        """
        return GameState(
            grid=np.copy(self.grid),
            hourglass_count=self.hourglass_count,
            score=score,
            last_score_delta=delta,
        )

    # =========================================================================
    # Hourglass Management (Task Group 4)
    # =========================================================================

    def can_afford(self, cost: int) -> bool:
        """Check if the hourglass balance can afford a cost.

        Args:
            cost: The cost to check.

        Returns:
            True if hourglass_count >= cost, False otherwise.
        """
        return self.hourglass_count >= cost

    def deduct(self, cost: int) -> GameState:
        """Return a new GameState with reduced hourglass count.

        Args:
            cost: The amount to deduct from hourglass_count.

        Returns:
            A new GameState with hourglass_count reduced by cost.
        """
        return GameState(
            grid=np.copy(self.grid),
            hourglass_count=self.hourglass_count - cost,
            score=self.score,
            last_score_delta=self.last_score_delta,
        )

    def with_hourglass(self, count: int) -> GameState:
        """Return a new GameState with specified hourglass count.

        Args:
            count: The new hourglass count.

        Returns:
            A new GameState with the specified hourglass_count.
        """
        return GameState(
            grid=np.copy(self.grid),
            hourglass_count=count,
            score=self.score,
            last_score_delta=self.last_score_delta,
        )

    # =========================================================================
    # Serialization Methods (Task Group 6)
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize the GameState to a JSON-compatible dictionary.

        Returns:
            A dictionary with all state properties in JSON-compatible format.
        """
        return {
            "grid": self.grid.tolist(),
            "hourglass_count": self.hourglass_count,
            "score": self.score,
            "last_score_delta": self.last_score_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        """Deserialize a GameState from a dictionary.

        Args:
            data: A dictionary containing grid, hourglass_count, score,
                  and last_score_delta.

        Returns:
            A new GameState instance.

        Raises:
            ValueError: If the data is invalid.
        """
        grid = np.array(data["grid"], dtype=np.int8)
        return cls(
            grid=grid,
            hourglass_count=data["hourglass_count"],
            score=data["score"],
            last_score_delta=data["last_score_delta"],
        )

    def to_json(self) -> str:
        """Serialize the GameState to a JSON string.

        Returns:
            A JSON string representation of the state.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> GameState:
        """Deserialize a GameState from a JSON string.

        Args:
            json_str: A JSON string representation of the state.

        Returns:
            A new GameState instance.

        Raises:
            ValueError: If the JSON is invalid or represents an invalid state.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
