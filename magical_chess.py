"""
Magical Chess ML - Convenience Imports

This module provides a simple import interface for the Magical Chess ML library.

Usage:
    from magical_chess import GameState, get_tier, get_level, MagicalChessEnv

Instead of:
    from src.engine import GameState, get_tier, get_level, MagicalChessEnv
"""

from src.engine import (
    GameState,
    get_tier,
    get_level,
    TIER_EMPTY,
    TIER_BRONZE,
    TIER_SILVER,
    TIER_GOLD,
    MAX_PIECE_LEVEL,
    MagicalChessEnv,
)

__all__ = [
    "GameState",
    "get_tier",
    "get_level",
    "TIER_EMPTY",
    "TIER_BRONZE",
    "TIER_SILVER",
    "TIER_GOLD",
    "MAX_PIECE_LEVEL",
    "MagicalChessEnv",
]
