"""Game engine module for Magical Chess ML."""

from .game_state import (
    GameState,
    get_tier,
    get_level,
    TIER_EMPTY,
    TIER_BRONZE,
    TIER_SILVER,
    TIER_GOLD,
    MAX_PIECE_LEVEL,
)

from .move_system import (
    find_merge_cluster,
    is_valid_move,
    get_valid_moves,
    execute_move,
    get_empty_cells,
    is_game_over,
    spawn_random_tile,
)

from .magical_chess_env import MagicalChessEnv

__all__ = [
    # game_state
    "GameState",
    "get_tier",
    "get_level",
    "TIER_EMPTY",
    "TIER_BRONZE",
    "TIER_SILVER",
    "TIER_GOLD",
    "MAX_PIECE_LEVEL",
    # move_system
    "find_merge_cluster",
    "is_valid_move",
    "get_valid_moves",
    "execute_move",
    "get_empty_cells",
    "is_game_over",
    "spawn_random_tile",
    # environment
    "MagicalChessEnv",
]
