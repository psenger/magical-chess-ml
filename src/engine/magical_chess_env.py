"""Gymnasium environment for Magical Chess ML.

This module provides a Gymnasium-compatible environment for training
reinforcement learning agents on the Magical Chess puzzle game.

The environment implements the standard Gymnasium interface with:
- Observation space: 4x4 grid (piece levels) + hourglass count
- Action space: Source tile (0-15) + Target tile (0-15) = 256 actions
- Reward: Score delta from merges (hourglass efficiency variant available)

Example usage:
    >>> from src.engine import MagicalChessEnv
    >>> env = MagicalChessEnv()
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game_state import GameState
from .move_system import (
    execute_move,
    get_valid_moves,
    is_game_over,
    spawn_random_tile,
)


class MagicalChessEnv(gym.Env):
    """Gymnasium environment for Magical Chess puzzle game.

    This environment implements the Magical Chess drag-and-drop merge puzzle
    as a Gymnasium-compatible RL environment. Agents learn to select source
    and target tiles to create merge clusters of 3+ identical pieces.

    Observation Space:
        Box space with shape (17,) containing:
        - grid[0:16]: Flattened 4x4 grid with piece levels (0-127)
        - hourglass: Hourglass count normalized by initial amount

    Action Space:
        Discrete(256) representing (source_tile * 16 + target_tile)
        where both source and target are indices 0-15 for 4x4 grid.

    Rewards:
        - Base: Score delta from successful merge
        - Invalid action: Small negative penalty
        - Game over: 0 (episode ends)

    Episode Termination:
        - Board full with no valid merges remaining

    Attributes:
        metadata: Environment metadata for Gymnasium compatibility.
        observation_space: The observation space specification.
        action_space: The action space specification.
        state: Current game state.
        initial_hourglass: Starting hourglass count for normalization.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        initial_hourglass: int = 100,
        initial_tiles: int = 2,
        invalid_action_penalty: float = -0.1,
        hourglass_cost: int = 1,
        reward_mode: str = "hourglass_efficiency",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the Magical Chess environment.

        Args:
            initial_hourglass: Starting hourglass count. Default 100.
            initial_tiles: Number of tiles to spawn at start. Default 2.
            invalid_action_penalty: Penalty for invalid actions. Default -0.1.
            hourglass_cost: Hourglass cost per move. Default 1.
            reward_mode: Reward calculation mode. Options:
                - "score_delta": Raw score from merge (merged_level * 10)
                - "hourglass_efficiency": Score per hourglass spent
                Default is "hourglass_efficiency".
            render_mode: Rendering mode ("ansi" or None). Default None.
            seed: Random seed for reproducibility. Default None.
        """
        super().__init__()

        self.initial_hourglass = initial_hourglass
        self.initial_tiles = initial_tiles
        self.invalid_action_penalty = invalid_action_penalty
        self.hourglass_cost = hourglass_cost
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        # Set up random number generator
        self._np_random: Optional[np.random.Generator] = None
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Define observation space:
        # - 16 values for flattened 4x4 grid (piece levels 0-127)
        # - 1 value for normalized hourglass count
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32,
        )

        # Define action space:
        # Source (0-15) * 16 + Target (0-15) = 256 possible actions
        self.action_space = spaces.Discrete(256)

        # Initialize state
        self.state: Optional[GameState] = None
        self._valid_moves_cache: Optional[list] = None

    def _get_obs(self) -> np.ndarray:
        """Convert current state to observation array.

        Returns:
            Observation array with normalized grid values and hourglass.
        """
        if self.state is None:
            return np.zeros(17, dtype=np.float32)

        # Flatten grid and normalize (max level ~127, use 128 for safety)
        grid_obs = self.state.grid.flatten().astype(np.float32) / 128.0

        # Normalize hourglass count
        hourglass_obs = np.array(
            [self.state.hourglass_count / self.initial_hourglass],
            dtype=np.float32,
        )

        return np.concatenate([grid_obs, hourglass_obs])

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state.

        Returns:
            Dictionary with valid_moves, score, hourglass, and game_over status.
        """
        if self.state is None:
            return {
                "valid_moves": [],
                "valid_actions": [],
                "score": 0,
                "hourglass": 0,
                "game_over": True,
            }

        # Cache valid moves for efficiency
        if self._valid_moves_cache is None:
            self._valid_moves_cache = get_valid_moves(self.state)

        # Convert valid moves to action indices
        valid_actions = [
            source[0] * 4 + source[1] + (target[0] * 4 + target[1]) * 16
            for source, target in self._valid_moves_cache
        ]
        # Correction: action = source_idx * 16 + target_idx
        valid_actions = [
            (source[0] * 4 + source[1]) * 16 + (target[0] * 4 + target[1])
            for source, target in self._valid_moves_cache
        ]

        return {
            "valid_moves": self._valid_moves_cache,
            "valid_actions": valid_actions,
            "score": self.state.score,
            "hourglass": self.state.hourglass_count,
            "hourglass_efficiency": (
                self.state.score / (self.initial_hourglass - self.state.hourglass_count)
                if self.state.hourglass_count < self.initial_hourglass else 0.0
            ),
            "game_over": is_game_over(self.state),
        }

    def _action_to_positions(
        self, action: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert action index to (source, target) positions.

        Args:
            action: Action index 0-255.

        Returns:
            Tuple of (source_pos, target_pos) where each is (row, col).
        """
        source_idx = action // 16
        target_idx = action % 16

        source = (source_idx // 4, source_idx % 4)
        target = (target_idx // 4, target_idx % 4)

        return source, target

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional configuration options.

        Returns:
            Tuple of (observation, info dictionary).
        """
        super().reset(seed=seed)

        # Update RNG if seed provided
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Create empty initial state
        self.state = GameState(
            grid=np.zeros((4, 4), dtype=np.int8),
            hourglass_count=self.initial_hourglass,
            score=0,
            last_score_delta=0,
        )

        # Spawn initial tiles
        for _ in range(self.initial_tiles):
            self.state = spawn_random_tile(self.state, self._np_random)

        # Clear valid moves cache
        self._valid_moves_cache = None

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action index 0-255 representing source*16 + target.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Clear valid moves cache (will be recomputed)
        self._valid_moves_cache = None

        # Convert action to positions
        source, target = self._action_to_positions(action)

        # Get valid moves and check if action is valid
        valid_moves = get_valid_moves(self.state)
        is_valid_action = (source, target) in valid_moves

        if not is_valid_action:
            # Invalid action - return penalty without changing state
            obs = self._get_obs()
            reward = self.invalid_action_penalty
            terminated = is_game_over(self.state)
            truncated = False
            info = self._get_info()
            return obs, reward, terminated, truncated, info

        # Execute valid move
        self.state = execute_move(self.state, source, target)

        # Calculate score from merge (use merged piece level as proxy for score)
        merged_value = self.state.grid[target[0], target[1]]
        score_delta = int(merged_value * 10)  # Reward proportional to merged level

        # Deduct hourglass cost
        hourglass_spent = self.hourglass_cost
        new_hourglass = max(0, self.state.hourglass_count - hourglass_spent)

        # Update state with score and hourglass
        self.state = GameState(
            grid=self.state.grid,
            hourglass_count=new_hourglass,
            score=self.state.score + score_delta,
            last_score_delta=score_delta,
        )

        # Spawn new tile after merge
        try:
            self.state = spawn_random_tile(self.state, self._np_random)
        except ValueError:
            # Board is full - will be detected in game over check
            pass

        # Calculate reward based on reward mode
        if self.reward_mode == "hourglass_efficiency":
            # Reward = score per hourglass spent
            # This encourages maximizing score relative to resources consumed
            if hourglass_spent > 0:
                reward = float(score_delta) / float(hourglass_spent)
            else:
                reward = float(score_delta)
        else:
            # Default: raw score delta
            reward = float(score_delta)

        # Check termination
        terminated = is_game_over(self.state)
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current state.

        Returns:
            String representation of the board if render_mode is "ansi".
        """
        if self.render_mode != "ansi" or self.state is None:
            return None

        tier_prefix = {
            0: "--",
            1: "B1",
            2: "B2",
            3: "B3",
            4: "S4",
            5: "S5",
            6: "S6",
        }

        lines = []
        lines.append("┌────┬────┬────┬────┐")

        for i, row in enumerate(self.state.grid):
            cells = []
            for val in row:
                if val == 0:
                    cells.append("--")
                elif val <= 6:
                    cells.append(tier_prefix.get(int(val), f"?{val}"))
                else:
                    cells.append(f"G{val}")
            lines.append(f"│ {cells[0]} │ {cells[1]} │ {cells[2]} │ {cells[3]} │")
            if i < 3:
                lines.append("├────┼────┼────┼────┤")

        lines.append("└────┴────┴────┴────┘")
        lines.append(f"Score: {self.state.score} | Hourglass: {self.state.hourglass_count}")

        return "\n".join(lines)

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions for action masking in RL algorithms.

        Returns:
            Boolean array of shape (256,) where True indicates valid action.
        """
        if self.state is None:
            return np.zeros(256, dtype=bool)

        mask = np.zeros(256, dtype=bool)
        valid_moves = get_valid_moves(self.state)

        for source, target in valid_moves:
            source_idx = source[0] * 4 + source[1]
            target_idx = target[0] * 4 + target[1]
            action = source_idx * 16 + target_idx
            mask[action] = True

        return mask
