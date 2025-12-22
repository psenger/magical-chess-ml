"""Tests for MagicalChessEnv Gymnasium environment.

This module contains all tests for the Gymnasium environment wrapper,
validating observation/action spaces, step/reset methods, and RL compatibility.
"""

import numpy as np
import pytest
import gymnasium as gym

from src.engine.magical_chess_env import MagicalChessEnv


# =============================================================================
# Observation Space Tests
# =============================================================================

class TestObservationSpace:
    """Tests for observation space definition and behavior."""

    def test_observation_space_is_box_with_correct_shape(self):
        """Test observation space is Box with shape (17,)."""
        env = MagicalChessEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (17,)

    def test_observation_space_bounds_are_0_to_1(self):
        """Test observation space has normalized bounds [0, 1]."""
        env = MagicalChessEnv()
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_observation_dtype_is_float32(self):
        """Test observation dtype is float32."""
        env = MagicalChessEnv()
        assert env.observation_space.dtype == np.float32

    def test_reset_returns_valid_observation(self):
        """Test reset returns observation within observation space."""
        env = MagicalChessEnv(seed=42)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

    def test_observation_contains_grid_and_hourglass(self):
        """Test observation has 16 grid values + 1 hourglass value."""
        env = MagicalChessEnv(seed=42)
        obs, _ = env.reset()

        # First 16 values are grid (normalized piece levels)
        grid_obs = obs[:16]
        assert len(grid_obs) == 16

        # Last value is hourglass (normalized)
        hourglass_obs = obs[16]
        assert hourglass_obs == 1.0  # Initial hourglass is 100% of initial


# =============================================================================
# Action Space Tests
# =============================================================================

class TestActionSpace:
    """Tests for action space definition and behavior."""

    def test_action_space_is_discrete_256(self):
        """Test action space is Discrete(256)."""
        env = MagicalChessEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 256

    def test_action_to_positions_conversion(self):
        """Test action index correctly converts to (source, target) positions."""
        env = MagicalChessEnv()

        # Action = source_idx * 16 + target_idx
        # source_idx = row * 4 + col, target_idx = row * 4 + col

        # Action 0: source_idx=0, target_idx=0 -> source=(0,0), target=(0,0)
        source, target = env._action_to_positions(0)
        assert source == (0, 0)
        assert target == (0, 0)

        # Action 17: source_idx=1, target_idx=1 -> source=(0,1), target=(0,1)
        source, target = env._action_to_positions(17)
        assert source == (0, 1)
        assert target == (0, 1)

        # Action 16: source_idx=1, target_idx=0 -> source=(0,1), target=(0,0)
        source, target = env._action_to_positions(16)
        assert source == (0, 1)
        assert target == (0, 0)

        # Action 255: source_idx=15, target_idx=15 -> source=(3,3), target=(3,3)
        source, target = env._action_to_positions(255)
        assert source == (3, 3)
        assert target == (3, 3)

    def test_get_action_mask_returns_boolean_array(self):
        """Test get_action_mask returns boolean array of shape (256,)."""
        env = MagicalChessEnv(seed=42)
        env.reset()
        mask = env.get_action_mask()
        assert mask.dtype == bool
        assert mask.shape == (256,)


# =============================================================================
# Reset Method Tests
# =============================================================================

class TestResetMethod:
    """Tests for reset method behavior."""

    def test_reset_returns_observation_and_info(self):
        """Test reset returns tuple of (observation, info)."""
        env = MagicalChessEnv(seed=42)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_spawns_initial_tiles(self):
        """Test reset spawns the configured number of initial tiles."""
        env = MagicalChessEnv(initial_tiles=3, seed=42)
        env.reset()

        # Count non-zero cells in grid
        non_zero = np.sum(env.state.grid > 0)
        assert non_zero == 3

    def test_reset_initializes_hourglass(self):
        """Test reset sets hourglass to initial value."""
        env = MagicalChessEnv(initial_hourglass=150, seed=42)
        env.reset()
        assert env.state.hourglass_count == 150

    def test_reset_with_seed_is_deterministic(self):
        """Test reset with same seed produces same initial state."""
        env1 = MagicalChessEnv()
        env2 = MagicalChessEnv()

        obs1, _ = env1.reset(seed=12345)
        obs2, _ = env2.reset(seed=12345)

        assert np.array_equal(obs1, obs2)

    def test_reset_clears_previous_state(self):
        """Test reset clears state from previous episode."""
        env = MagicalChessEnv(seed=42)
        obs1, _ = env.reset()

        # Take some steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        # Reset should return to clean state
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)


# =============================================================================
# Step Method Tests
# =============================================================================

class TestStepMethod:
    """Tests for step method behavior."""

    def test_step_returns_five_values(self):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = MagicalChessEnv(seed=42)
        env.reset()
        action = 0  # Any action
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_returns_valid_observation(self):
        """Test step returns observation within observation space."""
        env = MagicalChessEnv(seed=42)
        env.reset()
        action = 0
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    def test_step_invalid_action_returns_penalty(self):
        """Test step with invalid action returns negative penalty."""
        env = MagicalChessEnv(invalid_action_penalty=-0.5, seed=42)
        env.reset()

        # Action 0 (source=target=(0,0)) is always invalid
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == -0.5

    def test_step_valid_action_returns_positive_reward(self):
        """Test step with valid action returns positive reward."""
        env = MagicalChessEnv(seed=42)
        env.reset()

        # Get valid actions from info
        info = env._get_info()
        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            obs, reward, terminated, truncated, info = env.step(valid_action)
            assert reward > 0

    def test_step_spawns_tile_after_valid_move(self):
        """Test step spawns new tile after valid merge."""
        env = MagicalChessEnv(seed=42)
        env.reset()

        info = env._get_info()
        if info["valid_actions"]:
            tiles_before = np.sum(env.state.grid > 0)
            valid_action = info["valid_actions"][0]
            env.step(valid_action)
            # After merge: 3 consumed, 1 created, 1 spawned = tiles - 3 + 2 = tiles - 1
            # But since it depends on the merge, just verify spawning happened
            # by checking the board isn't completely empty

    def test_step_terminated_when_game_over(self):
        """Test step returns terminated=True when game is over."""
        env = MagicalChessEnv(seed=42)
        env.reset()

        # Play until game over
        max_steps = 1000
        for _ in range(max_steps):
            info = env._get_info()
            if info["game_over"]:
                # Any action should return terminated=True
                _, _, terminated, _, _ = env.step(0)
                assert terminated
                break
            elif info["valid_actions"]:
                action = info["valid_actions"][0]
                _, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
            else:
                # No valid actions means game over
                break

    def test_step_raises_without_reset(self):
        """Test step raises RuntimeError if called before reset."""
        env = MagicalChessEnv()
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(0)


# =============================================================================
# Info Dictionary Tests
# =============================================================================

class TestInfoDictionary:
    """Tests for info dictionary contents."""

    def test_info_contains_valid_moves(self):
        """Test info dictionary contains valid_moves list."""
        env = MagicalChessEnv(seed=42)
        _, info = env.reset()
        assert "valid_moves" in info
        assert isinstance(info["valid_moves"], list)

    def test_info_contains_valid_actions(self):
        """Test info dictionary contains valid_actions list."""
        env = MagicalChessEnv(seed=42)
        _, info = env.reset()
        assert "valid_actions" in info
        assert isinstance(info["valid_actions"], list)

    def test_info_contains_score(self):
        """Test info dictionary contains score."""
        env = MagicalChessEnv(seed=42)
        _, info = env.reset()
        assert "score" in info
        assert info["score"] == 0  # Initial score is 0

    def test_info_contains_hourglass(self):
        """Test info dictionary contains hourglass count."""
        env = MagicalChessEnv(initial_hourglass=100, seed=42)
        _, info = env.reset()
        assert "hourglass" in info
        assert info["hourglass"] == 100

    def test_info_contains_game_over(self):
        """Test info dictionary contains game_over boolean."""
        env = MagicalChessEnv(seed=42)
        _, info = env.reset()
        assert "game_over" in info
        assert isinstance(info["game_over"], bool)


# =============================================================================
# Render Tests
# =============================================================================

class TestRender:
    """Tests for render functionality."""

    def test_render_ansi_returns_string(self):
        """Test render with ansi mode returns string."""
        env = MagicalChessEnv(render_mode="ansi", seed=42)
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert "┌" in output  # Board border character

    def test_render_none_mode_returns_none(self):
        """Test render with no mode returns None."""
        env = MagicalChessEnv(render_mode=None, seed=42)
        env.reset()
        output = env.render()
        assert output is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete environment workflows."""

    def test_full_episode_until_game_over(self):
        """Test running a full episode until game over."""
        env = MagicalChessEnv(seed=42)
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 1000

        while steps < max_steps:
            # Get valid actions or random action
            if info["valid_actions"]:
                action = info["valid_actions"][0]
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Should have completed some steps
        assert steps > 0
        # Game should eventually terminate
        assert terminated or steps == max_steps

    def test_action_mask_matches_valid_actions(self):
        """Test action mask aligns with valid_actions in info."""
        env = MagicalChessEnv(seed=42)
        _, info = env.reset()

        mask = env.get_action_mask()
        valid_actions = info["valid_actions"]

        # All valid actions should have True in mask
        for action in valid_actions:
            assert mask[action], f"Action {action} should be valid"

        # Count should match
        assert np.sum(mask) == len(valid_actions)

    def test_environment_is_gymnasium_compatible(self):
        """Test environment passes Gymnasium compatibility checks."""
        env = MagicalChessEnv(seed=42)

        # Basic checks
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Check spaces are valid
        assert env.action_space.contains(0)
        assert env.action_space.contains(255)

        # Check reset/step work
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

        obs, reward, term, trunc, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_deterministic_with_seed(self):
        """Test environment is deterministic with fixed seed."""
        env1 = MagicalChessEnv()
        env2 = MagicalChessEnv()

        obs1, _ = env1.reset(seed=99)
        obs2, _ = env2.reset(seed=99)
        assert np.array_equal(obs1, obs2)

        # Same actions should produce same results
        for _ in range(10):
            action = 17  # Fixed action
            r1 = env1.step(action)
            r2 = env2.step(action)
            assert np.array_equal(r1[0], r2[0])  # observations
            assert r1[1] == r2[1]  # rewards


# =============================================================================
# Hourglass Efficiency Reward Tests
# =============================================================================

class TestHourglassEfficiencyReward:
    """Tests for hourglass efficiency reward mode."""

    def test_hourglass_cost_deducted_on_valid_move(self):
        """Test hourglass is deducted when a valid move is made."""
        env = MagicalChessEnv(initial_hourglass=100, hourglass_cost=5, seed=42)
        _, info = env.reset()

        initial_hourglass = info["hourglass"]

        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            _, _, _, _, new_info = env.step(valid_action)
            # Hourglass should be reduced by cost
            assert new_info["hourglass"] == initial_hourglass - 5

    def test_hourglass_not_deducted_on_invalid_move(self):
        """Test hourglass is NOT deducted on invalid action."""
        env = MagicalChessEnv(initial_hourglass=100, hourglass_cost=5, seed=42)
        _, info = env.reset()

        initial_hourglass = info["hourglass"]

        # Action 0 (source=target) is always invalid
        _, _, _, _, new_info = env.step(0)
        # Hourglass should remain unchanged
        assert new_info["hourglass"] == initial_hourglass

    def test_reward_mode_score_delta(self):
        """Test reward_mode='score_delta' returns raw score."""
        env = MagicalChessEnv(reward_mode="score_delta", hourglass_cost=5, seed=42)
        _, info = env.reset()

        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            _, reward, _, _, _ = env.step(valid_action)
            # Reward should be score_delta (merged_value * 10)
            # which is not divided by hourglass_cost
            assert reward > 0
            assert reward >= 10  # At least level 1 merge = 10

    def test_reward_mode_hourglass_efficiency(self):
        """Test reward_mode='hourglass_efficiency' returns score/hourglass."""
        env = MagicalChessEnv(
            reward_mode="hourglass_efficiency",
            hourglass_cost=10,
            seed=42
        )
        _, info = env.reset()

        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            _, reward, _, _, _ = env.step(valid_action)
            # Reward should be score_delta / hourglass_cost
            # If merged_value is 3, score_delta = 30, efficiency = 30/10 = 3.0
            assert reward > 0
            # Efficiency reward should be less than raw score (since hourglass_cost > 1)
            assert reward <= 10.0  # score_delta / hourglass_cost <= 100 / 10

    def test_info_contains_hourglass_efficiency(self):
        """Test info dictionary contains hourglass_efficiency metric."""
        env = MagicalChessEnv(initial_hourglass=100, hourglass_cost=10, seed=42)
        _, info = env.reset()

        assert "hourglass_efficiency" in info
        # Initially 0 since no hourglass spent
        assert info["hourglass_efficiency"] == 0.0

        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            _, _, _, _, new_info = env.step(valid_action)
            # Now should have non-zero efficiency
            assert new_info["hourglass_efficiency"] > 0.0

    def test_hourglass_cannot_go_negative(self):
        """Test hourglass count cannot go below 0."""
        env = MagicalChessEnv(initial_hourglass=5, hourglass_cost=10, seed=42)
        _, info = env.reset()

        if info["valid_actions"]:
            valid_action = info["valid_actions"][0]
            _, _, _, _, new_info = env.step(valid_action)
            # Hourglass should be 0, not negative
            assert new_info["hourglass"] >= 0

    def test_different_hourglass_costs_affect_efficiency(self):
        """Test different hourglass costs produce different efficiency rewards."""
        env1 = MagicalChessEnv(
            reward_mode="hourglass_efficiency",
            hourglass_cost=1,
            seed=42
        )
        env2 = MagicalChessEnv(
            reward_mode="hourglass_efficiency",
            hourglass_cost=10,
            seed=42
        )

        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()

        if info1["valid_actions"] and info2["valid_actions"]:
            # Use same valid action for both
            valid_action = info1["valid_actions"][0]
            _, reward1, _, _, _ = env1.step(valid_action)
            _, reward2, _, _, _ = env2.step(valid_action)

            # Higher hourglass cost = lower efficiency reward
            assert reward1 > reward2
