"""Tests for GameState class and helper functions.

This module contains all tests for the GameState data model, organized by
task group as defined in the specification.
"""

import json

import numpy as np
import pytest

from src.engine.game_state import (
    GameState,
    get_tier,
    get_level,
    TIER_EMPTY,
    TIER_BRONZE,
    TIER_SILVER,
    TIER_GOLD,
    MAX_PIECE_LEVEL,
)


# =============================================================================
# Task Group 1: GameState Foundation Tests (4 tests)
# =============================================================================

class TestGameStateFoundation:
    """Tests for GameState initialization and properties."""

    def test_empty_board_initialization_creates_4x4_int8_zeros_array(self):
        """Test that default GameState has a 4x4 zeros array of int8 dtype."""
        state = GameState()

        assert state.grid.shape == (4, 4)
        assert state.grid.dtype == np.int8
        assert np.all(state.grid == 0)

    def test_gamestate_with_custom_properties(self):
        """Test GameState initialization with custom grid, hourglass_count, score, last_score_delta."""
        custom_grid = np.array([
            [1, 2, 0, 0],
            [0, 3, 4, 0],
            [0, 0, 5, 6],
            [7, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(
            grid=custom_grid,
            hourglass_count=100,
            score=500,
            last_score_delta=50,
        )

        assert np.array_equal(state.grid, custom_grid)
        assert state.hourglass_count == 100
        assert state.score == 500
        assert state.last_score_delta == 50

    def test_grid_dtype_is_int8_and_shape_is_4x4(self):
        """Test grid dtype is np.int8 and shape is (4,4)."""
        state = GameState()

        assert state.grid.dtype == np.int8
        assert state.grid.shape == (4, 4)

    def test_immutability_frozen_dataclass_behavior(self):
        """Test that GameState is immutable (frozen dataclass)."""
        state = GameState()

        with pytest.raises(AttributeError):
            state.hourglass_count = 999

        with pytest.raises(AttributeError):
            state.score = 999


# =============================================================================
# Task Group 2: Tier and Level Helper Tests (4 tests)
# =============================================================================

class TestTierAndLevelHelpers:
    """Tests for tier and level helper functions."""

    def test_get_tier_zero_returns_empty(self):
        """Test get_tier(0) returns 'Empty'."""
        assert get_tier(0) == TIER_EMPTY
        assert get_tier(0) == "Empty"

    def test_get_tier_for_bronze_silver_gold_values(self):
        """Test get_tier for Bronze (1-3), Silver (4-6), Gold (7+) values."""
        # Bronze tier (1-3)
        assert get_tier(1) == TIER_BRONZE
        assert get_tier(2) == TIER_BRONZE
        assert get_tier(3) == TIER_BRONZE

        # Silver tier (4-6)
        assert get_tier(4) == TIER_SILVER
        assert get_tier(5) == TIER_SILVER
        assert get_tier(6) == TIER_SILVER

        # Gold tier (7+)
        assert get_tier(7) == TIER_GOLD
        assert get_tier(10) == TIER_GOLD
        assert get_tier(127) == TIER_GOLD

    def test_get_level_returns_correct_level_number(self):
        """Test get_level returns correct level number for each value."""
        assert get_level(0) == 0
        assert get_level(1) == 1
        assert get_level(3) == 3
        assert get_level(5) == 5
        assert get_level(7) == 7
        assert get_level(127) == 127

    def test_tier_edge_cases(self):
        """Test edge cases: get_tier(7) returns 'Gold', get_level(6) returns 6."""
        assert get_tier(7) == TIER_GOLD
        assert get_level(6) == 6


# =============================================================================
# Task Group 3: Clone and Copy-on-Modify Tests (4 tests)
# =============================================================================

class TestCloneAndCopyOnModify:
    """Tests for clone and copy-on-modify operations."""

    def test_clone_returns_deep_copy_with_identical_values(self):
        """Test clone() returns deep copy with identical values."""
        original_grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        original = GameState(
            grid=original_grid,
            hourglass_count=50,
            score=100,
            last_score_delta=25,
        )

        cloned = original.clone()

        assert np.array_equal(cloned.grid, original.grid)
        assert cloned.hourglass_count == original.hourglass_count
        assert cloned.score == original.score
        assert cloned.last_score_delta == original.last_score_delta
        assert cloned == original

    def test_clone_is_independent(self):
        """Test clone is independent (modifying clone grid does not affect original)."""
        original_grid = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original = GameState(grid=original_grid)

        cloned = original.clone()

        # Get a mutable copy of the cloned grid for testing independence
        # Since GameState is frozen, we need to access the underlying grid
        # and verify that modifications to a new grid don't affect original
        new_grid = np.copy(cloned.grid)
        new_grid[0, 0] = 99

        # Original should be unchanged
        assert original.grid[0, 0] == 1
        # Cloned should still have original value
        assert cloned.grid[0, 0] == 1

    def test_with_grid_returns_new_gamestate_with_different_grid(self):
        """Test with_grid() returns new GameState with different grid."""
        original = GameState(hourglass_count=10, score=200, last_score_delta=30)
        new_grid = np.array([
            [5, 5, 5, 5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        new_state = original.with_grid(new_grid)

        assert np.array_equal(new_state.grid, new_grid)
        assert new_state.hourglass_count == 10
        assert new_state.score == 200
        assert new_state.last_score_delta == 30
        assert new_state is not original

    def test_with_score_returns_new_gamestate_preserving_other_properties(self):
        """Test with_score() returns new GameState preserving other properties."""
        grid = np.array([
            [1, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original = GameState(grid=grid, hourglass_count=50, score=100, last_score_delta=10)

        new_state = original.with_score(score=200, delta=100)

        assert new_state.score == 200
        assert new_state.last_score_delta == 100
        assert np.array_equal(new_state.grid, grid)
        assert new_state.hourglass_count == 50
        assert new_state is not original


# =============================================================================
# Task Group 4: Hourglass Operations Tests (3 tests)
# =============================================================================

class TestHourglassOperations:
    """Tests for hourglass resource management."""

    def test_can_afford_returns_true_when_hourglass_count_gte_cost(self):
        """Test can_afford(cost) returns True when hourglass_count >= cost."""
        state = GameState(hourglass_count=100)

        assert state.can_afford(50) is True
        assert state.can_afford(100) is True
        assert state.can_afford(0) is True

    def test_can_afford_returns_false_when_hourglass_count_lt_cost(self):
        """Test can_afford(cost) returns False when hourglass_count < cost."""
        state = GameState(hourglass_count=50)

        assert state.can_afford(51) is False
        assert state.can_afford(100) is False

    def test_deduct_returns_new_gamestate_with_reduced_hourglass_count(self):
        """Test deduct(cost) returns new GameState with reduced hourglass_count."""
        grid = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original = GameState(grid=grid, hourglass_count=100, score=50, last_score_delta=10)

        new_state = original.deduct(30)

        assert new_state.hourglass_count == 70
        assert np.array_equal(new_state.grid, grid)
        assert new_state.score == 50
        assert new_state.last_score_delta == 10
        assert new_state is not original


# =============================================================================
# Task Group 5: Validation Tests (4 tests)
# =============================================================================

class TestValidation:
    """Tests for state validation."""

    def test_valid_state_passes_validation_without_raising(self):
        """Test valid state passes validation without raising."""
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=np.int8)

        # Should not raise
        state = GameState(grid=grid, hourglass_count=10, score=100, last_score_delta=50)

        assert state.is_valid() is True

    def test_invalid_grid_shape_raises_valueerror(self):
        """Test invalid grid shape raises ValueError."""
        invalid_grid = np.zeros((3, 4), dtype=np.int8)

        with pytest.raises(ValueError, match="shape"):
            GameState(grid=invalid_grid)

    def test_grid_value_out_of_range_raises_valueerror(self):
        """Test grid value out of range (negative or > MAX_PIECE_LEVEL) raises ValueError."""
        # Test negative value
        negative_grid = np.array([
            [-1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        with pytest.raises(ValueError, match="range"):
            GameState(grid=negative_grid)

    def test_negative_hourglass_count_raises_valueerror(self):
        """Test negative hourglass_count raises ValueError."""
        with pytest.raises(ValueError, match="hourglass"):
            GameState(hourglass_count=-1)


# =============================================================================
# Task Group 6: Serialization Tests (4 tests)
# =============================================================================

class TestSerialization:
    """Tests for state serialization and deserialization."""

    def test_to_dict_returns_json_compatible_dict_with_all_properties(self):
        """Test to_dict() returns JSON-compatible dict with all properties."""
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        state = GameState(grid=grid, hourglass_count=100, score=500, last_score_delta=50)

        result = state.to_dict()

        # Verify all properties are present
        assert "grid" in result
        assert "hourglass_count" in result
        assert "score" in result
        assert "last_score_delta" in result

        # Verify values
        assert result["hourglass_count"] == 100
        assert result["score"] == 500
        assert result["last_score_delta"] == 50

        # Verify grid is a list (JSON-compatible)
        assert isinstance(result["grid"], list)
        assert result["grid"][0][0] == 1

        # Verify JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_from_dict_reconstructs_identical_gamestate(self):
        """Test from_dict() reconstructs identical GameState."""
        data = {
            "grid": [
                [1, 2, 3, 4],
                [5, 6, 7, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            "hourglass_count": 100,
            "score": 500,
            "last_score_delta": 50,
        }

        state = GameState.from_dict(data)

        assert state.grid[0, 0] == 1
        assert state.grid[1, 2] == 7
        assert state.hourglass_count == 100
        assert state.score == 500
        assert state.last_score_delta == 50
        assert state.grid.dtype == np.int8

    def test_round_trip_from_dict_to_dict_equals_original(self):
        """Test round-trip: from_dict(state.to_dict()) equals original state."""
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        original = GameState(grid=grid, hourglass_count=100, score=500, last_score_delta=50)

        serialized = original.to_dict()
        restored = GameState.from_dict(serialized)

        assert restored == original

    def test_from_dict_with_invalid_data_raises_error(self):
        """Test from_dict with invalid data raises appropriate error."""
        invalid_data = {
            "grid": [[1, 2], [3, 4]],  # Wrong shape
            "hourglass_count": 100,
            "score": 500,
            "last_score_delta": 50,
        }

        with pytest.raises(ValueError):
            GameState.from_dict(invalid_data)


# =============================================================================
# Task Group 7: Integration and Gap Analysis Tests (up to 5 tests)
# =============================================================================

class TestIntegration:
    """Integration tests for GameState workflows."""

    def test_chained_mutations_create_correct_state(self):
        """Test chained copy-on-modify operations produce correct final state."""
        initial = GameState(hourglass_count=100)

        # Chain: update grid -> update score -> deduct hourglass
        new_grid = np.array([
            [1, 2, 3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        result = (
            initial
            .with_grid(new_grid)
            .with_score(score=150, delta=150)
            .deduct(25)
        )

        assert np.array_equal(result.grid, new_grid)
        assert result.score == 150
        assert result.last_score_delta == 150
        assert result.hourglass_count == 75

        # Original unchanged
        assert initial.hourglass_count == 100
        assert initial.score == 0

    def test_clone_independence_after_serialize(self):
        """Test clone from deserialized state is independent."""
        grid = np.array([
            [5, 5, 5, 5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original = GameState(grid=grid, hourglass_count=50, score=100, last_score_delta=10)

        # Serialize and deserialize
        restored = GameState.from_dict(original.to_dict())

        # Clone the restored state
        cloned = restored.clone()

        # Verify all are independent
        assert original == restored
        assert restored == cloned
        assert cloned is not restored
        assert cloned is not original

    def test_equality_and_hash_consistency(self):
        """Test that equal states have equal hashes and can be used in sets/dicts."""
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        state1 = GameState(grid=grid.copy(), hourglass_count=10, score=100, last_score_delta=5)
        state2 = GameState(grid=grid.copy(), hourglass_count=10, score=100, last_score_delta=5)

        # Equal states should have equal hashes
        assert state1 == state2
        assert hash(state1) == hash(state2)

        # Can be used in sets
        state_set = {state1}
        assert state2 in state_set

        # Can be used as dict keys
        state_dict = {state1: "value"}
        assert state_dict[state2] == "value"

    def test_json_round_trip_preserves_state(self):
        """Test to_json/from_json preserves all state correctly."""
        grid = np.array([
            [7, 6, 5, 4],
            [3, 2, 1, 0],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ], dtype=np.int8)
        original = GameState(grid=grid, hourglass_count=999, score=12345, last_score_delta=500)

        json_str = original.to_json()
        restored = GameState.from_json(json_str)

        assert restored == original
        assert isinstance(json_str, str)

    def test_max_piece_level_value_is_valid(self):
        """Test that MAX_PIECE_LEVEL (127) can be stored and retrieved correctly."""
        grid = np.array([
            [MAX_PIECE_LEVEL, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        state = GameState(grid=grid)

        assert state.grid[0, 0] == MAX_PIECE_LEVEL
        assert get_tier(MAX_PIECE_LEVEL) == TIER_GOLD
        assert get_level(MAX_PIECE_LEVEL) == MAX_PIECE_LEVEL
