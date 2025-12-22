"""Tests for move_system module.

This module contains all tests for the drag-drop move system, organized by
task group as defined in the specification.
"""

import numpy as np
import pytest

from src.engine.game_state import GameState
from src.engine.move_system import (
    find_merge_cluster,
    is_valid_move,
    get_valid_moves,
    execute_move,
    get_empty_cells,
    is_game_over,
    spawn_random_tile,
    _execute_merge_3,
    _execute_merge_5,
    _select_and_execute_merge,
)


# =============================================================================
# Task Group 1: Merge Cluster Detection Tests (4 tests)
# =============================================================================

class TestMergeClusterDetection:
    """Tests for find_merge_cluster BFS algorithm."""

    def test_bfs_finds_single_isolated_piece(self):
        """Test BFS finds single isolated piece (returns list with just that position)."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (0, 0))

        assert cluster == [(0, 0)]

    def test_bfs_finds_horizontal_cluster_of_3(self):
        """Test BFS finds horizontal cluster of 3 matching pieces."""
        grid = np.array([
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (0, 0))

        assert len(cluster) == 3
        assert (0, 0) in cluster
        assert (0, 1) in cluster
        assert (0, 2) in cluster

    def test_bfs_finds_l_shaped_cluster_of_4(self):
        """Test BFS finds L-shaped cluster of 4 matching pieces."""
        grid = np.array([
            [3, 3, 0, 0],
            [3, 0, 0, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (0, 0))

        assert len(cluster) == 4
        assert (0, 0) in cluster
        assert (0, 1) in cluster
        assert (1, 0) in cluster
        assert (2, 0) in cluster

    def test_bfs_excludes_diagonal_neighbors(self):
        """Test BFS excludes diagonal neighbors (orthogonal only)."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Each piece should be isolated (diagonals don't count)
        cluster_00 = find_merge_cluster(state, (0, 0))
        cluster_11 = find_merge_cluster(state, (1, 1))
        cluster_22 = find_merge_cluster(state, (2, 2))

        assert cluster_00 == [(0, 0)]
        assert cluster_11 == [(1, 1)]
        assert cluster_22 == [(2, 2)]


# =============================================================================
# Task Group 2: Merge-3 and Merge-5 Logic Tests (5 tests)
# =============================================================================

class TestMergeLogic:
    """Tests for merge execution logic."""

    def test_merge_3_produces_1_upgraded_piece(self):
        """Test Merge-3: exactly 3 matching pieces produces 1 upgraded piece."""
        grid = np.array([
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        cluster = [(0, 0), (0, 1), (0, 2)]
        target = (0, 1)

        new_grid = _execute_merge_3(grid, cluster, target)

        # All 3 positions cleared
        assert new_grid[0, 0] == 0
        assert new_grid[0, 2] == 0
        # Target has upgraded piece (level 2 -> level 3)
        assert new_grid[0, 1] == 3

    def test_merge_3_with_4_pieces_still_consumes_3(self):
        """Test Merge-3: 4 matching pieces still triggers Merge-3 (consumes 3)."""
        grid = np.array([
            [2, 2, 0, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        # BFS order from (0,0): (0,0), (0,1), (1,0), (1,1)
        cluster = [(0, 0), (0, 1), (1, 0), (1, 1)]
        target = (0, 0)

        new_grid = _execute_merge_3(grid, cluster, target)

        # First 3 positions cleared, 4th remains
        assert new_grid[0, 0] == 3  # Target has upgraded piece
        assert new_grid[0, 1] == 0  # Consumed
        assert new_grid[1, 0] == 0  # Consumed
        assert new_grid[1, 1] == 2  # Not consumed (4th piece)

    def test_merge_5_produces_2_upgraded_pieces(self):
        """Test Merge-5: 5+ matching pieces produces 2 upgraded pieces."""
        grid = np.array([
            [2, 2, 2, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        # 5 pieces in cluster
        cluster = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        target = (0, 0)

        new_grid = _execute_merge_5(grid, cluster, target)

        # Count upgraded pieces (level 3)
        upgraded_count = np.sum(new_grid == 3)
        assert upgraded_count == 2

        # Target has first upgraded piece
        assert new_grid[0, 0] == 3

    def test_merge_5_second_piece_at_first_empty_in_bfs_order(self):
        """Test Merge-5 second piece placement at first empty cell in BFS order."""
        grid = np.array([
            [2, 2, 2, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        # BFS order: (0,0), (0,1), (0,2), (1,0), (1,1)
        cluster = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        target = (0, 0)

        new_grid = _execute_merge_5(grid, cluster, target)

        # First upgraded at target
        assert new_grid[0, 0] == 3
        # Second upgraded at first empty in cluster (excluding target) = (0,1)
        assert new_grid[0, 1] == 3
        # Other positions cleared
        assert new_grid[0, 2] == 0
        assert new_grid[1, 0] == 0
        assert new_grid[1, 1] == 0

    def test_tier_transitions_b3_to_s4_and_s6_to_g7(self):
        """Test tier transitions: B3+B3+B3=S4, S6+S6+S6=G7."""
        # Test B3 -> S4
        grid_bronze = np.array([
            [3, 3, 3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        cluster_bronze = [(0, 0), (0, 1), (0, 2)]

        new_grid_bronze = _execute_merge_3(grid_bronze, cluster_bronze, (0, 0))
        assert new_grid_bronze[0, 0] == 4  # Silver 4

        # Test S6 -> G7
        grid_silver = np.array([
            [6, 6, 6, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        cluster_silver = [(0, 0), (0, 1), (0, 2)]

        new_grid_silver = _execute_merge_3(grid_silver, cluster_silver, (0, 0))
        assert new_grid_silver[0, 0] == 7  # Gold 7


# =============================================================================
# Task Group 3: Move Validation Tests (4 tests)
# =============================================================================

class TestMoveValidation:
    """Tests for move validation functions."""

    def test_is_valid_move_true_when_creates_cluster_of_3(self):
        """Test is_valid_move returns True when move creates cluster of 3+ pieces."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Moving (2,1) to (1,1) creates cluster of 3: (0,0), (0,1), (1,1)
        result = is_valid_move(state, (2, 1), (1, 1))

        assert result is True

    def test_is_valid_move_false_when_cluster_less_than_3(self):
        """Test is_valid_move returns False when move creates cluster < 3 pieces."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Moving (2,1) to (0,1) creates cluster of only 2
        result = is_valid_move(state, (2, 1), (0, 1))

        assert result is False

    def test_is_valid_move_false_for_empty_source(self):
        """Test is_valid_move returns False for empty source cell."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Source (1, 0) is empty
        result = is_valid_move(state, (1, 0), (1, 1))

        assert result is False

    def test_get_valid_moves_returns_correct_pairs(self):
        """Test get_valid_moves returns correct list of (source, target) pairs."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        moves = get_valid_moves(state)

        # Should include the valid move we identified
        assert ((2, 1), (1, 1)) in moves

        # All returned moves should be valid
        for source, target in moves:
            assert is_valid_move(state, source, target)


# =============================================================================
# Task Group 4: Execute Move Tests (4 tests)
# =============================================================================

class TestExecuteMove:
    """Tests for execute_move function."""

    def test_execute_move_with_valid_merge_3(self):
        """Test execute_move with valid Merge-3 returns correct new GameState."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid, hourglass_count=100, score=50)

        new_state = execute_move(state, (2, 1), (1, 1))

        # Target has upgraded piece
        assert new_state.grid[1, 1] == 3
        # Source and cluster pieces cleared
        assert new_state.grid[0, 0] == 0
        assert new_state.grid[0, 1] == 0
        assert new_state.grid[2, 1] == 0
        # Other state preserved
        assert new_state.hourglass_count == 100
        assert new_state.score == 50

    def test_execute_move_with_valid_merge_5(self):
        """Test execute_move with valid Merge-5 returns correct new GameState."""
        grid = np.array([
            [2, 2, 2, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Move (2,0) to (1,1) - this creates a cluster at (1,1) only with the moved piece
        # Actually let's use a better example - move a piece to join 4 others
        # Moving piece from (2,0) to (0,3) won't work as they're not connected

        # Better setup: 4 pieces plus 1 to move
        grid2 = np.array([
            [2, 2, 0, 0],
            [2, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state2 = GameState(grid=grid2)

        # Move (2,2) to (0,2) - creates cluster at top: (0,0), (0,1), (0,2), (1,0), (1,1)
        new_state = execute_move(state2, (2, 2), (0, 2))

        # Count upgraded pieces
        upgraded_count = np.sum(new_state.grid == 3)
        assert upgraded_count == 2

    def test_execute_move_clears_source_and_updates_target(self):
        """Test execute_move clears source cell and updates target cell."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        new_state = execute_move(state, (2, 1), (1, 1))

        # Source cell (2,1) should be cleared
        assert new_state.grid[2, 1] == 0
        # Target cell (1,1) should have upgraded piece
        assert new_state.grid[1, 1] == 3

    def test_execute_move_returns_immutable_new_state(self):
        """Test execute_move returns immutable new state (original unchanged)."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original_grid = np.copy(grid)
        state = GameState(grid=grid)

        new_state = execute_move(state, (2, 1), (1, 1))

        # Original state unchanged
        assert np.array_equal(state.grid, original_grid)
        # New state is different object
        assert new_state is not state
        # New state has different grid
        assert not np.array_equal(new_state.grid, state.grid)


# =============================================================================
# Task Group 5: Integration Tests (up to 8 additional tests)
# =============================================================================

class TestIntegration:
    """Integration tests for complete move workflows."""

    def test_full_workflow_move_to_cluster_to_merge(self):
        """Test complete workflow: valid move -> cluster detection -> merge execution."""
        grid = np.array([
            [3, 3, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Verify move is valid
        assert is_valid_move(state, (2, 1), (1, 1))

        # Execute move
        new_state = execute_move(state, (2, 1), (1, 1))

        # Verify result: B3 -> S4
        assert new_state.grid[1, 1] == 4
        assert new_state.grid[0, 0] == 0
        assert new_state.grid[0, 1] == 0

    def test_move_to_same_tier_different_level_no_merge(self):
        """Test moving to same-tier different-level does not create merge."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],  # Level 3 (Bronze), different from level 2
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Moving level 3 piece to join level 2 pieces should not be valid
        result = is_valid_move(state, (2, 1), (1, 1))

        assert result is False

    def test_cluster_at_board_corner(self):
        """Test cluster detection at board corner position."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (2, 3))

        assert len(cluster) == 3
        assert (2, 2) in cluster
        assert (2, 3) in cluster
        assert (3, 2) in cluster

    def test_cluster_at_board_edge(self):
        """Test cluster detection along board edge."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 2, 2, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (3, 1))

        assert len(cluster) == 3
        assert (3, 0) in cluster
        assert (3, 1) in cluster
        assert (3, 2) in cluster

    def test_execute_move_raises_for_invalid_move(self):
        """Test execute_move raises ValueError for invalid move."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        with pytest.raises(ValueError, match="Invalid move"):
            execute_move(state, (2, 1), (0, 1))

    def test_get_valid_moves_empty_for_no_possible_merges(self):
        """Test get_valid_moves returns empty list when no merges possible."""
        grid = np.array([
            [1, 0, 2, 0],
            [0, 0, 0, 0],
            [3, 0, 4, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        moves = get_valid_moves(state)

        assert moves == []

    def test_bfs_empty_cell_returns_empty_list(self):
        """Test find_merge_cluster on empty cell returns empty list."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        cluster = find_merge_cluster(state, (1, 1))

        assert cluster == []

    def test_bfs_out_of_bounds_returns_empty_list(self):
        """Test find_merge_cluster on out-of-bounds position returns empty list."""
        grid = np.zeros((4, 4), dtype=np.int8)
        state = GameState(grid=grid)

        assert find_merge_cluster(state, (-1, 0)) == []
        assert find_merge_cluster(state, (0, -1)) == []
        assert find_merge_cluster(state, (4, 0)) == []
        assert find_merge_cluster(state, (0, 4)) == []


# =============================================================================
# Game Over Detection Tests
# =============================================================================

class TestGameOverDetection:
    """Tests for get_empty_cells and is_game_over functions."""

    def test_get_empty_cells_returns_correct_positions(self):
        """Test get_empty_cells returns all empty positions for partially filled board."""
        grid = np.array([
            [2, 0, 3, 0],
            [0, 1, 0, 4],
            [5, 0, 0, 0],
            [0, 6, 0, 7],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        empty_cells = get_empty_cells(state)

        # There are 9 empty cells
        assert len(empty_cells) == 9
        # Verify specific empty positions
        assert (0, 1) in empty_cells
        assert (0, 3) in empty_cells
        assert (1, 0) in empty_cells
        assert (1, 2) in empty_cells
        assert (2, 1) in empty_cells
        assert (2, 2) in empty_cells
        assert (2, 3) in empty_cells
        assert (3, 0) in empty_cells
        assert (3, 2) in empty_cells
        # Verify non-empty positions are not included
        assert (0, 0) not in empty_cells
        assert (0, 2) not in empty_cells
        assert (1, 1) not in empty_cells

    def test_get_empty_cells_returns_empty_list_when_board_full(self):
        """Test get_empty_cells returns empty list when board is completely full."""
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        empty_cells = get_empty_cells(state)

        assert empty_cells == []

    def test_is_game_over_false_when_empty_cells_exist(self):
        """Test is_game_over returns False when board has empty cells."""
        grid = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        result = is_game_over(state)

        assert result is False

    def test_is_game_over_false_when_valid_moves_exist(self):
        """Test is_game_over returns False when board is full but merges are possible."""
        # Full board with three adjacent matching pieces (2, 2, 2 in first row)
        grid = np.array([
            [2, 2, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        result = is_game_over(state)

        # Valid merge exists (three 2s in first row)
        assert result is False

    def test_is_game_over_true_when_board_full_no_valid_merges(self):
        """Test is_game_over returns True when board is full with no valid merges."""
        # All unique values - no possible merges since no three matching pieces exist
        grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        result = is_game_over(state)

        assert result is True

    def test_functions_are_pure_no_state_mutation(self):
        """Test both get_empty_cells and is_game_over are pure (no state mutation)."""
        grid = np.array([
            [2, 0, 3, 0],
            [0, 1, 0, 4],
            [5, 0, 0, 0],
            [0, 6, 0, 7],
        ], dtype=np.int8)
        original_grid = np.copy(grid)
        state = GameState(grid=grid)

        # Call both functions
        get_empty_cells(state)
        is_game_over(state)

        # Verify grid was not mutated
        assert np.array_equal(state.grid, original_grid)


# =============================================================================
# Random Tile Spawning Tests
# =============================================================================

class TestRandomTileSpawning:
    """Tests for spawn_random_tile function."""

    def test_spawn_adds_one_tile_to_empty_cell(self):
        """Test spawn_random_tile adds exactly one tile to an empty cell."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)
        rng = np.random.default_rng(42)

        new_state = spawn_random_tile(state, rng)

        # Count pieces before and after
        pieces_before = np.sum(state.grid > 0)
        pieces_after = np.sum(new_state.grid > 0)
        assert pieces_after == pieces_before + 1

    def test_spawn_produces_b1_or_b2_only(self):
        """Test spawn_random_tile only produces B1 (1) or B2 (2) tiles."""
        grid = np.zeros((4, 4), dtype=np.int8)
        state = GameState(grid=grid)

        # Spawn many tiles and verify all are 1 or 2
        for seed in range(100):
            rng = np.random.default_rng(seed)
            new_state = spawn_random_tile(state, rng)
            spawned_value = new_state.grid[new_state.grid > 0][0]
            assert spawned_value in [1, 2], f"Spawned value {spawned_value} not in [1, 2]"

    def test_spawn_with_seeded_rng_is_deterministic(self):
        """Test spawn_random_tile with same seed produces same result."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        state = GameState(grid=grid)

        # Same seed should produce same result
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        new_state1 = spawn_random_tile(state, rng1)
        new_state2 = spawn_random_tile(state, rng2)

        assert np.array_equal(new_state1.grid, new_state2.grid)

    def test_spawn_raises_when_board_full(self):
        """Test spawn_random_tile raises ValueError when board is full."""
        grid = np.ones((4, 4), dtype=np.int8)
        state = GameState(grid=grid)
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="Cannot spawn tile: board is full"):
            spawn_random_tile(state, rng)

    def test_spawn_is_pure_no_state_mutation(self):
        """Test spawn_random_tile is pure (does not mutate original state)."""
        grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        original_grid = np.copy(grid)
        state = GameState(grid=grid)
        rng = np.random.default_rng(42)

        new_state = spawn_random_tile(state, rng)

        # Original state unchanged
        assert np.array_equal(state.grid, original_grid)
        # New state is different object
        assert new_state is not state

    def test_spawn_preserves_other_state_properties(self):
        """Test spawn_random_tile preserves hourglass_count, score, etc."""
        grid = np.zeros((4, 4), dtype=np.int8)
        state = GameState(grid=grid, hourglass_count=100, score=500, last_score_delta=25)
        rng = np.random.default_rng(42)

        new_state = spawn_random_tile(state, rng)

        assert new_state.hourglass_count == 100
        assert new_state.score == 500
        assert new_state.last_score_delta == 25
