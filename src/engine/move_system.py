"""Move system for drag-and-drop piece movement in Magical Chess ML.

This module provides the core move and merge logic for drag-and-drop piece
movement, enabling piece selection (source tile) and drop target mechanics
with automatic merge cluster detection and execution.

Key functions:
- find_merge_cluster: BFS-based connected component detection for matching pieces
- is_valid_move: Check if a specific move would create a valid merge cluster
- get_valid_moves: Return all valid (source, target) pairs for RL action masking
- execute_move: Execute a move and perform the resulting merge
- get_empty_cells: Return list of empty cell positions
- is_game_over: Detect terminal game state (no empty cells and no valid moves)
- spawn_random_tile: Spawn a new B1/B2 tile at a random empty position
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .game_state import GameState, get_tier, get_level


# =============================================================================
# Merge Cluster Detection (Task Group 1)
# =============================================================================

def find_merge_cluster(
    state: GameState,
    position: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find all pieces connected to the given position that can merge.

    Uses breadth-first search (BFS) to find all orthogonally adjacent pieces
    that have the same tier AND same level as the piece at the given position.
    The result is returned in BFS traversal order for deterministic behavior
    in RL training.

    Args:
        state: The current game state.
        position: The (row, col) position to start searching from.

    Returns:
        A list of (row, col) positions in BFS traversal order. Returns an
        empty list if the position is empty, out of bounds, or contains
        no matching neighbors.

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 2, 0, 0],
        ...     [2, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> cluster = find_merge_cluster(state, (0, 0))
        >>> len(cluster)
        3
        >>> (0, 0) in cluster
        True
        >>> (0, 1) in cluster
        True
        >>> (1, 0) in cluster
        True

        >>> # Empty cell returns empty list
        >>> find_merge_cluster(state, (3, 3))
        []

        >>> # Out of bounds returns empty list
        >>> find_merge_cluster(state, (-1, 0))
        []
    """
    row, col = position

    # Handle out of bounds
    if row < 0 or row >= 4 or col < 0 or col >= 4:
        return []

    # Handle empty cell
    start_value = state.grid[row, col]
    if start_value == 0:
        return []

    # Get tier and level for matching
    start_tier = get_tier(start_value)
    start_level = get_level(start_value)

    # BFS traversal
    visited: set[Tuple[int, int]] = set()
    cluster: List[Tuple[int, int]] = []
    queue: deque[Tuple[int, int]] = deque()

    queue.append(position)
    visited.add(position)

    # Orthogonal directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        curr_row, curr_col = queue.popleft()
        cluster.append((curr_row, curr_col))

        for dr, dc in directions:
            new_row, new_col = curr_row + dr, curr_col + dc

            # Check bounds
            if new_row < 0 or new_row >= 4 or new_col < 0 or new_col >= 4:
                continue

            # Check if already visited
            if (new_row, new_col) in visited:
                continue

            # Check if matching piece
            neighbor_value = state.grid[new_row, new_col]
            if neighbor_value == 0:
                continue

            neighbor_tier = get_tier(neighbor_value)
            neighbor_level = get_level(neighbor_value)

            if neighbor_tier == start_tier and neighbor_level == start_level:
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))

    return cluster


# =============================================================================
# Internal Merge Execution (Task Group 2)
# =============================================================================

def _get_next_level(current_value: int) -> int:
    """Calculate the next level value after a merge.

    Handles tier transitions: B3 -> S4, S6 -> G7.

    Args:
        current_value: The current piece value (level).

    Returns:
        The next level value.
    """
    return current_value + 1


def _execute_merge_3(
    grid: np.ndarray,
    cluster: List[Tuple[int, int]],
    target: Tuple[int, int]
) -> np.ndarray:
    """Execute a Merge-3 operation on the grid.

    Consumes 3 pieces from the cluster and produces 1 upgraded piece at the
    target position.

    Args:
        grid: The current grid (will not be mutated).
        cluster: List of positions in BFS order, must have at least 3 elements.
        target: The position where the upgraded piece will be placed.

    Returns:
        A new grid array with the merge applied.

    Raises:
        ValueError: If cluster has fewer than 3 pieces.
    """
    if len(cluster) < 3:
        raise ValueError(f"Merge-3 requires at least 3 pieces, got {len(cluster)}")

    new_grid = np.copy(grid)

    # Get the piece value (same for all in cluster)
    first_pos = cluster[0]
    piece_value = grid[first_pos[0], first_pos[1]]
    next_level = _get_next_level(piece_value)

    # Clear the first 3 pieces in BFS order
    for pos in cluster[:3]:
        new_grid[pos[0], pos[1]] = 0

    # Place upgraded piece at target
    new_grid[target[0], target[1]] = next_level

    return new_grid


def _execute_merge_5(
    grid: np.ndarray,
    cluster: List[Tuple[int, int]],
    target: Tuple[int, int]
) -> np.ndarray:
    """Execute a Merge-5 operation on the grid.

    Consumes 5 pieces from the cluster and produces 2 upgraded pieces.
    First piece at target, second at first empty cell in BFS order.

    Args:
        grid: The current grid (will not be mutated).
        cluster: List of positions in BFS order, must have at least 5 elements.
        target: The position where the first upgraded piece will be placed.

    Returns:
        A new grid array with the merge applied.

    Raises:
        ValueError: If cluster has fewer than 5 pieces.
    """
    if len(cluster) < 5:
        raise ValueError(f"Merge-5 requires at least 5 pieces, got {len(cluster)}")

    new_grid = np.copy(grid)

    # Get the piece value (same for all in cluster)
    first_pos = cluster[0]
    piece_value = grid[first_pos[0], first_pos[1]]
    next_level = _get_next_level(piece_value)

    # Clear the first 5 pieces in BFS order
    for pos in cluster[:5]:
        new_grid[pos[0], pos[1]] = 0

    # Place first upgraded piece at target
    new_grid[target[0], target[1]] = next_level

    # Find first empty cell in BFS order for second piece
    # The positions we just cleared are now empty, find the first one
    # that is not the target position
    second_pos = None
    for pos in cluster[:5]:
        if pos != target:
            second_pos = pos
            break

    if second_pos is not None:
        new_grid[second_pos[0], second_pos[1]] = next_level

    return new_grid


def _select_and_execute_merge(
    grid: np.ndarray,
    cluster: List[Tuple[int, int]],
    target: Tuple[int, int]
) -> np.ndarray:
    """Select and execute the appropriate merge based on cluster size.

    Automatically selects the largest possible merge:
    - If cluster size >= 5: execute Merge-5
    - If cluster size 3-4: execute Merge-3
    - If cluster size < 3: raise ValueError

    Args:
        grid: The current grid (will not be mutated).
        cluster: List of positions in BFS order.
        target: The position where the first upgraded piece will be placed.

    Returns:
        A new grid array with the merge applied.

    Raises:
        ValueError: If cluster has fewer than 3 pieces.
    """
    if len(cluster) < 3:
        raise ValueError(f"Invalid merge: cluster size {len(cluster)} < 3")

    if len(cluster) >= 5:
        return _execute_merge_5(grid, cluster, target)
    else:
        return _execute_merge_3(grid, cluster, target)


# =============================================================================
# Move Validation (Task Group 3)
# =============================================================================

def is_valid_move(
    state: GameState,
    source: Tuple[int, int],
    target: Tuple[int, int]
) -> bool:
    """Check if a move from source to target would create a valid merge.

    A move is valid if:
    - Source contains a non-empty piece
    - Target is within bounds
    - Source and target are different positions
    - Placing the source piece at target creates a cluster of 3+ pieces

    This function does not mutate the state; it simulates the move to check
    validity.

    Args:
        state: The current game state.
        source: The (row, col) position of the piece to move.
        target: The (row, col) position to move the piece to.

    Returns:
        True if the move would create a valid merge cluster, False otherwise.

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> # Moving (2,1) to (1,1) creates cluster of 3
        >>> is_valid_move(state, (2, 1), (1, 1))
        True
        >>> # Moving to (0,2) doesn't create cluster of 3
        >>> is_valid_move(state, (2, 1), (0, 2))
        False
    """
    src_row, src_col = source
    tgt_row, tgt_col = target

    # Check source is in bounds
    if src_row < 0 or src_row >= 4 or src_col < 0 or src_col >= 4:
        return False

    # Check target is in bounds
    if tgt_row < 0 or tgt_row >= 4 or tgt_col < 0 or tgt_col >= 4:
        return False

    # Check source is not empty
    source_value = state.grid[src_row, src_col]
    if source_value == 0:
        return False

    # Check source and target are different
    if source == target:
        return False

    # Simulate the move: create a temporary grid with piece moved
    temp_grid = np.copy(state.grid)
    temp_grid[src_row, src_col] = 0
    temp_grid[tgt_row, tgt_col] = source_value

    # Create temporary state for cluster detection
    temp_state = state.with_grid(temp_grid)

    # Find cluster at target position
    cluster = find_merge_cluster(temp_state, target)

    # Valid if cluster has 3+ pieces
    return len(cluster) >= 3


def get_valid_moves(
    state: GameState
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get all valid (source, target) move pairs for RL action masking.

    Iterates through all non-empty source positions and all target positions
    to find moves that would create valid merge clusters.

    Args:
        state: The current game state.

    Returns:
        A list of ((src_row, src_col), (tgt_row, tgt_col)) tuples representing
        valid moves. Suitable for RL action masking.

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> moves = get_valid_moves(state)
        >>> ((2, 1), (1, 1)) in moves
        True
    """
    valid_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    # Find all non-empty source positions
    for src_row in range(4):
        for src_col in range(4):
            if state.grid[src_row, src_col] == 0:
                continue

            source = (src_row, src_col)

            # Check all possible target positions
            for tgt_row in range(4):
                for tgt_col in range(4):
                    target = (tgt_row, tgt_col)

                    if is_valid_move(state, source, target):
                        valid_moves.append((source, target))

    return valid_moves


# =============================================================================
# Move Execution (Task Group 4)
# =============================================================================

def execute_move(
    state: GameState,
    source: Tuple[int, int],
    target: Tuple[int, int]
) -> GameState:
    """Execute a drag-drop move and perform the resulting merge.

    Validates the move, moves the source piece to target, finds the merge
    cluster at target, and executes the appropriate merge (Merge-3 or
    Merge-5).

    Args:
        state: The current game state (will not be mutated).
        source: The (row, col) position of the piece to move.
        target: The (row, col) position to move the piece to.

    Returns:
        A new GameState with the move and merge applied.

    Raises:
        ValueError: If the move is invalid (no valid merge cluster).

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> # Move piece at (2,1) to (1,1) to create cluster of 3
        >>> new_state = execute_move(state, (2, 1), (1, 1))
        >>> new_state.grid[1, 1]  # Upgraded piece at target
        3
        >>> new_state.grid[0, 0]  # Consumed
        0
        >>> new_state.grid[0, 1]  # Consumed
        0
        >>> new_state.grid[2, 1]  # Source cleared
        0
    """
    # Validate the move
    if not is_valid_move(state, source, target):
        raise ValueError(
            f"Invalid move: moving from {source} to {target} does not create "
            "a valid merge cluster of 3+ pieces"
        )

    src_row, src_col = source
    tgt_row, tgt_col = target

    # Create grid with piece moved from source to target
    new_grid = np.copy(state.grid)
    source_value = new_grid[src_row, src_col]
    new_grid[src_row, src_col] = 0
    new_grid[tgt_row, tgt_col] = source_value

    # Create temporary state to find cluster
    temp_state = state.with_grid(new_grid)

    # Find the merge cluster at target position
    cluster = find_merge_cluster(temp_state, target)

    # Execute the merge
    merged_grid = _select_and_execute_merge(new_grid, cluster, target)

    # Return new immutable state
    return state.with_grid(merged_grid)


# =============================================================================
# Game Over Detection
# =============================================================================

def get_empty_cells(state: GameState) -> List[Tuple[int, int]]:
    """Return list of empty cell positions on the grid.

    Iterates through the 4x4 grid and collects all positions where the cell
    value is 0 (empty). Useful for tile spawning logic and game over detection.

    Args:
        state: The current game state.

    Returns:
        A list of (row, col) tuples where the grid cell is empty (value 0).
        Returns an empty list when the board is completely full.

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 0, 0, 0],
        ...     [0, 3, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 1],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> empty = get_empty_cells(state)
        >>> len(empty)
        13
        >>> (0, 1) in empty
        True
        >>> (0, 0) in empty  # Has piece value 2
        False

        >>> # Full board returns empty list
        >>> full_grid = np.ones((4, 4), dtype=np.int8)
        >>> full_state = GameState(grid=full_grid)
        >>> get_empty_cells(full_state)
        []
    """
    empty_cells: List[Tuple[int, int]] = []

    for row in range(4):
        for col in range(4):
            if state.grid[row, col] == 0:
                empty_cells.append((row, col))

    return empty_cells


def is_game_over(state: GameState) -> bool:
    """Detect if the game has ended (terminal state).

    The game is over when both conditions are true:
    1. No empty cells remain on the board
    2. No valid merge moves are possible

    This function is pure and does not mutate the state.

    Args:
        state: The current game state.

    Returns:
        True if the game is over (no empty cells AND no valid moves),
        False otherwise.

    Examples:
        >>> import numpy as np
        >>> # Board with empty cells - game not over
        >>> grid = np.array([
        ...     [2, 2, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> is_game_over(state)
        False

        >>> # Full board with valid merges - game not over
        >>> grid_with_merges = np.array([
        ...     [2, 2, 2, 3],
        ...     [4, 5, 6, 7],
        ...     [1, 2, 3, 4],
        ...     [5, 6, 7, 8],
        ... ], dtype=np.int8)
        >>> state_with_merges = GameState(grid=grid_with_merges)
        >>> is_game_over(state_with_merges)
        False

        >>> # Full board with no valid merges - game over
        >>> grid_no_merges = np.array([
        ...     [1, 2, 1, 2],
        ...     [3, 4, 3, 4],
        ...     [1, 2, 1, 2],
        ...     [3, 4, 3, 4],
        ... ], dtype=np.int8)
        >>> state_no_merges = GameState(grid=grid_no_merges)
        >>> is_game_over(state_no_merges)
        True
    """
    # Check for empty cells first (quick check)
    empty_cells = get_empty_cells(state)
    if empty_cells:
        return False

    # Board is full - check if any valid moves exist
    valid_moves = get_valid_moves(state)
    return len(valid_moves) == 0


# =============================================================================
# Random Tile Spawning
# =============================================================================

def spawn_random_tile(
    state: GameState,
    rng: Optional[np.random.Generator] = None
) -> GameState:
    """Spawn a new tile at a random empty position.

    After each merge, this function spawns a new tile at a random empty
    position. Using the 1x multiplier rules, spawned tiles are B1 or B2
    (levels 1-2) with equal probability.

    Args:
        state: The current game state (will not be mutated).
        rng: Optional NumPy random generator for deterministic testing.
             If None, uses numpy's default random generator.

    Returns:
        A new GameState with a tile spawned at a random empty position.

    Raises:
        ValueError: If the board is full (no empty cells).

    Examples:
        >>> import numpy as np
        >>> grid = np.array([
        ...     [2, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.int8)
        >>> state = GameState(grid=grid)
        >>> # Use seeded RNG for deterministic test
        >>> rng = np.random.default_rng(42)
        >>> new_state = spawn_random_tile(state, rng)
        >>> # A new tile (1 or 2) should appear somewhere
        >>> np.sum(new_state.grid > 0) == 2  # Now 2 pieces on board
        True
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get empty cells
    empty_cells = get_empty_cells(state)

    if not empty_cells:
        raise ValueError("Cannot spawn tile: board is full")

    # Select random empty position
    idx = rng.integers(0, len(empty_cells))
    row, col = empty_cells[idx]

    # Spawn B1 or B2 with equal probability (1x multiplier rules)
    tile_value = rng.integers(1, 3)  # 1 or 2

    # Create new grid with spawned tile
    new_grid = np.copy(state.grid)
    new_grid[row, col] = tile_value

    return state.with_grid(new_grid)
