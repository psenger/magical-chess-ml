"""
Tests for Task Group 9: Training Visualization Window.

These tests verify:
1. Separate window/mode launches correctly
2. Board updates reflect training state
3. Episode counter increments
4. Speed slider controls visualization rate
5. Auto-pause triggers on game over
"""

import pytest
import numpy as np
import pygame

from src.ui.training_viz import TrainingVisualization


class TestTrainingVizInitialization:
    """Test training visualization initialization."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.viz = TrainingVisualization()

    def test_initial_state(self):
        """Test initial state values."""
        assert self.viz.episode == 0
        assert self.viz.score == 0
        assert not self.viz.paused
        assert not self.viz.game_over
        assert self.viz.speed == 0.5  # Half speed default

    def test_initial_grid_is_empty(self):
        """Test that grid starts empty."""
        assert self.viz.grid.shape == (4, 4)
        assert np.all(self.viz.grid == 0)


class TestStateUpdates:
    """Test state update functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.viz = TrainingVisualization()

    def test_update_state_changes_grid(self):
        """Test that update_state updates the grid."""
        new_grid = np.array([
            [1, 2, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 5],
        ], dtype=np.int8)

        self.viz.update_state(new_grid, 10, 100, False)

        assert np.array_equal(self.viz.grid, new_grid)
        assert self.viz.episode == 10
        assert self.viz.score == 100
        assert not self.viz.game_over

    def test_update_state_copies_grid(self):
        """Test that update_state makes a copy of the grid."""
        new_grid = np.ones((4, 4), dtype=np.int8)
        self.viz.update_state(new_grid, 1, 0, False)

        # Modify original grid
        new_grid[0, 0] = 99

        # Visualization grid should be unchanged
        assert self.viz.grid[0, 0] == 1


class TestAutoPause:
    """Test auto-pause on game over."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.viz = TrainingVisualization()

    def test_game_over_triggers_auto_pause(self):
        """Test that game over automatically pauses visualization."""
        grid = np.zeros((4, 4), dtype=np.int8)

        # Update with game_over=True
        self.viz.update_state(grid, 5, 200, game_over=True)

        assert self.viz.game_over
        assert self.viz.paused


class TestSpeedControl:
    """Test speed slider control."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.viz = TrainingVisualization()

    def test_set_speed_updates_value(self):
        """Test that set_speed updates speed."""
        self.viz.set_speed(0.75)
        assert self.viz.speed == 0.75

    def test_set_speed_clamps_to_valid_range(self):
        """Test that speed is clamped to 0.0-1.0."""
        self.viz.set_speed(1.5)
        assert self.viz.speed == 1.0

        self.viz.set_speed(-0.5)
        assert self.viz.speed == 0.0

    def test_set_speed_zero_pauses(self):
        """Test that setting speed to 0 pauses visualization."""
        self.viz.set_speed(0.0)
        assert self.viz.paused

    def test_toggle_pause(self):
        """Test pause toggle."""
        initial_paused = self.viz.paused

        self.viz.toggle_pause()
        assert self.viz.paused != initial_paused

        self.viz.toggle_pause()
        assert self.viz.paused == initial_paused

    def test_get_delay_ms_when_paused(self):
        """Test delay returns -1 when paused."""
        self.viz.paused = True
        assert self.viz.get_delay_ms() == -1

    def test_get_delay_ms_varies_with_speed(self):
        """Test delay varies inversely with speed."""
        self.viz.set_speed(1.0)
        fast_delay = self.viz.get_delay_ms()

        self.viz.set_speed(0.1)
        slow_delay = self.viz.get_delay_ms()

        assert slow_delay > fast_delay
