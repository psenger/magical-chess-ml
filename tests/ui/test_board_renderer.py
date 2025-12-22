"""
Tests for Task Group 2: Board Renderer Component.

These tests verify:
1. Empty board renders with correct background (#2D2D2D per cell)
2. Piece labels display correctly (B1, B2, B3, S4, S5, S6, G7-G12)
3. Tier colors applied correctly based on piece value
4. Grid position calculations (0-indexed row, col)
5. Cell click detection returns correct coordinates
"""

import pytest
import numpy as np
import pygame

from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    EMPTY_CELL,
)
from src.ui.constants import (
    CELL_SIZE,
    CELL_PADDING,
    GRID_SIZE,
)
from src.ui.board_renderer import BoardRenderer


class TestEmptyBoardRendering:
    """Test empty board renders with correct background."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_display):
        """Set up test fixtures."""
        self.screen = pygame_display
        self.renderer = BoardRenderer(x=50, y=50)

    def test_empty_cell_color_is_dark_gray(self):
        """Test that empty cells (value 0) use dark gray (#2D2D2D)."""
        color = self.renderer.get_tier_color(0)
        assert color == EMPTY_CELL
        assert color == (45, 45, 45)

    def test_empty_grid_renders_all_dark_gray_cells(self):
        """Test rendering an empty 4x4 grid."""
        grid = np.zeros((4, 4), dtype=np.int8)
        # Just verify no exceptions during rendering
        self.renderer.render(self.screen, grid)
        # Manually check a cell color in center
        for row in range(4):
            for col in range(4):
                assert self.renderer.get_tier_color(grid[row, col]) == EMPTY_CELL


class TestPieceLabelDisplay:
    """Test piece labels display correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.renderer = BoardRenderer(x=50, y=50)

    def test_empty_cell_has_no_label(self):
        """Test that empty cells have no label."""
        assert self.renderer.get_piece_label(0) == ""

    def test_bronze_tier_labels(self):
        """Test Bronze tier labels B1, B2, B3."""
        assert self.renderer.get_piece_label(1) == "B1"
        assert self.renderer.get_piece_label(2) == "B2"
        assert self.renderer.get_piece_label(3) == "B3"

    def test_silver_tier_labels(self):
        """Test Silver tier labels S4, S5, S6."""
        assert self.renderer.get_piece_label(4) == "S4"
        assert self.renderer.get_piece_label(5) == "S5"
        assert self.renderer.get_piece_label(6) == "S6"

    def test_gold_tier_labels(self):
        """Test Gold tier labels G7-G12."""
        assert self.renderer.get_piece_label(7) == "G7"
        assert self.renderer.get_piece_label(8) == "G8"
        assert self.renderer.get_piece_label(9) == "G9"
        assert self.renderer.get_piece_label(10) == "G10"
        assert self.renderer.get_piece_label(11) == "G11"
        assert self.renderer.get_piece_label(12) == "G12"

    def test_gold_tier_label_beyond_12(self):
        """Test Gold tier labels for values > 12."""
        assert self.renderer.get_piece_label(15) == "G15"
        assert self.renderer.get_piece_label(20) == "G20"


class TestTierColors:
    """Test tier colors applied correctly based on piece value."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.renderer = BoardRenderer(x=50, y=50)

    def test_bronze_tier_color(self):
        """Test Bronze tier values (1-3) use bronze color #CD7F32."""
        assert self.renderer.get_tier_color(1) == BRONZE
        assert self.renderer.get_tier_color(2) == BRONZE
        assert self.renderer.get_tier_color(3) == BRONZE

    def test_silver_tier_color(self):
        """Test Silver tier values (4-6) use silver color #C0C0C0."""
        assert self.renderer.get_tier_color(4) == SILVER
        assert self.renderer.get_tier_color(5) == SILVER
        assert self.renderer.get_tier_color(6) == SILVER

    def test_gold_tier_color(self):
        """Test Gold tier values (7+) use gold color #FFD700."""
        assert self.renderer.get_tier_color(7) == GOLD
        assert self.renderer.get_tier_color(10) == GOLD
        assert self.renderer.get_tier_color(15) == GOLD


class TestGridPositionCalculations:
    """Test grid position calculations (0-indexed row, col)."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.renderer = BoardRenderer(x=50, y=50)

    def test_cell_rect_for_origin(self):
        """Test cell rect for (0, 0) position."""
        rect = self.renderer.get_cell_rect(0, 0)
        assert rect.x == 50  # board x position
        assert rect.y == 50  # board y position
        assert rect.width == CELL_SIZE
        assert rect.height == CELL_SIZE

    def test_cell_rect_for_row_1_col_2(self):
        """Test cell rect for (1, 2) position."""
        rect = self.renderer.get_cell_rect(1, 2)
        expected_x = 50 + 2 * (CELL_SIZE + CELL_PADDING)
        expected_y = 50 + 1 * (CELL_SIZE + CELL_PADDING)
        assert rect.x == expected_x
        assert rect.y == expected_y

    def test_cell_rect_for_bottom_right(self):
        """Test cell rect for (3, 3) - bottom right corner."""
        rect = self.renderer.get_cell_rect(3, 3)
        expected_x = 50 + 3 * (CELL_SIZE + CELL_PADDING)
        expected_y = 50 + 3 * (CELL_SIZE + CELL_PADDING)
        assert rect.x == expected_x
        assert rect.y == expected_y


class TestCellClickDetection:
    """Test cell click detection returns correct coordinates."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.renderer = BoardRenderer(x=50, y=50)

    def test_click_in_first_cell_returns_0_0(self):
        """Test clicking in the first cell returns (0, 0)."""
        # Click in center of cell (0, 0)
        cell_center_x = 50 + CELL_SIZE // 2
        cell_center_y = 50 + CELL_SIZE // 2
        result = self.renderer.get_cell_at_position(cell_center_x, cell_center_y)
        assert result == (0, 0)

    def test_click_in_cell_1_2_returns_correct_coords(self):
        """Test clicking in cell (1, 2) returns (1, 2)."""
        # Calculate center of cell (1, 2)
        cell_x = 50 + 2 * (CELL_SIZE + CELL_PADDING)
        cell_y = 50 + 1 * (CELL_SIZE + CELL_PADDING)
        cell_center_x = cell_x + CELL_SIZE // 2
        cell_center_y = cell_y + CELL_SIZE // 2
        result = self.renderer.get_cell_at_position(cell_center_x, cell_center_y)
        assert result == (1, 2)

    def test_click_in_last_cell_returns_3_3(self):
        """Test clicking in the last cell returns (3, 3)."""
        # Calculate center of cell (3, 3)
        cell_x = 50 + 3 * (CELL_SIZE + CELL_PADDING)
        cell_y = 50 + 3 * (CELL_SIZE + CELL_PADDING)
        cell_center_x = cell_x + CELL_SIZE // 2
        cell_center_y = cell_y + CELL_SIZE // 2
        result = self.renderer.get_cell_at_position(cell_center_x, cell_center_y)
        assert result == (3, 3)

    def test_click_outside_board_returns_none(self):
        """Test clicking outside the board returns None."""
        result = self.renderer.get_cell_at_position(0, 0)  # Before board
        assert result is None

        # After board (very large coordinates)
        result = self.renderer.get_cell_at_position(1000, 1000)
        assert result is None

    def test_handle_click_returns_cell_and_button(self):
        """Test handle_click returns cell coordinates and button."""
        cell_center_x = 50 + CELL_SIZE // 2
        cell_center_y = 50 + CELL_SIZE // 2

        # Left click
        result = self.renderer.handle_click(cell_center_x, cell_center_y, 1)
        assert result == ((0, 0), 1)

        # Right click
        result = self.renderer.handle_click(cell_center_x, cell_center_y, 3)
        assert result == ((0, 0), 3)
