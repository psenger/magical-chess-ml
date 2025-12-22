"""
Tests for Task Group 3: Piece Palette Component.

These tests verify:
1. 12 buttons render (B1-B3, S4-S6, G7-G12)
2. Button click selects piece
3. Selected piece indicator displays correctly
4. Palette layout matches 3x4 grid specification
"""

import sys
import pytest
import pygame

# Check if pygame.font is available (broken in pygame 2.6.1 + Python 3.14)
try:
    import pygame.font
    PYGAME_FONT_AVAILABLE = hasattr(pygame.font, 'Font')
except (ImportError, NotImplementedError):
    PYGAME_FONT_AVAILABLE = False

from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    TEXT,
)
from src.ui.constants import (
    PALETTE_BUTTON_WIDTH,
    PALETTE_BUTTON_HEIGHT,
    PALETTE_BUTTON_PADDING,
    PALETTE_COLS,
    PALETTE_ROWS,
    AVAILABLE_PIECES,
)
from src.ui.piece_palette import PiecePalette


class TestPaletteButtonRendering:
    """Test 12 buttons render correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_display):
        """Set up test fixtures."""
        self.screen = pygame_display
        self.palette = PiecePalette(x=50, y=50)

    def test_all_12_pieces_available(self):
        """Test that all 12 pieces are available for selection."""
        assert len(AVAILABLE_PIECES) == 12
        assert AVAILABLE_PIECES == list(range(1, 13))

    def test_each_button_has_rect(self):
        """Test that each piece has a button rectangle."""
        for piece_value in AVAILABLE_PIECES:
            rect = self.palette.get_button_rect(piece_value)
            assert rect.width == PALETTE_BUTTON_WIDTH
            assert rect.height == PALETTE_BUTTON_HEIGHT

    @pytest.mark.skipif(
        not PYGAME_FONT_AVAILABLE,
        reason="pygame.font not available (Python 3.14 compatibility issue)"
    )
    def test_palette_renders_without_error(self):
        """Test that palette rendering doesn't raise exceptions."""
        self.palette.render(self.screen)


class TestButtonClickSelection:
    """Test button click selects piece."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.palette = PiecePalette(x=50, y=50)

    def test_initial_selection_is_none(self):
        """Test that no piece is selected initially."""
        assert self.palette.selected_piece is None

    def test_clicking_button_selects_piece(self):
        """Test clicking a button selects that piece."""
        # Get center of B1 button (piece value 1)
        rect = self.palette.get_button_rect(1)
        center_x = rect.centerx
        center_y = rect.centery

        result = self.palette.handle_click(center_x, center_y)

        assert result is True
        assert self.palette.selected_piece == 1

    def test_clicking_different_button_changes_selection(self):
        """Test clicking a different button changes selection."""
        # Select B1 first
        rect1 = self.palette.get_button_rect(1)
        self.palette.handle_click(rect1.centerx, rect1.centery)
        assert self.palette.selected_piece == 1

        # Now select S4
        rect4 = self.palette.get_button_rect(4)
        self.palette.handle_click(rect4.centerx, rect4.centery)
        assert self.palette.selected_piece == 4

    def test_clicking_outside_palette_returns_false(self):
        """Test clicking outside palette returns False."""
        result = self.palette.handle_click(0, 0)
        assert result is False
        assert self.palette.selected_piece is None

    def test_clear_selection_removes_selection(self):
        """Test clear_selection removes the current selection."""
        rect = self.palette.get_button_rect(5)
        self.palette.handle_click(rect.centerx, rect.centery)
        assert self.palette.selected_piece == 5

        self.palette.clear_selection()
        assert self.palette.selected_piece is None


class TestSelectedPieceIndicator:
    """Test selected piece indicator displays correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_display):
        """Set up test fixtures."""
        self.screen = pygame_display
        self.palette = PiecePalette(x=50, y=50)

    def test_selected_piece_tracked(self):
        """Test that selecting a piece updates the selected_piece attribute."""
        rect = self.palette.get_button_rect(7)
        self.palette.handle_click(rect.centerx, rect.centery)

        assert self.palette.selected_piece == 7

    @pytest.mark.skipif(
        not PYGAME_FONT_AVAILABLE,
        reason="pygame.font not available (Python 3.14 compatibility issue)"
    )
    def test_render_with_selection_no_error(self):
        """Test rendering with a selection doesn't raise exceptions."""
        self.palette.selected_piece = 5
        self.palette.render(self.screen)


class TestPaletteLayout:
    """Test palette layout matches 3x4 grid specification."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.palette = PiecePalette(x=50, y=50)

    def test_layout_is_3_columns_4_rows(self):
        """Test palette is arranged as 3 columns x 4 rows."""
        assert PALETTE_COLS == 3
        assert PALETTE_ROWS == 4

    def test_bronze_pieces_in_first_row(self):
        """Test B1, B2, B3 are in the first row (same y position)."""
        b1_rect = self.palette.get_button_rect(1)
        b2_rect = self.palette.get_button_rect(2)
        b3_rect = self.palette.get_button_rect(3)

        assert b1_rect.y == b2_rect.y == b3_rect.y

    def test_silver_pieces_in_second_row(self):
        """Test S4, S5, S6 are in the second row (same y position)."""
        s4_rect = self.palette.get_button_rect(4)
        s5_rect = self.palette.get_button_rect(5)
        s6_rect = self.palette.get_button_rect(6)

        assert s4_rect.y == s5_rect.y == s6_rect.y
        # And different from first row
        b1_rect = self.palette.get_button_rect(1)
        assert s4_rect.y > b1_rect.y

    def test_columns_have_increasing_x_positions(self):
        """Test columns are positioned left to right."""
        b1_rect = self.palette.get_button_rect(1)
        b2_rect = self.palette.get_button_rect(2)
        b3_rect = self.palette.get_button_rect(3)

        assert b1_rect.x < b2_rect.x < b3_rect.x

    def test_button_spacing_matches_padding(self):
        """Test buttons are spaced according to PALETTE_BUTTON_PADDING."""
        b1_rect = self.palette.get_button_rect(1)
        b2_rect = self.palette.get_button_rect(2)

        expected_spacing = PALETTE_BUTTON_WIDTH + PALETTE_BUTTON_PADDING
        actual_spacing = b2_rect.x - b1_rect.x

        assert actual_spacing == expected_spacing


class TestTierColors:
    """Test tier colors for palette buttons."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.palette = PiecePalette(x=50, y=50)

    def test_bronze_pieces_use_bronze_color(self):
        """Test B1-B3 use bronze color."""
        assert self.palette.get_tier_color(1) == BRONZE
        assert self.palette.get_tier_color(2) == BRONZE
        assert self.palette.get_tier_color(3) == BRONZE

    def test_silver_pieces_use_silver_color(self):
        """Test S4-S6 use silver color."""
        assert self.palette.get_tier_color(4) == SILVER
        assert self.palette.get_tier_color(5) == SILVER
        assert self.palette.get_tier_color(6) == SILVER

    def test_gold_pieces_use_gold_color(self):
        """Test G7-G12 use gold color."""
        for piece in range(7, 13):
            assert self.palette.get_tier_color(piece) == GOLD
