"""
Tests for Task Group 4: Context Menu System.

These tests verify:
1. Right-click on cell opens context menu
2. Menu displays all 12 piece options plus Clear Cell
3. "Clear All" option at bottom of menu
4. Menu dismisses on outside click or selection
"""

import pytest
import pygame

from src.ui.context_menu import (
    ContextMenu,
    ACTION_SELECT_PIECE,
    ACTION_CLEAR_CELL,
    ACTION_CLEAR_ALL,
)
from src.ui.constants import (
    CONTEXT_MENU_WIDTH,
    CONTEXT_MENU_ITEM_HEIGHT,
    AVAILABLE_PIECES,
)


class TestContextMenuDisplay:
    """Test context menu displays correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.menu = ContextMenu()

    def test_menu_starts_hidden(self):
        """Test that menu is hidden by default."""
        assert not self.menu.visible
        assert self.menu.target_cell is None

    def test_show_makes_menu_visible(self):
        """Test that show() makes the menu visible."""
        self.menu.show(100, 100, (1, 2))

        assert self.menu.visible
        assert self.menu.x == 100
        assert self.menu.y == 100
        assert self.menu.target_cell == (1, 2)

    def test_hide_makes_menu_invisible(self):
        """Test that hide() hides the menu."""
        self.menu.show(100, 100, (1, 2))
        self.menu.hide()

        assert not self.menu.visible
        assert self.menu.target_cell is None


class TestMenuItems:
    """Test menu displays all required options."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.menu = ContextMenu()

    def test_menu_has_12_piece_options(self):
        """Test that menu contains options for all 12 pieces."""
        piece_items = [
            (label, action, value)
            for label, action, value in self.menu._menu_items
            if action == ACTION_SELECT_PIECE
        ]

        assert len(piece_items) == 12

    def test_menu_has_clear_cell_option(self):
        """Test that menu contains 'Clear Cell' option."""
        clear_cell_items = [
            item for item in self.menu._menu_items
            if item[1] == ACTION_CLEAR_CELL
        ]

        assert len(clear_cell_items) == 1
        assert clear_cell_items[0][0] == "Clear Cell"

    def test_menu_has_clear_all_option(self):
        """Test that menu contains 'Clear All' option."""
        clear_all_items = [
            item for item in self.menu._menu_items
            if item[1] == ACTION_CLEAR_ALL
        ]

        assert len(clear_all_items) == 1
        assert clear_all_items[0][0] == "Clear All"

    def test_clear_all_is_at_bottom(self):
        """Test that 'Clear All' is the last menu item."""
        last_item = self.menu._menu_items[-1]
        assert last_item[1] == ACTION_CLEAR_ALL

    def test_total_menu_items_count(self):
        """Test total number of menu items (12 pieces + 2 actions)."""
        assert len(self.menu._menu_items) == 14


class TestMenuClickHandling:
    """Test menu click handling and dismissal."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.menu = ContextMenu()

    def test_clicking_outside_menu_dismisses(self):
        """Test that clicking outside menu dismisses it."""
        self.menu.show(100, 100, (0, 0))

        # Click outside menu bounds
        result = self.menu.handle_click(0, 0)

        assert not self.menu.visible
        assert result is None

    def test_clicking_piece_option_returns_action(self):
        """Test that clicking a piece option returns correct action."""
        self.menu.show(100, 100, (2, 3))

        # Click on first item (B1)
        first_item_rect = self.menu.get_item_rect(0)
        result = self.menu.handle_click(
            first_item_rect.centerx,
            first_item_rect.centery
        )

        assert result is not None
        action_type, piece_value, target_cell = result
        assert action_type == ACTION_SELECT_PIECE
        assert piece_value == 1  # B1
        assert target_cell == (2, 3)
        assert not self.menu.visible

    def test_clicking_clear_cell_returns_action(self):
        """Test that clicking 'Clear Cell' returns correct action."""
        self.menu.show(100, 100, (1, 1))

        # Find Clear Cell item index
        clear_cell_index = None
        for i, (_, action, _) in enumerate(self.menu._menu_items):
            if action == ACTION_CLEAR_CELL:
                clear_cell_index = i
                break

        item_rect = self.menu.get_item_rect(clear_cell_index)
        result = self.menu.handle_click(
            item_rect.centerx,
            item_rect.centery
        )

        assert result is not None
        action_type, piece_value, target_cell = result
        assert action_type == ACTION_CLEAR_CELL
        assert piece_value is None
        assert target_cell == (1, 1)

    def test_clicking_clear_all_returns_action(self):
        """Test that clicking 'Clear All' returns correct action."""
        self.menu.show(100, 100, (0, 0))

        # Clear All is the last item
        last_index = len(self.menu._menu_items) - 1
        item_rect = self.menu.get_item_rect(last_index)
        result = self.menu.handle_click(
            item_rect.centerx,
            item_rect.centery
        )

        assert result is not None
        action_type, piece_value, target_cell = result
        assert action_type == ACTION_CLEAR_ALL
        assert piece_value is None

    def test_selection_dismisses_menu(self):
        """Test that selecting any option dismisses the menu."""
        self.menu.show(100, 100, (0, 0))

        first_item_rect = self.menu.get_item_rect(0)
        self.menu.handle_click(
            first_item_rect.centerx,
            first_item_rect.centery
        )

        assert not self.menu.visible


class TestMenuPositioning:
    """Test menu position calculations."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.menu = ContextMenu()

    def test_item_rect_calculation(self):
        """Test that item rectangles are calculated correctly."""
        self.menu.show(50, 50, (0, 0))

        rect0 = self.menu.get_item_rect(0)
        rect1 = self.menu.get_item_rect(1)

        # Items should be stacked vertically
        assert rect1.y == rect0.y + CONTEXT_MENU_ITEM_HEIGHT

    def test_is_click_inside_when_visible(self):
        """Test is_click_inside when menu is visible."""
        self.menu.show(100, 100, (0, 0))

        # Click inside menu
        assert self.menu.is_click_inside(105, 105)

        # Click outside menu
        assert not self.menu.is_click_inside(0, 0)

    def test_is_click_inside_when_hidden(self):
        """Test is_click_inside when menu is hidden."""
        self.menu.hide()

        # Should return False even for positions that would be inside
        assert not self.menu.is_click_inside(105, 105)
