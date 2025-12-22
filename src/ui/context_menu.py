"""
Context Menu Component for Magical Chess UI.

This module provides the right-click context menu for quick piece
placement and cell clearing on the game board.
"""

from typing import Optional, List, Tuple

import pygame

from src.engine.game_state import get_tier, TIER_BRONZE, TIER_SILVER, TIER_GOLD
from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    TEXT,
    BACKGROUND,
    EMPTY_CELL,
)
from src.ui.constants import (
    CONTEXT_MENU_WIDTH,
    CONTEXT_MENU_ITEM_HEIGHT,
    CONTEXT_MENU_PADDING,
    SMALL_FONT_SIZE,
    FONT_NAME,
    AVAILABLE_PIECES,
    PIECE_LABELS,
    ADVISOR_WINDOW_WIDTH,
    ADVISOR_WINDOW_HEIGHT,
)


# Menu action types
ACTION_SELECT_PIECE = "select_piece"
ACTION_CLEAR_CELL = "clear_cell"
ACTION_CLEAR_ALL = "clear_all"


class ContextMenu:
    """Right-click context menu for board cell operations.

    Provides options to:
    - Place any of the 12 piece types (B1-B3, S4-S6, G7-G12)
    - Clear the selected cell
    - Clear all cells on the board

    Attributes:
        x: X position of the menu
        y: Y position of the menu
        visible: Whether the menu is currently displayed
        target_cell: The board cell (row, col) this menu targets
    """

    def __init__(self):
        """Initialize the context menu."""
        self.x = 0
        self.y = 0
        self.visible = False
        self.target_cell: Optional[Tuple[int, int]] = None
        self._font = None
        self._menu_items = self._build_menu_items()

    def _build_menu_items(self) -> List[Tuple[str, str, Optional[int]]]:
        """Build the list of menu items.

        Returns:
            List of (label, action_type, piece_value) tuples
        """
        items = []

        # Add piece options
        for piece_value in AVAILABLE_PIECES:
            label = PIECE_LABELS.get(piece_value, f"?{piece_value}")
            items.append((label, ACTION_SELECT_PIECE, piece_value))

        # Add separator-like gap (will be drawn differently)
        items.append(("Clear Cell", ACTION_CLEAR_CELL, None))
        items.append(("Clear All", ACTION_CLEAR_ALL, None))

        return items

    @property
    def font(self):
        """Get or create the font for rendering menu items.

        Note: Assumes pygame.init() has been called before accessing this property.
        """
        if self._font is None:
            import os
            from pygame import _freetype
            _freetype.init()
            # Find system font
            font_path = None
            for path in ['/System/Library/Fonts/Helvetica.ttc',
                         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                         'C:\\Windows\\Fonts\\arial.ttf']:
                if os.path.exists(path):
                    font_path = path
                    break
            self._font = _freetype.Font(font_path, SMALL_FONT_SIZE)
        return self._font

    @property
    def menu_height(self) -> int:
        """Calculate total height of the menu."""
        return len(self._menu_items) * CONTEXT_MENU_ITEM_HEIGHT + 2 * CONTEXT_MENU_PADDING

    def get_item_rect(self, index: int) -> pygame.Rect:
        """Get the rectangle for a menu item.

        Args:
            index: Index of the menu item

        Returns:
            Pygame Rect for the menu item
        """
        item_x = self.x + CONTEXT_MENU_PADDING
        item_y = self.y + CONTEXT_MENU_PADDING + index * CONTEXT_MENU_ITEM_HEIGHT
        item_width = CONTEXT_MENU_WIDTH - 2 * CONTEXT_MENU_PADDING
        return pygame.Rect(item_x, item_y, item_width, CONTEXT_MENU_ITEM_HEIGHT)

    def get_tier_color(self, piece_value: Optional[int]) -> Tuple[int, int, int]:
        """Get the background color for a menu item.

        Args:
            piece_value: Piece value or None for special actions

        Returns:
            RGB color tuple
        """
        if piece_value is None:
            return BACKGROUND

        tier = get_tier(piece_value)
        if tier == TIER_BRONZE:
            return BRONZE
        elif tier == TIER_SILVER:
            return SILVER
        elif tier == TIER_GOLD:
            return GOLD
        return BACKGROUND

    def show(self, x: int, y: int, target_cell: Tuple[int, int]) -> None:
        """Show the context menu at a position.

        Adjusts position to keep menu within window bounds.

        Args:
            x: X position for the menu
            y: Y position for the menu
            target_cell: The board cell (row, col) this menu is for
        """
        # Calculate menu dimensions
        menu_height = self.menu_height
        menu_width = CONTEXT_MENU_WIDTH

        # Adjust X position if menu would extend beyond right edge
        if x + menu_width > ADVISOR_WINDOW_WIDTH:
            x = ADVISOR_WINDOW_WIDTH - menu_width - 5

        # Adjust Y position if menu would extend beyond bottom edge
        if y + menu_height > ADVISOR_WINDOW_HEIGHT:
            # Position menu above the click point
            y = y - menu_height
            # Make sure it doesn't go above the top edge
            if y < 0:
                y = 5

        self.x = x
        self.y = y
        self.target_cell = target_cell
        self.visible = True

    def hide(self) -> None:
        """Hide the context menu."""
        self.visible = False
        self.target_cell = None

    def render(self, surface: pygame.Surface) -> None:
        """Render the context menu.

        Args:
            surface: Pygame surface to draw on
        """
        if not self.visible:
            return

        # Draw menu background
        menu_rect = pygame.Rect(
            self.x, self.y,
            CONTEXT_MENU_WIDTH,
            self.menu_height
        )
        pygame.draw.rect(surface, BACKGROUND, menu_rect)
        pygame.draw.rect(surface, TEXT, menu_rect, 1)  # Border

        # Draw menu items
        for index, (label, action_type, piece_value) in enumerate(self._menu_items):
            self._render_item(surface, index, label, action_type, piece_value)

    def _render_item(
        self,
        surface: pygame.Surface,
        index: int,
        label: str,
        action_type: str,
        piece_value: Optional[int]
    ) -> None:
        """Render a single menu item.

        Args:
            surface: Pygame surface to draw on
            index: Index of the menu item
            label: Text label to display
            action_type: Type of action for this item
            piece_value: Piece value or None
        """
        rect = self.get_item_rect(index)

        # Draw item background for piece options
        if action_type == ACTION_SELECT_PIECE and piece_value is not None:
            color = self.get_tier_color(piece_value)
            # Draw a small color indicator on the left
            indicator_rect = pygame.Rect(rect.x, rect.y + 2, 20, rect.height - 4)
            pygame.draw.rect(surface, color, indicator_rect)

        # Draw label
        if self.font is not None:
            text_surface, text_rect = self.font.render(label, TEXT)
            text_x = rect.x + 25  # Offset for color indicator
            text_y = rect.centery - text_rect.height // 2
            surface.blit(text_surface, (text_x, text_y))

    def get_item_at_position(self, mouse_x: int, mouse_y: int) -> Optional[int]:
        """Get the menu item index at a mouse position.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            Index of the item if click is on menu, None otherwise
        """
        if not self.visible:
            return None

        for index in range(len(self._menu_items)):
            rect = self.get_item_rect(index)
            if rect.collidepoint(mouse_x, mouse_y):
                return index
        return None

    def handle_click(
        self, mouse_x: int, mouse_y: int
    ) -> Optional[Tuple[str, Optional[int], Optional[Tuple[int, int]]]]:
        """Handle a mouse click on the menu.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            Tuple of (action_type, piece_value, target_cell) if clicked on item,
            None if click was outside menu
        """
        if not self.visible:
            return None

        # Check if click is inside menu bounds
        menu_rect = pygame.Rect(self.x, self.y, CONTEXT_MENU_WIDTH, self.menu_height)
        if not menu_rect.collidepoint(mouse_x, mouse_y):
            self.hide()
            return None

        # Find clicked item
        item_index = self.get_item_at_position(mouse_x, mouse_y)
        if item_index is not None:
            label, action_type, piece_value = self._menu_items[item_index]
            target = self.target_cell
            self.hide()
            return (action_type, piece_value, target)

        return None

    def is_click_inside(self, mouse_x: int, mouse_y: int) -> bool:
        """Check if a click is inside the menu bounds.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            True if click is inside menu, False otherwise
        """
        if not self.visible:
            return False

        menu_rect = pygame.Rect(self.x, self.y, CONTEXT_MENU_WIDTH, self.menu_height)
        return menu_rect.collidepoint(mouse_x, mouse_y)
