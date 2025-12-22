"""
Board Renderer Component for Magical Chess UI.

This module provides the 4x4 grid display for rendering game boards
in both Advisor and Training Visualization modes.
"""

from typing import Optional, Tuple

import numpy as np
import pygame

from src.engine.game_state import get_tier, TIER_BRONZE, TIER_SILVER, TIER_GOLD
from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    EMPTY_CELL,
    TEXT,
)
from src.ui.constants import (
    GRID_SIZE,
    CELL_SIZE,
    CELL_PADDING,
    BOARD_PADDING,
    CELL_FONT_SIZE,
    FONT_NAME,
    PIECE_LABELS,
)


class BoardRenderer:
    """Component for rendering the 4x4 Magical Chess game board.

    This class handles:
    - Drawing the grid with tier-colored cell backgrounds
    - Rendering piece labels (B1-B3, S4-S6, G7+)
    - Click detection for cell selection

    Attributes:
        x: X position of the board in the window
        y: Y position of the board in the window
        font: Pygame font for rendering piece labels
    """

    def __init__(self, x: int, y: int):
        """Initialize the board renderer.

        Args:
            x: X position of the board (top-left corner)
            y: Y position of the board (top-left corner)
        """
        self.x = x
        self.y = y
        self._font = None

    @property
    def font(self):
        """Get or create the font for rendering labels.

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
            self._font = _freetype.Font(font_path, CELL_FONT_SIZE)
        return self._font

    def get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the rectangle for a specific cell.

        Args:
            row: Row index (0-3)
            col: Column index (0-3)

        Returns:
            Pygame Rect for the cell at (row, col)
        """
        cell_x = self.x + col * (CELL_SIZE + CELL_PADDING)
        cell_y = self.y + row * (CELL_SIZE + CELL_PADDING)
        return pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)

    def get_tier_color(self, value: int) -> Tuple[int, int, int]:
        """Get the background color for a cell based on piece value.

        Args:
            value: Piece value (0 = empty, 1-3 = Bronze, 4-6 = Silver, 7+ = Gold)

        Returns:
            RGB color tuple for the cell background
        """
        if value == 0:
            return EMPTY_CELL

        tier = get_tier(value)
        if tier == TIER_BRONZE:
            return BRONZE
        elif tier == TIER_SILVER:
            return SILVER
        elif tier == TIER_GOLD:
            return GOLD
        else:
            return EMPTY_CELL

    def get_piece_label(self, value: int) -> str:
        """Get the display label for a piece value.

        Args:
            value: Piece value (0-12+)

        Returns:
            Label string (e.g., "B1", "S4", "G7", or "" for empty)
        """
        if value == 0:
            return ""

        # Use predefined labels for 1-12
        if value in PIECE_LABELS:
            return PIECE_LABELS[value]

        # For values > 12, use Gold tier with numeric suffix
        return f"G{value}"

    def render(self, surface: pygame.Surface, grid: np.ndarray) -> None:
        """Render the game board to a surface.

        Args:
            surface: Pygame surface to draw on
            grid: 4x4 NumPy array of piece values
        """
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = int(grid[row, col])
                self._render_cell(surface, row, col, value)

    def _render_cell(
        self, surface: pygame.Surface, row: int, col: int, value: int
    ) -> None:
        """Render a single cell.

        Args:
            surface: Pygame surface to draw on
            row: Row index (0-3)
            col: Column index (0-3)
            value: Piece value at this cell
        """
        rect = self.get_cell_rect(row, col)

        # Draw cell background with tier color
        color = self.get_tier_color(value)
        pygame.draw.rect(surface, color, rect)

        # Draw piece label if not empty
        label = self.get_piece_label(value)
        if label and self.font is not None:
            text_surface, text_rect = self.font.render(label, TEXT)
            text_rect.center = rect.center
            surface.blit(text_surface, text_rect)

    def get_cell_at_position(
        self, mouse_x: int, mouse_y: int
    ) -> Optional[Tuple[int, int]]:
        """Get the grid cell at a mouse position.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            Tuple of (row, col) if click is on a cell, None otherwise
        """
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = self.get_cell_rect(row, col)
                if rect.collidepoint(mouse_x, mouse_y):
                    return (row, col)
        return None

    def handle_click(
        self, mouse_x: int, mouse_y: int, button: int
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """Handle a mouse click on the board.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate
            button: Mouse button (1 = left, 3 = right)

        Returns:
            Tuple of ((row, col), button) if click is on a cell, None otherwise
        """
        cell = self.get_cell_at_position(mouse_x, mouse_y)
        if cell is not None:
            return (cell, button)
        return None
