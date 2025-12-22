"""
Piece Palette Component for Magical Chess UI.

This module provides the 3x4 grid of piece buttons for selecting
pieces to place on the game board.
"""

from typing import Optional, Tuple

import pygame

from src.engine.game_state import get_tier, TIER_BRONZE, TIER_SILVER, TIER_GOLD
from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    TEXT,
    BACKGROUND,
)
from src.ui.constants import (
    PALETTE_BUTTON_WIDTH,
    PALETTE_BUTTON_HEIGHT,
    PALETTE_BUTTON_PADDING,
    PALETTE_COLS,
    PALETTE_ROWS,
    BUTTON_FONT_SIZE,
    FONT_NAME,
    AVAILABLE_PIECES,
    PIECE_LABELS,
)


class PiecePalette:
    """Component for selecting pieces to place on the board.

    Displays a 3x4 grid of buttons (B1-B3, S4-S6, G7-G12) that allows
    users to select which piece they want to place on the board.

    Attributes:
        x: X position of the palette
        y: Y position of the palette
        selected_piece: Currently selected piece value (1-12, or None)
    """

    def __init__(self, x: int, y: int):
        """Initialize the piece palette.

        Args:
            x: X position of the palette (top-left corner)
            y: Y position of the palette (top-left corner)
        """
        self.x = x
        self.y = y
        self.selected_piece: Optional[int] = None
        self._font: Optional[pygame.font.Font] = None

    @property
    def font(self) -> Optional[pygame.font.Font]:
        """Get or create the font for rendering labels.

        Note: Assumes pygame.init() has been called before accessing this property.
        Returns None if font module is not available.
        """
        if self._font is None:
            try:
                self._font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)
            except (Exception, NotImplementedError):
                return None
        return self._font

    def get_button_rect(self, piece_value: int) -> pygame.Rect:
        """Get the rectangle for a specific piece button.

        Args:
            piece_value: Piece value (1-12)

        Returns:
            Pygame Rect for the button
        """
        # Pieces are arranged in 3 columns x 4 rows
        # Column 0: B1, S4, G7, G10
        # Column 1: B2, S5, G8, G11
        # Column 2: B3, S6, G9, G12
        index = piece_value - 1  # Convert to 0-indexed
        col = index % PALETTE_COLS
        row = index // PALETTE_COLS

        button_x = self.x + col * (PALETTE_BUTTON_WIDTH + PALETTE_BUTTON_PADDING)
        button_y = self.y + row * (PALETTE_BUTTON_HEIGHT + PALETTE_BUTTON_PADDING)

        return pygame.Rect(button_x, button_y, PALETTE_BUTTON_WIDTH, PALETTE_BUTTON_HEIGHT)

    def get_tier_color(self, piece_value: int) -> Tuple[int, int, int]:
        """Get the background color for a piece button.

        Args:
            piece_value: Piece value (1-12)

        Returns:
            RGB color tuple
        """
        tier = get_tier(piece_value)
        if tier == TIER_BRONZE:
            return BRONZE
        elif tier == TIER_SILVER:
            return SILVER
        elif tier == TIER_GOLD:
            return GOLD
        return BACKGROUND

    def render(self, surface: pygame.Surface) -> None:
        """Render the piece palette to a surface.

        Args:
            surface: Pygame surface to draw on
        """
        for piece_value in AVAILABLE_PIECES:
            self._render_button(surface, piece_value)

    def _render_button(self, surface: pygame.Surface, piece_value: int) -> None:
        """Render a single piece button.

        Args:
            surface: Pygame surface to draw on
            piece_value: Piece value (1-12)
        """
        rect = self.get_button_rect(piece_value)
        color = self.get_tier_color(piece_value)

        # Draw button background
        pygame.draw.rect(surface, color, rect)

        # Draw selection indicator (white border)
        if self.selected_piece == piece_value:
            pygame.draw.rect(surface, TEXT, rect, 3)  # 3px border

        # Draw piece label (skip if font not available)
        if self.font is not None:
            label = PIECE_LABELS.get(piece_value, f"?{piece_value}")
            text_surface = self.font.render(label, True, TEXT)
            text_rect = text_surface.get_rect(center=rect.center)
            surface.blit(text_surface, text_rect)

    def get_piece_at_position(self, mouse_x: int, mouse_y: int) -> Optional[int]:
        """Get the piece value at a mouse position.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            Piece value (1-12) if click is on a button, None otherwise
        """
        for piece_value in AVAILABLE_PIECES:
            rect = self.get_button_rect(piece_value)
            if rect.collidepoint(mouse_x, mouse_y):
                return piece_value
        return None

    def handle_click(self, mouse_x: int, mouse_y: int) -> bool:
        """Handle a mouse click on the palette.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            True if a piece was selected, False otherwise
        """
        piece = self.get_piece_at_position(mouse_x, mouse_y)
        if piece is not None:
            self.selected_piece = piece
            return True
        return False

    def clear_selection(self) -> None:
        """Clear the currently selected piece."""
        self.selected_piece = None
