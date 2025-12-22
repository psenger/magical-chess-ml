"""
Recommendations Panel Component for Magical Chess Advisor.

This module provides the panel that displays top 3 move recommendations
with confidence scores and expected gains.
"""

from typing import List, Dict, Any, Optional

import pygame

from src.ui import TEXT, BACKGROUND, EMPTY_CELL
from src.ui.constants import (
    RECOMMENDATIONS_PANEL_WIDTH,
    RECOMMENDATIONS_PANEL_HEIGHT,
    RECOMMENDATION_ENTRY_HEIGHT,
    SMALL_FONT_SIZE,
    FONT_NAME,
)


class RecommendationPanel:
    """Panel for displaying move recommendations.

    Shows up to 3 recommended moves with:
    - Source and target coordinates
    - Confidence percentage
    - Expected score gain

    Attributes:
        x: X position of the panel
        y: Y position of the panel
        recommendations: List of recommendation dictionaries
    """

    def __init__(self, x: int, y: int):
        """Initialize the recommendations panel.

        Args:
            x: X position of the panel
            y: Y position of the panel
        """
        self.x = x
        self.y = y
        self.recommendations: List[Dict[str, Any]] = []
        self._font = None
        self.loading = False

    @property
    def font(self):
        """Get or create the font for rendering text."""
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

    def set_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Update the displayed recommendations.

        Args:
            recommendations: List of recommendation dicts with keys:
                - source: (row, col)
                - target: (row, col)
                - confidence: percentage value
                - expected_score: score gain value
        """
        self.recommendations = recommendations[:3]  # Max 3 recommendations
        self.loading = False

    def set_loading(self, loading: bool) -> None:
        """Set the loading state.

        Args:
            loading: Whether the panel should show loading state
        """
        self.loading = loading

    def clear(self) -> None:
        """Clear all recommendations."""
        self.recommendations = []
        self.loading = False

    def render(self, surface: pygame.Surface) -> None:
        """Render the recommendations panel.

        Args:
            surface: Pygame surface to draw on
        """
        # Draw panel background
        panel_rect = pygame.Rect(
            self.x, self.y,
            RECOMMENDATIONS_PANEL_WIDTH,
            RECOMMENDATIONS_PANEL_HEIGHT
        )
        pygame.draw.rect(surface, EMPTY_CELL, panel_rect)
        pygame.draw.rect(surface, TEXT, panel_rect, 1)

        # Skip text rendering if font not available
        if self.font is None:
            return

        # Draw title
        title_surface, _ = self.font.render("Move Recommendations", TEXT)
        surface.blit(title_surface, (self.x + 10, self.y + 5))

        if self.loading:
            # Show loading message
            loading_surface, _ = self.font.render("Loading...", TEXT)
            surface.blit(loading_surface, (self.x + 10, self.y + 40))
            return

        if not self.recommendations:
            # Show empty state
            empty_surface, _ = self.font.render("Click 'Get Recommendations' to analyze", TEXT)
            surface.blit(empty_surface, (self.x + 10, self.y + 40))
            return

        # Draw recommendations
        for i, rec in enumerate(self.recommendations):
            y_offset = self.y + 30 + i * RECOMMENDATION_ENTRY_HEIGHT
            self._render_recommendation(surface, rec, y_offset, i + 1)

    def _render_recommendation(
        self,
        surface: pygame.Surface,
        rec: Dict[str, Any],
        y: int,
        rank: int
    ) -> None:
        """Render a single recommendation entry.

        Args:
            surface: Pygame surface to draw on
            rec: Recommendation dictionary
            y: Y position for this entry
            rank: Ranking number (1, 2, or 3)
        """
        source = rec.get("source", (0, 0))
        target = rec.get("target", (0, 0))
        confidence = rec.get("confidence", 0)
        expected_score = rec.get("expected_score", 0)

        # Format: "1. (0,1) → (2,3) | Conf: 45.2% | Score: +30"
        text = (
            f"{rank}. ({source[0]},{source[1]}) -> ({target[0]},{target[1]}) | "
            f"Conf: {confidence:.1f}% | Score: +{expected_score}"
        )

        text_surface, _ = self.font.render(text, TEXT)
        surface.blit(text_surface, (self.x + 10, y))

    def get_recommendation_count(self) -> int:
        """Get the number of current recommendations.

        Returns:
            Number of recommendations (0-3)
        """
        return len(self.recommendations)
