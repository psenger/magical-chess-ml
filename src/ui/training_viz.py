"""
Training Visualization Component for Magical Chess.

This module provides real-time visualization of training episodes,
allowing users to watch the agent play and learn.
"""

from typing import Optional
import threading
import time

import numpy as np
import pygame

from src.ui import BACKGROUND, TEXT
from src.ui.constants import (
    TRAINING_VIZ_WINDOW_WIDTH,
    TRAINING_VIZ_WINDOW_HEIGHT,
    TRAINING_BOARD_X,
    TRAINING_BOARD_Y,
    EPISODE_COUNTER_X,
    EPISODE_COUNTER_Y,
    SCORE_DISPLAY_X,
    SCORE_DISPLAY_Y,
    SPEED_SLIDER_X,
    SPEED_SLIDER_Y,
    SPEED_SLIDER_WIDTH,
    SPEED_SLIDER_HEIGHT,
    SMALL_FONT_SIZE,
    FPS,
)
from src.ui.board_renderer import BoardRenderer


class TrainingVisualization:
    """Real-time visualization of training episodes.

    Displays the game board as the agent plays, with controls for:
    - Speed adjustment (pause to full speed)
    - Episode and score display
    - Auto-pause on game over

    Attributes:
        board_renderer: Component for rendering the game board
        episode: Current episode number
        score: Current episode score
        speed: Visualization speed (0.0 = paused, 1.0 = full speed)
        paused: Whether visualization is currently paused
        game_over: Whether current episode has ended
    """

    def __init__(self):
        """Initialize the training visualization."""
        self.board_renderer = BoardRenderer(x=TRAINING_BOARD_X, y=TRAINING_BOARD_Y)
        self.episode = 0
        self.score = 0
        self.speed = 0.5  # Start at half speed
        self.paused = False
        self.game_over = False
        self.grid = np.zeros((4, 4), dtype=np.int8)
        self._font = None
        self._running = False
        self._update_callback = None

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

    def set_update_callback(self, callback) -> None:
        """Set callback for receiving state updates during training.

        Args:
            callback: Function to call with (grid, episode, score, game_over)
        """
        self._update_callback = callback

    def update_state(
        self,
        grid: np.ndarray,
        episode: int,
        score: int,
        game_over: bool
    ) -> None:
        """Update the visualization with new game state.

        Args:
            grid: Current 4x4 game board
            episode: Episode number
            score: Current score
            game_over: Whether the episode has ended
        """
        self.grid = grid.copy()
        self.episode = episode
        self.score = score
        self.game_over = game_over

        # Auto-pause on game over
        if game_over:
            self.paused = True

    def set_speed(self, speed: float) -> None:
        """Set the visualization speed.

        Args:
            speed: Speed value from 0.0 (paused) to 1.0 (full speed)
        """
        self.speed = max(0.0, min(1.0, speed))
        self.paused = (self.speed == 0.0)

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused

    def render(self, surface: pygame.Surface) -> None:
        """Render the training visualization.

        Args:
            surface: Pygame surface to draw on
        """
        # Render the board
        self.board_renderer.render(surface, self.grid)

        # Render speed slider (always drawn)
        self._render_speed_slider(surface)

        # Skip text rendering if font not available
        if self.font is None:
            return

        # Render episode counter
        episode_surface, episode_rect = self.font.render(f"Episode: {self.episode}", TEXT)
        episode_rect.center = (EPISODE_COUNTER_X, EPISODE_COUNTER_Y)
        surface.blit(episode_surface, episode_rect)

        # Render score display
        score_surface, score_rect = self.font.render(f"Score: {self.score}", TEXT)
        score_rect.center = (SCORE_DISPLAY_X, SCORE_DISPLAY_Y)
        surface.blit(score_surface, score_rect)

        # Render pause/game over indicator
        if self.game_over:
            status_surface, status_rect = self.font.render("GAME OVER (Auto-paused)", TEXT)
        elif self.paused:
            status_surface, status_rect = self.font.render("PAUSED", TEXT)
        else:
            status_surface, status_rect = self.font.render(f"Speed: {self.speed:.0%}", TEXT)

        status_rect.center = (TRAINING_VIZ_WINDOW_WIDTH // 2, SPEED_SLIDER_Y + 50)
        surface.blit(status_surface, status_rect)

    def _render_speed_slider(self, surface: pygame.Surface) -> None:
        """Render the speed control slider.

        Args:
            surface: Pygame surface to draw on
        """
        # Draw slider track
        track_rect = pygame.Rect(
            SPEED_SLIDER_X, SPEED_SLIDER_Y,
            SPEED_SLIDER_WIDTH, SPEED_SLIDER_HEIGHT
        )
        pygame.draw.rect(surface, (60, 60, 60), track_rect)

        # Draw slider position
        knob_x = SPEED_SLIDER_X + int(self.speed * (SPEED_SLIDER_WIDTH - 20))
        knob_rect = pygame.Rect(knob_x, SPEED_SLIDER_Y - 5, 20, SPEED_SLIDER_HEIGHT + 10)
        pygame.draw.rect(surface, TEXT, knob_rect)

    def handle_click(self, mouse_x: int, mouse_y: int) -> bool:
        """Handle mouse click for speed slider.

        Args:
            mouse_x: Mouse X coordinate
            mouse_y: Mouse Y coordinate

        Returns:
            True if click was handled, False otherwise
        """
        # Check if click is on speed slider
        if (SPEED_SLIDER_X <= mouse_x <= SPEED_SLIDER_X + SPEED_SLIDER_WIDTH and
            SPEED_SLIDER_Y - 10 <= mouse_y <= SPEED_SLIDER_Y + SPEED_SLIDER_HEIGHT + 10):

            # Calculate speed from click position
            relative_x = mouse_x - SPEED_SLIDER_X
            self.speed = max(0.0, min(1.0, relative_x / SPEED_SLIDER_WIDTH))
            self.paused = (self.speed == 0.0)
            return True

        return False

    def get_delay_ms(self) -> int:
        """Get the delay between updates based on current speed.

        Returns:
            Delay in milliseconds (0 for paused, lower for faster)
        """
        if self.paused or self.speed == 0:
            return -1  # Indicates paused

        # Map speed to delay: 1.0 = 50ms, 0.1 = 500ms
        return int(50 + (1 - self.speed) * 450)
