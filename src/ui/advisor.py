"""
Main Pygame application class for Magical Chess Advisor.

This module provides the main application window that hosts either:
- Advisor Mode: Input board states and receive AI recommendations
- Training Visualization Mode: Watch training episodes in real-time
"""

from typing import Optional

import numpy as np
import pygame

from src.engine.game_state import GameState
from src.ui import (
    BACKGROUND,
    TEXT,
    EMPTY_CELL,
    MODE_ADVISOR,
    MODE_TRAINING_VIZ,
)
from src.ui.constants import (
    ADVISOR_WINDOW_WIDTH,
    ADVISOR_WINDOW_HEIGHT,
    TRAINING_VIZ_WINDOW_WIDTH,
    TRAINING_VIZ_WINDOW_HEIGHT,
    BOARD_X,
    BOARD_Y,
    BOARD_SIZE,
    RECOMMENDATIONS_PANEL_X,
    RECOMMENDATIONS_PANEL_Y,
    FPS,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    SMALL_FONT_SIZE,
)
from src.ui.board_renderer import BoardRenderer
from src.ui.context_menu import ContextMenu, ACTION_SELECT_PIECE, ACTION_CLEAR_CELL, ACTION_CLEAR_ALL
from src.ui.recommendation_panel import RecommendationPanel

# Right panel position (to the right of the board)
RIGHT_PANEL_X = BOARD_X + BOARD_SIZE + 40
RIGHT_PANEL_Y = BOARD_Y


class MagicalChessAdvisor:
    """Main application class for the Magical Chess Advisor UI.

    This class manages the Pygame window, event loop, and mode switching
    between Advisor and Training Visualization modes.

    Attributes:
        mode: Current application mode (MODE_ADVISOR or MODE_TRAINING_VIZ)
        running: Whether the application event loop is running
        screen: Pygame display surface
        clock: Pygame clock for frame rate control
    """

    def __init__(self, mode: str = MODE_ADVISOR):
        """Initialize the Magical Chess Advisor application.

        Args:
            mode: Initial application mode (MODE_ADVISOR or MODE_TRAINING_VIZ)
        """
        self._mode = mode
        self.running = False
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._initialized = False

        # Game state
        self.grid = np.zeros((4, 4), dtype=np.int8)
        self.score = 0
        self.hourglass_count = 999  # Default to high value (effectively infinite)

        # UI Components
        self.board_renderer = BoardRenderer(x=BOARD_X, y=BOARD_Y)
        self.context_menu = ContextMenu()
        self.recommendation_panel = RecommendationPanel(
            x=RECOMMENDATIONS_PANEL_X,
            y=RECOMMENDATIONS_PANEL_Y
        )

        # Button rectangles (for click detection)
        self.clear_button_rect: Optional[pygame.Rect] = None
        self.recommend_button_rect: Optional[pygame.Rect] = None
        self.save_button_rect: Optional[pygame.Rect] = None
        self.load_button_rect: Optional[pygame.Rect] = None

        # Fonts (initialized after pygame.init())
        self._title_font = None
        self._label_font = None
        self._small_font = None

    @property
    def mode(self) -> str:
        """Get the current application mode."""
        return self._mode

    def _find_system_font(self) -> Optional[str]:
        """Find a system font file path for text rendering.

        Returns:
            Path to a system font file, or None to use default
        """
        import os
        font_paths = [
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            '/System/Library/Fonts/SFNS.ttf',  # macOS
            '/Library/Fonts/Arial.ttf',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            '/usr/share/fonts/TTF/DejaVuSans.ttf',  # Linux
            'C:\\Windows\\Fonts\\arial.ttf',  # Windows
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None  # Use default embedded font

    def initialize(self) -> None:
        """Initialize Pygame and create the display window.

        This method must be called before running the event loop.
        It initializes Pygame, creates the display surface, and sets up fonts.
        """
        if self._initialized:
            return

        pygame.init()

        # Set window dimensions based on mode
        if self._mode == MODE_ADVISOR:
            width = ADVISOR_WINDOW_WIDTH
            height = ADVISOR_WINDOW_HEIGHT
            title = "Magical Chess Advisor"
        else:
            width = TRAINING_VIZ_WINDOW_WIDTH
            height = TRAINING_VIZ_WINDOW_HEIGHT
            title = "Magical Chess Training Visualization"

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()

        # Initialize fonts using low-level _freetype (avoids circular import on Python 3.14)
        from pygame import _freetype
        _freetype.init()
        # Use system font path or None for default embedded font
        font_path = self._find_system_font()
        self._title_font = _freetype.Font(font_path, TITLE_FONT_SIZE)
        self._label_font = _freetype.Font(font_path, LABEL_FONT_SIZE)
        self._small_font = _freetype.Font(font_path, SMALL_FONT_SIZE)

        # Initialize button rectangles
        self._init_button_rects()

        self._initialized = True

    def _init_button_rects(self) -> None:
        """Initialize button rectangle positions for click detection."""
        button_width = 180
        button_height = 40

        # Hourglass input (just a display for now)
        self.hourglass_rect = pygame.Rect(
            RIGHT_PANEL_X, RIGHT_PANEL_Y + 30,
            button_width, 35
        )

        # Score display position
        self.score_y = RIGHT_PANEL_Y + 90

        # Get Recommendations button
        self.recommend_button_rect = pygame.Rect(
            RIGHT_PANEL_X, RIGHT_PANEL_Y + 150,
            button_width, button_height + 5
        )

        # Clear All button
        self.clear_button_rect = pygame.Rect(
            RIGHT_PANEL_X, RIGHT_PANEL_Y + 210,
            button_width, button_height
        )

        # Save/Load buttons (top right)
        self.save_button_rect = pygame.Rect(
            ADVISOR_WINDOW_WIDTH - 180, 20,
            80, 35
        )
        self.load_button_rect = pygame.Rect(
            ADVISOR_WINDOW_WIDTH - 90, 20,
            80, 35
        )

    def switch_mode(self, new_mode: str) -> None:
        """Switch between Advisor and Training Visualization modes.

        Args:
            new_mode: The mode to switch to (MODE_ADVISOR or MODE_TRAINING_VIZ)

        Raises:
            ValueError: If the mode is not valid
        """
        if new_mode not in (MODE_ADVISOR, MODE_TRAINING_VIZ):
            raise ValueError(f"Invalid mode: {new_mode}")

        if new_mode == self._mode:
            return

        self._mode = new_mode

        # Reinitialize display for new mode dimensions
        if self._initialized:
            self._initialized = False
            self.initialize()

    def handle_events(self) -> None:
        """Process Pygame events.

        Handles quit events and other user interactions.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event)

    def _handle_mouse_click(self, event: pygame.event.Event) -> None:
        """Handle mouse click events.

        Args:
            event: Pygame mouse event
        """
        mouse_x, mouse_y = event.pos
        button = event.button

        # Check context menu first
        if self.context_menu.visible:
            result = self.context_menu.handle_click(mouse_x, mouse_y)
            if result is not None:
                action_type, piece_value, target_cell = result
                self._handle_context_menu_action(action_type, piece_value, target_cell)
            return

        # Right-click on board - show context menu
        if button == 3:  # Right click
            cell = self.board_renderer.get_cell_at_position(mouse_x, mouse_y)
            if cell is not None:
                self.context_menu.show(mouse_x, mouse_y, cell)
            return

        # Left-click handling
        if button == 1:
            # Check button clicks first
            if self.clear_button_rect and self.clear_button_rect.collidepoint(mouse_x, mouse_y):
                self._handle_clear_all()
                return

            if self.recommend_button_rect and self.recommend_button_rect.collidepoint(mouse_x, mouse_y):
                self._handle_get_recommendations()
                return

            if self.save_button_rect and self.save_button_rect.collidepoint(mouse_x, mouse_y):
                self._handle_save()
                return

            if self.load_button_rect and self.load_button_rect.collidepoint(mouse_x, mouse_y):
                self._handle_load()
                return

    def _handle_context_menu_action(
        self,
        action_type: str,
        piece_value: Optional[int],
        target_cell: Optional[tuple]
    ) -> None:
        """Handle context menu action.

        Args:
            action_type: Type of action selected
            piece_value: Piece value if selecting a piece
            target_cell: Target cell coordinates
        """
        if action_type == ACTION_SELECT_PIECE and target_cell and piece_value:
            row, col = target_cell
            self.grid[row, col] = piece_value
        elif action_type == ACTION_CLEAR_CELL and target_cell:
            row, col = target_cell
            self.grid[row, col] = 0
        elif action_type == ACTION_CLEAR_ALL:
            self._handle_clear_all()

    def _handle_clear_all(self) -> None:
        """Clear the entire board."""
        self.grid = np.zeros((4, 4), dtype=np.int8)
        self.score = 0
        self.recommendation_panel.clear()

    def _handle_get_recommendations(self) -> None:
        """Get AI recommendations for the current board state.

        TODO: Implement model inference when model loader is connected.
        """
        # For now, show loading state then clear
        self.recommendation_panel.set_loading(True)
        # In the future, this will call the model loader
        # For now, just clear the loading state
        self.recommendation_panel.set_loading(False)

    def _handle_save(self) -> None:
        """Save the current board state to a JSON file.

        TODO: Implement file saving functionality.
        """
        pass

    def _handle_load(self) -> None:
        """Load a board state from a JSON file.

        TODO: Implement file loading functionality.
        """
        pass

    def update(self) -> None:
        """Update application state.

        Called once per frame to update game logic and UI state.
        """
        pass

    def render(self) -> None:
        """Render the current frame.

        Draws all UI components to the screen based on the current mode.
        """
        if self.screen is None:
            return

        # Clear screen with background color
        self.screen.fill(BACKGROUND)

        # Draw title bar with Save/Load buttons
        self._render_title_bar()

        # Draw board
        self.board_renderer.render(self.screen, self.grid)

        # Draw right panel (hourglass, score, buttons)
        self._render_right_panel()

        # Draw recommendations panel
        self.recommendation_panel.render(self.screen)

        # Draw context menu (on top)
        self.context_menu.render(self.screen)

        pygame.display.flip()

    def _render_title_bar(self) -> None:
        """Render the title bar with title and Save/Load buttons."""
        # Draw title
        if self._title_font:
            title_surface, title_rect = self._title_font.render("MAGICAL CHESS ADVISOR", TEXT)
            self.screen.blit(title_surface, (BOARD_X, 20))

        # Draw Save button
        if self.save_button_rect:
            pygame.draw.rect(self.screen, EMPTY_CELL, self.save_button_rect)
            pygame.draw.rect(self.screen, TEXT, self.save_button_rect, 1)
            if self._small_font:
                text_surface, text_rect = self._small_font.render("Save", TEXT)
                text_rect.center = self.save_button_rect.center
                self.screen.blit(text_surface, text_rect)

        # Draw Load button
        if self.load_button_rect:
            pygame.draw.rect(self.screen, EMPTY_CELL, self.load_button_rect)
            pygame.draw.rect(self.screen, TEXT, self.load_button_rect, 1)
            if self._small_font:
                text_surface, text_rect = self._small_font.render("Load", TEXT)
                text_rect.center = self.load_button_rect.center
                self.screen.blit(text_surface, text_rect)

    def _render_right_panel(self) -> None:
        """Render the right panel with hourglass, score, and buttons."""
        # Draw hourglass label and value
        if self._small_font:
            label_surface, _ = self._small_font.render("Hourglasses:", TEXT)
            self.screen.blit(label_surface, (RIGHT_PANEL_X, RIGHT_PANEL_Y))

        # Draw hourglass input box
        if hasattr(self, 'hourglass_rect'):
            pygame.draw.rect(self.screen, EMPTY_CELL, self.hourglass_rect)
            pygame.draw.rect(self.screen, TEXT, self.hourglass_rect, 1)
            if self._label_font:
                value_text = str(self.hourglass_count) if self.hourglass_count < 999 else "INF"
                text_surface, text_rect = self._label_font.render(value_text, TEXT)
                text_rect.center = self.hourglass_rect.center
                self.screen.blit(text_surface, text_rect)

        # Draw score display
        if self._label_font and hasattr(self, 'score_y'):
            text_surface, _ = self._label_font.render(f"Score: {self.score}", TEXT)
            self.screen.blit(text_surface, (RIGHT_PANEL_X, self.score_y))

        # Draw Get Recommendations button
        if self.recommend_button_rect:
            pygame.draw.rect(self.screen, EMPTY_CELL, self.recommend_button_rect)
            pygame.draw.rect(self.screen, TEXT, self.recommend_button_rect, 2)
            if self._small_font:
                text_surface, text_rect = self._small_font.render("GET RECOMMENDATIONS", TEXT)
                text_rect.center = self.recommend_button_rect.center
                self.screen.blit(text_surface, text_rect)

        # Draw Clear All button
        if self.clear_button_rect:
            pygame.draw.rect(self.screen, EMPTY_CELL, self.clear_button_rect)
            pygame.draw.rect(self.screen, TEXT, self.clear_button_rect, 1)
            if self._small_font:
                text_surface, text_rect = self._small_font.render("CLEAR ALL", TEXT)
                text_rect.center = self.clear_button_rect.center
                self.screen.blit(text_surface, text_rect)

        # Draw instruction text
        if self._small_font:
            instr1_surface, _ = self._small_font.render("Right-click board", TEXT)
            self.screen.blit(instr1_surface, (RIGHT_PANEL_X, RIGHT_PANEL_Y + 270))
            instr2_surface, _ = self._small_font.render("to place pieces", TEXT)
            self.screen.blit(instr2_surface, (RIGHT_PANEL_X, RIGHT_PANEL_Y + 290))

    def run(self) -> None:
        """Run the main application event loop.

        This method blocks until the application is closed.
        """
        self.initialize()
        self.running = True

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()

    def quit(self) -> None:
        """Quit the application and clean up resources."""
        self.running = False
        if self._initialized:
            pygame.quit()
            self._initialized = False
