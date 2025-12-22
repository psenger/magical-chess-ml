"""
Tests for Task Group 1: Pygame Application Foundation.

These tests verify:
1. Window initialization with correct dimensions
2. Event loop handles quit events
3. Color constants match spec
4. Application state transitions (advisor mode vs training viz mode)
"""

import pytest
import pygame

from src.ui import (
    BRONZE,
    SILVER,
    GOLD,
    EMPTY_CELL,
    BACKGROUND,
    TEXT,
    BRONZE_HEX,
    SILVER_HEX,
    GOLD_HEX,
    EMPTY_CELL_HEX,
    BACKGROUND_HEX,
    TEXT_HEX,
    MODE_ADVISOR,
    MODE_TRAINING_VIZ,
)
from src.ui.constants import (
    ADVISOR_WINDOW_WIDTH,
    ADVISOR_WINDOW_HEIGHT,
    TRAINING_VIZ_WINDOW_WIDTH,
    TRAINING_VIZ_WINDOW_HEIGHT,
)
from src.ui.advisor import MagicalChessAdvisor


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class TestWindowInitialization:
    """Test window initialization with correct dimensions."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures. Uses session-scoped pygame init."""
        pass

    def test_advisor_window_dimensions(self):
        """Test that advisor mode window has correct dimensions."""
        app = MagicalChessAdvisor(mode=MODE_ADVISOR)
        app.initialize()

        width, height = pygame.display.get_surface().get_size()
        assert width == ADVISOR_WINDOW_WIDTH, f"Expected width {ADVISOR_WINDOW_WIDTH}, got {width}"
        assert height == ADVISOR_WINDOW_HEIGHT, f"Expected height {ADVISOR_WINDOW_HEIGHT}, got {height}"

    def test_training_viz_window_dimensions(self):
        """Test that training visualization mode window has correct dimensions."""
        app = MagicalChessAdvisor(mode=MODE_TRAINING_VIZ)
        app.initialize()

        width, height = pygame.display.get_surface().get_size()
        assert width == TRAINING_VIZ_WINDOW_WIDTH, f"Expected width {TRAINING_VIZ_WINDOW_WIDTH}, got {width}"
        assert height == TRAINING_VIZ_WINDOW_HEIGHT, f"Expected height {TRAINING_VIZ_WINDOW_HEIGHT}, got {height}"


class TestQuitEventHandling:
    """Test event loop handles quit events."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures. Uses session-scoped pygame init."""
        pass

    def test_quit_event_stops_running(self):
        """Test that QUIT event sets running to False."""
        app = MagicalChessAdvisor(mode=MODE_ADVISOR)
        app.initialize()
        app.running = True

        # Post a QUIT event
        pygame.event.post(pygame.event.Event(pygame.QUIT))

        # Handle events
        app.handle_events()

        assert not app.running, "Application should stop running after QUIT event"


class TestColorConstants:
    """Test color constants match specification."""

    def test_background_color_matches_spec(self):
        """Test background color is #1A1A1A."""
        assert BACKGROUND == hex_to_rgb(BACKGROUND_HEX)
        assert BACKGROUND == (26, 26, 26)

    def test_text_color_matches_spec(self):
        """Test text color is #FFFFFF."""
        assert TEXT == hex_to_rgb(TEXT_HEX)
        assert TEXT == (255, 255, 255)

    def test_bronze_tier_color_matches_spec(self):
        """Test bronze tier color is #CD7F32."""
        assert BRONZE == hex_to_rgb(BRONZE_HEX)
        assert BRONZE == (205, 127, 50)

    def test_silver_tier_color_matches_spec(self):
        """Test silver tier color is #C0C0C0."""
        assert SILVER == hex_to_rgb(SILVER_HEX)
        assert SILVER == (192, 192, 192)

    def test_gold_tier_color_matches_spec(self):
        """Test gold tier color is #FFD700."""
        assert GOLD == hex_to_rgb(GOLD_HEX)
        assert GOLD == (255, 215, 0)

    def test_empty_cell_color_matches_spec(self):
        """Test empty cell color is #2D2D2D."""
        assert EMPTY_CELL == hex_to_rgb(EMPTY_CELL_HEX)
        assert EMPTY_CELL == (45, 45, 45)


class TestApplicationModeTransitions:
    """Test application state transitions between modes."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures. Uses session-scoped pygame init."""
        pass

    def test_initial_mode_is_advisor_by_default(self):
        """Test that default mode is advisor mode."""
        app = MagicalChessAdvisor()
        assert app.mode == MODE_ADVISOR

    def test_initial_mode_can_be_set_to_training_viz(self):
        """Test that initial mode can be set to training visualization."""
        app = MagicalChessAdvisor(mode=MODE_TRAINING_VIZ)
        assert app.mode == MODE_TRAINING_VIZ

    def test_switch_mode_from_advisor_to_training_viz(self):
        """Test switching from advisor mode to training visualization."""
        app = MagicalChessAdvisor(mode=MODE_ADVISOR)
        app.initialize()

        app.switch_mode(MODE_TRAINING_VIZ)

        assert app.mode == MODE_TRAINING_VIZ
        # Verify window resized
        width, height = pygame.display.get_surface().get_size()
        assert width == TRAINING_VIZ_WINDOW_WIDTH
        assert height == TRAINING_VIZ_WINDOW_HEIGHT

    def test_switch_mode_from_training_viz_to_advisor(self):
        """Test switching from training visualization to advisor mode."""
        app = MagicalChessAdvisor(mode=MODE_TRAINING_VIZ)
        app.initialize()

        app.switch_mode(MODE_ADVISOR)

        assert app.mode == MODE_ADVISOR
        width, height = pygame.display.get_surface().get_size()
        assert width == ADVISOR_WINDOW_WIDTH
        assert height == ADVISOR_WINDOW_HEIGHT

    def test_switch_mode_invalid_raises_error(self):
        """Test that switching to invalid mode raises ValueError."""
        app = MagicalChessAdvisor()

        with pytest.raises(ValueError, match="Invalid mode"):
            app.switch_mode("invalid_mode")

    def test_switch_mode_to_same_mode_is_noop(self):
        """Test that switching to the same mode does nothing."""
        app = MagicalChessAdvisor(mode=MODE_ADVISOR)
        app.initialize()

        # Get initial screen reference
        initial_screen = app.screen

        app.switch_mode(MODE_ADVISOR)

        # Mode unchanged
        assert app.mode == MODE_ADVISOR
        # Screen should be the same (no reinitialization)
        assert app.screen is initial_screen
