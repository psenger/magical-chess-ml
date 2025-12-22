"""
Pytest configuration for UI tests.

This module provides fixtures and configuration for pygame-based tests.
Due to circular import issues with pygame.font in Python 3.14, we need
to be careful about how pygame is initialized and cleaned up.
"""

import os
import pytest

# Set SDL to use dummy video driver for headless testing
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

# Pre-import pygame.font to avoid circular import issues
# This must be done before pygame.init() is called
try:
    import pygame.font
except ImportError:
    pass


@pytest.fixture(scope="session", autouse=True)
def pygame_init():
    """Initialize pygame once for the entire test session.

    This avoids issues with pygame.font module circular imports
    when pygame.quit() is called and then pygame.init() is called again.
    """
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture(scope="function")
def pygame_display():
    """Create a pygame display for tests that need one.

    Returns a 600x600 surface that can be used for rendering tests.
    """
    screen = pygame.display.set_mode((600, 600))
    yield screen
    # Don't quit pygame here - let the session fixture handle it
