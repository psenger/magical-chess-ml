"""
Pygame UI Module for Magical Chess Advisor and Training Visualization.

This module provides a graphical interface for:
- Inputting board states and receiving AI move recommendations (Advisor Mode)
- Visualizing training episodes in real-time (Training Visualization Mode)

Color Scheme Constants:
    BRONZE: #CD7F32 - Background color for Bronze tier pieces (B1-B3)
    SILVER: #C0C0C0 - Background color for Silver tier pieces (S4-S6)
    GOLD: #FFD700 - Background color for Gold tier pieces (G7+)
    EMPTY_CELL: #2D2D2D - Background color for empty cells
    BACKGROUND: #1A1A1A - Main window background color
    TEXT: #FFFFFF - Text color for labels and UI elements
"""

# Color scheme constants (RGB tuples for Pygame)
BRONZE = (205, 127, 50)  # #CD7F32
SILVER = (192, 192, 192)  # #C0C0C0
GOLD = (255, 215, 0)  # #FFD700
EMPTY_CELL = (45, 45, 45)  # #2D2D2D
BACKGROUND = (26, 26, 26)  # #1A1A1A
TEXT = (255, 255, 255)  # #FFFFFF

# Color hex strings for reference/testing
BRONZE_HEX = "#CD7F32"
SILVER_HEX = "#C0C0C0"
GOLD_HEX = "#FFD700"
EMPTY_CELL_HEX = "#2D2D2D"
BACKGROUND_HEX = "#1A1A1A"
TEXT_HEX = "#FFFFFF"

# Application modes
MODE_ADVISOR = "advisor"
MODE_TRAINING_VIZ = "training_viz"

__all__ = [
    # Color constants (RGB tuples)
    "BRONZE",
    "SILVER",
    "GOLD",
    "EMPTY_CELL",
    "BACKGROUND",
    "TEXT",
    # Color hex strings
    "BRONZE_HEX",
    "SILVER_HEX",
    "GOLD_HEX",
    "EMPTY_CELL_HEX",
    "BACKGROUND_HEX",
    "TEXT_HEX",
    # Application modes
    "MODE_ADVISOR",
    "MODE_TRAINING_VIZ",
]
