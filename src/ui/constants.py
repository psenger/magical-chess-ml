"""
Shared configuration constants for the Pygame UI.

This module contains all magic numbers and configuration values used
throughout the UI components.
"""

# =============================================================================
# Window Dimensions
# =============================================================================

# Advisor window dimensions
ADVISOR_WINDOW_WIDTH = 900
ADVISOR_WINDOW_HEIGHT = 700

# Training visualization window dimensions
TRAINING_VIZ_WINDOW_WIDTH = 600
TRAINING_VIZ_WINDOW_HEIGHT = 600

# =============================================================================
# Grid and Cell Sizing
# =============================================================================

# Board grid dimensions
GRID_SIZE = 4
CELL_SIZE = 100
CELL_PADDING = 5
BOARD_PADDING = 20

# Board total size
BOARD_SIZE = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * CELL_PADDING + 2 * BOARD_PADDING

# Board position in advisor window (left side)
BOARD_X = BOARD_PADDING
BOARD_Y = 80  # Leave room for title bar

# =============================================================================
# Piece Palette Configuration
# =============================================================================

# Palette position (right side of board)
PALETTE_X = BOARD_SIZE + 40
PALETTE_Y = BOARD_Y

# Palette button dimensions
PALETTE_BUTTON_WIDTH = 60
PALETTE_BUTTON_HEIGHT = 50
PALETTE_BUTTON_PADDING = 5

# Palette grid layout (3 columns x 4 rows)
PALETTE_COLS = 3
PALETTE_ROWS = 4

# =============================================================================
# Context Menu Configuration
# =============================================================================

CONTEXT_MENU_WIDTH = 120
CONTEXT_MENU_ITEM_HEIGHT = 30
CONTEXT_MENU_PADDING = 5

# =============================================================================
# UI Control Positions
# =============================================================================

# Hourglass input position
HOURGLASS_INPUT_X = PALETTE_X
HOURGLASS_INPUT_Y = PALETTE_Y + (PALETTE_ROWS * (PALETTE_BUTTON_HEIGHT + PALETTE_BUTTON_PADDING)) + 30
HOURGLASS_INPUT_WIDTH = 180
HOURGLASS_INPUT_HEIGHT = 35

# Model dropdown position
MODEL_DROPDOWN_X = PALETTE_X
MODEL_DROPDOWN_Y = HOURGLASS_INPUT_Y + HOURGLASS_INPUT_HEIGHT + 20
MODEL_DROPDOWN_WIDTH = 180
MODEL_DROPDOWN_HEIGHT = 35

# Get Recommendations button position
RECOMMEND_BUTTON_X = PALETTE_X
RECOMMEND_BUTTON_Y = MODEL_DROPDOWN_Y + MODEL_DROPDOWN_HEIGHT + 20
RECOMMEND_BUTTON_WIDTH = 180
RECOMMEND_BUTTON_HEIGHT = 45

# Save/Load buttons position (top bar)
SAVE_BUTTON_X = ADVISOR_WINDOW_WIDTH - 200
SAVE_BUTTON_Y = 20
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 35

LOAD_BUTTON_X = SAVE_BUTTON_X + BUTTON_WIDTH + 10
LOAD_BUTTON_Y = SAVE_BUTTON_Y

# =============================================================================
# Recommendations Panel Configuration
# =============================================================================

RECOMMENDATIONS_PANEL_X = BOARD_PADDING
RECOMMENDATIONS_PANEL_Y = BOARD_Y + BOARD_SIZE + 20
RECOMMENDATIONS_PANEL_WIDTH = ADVISOR_WINDOW_WIDTH - 2 * BOARD_PADDING
RECOMMENDATIONS_PANEL_HEIGHT = 150
RECOMMENDATION_ENTRY_HEIGHT = 40

# =============================================================================
# Move History Panel Configuration
# =============================================================================

HISTORY_PANEL_X = PALETTE_X
HISTORY_PANEL_Y = RECOMMEND_BUTTON_Y + RECOMMEND_BUTTON_HEIGHT + 20
HISTORY_PANEL_WIDTH = 180
HISTORY_PANEL_HEIGHT = 150

# =============================================================================
# Training Visualization Configuration
# =============================================================================

# Board position (centered)
TRAINING_BOARD_X = (TRAINING_VIZ_WINDOW_WIDTH - BOARD_SIZE) // 2 + BOARD_PADDING
TRAINING_BOARD_Y = 100

# Episode counter position
EPISODE_COUNTER_X = TRAINING_VIZ_WINDOW_WIDTH // 2
EPISODE_COUNTER_Y = 30

# Score display position
SCORE_DISPLAY_X = TRAINING_VIZ_WINDOW_WIDTH // 2
SCORE_DISPLAY_Y = 60

# Speed slider position
SPEED_SLIDER_X = 50
SPEED_SLIDER_Y = TRAINING_VIZ_WINDOW_HEIGHT - 80
SPEED_SLIDER_WIDTH = TRAINING_VIZ_WINDOW_WIDTH - 100
SPEED_SLIDER_HEIGHT = 20

# =============================================================================
# Font Configuration
# =============================================================================

FONT_NAME = None  # Use default system font
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 18
CELL_FONT_SIZE = 24
BUTTON_FONT_SIZE = 16
SMALL_FONT_SIZE = 14

# =============================================================================
# Piece Tier Mappings
# =============================================================================

# Value to label mapping
# Bronze: 1-6 (B1-B6), Silver: 7-12 (S1-S6), Gold: 13-18 (G1-G6)
PIECE_LABELS = {
    0: "",  # Empty cell - no label
    # Bronze tier (values 1-6)
    1: "B1",
    2: "B2",
    3: "B3",
    4: "B4",
    5: "B5",
    6: "B6",
    # Silver tier (values 7-12)
    7: "S1",
    8: "S2",
    9: "S3",
    10: "S4",
    11: "S5",
    12: "S6",
    # Gold tier (values 13-18)
    13: "G1",
    14: "G2",
    15: "G3",
    16: "G4",
    17: "G5",
    18: "G6",
}

# All available piece values for palette/context menu (1-18)
AVAILABLE_PIECES = list(range(1, 19))

# Label to value mapping (reverse lookup)
LABEL_TO_VALUE = {v: k for k, v in PIECE_LABELS.items() if k > 0}

# =============================================================================
# Frame Rate
# =============================================================================

FPS = 60
