# Magical Chess ML

> A Python-based reinforcement learning solution for the Magical Chess puzzle mini-game from Call of Antia

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)](#testing)

---

## Overview

**Magical Chess ML** trains reinforcement learning agents to play the Magical Chess (Merge Magic) puzzle game optimally, with a focus on **hourglass efficiency** - maximizing score per hourglass spent rather than raw score alone.

The project provides:
- A complete game engine implementing Magical Chess rules
- An RL environment compatible with Gymnasium/Stable Baselines3
- A Pygame-based advisor UI for getting move recommendations
- Optional live training visualization

## Game Mechanics

Magical Chess is a **drag-and-drop merge puzzle** (not 2048-style sliding):

| Mechanic        | Description                                               |
|-----------------|-----------------------------------------------------------|
| **Board**       | 4x4 grid                                                  |
| **Pieces**      | Chess-themed tiles: Pawn (1) through King (6), Super (7+) |
| **Tiers**       | Bronze (1-3), Silver (4-6), Gold (7+)                     |
| **Merge-3**     | 3 identical pieces → 1 piece of next level                |
| **Merge-5**     | 5 identical pieces → 2 pieces of next level (optimal)     |
| **Restriction** | Same-tier, same-level pieces only                         |

```
┌────────┬────────┬────────┬────────┐
│   B1   │   B2   │        │   B1   │
├────────┼────────┼────────┼────────┤
│   B3   │   B1   │   S4   │        │
├────────┼────────┼────────┼────────┤
│   B1   │   B3   │   B2   │   B1   │
├────────┼────────┼────────┼────────┤
│   S5   │        │   B3   │   B2   │
└────────┴────────┴────────┴────────┘
```

## Features

### Current (v0.1)
- [x] **Game State Model** - Immutable 4x4 grid with tier/level encoding
- [x] **Copy-on-Modify** - Safe state cloning for RL simulations
- [x] **Hourglass Tracking** - Resource management for move costs
- [x] **State Serialization** - JSON save/load support

### Planned
- [ ] Drag-drop move system
- [ ] Merge-3/Merge-5 logic with tier restrictions
- [ ] Gymnasium environment wrapper
- [ ] Hourglass efficiency reward function
- [ ] Pygame advisor UI
- [ ] Live training visualization
- [ ] Camera/OCR integration (future)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/magical-chess-ml.git
cd magical-chess-ml

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Demo

```bash
python main.py demo    # See GameState in action
python main.py test    # Run all tests
python main.py help    # Show available commands
```

### Using GameState

```python
import numpy as np
from magical_chess import GameState, get_tier, get_level

# Create empty game state
state = GameState()

# Create state with custom board
grid = np.array([
    [1, 1, 1, 0],  # Three B1 pawns (can merge!)
    [2, 0, 0, 0],
    [0, 0, 4, 4],  # Two S4 rooks
    [0, 0, 0, 0]
], dtype=np.int8)

state = GameState(
    grid=grid,
    hourglass_count=100,
    score=0,
    last_score_delta=0
)

# Check piece tier
print(get_tier(4))  # "Silver"
print(get_level(4)) # 4

# Clone for simulation (immutable pattern)
sim_state = state.clone()

# Check hourglass affordability
if state.can_afford(16):
    new_state = state.deduct(16)

# Serialize to JSON
json_str = state.to_json()
loaded_state = GameState.from_json(json_str)
```

## Project Structure

```
magical-chess-ml/
├── main.py                    # CLI entry point (demo, test, help)
├── magical_chess.py           # Convenience imports
├── src/
│   └── engine/
│       ├── __init__.py
│       └── game_state.py      # GameState class, tier/level helpers
├── tests/
│   ├── __init__.py
│   └── test_game_state.py     # Comprehensive tests
├── agent-os/
│   ├── product/
│   │   ├── mission.md         # Product vision
│   │   ├── roadmap.md         # Development phases
│   │   └── tech-stack.md      # Technology choices
│   └── specs/                 # Feature specifications
├── RULES.md                   # Complete game rules
├── CLAUDE.md                  # AI assistant instructions
├── requirements.txt           # Python dependencies
└── README.md
```

## Development

### Tech Stack

| Category       | Technology        |
|----------------|-------------------|
| Language       | Python 3.10+      |
| ML Framework   | Stable Baselines3 |
| RL Environment | Gymnasium         |
| Deep Learning  | PyTorch           |
| UI             | Pygame            |
| Testing        | pytest            |

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Style

The project follows these principles:
- **Immutability**: GameState uses frozen dataclasses; all mutations return new instances
- **Fail-fast validation**: Invalid states raise immediately with descriptive errors
- **RL compatibility**: State represented as NumPy arrays for efficient neural network processing

## Game Rules

See [RULES.md](RULES.md) for complete game mechanics including:
- Merge rules (Merge-3, Merge-5, tier restrictions)
- Chess piece mappings
- Hourglass costs and multipliers
- Scoring system
- ML/simulator design notes

## Roadmap

### Phase 1: Core Game Engine *(In Progress)*
- [x] Game State Model
- [ ] Drag-Drop Move System
- [ ] Merge Logic (Merge-3/Merge-5)
- [ ] Adjacency Detection
- [ ] Game Over Detection
- [ ] Random Tile Spawning
- [ ] Gymnasium Environment Wrapper
- [ ] Hourglass Efficiency Reward

### Phase 2: Training Pipeline
- [ ] Training Script (DQN, PPO)
- [ ] Hyperparameter Configuration
- [ ] TensorBoard Logging
- [ ] Model Checkpointing

### Phase 3: Pygame UI
- [ ] Board Display
- [ ] Board Input Interface
- [ ] Move Recommendations
- [ ] Live Training Visualization

### Phase 4: Polish & Documentation
- [ ] Chess Piece Images
- [ ] Model Optimization
- [ ] Performance Benchmarking

### Future: Camera/OCR
- [ ] Screen Capture
- [ ] Tile Detection & Recognition

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Game mechanics based on the Magical Chess mini-game from [Call of Antia](https://www.callofantia.com/)
- Built with [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- UI powered by [Pygame](https://www.pygame.org/)

---

<p align="center">
  <sub>Built with reinforcement learning for optimal puzzle solving</sub>
</p>
