<h1 align="center">Magical Chess ML</h1>

<p align="center">
  <strong>A reinforcement learning agent that masters the Magical Chess puzzle game</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-103%20passing-brightgreen.svg" alt="Tests: 103 passing"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [What is Magical Chess?](#what-is-magical-chess)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why This Project?

Magical Chess presents a unique challenge for reinforcement learning:

- **Resource Management**: Unlike 2048-style games, Magical Chess introduces hourglasses—a limited resource that constrains how many moves you can make
- **Optimization Target**: The goal isn't just high scores, but **hourglass efficiency**—maximizing points per hourglass spent
- **Merge Complexity**: With Merge-3 and Merge-5 mechanics, plus tier restrictions, the decision space is rich and non-trivial

This project aims to:

1. **Build a complete game simulator** faithful to the original mechanics
2. **Train RL agents** (DQN, PPO) that discover optimal play strategies
3. **Provide a move advisor** for human players to improve their gameplay
4. **Explore** how modern RL techniques handle constrained resource optimization

---

## What is Magical Chess?

Magical Chess is a **merge puzzle mini-game** from [Call of Antia](https://www.callofantia.com/). Unlike 2048-style sliding puzzles, it uses **drag-and-drop mechanics** with chess-themed pieces.

### Game Board

```
┌────┬────┬────┬────┐
│ B1 │ B2 │ -- │ B1 │
├────┼────┼────┼────┤
│ B3 │ B1 │ S4 │ -- │
├────┼────┼────┼────┤
│ B1 │ B3 │ B2 │ B1 │
├────┼────┼────┼────┤
│ S5 │ -- │ B3 │ B2 │
└────┴────┴────┴────┘
B = Bronze tier, S = Silver tier, G = Gold tier
Numbers indicate piece level (1-6+)
```

### Core Mechanics

| Mechanic | Description |
|----------|-------------|
| **Board** | 4×4 grid |
| **Pieces** | Chess-themed: Pawn (1) → Knight (2) → Bishop (3) → Rook (4) → Queen (5) → King (6) → Super (7+) |
| **Tiers** | Bronze (1-3), Silver (4-6), Gold (7+) |
| **Merge-3** | 3 identical pieces → 1 piece of next level |
| **Merge-5** | 5 identical pieces → 2 pieces of next level *(optimal!)* |
| **Restriction** | Only same-tier, same-level pieces can merge |
| **Hourglasses** | Resource spent on moves; multipliers cost more but spawn higher-level tiles |

> For complete rules, see [RULES.md](RULES.md)

---

## Features

### Implemented

- **Immutable Game State Model** — Thread-safe state representation using frozen dataclasses
- **Drag-Drop Move System** — Complete move validation and execution with cluster detection
- **Merge Logic** — Merge-3 and Merge-5 with tier restrictions fully implemented
- **Gymnasium Environment** — `MagicalChessEnv` compatible with Stable Baselines3
- **Hourglass Efficiency Reward** — Reward function optimizing points per hourglass
- **Training Pipeline** — PPO training with TensorBoard logging and model checkpointing
- **Evaluation Script** — Evaluate trained models and compare against random baseline
- **Random Tile Spawning** — Deterministic spawning with seeded RNG support
- **Game Over Detection** — Detects when no valid moves remain
- **State Serialization** — JSON import/export for debugging and replay
- **Comprehensive Test Suite** — 103 tests covering all components

### Planned

- Pygame-based move advisor UI
- Live training visualization
- Camera/OCR integration for live game assistance

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required for modern type hints |
| pip | Latest | Package manager |
| Git | Any | For cloning |

### Setup

```bash
# Clone the repository
git clone https://github.com/psenger/magical-chess-ml.git
cd magical-chess-ml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python main.py test
```

You should see all 103 tests passing.

---

## Quick Start

### Command Line Interface

```bash
python main.py demo    # Interactive demonstration
python main.py test    # Run test suite
python main.py help    # Show all commands
```

### Python API

```python
import numpy as np
from magical_chess import GameState, get_tier, get_level, MagicalChessEnv

# Create a game state with a custom board
grid = np.array([
    [1, 1, 1, 0],  # Three B1 pawns — ready to merge!
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

# Inspect piece properties
print(get_tier(4))   # → "Silver"
print(get_level(4))  # → 4

# Clone for simulation (immutable pattern)
sim_state = state.clone()

# Resource management
if state.can_afford(16):
    new_state = state.deduct(16)

# Serialize for storage/debugging
json_str = state.to_json()
restored = GameState.from_json(json_str)

# Use the Gymnasium environment
env = MagicalChessEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Training

### Train a New Model

```bash
python train.py
```

Training uses PPO by default with:
- TensorBoard logging
- Automatic model checkpointing to `models/`
- Hourglass efficiency reward function

### Evaluate a Trained Model

```bash
python evaluate.py
```

Compares trained model performance against a random baseline.

### Monitor Training

```bash
tensorboard --logdir=models/
```

---

## Project Structure

```
magical-chess-ml/
│
├── main.py                    # CLI entry point (demo, test, help)
├── train.py                   # Training script with PPO
├── evaluate.py                # Model evaluation script
├── magical_chess.py           # Public API exports
│
├── src/
│   └── engine/
│       ├── __init__.py
│       ├── game_state.py      # Core GameState class
│       ├── move_system.py     # Move validation, merging, spawning
│       └── magical_chess_env.py  # Gymnasium environment
│
├── tests/
│   ├── __init__.py
│   ├── test_game_state.py     # GameState unit tests
│   ├── test_move_system.py    # Move system tests
│   └── test_magical_chess_env.py  # Environment tests
│
├── models/                    # Saved trained models
│   └── ppo_*/                 # Timestamped model directories
│
├── agent-os/                  # Project management
│   ├── product/
│   │   ├── mission.md
│   │   ├── roadmap.md
│   │   └── tech-stack.md
│   └── specs/                 # Feature specifications
│
├── RULES.md                   # Complete game rules
├── CLAUDE.md                  # AI assistant context
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md
```

---

## Development

### Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Core implementation |
| **ML Framework** | Stable Baselines3 | RL algorithms (PPO) |
| **RL Environment** | Gymnasium | Standard RL interface |
| **Deep Learning** | PyTorch | Neural network backend |
| **Visualization** | TensorBoard | Training metrics |
| **Testing** | pytest | Test framework |

### Code Principles

1. **Immutability** — `GameState` is a frozen dataclass; all operations return new instances
2. **Fail-Fast Validation** — Invalid inputs raise descriptive errors immediately
3. **RL Compatibility** — States are NumPy arrays for efficient tensor conversion
4. **Pure Functions** — Move system functions have no side effects
5. **Test Coverage** — All public APIs have comprehensive tests

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_game_state.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Suite Overview

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_game_state.py` | GameState creation, cloning, serialization | Core state logic |
| `test_move_system.py` | Move validation, merging, spawning, game over | Game mechanics |
| `test_magical_chess_env.py` | Gymnasium interface, step, reset | RL environment |

**Total: 103 tests passing**

---

## Roadmap

### Phase 1: Core Game Engine — Complete

- [x] Immutable Game State Model
- [x] Drag-Drop Move System
- [x] Merge Logic (Merge-3 / Merge-5)
- [x] Adjacency Detection (BFS clustering)
- [x] Game Over Detection
- [x] Random Tile Spawning
- [x] Gymnasium Environment Wrapper
- [x] Hourglass Efficiency Reward Function

### Phase 2: Training Pipeline — Complete

- [x] Training Script (PPO)
- [x] TensorBoard Integration
- [x] Model Checkpointing
- [x] Evaluation Script
- [ ] Hyperparameter tuning
- [ ] DQN/A2C algorithm options

### Phase 3: Pygame Advisor UI — Planned

- [ ] Board Display
- [ ] Manual Board Input
- [ ] AI Move Recommendations
- [ ] Training Visualization
- [ ] Statistics Dashboard

### Phase 4: Polish & Distribution — Planned

- [ ] Pre-trained Model Weights (published)
- [ ] Performance Benchmarks
- [ ] PyPI Package
- [ ] Documentation Site

### Future: Vision Integration

- [ ] Screen Capture
- [ ] Tile Detection (OpenCV)
- [ ] OCR for Score Reading
- [ ] Real-time Advisor Mode

---

## Troubleshooting

### Common Issues

<details>
<summary><strong>ModuleNotFoundError: No module named 'magical_chess'</strong></summary>

Ensure you're running from the project root directory:

```bash
cd magical-chess-ml
python main.py demo
```

Or add the project to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/magical-chess-ml"
```

</details>

<details>
<summary><strong>Tests fail with import errors</strong></summary>

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

</details>

<details>
<summary><strong>CUDA/GPU errors during training</strong></summary>

The project works with CPU by default. If you have GPU issues:

```bash
# Force CPU-only
export CUDA_VISIBLE_DEVICES=""
python train.py
```

</details>

<details>
<summary><strong>TensorBoard not showing data</strong></summary>

Ensure you're pointing to the correct directory:

```bash
tensorboard --logdir=models/
```

Wait a few seconds for data to appear after training starts.

</details>

---

## FAQ

<details>
<summary><strong>Is this affiliated with Call of Antia?</strong></summary>

No. This is an independent fan project for educational and research purposes. Magical Chess is a mini-game within Call of Antia, developed by [JEEO Interactive](https://www.callofantia.com/).

</details>

<details>
<summary><strong>Can I use the trained models for actual gameplay?</strong></summary>

The move advisor (planned) is designed to help you make better decisions while playing. Currently you can evaluate the trained model using `evaluate.py`.

</details>

<details>
<summary><strong>What RL algorithms are supported?</strong></summary>

Currently implemented:
- **PPO** (Proximal Policy Optimization) — Default, works well for this problem

Planned:
- **DQN** (Deep Q-Network)
- **A2C** (Advantage Actor-Critic)

</details>

<details>
<summary><strong>Why not use 2048-style sliding mechanics?</strong></summary>

Magical Chess uses drag-and-drop merging, not sliding. This is a key distinction that affects the action space and strategy. See [RULES.md](RULES.md) for details.

</details>

<details>
<summary><strong>How does the hourglass efficiency reward work?</strong></summary>

Instead of rewarding raw score, the agent is rewarded based on score gained per hourglass spent. This teaches the agent to be resource-efficient, not just score-maximizing.

</details>

---

## Contributing

Contributions are welcome! Here's how to get started:

### Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/magical-chess-ml.git
   ```
3. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make** your changes
5. **Test** your changes:
   ```bash
   python -m pytest tests/ -v
   ```
6. **Commit** with a descriptive message:
   ```bash
   git commit -m "Add amazing feature"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open** a Pull Request

### Guidelines

- Add tests for new functionality
- Ensure all 103 tests pass before submitting
- Keep PRs focused on a single change

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2025 Philip A. Senger
```

---

## Acknowledgments

- **Game Inspiration** — [Call of Antia](https://www.callofantia.com/) by JEEO Interactive
- **RL Framework** — [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- **Environment Standard** — [Gymnasium](https://gymnasium.farama.org/)

---

<p align="center">
  <sub>Built with reinforcement learning for optimal puzzle solving</sub>
  <br>
  <sub>Made by <a href="https://github.com/psenger">Philip A. Senger</a></sub>
</p>
