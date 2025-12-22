#!/usr/bin/env python3
"""
Magical Chess ML - Entry Point

A reinforcement learning solution for the Magical Chess puzzle game.

Usage:
    python main.py              # Show help
    python main.py demo         # Run a demo showing GameState usage
    python main.py test         # Run all tests
"""

import sys


def show_help():
    """Display available commands."""
    print("""
Magical Chess ML
================

Commands:
    python main.py demo     Run a demo showing GameState usage
    python main.py test     Run all tests
    python main.py help     Show this help message

Quick Start (Python):
    from magical_chess import GameState, get_tier, get_level

    state = GameState()
    print(state.grid)
""")


def run_demo():
    """Run a demo showing GameState usage."""
    import numpy as np
    from src.engine import GameState, get_tier, get_level, TIER_BRONZE, TIER_SILVER

    print("=" * 50)
    print("Magical Chess ML - Demo")
    print("=" * 50)

    # Create empty state
    print("\n1. Creating empty GameState:")
    state = GameState()
    print(f"   Grid shape: {state.grid.shape}")
    print(f"   Hourglass: {state.hourglass_count}")
    print(f"   Score: {state.score}")

    # Create custom board
    print("\n2. Creating custom board with pieces:")
    grid = np.array([
        [1, 1, 1, 0],  # Three B1 pawns
        [2, 0, 0, 0],  # One B2 knight
        [0, 0, 4, 4],  # Two S4 rooks
        [0, 0, 0, 7]   # One G7 super
    ], dtype=np.int8)

    state = GameState(
        grid=grid,
        hourglass_count=100,
        score=500,
        last_score_delta=50
    )

    print("   Board:")
    print_board(state.grid)

    # Show tier detection
    print("\n3. Tier detection:")
    for val in [1, 3, 4, 6, 7]:
        tier = get_tier(val)
        level = get_level(val)
        print(f"   Value {val}: Tier={tier}, Level={level}")

    # Clone for simulation
    print("\n4. Cloning for simulation (immutable pattern):")
    cloned = state.clone()
    print(f"   Original == Clone: {state == cloned}")
    print(f"   Same object: {state is cloned}")

    # Hourglass operations
    print("\n5. Hourglass operations:")
    print(f"   Can afford 16? {state.can_afford(16)}")
    print(f"   Can afford 200? {state.can_afford(200)}")
    new_state = state.deduct(16)
    print(f"   After deduct(16): {new_state.hourglass_count} (original: {state.hourglass_count})")

    # Serialization
    print("\n6. Serialization:")
    json_str = state.to_json()
    print(f"   JSON length: {len(json_str)} chars")
    loaded = GameState.from_json(json_str)
    print(f"   Round-trip successful: {state == loaded}")

    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)


def print_board(grid):
    """Print the board in a readable format."""
    tier_prefix = {0: "--", 1: "B1", 2: "B2", 3: "B3", 4: "S4", 5: "S5", 6: "S6"}

    print("   ┌────┬────┬────┬────┐")
    for i, row in enumerate(grid):
        cells = []
        for val in row:
            if val == 0:
                cells.append("--")
            elif val <= 6:
                cells.append(tier_prefix.get(val, f"?{val}"))
            else:
                cells.append(f"G{val}")
        print(f"   │ {cells[0]} │ {cells[1]} │ {cells[2]} │ {cells[3]} │")
        if i < 3:
            print("   ├────┼────┼────┼────┤")
    print("   └────┴────┴────┴────┘")


def run_tests():
    """Run all tests using pytest."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd="."
    )
    sys.exit(result.returncode)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "demo":
        run_demo()
    elif command == "test":
        run_tests()
    elif command in ("help", "-h", "--help"):
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
