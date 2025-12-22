#!/usr/bin/env python3
"""
Magical Chess ML - Evaluation and Inference Script

Load trained models and evaluate their performance or get move recommendations.

Usage:
    python evaluate.py models/ppo_20240101_120000/best_model  # Evaluate model
    python evaluate.py models/best_model --episodes 50        # Run 50 episodes
    python evaluate.py models/best_model --infer              # Interactive inference

Example:
    python evaluate.py models/ppo_20240101_120000/final_model --episodes 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import DQN, PPO

from src.engine import MagicalChessEnv, GameState


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_path: str) -> Tuple[object, str]:
    """Load a trained model from disk.

    Automatically detects whether the model is PPO or DQN.

    Args:
        model_path: Path to the saved model (without .zip extension).

    Returns:
        Tuple of (model, algorithm_name).

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model format is not recognized.
    """
    model_path = Path(model_path)

    # Handle both with and without .zip extension
    if not model_path.suffix:
        zip_path = model_path.with_suffix(".zip")
    else:
        zip_path = model_path

    if not zip_path.exists():
        raise FileNotFoundError(f"Model not found: {zip_path}")

    # Try loading as PPO first, then DQN
    try:
        model = PPO.load(str(model_path))
        return model, "PPO"
    except Exception:
        pass

    try:
        model = DQN.load(str(model_path))
        return model, "DQN"
    except Exception:
        pass

    raise ValueError(f"Could not load model: {model_path}. Not a valid PPO or DQN model.")


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    initial_hourglass: int = 100,
    hourglass_cost: int = 1,
    reward_mode: str = "hourglass_efficiency",
    deterministic: bool = True,
    verbose: int = 1,
    seed: Optional[int] = None,
) -> Dict:
    """Evaluate a trained model over multiple episodes.

    Args:
        model_path: Path to the saved model.
        n_episodes: Number of episodes to evaluate.
        initial_hourglass: Starting hourglass count.
        hourglass_cost: Hourglass cost per move.
        reward_mode: Reward mode used during evaluation.
        deterministic: Whether to use deterministic actions.
        verbose: Verbosity level.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with evaluation metrics.
    """
    model, algo_name = load_model(model_path)

    if verbose > 0:
        print("=" * 60)
        print("Magical Chess ML - Model Evaluation")
        print("=" * 60)
        print(f"Model:            {model_path}")
        print(f"Algorithm:        {algo_name}")
        print(f"Episodes:         {n_episodes}")
        print(f"Deterministic:    {deterministic}")
        print("=" * 60)

    env = MagicalChessEnv(
        initial_hourglass=initial_hourglass,
        hourglass_cost=hourglass_cost,
        reward_mode=reward_mode,
        seed=seed,
    )

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    episode_efficiencies = []
    valid_action_rates = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        valid_actions = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Track if action was valid (reward > 0 or not penalty)
            if reward > 0 or reward != env.invalid_action_penalty:
                valid_actions += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info.get("score", 0))
        episode_efficiencies.append(info.get("hourglass_efficiency", 0))
        valid_action_rates.append(valid_actions / steps if steps > 0 else 0)

        if verbose > 1:
            print(
                f"Episode {ep + 1}/{n_episodes}: "
                f"Reward={total_reward:.2f}, "
                f"Score={info.get('score', 0)}, "
                f"Steps={steps}, "
                f"Efficiency={info.get('hourglass_efficiency', 0):.2f}"
            )

    env.close()

    # Compute statistics
    results = {
        "model_path": str(model_path),
        "algorithm": algo_name,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_score": float(np.mean(episode_scores)),
        "std_score": float(np.std(episode_scores)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_hourglass_efficiency": float(np.mean(episode_efficiencies)),
        "std_hourglass_efficiency": float(np.std(episode_efficiencies)),
        "mean_valid_action_rate": float(np.mean(valid_action_rates)),
        "max_score": int(np.max(episode_scores)),
        "min_score": int(np.min(episode_scores)),
    }

    if verbose > 0:
        print("\nEvaluation Results:")
        print("-" * 40)
        print(f"Mean Reward:          {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Mean Score:           {results['mean_score']:.1f} +/- {results['std_score']:.1f}")
        print(f"Max Score:            {results['max_score']}")
        print(f"Mean Episode Length:  {results['mean_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")
        print(f"Mean Efficiency:      {results['mean_hourglass_efficiency']:.2f} +/- {results['std_hourglass_efficiency']:.2f}")
        print(f"Valid Action Rate:    {results['mean_valid_action_rate']:.1%}")
        print("-" * 40)

    return results


# =============================================================================
# Inference (Move Recommendation)
# =============================================================================


def get_action_confidence(model, obs: np.ndarray, env: MagicalChessEnv) -> List[Tuple[int, float]]:
    """Get action probabilities/values for all valid actions.

    Args:
        model: The trained model.
        obs: Current observation.
        env: The environment instance.

    Returns:
        List of (action, confidence) tuples sorted by confidence descending.
    """
    action_mask = env.get_action_mask()
    valid_actions = np.where(action_mask)[0]

    if len(valid_actions) == 0:
        return []

    # Get model predictions
    action_scores = []

    # For PPO, we can get action probabilities
    if hasattr(model, "policy"):
        # Get the distribution over actions
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with model.policy.evaluator if hasattr(model.policy, 'evaluator') else model.policy:
            try:
                distribution = model.policy.get_distribution(obs_tensor)
                log_probs = distribution.log_prob(
                    model.policy.action_space.get_tensor_from_action(valid_actions)
                )
                probs = np.exp(log_probs.detach().numpy())
            except:
                # Fallback: just use the predicted action
                predicted_action, _ = model.predict(obs, deterministic=False)
                for action in valid_actions:
                    score = 1.0 if action == predicted_action else 0.1
                    action_scores.append((action, score))
                return sorted(action_scores, key=lambda x: x[1], reverse=True)

            for i, action in enumerate(valid_actions):
                action_scores.append((action, float(probs[i]) if i < len(probs) else 0.1))
    else:
        # For DQN, we could get Q-values but simplify for now
        predicted_action, _ = model.predict(obs, deterministic=True)
        for action in valid_actions:
            score = 1.0 if action == predicted_action else 0.5
            action_scores.append((action, score))

    return sorted(action_scores, key=lambda x: x[1], reverse=True)


def action_to_move_str(action: int) -> str:
    """Convert action index to human-readable move string."""
    source_idx = action // 16
    target_idx = action % 16

    source_row, source_col = source_idx // 4, source_idx % 4
    target_row, target_col = target_idx // 4, target_idx % 4

    return f"({source_row},{source_col}) -> ({target_row},{target_col})"


def recommend_moves(
    model_path: str,
    grid: Optional[np.ndarray] = None,
    hourglass: int = 100,
    top_k: int = 3,
    verbose: int = 1,
) -> List[Dict]:
    """Get move recommendations for a given board state.

    Args:
        model_path: Path to the saved model.
        grid: Optional 4x4 grid. If None, starts fresh game.
        hourglass: Current hourglass count.
        top_k: Number of top recommendations to return.
        verbose: Verbosity level.

    Returns:
        List of move recommendations with confidence scores.
    """
    model, algo_name = load_model(model_path)

    env = MagicalChessEnv(initial_hourglass=hourglass)
    obs, info = env.reset()

    # If custom grid provided, set it
    if grid is not None:
        env.state = GameState(
            grid=grid.astype(np.int8),
            hourglass_count=hourglass,
            score=0,
            last_score_delta=0,
        )
        obs = env._get_obs()
        info = env._get_info()

    if verbose > 0:
        print("=" * 60)
        print("Magical Chess ML - Move Recommendations")
        print("=" * 60)
        print(f"Model: {model_path} ({algo_name})")
        print(f"\nCurrent Board:")
        print(env.render() or "Board not rendered")
        print()

    valid_moves = info.get("valid_moves", [])
    if not valid_moves:
        if verbose > 0:
            print("No valid moves available!")
        return []

    # Get action confidences
    action_scores = get_action_confidence(model, obs, env)

    recommendations = []
    for action, confidence in action_scores[:top_k]:
        source_idx = action // 16
        target_idx = action % 16

        source = (source_idx // 4, source_idx % 4)
        target = (target_idx // 4, target_idx % 4)

        rec = {
            "action": int(action),
            "source": source,
            "target": target,
            "move_str": action_to_move_str(action),
            "confidence": float(confidence),
        }
        recommendations.append(rec)

        if verbose > 0:
            print(f"  #{len(recommendations)}: {rec['move_str']} (confidence: {confidence:.2%})")

    env.close()
    return recommendations


def interactive_inference(model_path: str, verbose: int = 1) -> None:
    """Run interactive inference session.

    Args:
        model_path: Path to the saved model.
        verbose: Verbosity level.
    """
    model, algo_name = load_model(model_path)

    print("=" * 60)
    print("Magical Chess ML - Interactive Inference")
    print("=" * 60)
    print(f"Model: {model_path} ({algo_name})")
    print("\nCommands:")
    print("  [Enter]  - Get recommendations for current state")
    print("  r        - Reset to new game")
    print("  p [action] - Play the specified action")
    print("  q        - Quit")
    print("=" * 60)

    env = MagicalChessEnv(render_mode="ansi")
    obs, info = env.reset()

    while True:
        print(f"\n{env.render()}")
        print(f"Valid actions: {len(info.get('valid_actions', []))}")

        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "r":
            obs, info = env.reset()
            print("Game reset!")
        elif cmd.startswith("p "):
            try:
                action = int(cmd[2:])
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Played action {action}, reward: {reward:.2f}")
                if terminated:
                    print("Game Over!")
            except ValueError:
                print("Invalid action number")
        elif cmd == "" or cmd == "rec":
            valid_actions = info.get("valid_actions", [])
            if not valid_actions:
                print("No valid moves!")
                continue

            action_scores = get_action_confidence(model, obs, env)
            print("\nTop Recommendations:")
            for i, (action, conf) in enumerate(action_scores[:3], 1):
                print(f"  {i}. {action_to_move_str(action)} (confidence: {conf:.2%})")
        else:
            print("Unknown command. Use Enter, r, p [action], or q.")

    env.close()
    print("Goodbye!")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Magical Chess ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py models/best_model                    # Evaluate with defaults
  python evaluate.py models/best_model --episodes 200     # Run 200 episodes
  python evaluate.py models/best_model --infer            # Interactive mode
  python evaluate.py models/best_model --output results.json
        """,
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model file",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )

    # Environment parameters
    parser.add_argument(
        "--initial-hourglass",
        type=int,
        default=100,
        help="Initial hourglass count (default: 100)",
    )
    parser.add_argument(
        "--hourglass-cost",
        type=int,
        default=1,
        help="Hourglass cost per move (default: 1)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    # Modes
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run interactive inference mode",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)",
    )

    args = parser.parse_args()

    if args.infer:
        interactive_inference(args.model_path, args.verbose)
    else:
        deterministic = not args.stochastic

        results = evaluate_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            initial_hourglass=args.initial_hourglass,
            hourglass_cost=args.hourglass_cost,
            deterministic=deterministic,
            verbose=args.verbose,
            seed=args.seed,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
