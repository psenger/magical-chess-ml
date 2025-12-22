#!/usr/bin/env python3
"""
Magical Chess ML - Training Script

Train reinforcement learning agents to play Magical Chess optimally,
with a focus on hourglass efficiency.

Usage:
    python train.py                     # Train with default settings (PPO)
    python train.py --algo dqn          # Train with DQN
    python train.py --timesteps 100000  # Train for 100k timesteps
    python train.py --help              # Show all options

Example:
    python train.py --algo ppo --timesteps 50000 --lr 0.0003 --tensorboard
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.engine import MagicalChessEnv


# =============================================================================
# Custom Callback for Hourglass Efficiency Tracking
# =============================================================================


class HourglassEfficiencyCallback(BaseCallback):
    """Callback for tracking hourglass efficiency during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_efficiencies = []

    def _on_step(self) -> bool:
        # Check if episode ended
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # Get efficiency from the info dict if available
                efficiency = info.get("hourglass_efficiency", 0.0)
                self.episode_efficiencies.append(efficiency)

                if self.verbose > 0 and len(self.episode_efficiencies) % 100 == 0:
                    avg_eff = np.mean(self.episode_efficiencies[-100:])
                    print(
                        f"Episode {len(self.episode_efficiencies)}: "
                        f"Avg Efficiency (last 100) = {avg_eff:.2f}"
                    )
        return True


# =============================================================================
# Training Functions
# =============================================================================


def create_env(
    initial_hourglass: int = 100,
    hourglass_cost: int = 1,
    reward_mode: str = "hourglass_efficiency",
    seed: Optional[int] = None,
) -> MagicalChessEnv:
    """Create a Magical Chess environment for training."""
    env = MagicalChessEnv(
        initial_hourglass=initial_hourglass,
        hourglass_cost=hourglass_cost,
        reward_mode=reward_mode,
        seed=seed,
    )
    return Monitor(env)


def train(
    algo: str = "ppo",
    timesteps: int = 100000,
    learning_rate: float = 0.0003,
    batch_size: int = 64,
    n_envs: int = 4,
    initial_hourglass: int = 100,
    hourglass_cost: int = 1,
    reward_mode: str = "hourglass_efficiency",
    save_dir: str = "models",
    log_dir: str = "logs",
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
    tensorboard: bool = True,
    seed: Optional[int] = None,
    verbose: int = 1,
) -> None:
    """Train a Magical Chess agent.

    Args:
        algo: Algorithm to use ("ppo" or "dqn").
        timesteps: Total training timesteps.
        learning_rate: Learning rate.
        batch_size: Batch size for training.
        n_envs: Number of parallel environments (PPO only).
        initial_hourglass: Starting hourglass count.
        hourglass_cost: Hourglass cost per move.
        reward_mode: Reward calculation mode.
        save_dir: Directory to save models.
        log_dir: Directory for TensorBoard logs.
        checkpoint_freq: Checkpoint save frequency (timesteps).
        eval_freq: Evaluation frequency (timesteps).
        eval_episodes: Number of evaluation episodes.
        tensorboard: Whether to log to TensorBoard.
        seed: Random seed for reproducibility.
        verbose: Verbosity level (0=none, 1=info, 2=debug).
    """
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{timestamp}"

    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(log_dir) / run_name if tensorboard else None
    if log_path:
        log_path.mkdir(parents=True, exist_ok=True)

    if verbose > 0:
        print("=" * 60)
        print("Magical Chess ML - Training")
        print("=" * 60)
        print(f"Algorithm:        {algo.upper()}")
        print(f"Timesteps:        {timesteps:,}")
        print(f"Learning Rate:    {learning_rate}")
        print(f"Batch Size:       {batch_size}")
        print(f"Parallel Envs:    {n_envs}")
        print(f"Reward Mode:      {reward_mode}")
        print(f"Save Directory:   {save_path}")
        if tensorboard:
            print(f"TensorBoard:      {log_path}")
        print("=" * 60)

    # Create environment(s)
    def make_env():
        return MagicalChessEnv(
            initial_hourglass=initial_hourglass,
            hourglass_cost=hourglass_cost,
            reward_mode=reward_mode,
            seed=seed,
        )

    if algo.lower() == "ppo":
        env = make_vec_env(make_env, n_envs=n_envs)
    else:
        # DQN doesn't support vectorized envs as well
        env = Monitor(make_env())

    # Create evaluation environment
    eval_env = Monitor(make_env())

    # Create model
    tensorboard_log = str(log_path) if tensorboard else None

    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
        )
    elif algo.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
            buffer_size=50000,
            learning_starts=1000,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'ppo' or 'dqn'.")

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // (n_envs if algo.lower() == "ppo" else 1),
        save_path=str(save_path),
        name_prefix="checkpoint",
        verbose=verbose,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(save_path),
        eval_freq=eval_freq // (n_envs if algo.lower() == "ppo" else 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=verbose,
    )
    callbacks.append(eval_callback)

    # Hourglass efficiency callback
    efficiency_callback = HourglassEfficiencyCallback(verbose=verbose)
    callbacks.append(efficiency_callback)

    # Train
    if verbose > 0:
        print("\nStarting training...")

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=verbose > 0,
    )

    # Save final model
    final_path = save_path / "final_model"
    model.save(str(final_path))

    if verbose > 0:
        print(f"\nTraining complete!")
        print(f"Final model saved to: {final_path}")

        # Final evaluation
        print("\nFinal Evaluation:")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        if efficiency_callback.episode_efficiencies:
            avg_efficiency = np.mean(efficiency_callback.episode_efficiencies[-100:])
            print(f"  Avg Hourglass Efficiency (last 100): {avg_efficiency:.2f}")

    env.close()
    eval_env.close()


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Magical Chess ML agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                           # Train with default settings
  python train.py --algo dqn                # Use DQN instead of PPO
  python train.py --timesteps 200000        # Train for 200k timesteps
  python train.py --lr 0.001 --batch-size 128
  python train.py --tensorboard             # Enable TensorBoard logging
  python train.py --seed 42                 # Set random seed
        """,
    )

    # Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm to use (default: ppo)",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments for PPO (default: 4)",
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
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="hourglass_efficiency",
        choices=["hourglass_efficiency", "score_delta"],
        help="Reward calculation mode (default: hourglass_efficiency)",
    )

    # Saving and logging
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs (default: logs)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Checkpoint save frequency in timesteps (default: 10000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency in timesteps (default: 5000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
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

    train(
        algo=args.algo,
        timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_envs=args.n_envs,
        initial_hourglass=args.initial_hourglass,
        hourglass_cost=args.hourglass_cost,
        reward_mode=args.reward_mode,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        tensorboard=args.tensorboard,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
