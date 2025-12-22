"""
Model Loading and Inference for Magical Chess Advisor.

This module handles loading trained models and performing inference
to generate move recommendations with confidence scores.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    from stable_baselines3 import PPO, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class ModelLoader:
    """Handles loading and inference with trained RL models.

    This class manages:
    - Scanning for available model files
    - Loading PPO or DQN models
    - Running inference to get action recommendations
    - Computing confidence scores using softmax

    Attributes:
        model: The loaded stable-baselines3 model
        model_path: Path to the loaded model file
        model_type: Type of model ('ppo' or 'dqn')
    """

    def __init__(self, models_dir: str = "models"):
        """Initialize the model loader.

        Args:
            models_dir: Directory containing model checkpoints
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.model_path: Optional[str] = None
        self.model_type: Optional[str] = None

    def get_available_models(self) -> List[str]:
        """Get list of available model files.

        Returns:
            List of paths to .zip model files
        """
        if not self.models_dir.exists():
            return []

        models = []
        # Look for .zip files in models directory and subdirectories
        for path in self.models_dir.rglob("*.zip"):
            models.append(str(path))

        # Sort by modification time, newest first
        models.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return models

    def load_model(self, model_path: str) -> bool:
        """Load a model from file.

        Args:
            model_path: Path to the .zip model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not SB3_AVAILABLE:
            return False

        if not os.path.exists(model_path):
            return False

        try:
            # Try loading as PPO first
            try:
                self.model = PPO.load(model_path)
                self.model_type = "ppo"
            except Exception:
                # Fall back to DQN
                self.model = DQN.load(model_path)
                self.model_type = "dqn"

            self.model_path = model_path
            return True

        except Exception:
            self.model = None
            self.model_path = None
            self.model_type = None
            return False

    def get_recommendations(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        top_k: int = 3,
        temperature: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Get top-k move recommendations with confidence scores.

        For DQN models, confidence is computed using softmax over Q-values.
        For PPO models, confidence is taken from policy action probabilities.

        The softmax function converts raw scores (Q-values) into probabilities
        that sum to 1. It works by:
        1. Exponentiating each Q-value: exp(Q_i / temperature)
        2. Normalizing by the sum: exp(Q_i) / sum(exp(Q_j))

        Higher Q-values get exponentially higher probabilities, representing
        the model's confidence in that action being optimal.

        Args:
            observation: Environment observation array
            action_mask: Boolean array of valid actions
            top_k: Number of recommendations to return
            temperature: Softmax temperature (higher = more uniform distribution)

        Returns:
            List of recommendation dicts with keys:
                - action: Action index
                - source: (row, col) source position
                - target: (row, col) target position
                - confidence: Confidence percentage (0-100)
                - expected_score: Estimated score gain (placeholder)
        """
        if self.model is None:
            return []

        recommendations = []

        try:
            if self.model_type == "dqn":
                # Get Q-values for DQN
                obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]
                q_values = self.model.policy.q_net(obs_tensor).detach().numpy().flatten()

                # Apply action mask (set invalid actions to very low value)
                masked_q = q_values.copy()
                masked_q[~action_mask] = -1e9

                # Apply softmax to get confidence scores
                # Formula: confidence_i = exp(Q_i / temp) / sum(exp(Q_j / temp))
                exp_q = np.exp((masked_q - masked_q.max()) / temperature)
                exp_q[~action_mask] = 0
                confidences = exp_q / exp_q.sum()

            else:  # PPO
                # Get action probabilities from policy
                obs_tensor = self.model.policy.obs_to_tensor(observation.reshape(1, -1))[0]
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.detach().numpy().flatten()

                # Apply action mask
                masked_probs = probs.copy()
                masked_probs[~action_mask] = 0
                if masked_probs.sum() > 0:
                    confidences = masked_probs / masked_probs.sum()
                else:
                    confidences = masked_probs

            # Get top-k actions
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                return []

            # Sort by confidence
            sorted_indices = np.argsort(confidences[valid_indices])[::-1]
            top_actions = valid_indices[sorted_indices[:top_k]]

            for action in top_actions:
                source, target = self._action_to_positions(action)
                recommendations.append({
                    "action": int(action),
                    "source": source,
                    "target": target,
                    "confidence": float(confidences[action] * 100),
                    "expected_score": 0  # Placeholder - would need env to compute
                })

        except Exception:
            pass

        return recommendations

    def _action_to_positions(
        self, action: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert action index to (source, target) positions.

        Args:
            action: Action index 0-255

        Returns:
            Tuple of (source_pos, target_pos) where each is (row, col)
        """
        source_idx = action // 16
        target_idx = action % 16

        source = (source_idx // 4, source_idx % 4)
        target = (target_idx // 4, target_idx % 4)

        return source, target
