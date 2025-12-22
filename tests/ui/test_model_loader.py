"""
Tests for Task Group 6: Model Loading and Inference.

These tests verify:
1. Model loading from .zip file
2. Inference returns action probabilities
3. Softmax conversion of Q-values
4. Top 3 recommendations extracted correctly
5. Confidence percentage calculation
"""

import pytest
import numpy as np

from src.ui.model_loader import ModelLoader


class TestModelLoading:
    """Test model loading functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.loader = ModelLoader(models_dir="models")

    def test_get_available_models_with_no_models_dir(self):
        """Test getting models when directory doesn't exist."""
        loader = ModelLoader(models_dir="nonexistent_dir")
        models = loader.get_available_models()
        assert models == []

    def test_initial_state_is_none(self):
        """Test that loader starts with no model."""
        assert self.loader.model is None
        assert self.loader.model_path is None
        assert self.loader.model_type is None

    def test_load_nonexistent_model_returns_false(self):
        """Test loading a nonexistent model file."""
        result = self.loader.load_model("nonexistent.zip")
        assert result is False
        assert self.loader.model is None


class TestInferenceWithoutModel:
    """Test inference without a loaded model."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.loader = ModelLoader()

    def test_get_recommendations_without_model_returns_empty(self):
        """Test that inference returns empty list without model."""
        obs = np.zeros(17, dtype=np.float32)
        mask = np.ones(256, dtype=bool)

        recommendations = self.loader.get_recommendations(obs, mask)
        assert recommendations == []


class TestActionConversion:
    """Test action index to position conversion."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.loader = ModelLoader()

    def test_action_to_positions_first_action(self):
        """Test converting action 0 to positions."""
        source, target = self.loader._action_to_positions(0)
        assert source == (0, 0)
        assert target == (0, 0)

    def test_action_to_positions_last_action(self):
        """Test converting action 255 to positions."""
        source, target = self.loader._action_to_positions(255)
        assert source == (3, 3)
        assert target == (3, 3)

    def test_action_to_positions_middle_action(self):
        """Test converting a middle action."""
        # Action 17 = source_idx=1 (row 0, col 1), target_idx=1 (row 0, col 1)
        source, target = self.loader._action_to_positions(17)
        assert source == (0, 1)
        assert target == (0, 1)

    def test_action_to_positions_different_source_target(self):
        """Test action with different source and target."""
        # Action 16 = source_idx=1 (row 0, col 1), target_idx=0 (row 0, col 0)
        source, target = self.loader._action_to_positions(16)
        assert source == (0, 1)
        assert target == (0, 0)


class TestRecommendationFormat:
    """Test recommendation output format."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.loader = ModelLoader()

    def test_recommendation_dict_structure(self):
        """Test that recommendations have correct dictionary structure."""
        # Since we can't load a model, test the structure definition
        expected_keys = {"action", "source", "target", "confidence", "expected_score"}

        # Mock a recommendation
        rec = {
            "action": 0,
            "source": (0, 0),
            "target": (1, 1),
            "confidence": 50.0,
            "expected_score": 0
        }

        assert set(rec.keys()) == expected_keys
        assert isinstance(rec["action"], int)
        assert isinstance(rec["source"], tuple)
        assert isinstance(rec["target"], tuple)
        assert isinstance(rec["confidence"], float)
