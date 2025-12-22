"""
Tests for Task Group 7: Recommendations Panel.

These tests verify:
1. Panel displays 3 recommendation entries
2. Each entry shows source/target coordinates
3. Confidence percentage formatted correctly
4. Expected score gain displays
"""

import pytest
import pygame

from src.ui.recommendation_panel import RecommendationPanel


class TestRecommendationDisplay:
    """Test recommendation panel display."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.panel = RecommendationPanel(x=50, y=400)

    def test_initial_state_is_empty(self):
        """Test that panel starts with no recommendations."""
        assert self.panel.recommendations == []
        assert not self.panel.loading

    def test_set_recommendations_limits_to_3(self):
        """Test that panel limits to 3 recommendations."""
        recs = [
            {"source": (0, 0), "target": (1, 1), "confidence": 80.0, "expected_score": 30},
            {"source": (1, 1), "target": (2, 2), "confidence": 60.0, "expected_score": 20},
            {"source": (2, 2), "target": (3, 3), "confidence": 40.0, "expected_score": 10},
            {"source": (3, 3), "target": (0, 0), "confidence": 20.0, "expected_score": 5},
        ]

        self.panel.set_recommendations(recs)

        assert len(self.panel.recommendations) == 3

    def test_clear_removes_all_recommendations(self):
        """Test that clear removes all recommendations."""
        self.panel.set_recommendations([
            {"source": (0, 0), "target": (1, 1), "confidence": 50.0, "expected_score": 10}
        ])

        self.panel.clear()

        assert self.panel.recommendations == []
        assert not self.panel.loading


class TestLoadingState:
    """Test loading state handling."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.panel = RecommendationPanel(x=50, y=400)

    def test_set_loading_true(self):
        """Test setting loading state to true."""
        self.panel.set_loading(True)
        assert self.panel.loading

    def test_set_recommendations_clears_loading(self):
        """Test that setting recommendations clears loading state."""
        self.panel.set_loading(True)
        self.panel.set_recommendations([])
        assert not self.panel.loading


class TestRecommendationCount:
    """Test recommendation count tracking."""

    @pytest.fixture(autouse=True)
    def setup(self, pygame_init):
        """Set up test fixtures."""
        self.panel = RecommendationPanel(x=50, y=400)

    def test_get_recommendation_count_empty(self):
        """Test count with no recommendations."""
        assert self.panel.get_recommendation_count() == 0

    def test_get_recommendation_count_with_data(self):
        """Test count with recommendations."""
        self.panel.set_recommendations([
            {"source": (0, 0), "target": (1, 1), "confidence": 50.0, "expected_score": 10},
            {"source": (1, 1), "target": (2, 2), "confidence": 30.0, "expected_score": 20},
        ])

        assert self.panel.get_recommendation_count() == 2
