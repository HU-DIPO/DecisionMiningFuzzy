"""Tests for baseclassifier.py."""
import pytest
import numpy as np

from decision_mining.core import baseclassifier


def test_baseclassifier() -> None:
    """Tests baseclassifier for CICD testing."""
    with pytest.raises(TypeError):
        baseclassifier.BaseClassifier()


def test_score() -> None:
    """Tests score functions."""
    class FakeClassifier:
        def __init__(self) -> None:
            super().__init__()

        def predict(self, X_true: np.ndarray) -> np.ndarray:
            return np.zeros(X_true.shape[0])

    X_true = np.zeros(100)
    y_true = np.ones(100)
    baseclassifier.BaseClassifier.score(FakeClassifier(), X_true, y_true) == 0
