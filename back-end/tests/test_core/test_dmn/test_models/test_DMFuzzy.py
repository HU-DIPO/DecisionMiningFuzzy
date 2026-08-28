"""Tests for DMFuzzy.py."""
from typing import Tuple

import numpy as np
import pytest

from decision_mining.core import fuzzy
from decision_mining.core.dmn.rule import Rule
from decision_mining.core.dmn.models import DMFuzzy


@pytest.fixture
def trained_fuzzy() -> fuzzy.FuzzyClassifier:
    """Basic FuzzyClassifier for testing. A new one is generated for each test.

    Yields:
        fuzzy.FuzzyClassifier: Fresh FuzzyClassifier for testing.
    """
    clss = fuzzy.FuzzyClassifier(continuous_cols=np.array([0, ]))
    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    clss.fit(X, y)
    yield clss


def test_make_model(lin_med: Tuple[np.ndarray, np.ndarray]) -> None:
    """Tests the make_model functions.

    Args:
        lin_med (Tuple[np.ndarray, np.ndarray]): Linearly separable medium\
            generated test set.
    """
    X, y = lin_med
    X = np.array([X, X]).T

    model = DMFuzzy.DMFuzzy().make_model(X, y, continuous_cols=np.array([0]))
    assert isinstance(
        model, fuzzy.FuzzyClassifier), f"Type should be fuzzy.FuzzyClassifier, not {type(model)}"


def test_extract_rules(trained_fuzzy: fuzzy.FuzzyClassifier) -> None:
    """Tests extract_rules function.

    As the underlying functions have already been tested fully, we're only checking typing.

    Args:
        trained_c45 (c45.C45Classifier): Fitted C45Classifier.
        trained_fuzzy (fuzzy.FuzzyClassifier): Fitted FuzzyClassifier.
    """
    rules = DMFuzzy.DMFuzzy().extract_rules([0, 1], trained_fuzzy)
    assert isinstance(rules, list), f"Should be list, not {type(rules)}"
    assert all(isinstance(rule, Rule) for rule in rules)


def test_fuzzy_encoder() -> None:
    """Tests the fuzzy_encoder function."""
    value = [DMFuzzy.fuzzy_encoder(index, 7, DMFuzzy.TERM_ENCODER) for index in range(5)]
    expected_value = ['Lowest', 'Low', 'Low+', 'Medium', 'High-']
    assert value == expected_value, f"Should be {expected_value}, not {value}"


test_fuzzy_encoder()
