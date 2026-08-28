"""Testing fuzzy.py."""

from typing import Tuple

import pytest

import numpy as np
from decision_mining.core import fuzzy


def test_gain(lin_small: Tuple[np.ndarray, np.ndarray],
              med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the gain function.

    Args:
        lin_small (Tuple[np.ndarray, np.ndarray]): Small generated test set.
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = lin_small
    mask = X[:, 0] == 0
    S = X[mask][:, 0]
    C = y[mask]
    assert abs((g := fuzzy.gain(S, C)) - (t := 0.)) < 1e-3, f"{g=}, expected {t}"
    X, y = med_set
    assert abs((g := fuzzy.gain(X, y)) - (t := 0.05977)
               ) < 1e-3, f"{g=}, expected {t}"


def test_split_info(med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the split_info function.

    Args:
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set

    assert abs((g := fuzzy.split_info(X, y)) - (t := 0.971)
               ) < 1e-3, f"{g=}, expected {t}"


def test_gain_ratio(med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the split_info function.

    Args:
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set

    assert abs((g := fuzzy.gain_ratio(X, y)) - (t := 0.0616)
               ) < 1e-3, f"{g=}, expected {t}"
    X = np.zeros(y.shape)
    assert (g := fuzzy.gain_ratio(X, y)) == (t := 0.), f"{g=}, expected {t}"


def test_find_threshold() -> None:
    """Tests the fuzzy.find_threshold function."""
    X = np.arange(100)
    y = (X > 50).astype(np.int32)

    threshold = 50.5
    gain = 1.

    thresh_test, gain_test = fuzzy.find_threshold(X, y)
    assert thresh_test == threshold, f"Expected threshold to be {threshold}, not {thresh_test}"
    assert abs(gain - gain_test) < 1e-3, f"Expected gain to be {gain}, not {gain_test}"

    y = (X > 75).astype(np.int32)
    threshold = 75.5
    thresh_test, gain_test = fuzzy.find_threshold(X, y)
    assert thresh_test == threshold, f"Expected threshold to be {threshold}, not {thresh_test}"
    assert abs(gain - gain_test) < 1e-3, f"Expected gain to be {gain}, not {gain_test}"


def test_get_all_thresholds(large_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test fuzzy.get_all_thresholds function.

    Args:
        large_set (Tuple[np.ndarray, np.ndarray]): Large pregenerated dataset
    """
    X, y = large_set
    X = np.squeeze(X)
    true_thresholds = [16.205649786709955,
                       16.58182020965245,
                       16.85645688713354,
                       41.73065170702169]
    # impractical split that ensures every possible split is covered
    thresholds = fuzzy.get_all_thresholds(X, y, 0.2)

    assert all(abs(pred - true) < 1e2
               for pred, true in
               zip(thresholds, true_thresholds)), f"{thresholds}, expected{true_thresholds}"

    true_thresholds = [53.900198555292775]
    thresholds = fuzzy.get_all_thresholds(X, y, 1)
    assert all(abs(pred - true) < 1e2
               for pred, true in
               zip(thresholds, true_thresholds)), f"{thresholds}, expected{true_thresholds}"


def test_extend_thresholds(large_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test fuzzy.extend_thresholds function.

    Args:
        large_set (Tuple[np.ndarray, np.ndarray]): Large pregenerated dataset
    """
    X, _ = large_set
    thresholds = [16.205649786709955, 29.538949160007757, 41.73065170702169]
    true_extended_thresholds = np.array([[15.326346016518073, 17.02242492103577],
                                         [29.184731427233057, 30.096656077047783],
                                         [40.91075723089042, 42.17171421173134]])
    extended_thresholds = np.array(fuzzy.extend_thresholds(X, thresholds))

    assert all(abs(pred - true) < 1e2 for pred, true in
               zip(extended_thresholds.flatten(),
                   true_extended_thresholds.flatten())), f"{extended_thresholds},\
                   expected{true_extended_thresholds}"
    thresholds = [53.900198555292775]
    true_extended_thresholds = np.array([(53.900198555292775, 54.900198555292775)])
    extended_thresholds = np.array(fuzzy.extend_thresholds(X, thresholds))

    assert all(abs(pred - true) < 1e2 for pred, true in
               zip(extended_thresholds.flatten(),
                   true_extended_thresholds.flatten())), f"{extended_thresholds},\
                   expected{true_extended_thresholds}"


def test_fuzzify_value() -> None:
    """Test fuzzy.fuzzify_value function."""
    thresholds = np.array([[14, 18],
                           [20, 30]])
    values_to_test = [13, 16, 19, 34]
    true_values = np.array([[1.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])

    for i in range(len(values_to_test)):
        fuzzy_value = fuzzy.fuzzify_value(values_to_test[i], thresholds)
        print(fuzzy_value)
        assert all(abs(pred - true) < 1e2 for pred, true in
                   zip(fuzzy_value, true_values[i])), f"{fuzzy_value}, expected {true_values[i]}"


def test_lin_equ() -> None:
    """Test fuzzy.lin_equ function."""
    dot1 = (0., 0.)
    dot2 = (10., 10.)
    m, c = fuzzy.lin_equ(dot1, dot2)
    assert (1.0, 0.0) == (m, c), f"{(m, c)}, expected{(1.0, 0.0)}"


@pytest.fixture
def clsfr() -> fuzzy.FuzzyClassifier:
    """Basic FuzzyClassifier for testing. A new one is generated for each test.

    Yields:
        fuzzy.FuzzyClassifier: Fresh FuzzyClassifier for testing.
    """
    classifier = fuzzy.FuzzyClassifier([0])
    yield classifier


def test_fuzzy_fit(clsfr: fuzzy.FuzzyClassifier,
                   mixed_small: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing FuzzyClassifier's fit function.

    Args:
        clsfr (fuzzy.FuzzyClassifier): Fresh FuzzyClassifier for testing.
        mixed_small (Tuple[np.ndarray, np.ndarray]): small mixed dataset
    """
    X, y = mixed_small
    clsfr.fit(X, y)
    value = clsfr.predict(np.array([[3, "True"]], dtype=object))
    assert value == [1], f"{value}, should be [1]"


def test_fuzzy_predict(clsfr: fuzzy.FuzzyClassifier,
                       large_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing FuzzyClassifier's predict function.

    Args:
        clsfr (fuzzy.FuzzyClassifier): Fresh FuzzyClassifier for testing.
        large_set (Tuple[np.ndarray, np.ndarray]): Large pregenerated dataset
    """
    X, y = large_set
    clsfr.fit(X.reshape((X.shape[0], 1)), y)
    prediction = clsfr.predict(np.array([[22], [44]]))

    true_value = np.array(['warm', 'hothot'])
    assert (prediction == true_value).all(), f"{prediction}, expected {true_value}"
