"""Fuzzy Algorithm.

The Fuzzy Algorithm makes fuzzy values from crisp continuous values.
these fuzzy values will be split based on the information gain ratio it has with it's target value.
this will be extended to a certain percentage to create a fuzzy transition period
between the fuzzy values.

This file contains one class `FuzzyClassifier` which is used in the API containing all of \
the fuzzy functions.
"""
from itertools import product
from statistics import mode
from typing import List, Tuple

import numpy as np
from scipy import stats

from decision_mining.core.baseclassifier import BaseClassifier


def split_info(attribute: np.ndarray, target: np.ndarray) -> float:
    """The SplitInfo equation, calculates the metric SplitInfo.

    SplitInfo(A, T) = - sum((length of T where v) / (length of T) *\
    log2((length of T where v) / (length of T)))

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: SplitInfo.
    """
    _split_info = 0.
    classes = np.unique(attribute)
    for value in classes:
        pkv = attribute == value
        split = np.count_nonzero(pkv) / target.shape[0]
        _split_info += split * np.log2(split)

    return -_split_info


def gain(attribute: np.ndarray, target: np.ndarray) -> float:
    """Information gain equation, used to calculate the metric information gain.

    Gain(A, T) = Entropy(A) - sum((length of T where v) / \
    (length of T) Entropy(A where v) for v in unique(A)

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: Information Gain.
    """
    classes, counts = np.unique(target, return_counts=True)
    entropy = stats.entropy(counts, base=2.)
    sub_entropy = 0.
    for value in np.unique(attribute):
        mask = attribute == value
        _, pkv = np.unique(target[mask], return_counts=True)
        normalizer = np.count_nonzero(mask) / target.shape[0]
        sub_entropy += normalizer * stats.entropy(pkv, base=2.)

    return entropy - sub_entropy


def gain_ratio(attribute: np.ndarray, target: np.ndarray) -> float:
    """The GainRatio equation, calculates the metric information gain ratio.

    The greater the GainRatio, the greater the information gain.

    0 <= GainRatio <= 1

    GainRatio(A, T) = Gain(A, T) / SplitInfo(A, T)

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: GainRatio(A, T).
    """
    # TODO: Vectorise GainRatio equation: speed+ readability-
    # If there's only one value left in the attribute, the GainRatio is 0.
    if (attribute == attribute[0]).all():
        return 0.
    return gain(attribute, target) / split_info(attribute, target)


def find_threshold(attribute: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Threshold function as defined in "Improved Use of Continuous Attributes in C4.5" by R. Quinlan.

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data, continuous values.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        Tuple[float, float]: threshold, gainratio of threshold.
    """
    attr_c = np.sort(attribute)  # Sorted copy of attribute
    thresholds = (attr_c[1:] + attr_c[:-1]) / 2  # All possible thresholds
    best_t = 0.  # Threshold with greatest gain ratio
    # Greatest gain ratio (gain ratio at threshold best_t)
    best_gain_ratio = 0.
    for t in thresholds:
        attr_t = attribute < t
        gain_ratio_at_t = gain_ratio(attr_t, target)
        if gain_ratio_at_t > best_gain_ratio:
            best_gain_ratio = gain_ratio_at_t
            best_t = t

    return best_t, best_gain_ratio


def get_all_thresholds(attribute: np.ndarray, target: np.ndarray,
                       minimal_gain_ratio: float = 0.3) -> List[float]:
    """Find multiple tresholds based on the gain_ratio.

    This function returns multiple thresholds on which the data can be split.
    For each data point in the attribute the gain_ratio will be defined. On the data point \
    with the highest gain_ratio a split will be made if it's higher than the minimal_gain_ratio\
    in the dataset.
    This will repeat until there is no gain_ratio higher than the minimal_gain_ratio.

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data. MUST be continuous.
        target (np.ndarray): Target or `y` in training data.
        minimal_gain_ratio (float, optional): Minimal gain ratio that is required to split\
                                              the dataset. Defaults to 0.3.

    Returns:
        List[float]: List of thresholds.
    """
    threshold = []
    masks = {}
    best_t, gain_ratio = find_threshold(attribute, target)
    if gain_ratio > minimal_gain_ratio:
        threshold.append(best_t)
        masks[best_t] = np.ones(attribute.shape, dtype=bool)
        i = 0
        while i < len(threshold):
            # combine with prev mask
            mask = attribute < threshold[i]  # TODO remove duplication
            low_mask = masks[threshold[i]] & mask
            high_mask = masks[threshold[i]] & ~mask
            low_t, low_gain_ratio = find_threshold(attribute[low_mask], target[low_mask])
            high_t, high_gain_ratio = find_threshold(attribute[high_mask], target[high_mask])
            if low_gain_ratio > minimal_gain_ratio:
                threshold.append(low_t)
                masks[low_t] = low_mask

            if high_gain_ratio > minimal_gain_ratio:
                threshold.append(high_t)
                masks[high_t] = high_mask

            if (low_gain_ratio > minimal_gain_ratio) & (high_gain_ratio > minimal_gain_ratio):
                threshold.remove(threshold[i])  # TODO is this the best
            else:
                i += 1

        return sorted(threshold)

    else:
        # Gain ratio too low for split
        return [attribute.max()]


def extend_thresholds(attribute: np.ndarray, thresholds: List[float],
                      percentage: int = 5) -> List[Tuple[float, float]]:
    """Extend threshold with a certain percentage of the subset.

    This will create a fuzzy transition period between two fuzzy values.

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        thresholds (List[float]): list of thresholds that splits the dataset.
        percentage (int): Percentage of which to increase the transition between threshold. \
                          Defaults to 5.

    Returns:
        List[Tuple[float, float]]: List of tuples, lower and upper best thresholds.
    """
    attribute = np.sort(attribute)
    percentage = percentage / 100
    if thresholds[0] == attribute.max():  # if there is no split made.
        return [(thresholds[0], thresholds[0] + 1)]

    subsets = []
    for t_index in range(len(thresholds) + 1):
        if t_index != 0:
            lower_limit = thresholds[t_index - 1]
        else:
            lower_limit = attribute.min()

        if t_index != len(thresholds):
            upper_limit = thresholds[t_index]
        else:
            upper_limit = attribute.max()

        subsets.append(attribute[(lower_limit <= attribute) & (attribute <= upper_limit)])

    extended_thresholds = []
    for i in range(len(subsets) - 1):
        index_low = int(subsets[i].size - percentage * subsets[i].size)
        index_high = int(percentage * subsets[i + 1].size)
        extended_thresholds.append((subsets[i][index_low], subsets[i + 1][index_high]))

    return extended_thresholds


def fuzzify_value(value: float, thresholds: List[List[float]]) -> np.ndarray:
    """Make fuzzy values from crisp values based on a list of thresholds.

    Args:
        value (float): Value to be fuzzified.
        thresholds (List[List[float]]): Thresholds that define the fuzzification.

    Returns:
        np.ndarray: Array of size (len(thresholds)).
    """
    fuzzy_value = np.zeros(len(thresholds) + 1)
    for i, t in enumerate(thresholds):
        if i == 0 and value < t[0]:
            fuzzy_value[0] = 1
            return fuzzy_value
        elif i == len(thresholds) - 1 and value > t[1]:
            fuzzy_value[-1] = 1
            return fuzzy_value
        elif t[0] <= value <= t[1]:
            m, c = lin_equ((t[0], 1,), (t[1], 0))
            fuzzy_value[i] = m * value + c
            fuzzy_value[i + 1] = 1 - (m * value + c)
            return fuzzy_value
        elif t[1] < value < thresholds[i + 1][0]:
            fuzzy_value[i + 1] = 1
            return fuzzy_value


def lin_equ(dot1: Tuple[int, int], dot2: Tuple[int, int]) -> Tuple[float, float]:
    """Get Linear Equation from 2 points.

    Args:
        dot1 (Tuple[int, int]): (x,y) point.
        dot2 (Tuple[int, int]): (x,y) point.

    Returns:
        Tuple[float, float]: m, c
    """
    m = (dot2[1] - dot1[1]) / (dot2[0] - dot1[0])
    c = (dot2[1] - (m * dot2[0]))
    return m, c


class FuzzyClassifier(BaseClassifier):
    """Fuzzy classifier.

    Classifier that transforms continuous crisp values to fuzzy values.
    By which it creates a ruleset with all possible combinations and the target's mode of that \
    combination. With these rulesets it is possible to create a prediction.
    If a combination of values is new to the ruleset the mode of the entire \
    y_train set will be given.

    Can only accept values that are not NaN.

    Args:
        continuous_cols (np.ndarray): Index array for continuous columns. Defaults to None.
    """

    categoricalize_continuous_values: bool = True

    def __init__(self, continuous_cols: np.ndarray = np.array([]),
                 minimal_gain_ratio: float = 0.3) -> None:
        """Init for FuzzyClassifier.

        Args:
            continuous_cols (np.ndarray): Index array for continuous columns. \
                                          Defaults to np.array([]).
            minimal_gain_ratio (float): Minimal gain ratio that is required to split\
                                        the dataset. Defaults to 0.3
        """
        super().__init__()
        self.continuous_cols = continuous_cols
        self.minimal_gain_ratio = minimal_gain_ratio
        self.threshold_cols = {}
        self.rules = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Build a classifier from the training set.

        Args:
            X_train (np.ndarray): Training input samples
            y_train (np.ndarray): The target values.
        """
        self.y_dtype = y_train.dtype
        self.x_dtype = X_train.dtype
        self.y_mode = mode(y_train)

        X: np.ndarray = X_train.copy()
        # Fuzzify continuous columns
        for column_idx in self.continuous_cols:
            thresholds = get_all_thresholds(X[:, column_idx], y_train, self.minimal_gain_ratio)
            thresholds = extend_thresholds(X[:, column_idx], thresholds)
            self.threshold_cols[column_idx] = thresholds
            fuzzy_values = np.stack([fuzzify_value(val, thresholds)
                                     for val in X[:, column_idx]], axis=0)
            X[:, column_idx] = fuzzy_values.argmax(axis=1)

        # Get combination of unique values and create rules.
        combinations = product(*[np.unique(column) for column in X.T])

        for combination in combinations:
            # TODO: create more efficient/cleaner mask
            mask = np.array([(row == np.array(combination,
                                              dtype=self.x_dtype)).all() for row in X])
            com_results = y_train[mask]
            if com_results.size != 0:  # TODO: temporary fix for empty combinations
                self.rules[tuple(combination)] = mode(com_results)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for X.

        Args:
            X (np.ndarray): Input samples.

        Returns:
            np.ndarray: The predicted classes.
        """
        for column_idx in self.continuous_cols:
            fuzzy_values = np.stack([fuzzify_value(val, self.threshold_cols[column_idx]
                                                   ) for val in X[:, column_idx]], axis=0)
            X[:, column_idx] = fuzzy_values.argmax(axis=1)

        prediction = np.array([self.rules.get(tuple(row), self.y_mode) for row in X],
                              dtype=self.y_dtype)

        return prediction


if __name__ == "__main__":  # pragma: no cover
    # create test dataset
    np.random.seed(10)
    cold = np.random.normal(10, 4, 50)
    warm = np.random.normal(25, 4, 50)
    hot = np.random.normal(35, 4, 50)
    hothot = np.random.normal(45, 4, 50)
    X = np.concatenate((cold, warm, hot, hothot))
    y = np.array([*["cold"] * 50,
                  *["warm"] * 50,
                  *["hot"] * 50,
                  *["hothot"] * 50])

    thresholds = get_all_thresholds(X, y, 0.3)
    print(f"The found thresholds are:\n\t{thresholds}")
    thresholds = extend_thresholds(X, thresholds)
    print(f"The extended thresholds are:\n\t{thresholds}")

    FCls = FuzzyClassifier([0])
    FCls.fit(X.reshape((X.shape[0], 1)), y)
    prediction = FCls.predict(np.array([[22], [44]]))
    print(prediction)
