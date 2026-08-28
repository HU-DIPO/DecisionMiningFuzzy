"""baseclassifier.py.

This module contains the BaseClassifier class, an Abstract Base Class used for
all classifiers in this package.
"""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score


class BaseClassifier(ABC):
    """Abstract Base Class for a classifier."""

    # True if the continuous values are categoricalized
    categoricalize_continuous_values: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Build a classifier from the training set.

        Args:
            X_train(np.ndarray): Training input samples
            y_train(np.ndarray): The target values
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for X.

        Args:
            X(np.ndarray): Input samples

        Returns:
            np.ndarray: The predicted classes
        """
        pass

    def score(self, X_true: np.ndarray, y_true: np.ndarray) -> float:
        """Return `accuracy_score` on the given test data and labels.

        Args:
            X_true(np.ndarray): Test input samples
            y_true(np.ndarray): True labels for `X_true`

        Returns:
            float: Score
        """
        y_pred = self.predict(X_true)
        return accuracy_score(y_true, y_pred)
