"""Creates an abstract model class for the models in the api. and a registry class for the\
    models."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from decision_mining.core.dmn.rule import Rule
from decision_mining.core.fuzzy import FuzzyClassifier

MODELS = FuzzyClassifier


class DecisionModel(ABC):
    """Factory for creating DecisionModels."""

    models = []

    @abstractmethod
    def make_model(self, X: np.ndarray, y: np.ndarray, continuous_cols: np.ndarray = None
                   ) -> MODELS:
        """Make a classifier Model, and train it on X and y.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): The target values.
            continuous_cols (np.ndarray, optional): Continuous column indices. Defaults to None.

        Returns:
            Trained classifier.
        """
        pass

    @abstractmethod
    def extract_rules(self, cols: List[str], model: MODELS) -> List[Rule]:
        """Extract rules from the classifier model.

        Args:
            cols (List[str]): columns

        Raises:
            ValueError: When the model is not trained.

        Returns:
            List[Rule]: rules
        """
        pass

    @property
    def model_name(self) -> str:
        """Get the name of the model.

        Can include spaces and special characters.
        """
        return self.model_info["name"]

    @property
    def model_id(self) -> str:
        """Get the name of the model.

        Can not include spaces and special characters.
        """
        return self.model_info["id"]

    @property
    def model_description(self) -> str:
        """Get the description of the model.

        A short description of the model. That will be shown in the UI.
        """
        return self.model_info["description"]

    @property
    def model_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the model.

        Returns:
            dict: parameters
        """
        return self.model_info["parameters"]

    def set_model_parameter(self, parameter: str, value: int) -> None:
        """Set the parameter of the model.

        Args:
            parameter (str): parameter
            value (int): value
        """
        if parameter in self.model_info["parameters"]:
            ValueError(f"Parameter {parameter} is not defined for this model.")
        self.model_info["parameters"][parameter]["value"] = value

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set multiple parameters of the model.

        Args:
            parameters (Dict[str, Any]): parameters
        """
        for parameter, value in parameters.items():
            self.set_model_parameter(parameter, value)

    def make_models(self, parsed_data: List[Tuple[np.ndarray, np.ndarray, List[str]]],
                    continuous_cols: np.ndarray) -> None:
        """Create multiple models and adds them to the self.models.

        Args:
            parsed_data (List[Tuple[np.ndarray, np.ndarray, List[str]]]): Parsed data from\
                pipeline
            continuous_cols (np.ndarray): Continuous column indices. Defaults to None
        """
        self.models = [self.make_model(X, y, cont_cols)
                       for (X, y, _), cont_cols in zip(parsed_data, continuous_cols)]

    def extract_rules_for_all_models(self, cols: List[List[str]]) -> List[List[Rule]]:
        """Extract rules from the all classifier model.

        Args:
            cols (List[List[str]]): a List of a List of columns

        Raises:
            ValueError: When the model is not trained.

        Returns:
            List[List[Rule]]: rules
        """
        if not self.models:
            raise ValueError("No models trained.")
        return [self.extract_rules(columns, model) for columns, model in zip(cols, self.models)]

    def score_models(
            self, parsed_data: List[Tuple[np.ndarray, np.ndarray, List[str]]]) -> List[float]:
        """Score all trained models on the given data.

        Args:
            parsed_data (List[Tuple[np.ndarray, np.ndarray, List[str]]]): Parsed data from\
                pipeline.

        Returns:
            List[float]: scores of the models
        """
        return [model.score(X, y) for (X, y, _), model in zip(parsed_data, self.models)]
