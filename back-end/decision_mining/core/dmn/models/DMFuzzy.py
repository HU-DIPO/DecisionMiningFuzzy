"""Decision Model for the Fuzzy algortime."""

from typing import Dict, List, Union
from math import floor

import numpy as np

from decision_mining.core.dmn.models.decisionModel import DecisionModel
from decision_mining.core.dmn.rule import Rule
from decision_mining.core.fuzzy import FuzzyClassifier

TERM_ENCODER = {
    "nadir": "Lowest",
    "Q1": "Low",
    "median": "Medium",
    "Q3": "High",
    "zenith": "Highest"
}


def fuzzy_encoder(index: int, max_index: int, term_encoder: dict = TERM_ENCODER) -> str:
    """Encode fuzzy value to fuzzy term.

    Args:
        value (int): value between 0 and max_value or max_value.
        len_array (int): Length of the array
        term_encoder (dict, optional): terms to which it will be encoded. Defaults to TERM_ENCODER.

    Returns:
        str: encoded term.
    """
    value_dict = {
        "median": floor(max_index * 0.50),
        "nadir": 0,
        "Q1": floor(max_index * 0.25),
        "Q3": floor(max_index * 0.75),
        "zenith": max_index
    }
    for key, val in value_dict.items():
        if index == val:
            return term_encoder[key]
    if index < value_dict["median"]:
        quarter = "Q1"
    else:
        quarter = "Q3"

    distance = int(index - value_dict[quarter])
    operator = '+'
    if distance < 0:
        operator = '-'
    return f"{term_encoder[quarter]}{operator * abs(distance)}"


class DMFuzzy(DecisionModel):
    """Decision Model for the Fuzzy Algortime."""

    def __init__(self) -> None:
        """Initialize the DecisionModel."""
        parameters = {
            "minimal_gain_ratio": {
                "value": 0.3,
                "min": 0.0,
                "max": 0.9,
                "step": 0.05,
                "type": "float",
                "description": (
                    "The minimal gain ratio that must be reached to be considered a new rule."
                )
            }
        }
        self.model_info = {
            "name": "DMFuzzy",
            "id": "DMFuzzy",
            "description": (
                "DMFuzzy is a classification model based on the model of the Discovery of Fuzzy "
                "DMN Decision Models from "
                "Event Logs(Bazhenova, Haarmann, Ihde, Solti, & Weske, 2017). "
                "It discretises the continuous values, resulting in fuzzy terms like "
                "'low', 'medium', and 'high'"
            ),
            "parameters": parameters
        }

    def make_model(self, X: np.ndarray, y: np.ndarray,
                   continuous_cols: np.ndarray = None) -> FuzzyClassifier:
        """Make a classifier Model, and train it on X and y.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): The target values.
            cols (List[str]): columns
            continuous_cols (np.ndarray, optional): Continuous column indices. Defaults to None.

        Returns:
            Trained classifier.
        """
        clsfr = FuzzyClassifier(continuous_cols=continuous_cols,
                                minimal_gain_ratio=self.model_info["parameters"]
                                ["minimal_gain_ratio"]["value"])
        clsfr.fit(X, y)

        return clsfr

    def extract_rules(self, cols: List[str], model: FuzzyClassifier) -> List[Rule]:
        """Extract rules from the classifier model.

        Args:
            cols (List[str]): columns
            model (FuzzyClassifier): Fuzzy classifier of the decision.

        Raises:
            ValueError: When the model is not trained.

        Returns:
            List[Rule]: rules
        """
        return self.make_rules(range(len(cols) - 1), model)

    @staticmethod
    def make_rules(cols: List[int], FuzzyClassifier: FuzzyClassifier) -> List[Rule]:
        """Make a list of Rule objects based on the FuzzyClassifier rules.

        Args:
            cols (list): List of column indices.
            FuzzyClassifier (FuzzyClassifier): Fuzzy classifier of the decision.

        Returns:
            List[Rule]: List of complete Rule objects.
        """
        RuleFactory = Rule.rule_generator(cols)

        rules = []
        for key, decision_val in FuzzyClassifier.rules.items():
            rule = RuleFactory()
            for attr, val in enumerate(key):
                if attr in FuzzyClassifier.continuous_cols:  # if fuzzy value
                    rule.cols[attr] = fuzzy_encoder(val,
                                                    len(FuzzyClassifier.threshold_cols[attr]))
                else:
                    rule.cols[attr] = val
            rule.decision = decision_val
            rules.append(rule)

        return rules
