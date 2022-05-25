"""DMN Rule class.

This module contains the Rule object that can be used \
for generating Decision Tables in a DMN Diagram.
"""
import uuid
from collections import OrderedDict
from typing import Callable, Dict, List, Union
from xml.etree import ElementTree


import numpy as np
from more_itertools import partition


class Rule:
    """Rule class for Decision Table generation.

    Each Rule object represents a row in a decision table.

    Attributes:
        cols (dict): Dictionary mapping column indices to\
        None | any | {threshold: float, "<": bool}
    """
    # TODO: Add support for label encoders

    def __init__(self, cols: List[int]) -> None:
        """Initialise Rule object instance for Decision Table generation.

        Args:
            cols (List[int]): List of column indices.
        """
        self.cols = OrderedDict.fromkeys(cols)
        self.decision = None

    def get_basic_rule(self) -> List[str]:  # TODO: should be implemented in the Rule class
        """Get basic representation of the Rule object.

        Returns:
            List[str]: A list of strings representing a basic decision table rule.
        """
        basic_rule = []
        for value in self.cols.values():
            if isinstance(value, list):
                basic_rule.append(self.get_continuous_bounds(value))
            elif value is None:
                basic_rule.append("None")
            else:
                basic_rule.append(str(value))
        basic_rule.append(self.decision)
        return basic_rule

    @staticmethod
    def rule_generator(cols: List[int]) -> Callable[..., "Rule"]:
        """Makes a rule factory based on passed columns.

        Args:
            cols (List[int]): List of column indices.

        Returns:
            Callable[..., Rule]: Rule factory.
        """
        def rule_factory() -> "Rule":
            return Rule(cols)
        return rule_factory

    @staticmethod
    def get_continuous_bounds(conditions: List[Dict[str, Union[float, bool]]]
                              ) -> List[Union[float, str]]:
        """Get the continuous bounds of a list of conditions.

        Args:
            conditions (List[Dict[str, Union[float, bool]]]): List of conditions.

        Returns:
            List[float]: List of continuous bounds.
        """
        greater, smaller = partition(lambda condition: condition["<"], conditions)
        greater = [condition["threshold"] for condition in greater]
        smaller = [condition["threshold"] for condition in smaller]
        if smaller:
            max_bound = min(smaller)
        else:
            max_bound = np.inf
        if greater:
            min_bound = max(greater)
        else:
            min_bound = -np.inf
        return [min_bound, max_bound]

    @staticmethod
    def parse_continuous(conditions: List[Dict[str, Union[float, bool]]]) -> str:
        """Parse a continuous columns to string for xml creation.

        Args:
            conditions (List[Dict[str, Union[float, bool]]]): Columns for continuous attribute.

        Returns:
            str: [{min_bound}..{max_bound}].
        """
        min_bound, max_bound = Rule.get_continuous_bounds(conditions)
        if np.isinf(min_bound):
            min_bound = ""
        if np.isinf(max_bound):
            max_bound = ""
        return f"[{min_bound}..{max_bound}]"

    def create_xml_element(self) -> ElementTree.Element:
        """Create a Rule XML element, each Rule element is a row in a decision table.

        Returns:
            ElementTree.Element: Rule XML element.
        """
        xml_rule_element = ElementTree.Element("rule")
        xml_rule_element.set("id", f"row-{uuid.uuid4()}")

        for _, value in self.cols.items():
            input_entry = ElementTree.SubElement(xml_rule_element, "inputEntry")
            input_entry.set("id", f"InputEntry-{uuid.uuid4()}")
            input_text = ElementTree.SubElement(input_entry, "text")
            if isinstance(value, list):
                input_text.text = self.parse_continuous(value)
            elif value is None:
                input_text.text = ""
            else:
                input_text.text = str(value)

        # OutputEntry
        xml_output_entry = ElementTree.SubElement(xml_rule_element, "outputEntry")
        xml_output_entry.set("id", f"OutputEntry-{uuid.uuid4()}")
        xml_output_text = ElementTree.SubElement(xml_output_entry, "text")
        xml_output_text.text = str(self.decision)

        return xml_rule_element

    def __repr__(self) -> str:  # pragma: no cover
        """String representation of a Rule.

        Returns:
            str: String representation of a Rule.
        """
        return f"Cols: {self.cols}; Decision: {self.decision}"


def get_basic_dt_rules(rules: List[Rule]) -> List[str]:
    """Create basic decision table from a list of Rule objects.

    Args:
        rules (List[Rule]): List of rules.

    Returns:
        List[str]: A list of basic rules.
    """
    dt_rules = [rule.get_basic_rule() for rule in rules]

    return dt_rules
