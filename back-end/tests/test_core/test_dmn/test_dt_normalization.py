"""Tests dt_normalization.py."""
import pytest
from typing import Any, List

import numpy as np

import decision_mining.core.dmn.dt_normalization as dt_norm
import decision_mining.core.dmn.rule as rule


@pytest.fixture
def basic_rules_complex() -> List[List[str]]:
    """Return a basic list of rules.

    Returns:
        List[List[str]]: A list of lists of rules.
    """
    return [
        ["Low", "Low", "-", 0],
        ["-", "High", [-np.inf, 20], 2],
        ["-", "High", [20, 40], 2],
        ["-", "High", [60, 70], 2],
        ["-", "High", [100, np.inf], 2]
    ]


@pytest.fixture
def basic_non_normalized_dt_rules() -> List[List[Any]]:
    """Return a non normalized basic list of rules.

    Returns:
        List[List[Any]]: A list of lists of rules.
    """
    return[
        [1, "female", [-np.inf, 8.0], 0],
        [3, "female", [-np.inf, 1.5], 1],
        [3, "female", [1.5, 2.5], 0],
        [3, "female", [2.5, 3.5], 0],
        [3, "female", [3.5, 5.5], 1],
        [3, "female", [5.5, 12.5], 0],
        [3, "female", [12.5, 13.5], 1],
        [3, "female", [13.5, 14.25], 0],
        [3, "female", [14.25, 14.75], 0],
        [3, "female", [14.75, 15.5], 1],
        [3, "female", [15.5, 16.5], 1],
        [3, "female", [16.5, 18.5], 0],
        [3, "female", [18.5, 19.5], 1],
        [2, "male", [0.96, 3.5], 1],
        [3, "male", [0.96, 1.5], 0],
        [3, "male", [1.5, 2.5], 0],
        [3, "male", [2.5, 3.5], 1],
        [1, "male", [3.5, 4.5], 1],
        [3, "male", [3.5, 4.5], 0],
        [1, "male", [4.5, 13.0], 1],
        [2, "male", [4.5, 13.0], 1],
        [3, "male", [4.5, 5.5], 0],
        [3, "male", [5.5, 6.5], 1],
        [3, "male", [6.5, 7.5], 0],
        [3, "male", [7.5, 8.5], 0],
        [3, "male", [8.5, 9.5], 0],
        [3, "male", [9.5, 11.5], 0],
        [3, "male", [11.5, 13.0], 1],
        [1, "male", [13.0, 17.5], 1],
        [2, "male", [13.0, 18.5], 0],
        [2, "male", [18.5, 20.0], 0],
        [3, "male", [13.0, 15.5], 0],
        [3, "male", [15.5, 16.5], 0],
        [3, "male", [16.5, 17.5], 0],
        [3, "male", [17.5, 18.5], 0],
        [3, "male", [18.5, 19.5], 0],
        [3, "male", [19.5, np.inf], 1]
    ]


def test_check_if_covered(basic_rules_complex: List[List[str]]) -> None:
    """Tests the check_if_covered function."""
    basic_rules_all_covered = [["-", "-", "-", 0]]

    combinations = [
        ("Low", "Low", "-"),
        ("High", "Low", "-"),
        ("-", "High", [20, 40]),
        ("-", "High", [25, 70]),
        ("-", "High", [40, 60]),
        ("-", "High", [120, 200])
    ]
    assert dt_norm.check_if_covered([], ("Low")) is False, "Should return False, not True"

    test_values = [dt_norm.check_if_covered(basic_rules_complex, combination, np.array([
        2]), "-") for combination in combinations]
    true_values = [True, False, True, True, False, True]
    assert test_values == true_values, f"test values should be {true_values}, not {test_values}"

    # Test where all values should be covered
    test_values = [dt_norm.check_if_covered(basic_rules_all_covered, combination, np.array([
        2]), "-") for combination in combinations]
    true_values = [True, True, True, True, True, True]
    assert test_values == true_values, f"test values should be {true_values}, not {test_values}"


def test_make_rules_of_lists(basic_rules_complex: List[List[str]]) -> None:
    """Tests the make_rules_of_lists function."""
    rules = dt_norm.make_rules_of_lists(basic_rules_complex)

    assert len(rules) == 5, "Should be 5 rules, not {}".format(len(rules))
    assert all(isinstance(rule_, rule.Rule) for rule_ in rules), "All rules should be of type Rule"

    complex_rule: rule.Rule = rules[2]
    cat_complex = "High"
    cont_complex = [{"threshold": 20, "<": False},
                    {"threshold": 40, "<": True}]
    assert complex_rule.decision == 2, f"Decision should be 2, not {complex_rule.decision}"
    assert complex_rule.cols[1] == cat_complex,\
        f"Column 0 should be {cat_complex}, not {complex_rule.cols[1]}"
    assert complex_rule.cols[2] == cont_complex, \
        f"Column 1 should be {cont_complex}, not {complex_rule.cols[2]}"

    complex_rule: rule.Rule = rules[1]
    cont_complex = [{"threshold": 20, "<": True}]

    assert complex_rule.cols[2] == cont_complex, \
        f"Column 1 should be {cont_complex}, not {complex_rule.cols[2]}"


def test_normalize_dt(basic_non_normalized_dt_rules: List[List[Any]]) -> None:
    """Tests the normalize_dt function."""
    non_normalized_dt_rules = dt_norm.make_rules_of_lists(basic_non_normalized_dt_rules)
    normalized_dt_rules = dt_norm.normalize_dt(non_normalized_dt_rules, np.array([2]))

    assert len(normalized_dt_rules) == 26, f"There should be 26 rules, not \
{len(normalized_dt_rules)}"
    # check if the first rule is the same
    cont_value = [{'threshold': 12.5, '<': False}, {'threshold': 13.0, '<': True}]
    assert normalized_dt_rules[0].cols[0] == '-', \
        f"Column 0 should be '-', not {normalized_dt_rules[0].cols[0]}"
    assert normalized_dt_rules[0].cols[1] == '-',\
        f"Column 1 should be '-', not {normalized_dt_rules[0].cols[1]}"
    assert normalized_dt_rules[0].cols[2] == cont_value,\
        f"Column 2 should be '(12.5, 13.0)', not {normalized_dt_rules[0].cols[2]}"
    assert normalized_dt_rules[0].decision == 1,\
        f"Column 3 should be 1, not {normalized_dt_rules[0].decision}"
