"""dt_normalization.py.

This module contains functions to
normalize a Decision Table.
"""
from typing import Any, Dict, List
import numpy as np
import numpy.ma as ma

from decision_mining.core.dmn.rule import Rule, get_basic_dt_rules
from itertools import product
from tqdm.contrib.itertools import product as tqdm_product


def check_if_covered(basic_rules: list, combination: tuple,
                     continuous_cols: np.array = np.array([]),
                     wildcard_value: str = "-") -> bool:
    """Check if combination is already covered by a rule in basic_new_rules.

    Args:
        basic_new_rules (list): List of covered rules.
        combination (tuple): Combination to check.
        continuous_cols (np.array): Continuous columns. Defaults to np.array([]).
        wildcard_value (str): Wildcard value. Defaults to "-".

    Returns:
        bool: Returns true if combination is covered, else False.
    """
    # Check if basic_rules is not empty
    if not basic_rules:
        return False

    basic_rules = np.array(basic_rules, dtype=object)

    covered_rules_bool = np.ones_like(basic_rules[:, :-1], dtype=bool)
    for column_idx, value in enumerate(combination):
        if value == wildcard_value:
            covered_rules_bool[:, column_idx] = True

        elif column_idx not in continuous_cols:
            covered_rules_bool[:, column_idx] = basic_rules[:, column_idx] == value

        else:
            cov_con_arr = np.array([i if i != '-' else [np.nan, np.nan]
                                   for i in basic_rules[:, column_idx]])
            min_, max_ = value

            cov_test1 = np.all([cov_con_arr[:, 0] <= min_, min_ < cov_con_arr[:, 1]], axis=0)
            cov_test2 = np.all([cov_con_arr[:, 0] < max_, max_ <= cov_con_arr[:, 1]], axis=0)
            covered_rules_bool[:, column_idx] = np.any([cov_test1, cov_test2], axis=0)

            covered_rules_bool[:, column_idx] = np.any(
                [covered_rules_bool[:, column_idx], np.isnan(cov_con_arr[:, 0])], axis=0)

    covered_rules_bool = np.logical_or(covered_rules_bool, basic_rules[:, :-1] == wildcard_value)
    return np.any(np.all(covered_rules_bool, axis=1))


def make_rules_of_lists(basic_rules: List[List[str]]) -> List[Rule]:
    """Create rule object of a list of basic rules.

    Args:
        basic_rules (List[List[str]]): List of basic rules.

    Returns:
        List[Rule]: List of rules.
    """
    RuleFactory = Rule.rule_generator(range(len(basic_rules[0]) - 2))

    rules = []
    for basic_rule in basic_rules:
        rule = RuleFactory()
        for column_idx, value in enumerate(basic_rule[:-1]):
            if isinstance(value, list) or isinstance(value, tuple):
                rule.cols[column_idx] = []
                for idx, val in enumerate(value):
                    # Do not include infinite values
                    if not np.isinf(val):
                        rule.cols[column_idx].append({"threshold": val, "<": bool(idx)})
            else:
                rule.cols[column_idx] = value
        rule.decision = basic_rule[-1]
        rules.append(rule)
    return rules


def normalize_dt(
        dt_rules: List[Rule],
        continuous_cols: np.array, wildcard_value: str = "-") -> List[Rule]:
    """Normalize rules.

    Args:
        dt_rules (List[Rule]): List of decision table rules.
        continuous_cols (np.array): Continuous columns.
        wildcard_value (str, optional): Wildcard Value. Defaults to "-".

    Returns:
        List[Rule]: Normalized rules.
    """
    rules = np.array(get_basic_dt_rules(dt_rules), dtype=object)

    basic_new_rules = []

    masks = {}
    # Creating masks for every column.
    for column_idx in range(rules.shape[1] - 1):
        column_mask = {}

        # Create wildcard mask, where every value is valid.
        column_mask[wildcard_value] = False

        # Creating mask for value in categorical column.
        if column_idx not in continuous_cols:
            unique_values = np.unique(rules[:, column_idx])
            for unique_value in unique_values:
                column_mask[unique_value] = (rules[:, column_idx] != unique_value)

        # Creating mask for value in continuous column.
        else:
            con_arr = ma.masked_array(np.vstack(rules[:, column_idx]), mask=False)
            unique_min = np.unique(con_arr[:, 0])
            unique_max = np.unique(con_arr[:, 1])

            # TODO: Creates a lot of masks....
            for min_, max_ in product(unique_min, unique_max[::-1]):
                if min_ > max_ or min_ == max_:  # TODO: find a way to avoid this
                    continue
                con_arr.mask = False

                con_arr.mask[:, 0] = (con_arr[:, 0] >= max_)
                con_arr.mask[:, 1] = (con_arr[:, 1] <= min_)

                masked_array = ma.mask_rows(con_arr)

                column_mask[(min_, max_)] = masked_array.mask.any(axis=1)

        masks[column_idx] = column_mask

    # Creating Rules for every combination of created masks.
    combinations = [column_mask_.keys() for column_mask_ in masks.values()]
    prod = tqdm_product(*combinations)
    for combination in prod:
        # Combine the masks of the combination to one mask
        combined_mask = np.zeros_like(rules, dtype=bool)
        for i, value in enumerate(combination):
            combined_mask[:, i] = masks[i][value]

        output_mask = ~np.any(combined_mask, axis=1)
        unique_outputs = np.unique(rules[:, -1][output_mask])
        # if combination only has 1 unique output
        if len(unique_outputs) == 1:
            # Check if combination isn't already covered
            if check_if_covered(basic_new_rules, combination, continuous_cols, wildcard_value):
                # Rule has already been covered
                continue

            basic_new_rules.append(list(combination) + [unique_outputs[0]])
    return make_rules_of_lists(basic_new_rules)


if __name__ == "__main__":  # pragma: no cover
    import pandas as pd

    def get_rules(data: dict, model_id: str = "DM45", parameters: Dict[str, Any] = None) -> None:
        """Get all rules the way the program normally will do."""
        from decision_mining.regester_models import registered_models
        from decision_mining.api.tools import pipeline as pp

        cols = [data["columns"]]
        output = [data["output"]]
        continuous_cols = [data["continuous_cols"]]
        data = [data["data"]]
        # parse received data
        parsed_data = list(map(pp.parse_data, data, cols, output))

        # Get DecisionModel
        decision_model = registered_models.get_model(model_id)

        # Set model parameters
        if parameters is not None:
            decision_model.set_model_parameters(parameters)

        decision_model.make_models(parsed_data, continuous_cols)
        rules = list(decision_model.extract_rules_for_all_models(cols))
        # scores = decision_model.score_models(parsed_data)

        return decision_model, rules

    def main(model_id: str = "DM45", parameters: Dict[str, Any] = None,) -> None:
        """Test all used functions."""
        # data_columns = ["Pclass", "Sex", "Age", "Fare", "Survived"]
        # data_columns = ["Pclass", "Sex", "Fare", "Survived"]
        data_columns = ["Pclass", "Sex", "Age", "Survived"]
        # continuous_cols = np.array([2, 3])
        continuous_cols = np.array([2])

        data = {
            "data": pd.read_csv(
                r"https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
                usecols=data_columns),
            "columns": data_columns,
            "continuous_cols": continuous_cols,
            "output": data_columns[-1]
        }
        decision_model, rules = get_rules(data, model_id, parameters)
        print(decision_model.translate_all_fuzzy_values())
        for model, rules in zip(decision_model.models, rules):
            continuous_cols = model.continuous_cols
            if model.categoricalize_continuous_values:
                continuous_cols = np.array([])
            df_before = pd.DataFrame(get_basic_dt_rules(rules), columns=data_columns)
            print(df_before)
            df_before.to_csv(f"{model_id}_default_rules.csv")
            normalized_rules = normalize_dt(rules, continuous_cols)
            df_after = pd.DataFrame(get_basic_dt_rules(normalized_rules), columns=data_columns)
            df_after.to_csv(f"{model_id}_normalized_rules.csv")

        normalized_rules = []
        for model, rules in zip(decision_model.models, rules):
            continuous_cols = model.continuous_cols
            if model.categoricalize_continuous_values:
                continuous_cols = np.array([])
            normalized_rules.append(normalize_dt(rules, continuous_cols))
        rules = normalized_rules

    main("DMFuzzy", {"minimal_gain_ratio": 0.05})
    # main("DM45", {"min_objs": 1})
    # main("DM45_pruned")
