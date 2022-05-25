"""API pipeline.

Contains pipeline functions that are executed while performing API requests.
"""
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from decision_mining.core.dmn import dmn_generation as dmng
from decision_mining.core.dmn.rule import Rule


def validate_input(headers: Dict, form: Dict) -> Dict[str, Union[int, str]]:
    """Validates Form Input.

    Args:
        headers (Dict): Flask Request Headers.
        form (Dict): Flask POST Request Formdata.

    Returns:
        Dict[str, Union[int, str]]: Status and error messages, if any.
    """
    messages = []

    if "token" not in headers:
        messages.append("`headers` does not contain `token`")
    elif not isinstance(headers.get("token"), str):
        messages.append("Field `token` must be of type `str`")

    if "cols" not in form:
        messages.append("Field `cols` is required")
    elif not isinstance(form.get("cols"), list):
        messages.append("Field `cols` must be of type `list`")

    if "output" not in form:
        messages.append("Field `output` is required")
    elif not isinstance(form.get("output"), list):
        messages.append("Field `output` must be of type `list`")

    if "model_id" not in form:
        messages.append("Field `model_id` is required")
    elif not isinstance(form.get("model_id"), str):
        messages.append("Field `model_id` must be of type `str`")

    if "normalize_bool" not in form:
        messages.append("Field `normalize_bool` is required")
    elif not isinstance(form.get("normalize_bool"), str):
        messages.append("Field `normalize_bool` must be of type `str`")

    if "continuous_cols" not in form:
        messages.append("Field `continuous_cols` is required")
    elif not isinstance(form.get("continuous_cols"), list):
        messages.append("Field `continuous_cols` must be of type `list`")

    if len(messages) == 0:
        return {"status": 200, "message": ""}

    return {"status": 400, "message": ";".join(messages)}


def parse_data(data: pd.DataFrame, columns: List,
               output: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Parses input DataFrame into [X, y] training values.

    Args:
        data (pd.DataFrame): Pandas DataFrame containing Input Data.
        columns (List): List of columns names.
        output (str): Column marked as output.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: X, y training values and column_names.
    """
    columns.append(columns.pop(columns.index(output)))
    data = data[columns]

    y = data[output].to_numpy()
    data = data.drop(columns=[output])
    X = data.to_numpy()

    return X, y, columns


def generate_dmn(column_names: List[List[str]],
                 rules: List[List[Rule]]) -> ElementTree.ElementTree:
    """Generate a DMN formatted XML tree containing the DRD and decision tables.

    Args:
        column_names (List[List[str]]): List of names per decision.
        rules (List[List[Rule]]): List of rules per decision.

    Returns:
        ElementTree.ElementTree: DMN formatted XML tree.
    """
    drd_objects = dmng.create_node_objects(column_names)
    decision_nodes = dmng.create_dependencies(column_names, drd_objects)
    for i, rule_list in enumerate(rules):
        decision_nodes[i].add_rules(rule_list)

    tree = dmng.create_xml(drd_objects, decision_nodes)
    return tree
