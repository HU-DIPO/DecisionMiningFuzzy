"""Tests decision_mining.core.tools.pipeline.py."""
from typing import Dict, List, Union
from xml.etree.ElementTree import ElementTree

import numpy as np
import pandas as pd

from decision_mining.api.tools import pipeline as pp


def test_validate_input(form_header_data: Dict[str, Union[List, Dict]]) -> None:
    """Tests the pipeline.validate_input function.

    Args:
        form_header_data (Dict[str, Union[List, Dict]]): Possibilities for formdata/headerdata.
    """
    base_form = form_header_data.get("base_form")
    bad_forms = form_header_data.get("bad_forms")
    base_headers = form_header_data.get("base_headers")
    bad_headers = form_header_data.get("bad_headers")
    bad_form_messages = form_header_data.get("bad_form_messages")
    bad_header_messages = form_header_data.get("bad_header_messages")

    for bad_header, bad_header_message in zip(bad_headers, bad_header_messages):
        validation = pp.validate_input(bad_header, base_form)
        assert validation.get("status") == 400, f"Response status should be 400,\
            got {validation.get('status')}"

        assert validation.get("message") == bad_header_message,\
            f"Expected error '{bad_header_message}', got {validation.get('message')}"

    for bad_form, bad_form_message in zip(bad_forms, bad_form_messages):
        validation = pp.validate_input(base_headers, bad_form)
        assert validation.get("status") == 400, f"Response status should be 400,\
            got {validation.get('status')}"

        assert validation.get("message") == bad_form_message,\
            f"Expected error '{bad_form_message}', got {validation.get('message')}"

    validation = pp.validate_input(base_headers, base_form)
    assert validation.get("status") == 200,\
        f"Response status should be 200, got {validation.get('status')}"

    assert validation.get("message") == "",\
        f"Expected error messages to be '' (empty), got {validation.get('message')}"


def test_parse_data() -> None:
    """Tests the pipeline.parse_data() function."""
    data = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ])

    data = pd.DataFrame(data)
    columns = ["col1", "col2", "col3"]
    data.columns = columns
    output = "col2"

    test_X, test_y, test_columns = pp.parse_data(data, columns, output)
    assert test_X.shape == (3, 2), f"Expected X.shape to be (3, 2), got {test_X.shape}"
    assert test_X.size == 6, f"Expected X.size to be 6, got {test_X.size}"
    assert len(test_X) == 3, f"Expected len(X) to be 3, got {len(test_X)}"

    assert test_y.shape == (3,), f"Expected y.shape to be (3,), got {test_y.shape}"
    assert test_y.size == 3, f"Expected y.size to be 3, got {test_y.size}"
    assert len(test_y) == 3, f"Expected len(y) to be 3, got {len(test_y)}"

    assert isinstance(test_columns, list), f"Expected `columns` type to be list,\
        got {type(test_columns)}"
    assert test_columns == ["col1", "col3", "col2"],\
        f"Columns list should consist of ['col1', 'col3', 'col2'], not {test_columns}"


def test_generate_dmn() -> None:
    """Tests the pipeline.generate_dmn function.

    Underlying functions have been tested fully, we're only checking typing and length.

    Args:
        trained_c45 (c45.C45Classifier): Fitted C45Classifier.
    """
    column_names = [
        ["duration", "premium", "amount", "risk"],
        ["premium", "amount", "risk", "approval"],
        ["duration", "premium", "amount", "check"],
    ]
    tree = pp.generate_dmn(column_names, [[], [], []])
    assert isinstance(tree, ElementTree), f"tree should be ElementTree, not {type(tree)}"

    root = tree.getroot()
    assert len(root) == 7, f"Expected 7 children, got {len(root)}"

    decisions = root.findall("decision")
    assert len(decisions) == 3, f"Should have 3 decisions, got {len(decisions)}"
    tables = decisions[0].findall("DecisionTable")
    assert len(tables) == 1, f"Should have 1 DecisionTable, got {len(tables)}"
