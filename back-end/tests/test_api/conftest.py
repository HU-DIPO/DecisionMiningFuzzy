"""Pytest fixtures.

Contains Pytest Fixtures for API endpoints.
"""
from typing import List, Dict, Union

import pytest


@pytest.fixture()
def form_header_data() -> Dict[str, Union[List, Dict]]:
    """Generate formdata for API requests.

    Makes list of formdata and headerdata dictionaries.
    formdata/headerdata contains possibilities for 'bad' requests.

    Returns:
        Dict[str, Union[List, Dict]]: correct form/headers, 'bad' form/headers, error messages
    """
    base_headers = {
        "token": "email@email.com"
    }

    headers1 = {
        "test": "test"
    }

    headers2 = {
        "token": ["hello", "there"]
    }

    base_form = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form1 = {
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form2 = {
        "cols": "hello there",
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form3 = {
        "cols": [["col1", "col2", "col3"]],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form4 = {
        "cols": [["col1", "col2", "col3"]],
        "output": "hello there",
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form5 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form6 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": ["hello there"],
        "normalize_bool": "false",
        "continuous_cols": [[0]]
    }

    form7 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
    }

    form8 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": "false",
        "continuous_cols": "hello there"
    }
    form9 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "continuous_cols": [[0]]
    }
    form10 = {
        "cols": [["col1", "col2", "col3"]],
        "output": ["col3"],
        "model_id": "DMFuzzy",
        "normalize_bool": 1,
        "continuous_cols": [[0]]
    }

    bad_forms = [form1, form2, form3, form4, form5, form6, form7, form8, form9, form10]
    bad_form_messages = [
        "Field `cols` is required",
        "Field `cols` must be of type `list`",
        "Field `output` is required",
        "Field `output` must be of type `list`",
        "Field `model_id` is required",
        "Field `model_id` must be of type `str`",
        "Field `continuous_cols` is required",
        "Field `continuous_cols` must be of type `list`",
        "Field `normalize_bool` is required",
        "Field `normalize_bool` must be of type `str`"
    ]

    bad_headers = [headers1, headers2]
    bad_header_messages = [
        "`headers` does not contain `token`",
        "Field `token` must be of type `str`"
    ]

    return {
        "base_form": base_form,
        "bad_forms": bad_forms,
        "base_headers": base_headers,
        "bad_headers": bad_headers,
        "bad_form_messages": bad_form_messages,
        "bad_header_messages": bad_header_messages
    }
