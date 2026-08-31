"""find_rules.py.

This module contains the endpoint for finding rules in the dataset(s).
"""
import copy
import json
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import pandas as pd
from flask import request
from flask_restful import Resource

from decision_mining.api.tools import pipeline as pp
from decision_mining.regester_models import registered_models
from decision_mining.core.dmn.rule import Rule


def _paper_unique_rules(rules: List[Rule], wildcard: str = "-") -> List[Rule]:
    """Create paper-style unique rules without aggressive wildcard expansion.

    For each rule, when a prerequisite is False, all trailing conditions are
    shown as wildcard (`-`). After that, duplicate rules are removed while
    preserving first-seen order.
    """
    unique_rules: List[Rule] = []
    seen = set()

    for rule in rules:
        values = list(rule.cols.values())
        keys = list(rule.cols.keys())
        for idx, value in enumerate(values):
            if str(value).lower() == "false":
                for trailing_key in keys[idx + 1:]:
                    rule.cols[trailing_key] = wildcard
                break

        identity = tuple(list(rule.cols.values()) + [rule.decision])
        if identity in seen:
            continue
        seen.add(identity)
        unique_rules.append(rule)

    return unique_rules


class FindRules(Resource):
    """FindRules Resource.

    Represents a Resource that can be used to train models and discover rules.
    Requires one or more CSV files and column data to execute.
    """

    def post(self) -> Tuple[Dict, int]:
        """POST FindRules.

        URL:
            /rules

        Method:
            POST

        URL Params:
            None

        Headers:
            token (str): User token for API verification

        Data Params:
            *  cols (List[List[str]]): n lists of column names, including output columns.
            *  output (List[str]): n columns names for output columns.
            *  model_id (str): model_id.
            *  normalize_bool(str): normalize_bool.
            *  continuous_cols (List[List[int]]):  n lists of continuous column indices.

        Files:
            n Files ending with .csv


        Returns:
            Tuple[Dict, int]: Success or error message.
        """
        # TODO: token validation

        if len(request.files) == 0:
            return {"message": "No files uploaded"}, 400

        form = request.form.to_dict()
        if "json" not in form:
            return {"message": "Form data must contain a field `json`, with JSON string"}, 400

        form = json.loads(form.get("json"))
        validation = pp.validate_input(request.headers, form)
        if validation["status"] == 400:
            return {"message": validation["message"]}, 400

        data = []
        files = request.files.getlist("file")
        for data_file in files:
            if not data_file.filename.endswith(".csv"):
                return {"message": "File not in CSV format"}, 400

            df = pd.read_csv(data_file)
            data.append(df)

        output = form.get("output")
        cols = form.get("cols")
        model_id = form.get("model_id")
        normalize_bool = bool(form.get("normalize_bool") == "true")
        continuous_cols = form.get("continuous_cols")
        parsed_data = list(map(pp.parse_data, data, cols, output))

        decision_model = registered_models.get_model(model_id)

        decision_model.make_models(parsed_data, continuous_cols)
        raw_rules = decision_model.extract_rules_for_all_models(cols)
        scores = decision_model.score_models(parsed_data)

        paper_rules = copy.deepcopy(raw_rules)
        for idx, rules_per_model in enumerate(paper_rules):
            paper_rules[idx] = _paper_unique_rules(rules_per_model)

        if normalize_bool:
            selected_rules = paper_rules
        else:
            selected_rules = raw_rules

        raw_dmn_tree = pp.generate_dmn([decision[-1] for decision in parsed_data], raw_rules)
        paper_dmn_tree = pp.generate_dmn([decision[-1] for decision in parsed_data], paper_rules)
        selected_dmn_tree = pp.generate_dmn([decision[-1] for decision in parsed_data], selected_rules)

        xml_raw = ElementTree.tostring(raw_dmn_tree.getroot(), encoding="unicode", method="xml")
        xml_paper = ElementTree.tostring(paper_dmn_tree.getroot(), encoding="unicode", method="xml")
        xml_selected = ElementTree.tostring(
            selected_dmn_tree.getroot(), encoding="unicode", method="xml")

        return {
            "message": "Success",
            "xml": xml_selected,
            "xml_raw": xml_raw,
            "xml_paper": xml_paper,
            "rules_view": "paper" if normalize_bool else "raw",
            "accuracy": scores,
        }, 200
