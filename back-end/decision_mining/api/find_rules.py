"""find_rules.py.

This module contains the endpoint for finding rules in the dataset(s).
"""
import json
from typing import Dict, Tuple
from xml.etree import ElementTree

import pandas as pd
import numpy as np
from flask import request
from flask_restful import Resource

from decision_mining.api.tools import pipeline as pp
from decision_mining.regester_models import registered_models
from decision_mining.core.dmn.dt_normalization import normalize_dt


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
        rules = decision_model.extract_rules_for_all_models(cols)
        scores = decision_model.score_models(parsed_data)

        if normalize_bool:
            normalized_rules = []
            for model, rules in zip(decision_model.models, rules):
                continuous_cols = model.continuous_cols
                if model.categoricalize_continuous_values:
                    continuous_cols = np.array([])
                normalized_rules.append(normalize_dt(rules, continuous_cols))
            rules = normalized_rules

        dmn_tree = pp.generate_dmn([decision[-1] for decision in parsed_data], rules)
        xml_str = ElementTree.tostring(dmn_tree.getroot(), encoding="unicode", method="xml")
        return {"message": "Success", "xml": xml_str, "accuracy": scores}, 200
