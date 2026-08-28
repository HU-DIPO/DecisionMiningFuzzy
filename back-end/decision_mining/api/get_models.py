"""get_rules.py.

This module contains the endpoint for retrieving models information.
"""
from typing import Dict, Tuple

from flask import jsonify
from flask_restful import Resource

from decision_mining.regester_models import registered_models


class GetModels(Resource):
    """GetModels Resource.

    Represents a Resource that can be used to retrieve all available information about models.
    """

    def get(self) -> Tuple[Dict, int]:
        """GET GetModels.

        URL:
            /models

        Method:
            GET

        URL Params:
            None

        Data Params:
            None

        Returns:
            Tuple[Dict, int]: Success or error message.
        """
        if not registered_models.get_all_models():  # pragma: no cover
            return {"message": "No models registered"}, 400

        message = {
            "message": "Success",
            "models": registered_models.get_all_models_info()
        }
        resp = jsonify(message)
        resp.status_code = 200
        return resp
