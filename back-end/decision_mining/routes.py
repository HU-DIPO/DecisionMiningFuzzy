"""routes.py.

This module contains the routes configuration
to all the REST API endpoints.
"""
import flask_restful
from decision_mining.api.find_rules import FindRules
from decision_mining.api.get_models import GetModels


def init_routes(api: flask_restful.Api) -> None:
    """Initializes every API endpoint into Flask RESTful.

    Args:
        api (flask_restful.Api): Flask RESTful API object.
    """
    api.add_resource(FindRules, "/rules")
    api.add_resource(GetModels, "/models")
