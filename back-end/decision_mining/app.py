"""Flask application.

This module contains the Flask main application.
It initializes a Flask RESTful REST API that can be run as backend system.
"""
from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from decision_mining.routes import init_routes

app = Flask(__name__)
cors = CORS(app)
api = Api(app)

init_routes(api)
