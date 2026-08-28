"""Flask application.

This module contains the Flask main application.
It initializes a Flask RESTful REST API that can be run as backend system.
"""
import os

from flask import Flask
from flask_cors import CORS
from flask_restful import Api

from decision_mining.routes import init_routes

app = Flask(__name__)
default_origins = "http://localhost:4200,http://127.0.0.1:4200"
allowed_origins = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", default_origins).split(",") if origin.strip()]
cors = CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    allow_headers=["Content-Type", "Authorization", "token"],
    methods=["GET", "POST", "OPTIONS"],
)
api = Api(app)

init_routes(api)
