"""Tests for regester_models.py."""
import pytest

from decision_mining import regester_models as r_models
from decision_mining.core.dmn.models.DMFuzzy import DMFuzzy


@pytest.fixture
def base_regestry() -> r_models.DecisionModelRegestry:
    """Fixture for DecisionModelRegestry."""
    regestry = r_models.DecisionModelRegestry()
    regestry.register(DMFuzzy())
    return regestry


def test_get_model(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_model function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    with pytest.raises(ValueError):
        base_regestry.get_model("Test")

    assert isinstance(DMFuzzy(), type(base_regestry.get_model("DMFuzzy"))), \
        "base_regestry.get_model('DMFuzzy') is not an instance of DMFuzzy"


def test_get_all_models(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    assert isinstance(DMFuzzy(), type(base_regestry.get_all_models()[0])), \
        "base_regestry.get_all_models()[0] is not an instance of DMFuzzy"


def test_get_all_models_parameters(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models_parameters function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    parameters = {
        "minimal_gain_ratio": {
            "value": 0.3,
            "min": 0.0,
            "max": 0.9,
            "step": 0.05,
            "type": "float",
            "description": (
                    "The minimal gain ratio that must be reached to be considered a new rule."
            )
        }
    }
    assert base_regestry.get_all_models_parameters()["DMFuzzy"] == parameters, \
        f"base_regestry.get_all_models_parameters()['DMFuzzy'] is not {parameters}"


def test_get_all_models_info(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models_info function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    parameters = {
        "minimal_gain_ratio": {
            "value": 0.3,
            "min": 0.0,
            "max": 0.9,
            "step": 0.05,
            "type": "float",
            "description": (
                    "The minimal gain ratio that must be reached to be considered a new rule."
            )
        }
    }
    model_info = {
        "name": "DMFuzzy",
        "id": "DMFuzzy",
        "description": (
                "DMFuzzy is a classification model based on the model of the Discovery of Fuzzy "
                "DMN Decision Models from "
                "Event Logs(Bazhenova, Haarmann, Ihde, Solti, & Weske, 2017). "
                "It discretises the continuous values, resulting in fuzzy terms like "
                "'low', 'medium', and 'high'"
        ),
        "parameters": parameters
    }
    assert base_regestry.get_all_models_info()["DMFuzzy"] == model_info, \
        f"base_regestry.get_all_models_info()['DMFuzzy'] is not equal to {model_info}"
