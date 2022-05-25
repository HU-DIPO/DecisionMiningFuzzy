"""Tests for decisionModel.py."""
import pytest
import numpy as np

from decision_mining.core.dmn.models import decisionModel


def test_decisionModel() -> None:
    """Tests decisionModel for CICD testing."""
    with pytest.raises(TypeError):
        decisionModel.DecisionModel()


@pytest.fixture
def fakeDecisionModel() -> decisionModel.DecisionModel:
    """Fixture for fake decision model.

    Returns:
        decisionModel.DecisionModel: fake decision model
    """

    class FakeClassifier:
        name = "FakeClassifier"

        def __init__(self) -> None:
            super().__init__()

        def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.array([1])

    class FakeDecisionModel(decisionModel.DecisionModel):
        """Fake decision model for testing."""

        def __init__(self) -> None:
            self.model_info = {
                "name": "FakeDecisionModel",
                "id": "fakeDecisionModel",
                "description": "Fake decision model for testing.",
                "parameters":
                {"param1": {"type": "int", "value": 1, "description": "param1 description."},
                 "param2": {"type": "int", "value": 2, "description": "param1 description."}}, }

        def make_model(self, X: int, y: int, continuous_cols: list) -> dict:
            """Fake make_model function for testing."""
            return FakeClassifier()

        def extract_rules(self, cols: list, model: FakeClassifier) -> dict:
            """Fake make_model function for testing."""
            return {"cols": cols, "model": model}

    return FakeDecisionModel()


def test_model_name(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the get_model_name function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    assert fakeDecisionModel.model_name == "FakeDecisionModel", \
        "get_model_name() should return the name of the model."


def test_get_model_id(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the get_model_id function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    assert fakeDecisionModel.model_id == "fakeDecisionModel", \
        "get_model_id() should return the id of the model."


def test_get_model_description(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the get_model_description function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    assert fakeDecisionModel.model_description == "Fake decision model for testing.", \
        "get_model_description() should return the description of the model."


def test_get_model_parameters(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the get_model_parameters function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    parameters = {"param1": {"type": "int", "value": 1, "description": "param1 description."},
                  "param2": {"type": "int", "value": 2, "description": "param1 description."}}
    assert fakeDecisionModel.model_parameters == parameters, \
        "get_model_parameters() should return the parameters of the model."


def test_set_model_parameter(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the set_model_parameter function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    fakeDecisionModel.set_model_parameter("param1", 2)
    paremeters = fakeDecisionModel.model_info["parameters"]
    assert paremeters["param1"]['value'] == 2, \
        f"param1 should be set to 2. Actual value: {paremeters['param1']['value']}"


def test_set_model_parameters(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the set_model_parameters function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    parameters = {"param1": 5, "param2": 10}
    fakeDecisionModel.set_model_parameters(parameters)
    current_parameters = fakeDecisionModel.model_info["parameters"]
    assert current_parameters["param1"]['value'] == 5, \
        f"param1 should be set to 5. Actual value: {current_parameters['param1']['value']}"
    assert current_parameters["param2"]['value'] == 10, \
        f"param2 should be set to 10. Actual value: {current_parameters['param2']['value']}"


def test_make_models(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the make_models function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    parsed_data = [(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]), ["col1", "col2"])]
    continuous_cols = [[]]

    fakeDecisionModel.make_models(parsed_data, continuous_cols)

    value = fakeDecisionModel.models[0].name
    assert value == "FakeClassifier", f"{value}, should be FakeClassifier"


def test_extract_rules_for_all_models(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the extract_rules_for_all_models function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    with pytest.raises(ValueError):
        cols = [["col1", "col2"]]
        value = fakeDecisionModel.extract_rules_for_all_models(cols)

    cols = [["col1", "col2"]]
    fakeDecisionModel.models = [fakeDecisionModel.make_model(1, 2, [])]
    value = fakeDecisionModel.extract_rules_for_all_models(cols)
    assert value[0]['cols'] == cols[0], f"{value[0]['cols']}, should be {cols[0]}"
    assert value[0]['model'].name == "FakeClassifier",  \
        f"{value[0]['model'].name}, should be FakeClassifier"


def test_score_models(fakeDecisionModel: decisionModel.DecisionModel) -> None:
    """Tests the score_models function.

    Args:
        fakeDecisionModel (decisionModel.DecisionModel): fake decision model
    """
    parsed_data = [(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]), ["col1", "col2"])]

    fakeDecisionModel.models = [fakeDecisionModel.make_model(1, 2, [])]
    value = fakeDecisionModel.score_models(parsed_data)
    assert value == [[1]], f"{value}, should be [[1]]"
