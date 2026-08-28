"""Tests the get_models resource."""
from flask.testing import FlaskClient

URL = "/models"


def test_get_models(client: FlaskClient) -> None:
    """Tests the get_models resource.

    Args:
        client (FlaskClient): Flask client
    """
    response = client.get(URL)
    assert response.status_code == 200, "Should return 200."
    assert response.headers["Content-Type"] == "application/json", "Should return json."
    print(response.json)

    assert len(response.json["models"]) == 1, \
        f"Should return 3 models, but returned {len(response.json['models'])}."
