"""Tests the find_rules resource."""
import json
from pathlib import Path
from xml.etree import ElementTree
from typing import Dict, List, Union

import flask
import numpy as np
import pytest
from flask.testing import FlaskClient

URL = "/rules"


@pytest.fixture()
def gen_csv(tmp_path: Path) -> Path:
    """Generate a .csv file and returns path.

    Args:
        tmp_path (Path): Pytest fixture.

    Returns:
        Path: Path to test.csv.
    """
    d = tmp_path / "data"
    if not d.exists():
        d.mkdir()
    file = d / "test.csv"

    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    with file.open("w") as f:
        f.write("col1,col2,col3\n")
        for x, _y in zip(X, y):
            f.write(f"{x[0]},{x[1]},{_y}\n")

    yield file


@pytest.fixture()
def gen_txt(tmp_path: Path) -> Path:
    """Generate a .txt file and returns path.

    Args:
        tmp_path (Path): Pytest fixture.

    Returns:
        Path: Path to test.txt.
    """
    d = tmp_path / "data"
    if not d.exists():
        d.mkdir()
    file = d / "test.txt"

    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    with file.open("w") as f:
        f.write("col2,col3,col5\n")
        for x, _y in zip(X, y):
            f.write(f"{x[0]},{x[1]},{_y}\n")

    yield file


@pytest.fixture()
def gen_request(gen_csv: Path) -> dict:
    """Create a generic request that will pass the resource post.

    Args:
        gen_csv (Path): Path to csv file.

    Returns:
        dict: data with json and file, headers with token.
    """
    cols = [["col1", "col2", "col3"]]
    cont_cols = [[0]]
    output = ["col3"]

    data = {
        "cols": cols,
        "output": output,
        "continuous_cols": cont_cols,
        "model_id": "DMFuzzy",
        "normalize_bool": "false"
    }

    headers = {
        "token": "email@email.com"
    }

    data = {"json": json.dumps(data)}
    data["file"] = (gen_csv.open("rb"), "test.csv")

    return {"data": data, "headers": headers}


@pytest.fixture()
def two_file_request(gen_csv: Path, gen_txt: Path) -> dict:
    """Create a generic request that will pass the resource post.

    Args:
        gen_csv (Path): Path to csv file.
        gen_txt (Path): Path fixture with file.

    Returns:
        dict: data with json and file, headers with token.
    """
    cols = [["col1", "col2", "col3"], ["col2", "col3", "col5"]]
    cont_cols = [[0], [0]]
    output = ["col3", "col5"]

    data = {
        "cols": cols,
        "output": output,
        "continuous_cols": cont_cols,
        "model_id": "DMFuzzy",
        "normalize_bool": "true"
    }

    headers = {
        "token": "email@email.com"
    }

    data = {"json": json.dumps(data)}
    data["file"] = [(gen_csv.open("rb"), "test1.csv"), (gen_txt.open("rb"), "test2.csv")]

    return {"data": data, "headers": headers}


def test_no_files(client: FlaskClient) -> None:
    """Tests the for existence of file(s).

    Args:
        client (FlaskClient): Flask testing client.
        gen_csv (Path): Path fixture with file.
        gen_txt (Path): Path fixture with file.
        gen_request (dict): Generic request fixture.
    """
    # No files passed
    data = {"nothing": "nothing"}
    res: flask.Response = client.post(URL, data=data)
    assert 400 == res.status_code, f"response status should be 400, got {res.status_code}"
    assert "No files uploaded" == res.get_json()["message"],\
        f"message should be `No files uploaded`, not {res.get_json()['message']}"


def test_one_file(client: FlaskClient, gen_csv: Path) -> None:
    """Tests for the existence of one file.

    Args:
        client (FlaskClient): Flask testing client.
        gen_csv (Path): Path fixture with file.
    """
    data = {"nothing": "nothing"}
    # One file passed, breaks on no json
    data["file"] = (gen_csv.open("rb"), "test.csv")
    res: flask.Response = client.post(URL, data=data, content_type='multipart/form-data')
    assert 400 == res.status_code, f"response status should be 400, got {res.status_code}"
    assert "Form data must contain a field `json`, with JSON string" == res.get_json()["message"],\
        f"Form data must contain a field `json`, with JSON string`,\
             not {res.get_json()['message']}"


def test_no_csv(client: FlaskClient, gen_txt: Path, gen_request: dict) -> None:
    """Tests for the existence of files other than .csv.

    Args:
        client (FlaskClient): Flask testing client.
        gen_txt (Path): Path fixture with file.
        gen_request (dict): Generic request fixture.
    """
    # Test a non-.csv file
    gen_request["data"]["file"] = (gen_txt.open("rb"), "test.txt")
    res: flask.Response = client.post(URL, **gen_request, content_type='multipart/form-data')
    assert 400 == res.status_code, f"response status should be 400, got {res.status_code}"
    assert "File not in CSV format" == res.get_json()["message"],\
        f"message should be `File not in CSV format`, not {res.get_json()['message']}"


def test_multiple_files(client: FlaskClient, two_file_request: dict) -> None:
    """Tests using multiple files.

    Args:
        client (FlaskClient): Flask test client fixture.
        two_file_request (dict): Request with two files.
    """
    res: flask.Response = client.post(URL, **two_file_request, content_type='multipart/form-data')
    assert 200 == res.status_code, f"response status should be 200, got {res.status_code}"
    assert [0.81, 0.81] == res.get_json()["accuracy"]


def test_arguments(client: FlaskClient, form_header_data: Dict[str, Union[List, Dict]],
                   gen_csv: Path) -> None:
    """Tests the argument verification for `find_rules`.

    Args:
        client (FlaskClient): Flask test_client
        form_header_data (Dict[str, Union[List, Dict]]): Possibilities for formdata/headerdata.
        gen_csv (Path): Path fixture with file.
    """
    base_form = form_header_data.get("base_form")
    bad_forms = form_header_data.get("bad_forms")
    base_headers = form_header_data.get("base_headers")
    bad_headers = form_header_data.get("bad_headers")
    bad_form_messages = form_header_data.get("bad_form_messages")
    bad_header_messages = form_header_data.get("bad_header_messages")

    # TODO: Token validation
    # Currently, we're skipping the last bad header
    # This is because we haven't implemented token validation
    # For those implementing token validation:
    # Please note that you're working with werkzeug's datastructures, not regular dicts
    for bad_header, bad_header_message in zip(bad_headers[:-1], bad_header_messages[:-1]):
        base_form_data = {
            "json": json.dumps(base_form),
            "file": (gen_csv.open("rb"), "test.csv")
        }
        result = client.post(URL, data=base_form_data, headers=bad_header)
        assert result.status_code == 400,\
            f"Response status should be 400, got {result.status_code}"

        result_json = result.get_json()
        assert result_json.get("message") == bad_header_message,\
            f"Expected error '{bad_header_message}', got '{result_json.get('message')}'"

    for bad_form, bad_form_message in zip(bad_forms, bad_form_messages):
        bad_form_data = {
            "json": json.dumps(bad_form),
            "file": (gen_csv.open("rb"), "test.csv")
        }
        result = client.post(URL, data=bad_form_data, headers=base_headers)
        assert result.status_code == 400,\
            f"Response status should be 400, got {result.status_code}"

        result_json = result.get_json()
        assert result_json.get("message") == bad_form_message,\
            f"Expected error '{bad_form_message}', got '{result_json.get('message')}'"

    base_form_data = {
        "json": json.dumps(base_form),
        "file": (gen_csv.open("rb"), "test.csv")
    }
    result = client.post(URL, data=base_form_data, headers=base_headers)
    assert result.status_code == 200,\
        f"Response status should be 200, got {result.status_code}"

    result_json = result.get_json()
    assert result_json.get("message") == "Success",\
        f"Expected error messages to be 'Success', got '{result_json.get('message')}'"


def test_json(client: FlaskClient, gen_csv: Path) -> None:
    """Tests the existence of the json field.

    Args:
        client (FlaskClient): Flask test_client.
        gen_csv (Path): Path to csv, fixture.
    """
    # Tests when `json` field exists
    data = dict()
    data["file"] = (gen_csv.open("rb"), "test.csv")
    json_dict = dict()
    json_dict["output"] = [0]
    data["json"] = json.dumps(json_dict)
    res: flask.Response = client.post(URL, data=data, content_type='multipart/form-data')
    assert 400 == res.status_code, f"response status should be 400, got {res.status_code}"
    assert "`headers` does not contain `token`" in res.get_json()["message"],\
        f"message should be `No files uploaded`, not {res.get_json()['message']}"

    # Test when `json` field does not exist.
    data = {"nothing": "nothing"}
    data["file"] = (gen_csv.open("rb"), "test.csv")
    res: flask.Response = client.post(URL, data=data, content_type='multipart/form-data')
    assert 400 == res.status_code, f"response status should be 400, got {res.status_code}"
    assert "Form data must contain a field `json`, with JSON string" == res.get_json()["message"],\
        f"Form data must contain a field `json`, with JSON string`,\
             not {res.get_json()['message']}"


def test_full(client: FlaskClient, gen_request: dict) -> None:
    """Tests the endpoint with correct input.

    Args:
        client (FlaskClient): Flask test client.
        gen_request (dict): Request arguments fixture.
    """
    res: flask.Response = client.post(URL, **gen_request, content_type='multipart/form-data')
    assert 200 == res.status_code, f"response status should be 200, got {res.status_code}"
    res_dict = res.get_json()
    assert abs(0.81 - res_dict["accuracy"][0]) < 1e2,\
        f"accuracy should be 0.81, not {res_dict['accuracy'][0]}"


def test_xml(client: FlaskClient, two_file_request: dict) -> None:
    """Testing the XML output for a multi-file post request.

    Args:
        client (FlaskClient): Flask test client.
        two_file_request (dict): Multi-file request arguments fixture.
    """
    res: flask.Response = client.post(URL, **two_file_request, content_type='multipart/form-data')
    assert 200 == res.status_code, f"response status should be 200, got {res.status_code}"
    res_dict = res.get_json()

    xml = res_dict["xml"]
    root = ElementTree.fromstring(xml)

    xml_namespace = "https://www.omg.org/spec/DMN/20191111/MODEL/"

    assert root[0].tag == f"{{{xml_namespace}}}decision", \
        f"Expected root[0].tag to be {{{xml_namespace}}}decision, got {root[0].tag}"
    assert root[1].tag == f"{{{xml_namespace}}}decision", \
        f"Expected root[1].tag to be {{{xml_namespace}}}decision, got {root[1].tag}"
    assert root[2].tag == f"{{{xml_namespace}}}inputData", \
        f"Expected root[2].tag to be {{{xml_namespace}}}inputData, got {root[2].tag}"

    namespaces = {"omg": xml_namespace,
                  "dmndi": "https://www.omg.org/spec/DMN/20191111/DMNDI/",
                  "dc": "http://www.omg.org/spec/DMN/20180521/DC/",
                  "di": "http://www.omg.org/spec/DMN/20180521/DI/"}

    decisions = root.findall("omg:decision", namespaces=namespaces)
    assert len(decisions) == 2, f"Expected two decision nodes, not {len(decisions)}"

    for i, decision in enumerate(decisions):
        requirements = decision.findall("omg:informationRequirement", namespaces=namespaces)
        assert len(requirements) == 2, \
            f"Decision {i} has {len(requirements)} informationRequirement(s), expected 2"

        tables = decision.findall("omg:DecisionTable", namespaces=namespaces)
        assert len(tables) == 1, f"Decision {i} has {len(tables)} DecisionTable(s), expected 1"
        table = tables[0]

        table_inputs = table.findall("omg:input", namespaces=namespaces)
        assert len(table_inputs) == 2, f"Decision {i} has {len(table_inputs)} input(s), expected 2"

        table_outputs = table.findall("omg:output", namespaces=namespaces)
        assert len(table_outputs) == 1, \
            f"Decision {i} has {len(table_outputs)} output(s), expected 1"

        rules = table.findall("omg:rule", namespaces=namespaces)
        assert len(rules) == 1, f"Decision {i} has {len(rules)} rule(s), expected 1"
        for j, rule in enumerate(rules):
            input_entries = rule.findall("omg:inputEntry", namespaces=namespaces)
            assert len(input_entries) == 2, \
                f"Decision {i};{j} has {len(input_entries)} inputEntry(s), expected 2"

            output_entries = rule.findall("omg:outputEntry", namespaces=namespaces)
            assert len(output_entries) == 1, \
                f"Decision {i};{j} has {len(output_entries)} outputEntry(s), expected 1"

    inputs = root.findall("omg:inputData", namespaces=namespaces)
    assert len(inputs) == 2, f"Expected two input nodes, not {len(inputs)}"

    dmndis = root.findall("dmndi:DMNDI", namespaces=namespaces)
    assert len(dmndis) == 1, f"Expected one DMNDI, not {len(dmndis)}"
    dmndi = dmndis[0]

    assert len(dmndi) == 1, f"Expected one subelement in DMNDI, not {len(dmndi)}"
    diagram = dmndi[0]
    assert (id := diagram.get("id")) == "DMNDiagram1",\
        f"Item should have ID 'DMNDiagram1', not {id}"

    shapes = diagram.findall("dmndi:DMNShape", namespaces=namespaces)
    assert len(shapes) == 4, f"Expected diagram to have 4 shapes, not {len(shapes)}"

    edges = diagram.findall("dmndi:DMNEdge", namespaces=namespaces)
    assert len(edges) == 4, f"Expected diagram to have 4 edges, not {len(edges)}"
