"""Tests dmn_generation.py."""
from xml.etree import ElementTree

import decision_mining.core.dmn.dmn_generation as dmng
from decision_mining.core.dmn import rule
import pytest


@pytest.fixture
def dmnnode() -> dmng.DMNNode:
    """Pytest fixture for creating a DMNNode.

    Yields:
        Iterator[dmng.DMNNode]: Test node with name "DMNNode".
    """
    yield dmng.DMNNode("DMNNode")


@pytest.fixture
def inputnode() -> dmng.InputNode:
    """Pytest fixture for creating an InputNode.

    Yields:
        Iterator[dmng.InputNode]: InputNode with name "InputNode".
    """
    yield dmng.InputNode("InputNode")


@pytest.fixture
def decisionnode() -> dmng.DecisionNode:
    """Pytest fixture for creating a DecisionNode.

    Yields:
        Iterator[dmng.DecisionNode]: DecisionNode with name "DecisionNode".
    """
    yield dmng.DecisionNode("DecisionNode")


def test_DMNNNode_create_dmndi_shape(dmnnode: dmng.DMNNode) -> None:
    """Tests the dmng.DMNNode.create_dmndi_shape function.

    Args:
        dmnnode (dmng.DMNNode): Fixture for DMNNode
    """
    # Set arguments
    parent = ElementTree.Element("dmndi_shape")
    x, y = 11, 12
    referer = "test dmndi_shape"

    # Execute function
    dmnnode.create_dmndi_shape(parent, referer, (x, y))
    # Test position
    assert dmnnode.x == x, f"Expected {x=}, got {dmnnode.x} instead"
    assert dmnnode.y == y, f"Expected {y=}, got {dmnnode.y} instead"

    # Take generated ElemenTree.Element
    elem = dmnnode.element
    # Validate tags/ids
    assert elem.tag == "dmndi:DMNShape", f"Tag should be dmndi:DMNShape, not {elem.tag}"
    assert elem.get("id") == "DMNShape_" + \
        referer.replace(" ", "_"), f"Expected id to be DMNShape_{referer}, not {elem.get('id')}"
    assert elem.get("dmnElementRef") == referer.replace(" ", "_")
    # Validate amount of children
    assert len(elem) == 1, f"DMNDI shape element should have 1 child, not {len(elem)}"

    # Validate positioning
    bounds = elem[0]
    assert bounds.tag == "dc:Bounds", f"Tag should be dc:Bounds, not {bounds.tag}"
    assert bounds.get("width") == "150", f"Expected width=150, got {bounds.get('width')} instead"
    assert bounds.get("height") == "50", f"Expected height=50, got {bounds.get('height')} instead"
    assert bounds.get("x") == str(x), f"Expected {x=}, got {bounds.get('x')} instead"
    assert bounds.get("y") == str(y), f"Expected {y=}, got {bounds.get('y')} instead"

    # Validate positioning with values for width, height
    width, height = 20, 30
    dmnnode.create_dmndi_shape(parent, referer, (x, y), width, height)
    elem = dmnnode.element
    bounds = elem[0]
    assert bounds.get("width") == str(
        width), f"Expected {width=}, got {bounds.get('width')} instead"
    assert bounds.get("height") == str(
        height), f"Expected {height=}, got {bounds.get('height')} instead"


def test_InputNode_set_xml_node(inputnode: dmng.InputNode) -> None:
    """Tests the dmn_generation.InputNode.set_xml_node function.

    Args:
        inputnode (dmng.InputNode): Fixture for InputNode.
    """
    # Set arguments
    parent = ElementTree.Element("root_input_test_element")

    inputnode.set_xml_node(parent)

    element = inputnode.element
    assert element.tag == "inputData", f"Element Tag should be 'inputData', not {element.tag}"
    assert element.get("id") == inputnode.name, f"Expected element id to be {inputnode.name}, \
        not {element.get('id')}"
    assert element.get("name") == inputnode.name, f"Expected name to be {inputnode.name}, \
        not {element.get('name')}"


def test_DecisionNode_add_dependencies(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode.add_dependencies function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    dependencies = [
        dmng.InputNode("TestInput1"),
        dmng.InputNode("TestInput2"),
        dmng.InputNode("TestInput3")
    ]
    decisionnode.add_dependencies(dependencies)

    assert len(decisionnode.dependencies) == len(dependencies), f"Expected no. \
        dependencies to be {len(dependencies)}, not {len(decisionnode.dependencies)}"


def test_DecisionNode_add_rules(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode.add_rules function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    r = rule.Rule([0, 1, 2])
    r.cols[0] = "Test"
    r.cols[1] = [{"threshold": 51.5, "<": False},
                 {"threshold": 79.5, "<": True}]
    r.decision = "True"

    rules = [r]
    decisionnode.add_rules(rules)

    assert len(decisionnode.rules) == 1, \
        f"Expected amount of rules to be 1, not {len(decisionnode.rules)}"
    assert isinstance(decisionnode.rules[0], rule.Rule), \
        f"Expected type to be 'Rule', not {type(decisionnode.rules[0])}"


def test_DecisionNode_set_xml_node(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode.set_xml_node function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    parent = ElementTree.Element("root_decision_test_element")
    decisionnode.dependencies = [dmng.InputNode("TestInput")]
    decisionnode.set_xml_node(parent)

    element = decisionnode.element
    assert element.tag == "decision", f"Element Tag should be 'decision', not {element.tag}"
    assert element.get(
        "id") == decisionnode.name, f"Expected element id to be {decisionnode.name}, \
        not {element.get('id')}"
    assert element.get("name") == decisionnode.name, f"Expected name to be {decisionnode.name}, \
        not {element.get('name')}"

    assert len(element) == 2, f"DecisionNode element should have 2 children, got {len(element)}"

    dependency_elem = element[0]
    assert dependency_elem.tag == "informationRequirement", f"Expected Dependency element tag \
        to be 'informationRequirement', not {dependency_elem.tag}"

    dependency_name = decisionnode.dependencies[0].name
    decision_name = decisionnode.name
    assert dependency_elem.get("id") == "info_req_" + dependency_name + "_" + decision_name, \
        f"Expected Dependency id to be info_req_{dependency_name}_{decision_name}, \
        not {dependency_elem.get('id')}"

    assert len(dependency_elem) == 1, f"Dependency should have 1 child: 'requirement', \
        got {len(dependency_elem)}"

    requirement = dependency_elem[0]
    assert requirement.get("href") == "#" + dependency_name, f"Expected requirement href to be \
        #{dependency_name}, not {requirement.get('href')}"


def test_DecisionNode_create_waypoints(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode._create_waypoints function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    # set attributes
    decisionnode.width = 150
    decisionnode.height = 50
    decisionnode.x = 200
    decisionnode.y = 200

    # Without anchor (2 waypoints)
    dependency1 = dmng.InputNode("TestInput1")
    dependency1.width = 150
    dependency1.height = 150
    dependency1.x = 200
    dependency1.y = 400

    # With anchor (3 waypoints)
    dependency2 = dmng.InputNode("TestInput2")
    dependency2.width = 150
    dependency2.height = 50
    dependency2.x = 400
    dependency2.y = 400

    dependency1_waypoints = decisionnode._create_waypoints(dependency1)
    dependency2_waypoints = decisionnode._create_waypoints(dependency2)

    assert len(dependency1_waypoints) == 2, \
        f"Expected 2 waypoints, not {len(dependency1_waypoints)}"
    assert dependency1_waypoints[0] == (275, 400), \
        f"Expected start (x,y) coordinates: (275, 400), got {dependency1_waypoints[0]}"
    assert dependency1_waypoints[1] == (275, 250), \
        f"Expected end (x,y) coordinates: (275, 250), got {dependency1_waypoints[1]}"

    assert len(dependency2_waypoints) == 3, \
        f"Expected 3 waypoints, not {len(dependency2_waypoints)}"
    assert dependency2_waypoints[0] == (475, 400), \
        f"Expected start (x,y) coordinates: (475, 400), got {dependency2_waypoints[0]}"
    assert dependency2_waypoints[1] == (275, 270), \
        f"Expected anchor (x,y) coordinates: (275, 270), got {dependency2_waypoints[1]}"
    assert dependency2_waypoints[2] == (275, 250), \
        f"Expected end (x,y) coordinates: (275, 250), got {dependency2_waypoints[2]}"


def test_DecisionNode_create_information_requirement(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode.create_information_requirement function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    decisionnode.width = 150
    decisionnode.height = 50
    decisionnode.x = 200
    decisionnode.y = 200

    parent = ElementTree.Element("root_test_element")
    referer_obj = dmng.InputNode("TestInput")

    referer_obj.width = 150
    referer_obj.height = 50
    referer_obj.x = 200
    referer_obj.y = 400

    decisionnode.create_information_requirement(parent, referer_obj)
    elem = decisionnode.element
    assert elem.tag == "dmndi:DMNEdge", f"Expected tag to be 'dmndi:DMNEdge', not {elem.tag}"
    assert elem.get("id") == "DMNEdge_" + referer_obj.name + "_" + decisionnode.name, \
        f"Expected element id to be 'DMNEdge_{referer_obj.name}_{decisionnode.name}', \
            not {elem.get('id')}"

    assert elem.get("dmnElementRef") == "info_req_" + referer_obj.name + "_" + decisionnode.name, \
        f"Expected dmnElementRef to be 'info_req_{referer_obj.name}_{decisionnode.name}', \
            not {elem.get('dmnElementRef')}"

    assert len(elem) == 2, f"Expected 2 waypoints, got {len(elem)}"
    start_waypoint = elem[0]
    end_waypoint = elem[1]

    assert start_waypoint.tag == "di:waypoint", \
        f"Expected waypoint tag to be 'di:waypoint', not {start_waypoint.tag}"
    assert start_waypoint.get("x") == "275", \
        f"start_waypoint x should be '275', got {start_waypoint.get('x')}"
    assert start_waypoint.get("y") == "400", \
        f"start_waypoint y should be '400', got {start_waypoint.get('y')}"

    assert end_waypoint.tag == "di:waypoint", \
        f"Expected waypoint tag to be 'di:waypoint', not {end_waypoint.tag}"
    assert end_waypoint.get("x") == "275", \
        f"end_waypoint x should be '275', got {end_waypoint.get('x')}"
    assert end_waypoint.get("y") == "250", \
        f"end_waypoint y should be '250', got {end_waypoint.get('y')}"


def test_DecisionNode_create_decision_table(decisionnode: dmng.DecisionNode) -> None:
    """Tests the dmn_generation.DecisionNode._create_decision_table function.

    Args:
        decisionnode (dmng.DecisionNode): Fixture for DecisionNode.
    """
    # Get required data
    r = rule.Rule([0, 1, 2])
    r.cols[0] = "Test"
    r.cols[1] = [{"threshold": 51.5, "<": False},
                 {"threshold": 79.5, "<": True}]
    r.decision = "True"
    decisionnode.rules = [r]
    decisionnode.dependencies = [dmng.InputNode("TestInput1"), dmng.InputNode("TestInput2")]
    decisionnode.element = ElementTree.Element("decision")
    # Execute function
    decisionnode._create_decision_table()

    # Checks
    assert len(decisionnode.element) == 1, \
        f"Function should create 1 child: 'DecisionTable', got {len(decisionnode.element)}"
    elem = decisionnode.element[0]

    assert elem.tag == "DecisionTable", \
        f"Element tag should be 'DecisionTable', not {elem.tag}"
    assert elem.get("id") == f"DecisionTable_{decisionnode.name}", \
        f"Expected element id to be 'DecisionTable_{decisionnode.name}', got '{elem.get('id')}'"
    assert elem.get("hitPolicy") == "UNIQUE", \
        f"Expected hitpolicy to be 'UNIQUE', got '{elem.get('hitPolicy')}'"

    assert len(elem) == len(decisionnode.dependencies) + len(decisionnode.rules) + 1, \
        f"Expected {len(decisionnode.dependencies) + len(decisionnode.rules) + 1} \
            (dependencies+rules+output), got {len(elem)}"

    # input nodes
    for i in range(len(decisionnode.dependencies)):
        assert elem[i].tag == "input", f"Element tag should be 'input', not '{elem[i].tag}'"
        assert elem[i].get("id") == f"DecisionTable{decisionnode.name}Input{i}", \
            f"Expected element id to be 'DecisionTable{decisionnode.name}Input{i}', \
                not '{elem[i].get('id')}'"

        assert elem[i].get("label") == decisionnode.dependencies[i].name, \
            f"Expected label to be '{decisionnode.dependencies[i].name}', \
                not '{elem[i].get('label')}'"

        assert len(elem[i]) == 1, \
            f"'input' should have 1 child: 'inputExpression', got {len(elem[i])}"

        input_expression = elem[i][0]
        assert input_expression.tag == "inputExpression", \
            f"Expected child element tag 'inputExpression', not {input_expression.tag}"
        assert input_expression.get("id").startswith(
            f"{decisionnode.dependencies[i].name}InputExpression-")
        assert input_expression.get("typeRef") == "[type]", \
            f"Expected typeRef of '[type]', got {input_expression.get('typeRef')}"

        assert len(input_expression) == 1, \
            f"input_expression should have 1 child: 'text', got {len(input_expression)}"

        ie_text = input_expression[0]
        assert ie_text.tag == "text", \
            f"input_expression text element tag should be 'tag', not {ie_text.tag}"
        assert ie_text.text == decisionnode.dependencies[i].name, \
            f"Expected element content to be '{decisionnode.dependencies[i].name}', \
                not '{ie_text.text}'"

    # Output node
    output_node = elem[len(decisionnode.dependencies)]
    assert output_node.tag == "output", \
        f"Output element tag should be 'output', not {output_node.tag}"
    assert output_node.get("id") == f"OutputClause{decisionnode.name}", \
        f"Output element id should be OutputClause{decisionnode.name}, \
            not {output_node.get('id')}"

    assert output_node.get("label") == decisionnode.name, \
        f"Ouptut element label should be {decisionnode.name}, not {output_node.get('label')}"
    assert output_node.get("name") == decisionnode.name, \
        f"Ouput element name should be {decisionnode.name}, not {output_node.get('name')}"
    assert output_node.get('typeRef') == "[type]", \
        f"Output typeRef should be '[type]', not {output_node.get('typeRef')}"


def test_create_node_objects() -> None:
    """Tests the dmn_generation.create_node_objects function."""
    column_names = [
        ["duration", "premium", "amount", "risk"],
        ["premium", "amount", "risk", "approval"],
        ["duration", "premium", "amount", "check"],
    ]
    node_objects = dmng.create_node_objects(column_names)
    assert len(node_objects) == 6, f"Amount of unique columns should be 6, not {len(node_objects)}"

    # Check every object type
    assert isinstance(node_objects["risk"], dmng.DecisionNode), \
        f"Expected 'risk' to be of type 'dmng.DecisionNode', not {type(node_objects['risk'])}"
    assert isinstance(node_objects["approval"], dmng.DecisionNode), \
        f"Expected 'approval' to be of type 'dmng.DecisionNode', \
            not {type(node_objects['approval'])}"
    assert isinstance(node_objects["check"], dmng.DecisionNode), \
        f"Expected 'check'to be of type 'dmng.DecisionNode', not {type(node_objects['check'])}"
    assert isinstance(node_objects["duration"], dmng.InputNode), \
        f"Expected 'duration' to be of type 'dmng.InputNode', not {type(node_objects['duration'])}"
    assert isinstance(node_objects['premium'], dmng.InputNode), \
        f"Expected 'premium' to be of type 'dmng.InputNode', not {type(node_objects['premium'])}"
    assert isinstance(node_objects["amount"], dmng.InputNode), \
        f"Expected 'amount to be of type 'dmng.InputNode', not {type(node_objects['amount'])}"


def test_create_dependencies() -> None:
    """Tests the dmn_generation.create_dependencies function."""
    column_names = [
        ["duration", "premium", "amount", "risk"],
        ["premium", "amount", "risk", "approval"],
        ["duration", "premium", "amount", "check"],
    ]

    node_objects = dmng.create_node_objects(column_names)
    decision_nodes = dmng.create_dependencies(column_names, node_objects)

    assert len(decision_nodes) == 3, f"Expected 3 decisions, got {len(decision_nodes)}"

    assert all(len(decision_node.dependencies) == 3 for decision_node in decision_nodes), \
        "Every decision should have 3 dependencies"


def test_create_xml() -> None:
    """Tests the dmn_generation.create_xml function."""
    column_names = [
        ["duration", "premium", "amount", "risk"],
        ["premium", "amount", "risk", "approval"],
        ["duration", "premium", "amount", "check"],
    ]

    node_objects = dmng.create_node_objects(column_names)
    decision_nodes = dmng.create_dependencies(column_names, node_objects)

    xml_tree = dmng.create_xml(node_objects, decision_nodes)
    root = xml_tree.getroot()
    assert len(root) == 7, f"Expected 7 children, got {len(root)}"

    # Check decision elements
    for i in range(0, 3):
        assert root[i].tag == "decision", \
            f"Expected first 3 element tags to be 'decision', not '{root[i].tag}'"

    # Check input elements
    for i in range(3, 6):
        assert root[i].tag == "inputData", \
            f"Expected last 3 element tags to be 'inputData', not '{root[i].tag}'"

    # Check DMNDI element
    assert root[-1].tag == "dmndi:DMNDI", \
        f"Expected last element tag to 'dmndi:DMNDI', not '{root[-1].tag}'"

    dmndi = root[-1]
    assert len(dmndi) == 1, f"DMNDI should have 1 child 'dmndi_diagram', got {len(dmndi)}"

    dmndi_diagram = dmndi[0]
    assert dmndi_diagram.tag == "dmndi:DMNDiagram", \
        f"Expected dmndi_diagram tag to be 'dmndi:DMNDiagram', not '{dmndi_diagram.tag}'"
    assert dmndi_diagram.get("id") == "DMNDiagram1", \
        f"Expected dmndi_diagram id to be 'DMNDiagram1, not '{dmndi_diagram.get('id')}'"

    assert len(dmndi_diagram) == 15, f"Expected 15 child elements, got {len(dmndi_diagram)}"
    assert all(dmndi_diagram[i].tag == "dmndi:DMNShape" for i in range(0, 6))
    assert all(dmndi_diagram[i].tag == "dmndi:DMNEdge" for i in range(6, 15))


def test_clean_id() -> None:
    """Tests clean_id."""
    assert dmng.clean_id('What the $%^&*1@') == "What_the_-1-", f"Expected \"What_the_-1-\",\
    got {dmng.clean_id('What the $%^&*1@')}"
