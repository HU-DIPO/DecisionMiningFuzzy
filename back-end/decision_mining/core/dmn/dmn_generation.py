"""dmn_generation.py.

This module contains classes and functions to
generate a Decision Requirements Diagram XML structure.

The XML is generated using DMN standard. See https://www.omg.org/dmn/.
"""
import re
import uuid
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree

import numpy as np

from decision_mining.core.dmn.rule import Rule


def clean_id(id_str: str) -> str:
    """Replaces invalid characters in id_str.

    Args:
        id_str (str): id string with possible invalid characters.

    Returns:
        str: String without invalid characters.
    """
    new_id = re.sub("[^0-9a-zA-Z-_.]+", "-",
                    id_str.replace(" ", "_").replace("?", ".").replace("!", "."))

    return new_id


class DMNNode:
    """DMN Base Node.

    Base node for representing an XML node with DMN standard.
    """

    def __init__(self, name: str) -> None:
        """Initializes DMN Node.

        Args:
            name (str): Name of DMN Node.
        """
        self.name = name

    def create_dmndi_shape(self, parent: ElementTree.Element, referer: str, pos: Tuple[int, int],
                           width: int = 150, height: int = 50) -> None:
        """Creates DMNDI Shape.

        Creates XML SubElement with dmndi DMNShape type.
        DMNDI is used for visualization of DMN elements.

        Args:
            parent (xml.etree.ElementTree.Element): XML Parent Element.
            referer (str): Name of element that will be visualized.
            pos (Tuple[int, int]): (x, y) position coordinates.
            width (int): Element width. Defaults to 150.
            height (int): Element height. Defaults to 50.
        """
        self.element = ElementTree.SubElement(parent, "dmndi:DMNShape")
        self.element.set("id", "DMNShape_" + clean_id(referer))
        self.element.set("dmnElementRef", clean_id(referer))

        self.width = width
        self.height = height
        self.x, self.y = pos

        bounds = ElementTree.SubElement(self.element, "dc:Bounds")
        bounds.set("width", str(self.width))
        bounds.set("height", str(self.height))
        bounds.set("x", str(self.x))
        bounds.set("y", str(self.y))


class InputNode(DMNNode):
    """DMN Input Node.

    Represents an XML DMN Input element.
    """

    def __init__(self, name: str) -> None:
        """Initializes DMN Input Node.

        Args:
            name (str): Name of InputNode.
        """
        super().__init__(name)

    def set_xml_node(self, parent: ElementTree.Element) -> None:
        """Creates XML InputNode Element.

        Creates an XML SubElement to represent a DMN 'inputData' element.

        Args:
            parent (xml.etree.ElementTree.Element): XML Parent Element.
        """
        self.element = ElementTree.SubElement(parent, "inputData")
        self.element.set("id", clean_id(self.name))
        self.element.set("name", self.name)


class DecisionNode(DMNNode):
    """DMN Decision Node.

    Represents an XML DMN Decision element.
    """

    def __init__(self, name: str) -> None:
        """Initializes DMN Decision Node.

        Args:
            name (str): Name of DecisionNode.
        """
        super().__init__(name)
        self.dependencies = []
        self.rules = []

    def add_dependencies(self, dependencies: List[Union["DecisionNode", InputNode]]) -> None:
        """Extends Decision dependencies.

        Extends the list of dependencies that the decision depends on.
        Dependencies can be InputNodes or DecisionNodes.

        Args:
            dependencies (list): List of InputNode/DecisionNode Objects.
        """
        self.dependencies.extend(dependencies)

    def add_rules(self, rules: List[Rule]) -> None:
        """Extends Decision Rules.

        Extends the list of discovered rules beloning to the decision.

        Args:
            rules (List[Rule]): List of Rule Objects.
        """
        self.rules.extend(rules)

    def set_xml_node(self, parent: ElementTree.Element) -> None:
        """Creates XML Element Node.

        Creates an XML 'decision' tag to make the decision exist in the DMN diagram.

        Args:
            parent (xml.etree.ElementTree.Element): XML Parent Element.
        """
        self.element = ElementTree.SubElement(parent, "decision")
        self.element.set("id", clean_id(self.name))
        self.element.set("name", self.name)

        for dependency in self.dependencies:
            name = dependency.name
            info_requirement = ElementTree.SubElement(
                self.element, "informationRequirement"
            )
            info_requirement.set("id", "info_req_" + clean_id(name + "_" + self.name))
            requirement = ElementTree.SubElement(info_requirement, "requiredInput")
            requirement.set("href", "#" + clean_id(name))

        self._create_decision_table()

    def _create_waypoints(self, dependency: DMNNode) -> Union[
            Tuple[Tuple[int, int], Tuple[int, int]],
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Creates an arrow connection between Decision and dependency.

        The arrow is being constructed by a minimum of two waypoints.
        The first waypoint is always the start of the arrow,
        the last waypoint is always the end of the arrow.
        More waypoints can be added, they function as anchor points.

        This function uses two waypoints if the objects are vertically aligned.
        If not, an anchor point is added to make sure the line doesn't pass through the object.

        Args:
            dependency (DMNNode): Decision dependency object.

        Returns:
            Union[Tuple[Tuple[int, int], Tuple[int, int]], \
                Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]: Tuple (start, end) or \
                    (start, anchor, end) (x,y) waypoint coordinates.
        """
        start = (dependency.width // 2 + dependency.x, dependency.y)
        end = (self.width // 2 + self.x, self.y + self.height)
        if abs(end[0] - start[0]) > 0:
            anchor = (end[0], int(end[1] + 0.1 * abs(end[0] - start[0])))
            return start, anchor, end
        return start, end

    def create_information_requirement(self, parent: ElementTree.Element,
                                       referer_obj: DMNNode) -> None:
        """Create DMN Information Requirement.

        An Information Requirement marks an object as dependency from DMN perspective.

        Args:
            parent (xml.etree.ElementTree.Element): XML Parent Element.
            referer_obj (DMNNode): Node object to connect arrow with.
        """
        self.element = ElementTree.SubElement(parent, "dmndi:DMNEdge")
        self.element.set("id", "DMNEdge_" + clean_id(referer_obj.name + "_" + self.name))
        self.element.set("dmnElementRef",
                         "info_req_" + clean_id(referer_obj.name + "_" + self.name))

        waypoints = self._create_waypoints(referer_obj)
        for x, y in waypoints:
            waypoint = ElementTree.SubElement(self.element, "di:waypoint")
            waypoint.set("x", str(x))
            waypoint.set("y", str(y))

    def _create_decision_table(self) -> None:
        """Creates a Decision Table for current DecisionNode instance.

        Creates an XML DMN structure to visualize a Decision Table,
        using discovered Rules to fill the table.
        """
        decision_table = ElementTree.SubElement(self.element, "DecisionTable")
        decision_table.set("id", clean_id(f"DecisionTable_{self.name}"))
        decision_table.set("hitPolicy", "UNIQUE")  # TODO: check hitpolicy behaviour

        # Create input nodes
        for i, dependency in enumerate(self.dependencies):
            xml_input_node = ElementTree.SubElement(decision_table, "input")
            xml_input_node.set("id", clean_id(f"DecisionTable{self.name}Input{i}"))
            xml_input_node.set("label", dependency.name)

            input_expression = ElementTree.SubElement(xml_input_node, "inputExpression")
            input_expression.set(
                "id", clean_id(f"{dependency.name}InputExpression-{uuid.uuid4()}"))
            input_expression.set("typeRef", "[type]")  # TODO: Add column type

            input_text = ElementTree.SubElement(input_expression, "text")
            input_text.text = dependency.name

        # Create output Node
        xml_output_node = ElementTree.SubElement(decision_table, "output")
        xml_output_node.set("id", clean_id(f"OutputClause{self.name}"))
        xml_output_node.set("label", self.name)
        xml_output_node.set("name", self.name)
        xml_output_node.set("typeRef", "[type]")  # TODO: Add output column type

        for rule in self.rules:
            rule_xml_element = rule.create_xml_element()
            decision_table.append(rule_xml_element)


def create_node_objects(attr_inputs: List[List[str]]) -> Dict[str, Union[DecisionNode, InputNode]]:
    """Creates Node objects.

    Creates DMN Node objects from dataset attributes.

    For each list of attributes, the right column is the output/decision,
    the other columns are dependencies to that decision.

    Args:
        attr_inputs (List[List[str]]): List of lists with dataset attribute names.

    Returns:
        Dict[str, Union[DecisionNode, InputNode]]: Dictionary with names and DMNNode objects.
    """
    attribute_objects = dict()

    decisions = [dataset[-1] for dataset in attr_inputs]
    for decision_name in decisions:
        attribute_objects[decision_name] = DecisionNode(decision_name)

    for decision_data in attr_inputs:
        for attribute in decision_data:
            if attribute in attribute_objects.keys():
                continue
            attribute_objects[attribute] = InputNode(attribute)
    return attribute_objects


def create_dependencies(inputs: List[List[str]],
                        node_objects: Dict[str, Union[DecisionNode, InputNode]]
                        ) -> List[DecisionNode]:
    """Creates Decision dependencies.

    Creates a list of dependencies for each decision in `inputs`.

    Args:
        inputs (List[List[str]]): List of lists of dataset attribute names.
        node_objects (Dict[str, Union[DecisionNode, InputNode]]): Dictionary with names and \
            DMNNode objects.

    Returns:
        List[DecisionNode]: List of DecisionNode objects.
    """
    decision_nodes = []
    for decision_data in inputs:
        decision = node_objects[decision_data[-1]]
        inputs = [node_objects[object_name] for object_name in decision_data[:-1]]
        decision.add_dependencies(inputs)
        decision_nodes.append(decision)
    return decision_nodes


def create_xml(node_objects: Dict[str, Union[DecisionNode, InputNode]],
               decision_nodes: List[DecisionNode]) -> ElementTree.ElementTree:
    """Creates XML DMN structure.

    Main function, creates XML elements and uses input to visualize a full DMN Diagram.

    Args:
        node_objects (Dict[str, Union[DecisionNode, InputNode]]): Dictionary with names and \
            DMNNode objects.
        decision_nodes (List[DecisionNode]): List of DecisionNode objects.

    Returns:
        ElementTree.ElementTree: Built XML tree with DMN standard.
    """
    root = ElementTree.Element("definitions")
    root.set("xmlns", "https://www.omg.org/spec/DMN/20191111/MODEL/")
    root.set("xmlns:dmndi", "https://www.omg.org/spec/DMN/20191111/DMNDI/")
    root.set("xmlns:dc", "http://www.omg.org/spec/DMN/20180521/DC/")
    root.set("xmlns:di", "http://www.omg.org/spec/DMN/20180521/DI/")
    root.set("id", "definitions")
    root.set("name", "definitions")
    root.set("namespace", "http://camunda.org/schema/1.0/dmn")

    for node_object in node_objects.values():
        node_object.set_xml_node(root)

    dmndi = ElementTree.SubElement(root, "dmndi:DMNDI")
    dmndi_diagram = ElementTree.SubElement(dmndi, "dmndi:DMNDiagram")
    dmndi_diagram.set("id", "DMNDiagram1")

    layer0 = [node for node in list(node_objects.values()) if isinstance(node, InputNode)]
    y = len(decision_nodes) * 150 + 150
    x = 100
    for node in layer0:
        node.create_dmndi_shape(dmndi_diagram, node.name, (x, y))
        x += 200

    x = 100
    y = len(decision_nodes) * 150 - 150
    for decision_node in decision_nodes:
        decision_node.create_dmndi_shape(dmndi_diagram, decision_node.name, (x, y))
        x += 200
        y -= 150

    for decision_node in decision_nodes:
        for dependency in decision_node.dependencies:
            decision_node.create_information_requirement(dmndi_diagram, dependency)

    tree = ElementTree.ElementTree(root)
    return tree


if __name__ == "__main__":  # pragma: no cover
    from decision_mining.core import c45, fuzzy
    from decision_mining.core.dmn import rule_c45, rule_fuzzy
    print(clean_id("What the $%^&*1@"))
    "What_the_-1-"

    column_names = [
        ["duration", "premium", "amount", "risk#$%^&*()"],
        ["premium", "amount", "risk#$%^&*()", "approval"],
        ["duration", "premium", "amount", "check"],
    ]
    clsfr = c45.C45Classifier(continuous_cols=np.array([0, ]))
    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    clsfr.fit(X, y)
    # print(clsfr.score(X, y))
    pathes = c45.traverse_c45(clsfr)
    rules = rule_c45.make_c45_rules([0, 1, 2], pathes)
    # for rule in rules:
    #     print(rule)

    drd_objects = create_node_objects(column_names)
    decision_nodes = create_dependencies(column_names, drd_objects)
    decision_nodes[0].rules = rules  # Temporay solution for adding rules.

    fuzzy_cls = fuzzy.FuzzyClassifier([0])
    fuzzy_cls.fit(X, y, 0.1)
    rules = rule_fuzzy.make_fuzzy_rules([0, 1, 2], fuzzy_cls)

    decision_nodes[1].rules = rules  # Temporay solution for adding rules.
    tree = create_xml(drd_objects, decision_nodes)
    tree.write("test.dmn")
