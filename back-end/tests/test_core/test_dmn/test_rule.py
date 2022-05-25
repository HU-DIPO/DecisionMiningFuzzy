"""Tests rule.py."""
from decision_mining.core.dmn import rule


def test_Rule_rule_generator() -> None:
    """Tests the rule.Rule.rule_generator function."""
    RuleFactory = rule.Rule.rule_generator([0, 1, 2])
    r = RuleFactory()

    assert list(r.cols.keys()) == [0, 1, 2], f"r should contain columns 0,1,2, not {r.cols}"

    assert all(val is None for val in r.cols.values()
               ), f"All values should equal None, not {r.cols.values()}"


def test_create_xml_element() -> None:
    """Tests the rule.Rule.create_xml_element function."""
    r = rule.Rule([0, 1, 2])
    r.cols[0] = "Test"
    r.cols[1] = [{"threshold": 51.5, "<": False},
                 {"threshold": 79.5, "<": True}]
    r.decision = "True"
    xml_elem = r.create_xml_element()
    col0, col1, col2, output = xml_elem
    assert xml_elem.tag == "rule", f"Tag should be rule, not {xml_elem.tag}"

    assert all(col.tag == "inputEntry" for col in (col0, col1, col2)
               ), "First three children should be inputEntry"
    assert all(col.get("id").startswith("InputEntry-") for col in (col0, col1, col2))
    assert output.tag == "outputEntry", f"Tag should be outputEntry, not {output.tag}"

    for entry, exp_txt in zip(xml_elem, ("Test", "[51.5..79.5]", "", "True")):
        assert entry[0].text == exp_txt, f"Entry text should be {exp_txt}, not {entry[0].text}"


def test_parse_continuous() -> None:
    """Tests the rule.Rule.parse_continuous function."""
    both_exist = [{"threshold": 51.5, "<": False},
                  {"threshold": 79.5, "<": True}]
    assert rule.Rule.parse_continuous(both_exist) == "[51.5..79.5]"
    no_greater = [{"threshold": 1, "<": True}]
    no_smaller = [{"threshold": 0, "<": False}]
    assert rule.Rule.parse_continuous(no_greater) == "[..1]"
    assert rule.Rule.parse_continuous(no_smaller) == "[0..]"


def test_get_basic_rule() -> None:
    """Tests the rule.Rule.get_basic_rule function."""
    r = rule.Rule([0, 1, 2])
    r.cols[0] = "Test"
    r.cols[1] = [{"threshold": 51.5, "<": False},
                 {"threshold": 79.5, "<": True}]
    r.cols[3] = None
    r.decision = True

    basic_rule = r.get_basic_rule()

    assert basic_rule[0] == "Test", f"First element should be 'Test', not {basic_rule[0]}"
    assert basic_rule[1] == [
        51.5, 79.5], f"{basic_rule[1]} should be [51.5, 79.5], not {basic_rule[1]}"
    assert basic_rule[2] == "None", f"{basic_rule[2]} should be None, not {basic_rule[2]}"
    assert basic_rule[-1] is True, f"Last element should be True, not {basic_rule[-1]}"


def test_get_basic_dt_rules() -> None:
    """Tests the rule.get_basic_dt_rules function."""
    rules = []
    for i in range(3):
        r = rule.Rule([0])
        r.cols[0] = f"Test_{i}"
        r.decision = f"True_{i}"
        rules.append(r)

    basic_rules = rule.get_basic_dt_rules(rules)

    assert isinstance(
        basic_rules[0],
        list), f"rules in basic_rules should be a list, not {type(basic_rules[0])}"
    assert len(basic_rules) == 3, f"basic_rules should have 3 rules, not {len(basic_rules)}"
