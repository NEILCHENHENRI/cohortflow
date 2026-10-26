from framework.logic import LogicExpression, assign_logic_node_ids
from interfaces.components.graph import _gate_svg


def test_vertical_logic_gate_svg_matches_top_to_bottom_graph_flow():
    svg = _gate_svg("AND", "#fff", "#000")
    assert 'viewBox="0 0 90 140"' in svg
    assert 'matrix(0 1 -1 0 90 0)' in svg


def test_nested_or_then_and_is_structurally_gate_to_gate():
    logic = assign_logic_node_ids(LogicExpression.and_(
        LogicExpression.or_(
            LogicExpression.predicate("p1", "A"),
            LogicExpression.predicate("p2", "B"),
        ),
        LogicExpression.predicate("p3", "C"),
    ))
    assert logic.kind == "and"
    assert logic.node_id == "g1"
    assert logic.children[0].kind == "or"
    assert logic.children[0].node_id == "g2"
    assert logic.children[0].predicate_ids() == ("p1", "p2")
    assert logic.children[1].predicate_id == "p3"


def test_renderer_layout_layers_database_before_fact_before_keyword_before_logic():
    from interfaces.components.graph import _compute_layered_layout

    node_ids = ["db", "code", "keyword", "gate", "result"]
    edges = [
        ("db", "code"),
        ("code", "keyword"),
        ("keyword", "gate"),
        ("gate", "result"),
    ]
    positions = _compute_layered_layout(node_ids, edges)

    assert positions["db"][1] < positions["code"][1]
    assert positions["code"][1] < positions["keyword"][1]
    assert positions["keyword"][1] < positions["gate"][1]
    assert positions["gate"][1] < positions["result"][1]
