import pytest

from framework.graph import (
    ExecutionGraph,
    ExecutionNode,
    NodeKind,
    NodeStatus,
)


def test_graph_topological_order_and_descendants():
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input",
        NodeKind.INPUT,
        outputs=("x",),
        status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "a",
        NodeKind.COMPUTATION,
        operator_name="op",
        inputs=("x",),
        outputs=("a_out",),
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "b",
        NodeKind.COMPUTATION,
        operator_name="op",
        inputs=("a_out",),
        outputs=("b_out",),
    ), depends_on=("a",))

    assert graph.topological_order() == ["input", "a", "b"]
    assert graph.descendants({"a"}) == {"b"}


def test_missing_parent_is_rejected():
    graph = ExecutionGraph()
    with pytest.raises(KeyError):
        graph.add_node(
            ExecutionNode(
                "a",
                NodeKind.COMPUTATION,
                operator_name="op",
            ),
            depends_on=("missing",),
        )


def test_topological_order_prioritizes_newly_ready_nodes_over_existing_frontier():
    graph = ExecutionGraph()
    for name, parents in (("z", ()), ("b", ()), ("a", ("b",)), ("c", ("a", "z"))):
        graph.add_node(
            ExecutionNode(name, NodeKind.COMPUTATION, operator_name="op"),
            depends_on=parents,
        )

    assert graph.topological_order() == ["b", "a", "z", "c"]


def test_topological_order_rejects_cycle_in_mutated_dependencies():
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode("a", NodeKind.INPUT))
    graph.add_node(ExecutionNode("b", NodeKind.INPUT), depends_on=("a",))
    graph.dependencies["a"].add("b")

    with pytest.raises(ValueError, match="execution graph contains a cycle"):
        graph.topological_order()
