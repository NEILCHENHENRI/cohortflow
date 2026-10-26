from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore
from framework.graph import (
    ExecutionGraph,
    ExecutionNode,
    NodeKind,
    NodeStatus,
)
from framework.reconcile import reconcile_graphs


def make_graph(multiplier: int) -> ExecutionGraph:
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input", NodeKind.INPUT,
        outputs=("x",),
        config={"value": 3},
        status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "compute", NodeKind.COMPUTATION,
        operator_name="multiply",
        inputs=("x",), outputs=("y",),
        config={"multiplier": multiplier},
        status=NodeStatus.COMPLETED,
    ), depends_on=("input",))
    return graph


def test_reconcile_reuses_unchanged_node():
    old = make_graph(2)
    new = make_graph(2)
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 3))
    store.put(Artifact("y", "number", 6))

    new_store, report = reconcile_graphs(old, new, store)

    assert report.reused_nodes == ("compute",)
    assert new_store.get("y").value == 6
    assert new.nodes["compute"].status == NodeStatus.COMPLETED


def test_reconcile_invalidates_changed_node():
    old = make_graph(2)
    new = make_graph(3)
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 3))
    store.put(Artifact("y", "number", 6))

    _, report = reconcile_graphs(old, new, store)

    assert report.invalidated_nodes == ("compute",)
    assert new.nodes["compute"].status == NodeStatus.INVALIDATED


def make_chain(input_value: int) -> ExecutionGraph:
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input", NodeKind.INPUT,
        outputs=("x",), config={"value": input_value},
        status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "compute", NodeKind.COMPUTATION,
        operator_name="identity", inputs=("x",), outputs=("y",),
        config={"stable": True}, status=NodeStatus.COMPLETED,
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "downstream", NodeKind.COMPUTATION,
        operator_name="identity", inputs=("y",), outputs=("z",),
        config={"stable": True}, status=NodeStatus.COMPLETED,
    ), depends_on=("compute",))
    return graph


def test_reconcile_invalidates_unchanged_nodes_when_upstream_input_changes():
    old = make_chain(1)
    new = make_chain(2)
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 1))
    store.put(Artifact("y", "number", 1))
    store.put(Artifact("z", "number", 1))

    new_store, report = reconcile_graphs(old, new, store)

    assert report.reused_nodes == ()
    assert report.invalidated_nodes == ("compute", "downstream")
    assert not new_store.contains("y")
    assert not new_store.contains("z")


def test_reconcile_does_not_reuse_completed_node_with_missing_output():
    old = make_graph(2)
    new = make_graph(2)
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 3))

    new_store, report = reconcile_graphs(old, new, store)

    assert report.reused_nodes == ()
    assert report.invalidated_nodes == ("compute",)
    assert not new_store.contains("y")
    assert new.nodes["compute"].status is NodeStatus.INVALIDATED


def test_reconcile_preserves_independent_branch_and_copies_timing_metadata():
    old = make_chain(1)
    new = make_chain(1)
    for graph in (old, new):
        graph.add_node(ExecutionNode(
            "independent", NodeKind.COMPUTATION,
            operator_name="identity", inputs=("x",), outputs=("other",),
            status=NodeStatus.COMPLETED,
        ), depends_on=("input",))
    new.nodes["compute"].config["stable"] = False
    old_node = old.nodes["independent"]
    old_node.duration_ms = 12.5
    old_node.component_timings_ms = {"query": 10.0}
    old_node.attempt_timings = [{"duration_ms": 12.5}]
    store = InMemoryArtifactStore()
    for output in ("x", "y", "z", "other"):
        store.put(Artifact(output, "number", 1))

    new_store, report = reconcile_graphs(old, new, store)

    assert report.reused_nodes == ("independent",)
    assert report.invalidated_nodes == ("compute", "downstream")
    assert new_store.get("other") is store.get("other")
    assert not new_store.contains("y")
    assert not new_store.contains("z")
    new_node = new.nodes["independent"]
    assert new_node.duration_ms == 12.5
    assert new_node.component_timings_ms == {"query": 10.0}
    assert new_node.attempt_timings == [{"duration_ms": 12.5}]
    new_node.component_timings_ms["query"] = 99.0
    new_node.attempt_timings[0]["duration_ms"] = 99.0
    assert old_node.component_timings_ms == {"query": 10.0}
    assert old_node.attempt_timings == [{"duration_ms": 12.5}]
