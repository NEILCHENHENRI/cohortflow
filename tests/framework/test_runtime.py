from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore
from framework.graph import (
    ExecutionGraph,
    ExecutionNode,
    NodeKind,
    NodeStatus,
)
from framework.operator import (
    OperatorRegistry,
    OperatorRole,
    OperatorSpec,
)
from framework.runtime import GraphRuntime


def test_runtime_executes_parallel_branches_and_merge():
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec("double", OperatorRole.TRANSFORM, "Double x."),
        lambda inputs, config, context: {"a": inputs["x"] * 2},
    )
    registry.register(
        OperatorSpec("square", OperatorRole.TRANSFORM, "Square y."),
        lambda inputs, config, context: {"b": inputs["y"] ** 2},
    )
    registry.register(
        OperatorSpec("merge", OperatorRole.AGGREGATE, "Add values."),
        lambda inputs, config, context: {
            "result": inputs["a"] + inputs["b"]
        },
    )

    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input:x", NodeKind.INPUT,
        outputs=("x",), status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "input:y", NodeKind.INPUT,
        outputs=("y",), status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "double", NodeKind.COMPUTATION,
        operator_name="double", inputs=("x",), outputs=("a",),
    ), depends_on=("input:x",))
    graph.add_node(ExecutionNode(
        "square", NodeKind.COMPUTATION,
        operator_name="square", inputs=("y",), outputs=("b",),
    ), depends_on=("input:y",))
    graph.add_node(ExecutionNode(
        "merge", NodeKind.COMPUTATION,
        operator_name="merge", inputs=("a", "b"),
        outputs=("result",),
    ), depends_on=("double", "square"))

    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 2))
    store.put(Artifact("y", "number", 3))

    result = GraphRuntime(registry, max_workers=2).run(
        graph, store, execution_version=1
    )

    assert result.stopped is False
    assert store.get("result").value == 13
    assert graph.nodes["merge"].status == NodeStatus.COMPLETED
    assert graph.nodes["merge"].duration_ms is not None


def _two_ready_node_graph():
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input:x", NodeKind.INPUT, outputs=("x",), status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "llm_node", NodeKind.COMPUTATION,
        operator_name="llm_op", inputs=("x",), outputs=("llm_out",),
    ), depends_on=("input:x",))
    graph.add_node(ExecutionNode(
        "plain_node", NodeKind.COMPUTATION,
        operator_name="plain_op", inputs=("x",), outputs=("plain_out",),
    ), depends_on=("input:x",))
    return graph


def test_parallel_mode_keeps_graph_execution_single_lane(monkeypatch):
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec("llm_op", OperatorRole.INFER, "May call an LLM.", metadata={"may_use_llm": True}),
        lambda inputs, config, context: {"llm_out": inputs["x"]},
    )
    registry.register(
        OperatorSpec("plain_op", OperatorRole.TRANSFORM, "No LLM."),
        lambda inputs, config, context: {"plain_out": inputs["x"]},
    )
    graph = _two_ready_node_graph()
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 1))
    from framework.runtime import ExecutionMode
    runtime = GraphRuntime(registry, max_workers=4, execution_mode=ExecutionMode.PARALLEL)
    runtime.run(graph, store, execution_version=1)
    assert graph.nodes["llm_node"].status is NodeStatus.COMPLETED
    assert graph.nodes["plain_node"].status is NodeStatus.COMPLETED

def test_sequential_mode_does_not_create_thread_pool(monkeypatch):
    import framework.runtime as runtime_module
    from framework.runtime import ExecutionMode

    registry = OperatorRegistry()
    registry.register(
        OperatorSpec("llm_op", OperatorRole.INFER, "May call an LLM.", metadata={"may_use_llm": True}),
        lambda inputs, config, context: {"llm_out": inputs["x"]},
    )
    registry.register(
        OperatorSpec("plain_op", OperatorRole.TRANSFORM, "No LLM."),
        lambda inputs, config, context: {"plain_out": inputs["x"]},
    )

    class ForbiddenPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("sequential mode must not create a thread pool")

    monkeypatch.setattr(runtime_module, "ThreadPoolExecutor", ForbiddenPool)
    graph = _two_ready_node_graph()
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 1))
    GraphRuntime(
        registry, max_workers=4, execution_mode=ExecutionMode.SEQUENTIAL,
    ).run(graph, store, execution_version=1)
