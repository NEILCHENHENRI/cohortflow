from __future__ import annotations

import pytest

from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore
from framework.execution_backend import ExecutionBackend
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus
from framework.langgraph_runtime import LangGraphRuntime
from framework.operator import OperatorRegistry, OperatorRole, OperatorSpec
from framework.runtime import ExecutionMode, GraphRuntime
from framework.session import SteerableSession


class _FakeSaver:
    pass


class _FakeCompiledGraph:
    def __init__(self, nodes, edges):
        self.nodes = dict(nodes)
        self.edges = list(edges)
        self.last_config = None

    def invoke(self, state, config=None):
        self.last_config = config
        completed: set[str] = set()
        current = dict(state)
        node_names = set(self.nodes)

        requirements: dict[str, set[str]] = {name: set() for name in node_names}
        for source, dest in self.edges:
            if dest not in node_names:
                continue
            if source == "__start__":
                continue
            if isinstance(source, (list, tuple)):
                requirements[dest].update(source)
            else:
                requirements[dest].add(source)

        while len(completed) < len(node_names):
            ready = sorted(
                name
                for name in node_names - completed
                if requirements[name].issubset(completed)
            )
            if not ready:
                raise RuntimeError("fake LangGraph stalled")
            for name in ready:
                update = self.nodes[name](current) or {}
                current.setdefault("completed", [])
                current["completed"] = current["completed"] + list(
                    update.get("completed", [])
                )
                completed.add(name)
        return current


class _FakeStateGraph:
    def __init__(self, _state_schema):
        self.nodes = {}
        self.edges = []

    def add_node(self, name, action):
        self.nodes[name] = action
        return self

    def add_edge(self, source, dest):
        self.edges.append((source, dest))
        return self

    def compile(self, checkpointer=None):
        assert isinstance(checkpointer, _FakeSaver)
        return _FakeCompiledGraph(self.nodes, self.edges)


def _fake_langgraph():
    return _FakeStateGraph, "__start__", "__end__", _FakeSaver


def _registry():
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
        OperatorSpec("merge", OperatorRole.AGGREGATE, "Merge results."),
        lambda inputs, config, context: {"result": inputs["a"] + inputs["b"]},
    )
    return registry


def _graph_and_store():
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input:x", NodeKind.INPUT, outputs=("x",), status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "input:y", NodeKind.INPUT, outputs=("y",), status=NodeStatus.COMPLETED,
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
        operator_name="merge", inputs=("a", "b"), outputs=("result",),
    ), depends_on=("double", "square"))
    store = InMemoryArtifactStore()
    store.put(Artifact("x", "number", 2))
    store.put(Artifact("y", "number", 3))
    return graph, store


def test_langgraph_adapter_matches_native_artifacts(monkeypatch):
    import framework.langgraph_runtime as module

    monkeypatch.setattr(module, "_load_langgraph", _fake_langgraph)

    native_graph, native_store = _graph_and_store()
    langgraph_graph, langgraph_store = _graph_and_store()
    registry = _registry()

    native = GraphRuntime(
        registry, max_workers=2, execution_mode=ExecutionMode.PARALLEL,
    ).run(native_graph, native_store, execution_version=1)
    adapted = LangGraphRuntime(
        registry, max_workers=2, execution_mode=ExecutionMode.PARALLEL,
    ).run(langgraph_graph, langgraph_store, execution_version=1)

    assert native.stopped is adapted.stopped is False
    assert native_store.get("result").value == langgraph_store.get("result").value == 13
    assert set(native.completed_nodes) == set(adapted.completed_nodes)
    assert langgraph_graph.nodes["merge"].status is NodeStatus.COMPLETED


def test_langgraph_execution_modes_translate_dependencies():
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            "llm_op", OperatorRole.INFER, "May use LLM.",
            metadata={"may_use_llm": True},
        ),
        lambda inputs, config, context: {"x": 1},
    )
    registry.register(
        OperatorSpec("plain_op", OperatorRole.TRANSFORM, "Plain."),
        lambda inputs, config, context: {"y": 1},
    )

    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input", NodeKind.INPUT, outputs=("seed",), status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "llm_a", NodeKind.COMPUTATION, operator_name="llm_op", outputs=("a",),
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "plain", NodeKind.COMPUTATION, operator_name="plain_op", outputs=("p",),
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "llm_b", NodeKind.COMPUTATION, operator_name="llm_op", outputs=("b",),
    ), depends_on=("input",))

    active = [graph.nodes[node_id] for node_id in ("llm_a", "plain", "llm_b")]

    expected = {"llm_a": set(), "plain": {"llm_a"}, "llm_b": {"plain"}}
    parallel = LangGraphRuntime(registry, execution_mode=ExecutionMode.PARALLEL)
    assert parallel._effective_parents(graph, active) == expected

    sequential = LangGraphRuntime(registry, execution_mode=ExecutionMode.SEQUENTIAL)
    assert sequential._effective_parents(graph, active) == expected


def test_langgraph_failure_uses_existing_failure_event_pipeline(monkeypatch):
    import framework.langgraph_runtime as module

    monkeypatch.setattr(module, "_load_langgraph", _fake_langgraph)
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec("boom", OperatorRole.TRANSFORM, "Raise."),
        lambda inputs, config, context: (_ for _ in ()).throw(ValueError("bad node")),
    )
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "boom", NodeKind.COMPUTATION, operator_name="boom", outputs=("x",),
    ))
    store = InMemoryArtifactStore()

    with pytest.raises(ValueError, match="bad node"):
        LangGraphRuntime(registry).run(graph, store, execution_version=4)

    node = graph.nodes["boom"]
    assert node.status is NodeStatus.FAILED
    assert node.failure is not None
    assert node.failure.node_id == "boom"
    assert node.failure.metadata["execution_backend"] == "langgraph"


def test_session_defaults_native_and_can_switch_backend():
    class DummyAgent:
        def operator_registry(self):
            return OperatorRegistry()

    session = SteerableSession(DummyAgent())
    assert session.execution_backend is ExecutionBackend.NATIVE
    assert type(session.runtime) is GraphRuntime

    session.set_execution_mode(ExecutionMode.PARALLEL, workers=4)
    session.set_execution_backend(ExecutionBackend.LANGGRAPH)
    assert session.execution_backend is ExecutionBackend.LANGGRAPH
    assert isinstance(session.runtime, LangGraphRuntime)
    assert session.runtime.execution_mode is ExecutionMode.PARALLEL
    assert session.runtime.max_workers == 1

    session.set_execution_backend(ExecutionBackend.NATIVE)
    assert session.execution_backend is ExecutionBackend.NATIVE
    assert type(session.runtime) is GraphRuntime
    assert session.runtime.execution_mode is ExecutionMode.PARALLEL


def test_real_langgraph_backend_equivalence_when_dependency_installed():
    pytest.importorskip("langgraph")
    native_graph, native_store = _graph_and_store()
    langgraph_graph, langgraph_store = _graph_and_store()
    registry = _registry()

    GraphRuntime(registry, max_workers=2, execution_mode=ExecutionMode.PARALLEL).run(
        native_graph, native_store, execution_version=1
    )
    LangGraphRuntime(
        registry, max_workers=2, execution_mode=ExecutionMode.PARALLEL,
    ).run(langgraph_graph, langgraph_store, execution_version=1)

    assert native_store.get("result").value == langgraph_store.get("result").value
    assert all(
        langgraph_graph.nodes[node].status is NodeStatus.COMPLETED
        for node in ("double", "square", "merge")
    )


@pytest.mark.parametrize("backend", [ExecutionBackend.NATIVE, ExecutionBackend.LANGGRAPH])
def test_session_recovery_continues_both_backends_without_repeating_completed_work(tmp_path, monkeypatch, backend):
    from unittest.mock import Mock
    import framework.langgraph_runtime as module
    from framework.session import AgentState
    from framework.task import TaskSpec, Objective
    from framework.agent import DomainAgent
    from framework.repair import RepairManager, RepairPolicy, RepairMode, RepairStatus
    monkeypatch.setattr(module, "_load_langgraph", _fake_langgraph)
    registry = _registry()
    registry.get("double").function = Mock(side_effect=[TimeoutError("mock"), {"a": 4}])
    square = registry.get("square").function
    registry.get("square").function = Mock(side_effect=square)
    class Agent:
        assess_failure = DomainAgent.assess_failure
        repair_replay = DomainAgent.repair_replay
        def operator_registry(self):
            return registry
    session = SteerableSession(Agent())
    session.repair_manager = RepairManager(repository_root=str(tmp_path), policy=RepairPolicy(mode=RepairMode.LOCAL))
    graph, store = _graph_and_store()
    session.state = AgentState(TaskSpec("test", Objective("test", "unchanged")), graph)
    runtime = session._build_runtime(backend, max_workers=1, execution_mode=ExecutionMode.SEQUENTIAL)
    result = runtime.run(graph, store, execution_version=1)
    assert not result.failed_nodes
    assert store.get("result").value == 13
    assert registry.get("double").function.call_count == 2
    assert registry.get("square").function.call_count == 1
    assert session.last_repair_outcome.status is RepairStatus.RECOVERED
    assert session.state.task.objective.description == "unchanged"
