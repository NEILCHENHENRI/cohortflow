from framework.artifact_store import InMemoryArtifactStore
from framework.graph import ExecutionGraph
from framework.reconcile import ReuseReport
from framework.session import AgentState, SessionStatus, StagedExecution
from framework.task import Objective, TaskSpec


def task(name):
    return TaskSpec(domain="x", objective=Objective("x", name))


def test_state_distinguishes_committed_and_staged_task():
    committed = task("committed")
    proposed = task("proposed")
    state = AgentState(committed, ExecutionGraph(), version=2)
    state.staged = StagedExecution(
        base_version=2,
        task=proposed,
        graph=ExecutionGraph(),
        artifacts=InMemoryArtifactStore(),
        reuse_report=ReuseReport((), (), (), ()),
    )
    assert state.task.objective.description == "committed"
    assert state.staged.task.objective.description == "proposed"
    assert state.status is SessionStatus.READY


def test_session_execution_mode_two_way_control():
    from framework.runtime import ExecutionMode
    from framework.operator import OperatorRegistry

    class DummyAgent:
        def operator_registry(self):
            return OperatorRegistry()

    from framework.session import SteerableSession
    session = SteerableSession(DummyAgent())

    session.set_execution_mode(ExecutionMode.SEQUENTIAL, workers=4)
    assert session.runtime.execution_mode is ExecutionMode.SEQUENTIAL
    assert session.runtime.max_workers == 1

    session.set_execution_mode(ExecutionMode.PARALLEL, workers=4)
    assert session.runtime.execution_mode is ExecutionMode.PARALLEL
    assert session.runtime.max_workers == 1


def _lineage_graph(input_value: int):
    from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus

    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input", NodeKind.INPUT, outputs=("x",),
        config={"value": input_value}, status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "middle", NodeKind.COMPUTATION, operator_name="middle",
        inputs=("x",), outputs=("y",), config={"stable": True},
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "child", NodeKind.COMPUTATION, operator_name="child",
        inputs=("y",), outputs=("z",), config={"stable": True},
    ), depends_on=("middle",))
    return graph


def test_artifact_cache_key_includes_transitive_semantic_lineage():
    from framework.operator import OperatorRegistry
    from framework.session import SteerableSession

    class DummyAgent:
        def operator_registry(self):
            return OperatorRegistry()

    session = SteerableSession(DummyAgent())
    old = _lineage_graph(1)
    new = _lineage_graph(2)

    assert old.nodes["child"].semantic_fingerprint == new.nodes["child"].semantic_fingerprint
    assert old.nodes["middle"].semantic_fingerprint == new.nodes["middle"].semantic_fingerprint
    assert session._node_cache_key(old, "child") != session._node_cache_key(new, "child")


def test_artifact_cache_does_not_restore_child_before_parent_is_valid():
    from framework.artifact import Artifact
    from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus
    from framework.operator import OperatorRegistry
    from framework.session import SteerableSession

    class DummyAgent:
        def operator_registry(self):
            return OperatorRegistry()

    session = SteerableSession(DummyAgent())
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode(
        "input", NodeKind.INPUT, outputs=("requirement",),
        config={"payload": "new"}, status=NodeStatus.COMPLETED,
    ))
    graph.add_node(ExecutionNode(
        "evaluate", NodeKind.COMPUTATION, operator_name="evaluate",
        inputs=("requirement",), outputs=("patients",),
        status=NodeStatus.INVALIDATED,
    ), depends_on=("input",))
    graph.add_node(ExecutionNode(
        "relation", NodeKind.COMPUTATION, operator_name="relation",
        inputs=("requirement",), outputs=("related",),
        status=NodeStatus.INVALIDATED,
    ), depends_on=("evaluate",))

    store = InMemoryArtifactStore()
    store.put(Artifact("requirement", "requirement", {"term": "new"}))
    cached = Artifact("related", "generic", {99})
    session.artifact_cache[session._node_cache_key(graph, "relation")] = (cached,)

    restored = session._restore_cached(graph, store)

    assert "relation" not in restored
    assert not store.contains("related")
    assert graph.nodes["relation"].status is NodeStatus.INVALIDATED


def test_composed_agent_executes_and_revises_a_non_cohort_task():
    import pytest
    from framework.agent import ComposedDomainAgent
    from framework.correction import CorrectionAnalysis, CorrectionIntent, CorrectionScope
    from framework.graph import ExecutionNode, NodeKind, NodeStatus
    from framework.operator import OperatorRegistry, OperatorRole, OperatorSpec
    from framework.task import TaskDelta

    class Parser:
        def bind_module_runner(self, runner):
            self.runner = runner

        def parse_task(self, request, on_progress=None):
            return self.runner("stock.parse", lambda: task(request), on_progress=on_progress)

        def interpret_correction(self, current_task, message, on_progress=None):
            return CorrectionAnalysis(
                intent=CorrectionIntent(
                    message, "Update quantity", message, CorrectionScope.LOCAL, (), 1.0,
                ),
                task_delta=TaskDelta(objective=Objective("x", message)),
            )

    class Planner:
        def build_graph(self, task):
            graph = ExecutionGraph()
            graph.add_node(ExecutionNode(
                "input", NodeKind.INPUT, outputs=("quantity",),
                config={"quantity": task.objective.description}, status=NodeStatus.COMPLETED,
            ))
            graph.add_node(ExecutionNode(
                "double", NodeKind.COMPUTATION, operator_name="double",
                inputs=("quantity",), outputs=("result",),
            ), depends_on=("input",))
            return graph

        def build_input_values(self, task):
            return {"quantity": int(task.objective.description)}

    class InventoryAgent(ComposedDomainAgent):
        domain = "x"

        def __init__(self):
            self.parser, self.planner = Parser(), Planner()
            self._registry = OperatorRegistry()
            self._registry.register(
                OperatorSpec("double", OperatorRole.TRANSFORM, "Double stock"),
                lambda inputs, config, context: {"result": 2 * inputs["quantity"]},
            )
            self._bind_parser_modules()

        def summarize(self, artifacts):
            return {"quantity": artifacts.get("result").value}

    agent = InventoryAgent()
    events = []
    parsed = agent.parse_task("3", on_progress=lambda k, v: events.append((k, v)))
    assert ("module_finished", "stock.parse") in events
    session = agent.create_session()
    session.initialize(parsed)
    session.execute()
    assert session.summary() == {"quantity": 6}
    analysis = agent.interpret_correction(parsed, "5")
    session.stage(analysis.task_delta)
    session.execute_staged()
    assert session.summary() == {"quantity": 10}
    assert session.state.version == 2
    with pytest.raises(NotImplementedError, match="manual fallback editing"):
        agent.build_manual_edit_form(parsed, version=1)


def test_domain_repair_guidance_also_applies_outside_operator(tmp_path):
    from unittest.mock import Mock
    from framework.session import SteerableSession
    from framework.operator import OperatorRegistry
    from framework.failure import FailureEvent
    from framework.repair import RepairManager, RepairPolicy, RepairMode, RepairStatus
    from agents.cohort.agent import CohortAgent
    class Agent:
        assess_failure = CohortAgent.assess_failure
        def operator_registry(self):
            return OperatorRegistry()
    session = SteerableSession(Agent())
    backend = Mock()
    session.repair_manager = RepairManager(repository_root=str(tmp_path), local_backend=backend,
        policy=RepairPolicy(mode=RepairMode.LOCAL))
    error = type("UnitConversionError", (ValueError,), {})("ambiguous units")
    outcome = session.repair_failure(FailureEvent.capture(error, origin="parsing"))
    assert outcome.status is RepairStatus.NEEDS_INPUT
    backend.analyze.assert_not_called()
    assert outcome.retry_attempts == 0
