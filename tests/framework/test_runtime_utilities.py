from __future__ import annotations

import time

import pytest

from framework.artifact import Artifact
from framework.artifact_store import InMemoryArtifactStore
from framework.graph import ExecutionGraph, ExecutionNode, NodeKind, NodeStatus
from framework.module import AgentModuleRole, AgentModuleRunner, AgentModuleSpec
from framework.progress import scoped_progress
from framework.resolution import ResolutionRequired, ResolutionRequest, ResponseMode
from framework.scheduler import Scheduler
from framework.steering import SteeringYield, steering_checkpoint
from framework.timing import measure, timing_session
from framework.errors import DomainValidationError
from framework.parsing import StructuredTaskParser


class QueueStructuredLLM:
    think = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append((kwargs, self.think))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class InventoryParser(StructuredTaskParser):
    def _validation_repair_rules(self, label):
        return "\nKeep the inventory quantity as an integer."


def generate_quantity(parser, **kwargs):
    return parser._generate_with_validation(
        system_prompt="Parse inventory quantities.", user_prompt="Count the stock.",
        schema={"type": "object"}, converter=lambda value: int(value["quantity"]),
        label="Inventory", **kwargs,
    )


def test_structured_parser_retries_with_domain_rules_and_restores_thinking():
    llm = QueueStructuredLLM([{"quantity": "bad"}, {"quantity": 7}])
    parser = InventoryParser(llm)
    events = []
    callback = lambda kind, payload: events.append((kind, payload))
    with parser._thinking_for("compilation"):
        assert generate_quantity(parser, on_progress=callback) == 7
    assert llm.think is True
    assert [thinking for _, thinking in llm.calls] == [False, False]
    first, second = [call for call, _ in llm.calls]
    assert first["user_prompt"] == "Count the stock."
    assert second["user_prompt"] == (
        "Count the stock.\n\nTHE PREVIOUS OUTPUT FAILED DETERMINISTIC VALIDATION.\n"
        "Validation error: ValueError: invalid literal for int() with base 10: 'bad'\n"
        "\nKeep the inventory quantity as an integer."
        "\nRegenerate the complete object without changing the intended semantics."
    )
    assert first["on_progress"] is second["on_progress"] is callback
    assert events == [("module_checkpoint", "semantic.validation_retry:Inventory")]


def test_structured_parser_retry_override_and_failure_restores_thinking():
    llm = QueueStructuredLLM([{}])
    parser = InventoryParser(llm, structural_retry_limit=3)
    with pytest.raises(DomainValidationError, match="after 1 attempt"):
        with parser._thinking_for("compilation"):
            generate_quantity(parser, on_progress=None, retry_limit=0)
    assert llm.think is True
    assert len(llm.calls) == 1
    assert "on_progress" not in llm.calls[0][0]


def test_structured_parser_does_not_retry_unexpected_failures():
    llm = QueueStructuredLLM([RuntimeError("transport unavailable")])
    with pytest.raises(RuntimeError, match="transport unavailable"):
        generate_quantity(InventoryParser(llm), on_progress=None)
    assert len(llm.calls) == 1


def test_structured_parser_yields_before_retrying_obsolete_work():
    class CorrectionWaiting:
        def __call__(self, kind, payload):
            pass

        def should_yield(self):
            return True

    llm = QueueStructuredLLM([{}, {"quantity": 9}])
    with pytest.raises(SteeringYield, match="semantic.validation_retry:Inventory"):
        generate_quantity(InventoryParser(llm), on_progress=CorrectionWaiting())
    assert len(llm.calls) == 1


def test_structured_parser_configuration_is_isolated_and_module_runner_can_bind():
    first = InventoryParser(QueueStructuredLLM([]))
    second = InventoryParser(QueueStructuredLLM([]))
    first.configure_parsing(thinking_by_pass={"compilation": True})
    assert second.thinking_by_pass["compilation"] is False
    with pytest.raises(ValueError, match="unknown parsing pass"):
        first.configure_parsing(thinking_by_pass={"not_a_pass": True})
    events = []
    callback = lambda kind, payload: events.append((kind, payload))
    assert first._run_module("inventory.parse", lambda: 3, on_progress=callback) == 3
    assert events == [
        ("module_started", "inventory.parse"),
        ("module_finished", "inventory.parse"),
        ("module_checkpoint", "inventory.parse"),
    ]
    calls = []

    def runner(name, function, *, on_progress):
        calls.append((name, on_progress))
        return function()

    first.bind_module_runner(runner)
    assert first._run_module("inventory.parse", lambda: 5, on_progress=callback) == 5
    assert calls == [("inventory.parse", callback)]
    assert len(events) == 3


def test_scoped_progress_namespaces_non_stage_and_labels_stage():
    events = []
    emit = scoped_progress(lambda kind, payload: events.append((kind, payload)), scope="parser", label="Parsing")
    assert emit is not None
    emit("token", "abc")
    emit("stage", "Compiling")
    assert events == [("token:parser", "abc"), ("stage", "Parsing: Compiling")]
    assert scoped_progress(None, scope="x", label="X") is None


def test_timing_session_records_nested_parent_and_aggregates_names():
    with timing_session() as recorder:
        with measure("outer", category="parse", metadata={"mode": "graphite"}):
            time.sleep(0.001)
            with measure("inner", category="llm"):
                time.sleep(0.001)
        with measure("inner", category="llm"):
            pass
    report = recorder.report()
    spans = report["spans"]
    assert report["total_wall_ms"] >= 0
    assert report["totals_by_name_ms"]["inner"] >= 0
    nested = next(s for s in spans if s["name"] == "inner" and s["parent"] == "outer")
    assert nested["category"] == "llm"
    assert next(s for s in spans if s["name"] == "outer")["metadata"] == {"mode": "graphite"}


def test_agent_module_runner_progress_success_failure_and_expected_interruptions():
    runner = AgentModuleRunner()
    spec = AgentModuleSpec("demo", AgentModuleRole.INTERPRET, "Demo module", "test", checkpoint_after=False)
    events = []
    assert runner.run(spec, lambda: 7, on_progress=lambda a, b: events.append((a, b))) == 7
    assert events == [("module_started", "demo"), ("module_finished", "demo")]

    events.clear()
    with pytest.raises(RuntimeError):
        runner.run(spec, lambda: (_ for _ in ()).throw(RuntimeError("boom")), on_progress=lambda a, b: events.append((a, b)))
    assert events[-1] == ("module_failed", "demo")

    request = ResolutionRequest(
        resolver_id="x", error_code="needs_resolution", message="choose", recoverable=True,
        input_payload={"field": "term", "query": "q"}, options=(),
        response_mode=ResponseMode.SELECT_ONE,
    )
    events.clear()
    with pytest.raises(ResolutionRequired):
        runner.run(spec, lambda: (_ for _ in ()).throw(ResolutionRequired(request, question="choose")), on_progress=lambda a, b: events.append((a, b)))
    assert events[-1][0] == "module_interrupted"


def test_steering_checkpoint_yields_only_when_callback_requests_it():
    class Callback:
        def __init__(self, should):
            self.should = should
            self.events = []
        def __call__(self, kind, payload):
            self.events.append((kind, payload))
        def should_yield(self):
            return self.should

    cb = Callback(False)
    steering_checkpoint(cb, "after_parse")
    assert cb.events == [("module_checkpoint", "after_parse")]

    cb = Callback(True)
    with pytest.raises(SteeringYield) as exc:
        steering_checkpoint(cb, "after_parse")
    assert exc.value.checkpoint == "after_parse"


def test_scheduler_requires_completed_parents_and_present_input_artifacts():
    graph = ExecutionGraph()
    graph.add_node(ExecutionNode("input", NodeKind.INPUT, outputs=("a",)))
    graph.nodes["input"].status = NodeStatus.COMPLETED
    graph.add_node(ExecutionNode("n1", NodeKind.COMPUTATION, operator_name="op", inputs=("a",), outputs=("b",)), depends_on=("input",))
    graph.add_node(ExecutionNode("n2", NodeKind.COMPUTATION, operator_name="op", inputs=("b",), outputs=("c",)), depends_on=("n1",))
    store = InMemoryArtifactStore()
    scheduler = Scheduler()

    assert scheduler.ready_nodes(graph, store) == []
    store.put(Artifact("a", "x", 1))
    assert [n.node_id for n in scheduler.ready_nodes(graph, store)] == ["n1"]
    graph.nodes["n1"].status = NodeStatus.COMPLETED
    store.put(Artifact("b", "x", 2))
    assert [n.node_id for n in scheduler.ready_nodes(graph, store)] == ["n2"]
    assert scheduler.unresolved_nodes(graph) == ("n2",)


@pytest.mark.parametrize("safe,valid", [(True, True), (False, True), (True, False)])
def test_operator_recovery_uses_detached_inputs_and_checks_outputs(tmp_path, safe, valid):
    from unittest.mock import Mock
    from framework.operator import OperatorRegistry, OperatorSpec, OperatorRole, SideEffectLevel
    from framework.runtime import GraphRuntime
    from framework.repair import RepairManager, RepairPolicy, RepairMode, RecoveryHooks, RepairStatus
    registry = OperatorRegistry()
    original = [1]
    def initial(inputs, config, context):
        inputs["in"].append(2)
        raise TimeoutError("mock transient failure")
    def repeated(inputs, config, context):
        assert inputs["in"] == [1]
        return {"out" if valid else "wrong": 3}
    operator = Mock(side_effect=[None])
    calls = [initial, repeated]
    operator.side_effect = lambda *args: calls.pop(0)(*args)
    registry.register(OperatorSpec("op", OperatorRole.TRANSFORM, "mock operation",
        side_effects=SideEffectLevel.NONE if safe else SideEffectLevel.IRREVERSIBLE), operator)
    manager = RepairManager(repository_root=str(tmp_path), policy=RepairPolicy(mode=RepairMode.LOCAL))
    outcomes = []
    def recover(failure, node, retry, permitted):
        outcome = manager.repair(failure, hooks=RecoveryHooks("op", lambda: "v1", retry, permitted))
        outcomes.append(outcome)
        return outcome
    runtime = GraphRuntime(registry, recovery_handler=recover)
    store = InMemoryArtifactStore()
    store.put(Artifact("in", "generic", original))
    node = ExecutionNode("n", NodeKind.COMPUTATION, "op", inputs=("in",), outputs=("out",))
    if safe and valid:
        runtime._execute_node(node, store, 1, None)
        assert store.get("out").value == 3
        assert node.failure is None
        assert outcomes[0].status is RepairStatus.RECOVERED
    else:
        with pytest.raises(TimeoutError):
            runtime._execute_node(node, store, 1, None)
        assert "out" not in store.ids()
        assert node.failure.failure_id == outcomes[0].failure.failure_id
    if safe:
        assert original == [1]
    assert operator.call_count == (2 if safe else 1)
