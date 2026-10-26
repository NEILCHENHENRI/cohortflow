from types import MethodType, SimpleNamespace

from framework.pending_interaction import ContinuationKind, PendingInteraction
from framework.runtime import ExecutionMode
from interfaces.controller import AgentInteractionController
from presentation.models import InteractionResult


class _Session:
    def __init__(self, *, mode=ExecutionMode.SEQUENTIAL, pending=None):
        self.state = SimpleNamespace(version=1, pending_interaction=pending)
        self.runtime = SimpleNamespace(execution_mode=mode)
        self.last_failure = None
        self.last_repair_proposal = None
        self.last_repair_outcome = None
        self.analyzed = []

    def analyze_correction(self, text, on_progress=None):
        self.analyzed.append((text, self.state.version))
        return {"text": text, "version": self.state.version}


def test_deferred_corrections_materialize_fifo_against_latest_spec():
    session = _Session()
    controller = AgentInteractionController(session, adapter=SimpleNamespace())
    seen = []

    def preanalyzed(self, text, analysis, on_progress=None):
        seen.append((text, "pre", session.state.version, analysis["version"]))
        session.state.version = 2
        return InteractionResult(graph_changed=True, graph_version=2)

    def ordinary(self, interpretation_text, displayed_text, on_progress, *, user_message=None, preanalyzed=None):
        seen.append((interpretation_text, "fresh", session.state.version, None))
        session.state.version = 3
        return InteractionResult(graph_changed=True, graph_version=3)

    controller.handle_preanalyzed_correction = MethodType(preanalyzed, controller)
    controller._handle_correction = MethodType(ordinary, controller)

    result = controller._apply_deferred_corrections_fifo(
        ["C1", "C2"], [{"version": 1}]
    )

    assert seen == [
        ("C1", "pre", 1, 1),
        ("C2", "fresh", 2, None),
    ]
    assert result.graph_version == 3


def test_note_confirmation_sequential_queues_correction_without_interpreting():
    pending = PendingInteraction.note_execution_confirmation(
        continuation=ContinuationKind.INITIAL_EXECUTION,
        question="Proceed with notes?",
    )
    session = _Session(mode=ExecutionMode.SEQUENTIAL, pending=pending)
    controller = AgentInteractionController(session, adapter=SimpleNamespace())

    result = controller._answer_pending_interaction("also require CKD", on_progress=None)

    assert session.analyzed == []
    assert pending.deferred_corrections == ("also require CKD",)
    assert "correction_deferred_until_note_confirmation" in result.notifications


def test_note_confirmation_parallel_preinterprets_only_first_waiting_correction():
    pending = PendingInteraction.note_execution_confirmation(
        continuation=ContinuationKind.INITIAL_EXECUTION,
        question="Proceed with notes?",
    )
    session = _Session(mode=ExecutionMode.PARALLEL, pending=pending)
    controller = AgentInteractionController(session, adapter=SimpleNamespace())

    controller._answer_pending_interaction("C1", on_progress=None)
    controller._answer_pending_interaction("C2", on_progress=None)

    assert session.analyzed == [("C1", 1)]
    assert pending.deferred_corrections == ("C1", "C2")
    assert len(pending.deferred_correction_analyses) == 1
