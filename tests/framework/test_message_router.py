"""Tests for framework.interaction.MessageRouter.

Consolidates what used to be three files that each tested one routing
scenario in isolation (pending-correction priority, manual-fallback
commands, and pending-resolution/inspect routing).
"""

from framework.graph import ExecutionGraph
from framework.interaction import MessageRoute, MessageRouter
from framework.operator import OperatorRegistry
from framework.pending_interaction import PendingInteraction
from framework.resolution import PendingResolution, ResolutionRequest
from framework.session import AgentState, SteerableSession
from framework.task import Objective, TaskSpec


class Dummy:
    domain = "x"

    def operator_registry(self):
        return OperatorRegistry()


def session(state=None):
    value = SteerableSession(Dummy())
    value.state = state or AgentState(
        TaskSpec("x", Objective("x", "x")),
        ExecutionGraph(),
    )
    return value


def test_no_active_session_routes_to_initial_task():
    value = SteerableSession(Dummy())
    assert MessageRouter().route("Find diabetes patients", value) is MessageRoute.INITIAL_TASK


def test_pending_resolution_takes_priority_over_everything_else():
    current = session().state
    current.pending_resolution = PendingResolution(
        resolution_id="r",
        request=ResolutionRequest(
            resolver_id="test",
            error_code="ambiguous",
            message="ambiguous",
            recoverable=True,
        ),
        question="Which concept?",
        graph_version=2,
    )
    route = MessageRouter().route("Hypertension stage 1", session(current))
    assert route is MessageRoute.ANSWER_PENDING


def test_pending_correction_clarification_takes_priority():
    current = AgentState(
        task=TaskSpec("x", Objective("x", "x")),
        graph=ExecutionGraph(),
        pending_interaction=PendingInteraction.correction_clarification(
            original_message="change patient",
            question="Which patient?",
        ),
    )
    assert MessageRouter().route("patient 12", session(current)) is MessageRoute.ANSWER_PENDING


def test_fallback_aliases_are_deterministic():
    router = MessageRouter()
    for command in ("/fallback", " /SOS ", "/incorrigible please", "/unsalvageable"):
        assert router.route(command, session()) is MessageRoute.ENTER_MANUAL_FALLBACK


def test_plain_word_containing_a_fallback_alias_does_not_trigger_fallback():
    assert (
        MessageRouter().route("this output is unsalvageable", session())
        is MessageRoute.MODIFY_TASK
    )


def test_inspect_and_control_prefixes_are_recognized():
    router = MessageRouter()
    value = session()
    assert router.route("why was patient 1 excluded?", value) is MessageRoute.INSPECT_RESULT
    assert router.route("pause", value) is MessageRoute.CONTROL_EXECUTION


def test_unrecognized_message_falls_back_to_modify_task():
    assert MessageRouter().route("also include CKD", session()) is MessageRoute.MODIFY_TASK


def test_initial_parse_clarification_routes_as_pending_before_session_state_exists():
    from framework.pending_interaction import PendingInteraction

    class Session:
        state = None
        pending_initial_interaction = PendingInteraction.initial_parse_clarification(
            original_message="ambiguous request",
            question="What did you mean?",
            rationale="Two materially different interpretations remain.",
        )

    assert MessageRouter().route("restated complete request", Session()) is MessageRoute.ANSWER_PENDING
