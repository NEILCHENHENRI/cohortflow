from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from agents.cohort.parser import CohortTaskParser
from framework.pending_interaction import PendingInteractionKind
from interfaces.controller import AgentInteractionController


class NoteLLM:
    def __init__(self):
        self.calls = 0
        self.think = False

    def generate_structured(self, **kwargs):
        self.calls += 1
        responses = [
            {
                "predicates": [{"predicate_id": "adherence", "text": "patient was non-adherent to prescribed medication"}],
                "logic": "adherence", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Note evidence required.",
            },
            {
                "requirements": [{
                    "requirement_id": "adherence", "kind": "note",
                    "payload": {"claim": "patient was non-adherent to prescribed medication"},
                    "polarity": "include", "required": True,
                }],
                "assumptions": [],
            },
        ]
        return responses[self.calls - 1]


class Resolver:
    pass


class Repository:
    source_label = "test"


class Matcher:
    think = True

    def patients_with_note(self, payload):
        raise AssertionError("note matching must not run before confirmation")


def test_initial_note_task_pauses_before_expensive_note_execution():
    agent = CohortAgent(
        parser=CohortTaskParser(NoteLLM()),
        resolver=Resolver(),
        repository=Repository(),
        note_matcher=Matcher(),
    )
    controller = AgentInteractionController(agent.create_session(), CohortInteractionAdapter())

    result = controller.handle_message("Find patients documented as non-adherent to medications")

    assert result.error is None
    assert "substantially longer" in result.messages[-1].content
    assert controller.session.state.pending_interaction.kind is PendingInteractionKind.NOTE_EXECUTION_CONFIRMATION
    assert controller.session.artifacts.contains("cohort_summary") is False
