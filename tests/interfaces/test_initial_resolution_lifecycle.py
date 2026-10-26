from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from agents.cohort.parser import CohortTaskParser
from framework.resolution import (
    ResolutionAction,
    ResolutionDecision,
    ResolutionOption,
    ResolutionRequest,
    ResolutionRequired,
)
from infrastructure.omop.concepts import OMOPConcept
from interfaces.controller import AgentInteractionController


class TaskLLM:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "predicates": [{"predicate_id": "diabetes", "text": "patient has type 2 diabetes"}],
                "logic": "diabetes",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Initial cohort.",
            }
        return {
            "requirements": [{
                "requirement_id": "diabetes",
                "kind": "condition",
                "payload": {"term": "type 2 diabetes"},
                "polarity": "include",
                "required": True,
            }],
            "assumptions": [],
        }


class Resolver:
    def __init__(self):
        self.selected = False

    def resolve_condition(self, term):
        if self.selected:
            return OMOPConcept(
                201826,
                "Type 2 diabetes mellitus",
                "Condition",
                "SNOMED",
            )
        raise ResolutionRequired(
            ResolutionRequest(
                resolver_id="omop_concept",
                error_code="no_exact_concept_match",
                message="ambiguous",
                recoverable=True,
                options=(
                    ResolutionOption(
                        "201826",
                        "Type 2 diabetes mellitus",
                    ),
                ),
            ),
            question="Which concept?",
        )

    def resolve_medication(self, term):
        raise AssertionError

    def record_human_selection(self, **kwargs):
        self.selected = True


class Repository:
    def patients_with_condition(self, concept, payload=None):
        return {1, 2}

    def patients_with_medication(self, concept, payload=None):
        return set()


class AnswerResolver:
    def resolve_human_answer(self, pending, response):
        return ResolutionDecision(
            action=ResolutionAction.SELECT_OPTION,
            confidence=1.0,
            rationale="Exact answer.",
            option_id=pending.options[0].option_id,
        )


def test_initial_resolution_resumes_without_staged_update():
    resolver = Resolver()
    agent = CohortAgent(
        CohortTaskParser(TaskLLM()),
        resolver,
        Repository(),
    )
    session = agent.create_session()
    controller = AgentInteractionController(
        session,
        CohortInteractionAdapter(),
        AnswerResolver(),
    )

    first = controller.handle_message("find type 2 diabetes")
    assert first.error is None
    assert session.state.pending_resolution is not None
    assert session.state.staged is None

    second = controller.handle_message("mellitus")
    assert second.error is None
    assert session.state.pending_resolution is None
    assert session.state.version == 1
    assert session.summary()["cohort_size"] == 2
