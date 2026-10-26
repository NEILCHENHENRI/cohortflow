from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from interfaces.controller import AgentInteractionController
from agents.cohort.parser import CohortTaskParser
from framework.resolution import (
    ResolutionRequired,
    ResolutionRequest,
    ResolutionOption,
)
from infrastructure.llm.recovery import LLMSemanticExceptionResolver
from infrastructure.omop.concepts import OMOPConcept


class FakeParserLLM:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        responses = [
            {
                "predicates": [{"predicate_id": "diabetes", "text": "patient has diabetes"}],
                "logic": "diabetes", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Initial cohort.",
            },
            {
                "requirements": [{"requirement_id": "diabetes", "kind": "condition",
                    "payload": {"term": "diabetes"}, "polarity": "include", "required": True}],
                "assumptions": [],
            },
            {"merged_request": "patients with diabetes or hypertension"},
            {
                "predicates": [
                    {"predicate_id": "diabetes", "text": "patient has diabetes"},
                    {"predicate_id": "hypertension", "text": "patient has hypertension"},
                ],
                "logic": "diabetes OR hypertension", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Include hypertension as an alternative.",
            },
            {
                "requirements": [
                    {"requirement_id": "diabetes", "kind": "condition",
                     "payload": {"term": "diabetes"}, "polarity": "include", "required": True},
                    {"requirement_id": "hypertension", "kind": "condition",
                     "payload": {"term": "hypertension"}, "polarity": "include", "required": True},
                ],
                "assumptions": [],
            },
        ]
        return responses[self.calls - 1]


class Resolver:
    def __init__(self):
        self.override = None

    def resolve_condition(self, term):
        if term == "diabetes":
            return OMOPConcept(
                1, "Diabetes", "Condition", "SNOMED"
            )
        if self.override is not None:
            return OMOPConcept(
                int(self.override),
                "Hypertension stage 1",
                "Condition",
                "SNOMED",
                resolution_source="human",
            )
        request = ResolutionRequest(
            resolver_id="omop_concept",
            error_code="no_exact_concept_match",
            message="No exact match",
            recoverable=True,
            input_payload={
                "term": "hypertension",
                "resolution_key": {
                    "operation": "resolve_omop_concept",
                    "domain": "Condition",
                    "normalized_input": "hypertension",
                    "scope_id": "session",
                },
            },
            options=(
                ResolutionOption(
                    "101",
                    "Hypertension stage 1",
                    {"concept_id": 101, "domain_id": "Condition"},
                ),
                ResolutionOption(
                    "102",
                    "Hypertension stage 2",
                    {"concept_id": 102, "domain_id": "Condition"},
                ),
            ),
        )
        raise ResolutionRequired(
            request, question="Which hypertension concept?"
        )

    def resolve_medication(self, term):
        raise AssertionError

    def record_human_selection(
        self,
        *,
        pending_context,
        options,
        rationale,
    ):
        self.override = options[0].option_id


class Repository:
    def patients_with_condition(self, concept, payload=None):
        if concept.concept_id == 1:
            return {1, 2, 3}
        return {2, 3}

    def patients_with_medication(self, concept, payload=None):
        return set()


class UnusedLLM:
    def generate_structured(self, **kwargs):
        raise AssertionError("Exact pending answer should not call the LLM")


def test_pending_answer_resumes_and_commits_staged_version():
    resolver = Resolver()
    agent = CohortAgent(
        CohortTaskParser(FakeParserLLM(), correction_mode="full_regeneration"),
        resolver,
        Repository(),
    )
    session = agent.create_session()
    controller = AgentInteractionController(
        session, CohortInteractionAdapter(),
        LLMSemanticExceptionResolver(UnusedLLM()),
    )

    initial = controller.handle_message("Find diabetes patients")
    assert initial.graph_version == 1

    paused = controller.handle_message("Also include hypertension")
    assert paused.error is None
    assert session.state.version == 1
    assert session.state.pending_resolution is not None
    assert session.state.staged is not None

    resumed = controller.handle_message("Hypertension stage 1")
    assert resumed.error is None
    assert session.state.version == 2
    assert session.state.pending_resolution is None
    assert session.state.staged is None
    assert session.summary()["cohort_size"] == 3


def test_none_of_candidates_keeps_pending_task_recoverable():
    resolver = Resolver()
    agent = CohortAgent(
        CohortTaskParser(FakeParserLLM(), correction_mode="full_regeneration"),
        resolver,
        Repository(),
    )
    session = agent.create_session()
    controller = AgentInteractionController(
        session, CohortInteractionAdapter(),
        LLMSemanticExceptionResolver(UnusedLLM()),
    )

    controller.handle_message("Find diabetes patients")
    controller.handle_message("Also include hypertension")
    assert session.state.pending_resolution is not None
    assert session.state.staged is not None
    version = session.state.version

    result = controller.handle_message("none of these")

    assert result.error is None
    assert session.state.version == version
    assert session.state.pending_resolution is not None
    assert session.state.staged is not None
    assert "refine" in result.messages[-1].content.lower()
