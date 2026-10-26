from agents.cohort.agent import CohortAgent
from agents.cohort.interaction import CohortInteractionAdapter
from agents.cohort.parser import CohortTaskParser
from interfaces.controller import AgentInteractionController


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        responses = [
            {
                "predicates": [{"predicate_id": "diabetes", "text": "patient has type 2 diabetes mellitus"}],
                "logic": "diabetes", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Initial cohort.",
            },
            {
                "requirements": [{"requirement_id": "diabetes", "kind": "condition",
                    "payload": {"term": "Type 2 diabetes mellitus"}, "polarity": "include", "required": True}],
                "assumptions": [],
            },
            {"merged_request": "patients with type 2 diabetes mellitus and chronic kidney disease"},
            {
                "predicates": [
                    {"predicate_id": "diabetes", "text": "patient has type 2 diabetes mellitus"},
                    {"predicate_id": "ckd", "text": "patient has chronic kidney disease"},
                ],
                "logic": "diabetes AND ckd", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Add CKD as mandatory.",
            },
            {
                "requirements": [
                    {"requirement_id": "diabetes", "kind": "condition",
                     "payload": {"term": "Type 2 diabetes mellitus"}, "polarity": "include", "required": True},
                    {"requirement_id": "ckd", "kind": "condition",
                     "payload": {"term": "Chronic kidney disease"}, "polarity": "include", "required": True},
                ],
                "assumptions": [],
            },
        ]
        return responses[self.calls - 1]


class Resolver:
    def resolve_condition(self, term):
        return term

    def resolve_medication(self, term):
        return term


class Repository:
    DATA = {
        "Type 2 diabetes mellitus": {1, 2, 3},
        "Chronic kidney disease": {2, 3},
    }

    def patients_with_condition(self, concept, payload=None):
        return set(self.DATA[concept])

    def patients_with_medication(self, concept, payload=None):
        return set()


def test_chat_creates_then_updates_graph():
    agent = CohortAgent(
        CohortTaskParser(FakeLLM(), correction_mode="full_regeneration"),
        Resolver(),
        Repository(),
    )
    session = agent.create_session()
    controller = AgentInteractionController(session, CohortInteractionAdapter())

    first = controller.handle_message("Find diabetes patients")
    second = controller.handle_message("Also require CKD")

    assert first.graph_version == 1
    assert second.graph_version == 2
    assert session.summary()["cohort_size"] == 2
    assert "reused" in second.messages[-1].content


class RobustFallbackLLM:
    def __init__(self):
        self.responses = [
            {
                "predicates": [{"predicate_id": "p1", "text": "patient has type 2 diabetes mellitus"}],
                "logic": "p1", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Initial cohort.",
            },
            {
                "requirements": [{"requirement_id": "p1", "kind": "condition",
                    "payload": {"term": "Type 2 diabetes mellitus"}, "polarity": "include", "required": True}],
                "assumptions": [],
            },
            {
                "actions": [{"action": "REMOVE", "target": "p99"}],
                "requires_clarification": False, "clarification_question": None,
            },
            {
                "actions": [{"action": "REMOVE", "target": "p99"}],
                "requires_clarification": False, "clarification_question": None,
            },
            {"merged_request": "patients with type 2 diabetes mellitus and chronic kidney disease"},
            {
                "predicates": [
                    {"predicate_id": "p1", "text": "patient has type 2 diabetes mellitus"},
                    {"predicate_id": "p2", "text": "patient has chronic kidney disease"},
                ],
                "logic": "p1 AND p2", "ambiguities": [], "requires_clarification": False,
                "clarification_question": None, "rationale": "Add CKD.",
            },
            {
                "requirements": [
                    {"requirement_id": "p1", "kind": "condition",
                     "payload": {"term": "Type 2 diabetes mellitus"}, "polarity": "include", "required": True},
                    {"requirement_id": "p2", "kind": "condition",
                     "payload": {"term": "Chronic kidney disease"}, "polarity": "include", "required": True},
                ],
                "assumptions": [],
            },
        ]
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def test_robust_failure_requires_explicit_full_regeneration_confirmation():
    from framework.correction import CorrectionMode
    from framework.pending_interaction import PendingInteractionKind

    llm = RobustFallbackLLM()
    parser = CohortTaskParser(llm, correction_mode=CorrectionMode.ROBUST_DELTA)
    agent = CohortAgent(parser, Resolver(), Repository())
    controller = AgentInteractionController(agent.create_session(), CohortInteractionAdapter())
    controller.handle_message("Find diabetes patients")

    failed = controller.handle_message("Also require CKD")
    assert controller.session.state.version == 1
    assert controller.session.state.pending_interaction.kind is PendingInteractionKind.CORRECTION_FALLBACK_CONFIRMATION
    assert "Use Full Regeneration" in failed.messages[-1].content
    assert len(llm.calls) == 4

    completed = controller.handle_message("use full regeneration")
    assert controller.session.state.version == 2
    assert controller.session.state.pending_interaction is None
    assert parser.correction_mode is CorrectionMode.ROBUST_DELTA
    assert controller.session.summary()["cohort_size"] == 2
    assert "correction_full_regeneration_confirmed" in completed.notifications
