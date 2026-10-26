from agents.cohort.agent import CohortAgent
from agents.cohort.parser import CohortTaskParser


class QueueLLM:
    def __init__(self):
        self.responses = [
            # Initial logical state.
            {
                "predicates": [
                    {"predicate_id": "diabetes", "text": "patient has type 2 diabetes"},
                ],
                "logic": "diabetes",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Initial cohort.",
            },
            # Initial predicate compilation.
            {
                "requirements": [
                    {
                        "requirement_id": "diabetes",
                        "kind": "condition",
                        "payload": {"term": "type 2 diabetes"},
                        "polarity": "include",
                        "required": True,
                    }
                ],
                "assumptions": [],
            },
            # Merge correction into a complete natural-language request.
            {"merged_request": "patients with type 2 diabetes or hypertension"},
            # Reparse using the same logical parser as an initial request.
            {
                "predicates": [
                    {"predicate_id": "diabetes", "text": "patient has type 2 diabetes"},
                    {"predicate_id": "hypertension", "text": "patient has hypertension"},
                ],
                "logic": "diabetes OR hypertension",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Hypertension is an alternative eligible population.",
            },
            # Verify old scheme + correction + reparsed scheme.
            # Compile the complete verified scheme before structured diffing.
            {
                "requirements": [
                    {
                        "requirement_id": "diabetes",
                        "kind": "condition",
                        "payload": {"term": "type 2 diabetes"},
                        "polarity": "include",
                        "required": True,
                    },
                    {
                        "requirement_id": "hypertension",
                        "kind": "condition",
                        "payload": {"term": "hypertension"},
                        "polarity": "include",
                        "required": True,
                    },
                ],
                "assumptions": [],
            },
        ]

    def generate_structured(self, **kwargs):
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class Resolver:
    def resolve_condition(self, term):
        return term

    def resolve_medication(self, term):
        return term


class Repository:
    def patients_with_condition(self, concept, payload=None):
        return {
            "type 2 diabetes": {1, 2},
            "hypertension": {3, 4},
        }[concept]

    def patients_with_medication(self, concept, payload=None):
        return set()

    def all_patients(self):
        return {1, 2, 3, 4}


def test_add_alternative_correction_adds_branch_and_recomputes_aggregation():
    llm = QueueLLM()
    agent = CohortAgent(CohortTaskParser(llm, correction_mode="full_regeneration"), Resolver(), Repository())
    session = agent.create_session()

    task = agent.parse_task("find patients with type 2 diabetes")
    session.initialize(task)
    session.execute()
    assert session.summary()["cohort_size"] == 2

    analysis = session.analyze_correction("include hypertension patients")
    assert analysis.task_delta.logic is not None
    assert analysis.task_delta.logic.kind == "or"
    assert [r.payload["term"] for r in analysis.task_delta.add_requirements] == [
        "hypertension"
    ]

    plan = session.plan_correction(analysis)
    report = session.stage_correction(analysis, plan)
    session.execute_staged()

    assert session.summary()["cohort_size"] == 4
    assert any(node.startswith("evaluate:requirement:") for node in report.added_nodes)
    assert "aggregate:cohort" in report.invalidated_nodes
    assert "deliver:summary" in report.invalidated_nodes


class AdultConstraintLLM:
    def __init__(self):
        self.responses = [
            # Initial OR cohort.
            {
                "predicates": [
                    {"predicate_id": "diabetes", "text": "patient has diabetes"},
                    {"predicate_id": "hypertension", "text": "patient has hypertension"},
                ],
                "logic": "diabetes OR hypertension",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Either condition is sufficient.",
            },
            {
                "requirements": [
                    {"requirement_id": "diabetes", "kind": "condition", "payload": {"term": "diabetes"}, "polarity": "include", "required": True},
                    {"requirement_id": "hypertension", "kind": "condition", "payload": {"term": "hypertension"}, "polarity": "include", "required": True},
                ],
                "assumptions": [],
            },
            # Correction: universal adult constraint must sit outside the OR group.
            {"merged_request": "patients who are adults and have either diabetes or hypertension"},
            {
                "predicates": [
                    {"predicate_id": "adult", "text": "patient is an adult"},
                    {"predicate_id": "diabetes", "text": "patient has diabetes"},
                    {"predicate_id": "hypertension", "text": "patient has hypertension"},
                ],
                "logic": "adult AND (diabetes OR hypertension)",
                "ambiguities": [],
                "requires_clarification": False,
                "clarification_question": None,
                "rationale": "Adult applies to every eligible patient.",
            },
            {
                "requirements": [
                    {"requirement_id": "adult", "kind": "patient", "payload": {"attribute": "age", "operator": ">=", "value": 18}, "polarity": "include", "required": True},
                    {"requirement_id": "diabetes", "kind": "condition", "payload": {"term": "diabetes"}, "polarity": "include", "required": True},
                    {"requirement_id": "hypertension", "kind": "condition", "payload": {"term": "hypertension"}, "polarity": "include", "required": True},
                ],
                "assumptions": [],
            },
        ]

    def generate_structured(self, **kwargs):
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class AdultRepository(Repository):
    def patients_with_condition(self, concept, payload=None):
        return {"diabetes": {1, 2}, "hypertension": {3, 4}}[concept]

    def patients_with_patient_attribute(self, payload):
        assert payload == {"attribute": "age", "operator": ">=", "value": 18}
        return {1, 3}


def test_universal_constraint_after_or_group_intersects_whole_existing_cohort():
    agent = CohortAgent(CohortTaskParser(AdultConstraintLLM(), correction_mode="full_regeneration"), Resolver(), AdultRepository())
    session = agent.create_session()

    task = agent.parse_task("find patients with diabetes or hypertension")
    session.initialize(task)
    session.execute()
    assert session.summary()["cohort_size"] == 4

    analysis = session.analyze_correction("enforce adults (age >= 18)")
    assert analysis.task_delta.logic is not None
    assert analysis.task_delta.logic.kind == "and"
    assert any(child.kind == "or" for child in analysis.task_delta.logic.children)

    plan = session.plan_correction(analysis)
    session.stage_correction(analysis, plan)
    session.execute_staged()

    assert session.summary()["cohort_size"] == 2
