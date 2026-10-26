from agents.cohort.parser import CohortTaskParser, _canonicalize_compiled_scheme
from framework.correction import CorrectionScope
from framework.logic import LogicExpression
from framework.task import Objective, Requirement, TaskSpec


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def merged(text):
    return {"merged_request": text}


def logic(predicates, expression, rationale="Parsed revised cohort."):
    return {
        "predicates": predicates, "logic": expression, "ambiguities": [],
        "requires_clarification": False, "clarification_question": None,
        "rationale": rationale,
    }


def req(rid, term):
    return {
        "requirement_id": rid, "kind": "condition", "payload": {"term": term},
        "polarity": "include", "required": True,
    }


def compiled(*requirements):
    return {"requirements": list(requirements), "assumptions": []}


def current_one():
    requirements = (Requirement("diabetes", "condition", {"term": "Diabetes"}),)
    logic_expr = LogicExpression.predicate("diabetes", "patient has diabetes")
    logic_expr, requirements = _canonicalize_compiled_scheme(logic_expr, requirements)
    return TaskSpec(domain="cohort_curation", objective=Objective("cohort", "Find patients"), requirements=requirements, logic=logic_expr)


def current_two(operator="and"):
    leaves = (
        LogicExpression.predicate("diabetes", "patient has diabetes"),
        LogicExpression.predicate("hypertension", "patient has hypertension"),
    )
    logic_expr = LogicExpression.and_(*leaves) if operator == "and" else LogicExpression.or_(*leaves)
    requirements = (
        Requirement("diabetes", "condition", {"term": "Diabetes"}),
        Requirement("hypertension", "condition", {"term": "Hypertension"}),
    )
    logic_expr, requirements = _canonicalize_compiled_scheme(logic_expr, requirements)
    return TaskSpec(domain="cohort_curation", objective=Objective("cohort", "Find patients"), requirements=requirements, logic=logic_expr)


def test_correction_regenerates_full_state_then_diffs_structured_fields():
    preds = [
        {"predicate_id": "diabetes", "text": "patient has diabetes"},
        {"predicate_id": "hypertension", "text": "patient has hypertension"},
    ]
    llm = QueueLLM([
        merged("patients with diabetes or hypertension"),
        logic(preds, "diabetes OR hypertension"),
        compiled(req("diabetes", "Diabetes"), req("hypertension", "Hypertension")),
    ])

    result = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current_one(), "also include hypertension patients")

    assert result.intent.scope is CorrectionScope.GLOBAL
    assert result.task_delta.logic is not None and result.task_delta.logic.kind == "or"
    assert [r.payload["term"] for r in result.task_delta.add_requirements] == ["Hypertension"]
    assert result.task_delta.replace_requirements == ()
    assert len(llm.calls) == 3
    assert "CURRENT COHORT LOGIC" in llm.calls[0]["user_prompt"]
    assert "LATEST CORRECTION" in llm.calls[0]["user_prompt"]
    # The merged natural-language request from pass 1 is what gets reparsed
    # in pass 2 — corrections reuse the exact initial-request parser rather
    # than patching structured state textually.
    assert llm.calls[1]["user_prompt"] == "patients with diabetes or hypertension"


def test_no_verifier_can_overwrite_a_valid_or_candidate():
    preds = [
        {"predicate_id": "diabetes", "text": "patient has diabetes"},
        {"predicate_id": "hypertension", "text": "patient has hypertension"},
    ]
    llm = QueueLLM([
        merged("patients with diabetes or hypertension"),
        logic(preds, "diabetes OR hypertension"),
        compiled(req("diabetes", "Diabetes"), req("hypertension", "Hypertension")),
    ])

    result = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current_one(), "include hypertension patients")

    assert result.task_delta.logic is not None and result.task_delta.logic.kind == "or"
    assert len(llm.calls) == 3


def test_pure_regrouping_only_changes_logic_after_structured_diff():
    preds = [
        {"predicate_id": "diabetes", "text": "patient has diabetes"},
        {"predicate_id": "hypertension", "text": "patient has hypertension"},
    ]
    llm = QueueLLM([
        merged("patients with either diabetes or hypertension"),
        logic(preds, "diabetes OR hypertension"),
        compiled(req("diabetes", "Diabetes"), req("hypertension", "Hypertension")),
    ])

    result = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current_two("and"), "Either diabetes or hypertension is sufficient.")

    assert result.task_delta.logic is not None and result.task_delta.logic.kind == "or"
    # Regrouping AND -> OR must not replace, add, or remove the underlying
    # requirement leaves. Only the Boolean structure over them changes.
    assert set(result.task_delta.logic.predicate_ids()) == set(
        current_two("and").logic.predicate_ids()
    )
    assert result.task_delta.add_requirements == ()
    assert result.task_delta.replace_requirements == ()
    assert result.task_delta.remove_requirement_ids == ()
    assert len(llm.calls) == 3


def test_structured_equivalence_recovers_old_id_after_wording_and_id_drift():
    preds = [
        {"predicate_id": "p7", "text": "diagnosed with diabetes"},
        {"predicate_id": "hypertension", "text": "patient has hypertension"},
    ]
    llm = QueueLLM([
        merged("patients diagnosed with diabetes or with hypertension"),
        logic(preds, "p7 OR hypertension"),
        compiled(req("p7", "Diabetes"), req("hypertension", "Hypertension")),
    ])

    result = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current_one(), "also include hypertension")

    assert result.task_delta.logic is not None
    assert len(set(result.task_delta.logic.predicate_ids())) == 2
    assert [r.payload["term"] for r in result.task_delta.add_requirements] == ["Hypertension"]
    assert result.task_delta.replace_requirements == ()
