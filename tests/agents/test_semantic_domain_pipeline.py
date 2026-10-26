"""Tests for the single-pass full-state cohort semantic pipeline."""

from agents.cohort.parser import CohortTaskParser, _canonicalize_compiled_scheme
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


def logic_result(predicates, logic, *, rationale="Parsed Boolean semantics."):
    return {
        "predicates": predicates,
        "logic": logic,
        "ambiguities": [],
        "requires_clarification": False,
        "clarification_question": None,
        "rationale": rationale,
    }


def predicate(pid, text):
    return {"predicate_id": pid, "text": text}


def compiled(*requirements):
    return {"requirements": list(requirements), "assumptions": []}


def requirement(pid, kind, payload):
    return {
        "requirement_id": pid,
        "kind": kind,
        "payload": payload,
        "polarity": "include",
        "required": True,
    }


def merged(text):
    return {"merged_request": text}


def _current_or_task():
    reqs = (
        Requirement("d", "condition", {"term": "diabetes"}),
        Requirement("h", "condition", {"term": "hypertension"}),
    )
    logic = LogicExpression.or_(
        LogicExpression.predicate("d", "patient has diabetes"),
        LogicExpression.predicate("h", "patient has hypertension"),
    )
    logic, reqs = _canonicalize_compiled_scheme(logic, reqs)
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort", "Find patients"),
        requirements=reqs,
        logic=logic,
    )


def test_simple_initial_request_uses_flat_logic_then_predicate_compilation():
    llm = QueueLLM([
        logic_result([predicate("diabetes", "patient has diabetes")], "diabetes"),
        compiled(requirement("diabetes", "condition", {"term": "diabetes"})),
    ])

    task = CohortTaskParser(llm, correction_mode="full_regeneration").parse_task("Find patients with diabetes")

    assert len(llm.calls) == 2
    assert task.logic is not None and task.logic.kind == "predicate"
    assert task.requirements[0].payload["term"] == "diabetes"
    assert "Boolean relationships are already fixed" in llm.calls[1]["system_prompt"]


def test_complicated_initial_request_supports_nested_boolean_logic_in_one_semantic_pass():
    predicates = [
        predicate("adult", "patient is at least 40 years old"),
        predicate("diabetes", "patient has type 2 diabetes"),
        predicate("ckd", "patient has chronic kidney disease"),
        predicate("hypertension", "patient has hypertension"),
        predicate("malignancy", "patient has active malignancy"),
    ]
    llm = QueueLLM([
        logic_result(
            predicates,
            "adult AND ((diabetes AND ckd) OR hypertension) AND NOT malignancy",
        ),
        compiled(
            requirement("adult", "patient", {"attribute": "age", "operator": ">=", "value": 40}),
            requirement("diabetes", "condition", {"term": "type 2 diabetes"}),
            requirement("ckd", "condition", {"term": "chronic kidney disease"}),
            requirement("hypertension", "condition", {"term": "hypertension"}),
            requirement("malignancy", "condition", {"term": "active malignancy"}),
        ),
    ])

    task = CohortTaskParser(llm, correction_mode="full_regeneration").parse_task(
        "Adults with either both type 2 diabetes and CKD, or hypertension, and no active malignancy"
    )

    assert len(llm.calls) == 2
    assert task.logic is not None and task.logic.kind == "and"
    assert task.logic.contains_not()
    assert any(child.kind == "or" for child in task.logic.children)
    assert "nested combinations are allowed at any depth" in llm.calls[0]["system_prompt"]


def test_correction_merge_preserves_or_group_and_adds_universal_and_constraint():
    current = _current_or_task()
    predicates = [
        predicate("d", "patient has diabetes"),
        predicate("h", "patient has hypertension"),
        predicate("adult", "patient is an adult"),
    ]
    llm = QueueLLM([
        merged("patients who are adults and have either diabetes or hypertension"),
        logic_result(predicates, "adult AND (d OR h)"),
        compiled(
            requirement("d", "condition", {"term": "diabetes"}),
            requirement("h", "condition", {"term": "hypertension"}),
            requirement("adult", "patient", {"attribute": "age", "operator": ">=", "value": 18}),
        ),
    ])

    delta = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current, "enforce adults (age >= 18)").task_delta

    assert len(llm.calls) == 3
    assert delta.logic is not None and delta.logic.kind == "and"
    assert any(child.kind == "or" for child in delta.logic.children)
    assert len(delta.add_requirements) == 1
    assert delta.add_requirements[0].kind == "patient"
    assert "CURRENT COHORT LOGIC" in llm.calls[0]["user_prompt"]
    assert "patient has diabetes OR patient has hypertension" in llm.calls[0]["user_prompt"]


def test_include_population_prompt_is_explicitly_open_to_or_broadening():
    current = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort", "Find patients"),
        requirements=(Requirement("d", "condition", {"term": "diabetes"}),),
        logic=LogicExpression.predicate("d", "patient has diabetes"),
    )
    llm = QueueLLM([
        merged("patients with diabetes or hypertension"),
        logic_result([
            predicate("d", "patient has diabetes"),
            predicate("h", "patient has hypertension"),
        ], "d OR h"),
        compiled(
            requirement("d", "condition", {"term": "diabetes"}),
            requirement("h", "condition", {"term": "hypertension"}),
        ),
    ])

    delta = CohortTaskParser(llm, correction_mode="full_regeneration").interpret_correction(current, "include hypertension patients").task_delta

    assert delta.logic is not None and delta.logic.kind == "or"
    assert "diabetes OR hypertension" in llm.calls[0]["system_prompt"]


