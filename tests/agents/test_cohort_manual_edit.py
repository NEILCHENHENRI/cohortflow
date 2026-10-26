from agents.cohort.manual_edit import CohortManualEditProvider
from framework.logic import LogicExpression
from framework.task import Objective, Requirement, TaskSpec, apply_task_delta


def task():
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort", "Find diabetes patients"),
        requirements=(
            Requirement(
                "diabetes",
                "condition",
                {"term": "Type 2 diabetes mellitus"},
            ),
            Requirement(
                "hypertension",
                "condition",
                {"term": "Hypertension stage 1"},
            ),
        ),
        logic=LogicExpression.and_(
            LogicExpression.predicate("diabetes", "patient has type 2 diabetes mellitus"),
            LogicExpression.predicate("hypertension", "patient has hypertension stage 1"),
        ),
    )


def test_manual_fallback_changes_nested_logic_and_term():
    provider = CohortManualEditProvider()
    result = provider.compile(
        task(),
        {
            "objective.description": "Find diabetes or hypertension",
            "logic.expression": "diabetes OR hypertension",
            "requirements.diabetes.kind": "condition",
            "requirements.diabetes.polarity": "include",
            "requirements.diabetes.required": True,
            "requirements.diabetes.payload.term": "Type 2 diabetes mellitus",
            "requirements.diabetes.remove": False,
            "requirements.hypertension.kind": "condition",
            "requirements.hypertension.polarity": "include",
            "requirements.hypertension.required": True,
            "requirements.hypertension.payload.term": "Hypertension",
            "requirements.hypertension.remove": False,
        },
    )
    assert result.valid
    updated = apply_task_delta(task(), result.task_delta)
    assert updated.logic is not None
    assert updated.logic.kind == "or"
    assert updated.requirements[1].payload["term"] == "Hypertension"


def test_manual_fallback_rejects_unknown_predicate_in_logic():
    provider = CohortManualEditProvider()
    result = provider.compile(
        task(),
        {
            "logic.expression": "diabetes OR unknown",
        },
    )
    assert not result.valid
    assert any(issue.field_id == "logic.expression" for issue in result.issues)
