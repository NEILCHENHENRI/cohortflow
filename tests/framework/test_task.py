import pytest

from framework.logic import LogicExpression
from framework.manual_edit import StructuredManualEditProvider, ManualEditForm

from framework.task import (
    Assumption,
    Deliverable,
    Objective,
    Requirement,
    TaskDelta,
    TaskSpec,
    apply_task_delta,
)


def base_task() -> TaskSpec:
    return TaskSpec(
        domain="demo",
        objective=Objective("find", "Find matching records"),
        deliverables=(
            Deliverable("records", "record_set"),
        ),
        requirements=(
            Requirement("r1", "condition", {"term": "A"}),
        ),
        context={"source": "db"},
        assumptions=(
            Assumption("a1", "Source is current"),
        ),
    )


def test_task_delta_add_remove_replace():
    task = base_task()
    delta = TaskDelta(
        add_requirements=(
            Requirement("r2", "condition", {"term": "B"}),
        ),
        replace_requirements=(
            Requirement("r1", "condition", {"term": "A2"}),
        ),
        context_updates={"snapshot": "v2"},
        metadata_updates={"language": "en"},
        remove_assumption_ids=("a1",),
    )

    updated = apply_task_delta(task, delta)

    assert [r.requirement_id for r in updated.requirements] == ["r1", "r2"]
    assert updated.requirements[0].payload["term"] == "A2"
    assert updated.context == {"source": "db", "snapshot": "v2"}
    assert updated.metadata["language"] == "en"
    assert updated.assumptions == ()


def test_duplicate_requirement_ids_rejected():
    with pytest.raises(ValueError):
        TaskSpec(
            domain="demo",
            objective=Objective("x", "y"),
            requirements=(
                Requirement("same", "a", {}),
                Requirement("same", "b", {}),
            ),
        )


class InventoryManualEditor(StructuredManualEditProvider):
    def build_form(self, task, *, version):
        return ManualEditForm("stock", "Stock", "Edit stock", version, ())

    def _validate_payload(self, kind, payload):
        if kind != "stock" or payload["quantity"] < 0:
            raise ValueError("Stock quantity must be non-negative")

    def _payload_from_submission(self, current, kind, submitted, prefix, issues):
        return {"quantity": submitted.get(f"{prefix}.quantity", current.payload["quantity"])}


def inventory_task():
    return TaskSpec(
        domain="inventory", objective=Objective("stock", "Available products"),
        requirements=(
            Requirement("a", "stock", {"quantity": 1}),
            Requirement("b", "stock", {"quantity": 2}),
        ),
        logic=LogicExpression.and_(
            LogicExpression.predicate("a", "First product"),
            LogicExpression.predicate("b", "Second product"),
        ),
    )


def test_generic_manual_editor_prunes_removed_leaves_without_mutating_source():
    original = inventory_task()
    before = original.to_dict()
    result = InventoryManualEditor().compile(original, {
        "requirements.a.remove": True,
        "requirements.b.quantity": 8,
    })
    assert result.valid
    updated = apply_task_delta(original, result.task_delta)
    assert updated.logic.kind == "predicate"
    assert updated.logic.predicate_id == "b"
    assert updated.logic.text == "Second product"
    assert updated.requirements == (Requirement("b", "stock", {"quantity": 8}),)
    assert original.to_dict() == before


@pytest.mark.parametrize("submitted,field", [
    ({"requirements.a.quantity": -1}, "requirements.a"),
    ({"requirements.a.remove": True, "requirements.b.remove": True}, "requirements"),
    ({"logic.expression": "a OR missing"}, "logic.expression"),
    ({"objective.description": " "}, "objective.description"),
])
def test_generic_manual_editor_rejects_invalid_edits_transactionally(submitted, field):
    original = inventory_task()
    before = original.to_dict()
    result = InventoryManualEditor().compile(original, submitted)
    assert not result.valid
    assert result.task_delta is None
    assert any(issue.field_id == field for issue in result.issues)
    assert original.to_dict() == before
