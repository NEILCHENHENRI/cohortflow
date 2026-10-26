import pytest

from framework.logic import LogicExpression, assign_logic_node_ids
from framework.task import Objective, Requirement, TaskDelta, TaskSpec, apply_task_delta
from framework.task_delta import (
    AddPredicateAct, AddRelationAct, AttachmentSpec, ChangeGroupOperatorAct,
    MovePredicateAct, RemovePredicateAct, RemoveRelationAct, ReplaceLogicSubtreeAct,
    ReplacePredicateAct, SetPredicateNegationAct, SetResolutionAct, UpdateAggregationAct,
    UpdatePredicateFieldsAct, UpdateRelationAct,
)


def req(rid, kind="condition", payload=None):
    return Requirement(rid, kind, payload or {"term": rid})


def task(logic=None, relations=None):
    requirements = (req("p1"), req("p2"), req("p3"))
    logic = logic or LogicExpression.and_(
        LogicExpression.predicate("p1", "A"),
        LogicExpression.or_(
            LogicExpression.predicate("p2", "B"),
            LogicExpression.predicate("p3", "C"),
        ),
    )
    return TaskSpec(
        "cohort_curation", Objective("cohort_retrieval", "test"),
        requirements=requirements,
        logic=assign_logic_node_ids(logic),
        context={"relations": relations or []},
    )


def replace_task_requirement(task_spec, rid, payload, *, kind):
    requirements = tuple(
        Requirement(r.requirement_id, kind, payload, polarity=r.polarity) if r.requirement_id == rid else r
        for r in task_spec.requirements
    )
    return TaskSpec(
        task_spec.domain, task_spec.objective, deliverables=task_spec.deliverables,
        requirements=requirements, logic=task_spec.logic, context=task_spec.context,
        assumptions=task_spec.assumptions, metadata=task_spec.metadata,
        original_request=task_spec.original_request,
    )


def group_by_kind(logic, kind):
    if logic.kind == kind:
        return logic
    for child in logic.children:
        if child.kind != "predicate":
            found = group_by_kind(child, kind)
            if found:
                return found
    return None


def test_delta_01_field_edit_preserves_logic_and_refs():
    original = task()
    before = original.logic.to_dict()
    updated = apply_task_delta(original, TaskDelta(acts=(
        UpdatePredicateFieldsAct("p1", {"count_min": 2}),
    )))
    assert updated.requirements[0].payload["count_min"] == 2
    assert updated.logic.to_dict() == before


def test_delta_02_add_predicate_to_global_and():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(
        AddPredicateAct(req("p4", "condition", {"term": "CKD"}), (AttachmentSpec("root", "and_child"),)),
    )))
    assert len(updated.requirements) == 4
    assert updated.logic.kind == "and"
    assert len(updated.logic.children) == 3


def test_delta_03_add_predicate_to_local_or_group():
    original = task()
    local_or = group_by_kind(original.logic, "or")
    updated = apply_task_delta(original, TaskDelta(acts=(
        AddPredicateAct(req("p4", "condition", {"term": "bronchitis"}), (AttachmentSpec(local_or.node_id, "or_child"),)),
    )))
    new_or = group_by_kind(updated.logic, "or")
    assert len(new_or.children) == 3
    assert set(new_or.predicate_ids()) == {"p2", "p3", next(r.requirement_id for r in updated.requirements if r.payload.get("term") == "bronchitis")}


def test_delta_04_add_predicate_to_existing_relation_scope():
    original = task(relations=[{"relation_id": "r1", "relation": "same_admission", "members": ["p1", "p2"]}])
    updated = apply_task_delta(original, TaskDelta(acts=(
        AddPredicateAct(req("p4", "medication", {"term": "lactulose"}), (
            AttachmentSpec("root", "and_child"),
            AttachmentSpec("r1", "relation_member"),
        )),
    )))
    relation = updated.context["relations"][0]
    assert len(relation["members"]) == 3
    assert relation["relation_id"] == "r1"


def test_delta_05_remove_predicate_cleans_logic_and_dangling_relation():
    original = task(relations=[{
        "relation_id": "r1", "relation": "before",
        "left_requirement_id": "p2", "right_requirement_id": "p3",
        "max_gap_hours": 4, "same_admission": True,
    }])
    updated = apply_task_delta(original, TaskDelta(acts=(RemovePredicateAct("p2"),)))
    assert "p2" not in updated.logic.predicate_ids()
    assert updated.context["relations"] == []
    assert updated.logic.kind == "and"


def test_delta_06_change_only_local_group_operator():
    original = task()
    local_or = group_by_kind(original.logic, "or")
    root_id = original.logic.node_id
    updated = apply_task_delta(original, TaskDelta(acts=(ChangeGroupOperatorAct(local_or.node_id, "and"),)))
    assert updated.logic.node_id == root_id
    assert updated.logic.kind == "and"
    assert updated.logic.children[1].kind == "and"
    assert updated.logic.children[1].node_id == local_or.node_id


def test_delta_07_replace_logic_subtree_preserves_outer_structure():
    original = task()
    local_or = group_by_kind(original.logic, "or")
    replacement = LogicExpression.and_(
        LogicExpression.predicate("p2", "B"),
        LogicExpression.predicate("p3", "C"),
    )
    updated = apply_task_delta(original, TaskDelta(acts=(ReplaceLogicSubtreeAct(local_or.node_id, replacement),)))
    assert updated.logic.kind == "and"
    assert updated.logic.children[1].kind == "and"
    assert updated.logic.children[1].node_id == local_or.node_id


def test_delta_08_move_predicate_from_local_or_to_root_and():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(MovePredicateAct("p2", AttachmentSpec("root", "and_child")),)))
    assert updated.logic.kind == "and"
    assert set(updated.logic.predicate_ids()) == {"p1", "p2", "p3"}
    # Removing p2 collapses OR(p2,p3) to p3 before p2 is reattached.
    assert len(updated.logic.children) == 3


def test_delta_09_add_new_before_relation_between_existing_predicates():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(AddRelationAct({
        "relation": "before", "left_requirement_id": "p2", "right_requirement_id": "p3",
        "max_gap_hours": 4, "same_admission": True,
    }),)))
    relation = updated.context["relations"][0]
    assert relation["relation_id"] == "r1"
    assert relation["left_requirement_id"] == "p2"
    assert relation["right_requirement_id"] == "p3"


def test_delta_10_reverse_relation_and_change_bound_without_touching_logic():
    original = task(relations=[{
        "relation_id": "r1", "relation": "before",
        "left_requirement_id": "p2", "right_requirement_id": "p3",
        "max_gap_hours": 24, "same_admission": True,
    }])
    before_logic = original.logic.to_dict()
    updated = apply_task_delta(original, TaskDelta(acts=(UpdateRelationAct("r1", {
        "left_requirement_id": "p3", "right_requirement_id": "p2", "max_gap_hours": 4,
    }),)))
    relation = updated.context["relations"][0]
    assert (relation["left_requirement_id"], relation["right_requirement_id"], relation["max_gap_hours"]) == ("p3", "p2", 4)
    assert updated.logic.to_dict() == before_logic


def test_delta_11_multi_act_transaction():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(
        RemovePredicateAct("p3"),
        UpdatePredicateFieldsAct("p1", {"count_min": 2}),
        AddRelationAct({"relation": "before", "left_requirement_id": "p1", "right_requirement_id": "p2", "max_gap_hours": 8, "same_admission": False}),
    )))
    assert len(updated.requirements) == 2
    assert next(r for r in updated.requirements if r.requirement_id == "p1").payload["count_min"] == 2
    assert updated.context["relations"][0]["relation"] == "before"


def test_delta_12_invalid_act_is_transactional_and_leaves_original_unchanged():
    original = task()
    snapshot = original.to_dict()
    with pytest.raises(ValueError, match="unknown logic group ref"):
        apply_task_delta(original, TaskDelta(acts=(
            AddPredicateAct(req("p4"), (AttachmentSpec("g999", "or_child"),)),
        )))
    assert original.to_dict() == snapshot


def test_delta_13_resolution_override_changes_context_not_semantics():
    original = task()
    before = original.logic.to_dict()
    updated = apply_task_delta(original, TaskDelta(acts=(
        SetResolutionAct("p1", {"concept_ids": [123]}),
    )))
    assert updated.logic.to_dict() == before
    assert updated.context["resolution_overrides"]["p1"] == {"concept_ids": [123]}


def test_delta_14_remove_relation_only_preserves_predicates():
    original = task(relations=[{
        "relation_id": "r1", "relation": "before", "left_requirement_id": "p1", "right_requirement_id": "p2",
        "max_gap_hours": None, "same_admission": False,
    }])
    updated = apply_task_delta(original, TaskDelta(acts=(RemoveRelationAct("r1"),)))
    assert updated.context["relations"] == []
    assert updated.logic.predicate_ids() == original.logic.predicate_ids()


def test_delta_15_replace_predicate_preserves_ref_and_structural_position():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(
        ReplacePredicateAct("p2", req("replacement", "condition", {"term": "hypertension"})),
    )))
    replacement = next(r for r in updated.requirements if r.requirement_id == "p2")
    assert replacement.payload["term"] == "hypertension"
    assert updated.logic.predicate_ids() == original.logic.predicate_ids()


def test_delta_16_update_fields_can_add_and_remove_constraints_without_touching_structure():
    original = task()
    original = replace_task_requirement(original, "p1", {"term": "furosemide", "route": "oral", "frequency": "daily"}, kind="medication")
    before = original.logic.to_dict()
    updated = apply_task_delta(original, TaskDelta(acts=(
        UpdatePredicateFieldsAct("p1", {"dose_value": 40, "dose_unit": "mg"}, remove_fields=("frequency",)),
    )))
    payload = next(r.payload for r in updated.requirements if r.requirement_id == "p1")
    assert payload == {"term": "furosemide", "route": "oral", "dose_value": 40, "dose_unit": "mg"}
    assert updated.logic.to_dict() == before


def test_delta_17_predicate_negation_round_trip_is_local():
    original = task()
    excluded = apply_task_delta(original, TaskDelta(acts=(SetPredicateNegationAct("p2", True),)))
    assert next(r for r in excluded.requirements if r.requirement_id == "p2").polarity == "exclude"
    assert "p2" in excluded.logic.predicate_ids()
    restored = apply_task_delta(excluded, TaskDelta(acts=(SetPredicateNegationAct("p2", False),)))
    assert next(r for r in restored.requirements if r.requirement_id == "p2").polarity == "include"
    assert restored.logic.predicate_ids() == original.logic.predicate_ids()


def test_delta_18_aggregation_update_preserves_requirement_identity():
    original = task()
    updated = apply_task_delta(original, TaskDelta(acts=(
        UpdateAggregationAct("p3", {"count_min": 2, "aggregation": "latest"}),
    )))
    requirement = next(r for r in updated.requirements if r.requirement_id == "p3")
    assert requirement.payload["count_min"] == 2
    assert requirement.payload["aggregation"] == "latest"
    assert updated.logic.predicate_ids() == original.logic.predicate_ids()


def test_delta_19_relation_type_endpoint_and_scope_can_change_without_leaf_drift():
    original = task(relations=[{
        "relation_id": "r1", "relation": "before",
        "left_requirement_id": "p2", "right_requirement_id": "p3",
        "max_gap_hours": 24, "same_admission": False,
    }])
    before = original.logic.to_dict()
    updated = apply_task_delta(original, TaskDelta(acts=(UpdateRelationAct("r1", {
        "relation": "after", "left_requirement_id": "p1",
        "right_requirement_id": "p2", "max_gap_hours": 6, "same_admission": True,
    }),)))
    relation = updated.context["relations"][0]
    assert relation["relation"] == "after"
    assert relation["left_requirement_id"] == "p1"
    assert relation["right_requirement_id"] == "p2"
    assert relation["max_gap_hours"] == 6
    assert relation["same_admission"] is True
    assert updated.logic.to_dict() == before


def test_delta_20_preserves_surface_text_for_structured_patient_predicates():
    """Robust Delta must not turn age criteria into canonical-ID/hash labels."""
    original = task()
    age = Requirement("p4", "patient", {"attribute": "age", "operator": ">=", "value": 65})
    added = apply_task_delta(original, TaskDelta(acts=(
        AddPredicateAct(
            age,
            (AttachmentSpec("root", "and_child"),),
            display_text="age at first admission is at least 65 years",
        ),
    )))
    age_requirement = next(r for r in added.requirements if r.kind == "patient")
    age_leaf = next(x for x in added.logic.leaves() if x.predicate_id == age_requirement.requirement_id)
    assert age_leaf.text == "age at first admission is at least 65 years"
    assert age_requirement.requirement_id not in age_leaf.text

    replaced = apply_task_delta(added, TaskDelta(acts=(
        ReplacePredicateAct(
            age_requirement.requirement_id,
            Requirement("temporary", "patient", {"attribute": "age", "operator": ">=", "value": 70}),
            display_text="age at first admission is at least 70 years",
        ),
    )))
    replaced_leaf = next(x for x in replaced.logic.leaves() if x.predicate_id == age_requirement.requirement_id)
    assert replaced_leaf.text == "age at first admission is at least 70 years"


def test_delta_21_moving_patient_predicate_keeps_existing_surface_text():
    logic = LogicExpression.and_(
        LogicExpression.predicate("p1", "age at first admission is at least 50 years"),
        LogicExpression.or_(
            LogicExpression.predicate("p2", "B"),
            LogicExpression.predicate("p3", "C"),
        ),
    )
    original = task(logic=logic)
    original = replace_task_requirement(
        original, "p1", {"attribute": "age", "operator": ">=", "value": 50}, kind="patient"
    )
    moved = apply_task_delta(original, TaskDelta(acts=(
        MovePredicateAct("p1", AttachmentSpec("root", "or_child")),
    )))
    leaf = next(x for x in moved.logic.leaves() if x.predicate_id == "p1")
    assert leaf.text == "age at first admission is at least 50 years"
