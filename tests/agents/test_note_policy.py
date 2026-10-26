import pytest

from tests.cohort_fixtures import temporal_note_task, encounter_note_task
from agents.cohort.note_policy import without_note_requirements
from framework.logic import LogicExpression
from framework.task import Deliverable, Objective, Requirement, TaskSpec


def _ids(task):
    return {requirement.requirement_id for requirement in task.requirements}


def test_without_notes_prunes_nested_logic_and_same_admission_members():
    task = without_note_requirements(encounter_note_task())
    assert {"p10", "p11"}.isdisjoint(_ids(task))
    assert {"p10", "p11"}.isdisjoint(task.logic.predicate_ids())
    relation = next(item for item in task.context["relations"] if item["relation"] == "same_admission")
    assert all(
        not isinstance(member, dict)
        or not {"p10", "p11"}.intersection(member.get("requirement_ids", ()))
        for member in relation["members"]
    )
    assert task.metadata["notes_skipped"] is True


def test_without_notes_preserves_non_note_temporal_relation():
    task = without_note_requirements(temporal_note_task())
    before = next(item for item in task.context["relations"] if item["relation"] == "before")
    assert before["left_requirement_id"] == "p6"
    assert before["right_requirement_id"] == "p5"


def test_without_notes_drops_relation_whose_endpoint_is_note():
    requirements = (
        Requirement("p1", "condition", {"term": "diabetes"}),
        Requirement("p2", "note", {"claim": "non-adherence"}),
    )
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "x"),
        deliverables=(Deliverable("patients", "patient_set"),),
        requirements=requirements,
        logic=LogicExpression.and_(
            LogicExpression.predicate("p1", "diabetes"),
            LogicExpression.predicate("p2", "non-adherence"),
        ),
        context={"relations": [{
            "relation_id": "r1", "relation": "before",
            "left_requirement_id": "p2", "right_requirement_id": "p1",
        }]},
    )
    stripped = without_note_requirements(task)
    assert stripped.logic.render_boolean() == "p1"
    assert stripped.context["relations"] == []


def test_without_notes_refuses_note_only_cohort():
    task = TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", "x"),
        requirements=(Requirement("p1", "note", {"claim": "non-adherence"}),),
        logic=LogicExpression.predicate("p1", "non-adherence"),
    )
    with pytest.raises(ValueError, match="every cohort requirement is note-based"):
        without_note_requirements(task)
