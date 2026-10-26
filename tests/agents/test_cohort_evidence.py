import threading

from agents.cohort.operators import (
    CohortOperatorServices,
    _aggregate_cohort,
    _evaluate_requirement,
)
from framework.operator import OperatorContext
from framework.logic import LogicExpression
from framework.task import Requirement


class Concept:
    concept_id = 123
    concept_name = "Condition X"
    domain_id = "Condition"
    vocabulary_id = "SNOMED"
    resolution_source = "human"


class Resolver:
    def resolve_condition(self, term):
        return Concept()

    def resolve_medication(self, term):
        return Concept()


class Repo:
    def patients_with_condition(self, concept, payload=None):
        return {1, 2}

    def patients_with_medication(self, concept, payload=None):
        return set()


def context():
    return OperatorContext(
        execution_version=1,
        stop_event=threading.Event(),
        node_id="x",
    )


def test_requirement_evidence_contains_concept_and_patients():
    execute = _evaluate_requirement(
        CohortOperatorServices(Resolver(), Repo())
    )
    output = execute(
        {
            "requirement": Requirement(
                "condition_x",
                "condition",
                {"term": "Condition X"},
            )
        },
        {
            "output_artifact": "patients:x",
            "evidence_artifact": "evidence:x",
        },
        context(),
    )
    assert output["patients:x"] == {1, 2}
    assert output["evidence:x"]["patient_count"] == 2
    assert output["evidence:x"]["resolved_concept"]["concept_id"] == 123


def test_aggregate_evidence_explains_membership():
    output = _aggregate_cohort(
        {
            "patients:a": {1, 2},
            "patients:b": {2, 3},
        },
        {
            "logic": LogicExpression.or_(
                LogicExpression.predicate("a", "A"),
                LogicExpression.predicate("b", "B"),
            ).to_dict(),
            "artifact_by_requirement": {"a": "patients:a", "b": "patients:b"},
            "universe_artifact": None,
        },
        context(),
    )
    assert output["cohort_patient_ids"] == {1, 2, 3}
    assert output["cohort_evidence"]["result_count"] == 3
    assert output["cohort_evidence"]["patient_membership"][1][
        "matched_include_requirements"
    ] == ["a"]
