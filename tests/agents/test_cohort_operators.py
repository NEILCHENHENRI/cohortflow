from agents.cohort.operators import (
    CohortOperatorServices,
    register_cohort_operators,
)
from framework.logic import LogicExpression
from framework.operator import OperatorContext
from framework.task import Requirement
import threading


class FakeResolver:
    def resolve_condition(self, term):
        return f"condition:{term}"

    def resolve_medication(self, term):
        return f"drug:{term}"


class FakeRepository:
    def patients_with_condition(self, concept, payload=None):
        return {
            "condition:A": {1, 2, 3},
            "condition:B": {3},
        }[concept]

    def patients_with_medication(self, concept, payload=None):
        return {1, 2}


def context(node_id="node"):
    return OperatorContext(
        execution_version=1,
        stop_event=threading.Event(),
        node_id=node_id,
    )


def test_requirement_evaluation_and_aggregation():
    registry = register_cohort_operators(
        CohortOperatorServices(
            FakeResolver(),
            FakeRepository(),
        )
    )
    evaluate = registry.get("evaluate_requirement").function
    aggregate = registry.get("aggregate_cohort").function

    a = evaluate(
        {"requirement:a": Requirement(
            "a", "condition", {"term": "A"}
        )},
        {"output_artifact": "patients:a"},
        context("a"),
    )
    b = evaluate(
        {"requirement:b": Requirement(
            "b", "condition", {"term": "B"}, polarity="exclude"
        )},
        {"output_artifact": "patients:b"},
        context("b"),
    )
    result = aggregate(
        {**a, **b, "all_patient_ids": {1, 2, 3}},
        {
            "logic": LogicExpression.and_(
                LogicExpression.predicate("a", "A"),
                LogicExpression.not_(LogicExpression.predicate("b", "B")),
            ).to_dict(),
            "artifact_by_requirement": {"a": "patients:a", "b": "patients:b"},
            "universe_artifact": "all_patient_ids",
        },
        context("aggregate"),
    )

    assert a["patients:a"] == {1, 2, 3}
    assert result["cohort_patient_ids"] == {1, 2}
