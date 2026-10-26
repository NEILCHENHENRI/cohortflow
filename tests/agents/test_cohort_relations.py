import threading

from agents.cohort.operators import CohortOperatorServices, register_cohort_operators
from framework.operator import OperatorContext
from framework.task import Requirement


class Resolver:
    def resolve_procedure(self, term):
        return f"procedure:{term}"

    def resolve_lab(self, term):
        return f"lab:{term}"

    def resolve_condition(self, term):
        return f"condition:{term}"

    def resolve_medication(self, term):
        return f"medication:{term}"


class Repository:
    def admissions_matching_requirement(self, kind, payload, concept=None):
        term = payload.get("term")
        if kind == "procedure" and term == "paracentesis":
            return {(1, 100), (2, 200)}
        if kind == "lab" and term == "potassium":
            return {(1, 100), (2, 201)}
        raise AssertionError((kind, payload, concept))

    def patients_satisfying_relation(self, **kwargs):
        assert kwargs["relation"] == "before"
        assert kwargs["left_kind"] == "lab"
        assert kwargs["right_kind"] == "procedure"
        assert kwargs["max_gap_hours"] == 168
        assert kwargs["same_admission"] is True
        return {1}


class NoteMatcher:
    think = True

    def admissions_with_note(self, payload):
        claim = payload["claim"]
        if "non-adherent" in claim:
            return {(1, 100)}
        return {(2, 200)}

    def patients_with_note(self, payload):
        return {subject_id for subject_id, _ in self.admissions_with_note(payload)}


def context():
    return OperatorContext(execution_version=1, stop_event=threading.Event(), node_id="relation")


def test_same_admission_relation_supports_nested_note_or():
    registry = register_cohort_operators(CohortOperatorServices(Resolver(), Repository(), NoteMatcher()))
    evaluate = registry.get("evaluate_relation").function
    requirements = [
        Requirement("p6", "procedure", {"term": "paracentesis"}),
        Requirement("p7", "lab", {"term": "potassium", "operator": ">", "value": 5.5, "count_min": 2}),
        Requirement("p10", "note", {"claim": "patient was non-adherent"}),
        Requirement("p11", "note", {"claim": "patient experienced confusion or hallucinations"}),
    ]
    result = evaluate(
        {f"r:{r.requirement_id}": r for r in requirements},
        {
            "relation": {
                "relation": "same_admission",
                "members": [
                    {"kind": "predicate", "requirement_id": "p6"},
                    {"kind": "predicate", "requirement_id": "p7"},
                    {"kind": "or", "requirement_ids": ["p10", "p11"]},
                ],
            },
            "requirement_ids": ["p6", "p7", "p10", "p11"],
            "output_artifact": "relation_patients:1",
        },
        context(),
    )
    assert result["relation_patients:1"] == {1}


def test_before_relation_delegates_to_generic_event_relation_execution():
    registry = register_cohort_operators(CohortOperatorServices(Resolver(), Repository(), NoteMatcher()))
    evaluate = registry.get("evaluate_relation").function
    result = evaluate(
        {
            "lab": Requirement("p7", "lab", {"term": "potassium", "operator": ">", "value": 5.5}),
            "procedure": Requirement("p6", "procedure", {"term": "paracentesis"}),
        },
        {
            "relation": {
                "relation": "before",
                "left_requirement_id": "p7",
                "right_requirement_id": "p6",
                "max_gap_hours": 168,
                "same_admission": True,
            },
            "requirement_ids": ["p7", "p6"],
            "output_artifact": "relation_patients:2",
        },
        context(),
    )
    assert result["relation_patients:2"] == {1}
