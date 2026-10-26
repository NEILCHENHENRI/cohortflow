from __future__ import annotations

"""Structured task fixtures for note-policy and presentation unit tests."""

from framework.logic import LogicExpression, assign_logic_node_ids
from framework.task import Deliverable, Objective, Requirement, TaskSpec


TEMPORAL_NOTE_PROMPT = (
    "Adults age 50 or older with at least two hospital admissions, including at "
    "least one emergency admission, who have a diagnosis of cirrhosis and a "
    "documented paracentesis procedure, had a potassium level greater than 5.5 "
    "during the same admission as the paracentesis, received oral furosemide at "
    "a dose of at least 40 mg during that admission, and the discharge summary "
    "from that hospitalization documents medication non-adherence. Exclude "
    "patients who died during that hospitalization. Require the elevated "
    "potassium to occur before the paracentesis, within 7 days."
)

ENCOUNTER_NOTE_PROMPT = (
    "Adults age 50–80 with at least two emergency admissions who have cirrhosis "
    "and COPD, underwent paracentesis, had at least two potassium measurements "
    "above 5.5 during the same hospitalization, received oral furosemide at ≥40 "
    "mg and lactulose during that admission, and the discharge summary from that "
    "hospitalization documents either medication non-adherence or confusion/"
    "hallucinations. Require at least one of those elevated potassium measurements "
    "to occur before the paracentesis, within 7 days, and exclude patients who "
    "died during that hospitalization."
)


def _r(requirement_id: str, kind: str, payload: dict) -> Requirement:
    return Requirement(requirement_id, kind, payload, polarity="include", required=True)


def _leaf(requirement_id: str, text: str) -> LogicExpression:
    return LogicExpression.predicate(requirement_id, text)


def _task(prompt: str, requirements: tuple[Requirement, ...], logic: LogicExpression, relations: list[dict], name: str) -> TaskSpec:
    return TaskSpec(
        domain="cohort_curation",
        objective=Objective("cohort_retrieval", prompt),
        deliverables=(Deliverable(
            "patients", "patient_set", "Patients matching the cohort specification", True, "json"
        ),),
        requirements=requirements,
        logic=assign_logic_node_ids(logic),
        context={"relations": relations},
        metadata={
            "fixture": name,
            "parser_bypassed": True,
            "logic_requires_clarification": False,
            "logic_clarification_question": None,
        },
        original_request=prompt,
    )


def temporal_note_task() -> TaskSpec:
    requirements = (
        _r("p1", "patient", {"attribute": "age", "operator": ">=", "value": 50}),
        _r("p2", "admission", {"count_min": 2}),
        _r("p3", "admission", {"admission_type": "emergency", "count_min": 1}),
        _r("p4", "condition", {"term": "cirrhosis"}),
        _r("p5", "procedure", {"term": "paracentesis"}),
        _r("p6", "lab", {"term": "potassium", "operator": ">", "value": 5.5}),
        _r("p7", "medication", {
            "term": "furosemide", "dose_operator": ">=", "dose_value": 40,
            "dose_unit": "mg", "route": "oral",
        }),
        _r("p8", "note", {"claim": "the patient was non-adherent to prescribed medication"}),
        _r("p9", "admission", {"hospital_expire_flag": 0}),
    )
    leaves = {
        "p1": _leaf("p1", "age at first admission is at least 50 years"),
        "p2": _leaf("p2", "at least two hospital admissions"),
        "p3": _leaf("p3", "at least one emergency admission"),
        "p4": _leaf("p4", "cirrhosis"),
        "p5": _leaf("p5", "paracentesis"),
        "p6": _leaf("p6", "potassium greater than 5.5"),
        "p7": _leaf("p7", "oral furosemide at least 40 mg"),
        "p8": _leaf("p8", "discharge note documents medication non-adherence"),
        "p9": _leaf("p9", "survived the relevant hospitalization"),
    }
    logic_ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
    logic = LogicExpression.and_(*(leaves[pid] for pid in logic_ids))
    same_admission_members = ["p5", "p6", "p7", "p8", "p9"]
    relations = [
        {
            "relation": "same_admission",
            "members": same_admission_members,
        },
        {
            "relation": "before",
            "left_requirement_id": "p6",
            "right_requirement_id": "p5",
            "max_gap_hours": 168,
            "same_admission": True,
        },
    ]
    return _task(TEMPORAL_NOTE_PROMPT, requirements, logic, relations, "temporal_note")


def encounter_note_task() -> TaskSpec:
    # Age range is represented by two ordinary patient predicates because the
    # canonical patient schema uses one comparator/value per atomic criterion.
    requirements = (
        _r("p1", "patient", {"attribute": "age", "operator": ">=", "value": 50}),
        _r("p2", "patient", {"attribute": "age", "operator": "<=", "value": 80}),
        _r("p3", "admission", {"admission_type": "emergency", "count_min": 2}),
        _r("p4", "condition", {"term": "cirrhosis"}),
        _r("p5", "condition", {"term": "COPD"}),
        _r("p6", "procedure", {"term": "paracentesis"}),
        _r("p7", "lab", {"term": "potassium", "operator": ">", "value": 5.5, "count_min": 2}),
        _r("p8", "medication", {
            "term": "furosemide", "dose_operator": ">=", "dose_value": 40,
            "dose_unit": "mg", "route": "oral",
        }),
        _r("p9", "medication", {"term": "lactulose"}),
        _r("p10", "note", {"claim": "the patient was non-adherent to prescribed medication"}),
        _r("p11", "note", {"claim": "the patient experienced confusion or hallucinations"}),
        _r("p12", "admission", {"hospital_expire_flag": 0}),
    )
    leaves = {
        "p1": _leaf("p1", "age at first admission is at least 50 years"),
        "p2": _leaf("p2", "age at first admission is at most 80 years"),
        "p3": _leaf("p3", "at least two emergency admissions"),
        "p4": _leaf("p4", "cirrhosis"),
        "p5": _leaf("p5", "COPD"),
        "p6": _leaf("p6", "paracentesis"),
        "p7": _leaf("p7", "at least two potassium measurements above 5.5"),
        "p8": _leaf("p8", "oral furosemide at least 40 mg"),
        "p9": _leaf("p9", "lactulose"),
        "p10": _leaf("p10", "discharge note documents medication non-adherence"),
        "p11": _leaf("p11", "discharge note documents confusion or hallucinations"),
        "p12": _leaf("p12", "survived the relevant hospitalization"),
    }
    logic = LogicExpression.and_(
        leaves["p1"], leaves["p2"], leaves["p3"], leaves["p4"], leaves["p5"],
        leaves["p6"], leaves["p7"], leaves["p8"], leaves["p9"],
        LogicExpression.or_(leaves["p10"], leaves["p11"]),
        leaves["p12"],
    )
    same_admission_members = [
        "p6", "p7", "p8", "p9",
        {"kind": "or", "requirement_ids": ["p10", "p11"]},
        "p12",
    ]
    relations = [
        {
            "relation": "same_admission",
            "members": same_admission_members,
        },
        {
            "relation": "before",
            "left_requirement_id": "p7",
            "right_requirement_id": "p6",
            "max_gap_hours": 168,
            "same_admission": True,
        },
    ]
    return _task(ENCOUNTER_NOTE_PROMPT, requirements, logic, relations, "encounter_note")

