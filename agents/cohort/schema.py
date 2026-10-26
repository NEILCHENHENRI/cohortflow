from __future__ import annotations

from typing import Any

from framework.errors import DomainValidationError
from framework.task import Requirement
from agents.cohort.structured import validate_structured_payload


CONDITION_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["term"],
    "properties": {
        "term": {"type": "string", "minLength": 1},
        "count_min": {"type": "integer", "minimum": 0},
        "seq_num": {"type": "integer", "minimum": 0},
    },
}

PROCEDURE_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["term"],
    "properties": {
        "term": {"type": "string", "minLength": 1},
        "count_min": {"type": "integer", "minimum": 0},
    },
}

LAB_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["term"],
    "properties": {
        "term": {"type": "string", "minLength": 1},
        "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<="]},
        "value": {"type": "number"},
        "value_text": {"type": "string"},
        "unit": {"type": "string"},
        "abnormal_flag": {"type": "string"},
        "count_min": {"type": "integer", "minimum": 0},
        "aggregation": {"type": "string", "enum": ["any", "latest", "min", "max", "avg"]},
        "start_time": {"type": "string"},
        "end_time": {"type": "string"},
    },
}

MEDICATION_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["term"],
    "properties": {
        "term": {"type": "string", "minLength": 1},
        "dose_operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<="]},
        "dose_value": {"type": "number"},
        "dose_unit": {"type": "string"},
        "route": {"type": "string"},
        "frequency": {"type": "string"},
        "count_min": {"type": "integer", "minimum": 0},
        "start_time": {"type": "string"},
        "end_time": {"type": "string"},
        "exposure_mode": {"type": "string", "enum": ["any_recorded", "active_at_index"]},
    },
}

PATIENT_PAYLOAD_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["attribute", "operator", "value"],
            "properties": {
                "attribute": {"const": "age"},
                "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<="]},
                "value": {"type": "number"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["attribute", "operator", "value"],
            "properties": {
                "attribute": {"const": "gender"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {"type": "string", "minLength": 1},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["attribute", "operator", "value"],
            "properties": {
                "attribute": {"const": "deceased"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "integer", "enum": [0, 1]},
                    ]
                },
            },
        },
    ]
}

ADMISSION_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "minProperties": 1,
    "properties": {
        "admission_type": {"type": "string"},
        "admission_location": {"type": "string"},
        "discharge_location": {"type": "string"},
        "insurance": {"type": "string"},
        "language": {"type": "string"},
        "marital_status": {"type": "string"},
        "race": {"type": "string"},
        "hospital_expire_flag": {
            "oneOf": [
                {"type": "boolean"},
                {"type": "integer", "enum": [0, 1]},
            ]
        },
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "duration_operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<="]},
        "duration_days": {"type": "number"},
        "count_min": {"type": "integer", "minimum": 0},
    },
}

NOTE_PAYLOAD_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claim"],
    "properties": {
        "claim": {"type": "string", "minLength": 1},
        "count_min": {"type": "integer", "minimum": 0},
    },
}

def requirement_schema(kinds: tuple[str, ...] | None = None) -> dict[str, Any]:
    base = {
        "type": "object", "additionalProperties": False,
        "required": ["requirement_id", "kind", "payload", "polarity", "required"],
        "properties": {
            "requirement_id": {"type": "string"},
            "kind": {"type": "string"},
            "payload": {"type": "object"},
            "polarity": {"type": "string", "enum": ["include", "exclude"]},
            "required": {"type": "boolean"},
        },
    }
    schemas = {
        "condition": CONDITION_PAYLOAD_SCHEMA,
        "procedure": PROCEDURE_PAYLOAD_SCHEMA,
        "lab": LAB_PAYLOAD_SCHEMA,
        "medication": MEDICATION_PAYLOAD_SCHEMA,
        "patient": PATIENT_PAYLOAD_SCHEMA,
        "admission": ADMISSION_PAYLOAD_SCHEMA,
        "note": NOTE_PAYLOAD_SCHEMA,
    }
    selected = schemas if kinds is None else {kind: schemas[kind] for kind in kinds}
    return {"oneOf": [
        {**base, "properties": {**base["properties"], "kind": {"const": kind}, "payload": schema}}
        for kind, schema in selected.items()
    ]}


DELIVERABLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "deliverable_id",
        "kind",
        "description",
        "required",
        "format",
    ],
    "properties": {
        "deliverable_id": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": [
                "patient_set",
                "cohort_summary",
                "eligibility_evidence",
                "sql",
            ],
        },
        "description": {"type": "string"},
        "required": {"type": "boolean"},
        "format": {"type": ["string", "null"]},
    },
}

ASSUMPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assumption_id", "statement", "status"],
    "properties": {
        "assumption_id": {"type": "string"},
        "statement": {"type": "string"},
        "status": {"type": "string"},
    },
}

def validate_requirement_payload(kind: str, payload: dict[str, Any]) -> None:
    from agents.cohort.structured import validate_structured_payload

    validate_structured_payload(kind, payload)


def _requirement_from_payload(item: dict[str, Any]) -> Requirement:
    payload = dict(item.get("payload", {}))
    validate_requirement_payload(item["kind"], payload)
    return Requirement(
        requirement_id=item["requirement_id"],
        kind=item["kind"],
        payload=payload,
        polarity=item.get("polarity", "include"),
        required=item.get("required", True),
    )


# Structured response contracts for the cohort parsing and correction passes.
def _flat_logic_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "predicates", "logic", "ambiguities", "requires_clarification",
            "clarification_question", "rationale",
        ],
        "properties": {
            "predicates": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["predicate_id", "text"],
                    "properties": {
                        "predicate_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                },
            },
            "logic": {"type": "string"},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "requires_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "rationale": {"type": "string"},
        },
    }


def _predicate_extraction_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["predicates", "ambiguities", "requires_clarification", "clarification_question", "rationale"],
        "properties": {
            "predicates": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["predicate_id", "text"],
                    "properties": {"predicate_id": {"type": "string"}, "text": {"type": "string"}},
                },
            },
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "requires_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "rationale": {"type": "string"},
        },
    }


def _relationship_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["logic", "ambiguities", "requires_clarification", "clarification_question", "rationale"],
        "properties": {
            "logic": {"type": "string"},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "requires_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "rationale": {"type": "string"},
        },
    }


def _carbon_schema() -> dict[str, Any]:
    base = _flat_logic_result_schema()
    props = dict(base["properties"])
    props["requirements"] = {"type": "array", "items": requirement_schema()}
    props["assumptions"] = {"type": "array", "items": ASSUMPTION_SCHEMA}
    required = list(base["required"]) + ["requirements", "assumptions"]
    return {"type": "object", "additionalProperties": False, "required": required, "properties": props}


def _merged_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["merged_request"],
        "properties": {
            "merged_request": {"type": "string", "minLength": 1},
        },
    }




def _correction_actions_schema(
    *, predicate_refs: tuple[str, ...], group_refs: tuple[str, ...], relation_refs: tuple[str, ...],
) -> dict[str, Any]:
    anchors = ["ROOT", *predicate_refs, *group_refs]

    def obj(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    action = lambda name: {"type": "string", "const": name}
    p_ref = {"type": "string", "enum": list(predicate_refs)}
    g_ref = {"type": "string", "enum": list(group_refs)}
    r_ref = {"type": "string", "enum": list(relation_refs)}
    anchor = {"type": "string"}
    bool_op = {"type": "string", "enum": ["AND", "OR"]}
    placement = {"type": "string", "enum": ["WITH", "INSIDE"]}
    local_p = {"type": "string", "pattern": "^new_[1-9][0-9]*$"}
    local_r = {"type": "string", "pattern": "^new_r[1-9][0-9]*$"}
    endpoint = {"type": "string"}
    relation_type = {"type": "string", "enum": ["BEFORE", "AFTER", "SAME_ADMISSION"]}
    gap_unit = {"type": "string", "enum": ["hour", "day", "week"]}

    variants = [
        obj(
            ["action", "local_ref", "requirement_text", "excluded", "anchor", "operator", "relation_memberships"],
            {
                "action": action("ADD"), "local_ref": local_p,
                "requirement_text": {"type": "string", "minLength": 1},
                "excluded": {"type": "boolean"}, "anchor": anchor,
                "operator": bool_op,
                "relation_memberships": {"type": "array", "items": r_ref},
            },
        ),
        obj(
            ["action", "target", "requirement_text"],
            {"action": action("MODIFY"), "target": p_ref, "requirement_text": {"type": "string", "minLength": 1}},
        ),
        obj(["action", "target"], {"action": action("REMOVE"), "target": p_ref}),
        obj(
            ["action", "target", "anchor", "operator", "placement"],
            {"action": action("MOVE"), "target": p_ref, "anchor": anchor, "operator": bool_op, "placement": placement},
        ),
        obj(
            ["action", "target", "excluded"],
            {"action": action("SET_EXCLUDED"), "target": p_ref, "excluded": {"type": "boolean"}},
        ),
        obj(
            ["action", "target", "operator"],
            {"action": action("CHANGE_OPERATOR"), "target": g_ref, "operator": bool_op},
        ),
        obj(
            ["action", "local_ref", "relation"],
            {
                "action": action("ADD_RELATION"), "local_ref": local_r,
                "relation": relation_type,
                "left": endpoint, "right": endpoint,
                "members": {"type": "array", "items": endpoint, "minItems": 2},
                "max_gap_value": {"type": "number", "minimum": 0},
                "max_gap_unit": gap_unit,
                "same_admission": {"type": "boolean"},
            },
        ),
        obj(
            ["action", "target"],
            {
                "action": action("MODIFY_RELATION"), "target": r_ref,
                "relation": relation_type,
                "left": endpoint, "right": endpoint,
                "members": {"type": "array", "items": endpoint, "minItems": 2},
                "max_gap_value": {"type": "number", "minimum": 0},
                "max_gap_unit": gap_unit,
                "clear_max_gap": {"type": "boolean"},
                "same_admission": {"type": "boolean"},
            },
        ),
        obj(["action", "target"], {"action": action("REMOVE_RELATION"), "target": r_ref}),
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["actions", "requires_clarification", "clarification_question", "clarification_rationale"],
        "properties": {
            "actions": {"type": "array", "items": {"oneOf": variants}},
            "requires_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "clarification_rationale": {"type": ["string", "null"]},
        },
    }


def _predicate_compilation_schema(kinds: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirements", "assumptions"],
        "properties": {
            "requirements": {"type": "array", "items": requirement_schema(kinds)},
            "assumptions": {"type": "array", "items": ASSUMPTION_SCHEMA},
        },
    }


def _correction_extraction_schema() -> dict[str, Any]:
    """Extract intent/text without committing to structural targets or attachments."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["intents", "requires_clarification", "clarification_question", "clarification_rationale"],
        "properties": {
            "intents": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["intent_id", "action", "instruction", "requirement_text"],
                "properties": {
                    "intent_id": {"type": "string", "pattern": "^c[1-9][0-9]*$"},
                    "action": {"type": "string", "enum": [
                        "ADD", "MODIFY", "REMOVE", "MOVE", "SET_EXCLUDED", "CHANGE_OPERATOR",
                        "ADD_RELATION", "MODIFY_RELATION", "REMOVE_RELATION",
                    ]},
                    "instruction": {"type": "string", "minLength": 1},
                    "requirement_text": {"type": ["string", "null"]},
                },
            }},
            "requires_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "clarification_rationale": {"type": ["string", "null"]},
        },
    }


def _correction_attachment_schema(refs, intents) -> dict[str, Any]:
    schema = _correction_actions_schema(
        predicate_refs=refs["predicates"], group_refs=refs["groups"], relation_refs=refs["relations"],
    )
    for variant in schema["properties"]["actions"]["items"]["oneOf"]:
        variant["required"].append("intent_id")
        variant["properties"]["intent_id"] = {
            "type": "string", "enum": [item["intent_id"] for item in intents],
        }
    return schema


def _carbon_correction_schema(refs) -> dict[str, Any]:
    schema = _correction_actions_schema(
        predicate_refs=refs["predicates"], group_refs=refs["groups"], relation_refs=refs["relations"],
    )
    compiled = _predicate_compilation_schema()
    schema["properties"].update(compiled["properties"])
    schema["required"].extend(compiled["required"])
    return schema
