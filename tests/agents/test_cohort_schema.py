from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from agents.cohort.schema import requirement_schema, validate_requirement_payload
from framework.errors import DomainValidationError


def _requirement(kind: str, payload: dict):
    return {
        "requirement_id": "p1",
        "kind": kind,
        "payload": payload,
        "polarity": "include",
        "required": True,
    }


def _assert_schema_valid(item: dict) -> None:
    Draft202012Validator(requirement_schema()).validate(item)


def _assert_schema_invalid(item: dict) -> None:
    errors = list(Draft202012Validator(requirement_schema()).iter_errors(item))
    assert errors, f"expected schema rejection for {item!r}"


def test_patient_age_schema_requires_bare_number():
    _assert_schema_valid(_requirement("patient", {"attribute": "age", "operator": ">=", "value": 18}))
    _assert_schema_invalid(
        _requirement(
            "patient",
            {"attribute": "age", "operator": ">=", "value": {"type": "number", "value": 18}},
        )
    )


def test_patient_schema_conditions_value_and_operator_on_attribute():
    _assert_schema_valid(_requirement("patient", {"attribute": "gender", "operator": "=", "value": "female"}))
    _assert_schema_invalid(_requirement("patient", {"attribute": "gender", "operator": ">", "value": "female"}))
    _assert_schema_invalid(_requirement("patient", {"attribute": "gender", "operator": "=", "value": 1}))

    _assert_schema_valid(_requirement("patient", {"attribute": "deceased", "operator": "=", "value": True}))
    _assert_schema_valid(_requirement("patient", {"attribute": "deceased", "operator": "!=", "value": 0}))
    _assert_schema_invalid(_requirement("patient", {"attribute": "deceased", "operator": ">", "value": True}))
    _assert_schema_invalid(_requirement("patient", {"attribute": "deceased", "operator": "=", "value": 2}))


def test_domain_validator_matches_patient_schema_contract():
    validate_requirement_payload("patient", {"attribute": "age", "operator": ">=", "value": 18})
    validate_requirement_payload("patient", {"attribute": "gender", "operator": "=", "value": "female"})
    validate_requirement_payload("patient", {"attribute": "deceased", "operator": "=", "value": True})

    with pytest.raises(DomainValidationError, match="age value must be numeric"):
        validate_requirement_payload(
            "patient",
            {"attribute": "age", "operator": ">=", "value": {"type": "number", "value": 18}},
        )
    with pytest.raises(DomainValidationError, match="gender supports only"):
        validate_requirement_payload("patient", {"attribute": "gender", "operator": ">", "value": "female"})
    with pytest.raises(DomainValidationError, match="deceased value"):
        validate_requirement_payload("patient", {"attribute": "deceased", "operator": "=", "value": 2})


def test_numeric_payload_schemas_reject_schema_descriptor_wrappers():
    cases = [
        ("lab", {"term": "potassium", "operator": ">", "value": {"type": "number", "value": 5.5}}),
        (
            "medication",
            {"term": "furosemide", "dose_operator": ">=", "dose_value": {"type": "number", "value": 40}},
        ),
        (
            "admission",
            {"duration_operator": ">=", "duration_days": {"type": "number", "value": 7}},
        ),
    ]
    for kind, payload in cases:
        _assert_schema_invalid(_requirement(kind, payload))


def test_admission_expire_flag_schema_rejects_non_boolean_integer():
    _assert_schema_valid(_requirement("admission", {"hospital_expire_flag": 0}))
    _assert_schema_valid(_requirement("admission", {"hospital_expire_flag": True}))
    _assert_schema_invalid(_requirement("admission", {"hospital_expire_flag": 2}))
