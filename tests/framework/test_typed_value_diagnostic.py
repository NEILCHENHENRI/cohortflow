"""Diagnostic for the typed-value boundary in cohort requirement validation.

This test uses the actual project modules:

    framework.task.Requirement
    agents.cohort.schema.validate_requirement_payload

It reproduces the value shape observed in the LLM activity panel:

    {"type": "number", "value": 18}

Run:

    PYTHONPATH=. pytest -q -s tests/framework/test_typed_value_diagnostic.py
"""

import pytest

from agents.cohort.schema import validate_requirement_payload
from framework.errors import DomainValidationError
from framework.task import Requirement


TYPED_NUMBER = {"type": "number", "value": 18}
RAW_NUMBER = 18


def _payload(value):
    return {
        "attribute": "age",
        "operator": ">",
        "value": value,
    }


def test_requirement_itself_preserves_typed_number_wrapper():
    """Boundary 1: Requirement does not normalize payload values."""
    req = Requirement(
        requirement_id="p_diag",
        kind="patient",
        payload=_payload(TYPED_NUMBER),
        polarity="include",
        required=True,
    )

    print("\n--- REQUIREMENT PAYLOAD ---")
    print("payload:", req.payload)
    print("value type:", type(req.payload["value"]))
    print("value repr:", repr(req.payload["value"]))

    assert req.payload["value"] == TYPED_NUMBER


def test_raw_numeric_age_is_accepted_by_real_domain_validator():
    """Control: the current validator accepts the canonical scalar form."""
    payload = _payload(RAW_NUMBER)

    print("\n--- RAW NUMERIC CONTROL ---")
    print("payload:", payload)

    validate_requirement_payload("patient", payload)


def test_typed_numeric_age_boundary_behavior():
    """Diagnostic: show exactly how the real validator treats the wrapper.

    If this test reports REJECTED while the raw-number control passes, then the
    representation-boundary mismatch is confirmed: the validator expects a
    scalar number but the upstream path supplied a typed-value wrapper.
    """
    payload = _payload(TYPED_NUMBER)

    print("\n--- TYPED NUMBER ENTERING REAL DOMAIN VALIDATOR ---")
    print("payload:", payload)
    print("value type:", type(payload["value"]))
    print("value repr:", repr(payload["value"]))

    try:
        validate_requirement_payload("patient", payload)
    except DomainValidationError as exc:
        print("\nRESULT: REJECTED")
        print("exception type:", type(exc).__name__)
        print("exception:", exc)

        # This assertion deliberately documents today's suspected behavior.
        # If it passes, the diagnostic has confirmed the mismatch.
        assert "age value must be numeric" in str(exc)
        return

    print("\nRESULT: ACCEPTED")
    pytest.fail(
        "Typed numeric wrapper unexpectedly passed validation. "
        "The failure must therefore occur at a different boundary."
    )


def test_lab_and_medication_have_same_typed_number_risk():
    """Check whether the mismatch is systemic across numeric cohort fields."""
    cases = [
        (
            "lab",
            {
                "term": "potassium",
                "operator": ">",
                "value": {"type": "number", "value": 5.5},
            },
            "lab value must be numeric",
        ),
        (
            "medication",
            {
                "term": "furosemide",
                "dose_operator": ">=",
                "dose_value": {"type": "number", "value": 40},
            },
            "dose_value must be numeric",
        ),
    ]

    for kind, payload, expected_error in cases:
        print(f"\n--- {kind.upper()} TYPED NUMERIC CHECK ---")
        print("payload:", payload)

        try:
            validate_requirement_payload(kind, payload)
        except DomainValidationError as exc:
            print("RESULT: REJECTED")
            print("exception:", exc)
            assert expected_error in str(exc)
        else:
            pytest.fail(
                f"{kind} typed numeric wrapper unexpectedly passed validation"
            )
