from __future__ import annotations

"""Backend-independent structured semantics for cohort criteria.

This module is intentionally LLM-free. Domain parsers produce these payloads.
Database adapters consume them. No physical table or column knowledge belongs
here.
"""

from dataclasses import dataclass
from typing import Any

from framework.errors import DomainValidationError
from framework.task_delta import RELATION_KINDS, relation_constraints


SUPPORTED_DOMAINS = (
    "patient",
    "admission",
    "condition",
    "procedure",
    "lab",
    "medication",
    "note",
)

COMPARATORS = ("=", "!=", ">", ">=", "<", "<=")


@dataclass(frozen=True, slots=True)
class EventRelation:
    relation: str
    left_requirement_id: str
    right_requirement_id: str
    max_gap_hours: int | None = None
    same_admission: bool = False

    def __post_init__(self) -> None:
        if self.relation not in {"before", "after"}:
            raise DomainValidationError("relation must be before or after")
        if not self.left_requirement_id or not self.right_requirement_id:
            raise DomainValidationError("relation endpoints must be non-empty")
        if self.max_gap_hours is not None and self.max_gap_hours < 0:
            raise DomainValidationError("max_gap_hours cannot be negative")


def validate_structured_payload(kind: str, payload: dict[str, Any]) -> None:
    """Validate domain-specific payloads without looking at a database.

    Optional fields are intentionally sparse: unsupported attributes fail
    closed instead of being silently ignored or converted to SQL.
    """

    if kind == "condition":
        _keys(payload, required={"term"}, optional={"count_min", "seq_num"})
        _text(payload, "term")
        _optional_nonnegative_int(payload, "count_min")
        _optional_nonnegative_int(payload, "seq_num")
        return

    if kind == "procedure":
        _keys(payload, required={"term"}, optional={"count_min"})
        _text(payload, "term")
        _optional_nonnegative_int(payload, "count_min")
        return

    if kind == "lab":
        _keys(
            payload,
            required={"term"},
            optional={
                "operator", "value", "value_text", "unit", "abnormal_flag",
                "count_min", "aggregation", "start_time", "end_time",
            },
        )
        _text(payload, "term")
        _optional_comparator(payload)
        if "operator" in payload and "value" not in payload:
            raise DomainValidationError("lab operator requires numeric value")
        if "value" in payload and isinstance(payload["value"], bool):
            raise DomainValidationError("lab value must be numeric")
        if "value" in payload and not isinstance(payload["value"], (int, float)):
            raise DomainValidationError("lab value must be numeric")
        for key in ("value_text", "unit", "abnormal_flag", "start_time", "end_time"):
            _optional_text(payload, key)
        _optional_nonnegative_int(payload, "count_min")
        aggregation = payload.get("aggregation", "any")
        if aggregation not in {"any", "latest", "min", "max", "avg"}:
            raise DomainValidationError("unsupported lab aggregation")
        return

    if kind == "medication":
        _keys(
            payload,
            required={"term"},
            optional={
                "dose_operator", "dose_value", "dose_unit", "route",
                "frequency", "count_min", "start_time", "end_time",
                "exposure_mode",
            },
        )
        _text(payload, "term")
        if "dose_operator" in payload:
            if payload["dose_operator"] not in COMPARATORS:
                raise DomainValidationError("invalid medication dose operator")
            if "dose_value" not in payload:
                raise DomainValidationError("dose_operator requires dose_value")
        if "dose_value" in payload and (
            isinstance(payload["dose_value"], bool)
            or not isinstance(payload["dose_value"], (int, float))
        ):
            raise DomainValidationError("dose_value must be numeric")
        for key in ("dose_unit", "route", "frequency", "start_time", "end_time"):
            _optional_text(payload, key)
        _optional_nonnegative_int(payload, "count_min")
        exposure_mode = payload.get("exposure_mode", "any_recorded")
        if exposure_mode not in {"any_recorded", "active_at_index"}:
            raise DomainValidationError("unsupported medication exposure_mode")
        return

    if kind == "patient":
        _keys(payload, required={"attribute", "operator", "value"})
        attribute = payload["attribute"]
        if attribute not in {"gender", "age", "deceased"}:
            raise DomainValidationError("unsupported patient attribute")
        if payload["operator"] not in COMPARATORS:
            raise DomainValidationError("invalid patient comparator")
        if attribute == "age":
            if (
                isinstance(payload["value"], bool)
                or not isinstance(payload["value"], (int, float))
            ):
                raise DomainValidationError("patient age value must be numeric")
            return
        if attribute == "gender":
            if payload["operator"] not in {"=", "!="}:
                raise DomainValidationError("patient gender supports only = or !=")
            if not isinstance(payload["value"], str) or not payload["value"].strip():
                raise DomainValidationError("patient gender value must be non-empty text")
            return
        if attribute == "deceased":
            if payload["operator"] not in {"=", "!="}:
                raise DomainValidationError("patient deceased supports only = or !=")
            if payload["value"] not in {0, 1, False, True}:
                raise DomainValidationError("patient deceased value must be boolean/0/1")
            return
        return

    if kind == "admission":
        _keys(
            payload,
            required=set(),
            optional={
                "admission_type", "admission_location", "discharge_location",
                "insurance", "language", "marital_status", "race",
                "hospital_expire_flag", "start_date", "end_date",
                "duration_operator", "duration_days", "count_min",
            },
        )
        if not payload:
            raise DomainValidationError("admission requirement cannot be empty")
        for key in (
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race",
            "start_date", "end_date",
        ):
            _optional_text(payload, key)
        if "hospital_expire_flag" in payload and payload["hospital_expire_flag"] not in {0, 1, False, True}:
            raise DomainValidationError("hospital_expire_flag must be boolean/0/1")
        if "duration_operator" in payload:
            if payload["duration_operator"] not in COMPARATORS:
                raise DomainValidationError("invalid duration operator")
            if "duration_days" not in payload:
                raise DomainValidationError("duration_operator requires duration_days")
        if "duration_days" in payload and (
            isinstance(payload["duration_days"], bool)
            or not isinstance(payload["duration_days"], (int, float))
        ):
            raise DomainValidationError("duration_days must be numeric")
        _optional_nonnegative_int(payload, "count_min")
        return

    if kind == "note":
        # New parser output uses a normalized evidence claim. ``query`` is
        # accepted only as a compatibility alias for direct repository/tests.
        has_claim = isinstance(payload.get("claim"), str) and payload.get("claim", "").strip()
        has_query = isinstance(payload.get("query"), str) and payload.get("query", "").strip()
        if not (has_claim or has_query):
            raise DomainValidationError("note requirement requires claim")
        allowed = {"claim", "query", "count_min", "match_mode", "section"}
        extra = set(payload) - allowed
        if extra:
            raise DomainValidationError(f"unexpected {sorted(extra)}")
        if has_claim:
            _text(payload, "claim")
        if has_query:
            _text(payload, "query")
        _optional_nonnegative_int(payload, "count_min")
        return

    raise DomainValidationError(f"unsupported structured requirement kind: {kind}")


def _keys(payload: dict[str, Any], *, required: set[str], optional: set[str] = set()) -> None:
    actual = set(payload)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        raise DomainValidationError(", ".join(parts))


def _text(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{key} must be non-empty text")


def _optional_text(payload: dict[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None:
        _text(payload, key)


def _optional_nonnegative_int(payload: dict[str, Any], key: str) -> None:
    if key not in payload or payload[key] is None:
        return
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainValidationError(f"{key} must be a non-negative integer")


def _optional_comparator(payload: dict[str, Any]) -> None:
    if "operator" in payload and payload["operator"] not in COMPARATORS:
        raise DomainValidationError("invalid comparator")
