from __future__ import annotations

"""Semantic field metadata exposed to cohort compilation LLMs.

This is deliberately database-agnostic: descriptions explain meaning, never
show physical column names or example/database-native values.  The future
universal adapter can replace/extend this catalog without changing parser flow.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemanticField:
    domain: str
    name: str
    description: str

    @property
    def qualified_name(self) -> str:
        return f"{self.domain}.{self.name}"


_FIELDS: tuple[SemanticField, ...] = (
    SemanticField("condition", "term", "diagnosis or clinical condition concept"),
    SemanticField("condition", "count_min", "minimum number of matching condition occurrences"),
    SemanticField("condition", "seq_num", "diagnosis sequence or priority number when explicitly requested"),
    SemanticField("procedure", "term", "performed procedure concept"),
    SemanticField("procedure", "count_min", "minimum number of matching procedure occurrences"),
    SemanticField("lab", "term", "laboratory measurement or test concept"),
    SemanticField("lab", "operator", "comparison operator for a numeric laboratory result"),
    SemanticField("lab", "value", "numeric laboratory result threshold"),
    SemanticField("lab", "value_text", "explicit non-numeric laboratory result text"),
    SemanticField("lab", "unit", "unit explicitly stated with a laboratory value"),
    SemanticField("lab", "abnormal_flag", "explicit abnormal-result status or flag"),
    SemanticField("lab", "count_min", "minimum number of matching laboratory measurements"),
    SemanticField("lab", "aggregation", "requested aggregation such as latest, minimum, maximum, or average"),
    SemanticField("lab", "start_time", "lower time bound for the laboratory evidence"),
    SemanticField("lab", "end_time", "upper time bound for the laboratory evidence"),
    SemanticField("medication", "term", "medication or drug concept"),
    SemanticField("medication", "dose_operator", "comparison operator for medication dose"),
    SemanticField("medication", "dose_value", "numeric medication dose threshold"),
    SemanticField("medication", "dose_unit", "unit explicitly stated with medication dose"),
    SemanticField("medication", "route", "route of medication administration"),
    SemanticField("medication", "frequency", "medication administration frequency"),
    SemanticField("medication", "count_min", "minimum number of matching medication records"),
    SemanticField("medication", "start_time", "lower time bound for medication exposure"),
    SemanticField("medication", "end_time", "upper time bound for medication exposure"),
    SemanticField("medication", "exposure_mode", "whether any recorded exposure or exposure active at an index time is required"),
    SemanticField("patient", "attribute", "patient demographic/status attribute: age, gender, or deceased status"),
    SemanticField("patient", "operator", "comparison operator applied to the patient attribute"),
    SemanticField("patient", "value", "semantic value or threshold for the patient attribute"),
    SemanticField("admission", "admission_type", "type or class of hospital admission"),
    SemanticField("admission", "admission_location", "location or source from which the patient was admitted"),
    SemanticField("admission", "discharge_location", "destination or disposition location after discharge"),
    SemanticField("admission", "insurance", "insurance or payer category recorded for the admission"),
    SemanticField("admission", "language", "patient language recorded for the admission"),
    SemanticField("admission", "marital_status", "patient marital status recorded for the admission"),
    SemanticField("admission", "race", "race/ethnicity category recorded for the admission"),
    SemanticField("admission", "hospital_expire_flag", "whether the patient died during the hospitalization"),
    SemanticField("admission", "start_date", "lower calendar/date bound for admission start"),
    SemanticField("admission", "end_date", "upper calendar/date bound for admission/end timing"),
    SemanticField("admission", "duration_operator", "comparison operator for hospital length of stay"),
    SemanticField("admission", "duration_days", "hospital length-of-stay threshold in days"),
    SemanticField("admission", "count_min", "minimum number of qualifying admissions"),
    SemanticField("note", "claim", "clinical claim that must be supported by free-text note evidence"),
    SemanticField("note", "count_min", "minimum number of note-supported occurrences/admissions"),
)


def semantic_fields(domain: str | None = None) -> tuple[SemanticField, ...]:
    if domain is None:
        return _FIELDS
    key = domain.casefold().strip()
    return tuple(field for field in _FIELDS if field.domain == key)


def semantic_field_count() -> int:
    return len(_FIELDS)


def supported_domains() -> tuple[str, ...]:
    return tuple(dict.fromkeys(field.domain for field in _FIELDS))


def render_semantic_fields(domains: tuple[str, ...] | None = None) -> str:
    selected = _FIELDS if domains is None else tuple(
        field for domain in domains for field in semantic_fields(domain)
    )
    grouped: dict[str, list[SemanticField]] = {}
    for field in selected:
        grouped.setdefault(field.domain, []).append(field)
    lines = [
        "SUPPORTED SEMANTIC FIELDS (descriptions only. Preserve user values in natural clinical terminology):"
    ]
    for domain, fields in grouped.items():
        lines.append(f"{domain}:")
        for field in fields:
            lines.append(f"- {field.name}: {field.description}")
    return "\n".join(lines)
