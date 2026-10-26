from __future__ import annotations

from datetime import date
import re
from typing import Any

from framework.errors import UnsupportedCriterionError
from infrastructure.database.duckdb import DuckDBConnectionFactory
from infrastructure.mimic_hosp.concepts import MimicHospConcept
from agents.cohort.structured import validate_structured_payload


class MimicHospCohortRepository:
    """Read-only structured cohort execution over the 10 downloaded MIMIC files.

    Seven logical domains are exposed: patient, admission, condition,
    procedure, lab, medication, and discharge-note text. Dictionary tables are
    used only for semantic resolution. Fact tables are used for membership.

    The repository owns all physical column/table knowledge. Upstream LLM code
    emits domain semantics only and never generates SQL.
    """

    source_label = "MIMIC-IV hosp + note DuckDB"

    REQUIRED_TABLES = (
        "patients", "admissions",
        "diagnoses_icd", "d_icd_diagnoses",
        "procedures_icd", "d_icd_procedures",
        "labevents", "d_labitems",
        "prescriptions", "discharge",
    )

    def __init__(self, clinical: DuckDBConnectionFactory) -> None:
        self._clinical = clinical
        core = ("patients", "admissions")
        missing = [name for name in core if not clinical.table_exists(name)]
        if missing:
            raise ValueError(
                "clinical database is missing required MIMIC core tables: " + ", ".join(missing)
            )
        for name in self.REQUIRED_TABLES:
            setattr(
                self, f"_{name}",
                clinical.qualified_table_name(name) if clinical.table_exists(name) else None,
            )

    def all_patients(self) -> set[int]:
        return self._patient_set(
            f"SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) FROM {self._patients} "
            "WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL"
        )

    # ------------------------------------------------------------------
    # Public domain methods used by cohort operators.
    # ------------------------------------------------------------------
    def patients_with_condition(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        payload = {"term": "resolved", **(payload or {})}
        sql, params = self.compile_requirement_sql("condition", payload, concept)
        return self._patient_set(sql, params)

    def patients_with_procedure(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        payload = {"term": "resolved", **(payload or {})}
        sql, params = self.compile_requirement_sql("procedure", payload, concept)
        return self._patient_set(sql, params)

    def patients_with_lab(self, concept: Any, payload: dict[str, Any]) -> set[int]:
        sql, params = self.compile_requirement_sql("lab", payload, concept)
        return self._patient_set(sql, params)

    def patients_with_medication(self, concept: Any, payload: dict[str, Any] | None = None) -> set[int]:
        payload = {"term": "resolved", **(payload or {})}
        if payload.get("exposure_mode", "any_recorded") == "active_at_index":
            raise UnsupportedCriterionError(
                "active_at_index requires an explicit index event. Use a relation/scope rule"
            )
        sql, params = self.compile_requirement_sql("medication", payload, concept)
        return self._patient_set(sql, params)

    def patients_with_patient_attribute(self, payload: dict[str, Any]) -> set[int]:
        sql, params = self.compile_requirement_sql("patient", payload)
        return self._patient_set(sql, params)

    def patients_with_admission(self, payload: dict[str, Any]) -> set[int]:
        sql, params = self.compile_requirement_sql("admission", payload)
        return self._patient_set(sql, params)

    def patients_with_note(self, payload: dict[str, Any]) -> set[int]:
        sql, params = self.compile_requirement_sql("note", payload)
        return self._patient_set(sql, params)

    def admissions_matching_requirement(
        self,
        kind: str,
        payload: dict[str, Any],
        concept: Any | None = None,
    ) -> set[tuple[int, int]]:
        """Return admissions in which one structured requirement is satisfied.

        Occurrence counts are scoped to the admission here. This is intentionally
        distinct from the ordinary patient-level requirement query, where
        ``count_min`` counts across the patient's available history.
        """
        if kind == "patient":
            raise UnsupportedCriterionError("patient attributes are not admission-scoped events")
        event_payload = dict(payload)
        count_min = event_payload.pop("count_min", None)
        sql, params = self.event_sql(kind, event_payload, concept)
        if count_min is not None:
            rows = self._clinical.execute(
                "WITH events AS (" + sql + ") "
                "SELECT subject_id, hadm_id FROM events "
                "WHERE hadm_id IS NOT NULL GROUP BY 1,2 HAVING COUNT(*) >= ?",
                [*params, int(count_min)],
            )
        else:
            rows = self._clinical.execute(
                "WITH events AS (" + sql + ") "
                "SELECT DISTINCT subject_id, hadm_id FROM events WHERE hadm_id IS NOT NULL",
                params,
            )
        return {
            (int(subject_id), int(hadm_id))
            for subject_id, hadm_id in rows
            if subject_id is not None and hadm_id is not None
        }

    # ------------------------------------------------------------------
    # Deterministic structured representation -> SQL compiler.
    # ------------------------------------------------------------------
    def compile_requirement_sql(
        self,
        kind: str,
        payload: dict[str, Any],
        concept: Any | None = None,
    ) -> tuple[str, list[Any]]:
        validate_structured_payload(kind, payload)
        if kind == "condition":
            return self._compile_condition(payload, concept)
        if kind == "procedure":
            return self._compile_procedure(payload, concept)
        if kind == "lab":
            return self._compile_lab(payload, concept)
        if kind == "medication":
            return self._compile_medication(payload, concept)
        if kind == "patient":
            return self._compile_patient(payload)
        if kind == "admission":
            return self._compile_admission(payload)
        if kind == "note":
            return self._compile_note(payload)
        raise ValueError(f"no MIMIC SQL compiler for {kind!r}")

    def patients_satisfying_relation(
        self,
        *,
        relation: str,
        left_kind: str,
        left_payload: dict[str, Any],
        right_kind: str,
        right_payload: dict[str, Any],
        left_concept: Any | None = None,
        right_concept: Any | None = None,
        max_gap_hours: int | None = None,
        same_admission: bool = False,
    ) -> set[int]:
        """Generic BEFORE/AFTER relation across event-producing domains."""
        if relation not in {"before", "after"}:
            raise ValueError("relation must be before or after")
        if max_gap_hours is not None and max_gap_hours < 0:
            raise ValueError("max_gap_hours cannot be negative")
        lsql, lp = self.event_sql(left_kind, left_payload, left_concept)
        rsql, rp = self.event_sql(right_kind, right_payload, right_concept)
        op = "<" if relation == "before" else ">"
        clauses = [f"l.event_time {op} r.event_time"]
        if same_admission:
            clauses.append("l.hadm_id = r.hadm_id")
        params = [*lp, *rp]
        if max_gap_hours is not None:
            clauses.append("ABS(DATE_DIFF('hour', l.event_time, r.event_time)) <= ?")
            params.append(max_gap_hours)
        return self._patient_set(
            f"WITH l AS ({lsql}), r AS ({rsql}) "
            "SELECT DISTINCT l.subject_id FROM l JOIN r ON l.subject_id = r.subject_id "
            f"WHERE {' AND '.join(clauses)}",
            params,
        )

    def event_sql(
        self, kind: str, payload: dict[str, Any], concept: Any | None = None,
    ) -> tuple[str, list[Any]]:
        """Return subject_id, hadm_id, event_time for relation execution."""
        validate_structured_payload(kind, payload)
        if kind == "condition":
            # diagnoses_icd is joined to admissions for an event timestamp, so
            # every diagnoses column in the predicate must be qualified.
            where, params = self._condition_where(payload, concept, alias="d")
            return (
                f"SELECT TRY_CAST(d.subject_id AS BIGINT) subject_id, "
                "TRY_CAST(d.hadm_id AS BIGINT) hadm_id, a.admittime event_time "
                f"FROM {self._diagnoses_icd} d JOIN {self._admissions} a "
                "ON TRY_CAST(d.hadm_id AS BIGINT)=TRY_CAST(a.hadm_id AS BIGINT) "
                f"WHERE {where} AND a.admittime IS NOT NULL",
                params,
            )
        if kind == "procedure":
            where, params = self._procedure_where(payload, concept)
            return (
                "SELECT TRY_CAST(subject_id AS BIGINT) subject_id, "
                "TRY_CAST(hadm_id AS BIGINT) hadm_id, "
                "COALESCE(CAST(chartdate AS TIMESTAMP), CAST(NULL AS TIMESTAMP)) event_time "
                f"FROM {self._procedures_icd} WHERE {where} AND chartdate IS NOT NULL",
                params,
            )
        if kind == "lab":
            where, params = self._lab_where(payload, concept, include_value=True)
            return (
                "SELECT TRY_CAST(subject_id AS BIGINT) subject_id, "
                "TRY_CAST(hadm_id AS BIGINT) hadm_id, charttime event_time "
                f"FROM {self._labevents} WHERE {where} AND charttime IS NOT NULL",
                params,
            )
        if kind == "medication":
            where, params = self._medication_where(payload, concept)
            return (
                "SELECT TRY_CAST(subject_id AS BIGINT) subject_id, "
                "TRY_CAST(hadm_id AS BIGINT) hadm_id, starttime event_time "
                f"FROM {self._prescriptions} WHERE {where} AND starttime IS NOT NULL",
                params,
            )
        if kind == "admission":
            where, params = self._admission_where(payload)
            return (
                "SELECT TRY_CAST(subject_id AS BIGINT) subject_id, "
                "TRY_CAST(hadm_id AS BIGINT) hadm_id, admittime event_time "
                f"FROM {self._admissions} WHERE {where} AND admittime IS NOT NULL",
                params,
            )
        if kind == "note":
            where, params = self._note_where(payload)
            return (
                "SELECT TRY_CAST(subject_id AS BIGINT) subject_id, "
                "TRY_CAST(hadm_id AS BIGINT) hadm_id, charttime event_time "
                f"FROM {self._discharge} WHERE {where} AND charttime IS NOT NULL",
                params,
            )
        raise UnsupportedCriterionError(f"{kind!r} does not produce event evidence")

    # ------------------------------------------------------------------
    # Domain compilers.
    # ------------------------------------------------------------------
    def _compile_condition(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        where, params = self._condition_where(payload, concept)
        return self._select_with_count(self._diagnoses_icd, where, params, payload.get("count_min"))

    def _condition_where(
        self, payload: dict[str, Any], concept: Any, *, alias: str | None = None,
    ) -> tuple[str, list[Any]]:
        codes = _condition_codes(concept)
        prefix = f"{alias}." if alias else ""
        clauses, params = [
            _code_pair_clause(f"{prefix}icd_code", f"{prefix}icd_version", len(codes))
        ], _flatten(codes)
        if payload.get("seq_num") is not None:
            clauses.append(f"TRY_CAST({prefix}seq_num AS INTEGER) = ?")
            params.append(payload["seq_num"])
        clauses.append(f"TRY_CAST({prefix}subject_id AS BIGINT) IS NOT NULL")
        return " AND ".join(clauses), params

    def _compile_procedure(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        where, params = self._procedure_where(payload, concept)
        return self._select_with_count(self._procedures_icd, where, params, payload.get("count_min"))

    def _procedure_where(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        codes = _procedure_codes(concept)
        return (
            _code_pair_clause("icd_code", "icd_version", len(codes))
            + " AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL",
            _flatten(codes),
        )

    def _compile_lab(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        itemids = _lab_itemids(concept)
        aggregation = payload.get("aggregation", "any")
        count_min = payload.get("count_min")
        # For aggregate modes, compare the aggregate numeric value rather than
        # filtering rows first.
        if aggregation in {"min", "max", "avg"}:
            where, params = self._lab_where(payload, concept, include_value=False)
            op, value = payload.get("operator"), payload.get("value")
            if op is None or value is None:
                raise ValueError(f"lab aggregation={aggregation} requires operator and value")
            return (
                f"SELECT TRY_CAST(subject_id AS BIGINT) subject_id FROM {self._labevents} "
                f"WHERE {where} GROUP BY 1 HAVING {aggregation.upper()}(valuenum) {op} ?",
                [*params, value],
            )
        where, params = self._lab_where(payload, concept, include_value=True)
        if aggregation == "latest":
            return (
                "WITH ranked AS (SELECT TRY_CAST(subject_id AS BIGINT) subject_id, valuenum, "
                f"ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY charttime DESC) rn FROM {self._labevents} "
                f"WHERE {where}) SELECT DISTINCT subject_id FROM ranked WHERE rn=1",
                params,
            )
        return self._select_with_count(self._labevents, where, params, count_min)

    def _lab_where(self, payload: dict[str, Any], concept: Any, *, include_value: bool) -> tuple[str, list[Any]]:
        itemids = _lab_itemids(concept)
        placeholders = ",".join("?" for _ in itemids)
        clauses = [f"TRY_CAST(itemid AS BIGINT) IN ({placeholders})", "TRY_CAST(subject_id AS BIGINT) IS NOT NULL"]
        params: list[Any] = list(itemids)
        if include_value and payload.get("operator") is not None:
            clauses.append(f"valuenum {payload['operator']} ?")
            params.append(payload["value"])
        if payload.get("value_text"):
            clauses.append("lower(value) = lower(?)")
            params.append(payload["value_text"])
        if payload.get("unit"):
            clauses.append("lower(valueuom) = lower(?)")
            params.append(payload["unit"])
        if payload.get("abnormal_flag"):
            clauses.append("lower(flag) = lower(?)")
            params.append(payload["abnormal_flag"])
        if payload.get("start_time"):
            clauses.append("charttime >= CAST(? AS TIMESTAMP)")
            params.append(payload["start_time"])
        if payload.get("end_time"):
            clauses.append("charttime <= CAST(? AS TIMESTAMP)")
            params.append(payload["end_time"])
        return " AND ".join(clauses), params

    def _compile_medication(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        where, params = self._medication_where(payload, concept)
        return self._select_with_count(self._prescriptions, where, params, payload.get("count_min"))

    def _medication_where(self, payload: dict[str, Any], concept: Any) -> tuple[str, list[Any]]:
        names = _medication_names(concept)
        placeholders = ",".join("?" for _ in names)
        clauses = [f"lower(drug) IN ({placeholders})", "TRY_CAST(subject_id AS BIGINT) IS NOT NULL"]
        params: list[Any] = [name.casefold() for name in names]
        if payload.get("dose_operator"):
            clauses.append(f"TRY_CAST(dose_val_rx AS DOUBLE) {payload['dose_operator']} ?")
            params.append(payload["dose_value"])
        if payload.get("dose_unit"):
            clauses.append("lower(dose_unit_rx) = lower(?)")
            params.append(payload["dose_unit"])
        if payload.get("route"):
            routes = _medication_route_aliases(str(payload["route"]))
            placeholders = ",".join("?" for _ in routes)
            clauses.append(f"lower(route) IN ({placeholders})")
            params.extend(route.casefold() for route in routes)
        if payload.get("frequency"):
            clauses.append("lower(frequency) = lower(?)")
            params.append(payload["frequency"])
        if payload.get("start_time"):
            clauses.append("starttime >= CAST(? AS TIMESTAMP)")
            params.append(payload["start_time"])
        if payload.get("end_time"):
            clauses.append("starttime <= CAST(? AS TIMESTAMP)")
            params.append(payload["end_time"])
        return " AND ".join(clauses), params

    def _compile_patient(self, payload: dict[str, Any]) -> tuple[str, list[Any]]:
        attr, op, value = payload["attribute"], payload["operator"], payload["value"]
        if attr == "gender":
            normalized = {"male": "M", "female": "F"}.get(str(value).casefold(), str(value))
            return (
                f"SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) FROM {self._patients} "
                f"WHERE upper(gender) {op} upper(?) AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL",
                [normalized],
            )
        if attr == "deceased":
            expected = bool(value)
            expression = "dod IS NOT NULL" if expected else "dod IS NULL"
            if op == "!=":
                expression = "dod IS NULL" if expected else "dod IS NOT NULL"
            elif op != "=":
                raise ValueError("deceased supports only = or !=")
            return (
                f"SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) FROM {self._patients} "
                f"WHERE {expression} AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL",
                [],
            )
        return (
            self._age_cte_sql()
            + f" SELECT DISTINCT subject_id FROM ages WHERE age_at_first_visit {op} ?",
            [value],
        )

    def _compile_admission(self, payload: dict[str, Any]) -> tuple[str, list[Any]]:
        where, params = self._admission_where(payload)
        return self._select_with_count(self._admissions, where, params, payload.get("count_min"))

    def _admission_where(self, payload: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = ["TRY_CAST(subject_id AS BIGINT) IS NOT NULL"]
        params: list[Any] = []
        for key in (
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race",
        ):
            if payload.get(key) is not None:
                value = payload[key]
                if key == "admission_type" and str(value).strip().casefold() == "emergency":
                    clauses.append("lower(admission_type) LIKE '%emer%'")
                else:
                    clauses.append(f"lower({key}) = lower(?)")
                    params.append(value)
        if "hospital_expire_flag" in payload:
            clauses.append("TRY_CAST(hospital_expire_flag AS INTEGER) = ?")
            params.append(int(bool(payload["hospital_expire_flag"])))
        if payload.get("start_date"):
            clauses.append("CAST(admittime AS DATE) >= CAST(? AS DATE)")
            params.append(payload["start_date"])
        if payload.get("end_date"):
            clauses.append("CAST(admittime AS DATE) <= CAST(? AS DATE)")
            params.append(payload["end_date"])
        if payload.get("duration_operator"):
            clauses.append(
                f"DATE_DIFF('hour', admittime, dischtime) / 24.0 {payload['duration_operator']} ?"
            )
            params.append(payload["duration_days"])
        return " AND ".join(clauses), params

    def _compile_note(self, payload: dict[str, Any]) -> tuple[str, list[Any]]:
        where, params = self._note_where(payload)
        return self._select_with_count(self._discharge, where, params, payload.get("count_min"))

    def _note_where(self, payload: dict[str, Any]) -> tuple[str, list[Any]]:
        query = str(payload.get("claim") or payload.get("query") or "").strip()
        if not query:
            raise ValueError("note requirement requires claim")
        clauses = ["TRY_CAST(subject_id AS BIGINT) IS NOT NULL"]
        params: list[Any] = []
        section = payload.get("section")
        if section:
            # Discharge summaries are semi-structured. Restrict a textual
            # query to the section beginning at the requested heading and
            # ending at the next blank-line heading-like block (best-effort,
            # deterministic RE2 matching. No LLM required).
            pattern = rf"{re.escape(section)}\s*:\s*[\s\S]*?{re.escape(query)}"
            clauses.append("regexp_matches(text, ?, 'i')")
            params.append(pattern)
        else:
            clauses.append("lower(text) LIKE lower(?)")
            params.append(f"%{query}%")
        return " AND ".join(clauses), params

    def _select_with_count(
        self, table: str, where: str, params: list[Any], count_min: int | None,
    ) -> tuple[str, list[Any]]:
        if count_min is not None:
            return (
                f"SELECT TRY_CAST(subject_id AS BIGINT) subject_id FROM {table} "
                f"WHERE {where} GROUP BY 1 HAVING COUNT(*) >= ?",
                [*params, count_min],
            )
        return (
            f"SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) subject_id FROM {table} WHERE {where}",
            params,
        )

    def _age_cte_sql(self) -> str:
        return f"""
        WITH first_admission AS (
            SELECT TRY_CAST(subject_id AS BIGINT) subject_id, MIN(admittime) first_admittime
            FROM {self._admissions} GROUP BY 1
        ), ages AS (
            SELECT TRY_CAST(p.subject_id AS BIGINT) subject_id,
                   TRY_CAST(p.anchor_age AS INTEGER)
                     + (EXTRACT(YEAR FROM f.first_admittime)::INTEGER - TRY_CAST(p.anchor_year AS INTEGER))
                     AS age_at_first_visit
            FROM {self._patients} p JOIN first_admission f
              ON TRY_CAST(p.subject_id AS BIGINT)=f.subject_id
            WHERE p.anchor_age IS NOT NULL AND p.anchor_year IS NOT NULL
              AND f.first_admittime IS NOT NULL
        )
        """

    def _patient_set(self, sql: str, parameters: list[Any] | tuple[Any, ...] = ()) -> set[int]:
        rows = self._clinical.execute(sql, parameters)
        return {int(row[0]) for row in rows if row and row[0] is not None}


def _medication_route_aliases(route: str) -> tuple[str, ...]:
    normalized = route.strip().casefold()
    if normalized in {"oral", "by mouth"}:
        return ("PO", "ORAL", "P.O.")
    if normalized == "intravenous":
        return ("IV", "INTRAVENOUS", "I.V.")
    return (route.strip(),)


def _condition_codes(concept: Any) -> tuple[tuple[str, int], ...]:
    if isinstance(concept, MimicHospConcept):
        return concept.condition_codes
    if isinstance(concept, dict) and "icd_code" in concept:
        return ((str(concept["icd_code"]), int(concept.get("icd_version", 10))),)
    raise TypeError("condition concept must provide ICD code/version")


def _procedure_codes(concept: Any) -> tuple[tuple[str, int], ...]:
    if isinstance(concept, MimicHospConcept):
        return concept.procedure_codes
    if isinstance(concept, dict) and "icd_code" in concept:
        return ((str(concept["icd_code"]), int(concept.get("icd_version", 10))),)
    raise TypeError("procedure concept must provide ICD code/version")


def _lab_itemids(concept: Any) -> tuple[int, ...]:
    if isinstance(concept, MimicHospConcept):
        return concept.lab_itemids
    if isinstance(concept, dict) and "itemid" in concept:
        return (int(concept["itemid"]),)
    if isinstance(concept, int):
        return (concept,)
    raise TypeError("lab concept must provide itemid")


def _medication_names(concept: Any) -> tuple[str, ...]:
    if isinstance(concept, MimicHospConcept):
        return concept.medication_names
    if isinstance(concept, str):
        return (concept,)
    raise TypeError("medication concept must provide drug name")


def _code_pair_clause(code_col: str, version_col: str, count: int) -> str:
    if count < 1:
        raise ValueError("at least one code is required")
    return "(" + " OR ".join(f"({code_col} = ? AND {version_col} = ?)" for _ in range(count)) + ")"


def _flatten(pairs: tuple[tuple[Any, Any], ...]) -> list[Any]:
    return [value for pair in pairs for value in pair]
