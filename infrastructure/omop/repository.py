from __future__ import annotations

from datetime import date
from typing import Any

from infrastructure.database.duckdb import DuckDBConnectionFactory
from infrastructure.omop.concepts import OMOPConcept


class DuckDBCohortRepository:
    """Read-only OMOP CDM cohort queries backed by DuckDB.

    Each method opens its own connection through the factory, making the
    adapter safe to use from independent runtime worker threads.
    """

    source_label = "OMOP clinical DuckDB"

    def __init__(
        self,
        clinical: DuckDBConnectionFactory,
    ) -> None:
        self._clinical = clinical
        required = {
            "person",
            "condition_occurrence",
            "drug_exposure",
            "visit_occurrence",
        }
        missing = {
            table
            for table in required
            if not self._clinical.table_exists(table)
        }
        if missing:
            raise ValueError(
                "clinical database is missing OMOP tables: "
                + ", ".join(sorted(missing))
            )

    def all_patients(self) -> set[int]:
        return self._patient_set(
            """
            SELECT DISTINCT TRY_CAST(person_id AS BIGINT)
            FROM person
            WHERE TRY_CAST(person_id AS BIGINT) IS NOT NULL
            """
        )

    def patients_with_condition(
        self, concept: Any, payload: dict[str, Any] | None = None,
    ) -> set[int]:
        return self._patients_with_concept(
            concept, payload, table="condition_occurrence",
            concept_column="condition_concept_id",
            date_column="condition_start_date",
            datetime_column="condition_start_datetime",
        )

    def patients_with_medication(
        self, concept: Any, payload: dict[str, Any] | None = None,
    ) -> set[int]:
        values = dict(payload or {})
        exposure_mode = values.pop("exposure_mode", "any_recorded")
        if exposure_mode != "any_recorded":
            raise ValueError("OMOP demo medication supports only any_recorded exposure")
        return self._patients_with_concept(
            concept, values, table="drug_exposure",
            concept_column="drug_concept_id",
            date_column="drug_exposure_start_date",
            datetime_column="drug_exposure_start_datetime",
        )

    def _patients_with_concept(
        self, concept, payload, *, table, concept_column, date_column, datetime_column,
    ) -> set[int]:
        values = payload or {}
        unsupported = set(values) - {"term", "count_min", "start_time", "end_time"}
        if unsupported:
            raise ValueError(
                "OMOP demo does not support fields: " + ", ".join(sorted(unsupported))
            )
        clauses = [
            f"TRY_CAST({concept_column} AS BIGINT) = ?",
            "TRY_CAST(person_id AS BIGINT) IS NOT NULL",
        ]
        parameters = [_concept_id(concept)]
        for field, operator in (("start_time", ">="), ("end_time", "<=")):
            if values.get(field) is not None:
                clauses.append(
                    f"COALESCE({date_column}, CAST({datetime_column} AS DATE)) {operator} ?"
                )
                parameters.append(_date_value(values[field], field))
        minimum = values.get("count_min", 1)
        minimum = 1 if minimum is None else _int_value(minimum, "count_min")
        if minimum < 0:
            raise ValueError("count_min cannot be negative")
        parameters.append(minimum)
        return self._patient_set(
            f"SELECT TRY_CAST(person_id AS BIGINT) FROM {table} WHERE "
            + " AND ".join(clauses) + " GROUP BY 1 HAVING COUNT(*) >= ?",
            parameters,
        )

    def _patients_in_age_range(
        self,
        minimum: int | None,
        maximum: int | None,
    ) -> set[int]:
        if minimum is None and maximum is None:
            return self._patient_set(
                '''
                SELECT DISTINCT TRY_CAST(person_id AS BIGINT)
                FROM person
                WHERE TRY_CAST(person_id AS BIGINT) IS NOT NULL
                '''
            )
        if minimum is not None and minimum < 0:
            raise ValueError("minimum age cannot be negative")
        if maximum is not None and maximum < 0:
            raise ValueError("maximum age cannot be negative")
        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError("minimum age cannot exceed maximum age")

        clauses = []
        parameters: list[Any] = []
        if minimum is not None:
            clauses.append("age_at_first_visit >= ?")
            parameters.append(minimum)
        if maximum is not None:
            clauses.append("age_at_first_visit <= ?")
            parameters.append(maximum)

        return self._patient_set(
            f'''
            WITH first_visits AS (
                SELECT
                    TRY_CAST(person_id AS BIGINT) AS person_id,
                    MIN(
                        COALESCE(
                            visit_start_date,
                            CAST(visit_start_datetime AS DATE)
                        )
                    ) AS first_visit
                FROM visit_occurrence
                GROUP BY 1
            ),
            ages AS (
                SELECT
                    TRY_CAST(p.person_id AS BIGINT) AS person_id,
                    EXTRACT(YEAR FROM fv.first_visit)
                        - TRY_CAST(p.year_of_birth AS INTEGER)
                        - CASE
                            WHEN p.month_of_birth IS NOT NULL
                             AND p.day_of_birth IS NOT NULL
                             AND MAKE_DATE(
                                 EXTRACT(YEAR FROM fv.first_visit)::INTEGER,
                                 p.month_of_birth::INTEGER,
                                 p.day_of_birth::INTEGER
                             ) > fv.first_visit
                            THEN 1
                            ELSE 0
                          END AS age_at_first_visit
                FROM person AS p
                JOIN first_visits AS fv
                  ON TRY_CAST(p.person_id AS BIGINT) = fv.person_id
                WHERE p.year_of_birth IS NOT NULL
                  AND fv.first_visit IS NOT NULL
            )
            SELECT DISTINCT person_id
            FROM ages
            WHERE {' AND '.join(clauses)}
              AND person_id IS NOT NULL
            ''',
            parameters,
        )

    def patients_with_admission(
        self,
        payload: dict[str, Any],
    ) -> set[int]:
        unsupported = set(payload) - {
            "visit_concept_id", "start_date", "end_date",
            "duration_operator", "duration_days", "count_min"
        }
        if unsupported:
            raise ValueError(
                "OMOP admission adapter does not support fields: "
                + ", ".join(sorted(unsupported))
            )
        clauses = ["TRY_CAST(person_id AS BIGINT) IS NOT NULL"]
        parameters: list[Any] = []
        if payload.get("visit_concept_id") is not None:
            clauses.append("TRY_CAST(visit_concept_id AS BIGINT) = ?")
            parameters.append(int(payload["visit_concept_id"]))
        if payload.get("start_date") is not None:
            clauses.append("COALESCE(visit_start_date, CAST(visit_start_datetime AS DATE)) >= ?")
            parameters.append(_date_value(payload["start_date"], "start_date"))
        if payload.get("end_date") is not None:
            clauses.append("COALESCE(visit_start_date, CAST(visit_start_datetime AS DATE)) <= ?")
            parameters.append(_date_value(payload["end_date"], "end_date"))
        if payload.get("duration_operator") is not None:
            op = payload["duration_operator"]
            if op not in {"=", "!=", ">", ">=", "<", "<="}:
                raise ValueError("invalid duration operator")
            clauses.append(
                f"DATE_DIFF('day', COALESCE(visit_start_date, CAST(visit_start_datetime AS DATE)), "
                f"COALESCE(visit_end_date, CAST(visit_end_datetime AS DATE))) {op} ?"
            )
            parameters.append(float(payload["duration_days"]))
        count_min = payload.get("count_min")
        if count_min is not None:
            sql = (
                "SELECT TRY_CAST(person_id AS BIGINT) FROM visit_occurrence WHERE "
                + " AND ".join(clauses)
                + " GROUP BY 1 HAVING COUNT(*) >= ?"
            )
            parameters.append(int(count_min))
            return self._patient_set(sql, parameters)
        return self._patient_set(
            "SELECT DISTINCT TRY_CAST(person_id AS BIGINT) FROM visit_occurrence WHERE "
            + " AND ".join(clauses), parameters
        )

    def patients_with_patient_attribute(self, payload: dict[str, Any]) -> set[int]:
        attribute = payload.get("attribute")
        operator = payload.get("operator")
        value = payload.get("value")
        if operator not in {"=", "!=", ">", ">=", "<", "<="}:
            raise ValueError("invalid patient comparator")
        if attribute == "age":
            numeric = int(value)
            if operator == ">=":
                return self._patients_in_age_range(numeric, None)
            if operator == ">":
                return self._patients_in_age_range(numeric + 1, None)
            if operator == "<=":
                return self._patients_in_age_range(None, numeric)
            if operator == "<":
                return self._patients_in_age_range(None, numeric - 1)
            if operator == "=":
                return self._patients_in_age_range(numeric, numeric)
            universe = self.all_patients()
            return universe - self._patients_in_age_range(numeric, numeric)
        if attribute == "gender":
            normalized = str(value).strip().lower()
            concept_id = {"male": 8507, "m": 8507, "female": 8532, "f": 8532}.get(normalized)
            if concept_id is None or operator not in {"=", "!="}:
                raise ValueError("OMOP gender supports only equality with male/female")
            matched = self._patient_set(
                "SELECT DISTINCT TRY_CAST(person_id AS BIGINT) FROM person "
                "WHERE TRY_CAST(gender_concept_id AS BIGINT) = ?", [concept_id]
            )
            return matched if operator == "=" else self.all_patients() - matched
        if attribute == "deceased":
            raise ValueError("OMOP deceased attribute requires a death table adapter")
        raise ValueError(f"unsupported patient attribute: {attribute!r}")

    def _patient_set(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] = (),
    ) -> set[int]:
        rows = self._clinical.execute(sql, parameters)
        return {
            int(row[0])
            for row in rows
            if row and row[0] is not None
        }


def _concept_id(concept: Any) -> int:
    if isinstance(concept, OMOPConcept):
        return concept.concept_id
    if isinstance(concept, int) and not isinstance(concept, bool):
        return concept
    value = getattr(concept, "concept_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(
        "concept must be an OMOPConcept, integer ID, or object with "
        "an integer concept_id"
    )


def _event_spec(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} event must be an object")
    table = value.get("table")
    allowed = {"condition_occurrence", "drug_exposure"}
    if table not in allowed:
        raise ValueError(
            f"{field}.table must be one of {sorted(allowed)}"
        )
    concept_id = _int_value(
        value.get("concept_id"),
        f"{field}.concept_id",
    )
    return {"table": table, "concept_id": concept_id}


def _event_sql(spec: dict[str, Any], alias: str) -> str:
    if spec["table"] == "condition_occurrence":
        return f'''
            SELECT
                TRY_CAST(person_id AS BIGINT) AS person_id,
                COALESCE(
                    condition_start_date,
                    CAST(condition_start_datetime AS DATE)
                ) AS event_date
            FROM condition_occurrence
            WHERE TRY_CAST(condition_concept_id AS BIGINT) = ?
              AND TRY_CAST(person_id AS BIGINT) IS NOT NULL
              AND COALESCE(
                    condition_start_date,
                    CAST(condition_start_datetime AS DATE)
                  ) IS NOT NULL
        '''
    return f'''
        SELECT
            TRY_CAST(person_id AS BIGINT) AS person_id,
            COALESCE(
                drug_exposure_start_date,
                CAST(drug_exposure_start_datetime AS DATE)
            ) AS event_date
        FROM drug_exposure
        WHERE TRY_CAST(drug_concept_id AS BIGINT) = ?
          AND TRY_CAST(person_id AS BIGINT) IS NOT NULL
          AND COALESCE(
                drug_exposure_start_date,
                CAST(drug_exposure_start_datetime AS DATE)
              ) IS NOT NULL
    '''


def _int_value(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _date_value(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{field} must use YYYY-MM-DD format"
            ) from exc
    raise ValueError(f"{field} must be a date or YYYY-MM-DD string")
