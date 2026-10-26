from __future__ import annotations

from typing import Any

from infrastructure.database.duckdb import DuckDBConnectionFactory


class OMOPPatientDataProvider:
    def __init__(
        self,
        clinical: DuckDBConnectionFactory,
        vocabulary: DuckDBConnectionFactory | None = None,
        *,
        event_limit: int = 100,
    ) -> None:
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        self.clinical = clinical
        self.vocabulary = vocabulary
        self.event_limit = event_limit

    def patient_detail(self, patient_id: int) -> dict[str, Any]:
        rows = self.clinical.execute(
            """
            SELECT TRY_CAST(person_id AS BIGINT), gender_concept_id,
                   year_of_birth, month_of_birth, day_of_birth,
                   race_concept_id, ethnicity_concept_id
            FROM person
            WHERE TRY_CAST(person_id AS BIGINT) = ?
            LIMIT 1
            """,
            [patient_id],
        )
        if not rows:
            raise LookupError(f"patient {patient_id} does not exist")
        row = rows[0]
        return {
            "demographics": {
                "Patient ID": int(row[0]),
                "Gender concept ID": row[1],
                "Year of birth": row[2],
                "Month of birth": row[3],
                "Day of birth": row[4],
                "Race concept ID": row[5],
                "Ethnicity concept ID": row[6],
            },
            "conditions": self._conditions(patient_id),
            "medications": self._medications(patient_id),
            "visits": self._visits(patient_id),
        }

    def _conditions(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            """
            SELECT TRY_CAST(condition_concept_id AS BIGINT),
                   COALESCE(condition_start_date,
                            CAST(condition_start_datetime AS DATE)),
                   COALESCE(condition_end_date,
                            CAST(condition_end_datetime AS DATE))
            FROM condition_occurrence
            WHERE TRY_CAST(person_id AS BIGINT) = ?
            ORDER BY 2 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "concept_id": row[0],
                "concept_name": self._concept_name(row[0]),
                "start_date": row[1],
                "end_date": row[2],
            }
            for row in rows
        ]

    def _medications(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            """
            SELECT TRY_CAST(drug_concept_id AS BIGINT),
                   COALESCE(drug_exposure_start_date,
                            CAST(drug_exposure_start_datetime AS DATE)),
                   COALESCE(drug_exposure_end_date,
                            CAST(drug_exposure_end_datetime AS DATE))
            FROM drug_exposure
            WHERE TRY_CAST(person_id AS BIGINT) = ?
            ORDER BY 2 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "concept_id": row[0],
                "concept_name": self._concept_name(row[0]),
                "start_date": row[1],
                "end_date": row[2],
            }
            for row in rows
        ]

    def _visits(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            """
            SELECT TRY_CAST(visit_occurrence_id AS BIGINT),
                   TRY_CAST(visit_concept_id AS BIGINT),
                   COALESCE(visit_start_date,
                            CAST(visit_start_datetime AS DATE)),
                   COALESCE(visit_end_date,
                            CAST(visit_end_datetime AS DATE))
            FROM visit_occurrence
            WHERE TRY_CAST(person_id AS BIGINT) = ?
            ORDER BY 3 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "visit_id": row[0],
                "concept_id": row[1],
                "concept_name": self._concept_name(row[1]),
                "start_date": row[2],
                "end_date": row[3],
            }
            for row in rows
        ]

    def _concept_name(self, concept_id: Any) -> str | None:
        if concept_id is None or self.vocabulary is None:
            return None
        rows = self.vocabulary.execute(
            """
            SELECT concept_name
            FROM concept
            WHERE TRY_CAST(concept_id AS BIGINT) = ?
            LIMIT 1
            """,
            [int(concept_id)],
        )
        return str(rows[0][0]) if rows else None
