from __future__ import annotations

from typing import Any

from infrastructure.database.duckdb import DuckDBConnectionFactory


class MimicHospPatientDataProvider:
    """Patient-detail lookups against raw MIMIC-IV hosp-module tables."""

    def __init__(
        self,
        clinical: DuckDBConnectionFactory,
        *,
        event_limit: int = 100,
    ) -> None:
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        self.clinical = clinical
        self.event_limit = event_limit
        self._patients = clinical.qualified_table_name("patients")
        self._admissions = clinical.qualified_table_name("admissions")
        self._diagnoses_icd = clinical.qualified_table_name("diagnoses_icd")
        self._prescriptions = clinical.qualified_table_name("prescriptions")
        self._d_icd_diagnoses = clinical.qualified_table_name("d_icd_diagnoses")

    def patient_detail(self, patient_id: int) -> dict[str, Any]:
        rows = self.clinical.execute(
            f"""
            SELECT TRY_CAST(subject_id AS BIGINT), gender,
                   anchor_age, anchor_year, dod
            FROM {self._patients}
            WHERE TRY_CAST(subject_id AS BIGINT) = ?
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
                "Gender": row[1],
                "Anchor age": row[2],
                "Anchor year": row[3],
                "Date of death": row[4],
            },
            "conditions": self._conditions(patient_id),
            "medications": self._medications(patient_id),
            "visits": self._visits(patient_id),
        }

    def _conditions(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            f"""
            SELECT d.icd_code, d.icd_version, dd.long_title, a.admittime
            FROM {self._diagnoses_icd} AS d
            JOIN {self._admissions} AS a
              ON TRY_CAST(d.subject_id AS BIGINT) = TRY_CAST(a.subject_id AS BIGINT)
             AND TRY_CAST(d.hadm_id AS BIGINT) = TRY_CAST(a.hadm_id AS BIGINT)
            LEFT JOIN {self._d_icd_diagnoses} AS dd
              ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
            WHERE TRY_CAST(d.subject_id AS BIGINT) = ?
            ORDER BY 4 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "icd_code": row[0],
                "icd_version": row[1],
                "condition": row[2],
                "admission_date": row[3],
            }
            for row in rows
        ]

    def _medications(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            f"""
            SELECT drug, starttime, stoptime
            FROM {self._prescriptions}
            WHERE TRY_CAST(subject_id AS BIGINT) = ?
            ORDER BY 2 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "medication": row[0],
                "start_date": row[1],
                "end_date": row[2],
            }
            for row in rows
        ]

    def _visits(self, patient_id: int) -> list[dict[str, Any]]:
        rows = self.clinical.execute(
            f"""
            SELECT TRY_CAST(hadm_id AS BIGINT), admission_type,
                   admittime, dischtime
            FROM {self._admissions}
            WHERE TRY_CAST(subject_id AS BIGINT) = ?
            ORDER BY 3 NULLS LAST
            LIMIT ?
            """,
            [patient_id, self.event_limit],
        )
        return [
            {
                "visit_id": row[0],
                "admission_type": row[1],
                "start_date": row[2],
                "end_date": row[3],
            }
            for row in rows
        ]
