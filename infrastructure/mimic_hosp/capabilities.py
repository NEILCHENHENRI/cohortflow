from __future__ import annotations

from infrastructure.capabilities import DataCapabilities
from infrastructure.database.duckdb import DuckDBConnectionFactory


def inspect_mimic_hosp_capabilities(clinical: DuckDBConnectionFactory) -> DataCapabilities:
    has = clinical.table_exists
    return DataCapabilities(
        supports_conditions=has("diagnoses_icd") and has("d_icd_diagnoses"),
        supports_procedures=has("procedures_icd") and has("d_icd_procedures"),
        supports_labs=has("labevents") and has("d_labitems"),
        supports_medications=has("prescriptions"),
        supports_patient_attributes=has("patients") and has("admissions"),
        supports_admissions=has("admissions"),
        supports_notes=has("discharge"),
        supports_active_medication=False,
    )
