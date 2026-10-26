from __future__ import annotations

from infrastructure.capabilities import DataCapabilities
from infrastructure.database.duckdb import DuckDBConnectionFactory

__all__ = ["DataCapabilities", "inspect_omop_capabilities"]


def inspect_omop_capabilities(
    clinical: DuckDBConnectionFactory,
) -> DataCapabilities:
    has = clinical.table_exists
    return DataCapabilities(
        supports_conditions=has("condition_occurrence"),
        supports_medications=has("drug_exposure"),
        supports_patient_attributes=has("person") and has("visit_occurrence"),
        supports_admissions=has("visit_occurrence"),
        supports_procedures=False,
        supports_labs=False,
        supports_notes=False,
    )
