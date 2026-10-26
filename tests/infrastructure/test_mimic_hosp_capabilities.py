import pytest

duckdb = pytest.importorskip("duckdb")

from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.mimic_hosp.capabilities import inspect_mimic_hosp_capabilities


def test_capabilities_reflect_present_tables(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE patients (subject_id BIGINT)")
        con.execute("CREATE TABLE admissions (subject_id BIGINT)")
        con.execute("CREATE TABLE diagnoses_icd (subject_id BIGINT)")
        con.execute("CREATE TABLE d_icd_diagnoses (icd_code VARCHAR)")
        con.execute("CREATE TABLE prescriptions (subject_id BIGINT)")

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    capabilities = inspect_mimic_hosp_capabilities(factory)

    assert capabilities.supports_conditions is True
    assert capabilities.supports_medications is True
    assert capabilities.supports_patient_attributes is True
    assert capabilities.supports_admissions is True
    assert capabilities.supports_active_medication is False


def test_capabilities_report_missing_tables(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE patients (subject_id BIGINT)")

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    capabilities = inspect_mimic_hosp_capabilities(factory)

    assert capabilities.supports_conditions is False
    assert capabilities.supports_medications is False
