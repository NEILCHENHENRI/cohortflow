import pytest

duckdb = pytest.importorskip("duckdb")

from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.mimic_hosp.presentation import MimicHospPatientDataProvider


def make_clinical(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE patients (
                subject_id BIGINT, gender VARCHAR, anchor_age INTEGER,
                anchor_year INTEGER, dod DATE
            )
        """)
        con.execute("""
            CREATE TABLE admissions (
                subject_id BIGINT, hadm_id BIGINT,
                admittime TIMESTAMP, dischtime TIMESTAMP, admission_type VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE diagnoses_icd (
                subject_id BIGINT, hadm_id BIGINT, seq_num INTEGER,
                icd_code VARCHAR, icd_version INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE d_icd_diagnoses (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, drug VARCHAR,
                starttime TIMESTAMP, stoptime TIMESTAMP
            )
        """)
        con.execute("INSERT INTO patients VALUES (1, 'M', 40, 2020, NULL)")
        con.execute(
            "INSERT INTO admissions VALUES (1, 501, '2020-01-01', '2020-01-03', 'URGENT')"
        )
        con.execute(
            "INSERT INTO diagnoses_icd VALUES (1, 501, 1, '25000', 9)"
        )
        con.execute(
            "INSERT INTO d_icd_diagnoses VALUES "
            "('25000', 9, 'Diabetes mellitus without mention of complication')"
        )
        con.execute(
            "INSERT INTO prescriptions VALUES (1, 501, 'Metformin', '2020-01-05', NULL)"
        )
    return path


def test_patient_detail_assembles_demographics_and_events(tmp_path):
    factory = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    provider = MimicHospPatientDataProvider(factory)

    detail = provider.patient_detail(1)

    assert detail["demographics"]["Patient ID"] == 1
    assert detail["demographics"]["Anchor age"] == 40
    assert detail["conditions"][0]["condition"] == (
        "Diabetes mellitus without mention of complication"
    )
    assert detail["medications"][0]["medication"] == "Metformin"
    assert detail["visits"][0]["visit_id"] == 501


def test_unknown_patient_raises_lookup_error(tmp_path):
    factory = DuckDBConnectionFactory(
        DuckDBConfig(make_clinical(tmp_path), read_only=True)
    )
    provider = MimicHospPatientDataProvider(factory)

    with pytest.raises(LookupError):
        provider.patient_detail(999)


def test_provider_works_when_dictionary_table_lives_in_a_non_default_schema(tmp_path):
    path = tmp_path / "mimic_hosp_namespaced.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA vocab")
        con.execute("""
            CREATE TABLE patients (
                subject_id BIGINT, gender VARCHAR, anchor_age INTEGER,
                anchor_year INTEGER, dod DATE
            )
        """)
        con.execute("""
            CREATE TABLE admissions (
                subject_id BIGINT, hadm_id BIGINT,
                admittime TIMESTAMP, dischtime TIMESTAMP, admission_type VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE diagnoses_icd (
                subject_id BIGINT, hadm_id BIGINT, seq_num INTEGER,
                icd_code VARCHAR, icd_version INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE vocab.d_icd_diagnoses (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, drug VARCHAR,
                starttime TIMESTAMP, stoptime TIMESTAMP
            )
        """)
        con.execute("INSERT INTO patients VALUES (1, 'M', 40, 2020, NULL)")
        con.execute(
            "INSERT INTO admissions VALUES (1, 501, '2020-01-01', '2020-01-03', 'URGENT')"
        )
        con.execute("INSERT INTO diagnoses_icd VALUES (1, 501, 1, '25000', 9)")
        con.execute(
            "INSERT INTO vocab.d_icd_diagnoses VALUES "
            "('25000', 9, 'Diabetes mellitus without mention of complication')"
        )
        con.execute(
            "INSERT INTO prescriptions VALUES (1, 501, 'Metformin', '2020-01-05', NULL)"
        )

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    provider = MimicHospPatientDataProvider(factory)

    detail = provider.patient_detail(1)

    assert detail["conditions"][0]["condition"] == (
        "Diabetes mellitus without mention of complication"
    )
