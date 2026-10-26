from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from framework.errors import UnsupportedCriterionError
from infrastructure.database.duckdb import (
    DuckDBConfig,
    DuckDBConnectionFactory,
)
from infrastructure.mimic_hosp.concepts import MimicHospConcept, MimicHospConceptEntry
from infrastructure.mimic_hosp.repository import MimicHospCohortRepository


def make_clinical(tmp_path):
    path = tmp_path / "mimic_hosp.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute('''
            CREATE TABLE patients (
                subject_id BIGINT,
                gender VARCHAR,
                anchor_age INTEGER,
                anchor_year INTEGER,
                dod DATE
            )
        ''')
        con.execute('''
            CREATE TABLE admissions (
                subject_id BIGINT,
                hadm_id BIGINT,
                admittime TIMESTAMP,
                dischtime TIMESTAMP,
                admission_type VARCHAR
            )
        ''')
        con.execute('''
            CREATE TABLE diagnoses_icd (
                subject_id BIGINT,
                hadm_id BIGINT,
                seq_num INTEGER,
                icd_code VARCHAR,
                icd_version INTEGER
            )
        ''')
        con.execute('''
            CREATE TABLE d_icd_diagnoses (
                icd_code VARCHAR,
                icd_version INTEGER,
                long_title VARCHAR
            )
        ''')
        con.execute('''
            CREATE TABLE prescriptions (
                subject_id BIGINT,
                hadm_id BIGINT,
                drug VARCHAR,
                starttime TIMESTAMP,
                stoptime TIMESTAMP
            )
        ''')
        con.executemany(
            "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
            [
                (1, "M", 40, 2020, None),
                (2, "F", 10, 2020, None),
                (3, "F", 50, 2020, None),
            ],
        )
        con.executemany(
            "INSERT INTO admissions VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, "2020-01-01 08:00:00", "2020-01-03 08:00:00", "URGENT"),
                (2, 502, "2020-01-01 08:00:00", "2020-01-03 08:00:00", "URGENT"),
                (3, 503, "2020-01-01 08:00:00", "2020-01-03 08:00:00", "ELECTIVE"),
            ],
        )
        con.executemany(
            "INSERT INTO diagnoses_icd VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, 1, "25000", 9),
                (2, 502, 1, "25000", 9),
                (3, 503, 1, "I10", 10),
            ],
        )
        con.executemany(
            "INSERT INTO d_icd_diagnoses VALUES (?, ?, ?)",
            [
                ("25000", 9, "Diabetes mellitus without mention of complication"),
                ("I10", 10, "Essential (primary) hypertension"),
            ],
        )
        con.executemany(
            "INSERT INTO prescriptions VALUES (?, ?, ?, ?, ?)",
            [
                (1, 501, "Metformin", "2020-01-05 09:00:00", None),
                # Patient 2's prescription predates their diagnosis, so it
                # must NOT satisfy a "diagnosis before medication" rule.
                (2, 502, "Metformin", "2018-01-01 09:00:00", None),
                (3, 503, "Lisinopril", "2020-01-01 09:00:00", None),
            ],
        )
    return path


def repository(tmp_path):
    return MimicHospCohortRepository(
        DuckDBConnectionFactory(
            DuckDBConfig(make_clinical(tmp_path), read_only=True)
        )
    )


def condition(icd_code, icd_version):
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=icd_code,
            display_name="x",
            vocabulary_id="ICD9CM" if icd_version == 9 else "ICD10CM",
        ),),
        domain_id="Condition",
    )


def conditions(*pairs):
    return MimicHospConcept(
        entries=tuple(
            MimicHospConceptEntry(
                code=icd_code,
                display_name="x",
                vocabulary_id="ICD9CM" if icd_version == 9 else "ICD10CM",
            )
            for icd_code, icd_version in pairs
        ),
        domain_id="Condition",
    )


def medication(drug):
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(
            code=drug, display_name=drug, vocabulary_id="prescriptions",
        ),),
        domain_id="Drug",
    )


def test_all_patients(tmp_path):
    assert repository(tmp_path).all_patients() == {1, 2, 3}


def test_condition_and_medication_queries(tmp_path):
    repo = repository(tmp_path)

    assert repo.patients_with_condition(condition("25000", 9)) == {1, 2}
    assert repo.patients_with_condition(condition("I10", 10)) == {3}
    assert repo.patients_with_medication(medication("Metformin")) == {1, 2}
    assert repo.patients_with_medication(medication("metformin")) == {1, 2}


def test_condition_with_multiple_codes_matches_via_or(tmp_path):
    repo = repository(tmp_path)

    # Patient 1/2 are coded ICD-9 25000, patient 3 is coded ICD-10 I10. A
    # concept spanning both codes (e.g. a human picked both from a
    # disambiguation list) must match the union, not require both codes on
    # the same patient.
    combined = conditions(("25000", 9), ("I10", 10))
    assert repo.patients_with_condition(combined) == {1, 2, 3}


def test_medication_with_multiple_names_matches_via_or(tmp_path):
    repo = repository(tmp_path)

    combined = MimicHospConcept(
        entries=(
            MimicHospConceptEntry("Metformin", "Metformin", "prescriptions"),
            MimicHospConceptEntry("Lisinopril", "Lisinopril", "prescriptions"),
        ),
        domain_id="Drug",
    )
    assert repo.patients_with_medication(combined) == {1, 2, 3}


def test_patient_age_uses_anchor_age_and_first_admission_year(tmp_path):
    repo = repository(tmp_path)

    assert repo.patients_with_patient_attribute({"attribute": "age", "operator": ">=", "value": 18}) == {1, 3}
    assert repo.patients_with_patient_attribute({"attribute": "age", "operator": "<=", "value": 18}) == {2}


def test_admission_date_range(tmp_path):
    repo = repository(tmp_path)

    assert repo.patients_with_admission({"start_date": "2020-01-01"}) == {1, 2, 3}
    assert repo.patients_with_admission({"end_date": "2019-12-31"}) == set()


def test_repository_works_when_diagnosis_tables_live_in_a_non_default_schema(tmp_path):
    # Reproduces a real deployment layout: dictionary/vocabulary-adjacent
    # tables under a non-default schema, clinical fact tables under main.
    # DuckDB will not resolve an unqualified `FROM diagnoses_icd` against a
    # table that only exists under e.g. `vocab`, even though table_exists()
    # (which searches information_schema.tables across every schema) reports
    # it as present.
    path = tmp_path / "mimic_hosp_namespaced.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA vocab")
        con.execute('''
            CREATE TABLE patients (
                subject_id BIGINT, anchor_age INTEGER, anchor_year INTEGER
            )
        ''')
        con.execute('''
            CREATE TABLE admissions (
                subject_id BIGINT, hadm_id BIGINT, admittime TIMESTAMP
            )
        ''')
        con.execute('''
            CREATE TABLE vocab.diagnoses_icd (
                subject_id BIGINT, hadm_id BIGINT, icd_code VARCHAR, icd_version INTEGER
            )
        ''')
        con.execute('''
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, drug VARCHAR, starttime TIMESTAMP
            )
        ''')
        con.execute("INSERT INTO patients VALUES (1, 40, 2020)")
        con.execute("INSERT INTO admissions VALUES (1, 501, '2020-01-01')")
        con.execute("INSERT INTO vocab.diagnoses_icd VALUES (1, 501, '25000', 9)")
        con.execute(
            "INSERT INTO prescriptions VALUES (1, 501, 'Metformin', '2020-01-05')"
        )

    repo = MimicHospCohortRepository(
        DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))
    )

    assert repo.patients_with_condition(
        condition("25000", 9)
    ) == {1}
