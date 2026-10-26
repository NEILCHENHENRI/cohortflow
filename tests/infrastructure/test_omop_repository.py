from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from infrastructure.database.duckdb import (
    DuckDBConfig,
    DuckDBConnectionFactory,
)
from infrastructure.omop.concepts import OMOPConcept
from infrastructure.omop.repository import DuckDBCohortRepository


def make_clinical(tmp_path):
    path = tmp_path / "clinical.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute('''
            CREATE TABLE person (
                person_id BIGINT,
                year_of_birth INTEGER,
                month_of_birth INTEGER,
                day_of_birth INTEGER
            )
        ''')
        con.execute('''
            CREATE TABLE visit_occurrence (
                person_id BIGINT,
                visit_concept_id BIGINT,
                visit_start_date DATE,
                visit_start_datetime TIMESTAMP
            )
        ''')
        con.execute('''
            CREATE TABLE condition_occurrence (
                person_id BIGINT,
                condition_concept_id BIGINT,
                condition_start_date DATE,
                condition_start_datetime TIMESTAMP
            )
        ''')
        con.execute('''
            CREATE TABLE drug_exposure (
                person_id BIGINT,
                drug_concept_id BIGINT,
                drug_exposure_start_date DATE,
                drug_exposure_start_datetime TIMESTAMP
            )
        ''')
        con.executemany(
            "INSERT INTO person VALUES (?, ?, ?, ?)",
            [
                (1, 1980, 1, 1),
                (2, 2010, 1, 1),
                (3, 1970, None, None),
            ],
        )
        con.executemany(
            "INSERT INTO visit_occurrence VALUES (?, ?, ?, ?)",
            [
                (1, 9201, date(2020, 1, 1), None),
                (2, 9201, date(2020, 1, 1), None),
                (3, 9202, date(2020, 1, 1), None),
            ],
        )
        con.executemany(
            "INSERT INTO condition_occurrence VALUES (?, ?, ?, ?)",
            [
                (1, 100, date(2019, 1, 1), None),
                (2, 100, date(2021, 1, 1), None),
                (3, 200, date(2018, 1, 1), None),
            ],
        )
        con.executemany(
            "INSERT INTO drug_exposure VALUES (?, ?, ?, ?)",
            [
                (1, 300, date(2020, 1, 1), None),
                (2, 300, date(2020, 1, 1), None),
                (3, 400, date(2017, 1, 1), None),
            ],
        )
    return path


def repository(tmp_path):
    return DuckDBCohortRepository(
        DuckDBConnectionFactory(
            DuckDBConfig(make_clinical(tmp_path), read_only=True)
        )
    )


def concept(concept_id, domain):
    return OMOPConcept(
        concept_id=concept_id,
        concept_name="x",
        domain_id=domain,
        vocabulary_id="test",
    )


def test_condition_drug_age_and_visit_queries(tmp_path):
    repo = repository(tmp_path)

    assert repo.patients_with_condition(
        concept(100, "Condition")
    ) == {1, 2}
    assert repo.patients_with_medication(
        concept(300, "Drug")
    ) == {1, 2}
    assert repo.patients_with_patient_attribute({"attribute": "age", "operator": ">=", "value": 18}) == {1, 3}
    assert repo.patients_with_admission({
        "visit_concept_id": 9201
    }) == {1, 2}
