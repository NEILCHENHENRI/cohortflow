from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path


# These are intentionally synthetic identifiers, not official Athena/OMOP IDs.
CONCEPTS = (
    (9000001, "Type 2 diabetes mellitus", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-T2DM", None),
    (9000002, "Hypertension", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-HTN", None),
    (9000003, "Hypertension stage 1", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-HTN1", None),
    (9000004, "Hypertension stage 2", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-HTN2", None),
    (9000005, "Chronic kidney disease", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-CKD", None),
    (9000006, "Asthma", "Condition", "DEMO", "Clinical Finding", "S", "DEMO-COND-ASTHMA", None),
    (9000101, "Metformin", "Drug", "DEMO", "Ingredient", "S", "DEMO-DRUG-METFORMIN", None),
    (9000102, "Insulin", "Drug", "DEMO", "Ingredient", "S", "DEMO-DRUG-INSULIN", None),
    (9000103, "Insulin glargine", "Drug", "DEMO", "Ingredient", "S", "DEMO-DRUG-GLARGINE", None),
    (9000104, "Insulin lispro", "Drug", "DEMO", "Ingredient", "S", "DEMO-DRUG-LISPRO", None),
    (9000105, "Lisinopril", "Drug", "DEMO", "Ingredient", "S", "DEMO-DRUG-LISINOPRIL", None),
    (9000201, "Inpatient Visit", "Visit", "DEMO", "Visit", "S", "DEMO-VISIT-INPATIENT", None),
    (9000202, "Outpatient Visit", "Visit", "DEMO", "Visit", "S", "DEMO-VISIT-OUTPATIENT", None),
    (9000203, "Emergency Room Visit", "Visit", "DEMO", "Visit", "S", "DEMO-VISIT-ED", None),
)


def ensure_demo_databases(output_dir: str | Path) -> tuple[Path, Path]:
    """Create the synthetic demo databases when either file is absent."""
    directory = Path(output_dir).expanduser().resolve()
    clinical = directory / "clinical.duckdb"
    vocabulary = directory / "vocabulary.duckdb"
    if clinical.exists() and vocabulary.exists():
        return clinical, vocabulary
    build_demo_databases(directory, overwrite=True)
    return clinical, vocabulary


def build_demo_databases(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Build small, fully synthetic OMOP-like DuckDB databases."""
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Demo database construction requires DuckDB. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    clinical = directory / "clinical.duckdb"
    vocabulary = directory / "vocabulary.duckdb"

    for path in (clinical, vocabulary):
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} exists. Pass overwrite=True to replace it"
                )
            path.unlink()

    # Keep generated demo files readable by every supported DuckDB 1.x reader,
    # even when the builder runs under DuckDB 2.x.
    storage_config = {"storage_compatibility_version": "v1.0.0"}
    with duckdb.connect(str(vocabulary), config=storage_config) as connection:
        connection.execute(
            """
            CREATE TABLE concept (
                concept_id BIGINT,
                concept_name VARCHAR,
                domain_id VARCHAR,
                vocabulary_id VARCHAR,
                concept_class_id VARCHAR,
                standard_concept VARCHAR,
                concept_code VARCHAR,
                invalid_reason VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO concept VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            CONCEPTS,
        )

    with duckdb.connect(str(clinical), config=storage_config) as connection:
        _create_clinical_schema(connection)
        _insert_people(connection)
        _insert_visits(connection)
        _insert_conditions(connection)
        _insert_drugs(connection)

    verify_demo_databases(clinical, vocabulary)
    return clinical, vocabulary


def verify_demo_databases(
    clinical_database: str | Path,
    vocabulary_database: str | Path,
) -> None:
    import duckdb

    with duckdb.connect(str(vocabulary_database), read_only=True) as connection:
        count = connection.execute("SELECT COUNT(*) FROM concept").fetchone()[0]
        if count != len(CONCEPTS):
            raise RuntimeError("demo vocabulary verification failed")

    with duckdb.connect(str(clinical_database), read_only=True) as connection:
        expected = {
            "person": 24,
            "t2dm": 12,
            "hypertension_stage_1": 10,
            "metformin": 8,
        }
        observed = {
            "person": connection.execute(
                "SELECT COUNT(*) FROM person"
            ).fetchone()[0],
            "t2dm": connection.execute(
                """SELECT COUNT(DISTINCT person_id)
                   FROM condition_occurrence
                   WHERE condition_concept_id = 9000001"""
            ).fetchone()[0],
            "hypertension_stage_1": connection.execute(
                """SELECT COUNT(DISTINCT person_id)
                   FROM condition_occurrence
                   WHERE condition_concept_id = 9000003"""
            ).fetchone()[0],
            "metformin": connection.execute(
                """SELECT COUNT(DISTINCT person_id)
                   FROM drug_exposure
                   WHERE drug_concept_id = 9000101"""
            ).fetchone()[0],
        }
        if observed != expected:
            raise RuntimeError(
                f"demo clinical verification failed: {observed!r}"
            )


def _create_clinical_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE person (
            person_id BIGINT,
            gender_concept_id BIGINT,
            year_of_birth INTEGER,
            month_of_birth INTEGER,
            day_of_birth INTEGER,
            race_concept_id BIGINT,
            ethnicity_concept_id BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE visit_occurrence (
            visit_occurrence_id BIGINT,
            person_id BIGINT,
            visit_concept_id BIGINT,
            visit_start_date DATE,
            visit_start_datetime TIMESTAMP,
            visit_end_date DATE,
            visit_end_datetime TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE condition_occurrence (
            condition_occurrence_id BIGINT,
            person_id BIGINT,
            condition_concept_id BIGINT,
            condition_start_date DATE,
            condition_start_datetime TIMESTAMP,
            condition_end_date DATE,
            condition_end_datetime TIMESTAMP,
            visit_occurrence_id BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE drug_exposure (
            drug_exposure_id BIGINT,
            person_id BIGINT,
            drug_concept_id BIGINT,
            drug_exposure_start_date DATE,
            drug_exposure_start_datetime TIMESTAMP,
            drug_exposure_end_date DATE,
            drug_exposure_end_datetime TIMESTAMP,
            visit_occurrence_id BIGINT
        )
        """
    )


def _insert_people(connection) -> None:
    people = []
    for person_id in range(1, 25):
        people.append(
            (
                person_id,
                8507 if person_id % 2 else 8532,
                1940 + ((person_id * 7) % 58),
                1 + person_id % 12,
                1 + person_id % 27,
                0,
                0,
            )
        )
    connection.executemany(
        "INSERT INTO person VALUES (?, ?, ?, ?, ?, ?, ?)",
        people,
    )


def _insert_visits(connection) -> None:
    visits = []
    visit_id = 1
    for person_id in range(1, 25):
        first = date(2022, 1, 1) + timedelta(days=person_id * 9)
        visit_concept = (
            9000201
            if person_id % 6 == 0
            else 9000203
            if person_id % 5 == 0
            else 9000202
        )
        visits.append(
            (visit_id, person_id, visit_concept, first, None, first + timedelta(days=1), None)
        )
        visit_id += 1
        if person_id % 3 == 0:
            second = first + timedelta(days=180)
            visits.append((visit_id, person_id, 9000202, second, None, second, None))
            visit_id += 1
    connection.executemany(
        "INSERT INTO visit_occurrence VALUES (?, ?, ?, ?, ?, ?, ?)",
        visits,
    )


def _insert_conditions(connection) -> None:
    condition_sets = {
        9000001: range(1, 13),
        9000003: range(7, 17),
        9000004: range(17, 21),
        9000005: (4, 8, 12, 16, 20, 24),
        9000006: (2, 6, 10, 14, 18, 22),
    }
    rows = []
    occurrence_id = 1
    for concept_id, patient_ids in condition_sets.items():
        for person_id in patient_ids:
            start = date(2022, 2, 1) + timedelta(days=person_id * 5)
            rows.append(
                (occurrence_id, person_id, concept_id, start, None, None, None, person_id)
            )
            occurrence_id += 1
    connection.executemany(
        "INSERT INTO condition_occurrence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _insert_drugs(connection) -> None:
    drug_sets = {
        9000101: range(1, 9),
        9000102: range(5, 11),
        9000103: range(9, 13),
        9000104: range(11, 15),
        9000105: range(7, 17),
    }
    rows = []
    exposure_id = 1
    for concept_id, patient_ids in drug_sets.items():
        for person_id in patient_ids:
            start = date(2022, 3, 1) + timedelta(days=person_id * 6)
            rows.append(
                (
                    exposure_id,
                    person_id,
                    concept_id,
                    start,
                    None,
                    start + timedelta(days=90),
                    None,
                    person_id,
                )
            )
            exposure_id += 1
    connection.executemany(
        "INSERT INTO drug_exposure VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
