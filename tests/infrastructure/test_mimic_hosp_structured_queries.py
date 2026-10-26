import pytest

duckdb = pytest.importorskip("duckdb")

from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
from infrastructure.mimic_hosp.concepts import (
    MimicHospConcept,
    MimicHospConceptEntry,
    MimicHospConceptResolver,
)
from infrastructure.mimic_hosp.repository import MimicHospCohortRepository


def _concept(domain, *entries):
    return MimicHospConcept(
        entries=tuple(MimicHospConceptEntry(*entry) for entry in entries),
        domain_id=domain,
    )


def make_db(tmp_path):
    path = tmp_path / "mimic_all_domains.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE patients (
                subject_id BIGINT, gender VARCHAR, anchor_age INTEGER,
                anchor_year INTEGER, anchor_year_group VARCHAR, dod DATE
            )
        """)
        con.execute("""
            CREATE TABLE admissions (
                subject_id BIGINT, hadm_id BIGINT, admittime TIMESTAMP,
                dischtime TIMESTAMP, deathtime TIMESTAMP, admission_type VARCHAR,
                admit_provider_id VARCHAR, admission_location VARCHAR,
                discharge_location VARCHAR, insurance VARCHAR, language VARCHAR,
                marital_status VARCHAR, race VARCHAR, edregtime TIMESTAMP,
                edouttime TIMESTAMP, hospital_expire_flag INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE d_icd_diagnoses (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE diagnoses_icd (
                subject_id BIGINT, hadm_id BIGINT, seq_num INTEGER,
                icd_code VARCHAR, icd_version INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE d_icd_procedures (
                icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE procedures_icd (
                subject_id BIGINT, hadm_id BIGINT, seq_num INTEGER,
                chartdate DATE, icd_code VARCHAR, icd_version INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE d_labitems (
                itemid BIGINT, label VARCHAR, fluid VARCHAR, category VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE labevents (
                labevent_id BIGINT, subject_id BIGINT, hadm_id BIGINT,
                specimen_id BIGINT, itemid BIGINT, order_provider_id VARCHAR,
                charttime TIMESTAMP, storetime TIMESTAMP, value VARCHAR,
                valuenum DOUBLE, valueuom VARCHAR, ref_range_lower DOUBLE,
                ref_range_upper DOUBLE, flag VARCHAR, priority VARCHAR,
                comments VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE prescriptions (
                subject_id BIGINT, hadm_id BIGINT, pharmacy_id BIGINT,
                poe_id VARCHAR, poe_seq BIGINT, order_provider_id VARCHAR,
                starttime TIMESTAMP, stoptime TIMESTAMP, drug_type VARCHAR,
                drug VARCHAR, formulary_drug_cd VARCHAR, gsn VARCHAR,
                ndc VARCHAR, prod_strength VARCHAR, form_rx VARCHAR,
                dose_val_rx VARCHAR, dose_unit_rx VARCHAR,
                form_val_disp VARCHAR, form_unit_disp VARCHAR,
                doses_per_24_hrs DOUBLE, route VARCHAR, frequency VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE discharge (
                note_id VARCHAR, subject_id BIGINT, hadm_id BIGINT,
                note_type VARCHAR, note_seq INTEGER, charttime TIMESTAMP,
                storetime TIMESTAMP, text VARCHAR
            )
        """)

        con.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?)", [
            (1, "F", 60, 2020, "2017 - 2019", None),
            (2, "M", 45, 2020, "2017 - 2019", None),
            (3, "F", 70, 2020, "2017 - 2019", "2020-04-10"),
            (4, "M", 30, 2020, "2017 - 2019", None),
        ])
        con.executemany("INSERT INTO admissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (1, 101, "2020-01-01 08:00", "2020-01-05 08:00", None, "EW EMER.", "P1", "EMERGENCY ROOM", "HOME", "Private", "English", "MARRIED", "WHITE", None, None, 0),
            (1, 102, "2020-03-01 08:00", "2020-03-10 08:00", None, "URGENT", "P2", "TRANSFER FROM HOSPITAL", "HOME HEALTH CARE", "Private", "English", "MARRIED", "WHITE", None, None, 0),
            (2, 201, "2020-02-01 08:00", "2020-02-02 08:00", None, "ELECTIVE", "P3", "PHYSICIAN REFERRAL", "HOME", "Medicare", "English", "SINGLE", "BLACK/AFRICAN AMERICAN", None, None, 0),
            (3, 301, "2020-04-01 08:00", "2020-04-10 08:00", "2020-04-10 07:00", "EW EMER.", "P4", "EMERGENCY ROOM", "HOSPICE", "Medicaid", "Spanish", "WIDOWED", "WHITE", None, None, 1),
            (4, 401, "2020-05-01 08:00", "2020-05-03 08:00", None, "URGENT", "P5", "WALK-IN/SELF REFERRAL", "HOME", "Private", "English", "SINGLE", "OTHER", None, None, 0),
        ])
        con.executemany("INSERT INTO d_icd_diagnoses VALUES (?, ?, ?)", [
            ("E11", 10, "Type 2 diabetes mellitus"),
            ("I10", 10, "Essential (primary) hypertension"),
        ])
        con.executemany("INSERT INTO diagnoses_icd VALUES (?, ?, ?, ?, ?)", [
            (1, 101, 1, "E11", 10), (1, 102, 2, "I10", 10),
            (2, 201, 1, "I10", 10), (3, 301, 1, "E11", 10),
        ])
        con.executemany("INSERT INTO d_icd_procedures VALUES (?, ?, ?)", [
            ("5A1D70Z", 10, "Performance of urinary filtration, intermittent"),
            ("0W9G3ZZ", 10, "Drainage of peritoneal cavity, percutaneous approach"),
        ])
        con.executemany("INSERT INTO procedures_icd VALUES (?, ?, ?, ?, ?, ?)", [
            (1, 102, 1, "2020-03-05", "5A1D70Z", 10),
            (3, 301, 1, "2020-04-05", "0W9G3ZZ", 10),
        ])
        con.executemany("INSERT INTO d_labitems VALUES (?, ?, ?, ?)", [
            (50971, "Potassium", "Blood", "Chemistry"),
            (50912, "Creatinine", "Blood", "Chemistry"),
        ])
        con.executemany("INSERT INTO labevents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (1, 1, 102, 1, 50971, None, "2020-03-04 08:00", None, "6.2", 6.2, "mEq/L", 3.5, 5.1, "abnormal", "STAT", None),
            (2, 1, 102, 2, 50912, None, "2020-03-04 09:00", None, "2.4", 2.4, "mg/dL", 0.6, 1.2, "abnormal", "ROUTINE", None),
            (3, 1, 102, 3, 50912, None, "2020-03-04 12:00", None, "2.2", 2.2, "mg/dL", 0.6, 1.2, "abnormal", "ROUTINE", None),
            (4, 2, 201, 4, 50971, None, "2020-02-01 10:00", None, "4.0", 4.0, "mEq/L", 3.5, 5.1, None, "ROUTINE", None),
            (5, 3, 301, 5, 50971, None, "2020-04-02 10:00", None, "6.8", 6.8, "mEq/L", 3.5, 5.1, "abnormal", "STAT", None),
        ])
        con.executemany("INSERT INTO prescriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (1, 102, 1, None, None, None, "2020-03-04 10:00", "2020-03-08 10:00", "MAIN", "Furosemide", None, None, None, "40 mg", None, "40", "mg", None, None, 1, "PO", "DAILY"),
            (1, 102, 2, None, None, None, "2020-03-04 11:00", "2020-03-04 12:00", "MAIN", "Insulin", None, None, None, "10 units", None, "10", "UNIT", None, None, 1, "IV", "ONCE"),
            (2, 201, 3, None, None, None, "2020-02-01 09:00", None, "MAIN", "Furosemide", None, None, None, "20 mg", None, "20", "mg", None, None, 1, "PO", "DAILY"),
        ])
        con.executemany("INSERT INTO discharge VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            ("1-DS-1", 1, 102, "DS", 1, "2020-03-10", "2020-03-11", "Brief Hospital Course:\nPatient had medication non-adherence and improved after dialysis.\nDischarge Diagnosis:\nAKI"),
            ("2-DS-1", 2, 201, "DS", 1, "2020-02-02", "2020-02-03", "Brief Hospital Course:\nRoutine elective admission.\nDischarge Diagnosis:\nHypertension"),
            ("3-DS-1", 3, 301, "DS", 1, "2020-04-10", "2020-04-11", "Goals of Care:\nPatient elected hospice care.\nDischarge Diagnosis:\nCirrhosis"),
        ])
    return path


def stack(tmp_path):
    factory = DuckDBConnectionFactory(DuckDBConfig(make_db(tmp_path), read_only=True))
    return MimicHospCohortRepository(factory), MimicHospConceptResolver(factory)


def test_structured_patient_and_admission_requirements_compile_to_sql_and_execute(tmp_path):
    repo, _ = stack(tmp_path)
    assert repo.patients_with_patient_attribute({"attribute": "gender", "operator": "=", "value": "female"}) == {1, 3}
    assert repo.patients_with_patient_attribute({"attribute": "age", "operator": ">=", "value": 65}) == {3}
    assert repo.patients_with_patient_attribute({"attribute": "deceased", "operator": "=", "value": True}) == {3}
    assert repo.patients_with_admission({"admission_type": "EW EMER.", "count_min": 1}) == {1, 3}
    assert repo.patients_with_admission({"duration_operator": ">=", "duration_days": 7}) == {1, 3}
    assert repo.patients_with_admission({"discharge_location": "HOSPICE"}) == {3}


def test_structured_condition_and_procedure_use_dictionary_resolutions(tmp_path):
    repo, resolver = stack(tmp_path)
    diabetes = resolver.resolve_condition("Type 2 diabetes mellitus")
    dialysis = resolver.resolve_procedure("Performance of urinary filtration, intermittent")
    assert repo.patients_with_condition(diabetes, {"term": "Type 2 diabetes mellitus"}) == {1, 3}
    assert repo.patients_with_procedure(dialysis, {"term": "Performance of urinary filtration, intermittent"}) == {1}


def test_structured_lab_threshold_count_and_aggregation(tmp_path):
    repo, resolver = stack(tmp_path)
    potassium = resolver.resolve_lab("Potassium")
    creatinine = resolver.resolve_lab("Creatinine")
    assert repo.patients_with_lab(potassium, {"term": "Potassium", "operator": ">", "value": 6.0}) == {1, 3}
    assert repo.patients_with_lab(creatinine, {"term": "Creatinine", "operator": ">", "value": 2.0, "count_min": 2}) == {1}
    assert repo.patients_with_lab(creatinine, {"term": "Creatinine", "aggregation": "avg", "operator": ">", "value": 2.2}) == {1}


def test_structured_medication_dose_route_frequency(tmp_path):
    repo, resolver = stack(tmp_path)
    drug = resolver.resolve_medication("Furosemide")
    assert repo.patients_with_medication(drug, {
        "term": "Furosemide", "dose_operator": ">=", "dose_value": 40,
        "dose_unit": "mg", "route": "PO", "frequency": "DAILY",
    }) == {1}



def test_medication_route_aliases_accept_semantic_and_native_values(tmp_path):
    repo, resolver = stack(tmp_path)
    furosemide = resolver.resolve_medication("Furosemide")
    insulin = resolver.resolve_medication("Insulin")

    # Route-only filtering should include both patients because both fixture
    # Furosemide rows are recorded as PO. Dose/frequency constraints are tested
    # separately above and correctly narrow the result to patient 1.
    assert repo.patients_with_medication(furosemide, {
        "term": "Furosemide", "route": "oral",
    }) == {1, 2}
    assert repo.patients_with_medication(furosemide, {
        "term": "Furosemide", "route": "PO",
    }) == {1, 2}
    assert repo.patients_with_medication(insulin, {
        "term": "Insulin", "route": "intravenous",
    }) == {1}
    assert repo.patients_with_medication(insulin, {
        "term": "Insulin", "route": "IV",
    }) == {1}

def test_discharge_free_text_is_a_first_class_deterministic_evidence_source(tmp_path):
    repo, _ = stack(tmp_path)
    assert repo.patients_with_note({"query": "medication non-adherence"}) == {1}
    assert repo.patients_with_note({"query": "hospice care"}) == {3}
    assert repo.patients_with_note({"query": "medication non-adherence", "section": "Brief Hospital Course"}) == {1}


def test_generic_before_relation_works_across_lab_and_procedure(tmp_path):
    repo, resolver = stack(tmp_path)
    creatinine = resolver.resolve_lab("Creatinine")
    dialysis = resolver.resolve_procedure("Performance of urinary filtration, intermittent")
    assert repo.patients_satisfying_relation(
        relation="before",
        left_kind="lab",
        left_payload={"term": "Creatinine", "operator": ">", "value": 2.0},
        left_concept=creatinine,
        right_kind="procedure",
        right_payload={"term": "Performance of urinary filtration, intermittent"},
        right_concept=dialysis,
        max_gap_hours=48,
        same_admission=True,
    ) == {1}


def test_generic_before_relation_works_across_lab_and_medication(tmp_path):
    repo, resolver = stack(tmp_path)
    potassium = resolver.resolve_lab("Potassium")
    insulin = resolver.resolve_medication("Insulin")
    assert repo.patients_satisfying_relation(
        relation="before",
        left_kind="lab",
        left_payload={"term": "Potassium", "operator": ">", "value": 6.0},
        left_concept=potassium,
        right_kind="medication",
        right_payload={"term": "Insulin", "route": "IV"},
        right_concept=insulin,
        max_gap_hours=4,
        same_admission=True,
    ) == {1}


def test_sql_compiler_does_not_need_llm(tmp_path):
    repo, resolver = stack(tmp_path)
    potassium = resolver.resolve_lab("Potassium")
    sql, params = repo.compile_requirement_sql(
        "lab", {"term": "Potassium", "operator": ">=", "value": 6.0, "unit": "mEq/L"}, potassium
    )
    assert "labevents" in sql
    assert "valuenum >= ?" in sql
    assert params[-2:] == [6.0, "mEq/L"]
