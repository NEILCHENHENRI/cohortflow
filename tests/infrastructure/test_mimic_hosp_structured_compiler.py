from infrastructure.mimic_hosp.concepts import MimicHospConcept, MimicHospConceptEntry
from infrastructure.mimic_hosp.repository import MimicHospCohortRepository


class SchemaOnlyFactory:
    def table_exists(self, table_name):
        return True

    def qualified_table_name(self, table_name):
        return f'"main"."{table_name}"'

    def execute(self, sql, parameters=()):
        raise AssertionError("compile-only tests must not execute SQL")


def concept(domain, code, name, vocabulary):
    return MimicHospConcept(
        entries=(MimicHospConceptEntry(code, name, vocabulary),),
        domain_id=domain,
    )


def repo():
    return MimicHospCohortRepository(SchemaOnlyFactory())


def test_condition_procedure_lab_medication_compile_from_structured_representation():
    r = repo()
    diabetes = concept("Condition", "E11", "Type 2 diabetes mellitus", "ICD10CM")
    dialysis = concept("Procedure", "5A1D70Z", "Dialysis", "ICD10PCS")
    potassium = concept("Measurement", "50971", "Potassium", "d_labitems")
    furosemide = concept("Drug", "Furosemide", "Furosemide", "prescriptions")

    sql, params = r.compile_requirement_sql("condition", {"term": "diabetes", "seq_num": 1}, diabetes)
    assert '"main"."diagnoses_icd"' in sql and "seq_num" in sql
    assert params == ["E11", 10, 1]

    sql, params = r.compile_requirement_sql("procedure", {"term": "dialysis"}, dialysis)
    assert '"main"."procedures_icd"' in sql
    assert params == ["5A1D70Z", 10]

    sql, params = r.compile_requirement_sql(
        "lab",
        {"term": "potassium", "operator": ">", "value": 5.5, "unit": "mEq/L", "count_min": 2},
        potassium,
    )
    assert '"main"."labevents"' in sql
    assert "valuenum > ?" in sql and "HAVING COUNT(*) >= ?" in sql
    assert params == [50971, 5.5, "mEq/L", 2]

    sql, params = r.compile_requirement_sql(
        "medication",
        {"term": "furosemide", "dose_operator": ">=", "dose_value": 40, "dose_unit": "mg", "route": "PO"},
        furosemide,
    )
    assert '"main"."prescriptions"' in sql
    assert "TRY_CAST(dose_val_rx AS DOUBLE) >= ?" in sql and "lower(route)" in sql
    assert params == ["furosemide", 40, "mg", "po"]


def test_patient_admission_and_note_compile_from_structured_representation():
    r = repo()

    sql, params = r.compile_requirement_sql("patient", {"attribute": "age", "operator": ">=", "value": 65})
    assert "first_admission" in sql and "age_at_first_visit >= ?" in sql
    assert params == [65]

    sql, params = r.compile_requirement_sql(
        "admission",
        {"admission_type": "EW EMER.", "duration_operator": ">=", "duration_days": 7, "count_min": 2},
    )
    assert '"main"."admissions"' in sql
    assert "DATE_DIFF('hour'" in sql and "HAVING COUNT(*) >= ?" in sql
    assert params == ["EW EMER.", 7, 2]

    sql, params = r.compile_requirement_sql(
        "note", {"query": "medication non-adherence", "section": "Brief Hospital Course"}
    )
    assert '"main"."discharge"' in sql and "regexp_matches" in sql
    assert len(params) == 1 and "medication\\ non\\-adherence" in params[0]


def test_event_sql_exposes_common_subject_admission_time_shape_for_relations():
    r = repo()
    potassium = concept("Measurement", "50971", "Potassium", "d_labitems")
    dialysis = concept("Procedure", "5A1D70Z", "Dialysis", "ICD10PCS")

    lsql, lparams = r.event_sql("lab", {"term": "potassium", "operator": ">", "value": 6}, potassium)
    rsql, rparams = r.event_sql("procedure", {"term": "dialysis"}, dialysis)

    for sql in (lsql, rsql):
        assert "subject_id" in sql
        assert "hadm_id" in sql
        assert "event_time" in sql
    assert lparams == [50971, 6]
    assert rparams == ["5A1D70Z", 10]


def test_condition_event_sql_qualifies_columns_after_admissions_join():
    r = repo()
    diabetes = concept("Condition", "25000", "Diabetes mellitus", "ICD9CM")

    sql, params = r.event_sql("condition", {"term": "diabetes"}, diabetes)

    assert "TRY_CAST(d.subject_id AS BIGINT)" in sql
    assert "d.icd_code = ?" in sql
    assert "d.icd_version = ?" in sql
    assert "TRY_CAST(d.subject_id AS BIGINT) IS NOT NULL" in sql
    assert "TRY_CAST(subject_id AS BIGINT) IS NOT NULL" not in sql
    assert params == ["25000", 9]


def test_mimic_semantic_route_and_emergency_admission_normalization():
    r = repo()
    furosemide = concept("Drug", "Furosemide", "Furosemide", "prescriptions")

    sql, params = r.compile_requirement_sql(
        "medication",
        {"term": "furosemide", "route": "oral"},
        furosemide,
    )
    assert "lower(route) IN" in sql
    assert params == ["furosemide", "po", "oral", "p.o."]

    insulin = concept("Drug", "Insulin", "Insulin", "prescriptions")
    sql, params = r.compile_requirement_sql(
        "medication",
        {"term": "insulin", "route": "IV"},
        insulin,
    )
    assert "lower(route) IN" in sql
    assert params == ["insulin", "iv"]

    sql, params = r.compile_requirement_sql(
        "admission", {"admission_type": "emergency", "count_min": 2}
    )
    assert "lower(admission_type) LIKE '%emer%'" in sql
    assert params == [2]
