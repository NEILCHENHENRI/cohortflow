from __future__ import annotations

import json
from pathlib import Path

from agents.cohort.adapters.inspector import CSVInspector
from agents.cohort.adapters.registry import FieldRegistry
import pytest


def write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def test_inspection_cannot_replace_database_source(tmp_path):
    import sqlite3
    from agents.cohort.adapters.inspector import SQLInspector

    source = tmp_path / "construction_spec.json"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE labs (code TEXT)")
        connection.execute("INSERT INTO labs VALUES ('001')")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite source"):
        SQLInspector().inspect(source, tmp_path, dialect="sqlite")
    assert source.read_bytes() == original
    assert not (tmp_path / "field_catalog.json").exists()


def test_csv_inspector_generates_auditable_adapter_artifacts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "adapter"
    source.mkdir()
    write_csv(source / "admissions.csv", """
subject_id,hadm_id,admittime,admission_type
001,100,2026-01-02 12:00:00,EMERGENCY
002,101,2026-01-03 13:30:00,ELECTIVE
""")
    write_csv(source / "labs.csv", """
subject_id,hadm_id,test,value,unit
001,100,potassium,5.5,mmol/L
002,101,potassium,4.2,mmol/L
""")

    profile = CSVInspector(sample_limit=100).inspect(source, output)

    assert {table.name for table in profile.tables} == {"admissions", "labs"}
    assert (output / "report.md").exists()
    assert (output / "schema_profile.json").exists()
    assert (output / "field_catalog.json").exists()
    assert (output / "construction_spec.json").exists()

    catalog = json.loads((output / "field_catalog.json").read_text())
    fields = {item["semantic_field"]: item for item in catalog}
    assert fields["admissions.subject_id"]["is_identifier_or_code"] is True
    assert fields["admissions.admittime"]["is_datetime"] is True
    assert fields["labs.value"]["is_numeric"] is True
    assert fields["labs.value"]["unit_column"] == "unit"

    construction = json.loads((output / "construction_spec.json").read_text())
    derived = {(item["table"], item["source_column"]): item for item in construction["derived_fields"]}
    assert derived[("admissions", "admittime")]["kind"] == "datetime"
    assert derived[("labs", "value")]["kind"] == "numeric"
    assert derived[("labs", "value")]["unit_source_column"] == "unit"


def test_field_registry_uses_normalized_columns(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "adapter"
    source.mkdir()
    write_csv(source / "measurements.csv", """
patient_id,value,value_unit,when_time,label
001,10.5,mg,2026-01-02 10:00:00,a
002,20.0,mg,2026-01-03 11:00:00,b
""")
    CSVInspector(sample_limit=100).inspect(source, output)
    registry = FieldRegistry.from_adapter_dir(output)

    value = registry.get("measurements.value")
    assert value.value_kind == "numeric"
    assert value.value_column == "value__norm"
    assert value.unit_column == "value_unit__norm"

    timestamp = registry.get("measurements.when_time")
    assert timestamp.value_kind == "datetime"
    assert timestamp.value_column == "when_time__norm"

    patient_id = registry.get("measurements.patient_id")
    assert patient_id.value_kind == "identifier"
    assert patient_id.value_column == "patient_id"


def test_inspector_constructor_handoff_is_internally_consistent(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "adapter"
    source.mkdir()
    write_csv(source / "events.csv", """
patient_id,value,value_unit,event_time,label
001,1.5,g,2026-01-02 10:00:00,a
002,500,mg,2026-01-03 11:00:00,b
""")

    CSVInspector(sample_limit=100).inspect(source, output)
    catalog = json.loads((output / "field_catalog.json").read_text())
    construction = json.loads((output / "construction_spec.json").read_text())

    assert construction["source_kind"] == "csv"
    fields = {(item["table"], item["column"]): item for item in catalog}
    derived = {(item["table"], item["source_column"]): item for item in construction["derived_fields"]}

    value_field = fields[("events", "value")]
    value_derived = derived[("events", "value")]
    assert value_field["is_numeric"] is True
    assert value_field["unit_column"] == "value_unit"
    assert value_derived["target_column"] == "value__norm"
    assert value_derived["unit_source_column"] == "value_unit"
    assert value_derived["unit_target_column"] == "value_unit__norm"

    time_field = fields[("events", "event_time")]
    time_derived = derived[("events", "event_time")]
    assert time_field["is_datetime"] is True
    assert time_derived["kind"] == "datetime"
    assert time_derived["target_column"] == "event_time__norm"

    patient_field = fields[("events", "patient_id")]
    assert patient_field["is_identifier_or_code"] is True
    assert ("events", "patient_id") not in derived


def test_inspector_rejects_duplicate_table_names_before_publishing(tmp_path):
    source = tmp_path / "source"
    for folder in (source / "a", source / "b"):
        folder.mkdir(parents=True)
        (folder / "labs.csv").write_text("value\n1\n")
    output = tmp_path / "adapter"
    with pytest.raises(ValueError, match="duplicate logical table"):
        CSVInspector().inspect(source, output)
    assert not output.exists()


@pytest.mark.parametrize("header", ["value,value", "value,VALUE", "value,"])
def test_inspector_rejects_ambiguous_csv_headers(tmp_path, header):
    source = tmp_path / "labs.csv"
    source.write_text(header + "\n1,2\n")
    with pytest.raises(ValueError, match="column names"):
        CSVInspector().inspect(source, tmp_path / "adapter")


def test_inspector_does_not_present_capped_distinct_count_as_exact(tmp_path):
    source = tmp_path / "labs.csv"
    source.write_text("value\n" + "\n".join(str(i) for i in range(10)) + "\n")
    profile = CSVInspector(distinct_sample_limit=5).inspect(source, tmp_path / "adapter")
    assert profile.tables[0].columns[0].unique_count is None


def test_failed_bundle_publication_preserves_existing_adapter(tmp_path, monkeypatch):
    import agents.cohort.adapters.models as models
    source = tmp_path / "labs.csv"
    source.write_text("value\n1\n")
    output = tmp_path / "adapter"
    CSVInspector().inspect(source, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    source.write_text("value\n2\n")
    replace_file = models.os.replace

    def fail_third(src, dst):
        if Path(dst).name == "construction_spec.json" and Path(src).parent.name.startswith(".inspection-"):
            raise OSError("disk error")
        return replace_file(src, dst)

    monkeypatch.setattr(models.os, "replace", fail_third)
    with pytest.raises(OSError, match="disk error"):
        CSVInspector().inspect(source, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
