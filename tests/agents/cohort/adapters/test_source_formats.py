from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agents.cohort.adapters.constructor import DatasetConstructor
from agents.cohort.adapters.inspector import DatasetInspector, ParquetInspector, SQLInspector


duckdb = pytest.importorskip("duckdb")


def _assert_common_artifacts(output: Path) -> None:
    assert (output / "report.md").exists()
    assert (output / "schema_profile.json").exists()
    assert (output / "field_catalog.json").exists()
    assert (output / "construction_spec.json").exists()


def test_parquet_inspector_and_constructor_use_common_adapter_contract(tmp_path):
    source = tmp_path / "parquet"
    source.mkdir()
    parquet = source / "labs.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE labs(subject_id VARCHAR, value VARCHAR, value_unit VARCHAR, charttime VARCHAR)")
    con.execute("INSERT INTO labs VALUES ('001','5.5','mmol/L','2026-01-02 10:00:00'), ('002','4.2','mmol/L','2026-01-03 11:00:00')")
    con.execute(f"COPY labs TO '{str(parquet).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET)")
    con.close()

    output = tmp_path / "adapter"
    profile = ParquetInspector(sample_limit=100).inspect(source, output)
    assert profile.source_kind == "parquet"
    assert [table.name for table in profile.tables] == ["labs"]
    _assert_common_artifacts(output)

    catalog = json.loads((output / "field_catalog.json").read_text())
    by_field = {item["semantic_field"]: item for item in catalog}
    assert by_field["labs.value"]["is_numeric"] is True
    assert by_field["labs.value"]["unit_column"] == "value_unit"
    assert by_field["labs.charttime"]["is_datetime"] is True

    db = tmp_path / "normalized.duckdb"
    result = DatasetConstructor().construct(output / "construction_spec.json", db)
    assert result.tables == ("labs",)
    check = duckdb.connect(str(db), read_only=True)
    row = check.execute("SELECT value__norm, value_unit__norm, charttime__norm FROM labs WHERE subject_id='001'").fetchone()
    check.close()
    assert row[0] == pytest.approx(5.5)
    assert row[1] == "mmol/L"
    assert str(row[2]).startswith("2026-01-02 10:00:00")


def test_duckdb_sql_inspector_and_constructor(tmp_path):
    source = tmp_path / "clinical.duckdb"
    con = duckdb.connect(str(source))
    con.execute("CREATE TABLE admissions(subject_id VARCHAR, admittime VARCHAR, admission_type VARCHAR)")
    con.execute("INSERT INTO admissions VALUES ('001','2026-01-02 12:00:00','EMERGENCY')")
    con.close()

    output = tmp_path / "adapter"
    profile = SQLInspector(sample_limit=100).inspect(source, output)
    assert profile.source_kind == "sql:duckdb"
    assert [table.name for table in profile.tables] == ["admissions"]
    _assert_common_artifacts(output)

    db = tmp_path / "normalized.duckdb"
    DatasetConstructor().construct(output / "construction_spec.json", db)
    check = duckdb.connect(str(db), read_only=True)
    value = check.execute("SELECT admittime__norm FROM admissions").fetchone()[0]
    check.close()
    assert str(value).startswith("2026-01-02 12:00:00")


def test_sqlite_sql_inspector_and_constructor(tmp_path):
    source = tmp_path / "clinical.sqlite"
    con = sqlite3.connect(source)
    con.execute("CREATE TABLE meds(subject_id TEXT, dose TEXT, dose_unit TEXT, route TEXT)")
    con.execute("INSERT INTO meds VALUES ('001','40','mg','PO')")
    con.commit()
    con.close()

    output = tmp_path / "adapter"
    profile = DatasetInspector(sample_limit=100).inspect(source, output)
    assert profile.source_kind == "sql:sqlite"
    _assert_common_artifacts(output)

    db = tmp_path / "normalized.duckdb"
    DatasetConstructor().construct(output / "construction_spec.json", db)
    check = duckdb.connect(str(db), read_only=True)
    value, unit = check.execute("SELECT dose__norm, dose_unit__norm FROM meds").fetchone()
    check.close()
    assert value == pytest.approx(40.0)
    assert unit == "mg"


def test_dataset_inspector_auto_detects_parquet_and_sql(tmp_path):
    parquet = tmp_path / "x.parquet"
    con = duckdb.connect()
    con.execute("COPY (SELECT 1 AS patient_id, '3.2' AS value) TO ? (FORMAT PARQUET)", [str(parquet)])
    con.close()
    profile = DatasetInspector(sample_limit=100).inspect(parquet, tmp_path / "pa")
    assert profile.source_kind == "parquet"

    sqlite = tmp_path / "x.db"
    con = sqlite3.connect(sqlite)
    con.execute("CREATE TABLE t(patient_id TEXT)")
    con.execute("INSERT INTO t VALUES ('001')")
    con.commit()
    con.close()
    profile = DatasetInspector(sample_limit=100).inspect(sqlite, tmp_path / "sa")
    assert profile.source_kind == "sql:sqlite"


def test_mixed_directory_requires_an_explicit_format(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.PARQUET").write_bytes(b"not read during detection")
    with pytest.raises(ValueError, match="mixed source formats"):
        DatasetInspector().inspect(tmp_path, tmp_path / "adapter")


def test_duckdb_schema_names_remain_distinct_in_registry(tmp_path):
    from agents.cohort.adapters.registry import FieldRegistry
    source = tmp_path / "source.duckdb"
    with duckdb.connect(str(source)) as con:
        for schema in ("ward_a", "ward_b"):
            con.execute(f"CREATE SCHEMA {schema}")
            con.execute(f"CREATE TABLE {schema}.labs(value VARCHAR)")
            con.execute(f"INSERT INTO {schema}.labs VALUES ('1.5')")
    adapter = tmp_path / "adapter"
    DatasetInspector().inspect(source, adapter)
    registry = FieldRegistry.from_adapter_dir(adapter)
    assert {field.semantic_field for field in registry.all()} == {"ward_a.labs.value", "ward_b.labs.value"}
    result = DatasetConstructor().construct(adapter / "construction_spec.json", tmp_path / "out.duckdb")
    assert result.tables == ("ward_a.labs", "ward_b.labs")


def test_sqlite_uri_special_characters_are_not_interpreted_as_options(tmp_path):
    source = tmp_path / "source?#.sqlite"
    with sqlite3.connect(source) as con:
        con.execute("CREATE TABLE t(value TEXT)")
        con.execute("INSERT INTO t VALUES ('1.5')")
    adapter = tmp_path / "adapter"
    DatasetInspector().inspect(source, adapter)
    result = DatasetConstructor().construct(adapter / "construction_spec.json", tmp_path / "out.duckdb")
    assert result.tables == ("t",)


def test_sqlite_preserves_native_integer_real_and_blob(tmp_path):
    source = tmp_path / 'native.sqlite'
    with sqlite3.connect(source) as con:
        con.execute('CREATE TABLE samples(patient_id TEXT, amount INTEGER, ratio REAL, bytes BLOB)')
        con.execute('INSERT INTO samples VALUES (?, ?, ?, ?)', ('001', 9007199254740993, 1.25, b'\x00\xff'))
    package = tmp_path / 'adapter'
    DatasetInspector().inspect(source, package)
    target = tmp_path / 'out.duckdb'
    DatasetConstructor().construct(package / 'construction_spec.json', target)
    with duckdb.connect(str(target)) as con:
        assert con.execute('SELECT patient_id, amount, ratio, bytes FROM samples').fetchone() == ('001', 9007199254740993, 1.25, b'\x00\xff')
        assert con.execute("SELECT typeof(amount), typeof(ratio), typeof(bytes) FROM samples").fetchone() == ('BIGINT', 'DOUBLE', 'BLOB')


def test_sqlite_mixed_storage_preserves_tags_and_does_not_trust_affinity(tmp_path):
    source = tmp_path / 'mixed.sqlite'
    with sqlite3.connect(source) as con:
        con.execute('CREATE TABLE mixed(value INTEGER)')
        con.executemany('INSERT INTO mixed VALUES (?)', [(9007199254740993,), ('bad',), (1.25,), (None,)])
    package = tmp_path / 'adapter'
    profile = DatasetInspector().inspect(source, package)
    assert profile.tables[0].columns[0].storage_type == 'SQLITE_MIXED'
    target = tmp_path / 'out.duckdb'
    DatasetConstructor().construct(package / 'construction_spec.json', target)
    with duckdb.connect(str(target)) as con:
        assert con.execute('SELECT value, union_tag(value) FROM mixed').fetchall() == [
            (9007199254740993, 'i'), ('bad', 's'), (1.25, 'r'), (None, None)]
