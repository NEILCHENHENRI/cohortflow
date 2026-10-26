from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.cohort.adapters.constructor import DatasetConstructor
from agents.cohort.adapters.models import ConstructionSpec, DerivedFieldSpec


def _derived(**overrides):
    payload = dict(
        table="labs",
        source_column="value",
        target_column="value__norm",
        kind="numeric",
        sql_expression='TRY_CAST("value" AS DOUBLE)',
        parse_coverage=1.0,
        unit_source_column=None,
        unit_target_column=None,
        canonical_unit=None,
        observed_units=[],
        unit_conversions={},
    )
    payload.update(overrides)
    return DerivedFieldSpec(**payload)


def test_validate_spec_rejects_unsupported_source_and_output(tmp_path):
    constructor = DatasetConstructor()
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n")
    with pytest.raises(ValueError, match="unsupported source_kind"):
        constructor.validate_spec(ConstructionSpec("xml", str(source), []))
    with pytest.raises(ValueError, match="unsupported output_format"):
        constructor.validate_spec(ConstructionSpec("csv", str(source), [], output_format="parquet"))


def test_validate_spec_rejects_duplicate_targets_and_bad_unit_plan(tmp_path):
    constructor = DatasetConstructor()
    source = tmp_path / "source.csv"
    source.write_text("value,unit\n1,mg\n")
    duplicate = ConstructionSpec(
        "csv",
        str(source),
        [
            _derived(),
            _derived(source_column="other", target_column="value__norm"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate derived target"):
        constructor.validate_spec(duplicate)

    missing_unit = ConstructionSpec(
        "csv",
        str(source),
        [_derived(unit_conversions={"g": {"factor": 1000.0}}, canonical_unit="mg")],
    )
    with pytest.raises(ValueError, match="unit conversions require unit_source_column"):
        constructor.validate_spec(missing_unit)

    missing_canonical = ConstructionSpec(
        "csv",
        str(source),
        [_derived(
            unit_source_column="unit",
            unit_target_column="unit__norm",
            unit_conversions={"g": {"factor": 1000.0}},
        )],
    )
    with pytest.raises(ValueError, match="unit conversions require canonical_unit"):
        constructor.validate_spec(missing_canonical)


def test_unit_expression_is_case_sensitive_and_fail_closed():
    constructor = DatasetConstructor()
    derived = _derived(
        unit_source_column="unit",
        unit_target_column="unit__norm",
        canonical_unit="mg",
        observed_units=["mg", "g", "tablet"],
        unit_conversions={
            "mg": {"factor": 1.0},
            "g": {"factor": 1000.0},
        },
    )
    value_sql, unit_sql = constructor._derived_expressions(derived)
    assert "REPLACE(TRIM" in value_sql
    assert "LOWER(" not in value_sql
    assert "WHEN 'g'" in value_sql
    assert "* 1000.0" in value_sql
    assert "ELSE NULL" in value_sql
    assert unit_sql is not None
    assert "THEN 'mg'" in unit_sql


def test_unit_expression_preserves_raw_unit_when_no_conversion_plan():
    constructor = DatasetConstructor()
    derived = _derived(
        unit_source_column="unit",
        unit_target_column="unit__norm",
    )
    value_sql, unit_sql = constructor._derived_expressions(derived)
    assert value_sql == 'TRY_CAST("value" AS DOUBLE)'
    assert unit_sql == 'NULLIF(TRIM(CAST("unit" AS VARCHAR)), \'\')'


def test_load_spec_round_trips_unit_conversion_metadata(tmp_path):
    raw = {
        "source_kind": "csv",
        "source_path": str(tmp_path),
        "output_format": "duckdb",
        "derived_fields": [{
            "table": "labs",
            "source_column": "value",
            "target_column": "value__norm",
            "kind": "numeric",
            "sql_expression": 'TRY_CAST("value" AS DOUBLE)',
            "parse_coverage": 0.99,
            "unit_source_column": "unit",
            "unit_target_column": "unit__norm",
            "canonical_unit": "mg",
            "observed_units": ["mg", "g"],
            "unit_conversions": {
                "mg": {"factor": 1.0},
                "g": {"factor": 1000.0},
            },
        }],
    }
    path = tmp_path / "construction_spec.json"
    path.write_text(json.dumps(raw))
    spec = DatasetConstructor().load_spec(path)
    assert spec.derived_fields[0].canonical_unit == "mg"
    assert spec.derived_fields[0].unit_conversions["g"]["factor"] == 1000.0


try:
    import duckdb
except ImportError:  # pragma: no cover - exercised in dependency-light CI
    duckdb = None

requires_duckdb = pytest.mark.skipif(duckdb is None, reason="duckdb not installed")


def _write_spec(path: Path, source: Path, derived_fields: list[dict]) -> None:
    path.write_text(json.dumps({
        "source_kind": "csv",
        "source_path": str(source),
        "output_format": "duckdb",
        "derived_fields": derived_fields,
    }))


@requires_duckdb
def test_constructor_materializes_normalized_columns_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    csv_path = source / "labs.csv"
    csv_path.write_text("subject_id,value,unit,charttime\n001,5.5,mmol/L,2026-01-02 12:00:00\n")
    spec_path = tmp_path / "construction_spec.json"
    _write_spec(spec_path, source, [
        {
            "table": "labs", "source_column": "value", "target_column": "value__norm",
            "kind": "numeric", "sql_expression": 'TRY_CAST("value" AS DOUBLE)',
            "parse_coverage": 1.0, "unit_source_column": "unit", "unit_target_column": "unit__norm",
            "canonical_unit": None, "observed_units": ["mmol/L"], "unit_conversions": {},
        },
        {
            "table": "labs", "source_column": "charttime", "target_column": "charttime__norm",
            "kind": "datetime", "sql_expression": 'TRY_CAST("charttime" AS TIMESTAMP)',
            "parse_coverage": 1.0, "unit_source_column": None, "unit_target_column": None,
            "canonical_unit": None, "observed_units": [], "unit_conversions": {},
        },
    ])
    output = tmp_path / "adapted.duckdb"

    result = DatasetConstructor().construct(spec_path, output)
    con = duckdb.connect(str(output))
    row = con.execute("SELECT subject_id, value, value__norm, unit__norm, charttime__norm FROM labs").fetchone()
    con.close()
    assert row[0] == 1 or str(row[0]) == "001"
    assert str(row[1]) == "5.5"
    assert row[2] == pytest.approx(5.5)
    assert row[3] == "mmol/L"
    assert row[4] is not None
    assert "001,5.5,mmol/L" in csv_path.read_text()
    assert result.report_path is not None and result.report_path.exists()


@requires_duckdb
def test_constructor_applies_audited_unit_conversions_and_flags_unknown_units(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "meds.csv").write_text(
        "subject_id,dose,dose_unit\n"
        "001,1,g\n"
        "002,500,mg\n"
        "003,2,G\n"
        "004,1,tablet\n"
    )
    spec_path = tmp_path / "construction_spec.json"
    _write_spec(spec_path, source, [{
        "table": "meds", "source_column": "dose", "target_column": "dose__norm",
        "kind": "numeric", "sql_expression": 'TRY_CAST("dose" AS DOUBLE)',
        "parse_coverage": 1.0, "unit_source_column": "dose_unit", "unit_target_column": "dose_unit__norm",
        "canonical_unit": "mg", "observed_units": ["g", "mg", "tablet"],
        "unit_conversions": {
            "mg": {"factor": 1.0},
            "g": {"factor": 1000.0},
        },
    }])
    output = tmp_path / "adapted.duckdb"
    result = DatasetConstructor().construct(spec_path, output)

    con = duckdb.connect(str(output), read_only=True)
    rows = con.execute(
        "SELECT CAST(subject_id AS VARCHAR), dose__norm, dose_unit__norm FROM meds ORDER BY subject_id"
    ).fetchall()
    con.close()
    assert rows[0][1:] == (1000.0, "mg")
    assert rows[1][1:] == (500.0, "mg")
    assert rows[2][1:] == (None, None)  # G is not the SI gram symbol g
    assert rows[3][1:] == (None, None)
    assert result.field_stats[0].unsupported_unit_rows == 2
    report = json.loads(result.report_path.read_text())
    assert report["derived_fields"][0]["unsupported_unit_rows"] == 2


@requires_duckdb
def test_constructor_reports_parse_coverage_from_materialized_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "labs.csv").write_text("value,when_time\n5.5,2026-01-02\nbad,not-a-date\n6.0,2026-01-03\n")
    spec_path = tmp_path / "construction_spec.json"
    _write_spec(spec_path, source, [
        {
            "table": "labs", "source_column": "value", "target_column": "value__norm",
            "kind": "numeric", "sql_expression": 'TRY_CAST("value" AS DOUBLE)',
            "parse_coverage": None, "unit_source_column": None, "unit_target_column": None,
            "canonical_unit": None, "observed_units": [], "unit_conversions": {},
        },
        {
            "table": "labs", "source_column": "when_time", "target_column": "when_time__norm",
            "kind": "datetime", "sql_expression": 'TRY_CAST("when_time" AS TIMESTAMP)',
            "parse_coverage": None, "unit_source_column": None, "unit_target_column": None,
            "canonical_unit": None, "observed_units": [], "unit_conversions": {},
        },
    ])
    result = DatasetConstructor().construct(spec_path, tmp_path / "out.duckdb")
    stats = {item.source_column: item for item in result.field_stats}
    assert stats["value"].normalized_fraction == pytest.approx(2 / 3)
    assert stats["when_time"].normalized_fraction == pytest.approx(2 / 3)


@requires_duckdb
def test_constructor_rejects_missing_source_column_without_replacing_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "labs.csv").write_text("value\n5.5\n")
    spec_path = tmp_path / "construction_spec.json"
    _write_spec(spec_path, source, [{
        "table": "labs", "source_column": "missing", "target_column": "missing__norm",
        "kind": "numeric", "sql_expression": 'TRY_CAST("missing" AS DOUBLE)',
        "parse_coverage": None, "unit_source_column": None, "unit_target_column": None,
        "canonical_unit": None, "observed_units": [], "unit_conversions": {},
    }])
    output = tmp_path / "out.duckdb"
    output.write_bytes(b"existing-output-must-survive")
    original = output.read_bytes()
    with pytest.raises(ValueError, match="missing source column"):
        DatasetConstructor().construct(spec_path, output)
    assert output.read_bytes() == original


@requires_duckdb
def test_constructor_rejects_unknown_table(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "labs.csv").write_text("value\n5.5\n")
    spec_path = tmp_path / "construction_spec.json"
    _write_spec(spec_path, source, [{
        "table": "not_labs", "source_column": "value", "target_column": "value__norm",
        "kind": "numeric", "sql_expression": 'TRY_CAST("value" AS DOUBLE)',
        "parse_coverage": None, "unit_source_column": None, "unit_target_column": None,
        "canonical_unit": None, "observed_units": [], "unit_conversions": {},
    }])
    with pytest.raises(ValueError, match="unknown table"):
        DatasetConstructor().construct(spec_path, tmp_path / "out.duckdb")


@requires_duckdb
def test_csv_lexical_values_survive_inspection_and_construction(tmp_path):
    from agents.cohort.adapters.inspector import CSVInspector
    source = tmp_path / "LABS.CSV"
    source.write_text('account,value,label\n001,1.50,NA\n002,2.00,NULL\n')
    adapter = tmp_path / "adapter"
    profile = CSVInspector().inspect(source, adapter)
    account = profile.tables[0].columns[0]
    assert account.sample_values == ["001", "002"]
    assert account.is_identifier_or_code
    output = tmp_path / "out.duckdb"
    DatasetConstructor().construct(adapter / "construction_spec.json", output)
    with duckdb.connect(str(output), read_only=True) as con:
        assert con.execute('SELECT account, value, label FROM LABS ORDER BY account').fetchall() == [
            ("001", "1.50", "NA"), ("002", "2.00", "NULL"),
        ]


@requires_duckdb
@pytest.mark.parametrize("destination", ["source", "spec", "same_report", "report_source"])
def test_constructor_rejects_destructive_destination_aliases(tmp_path, destination):
    source = tmp_path / "labs.csv"
    source.write_text("value\n5.5\n")
    spec = tmp_path / "spec.json"
    _write_spec(spec, source, [])
    output = tmp_path / "out.duckdb"
    report = None
    if destination == "source":
        output = source
    elif destination == "spec":
        output = spec
    elif destination == "same_report":
        report = output
    else:
        report = source
    source_before, spec_before = source.read_bytes(), spec.read_bytes()
    with pytest.raises(ValueError, match="overwrite|different"):
        DatasetConstructor().construct(spec, output, report_path=report)
    assert source.read_bytes() == source_before
    assert spec.read_bytes() == spec_before


@requires_duckdb
def test_failed_database_publication_restores_previous_report(tmp_path, monkeypatch):
    import agents.cohort.adapters.constructor as module
    source = tmp_path / "labs.csv"
    source.write_text("value\n5.5\n")
    spec = tmp_path / "spec.json"
    _write_spec(spec, source, [])
    output, report = tmp_path / "out.duckdb", tmp_path / "report.json"
    output.write_bytes(b"previous database")
    report.write_bytes(b"previous report")
    replace = module.os.replace

    def fail_database(src, dst):
        if Path(dst) == output:
            raise OSError("publication failed")
        return replace(src, dst)

    monkeypatch.setattr(module.os, "replace", fail_database)
    with pytest.raises(OSError, match="publication failed"):
        DatasetConstructor().construct(spec, output, report_path=report)
    assert output.read_bytes() == b"previous database"
    assert report.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".*building-*"))
    assert not list(tmp_path.glob(".*backup-*"))


@requires_duckdb
@pytest.mark.parametrize("expression", [
    "'not a number'",
    '1); CREATE TABLE injected(x INTEGER); SELECT (1',
])
def test_constructor_rejects_invalid_normalization_contract(tmp_path, expression):
    source = tmp_path / "labs.csv"
    source.write_text("value\n5.5\n")
    spec = tmp_path / "spec.json"
    _write_spec(spec, source, [{
        "table": "labs", "source_column": "value", "target_column": "normalized",
        "kind": "numeric", "sql_expression": expression,
    }])
    with pytest.raises(ValueError, match="numeric normalization|single SELECT"):
        DatasetConstructor().construct(spec, tmp_path / "out.duckdb")
    assert not (tmp_path / "out.duckdb").exists()


@requires_duckdb
def test_raw_table_names_do_not_collapse_punctuation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a-b.csv").write_text("value\n1\n")
    (source / "a_b.csv").write_text("value\n2\n")
    spec = tmp_path / "spec.json"
    _write_spec(spec, source, [])
    output = tmp_path / "out.duckdb"
    DatasetConstructor().construct(spec, output)
    with duckdb.connect(str(output), read_only=True) as con:
        assert con.execute('SELECT value FROM "a-b"').fetchone() == ("1",)
        assert con.execute('SELECT value FROM a_b').fetchone() == ("2",)


@pytest.mark.parametrize("overrides,match", [
    ({"unit_source_column": "unit", "canonical_unit": "mg", "unit_conversions": {"g": {}}}, "unit_target_column"),
    ({"unit_source_column": "unit", "unit_target_column": "unit_norm", "canonical_unit": "mg", "unit_conversions": {"µg": {}, " μg ": {}}}, "duplicate normalized"),
])
def test_constructor_rejects_incomplete_or_ambiguous_unit_plans(tmp_path, overrides, match):
    source = tmp_path / "labs.csv"
    source.write_text("value,unit\n1,g\n")
    with pytest.raises(ValueError, match=match):
        DatasetConstructor().validate_spec(ConstructionSpec("csv", str(source), [_derived(**overrides)]))


@requires_duckdb
def test_inspector_generates_usable_dates_and_unique_derived_names(tmp_path):
    from agents.cohort.adapters.inspector import CSVInspector
    source = tmp_path / "labs.csv"
    source.write_text('value,value__norm,when_time\n1.5,2.5,01/02/2026\n')
    adapter = tmp_path / "adapter"
    CSVInspector().inspect(source, adapter)
    result = DatasetConstructor().construct(adapter / "construction_spec.json", tmp_path / "out.duckdb")
    with duckdb.connect(str(result.output_path), read_only=True) as con:
        row = con.execute('SELECT value__norm, value__norm_2, when_time__norm FROM labs').fetchone()
        assert row[:2] == ("2.5", 1.5)
        assert str(row[2]).startswith("2026-01-02")


@requires_duckdb
def test_normalization_cannot_read_external_files(tmp_path):
    source = tmp_path / "labs.csv"
    source.write_text("value\n1\n")
    external = tmp_path / "external.csv"
    external.write_text("secret\n42\n")
    spec = tmp_path / "spec.json"
    _write_spec(spec, source, [{
        "table": "labs", "source_column": "value", "target_column": "normalized",
        "kind": "numeric", "sql_expression": f"(SELECT secret FROM read_csv_auto('{external}'))",
    }])
    with pytest.raises(duckdb.Error, match="disabled|external"):
        DatasetConstructor().construct(spec, tmp_path / "out.duckdb")
    assert not (tmp_path / "out.duckdb").exists()


@requires_duckdb
def test_normalization_rejects_array_and_nonfinite_outputs(tmp_path):
    source = tmp_path / 'numbers.csv'
    source.write_text('value\nNaN\nInfinity\n1e309\n1.5\n')
    spec = tmp_path / 'spec.json'
    _write_spec(spec, source, [{'table': 'numbers', 'source_column': 'value', 'target_column': 'normalized', 'kind': 'numeric', 'sql_expression': 'TRY_CAST(value AS DOUBLE)'}])
    result = DatasetConstructor().construct(spec, tmp_path / 'out.duckdb')
    assert result.field_stats[0].normalized_non_null == 1
    assert result.field_stats[0].invalid_value_rows == 3
    _write_spec(spec, source, [{'table': 'numbers', 'source_column': 'value', 'target_column': 'normalized', 'kind': 'numeric', 'sql_expression': '[1, 2]'}])
    with pytest.raises(ValueError, match='numeric normalization returned'):
        DatasetConstructor().construct(spec, tmp_path / 'bad.duckdb')


@requires_duckdb
def test_exact_text_numeric_normalization_and_native_storage(tmp_path):
    from agents.cohort.adapters.inspector import DatasetInspector
    from decimal import Decimal
    source = tmp_path / 'source'; source.mkdir()
    (source / 'numbers.csv').write_text('amount,small\n9007199254740993,1.25\n9007199254740994,1e-3\n')
    package = tmp_path / 'package'
    DatasetInspector().inspect(source, package)
    db = tmp_path / 'out.duckdb'
    DatasetConstructor().construct(package / 'construction_spec.json', db)
    with duckdb.connect(str(db)) as con:
        assert con.execute('select amount, amount__norm, small__norm from numbers order by amount').fetchall() == [
            ('9007199254740993', 9007199254740993, Decimal('1.25')),
            ('9007199254740994', 9007199254740994, Decimal('.001')),
        ]
    # A late fractional value must not be rounded just because the sample was integer.
    spec = tmp_path / 'spec.json'
    _write_spec(spec, source, [{'table': 'numbers', 'source_column': 'small', 'target_column': 'normalized', 'kind': 'numeric',
                              'sql_expression': 'CASE WHEN adapter_numeric_exact(small, CAST(TRY_CAST(small AS HUGEINT) AS VARCHAR)) THEN TRY_CAST(small AS HUGEINT) END'}])
    result = DatasetConstructor().construct(spec, tmp_path / 'integers.duckdb')
    assert result.field_stats[0].invalid_value_rows == 2


@requires_duckdb
def test_full_adapter_preserves_previous_outputs_on_hard_failure(tmp_path):
    from agents.cohort.adapters.constructor import adapt_dataset
    source = tmp_path / 'source'; source.mkdir()
    (source / 'patients.csv').write_text('patient_id,gender\n001,F\n')
    root = tmp_path / 'out'
    assert adapt_dataset(source, root)['status'] == 'ready'
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    (source / 'patients.csv').write_text('bad,bad\nx,y\n')
    with pytest.raises(ValueError, match='duplicate CSV'):
        adapt_dataset(source, root)
    assert all((root / name).read_bytes() == value for name, value in before.items())
    assert json.loads((root / 'adaptation_failure.json').read_text())['status'] == 'failed'


@requires_duckdb
def test_unified_adapter_cannot_publish_over_its_source(tmp_path):
    from agents.cohort.adapters.constructor import adapt_dataset
    source = tmp_path / 'dataset.duckdb'
    with duckdb.connect(str(source)) as con:
        con.execute('CREATE TABLE data(value INTEGER)')
    before = source.read_bytes()
    with pytest.raises(ValueError, match='overwrite source'):
        adapt_dataset(source, tmp_path)
    assert source.read_bytes() == before
