from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from infrastructure.database.duckdb import (
    DuckDBConfig,
    DuckDBConnectionFactory,
)


def test_connection_factory_executes_and_closes(tmp_path):
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE numbers(value INTEGER)")
        connection.execute("INSERT INTO numbers VALUES (1), (2)")

    factory = DuckDBConnectionFactory(
        DuckDBConfig(path, read_only=True)
    )

    assert factory.execute(
        "SELECT SUM(value) FROM numbers"
    ) == [(3,)]
    assert factory.table_exists("numbers")
    assert not factory.table_exists("missing")


def test_qualified_table_name_resolves_non_default_schema(tmp_path):
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SCHEMA vocab")
        connection.execute("CREATE TABLE vocab.d_icd_diagnoses(icd_code VARCHAR)")
        connection.execute("INSERT INTO vocab.d_icd_diagnoses VALUES ('25000')")
        connection.execute("CREATE TABLE patients(subject_id BIGINT)")

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))

    # table_exists() alone would report this table as present even though a
    # plain, unqualified query against it fails: it searches
    # information_schema.tables across every schema, not just the ones on
    # DuckDB's default search_path.
    assert factory.table_exists("d_icd_diagnoses")

    qualified = factory.qualified_table_name("d_icd_diagnoses")
    assert qualified == '"vocab"."d_icd_diagnoses"'
    assert factory.execute(f"SELECT COUNT(*) FROM {qualified}") == [(1,)]

    # A table already in the default schema still resolves correctly.
    assert factory.qualified_table_name("patients") == '"main"."patients"'


def test_qualified_table_name_rejects_missing_or_ambiguous_names(tmp_path):
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SCHEMA a")
        connection.execute("CREATE SCHEMA b")
        connection.execute("CREATE TABLE a.dup(x INTEGER)")
        connection.execute("CREATE TABLE b.dup(x INTEGER)")

    factory = DuckDBConnectionFactory(DuckDBConfig(path, read_only=True))

    with pytest.raises(ValueError, match="not found"):
        factory.qualified_table_name("missing")

    with pytest.raises(ValueError, match="multiple schemas"):
        factory.qualified_table_name("dup")
