from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import hashlib
import shutil
import re

from agents.cohort.adapters.models import ConstructionSpec, DerivedFieldSpec
from agents.cohort.adapters.semantics import source_table_name, validate_table_names, validate_csv_header, numeric_storage_type, normalized_unit


@dataclass(frozen=True, slots=True)
class FieldConstructionStats:
    table: str
    source_column: str
    target_column: str
    kind: str
    rows: int
    source_non_null: int
    normalized_non_null: int
    normalized_fraction: float
    unsupported_unit_rows: int = 0
    invalid_value_rows: int = 0


@dataclass(frozen=True, slots=True)
class ConstructionResult:
    output_path: Path
    tables: tuple[str, ...]
    derived_fields: int
    report_path: Path | None = None
    field_stats: tuple[FieldConstructionStats, ...] = field(default_factory=tuple)


class DatasetConstructor:
    """Deterministically materializes an adapted DuckDB from an audited spec.

    The constructor never edits source data. It builds into a temporary DuckDB and
    atomically replaces the requested destination only after all source tables,
    derived expressions, and conversion instructions have validated successfully.
    """

    _SUPPORTED_SOURCE_KINDS = {"csv", "parquet", "sql:duckdb", "sql:sqlite"}
    _SUPPORTED_DERIVED_KINDS = {"numeric", "datetime"}

    def load_spec(self, path: str | Path) -> ConstructionSpec:
        raw = json.loads(Path(path).read_text())
        return ConstructionSpec(
            source_kind=str(raw["source_kind"]),
            source_path=str(raw["source_path"]),
            output_format=str(raw.get("output_format", "duckdb")),
            warnings=list(raw.get("warnings", [])),
            derived_fields=[DerivedFieldSpec(**item) for item in raw.get("derived_fields", [])],
        )

    def validate_spec(self, spec: ConstructionSpec) -> None:
        if spec.source_kind not in self._SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"unsupported source_kind: {spec.source_kind}")
        if spec.output_format.lower() != "duckdb":
            raise ValueError(f"unsupported output_format: {spec.output_format}")
        source = Path(spec.source_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(source)

        targets: set[tuple[str, str]] = set()
        for item in spec.derived_fields:
            if item.kind not in self._SUPPORTED_DERIVED_KINDS:
                raise ValueError(
                    f"unsupported derived field kind {item.kind!r} for "
                    f"{item.table}.{item.source_column}"
                )
            if not item.table or not item.source_column or not item.target_column:
                raise ValueError("derived field table/source_column/target_column must be non-empty")
            key = (item.table, item.target_column)
            if key in targets:
                raise ValueError(f"duplicate derived target: {item.table}.{item.target_column}")
            targets.add(key)
            if not isinstance(item.sql_expression, str) or not item.sql_expression.strip():
                raise ValueError("derived field sql_expression must be non-empty")
            if item.unit_target_column and not item.unit_source_column:
                raise ValueError(
                    f"unit_target_column requires unit_source_column for "
                    f"{item.table}.{item.source_column}"
                )
            if item.unit_conversions and not item.unit_source_column:
                raise ValueError(
                    f"unit conversions require unit_source_column for "
                    f"{item.table}.{item.source_column}"
                )
            if item.unit_conversions and not item.canonical_unit:
                raise ValueError(
                    f"unit conversions require canonical_unit for "
                    f"{item.table}.{item.source_column}"
                )
            if item.unit_conversions and (item.kind != "numeric" or not item.unit_target_column):
                raise ValueError("unit conversions require a numeric field and unit_target_column")
            normalized_units = [normalized_unit(str(unit)) for unit in item.unit_conversions]
            if len(normalized_units) != len(set(normalized_units)):
                raise ValueError("duplicate normalized unit conversion keys")
            for raw_unit, conversion in item.unit_conversions.items():
                self._validate_unit_conversion(raw_unit, conversion, item)

    @staticmethod
    def _validate_unit_conversion(
        raw_unit: str,
        conversion: dict[str, Any],
        item: DerivedFieldSpec,
    ) -> None:
        if not str(raw_unit).strip():
            raise ValueError(
                f"empty unit conversion key for {item.table}.{item.source_column}"
            )
        if not isinstance(conversion, dict):
            raise ValueError(
                f"unit conversion for {raw_unit!r} must be an object"
            )
        try:
            factor = float(conversion.get("factor", 1.0))
            offset = float(conversion.get("offset", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid unit conversion for {raw_unit!r} on "
                f"{item.table}.{item.source_column}"
            ) from exc
        if not (factor == factor and abs(factor) != float("inf")) or factor <= 0:
            raise ValueError(f"unit factor must be finite and positive for {raw_unit!r}")
        if not (offset == offset and abs(offset) != float("inf")):
            raise ValueError(f"non-finite unit offset for {raw_unit!r}")

    def construct(
        self,
        spec_path: str | Path,
        output_path: str | Path,
        *,
        report_path: str | Path | None = None,
    ) -> ConstructionResult:
        spec = self.load_spec(spec_path)
        self.validate_spec(spec)
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("duckdb is required to construct an adapted dataset") from exc

        output = Path(output_path).expanduser().resolve()
        report = (
            Path(report_path).expanduser().resolve()
            if report_path is not None
            else output.with_suffix(output.suffix + ".construction.json")
        )
        source = Path(spec.source_path).expanduser().resolve()
        protected = [Path(spec_path).resolve()]
        if source.is_dir():
            protected.extend(path for path in source.rglob("*") if path.is_file())
        else:
            protected.append(source)
        for destination in (output, report):
            for path in protected:
                if destination == path or (destination.exists() and destination.samefile(path)):
                    raise ValueError(f"destination would overwrite source/specification: {destination}")
        if output == report or (output.exists() and report.exists() and output.samefile(report)):
            raise ValueError("database and report destinations must be different")
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)

        # Build away from the destination. os.replace() is atomic on the same filesystem.
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{output.name}.building-", suffix=".duckdb", dir=output.parent
        )
        os.close(fd)
        temp_output = Path(temp_name)
        temp_report = None
        try:
            report_fd, report_name = tempfile.mkstemp(prefix=f".{report.name}.building-", dir=report.parent)
            os.close(report_fd)
            temp_report = Path(report_name)
            # DuckDB expects to initialize the file itself.
            temp_output.unlink(missing_ok=True)
            result = self._construct_into(spec, temp_output, temp_report)
            self._publish(temp_output, output, temp_report, report)
            return ConstructionResult(
                output_path=output,
                tables=result.tables,
                derived_fields=result.derived_fields,
                report_path=report,
                field_stats=result.field_stats,
            )
        finally:
            temp_output.unlink(missing_ok=True)
            if temp_report is not None:
                temp_report.unlink(missing_ok=True)

    @staticmethod
    def _publish(temp_output: Path, output: Path, temp_report: Path, report: Path) -> None:
        """Publish both completed artifacts, restoring the report on DB failure.

        Each replacement is atomic. The pair is not a filesystem transaction.
        A normal publication error leaves the previous database/report intact.
        """
        backup = None
        retain_backup = False
        if report.exists():
            fd, name = tempfile.mkstemp(prefix=f".{report.name}.backup-", dir=report.parent)
            os.close(fd)
            backup = Path(name)
        try:
            if backup is not None:
                shutil.copy2(report, backup)
            os.replace(temp_report, report)
            try:
                os.replace(temp_output, output)
            except Exception as original_error:
                try:
                    if backup is None:
                        report.unlink(missing_ok=True)
                    else:
                        os.replace(backup, report)
                except OSError as exc:
                    retain_backup = True
                    original_error.add_note(f"Report rollback failed: {exc}. Backup retained at {backup}")
                raise
        finally:
            if backup is not None and not retain_backup:
                backup.unlink(missing_ok=True)

    def _construct_into(
        self,
        spec: ConstructionSpec,
        output: Path,
        report: Path,
    ) -> ConstructionResult:
        import duckdb

        by_table: dict[str, list[DerivedFieldSpec]] = {}
        for item in spec.derived_fields:
            by_table.setdefault(item.table, []).append(item)

        connection = duckdb.connect(str(output))
        field_stats: list[FieldConstructionStats] = []
        try:
            from decimal import Decimal, InvalidOperation
            def exact_numeric(raw, converted):
                if raw is None or converted is None:
                    return False
                try:
                    original, normalized = Decimal(raw.strip()), Decimal(converted)
                    return original.is_finite() and normalized.is_finite() and original == normalized
                except InvalidOperation:
                    return False
            connection.create_function("adapter_numeric_exact", exact_numeric, ["VARCHAR", "VARCHAR"], "BOOLEAN", null_handling="special")
            tables = self._materialize_source(connection, spec)
            validate_table_names(tables)
            # All source data is local to this disposable database now. Audited
            # expressions must not read files, install extensions, or attach DBs.
            connection.execute("SET enable_external_access=false")
            known_tables = set(tables)
            unknown_tables = sorted(set(by_table) - known_tables)
            if unknown_tables:
                raise ValueError(
                    "construction spec references unknown table(s): " + ", ".join(unknown_tables)
                )

            for table in tables:
                raw_table = f"__raw_{self._safe_name(table)}"
                raw_columns = [
                    row[1]
                    for row in connection.execute(
                        f"PRAGMA table_info({self._quote(raw_table)})"
                    ).fetchall()
                ]
                raw_set = set(raw_columns)
                derived_for_table = by_table.get(table, [])
                self._validate_table_spec(table, raw_set, derived_for_table)

                expressions = [self._quote(column) for column in raw_columns]
                for derived in derived_for_table:
                    value_expression, unit_expression = self._derived_expressions(derived)
                    self._validate_expression(connection, raw_table, value_expression, kind=derived.kind)
                    present = f"{self._quote(derived.source_column)} IS NOT NULL"
                    if derived.kind == "numeric":
                        value_expression = f"CASE WHEN {present} AND isfinite({value_expression}) THEN ({value_expression}) ELSE NULL END"
                    else:
                        value_expression = f"CASE WHEN {present} THEN ({value_expression}) ELSE NULL END"
                    expressions.append(
                        f"({value_expression}) AS {self._quote(derived.target_column)}"
                    )
                    if derived.unit_source_column and derived.unit_target_column:
                        assert unit_expression is not None
                        self._validate_expression(connection, raw_table, unit_expression)
                        expressions.append(
                            f"({unit_expression}) AS {self._quote(derived.unit_target_column)}"
                        )

                connection.execute(
                    f"CREATE TABLE {self._quote(table)} AS "
                    f"SELECT {', '.join(expressions)} FROM {self._quote(raw_table)}"
                )
                for derived in derived_for_table:
                    field_stats.append(self._field_stats(connection, table, derived))
                connection.execute(f"DROP TABLE {self._quote(raw_table)}")
        finally:
            connection.close()

        self._write_construction_report(report, spec, tables, field_stats)
        return ConstructionResult(
            output,
            tuple(tables),
            len(spec.derived_fields),
            report,
            tuple(field_stats),
        )

    def _validate_table_spec(
        self,
        table: str,
        raw_columns: set[str],
        derived: list[DerivedFieldSpec],
    ) -> None:
        target_names = {item.target_column for item in derived}
        unit_targets = {item.unit_target_column for item in derived if item.unit_target_column}
        all_targets = [name for item in derived for name in (item.target_column, item.unit_target_column) if name]
        if len({name.casefold() for name in all_targets}) != len(all_targets):
            raise ValueError(f"duplicate derived value/unit targets in table {table}")
        collisions = sorted(name for name in target_names | unit_targets if name.casefold() in {raw.casefold() for raw in raw_columns})
        if collisions:
            raise ValueError(
                f"derived column(s) already exist in source table {table}: "
                + ", ".join(collisions)
            )
        if target_names & unit_targets:
            duplicates = sorted(target_names & unit_targets)
            raise ValueError(
                f"derived value/unit targets collide in table {table}: " + ", ".join(duplicates)
            )
        for item in derived:
            if item.source_column not in raw_columns:
                raise ValueError(
                    f"missing source column {table}.{item.source_column}"
                )
            if item.unit_source_column and item.unit_source_column not in raw_columns:
                raise ValueError(
                    f"missing unit source column {table}.{item.unit_source_column}"
                )

    def _derived_expressions(
        self,
        derived: DerivedFieldSpec,
    ) -> tuple[str, str | None]:
        base = derived.sql_expression
        if not derived.unit_source_column or not derived.unit_target_column:
            return base, None

        unit_col = self._quote(derived.unit_source_column)
        unit_expression_key = f"REPLACE(TRIM(CAST({unit_col} AS VARCHAR)), 'µ', 'μ')"
        if not derived.unit_conversions:
            return (
                base,
                f"NULLIF(TRIM(CAST({unit_col} AS VARCHAR)), '')",
            )

        # A non-empty conversion plan is fail-closed: only explicitly audited units
        # receive normalized values. Unknown units keep the raw source columns but
        # produce NULL normalized value/unit fields.
        value_cases: list[str] = []
        unit_cases: list[str] = []
        for raw_unit, conversion in sorted(
            derived.unit_conversions.items(), key=lambda item: str(item[0]).casefold()
        ):
            key = self._sql_literal(normalized_unit(str(raw_unit)))
            from decimal import Decimal
            factor = Decimal(str(conversion.get("factor", 1)))
            offset = Decimal(str(conversion.get("offset", 0)))
            converted = base if factor == 1 and offset == 0 else f"(({base}) * {format(factor, 'f')} + {format(offset, 'f')})"
            value_cases.append(f"WHEN {key} THEN {converted}")
            unit_cases.append(
                f"WHEN {key} THEN {self._sql_literal(str(derived.canonical_unit))}"
            )
        value_expression = (
            f"CASE {unit_expression_key} {' '.join(value_cases)} ELSE NULL END"
        )
        unit_expression = (
            f"CASE {unit_expression_key} {' '.join(unit_cases)} ELSE NULL END"
        )
        return value_expression, unit_expression

    def _field_stats(
        self,
        connection: Any,
        table: str,
        derived: DerivedFieldSpec,
    ) -> FieldConstructionStats:
        qtable = self._quote(table)
        qsource = self._quote(derived.source_column)
        qtarget = self._quote(derived.target_column)
        row = connection.execute(
            f"SELECT COUNT(*), COUNT({qsource}), COUNT({qtarget}) FROM {qtable}"
        ).fetchone()
        rows, source_non_null, normalized_non_null = (int(row[0]), int(row[1]), int(row[2]))
        unsupported = 0
        if (
            derived.unit_source_column
            and derived.unit_target_column
            and derived.unit_conversions
        ):
            qunit_source = self._quote(derived.unit_source_column)
            qunit_target = self._quote(derived.unit_target_column)
            unsupported = int(connection.execute(
                f"SELECT COUNT(*) FROM {qtable} "
                f"WHERE {qsource} IS NOT NULL AND {qunit_source} IS NOT NULL "
                f"AND {qunit_target} IS NULL"
            ).fetchone()[0])
        fraction = normalized_non_null / source_non_null if source_non_null else 1.0
        return FieldConstructionStats(
            table=table,
            source_column=derived.source_column,
            target_column=derived.target_column,
            kind=derived.kind,
            rows=rows,
            source_non_null=source_non_null,
            normalized_non_null=normalized_non_null,
            normalized_fraction=fraction,
            unsupported_unit_rows=unsupported,
            invalid_value_rows=max(0, source_non_null - normalized_non_null - unsupported),
        )

    def _write_construction_report(
        self,
        path: Path,
        spec: ConstructionSpec,
        tables: list[str],
        field_stats: list[FieldConstructionStats],
    ) -> None:
        payload = {
            "source_kind": spec.source_kind,
            "source_path": spec.source_path,
            "output_format": spec.output_format,
            "tables": tables,
            "derived_fields": [asdict(item) for item in field_stats],
            "warnings": spec.warnings,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def _materialize_source(self, connection: Any, spec: ConstructionSpec) -> list[str]:
        source = Path(spec.source_path).expanduser().resolve()
        if spec.source_kind == "csv":
            files = self._discover_csvs(source)
            if not files:
                raise FileNotFoundError(f"no CSV files found under {source}")
            tables = []
            validate_table_names([source_table_name(file) for file in files])
            for file in files:
                table = source_table_name(file)
                validate_csv_header(file)
                source_sql = f"read_csv_auto('{self._escape_path(file)}', header=true, all_varchar=true)"
                self._create_raw(connection, table, source_sql)
                tables.append(table)
            return tables
        if spec.source_kind == "parquet":
            files = self._discover_parquet(source)
            if not files:
                raise FileNotFoundError(f"no Parquet files found under {source}")
            tables = []
            validate_table_names([source_table_name(file) for file in files])
            for file in files:
                table = source_table_name(file)
                source_sql = f"read_parquet('{self._escape_path(file)}')"
                self._create_raw(connection, table, source_sql)
                tables.append(table)
            return tables
        if spec.source_kind == "sql:duckdb":
            return self._materialize_duckdb(connection, source)
        if spec.source_kind == "sql:sqlite":
            return self._materialize_sqlite(connection, source)
        raise ValueError(f"unsupported source_kind: {spec.source_kind}")

    def _create_raw(self, connection: Any, table: str, source_sql: str) -> None:
        raw_table = f"__raw_{self._safe_name(table)}"
        connection.execute(
            f"CREATE TABLE {self._quote(raw_table)} AS SELECT * FROM {source_sql}"
        )

    def _materialize_duckdb(self, connection: Any, source: Path) -> list[str]:
        import duckdb

        source_connection = duckdb.connect(str(source), read_only=True)
        try:
            rows = source_connection.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type='BASE TABLE' AND table_schema NOT IN ('information_schema','pg_catalog') "
                "ORDER BY table_schema, table_name"
            ).fetchall()
        finally:
            source_connection.close()
        alias = "__adapter_source"
        escaped = self._escape_path(source)
        connection.execute(f"ATTACH '{escaped}' AS {self._quote(alias)} (READ_ONLY)")
        try:
            tables: list[str] = []
            for schema, table_name in rows:
                logical = str(table_name) if schema in {"main", "public"} else f"{schema}.{table_name}"
                source_sql = (
                    f"{self._quote(alias)}.{self._quote(str(schema))}.{self._quote(str(table_name))}"
                )
                self._create_raw(connection, logical, source_sql)
                tables.append(logical)
            return tables
        finally:
            connection.execute(f"DETACH {self._quote(alias)}")

    def _materialize_sqlite(self, connection: Any, source: Path) -> list[str]:
        import sqlite3

        con = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        try:
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            materialized: list[str] = []
            for table in tables:
                cursor = con.execute(f"SELECT * FROM {self._quote(table)}")
                columns = [item[0] for item in cursor.description or ()]
                if not columns:
                    continue
                raw_table = f"__raw_{self._safe_name(table)}"
                # Preserve SQLite's actual storage classes rather than declared
                # affinity. Mixed columns retain their tags in a DuckDB UNION.
                storage_types = []
                for column in columns:
                    kinds = {row[0] for row in con.execute(
                        f"SELECT DISTINCT typeof({self._quote(column)}) FROM {self._quote(table)} "
                        f"WHERE {self._quote(column)} IS NOT NULL"
                    )}
                    storage_types.append("UNION(i BIGINT, r DOUBLE, s VARCHAR, b BLOB)" if len(kinds) > 1 else
                        {"integer": "BIGINT", "real": "DOUBLE", "blob": "BLOB", "text": "VARCHAR"}.get(next(iter(kinds), "text")))
                definitions = ", ".join(f"{self._quote(column)} {kind}" for column, kind in zip(columns, storage_types))
                connection.execute(f"CREATE TABLE {self._quote(raw_table)} ({definitions})")
                from collections import defaultdict
                while rows := cursor.fetchmany(5000):
                    batches = defaultdict(list)
                    for row in rows:
                        expressions, values = [], []
                        for value, kind in zip(row, storage_types):
                            if kind.startswith("UNION") and value is not None:
                                tag, member = {int: ("i", "BIGINT"), float: ("r", "DOUBLE"), str: ("s", "VARCHAR"), bytes: ("b", "BLOB")}[type(value)]
                                expressions.append(f"union_value({tag} := CAST(? AS {member}))")
                            else:
                                expressions.append("?")
                            values.append(value)
                        batches[", ".join(expressions)].append(values)
                    for expressions, values in batches.items():
                        connection.executemany(f"INSERT INTO {self._quote(raw_table)} VALUES ({expressions})", values)
                materialized.append(table)
            return materialized
        finally:
            con.close()

    @staticmethod
    def _validate_expression(connection: Any, raw_table: str, expression: str, *, kind: str | None = None) -> None:
        query = f"SELECT ({expression}) FROM {DatasetConstructor._quote(raw_table)} LIMIT 50"
        statements = connection.extract_statements(query)
        if len(statements) != 1 or str(statements[0].type) != "StatementType.SELECT":
            raise ValueError("normalization must be a single SELECT expression")
        result = connection.execute(query)
        if len(result.description) != 1:
            raise ValueError("normalization must return one column")
        value_type = str(result.description[0][1]).upper()
        numeric = numeric_storage_type(value_type)
        if kind == "numeric" and not numeric:
            raise ValueError(f"numeric normalization returned {value_type}")
        if kind == "datetime" and value_type not in {"DATE", "TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S", "TIMESTAMP WITH TIME ZONE"}:
            raise ValueError(f"datetime normalization returned {value_type}")
        result.fetchall()

    @staticmethod
    def _discover_csvs(source: Path) -> list[Path]:
        if source.is_file():
            return [source]
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and (path.name.lower().endswith(".csv") or path.name.lower().endswith(".csv.gz"))
        )

    @staticmethod
    def _discover_parquet(source: Path) -> list[Path]:
        if source.is_file():
            return [source]
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".parquet")

    @staticmethod
    def _safe_name(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _escape_path(path: Path) -> str:
        return str(path.resolve()).replace("'", "''")

    @staticmethod
    def _sql_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"


def adapt_dataset(source, output_dir, *, source_format="auto", sql_dialect="auto", llm_inspector=None, mapping=None):
    """Inspect, construct, validate and publish a DB + catalog + adaptation log.

    A missing/ambiguous mapping produces a useful database marked needs_review.
    It never fabricates patient identity. Ordinary build failures preserve the
    previous output bundle and write a separate failure log.
    """
    from agents.cohort.adapters.inspector import DatasetInspector
    from agents.cohort.adapters.models import publish_files, write_json
    from agents.cohort.adapters.registry import CustomDataset
    output = Path(output_dir).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    if output == source or (source.is_dir() and output.is_relative_to(source)):
        raise ValueError("adapter output must be outside the source dataset")
    bundle_names = ("dataset.duckdb", "dataset.duckdb.construction.json", "schema_profile.json",
                    "field_catalog.json", "construction_spec.json", "cohort_mapping.json",
                    "report.md", "adaptation_log.json", "adaptation_failure.json")
    if source.is_file():
        for name in bundle_names:
            destination = output / name
            if destination == source or (destination.exists() and destination.samefile(source)):
                raise ValueError(f"adapter output would overwrite source: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".adapt-", dir=output) as directory:
            stage = Path(directory)
            profile = DatasetInspector(llm_inspector=llm_inspector).inspect(
                source, stage, source_format=source_format, sql_dialect=sql_dialect,
            )
            built = DatasetConstructor().construct(stage / "construction_spec.json", stage / "dataset.duckdb")
            from datetime import datetime, timezone
            import uuid
            log = {
                "run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                "source_path": str(source), "source_kind": profile.source_kind,
                "relationship_candidates": audit_relationships(stage / "dataset.duckdb", profile),
                "status": "needs_review", "database_created": True,
                "database": "dataset.duckdb", "normalization": [asdict(x) for x in built.field_stats],
                "timezone_policy": "discard offsets. Assume consistent patient-local timestamps",
                "molar_conversion_policy": "no substance-dependent conversions. Clinician review required",
                "issues": list(json.loads((stage / "construction_spec.json").read_text()).get("warnings", [])), "mapping_validation": [],
            }
            if mapping is None:
                # Names only nominate an anchor. Full-data uniqueness, types and
                # overlap are checked below before it becomes executable.
                candidates = []
                for table in profile.tables:
                    if table.name.casefold() in {"patients", "patient", "person"}:
                        for column in table.columns:
                            if column.name.casefold() in {"subject_id", "patient_id", "person_id"}:
                                candidates.append((table.name, column.name))
                if len(candidates) == 1:
                    table, column = candidates[0]
                    mapping = {"patient": {"table": table, "columns": [column]}, "tables": {
                        t.name: {"patient_columns": [column]} for t in profile.tables
                        if any(c.name == column for c in t.columns)
                    }}
                    log["issues"].append({"code": "inferred_identity", "message": "Patient identity nominated by conventional names and validated against all rows. Review its clinical meaning."})
                else:
                    mapping = {"patient": {}, "tables": {}}
                    log["issues"].append({"code": "patient_mapping_required", "message": "No unambiguous patient anchor. Supply a reviewed mapping. Names/overlap alone cannot establish identity."})
            write_json(stage / "cohort_mapping.json", mapping)
            if mapping.get("patient") and mapping.get("tables"):
                try:
                    dataset = CustomDataset(stage)
                    log["mapping_validation"] = dataset.validation
                    log["status"] = "ready"
                except Exception as exc:
                    log["issues"].append({"code": "mapping_invalid", "message": str(exc)})
            excluded_fields = [f"{t.name}.{c.name}" for t in profile.tables for c in t.columns
                               if any(token in c.storage_type.upper() for token in ("[", "STRUCT", "MAP(", "UNION", "BLOB", "SQLITE_MIXED"))]
            if excluded_fields:
                log["issues"].append({"code": "non_scalar_or_mixed_fields", "fields": excluded_fields,
                    "message": "Raw typed values retained. Fields excluded from prompts until reviewed."})
            unmapped = sorted({t.name for t in profile.tables} - set(mapping.get("tables", {})))
            if unmapped:
                log["issues"].append({"code": "unmapped_tables", "tables": unmapped,
                    "message": "Retained in database, unavailable as cohort domains until patient mapping is reviewed."})
            for stat in built.field_stats:
                if stat.invalid_value_rows or stat.unsupported_unit_rows:
                    log["issues"].append({"code": "normalization_failures", "table": stat.table,
                        "column": stat.source_column, "invalid_value_rows": stat.invalid_value_rows,
                        "unsupported_unit_rows": stat.unsupported_unit_rows,
                        "message": "Raw values retained. Normalized failures are NULL."})
            write_json(stage / "adaptation_log.json", log)
            names = [p.name for p in stage.iterdir() if p.is_file()]
            publish_files([(stage / name, output / name) for name in names])
        return log
    except Exception as exc:
        # Does not overwrite the last successful run's log/database.
        write_json(output / "adaptation_failure.json", {"status": "failed", "error": str(exc)})
        raise


def audit_relationships(database_path, profile):
    """Measure nominated key pairs without treating overlap as semantic proof."""
    import duckdb
    from dataclasses import asdict
    q = DatasetConstructor._quote
    results = []
    with duckdb.connect(str(database_path), read_only=True) as con:
        for candidate in profile.relationships:
            entry = asdict(candidate)
            left, right = q(candidate.left_column), q(candidate.right_column)
            lt, rt = q(candidate.left_table), q(candidate.right_table)
            left_type = con.execute(f'SELECT {left} FROM {lt} LIMIT 0').description[0][1]
            right_type = con.execute(f'SELECT {right} FROM {rt} LIMIT 0').description[0][1]
            entry['compatible_types'] = str(left_type) == str(right_type)
            entry['status'] = 'proposal_only'
            for label, table, column in [('left', lt, left), ('right', rt, right)]:
                total, present, unique = con.execute(f'SELECT count(*), count({column}), count(DISTINCT {column}) FROM {table}').fetchone()
                entry[label + '_rows'] = total
                entry[label + '_null_rows'] = total - present
                entry[label + '_unique_key'] = total == present == unique
            if entry['compatible_types']:
                count, overlap = con.execute(
                    f'SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {rt} r WHERE r.{right} = l.{left})) FROM {lt} l'
                ).fetchone()
                entry['overlap_fraction'] = overlap / count if count else 0.0
                entry['unmatched_left_rows'] = count - overlap
            entry['confidence'] = 0.0  # Data compatibility is not proof of patient identity.
            results.append(entry)
    return results
