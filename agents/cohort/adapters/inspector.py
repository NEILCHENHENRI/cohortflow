from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math
import re
import tempfile
import logging

from agents.cohort.adapters.llm_inspector import AdapterLLMInspector
from agents.cohort.adapters.models import (
    ColumnProfile,
    ConstructionSpec,
    DatabaseProfile,
    DerivedFieldSpec,
    FieldCatalogEntry,
    RelationshipProfile,
    TableProfile,
    write_json,
    publish_files,
)
from agents.cohort.adapters.semantics import (
    numeric_storage_type,
    default_semantic_description,
    identifier_or_code_hint,
    semantic_field_name,
    source_table_name,
    validate_table_names,
    csv_text_stream,
    validate_csv_header,
)

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_DATE_PATTERNS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
)


class _InspectorCommon:
    """Shared finalization logic for tabular data-source inspectors."""

    def _finalize_profile(
        self,
        *,
        source_kind: str,
        source_path: str,
        tables: list[TableProfile],
        output_dir: str | Path,
    ) -> DatabaseProfile:
        validate_table_names([table.name for table in tables])
        self._apply_llm_semantics(tables)
        relationships = self._infer_relationships(tables)
        profile = DatabaseProfile(
            source_kind=source_kind,
            source_path=source_path,
            tables=tables,
            relationships=relationships,
        )
        field_catalog = self._field_catalog(profile)
        field_names = [item.semantic_field for item in field_catalog]
        if len(set(field_names)) != len(field_names):
            raise ValueError("ambiguous semantic field names. Rename dotted table/column names")
        construction = self._construction_spec(profile)
        construction.source_kind = source_kind

        output = Path(output_dir).expanduser().resolve()
        names = ("schema_profile.json", "field_catalog.json", "construction_spec.json", "report.md")
        sources = {Path(source_path).resolve(), *(Path(table.source_path).resolve() for table in tables)}
        for name in names:
            destination = output / name
            if any(destination.resolve() == source or (
                destination.exists() and source.is_file() and destination.samefile(source)
            ) for source in sources):
                raise ValueError(f"inspection destination would overwrite source: {destination}")
        output.mkdir(parents=True, exist_ok=True)
        # Prepare the entire package before replacing any existing audited files.
        with tempfile.TemporaryDirectory(prefix=".inspection-", dir=output) as directory:
            staging = Path(directory)
            write_json(staging / "schema_profile.json", profile)
            write_json(staging / "field_catalog.json", [asdict(item) for item in field_catalog])
            write_json(staging / "construction_spec.json", construction)
            self._write_report(staging / "report.md", profile, field_catalog, construction)
            publish_files([(staging / name, output / name) for name in names])
        return profile


class CSVInspector(_InspectorCommon):
    """Profiles arbitrary CSV/CSV.GZ collections without mutating them."""

    def __init__(
        self,
        *,
        llm_inspector: AdapterLLMInspector | None = None,
        sample_limit: int = 5000,
        distinct_sample_limit: int = 50,
    ) -> None:
        self._llm = llm_inspector
        self.sample_limit = max(100, sample_limit)
        self.distinct_sample_limit = max(5, distinct_sample_limit)

    def inspect(self, source: str | Path, output_dir: str | Path) -> DatabaseProfile:
        source_path = Path(source).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        files = self._discover(source_path)
        if not files:
            raise FileNotFoundError(f"no CSV files found under {source_path}")

        tables = [self._profile_file(path) for path in files]
        return self._finalize_profile(
            source_kind="csv",
            source_path=str(source_path),
            tables=tables,
            output_dir=output,
        )

    def _discover(self, source: Path) -> list[Path]:
        if source.is_file():
            return [source] if self._is_csv(source) else []
        return sorted(path for path in source.rglob("*") if path.is_file() and self._is_csv(path))

    @staticmethod
    def _is_csv(path: Path) -> bool:
        name = path.name.lower()
        return name.endswith(".csv") or name.endswith(".csv.gz")

    def _profile_file(self, path: Path) -> TableProfile:
        total = 0
        samples: dict[str, list[str]] = {}
        distinct: dict[str, set[str]] = {}
        nulls: Counter[str] = Counter()
        numeric_min: dict[str, float] = {}
        numeric_max: dict[str, float] = {}
        numeric_parsed: Counter[str] = Counter()
        datetime_parsed: Counter[str] = Counter()
        seen_non_null: Counter[str] = Counter()
        storage_types: dict[str, str] = {}

        # Stream lexical values before type inference. Codes such as 001 and NA
        # remain intact, and profiling does not require pandas or load whole tables.
        header = validate_csv_header(path)
        storage_types = {name: "VARCHAR" for name in header}
        samples = {name: [] for name in header}
        distinct = {name: set() for name in header}
        with csv_text_stream(path) as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"CSV row {reader.line_num} has {len(row)} fields. Expected {len(header)}: {path}")
                total += 1
                for name, raw in zip(header, row):
                    if raw == "":
                        nulls[name] += 1
                        continue
                    value = raw.strip()
                    if not value:
                        continue
                    seen_non_null[name] += 1
                    if len(samples[name]) < 20 and value not in samples[name]:
                        samples[name].append(value)
                    if len(distinct[name]) <= self.distinct_sample_limit:
                        distinct[name].add(value)
                    if _NUMERIC_RE.fullmatch(value):
                        number = float(value)
                        if math.isfinite(number):
                            numeric_parsed[name] += 1
                            numeric_min[name] = number if name not in numeric_min else min(numeric_min[name], number)
                            numeric_max[name] = number if name not in numeric_max else max(numeric_max[name], number)
                    if self._parse_datetime(value):
                        datetime_parsed[name] += 1

        columns: list[ColumnProfile] = []
        for name, storage_type in storage_types.items():
            non_null_count = max(0, total - nulls[name])
            denominator = max(1, seen_non_null[name])
            numeric_rate = numeric_parsed[name] / denominator
            datetime_rate = datetime_parsed[name] / denominator
            id_code = identifier_or_code_hint(name, samples[name])
            columns.append(ColumnProfile(
                name=name,
                storage_type=storage_type,
                null_fraction=(nulls[name] / total if total else 0.0),
                non_null_count=non_null_count,
                unique_count=(len(distinct[name]) if len(distinct[name]) <= self.distinct_sample_limit else None),
                sample_values=samples[name],
                min_value=numeric_min.get(name),
                max_value=numeric_max.get(name),
                semantic_description=default_semantic_description(path.name, name),
                is_numeric=(numeric_rate >= 0.98 and not id_code),
                is_datetime=(datetime_rate >= 0.98 and not id_code),
                is_identifier_or_code=id_code,
                numeric_parse_rate=numeric_rate,
                datetime_parse_rate=datetime_rate,
            ))
        return TableProfile(source_table_name(path), str(path), total, columns)

    @staticmethod
    def _parse_datetime(value: str) -> bool:
        if not any(mark in value for mark in ("-", "/", "T", ":")):
            return False
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            pass
        for pattern in _DATE_PATTERNS:
            try:
                datetime.strptime(value, pattern)
                return True
            except ValueError:
                continue
        return False

    def _apply_llm_semantics(self, tables: list[TableProfile]) -> None:
        if self._llm is None:
            return
        fields: list[dict[str, Any]] = []
        lookup: dict[str, ColumnProfile] = {}
        for table in tables:
            for column in table.columns:
                field_id = f"{table.name}.{column.name}"
                lookup[field_id] = column
                fields.append({
                    "field_id": field_id,
                    "table": table.name,
                    "column": column.name,
                    "storage_type": column.storage_type,
                    "identifier_or_code_hint": column.is_identifier_or_code,
                })
        # Keep prompts bounded while still allowing arbitrary schemas.
        for offset in range(0, len(fields), 60):
            result = self._llm.describe_fields(fields[offset:offset + 60])
            for field_id, item in result.items():
                column = lookup.get(field_id)
                if column is None:
                    continue
                description = str(item.get("description", "")).strip()
                if description:
                    column.semantic_description = description
                if not column.is_identifier_or_code:
                    column.is_numeric = bool(item.get("numeric", column.is_numeric))
                    column.is_datetime = bool(item.get("datetime", column.is_datetime))
                    if column.is_numeric and column.is_datetime:
                        # Do not allow an impossible dual type to propagate.
                        column.is_datetime = column.datetime_parse_rate >= column.numeric_parse_rate
                        column.is_numeric = not column.is_datetime

    def _infer_relationships(self, tables: list[TableProfile]) -> list[RelationshipProfile]:
        identifiers = [(table.name, column) for table in tables for column in table.columns
                       if column.is_identifier_or_code]
        result = []
        for index, (left_table, left) in enumerate(identifiers):
            for right_table, right in identifiers[index + 1:]:
                if left_table == right_table:
                    continue
                left_values, right_values = set(left.sample_values), set(right.sample_values)
                overlap = len(left_values & right_values)
                sampled_match = overlap >= 2 and overlap / max(1, min(len(left_values), len(right_values))) >= .8
                if left.name.casefold() != right.name.casefold() and not sampled_match:
                    continue
                # Both names and sampled overlap nominate candidates only. The
                # unified adapter measures full-data overlap/uniqueness. Patient
                # execution still requires an explicit or validated anchor mapping.
                result.append(RelationshipProfile(
                    left_table=left_table, left_column=left.name,
                    right_table=right_table, right_column=right.name,
                    overlap_fraction=0.0, confidence=0.0,
                ))
        return result

    def _field_catalog(self, profile: DatabaseProfile) -> list[FieldCatalogEntry]:
        fields: list[FieldCatalogEntry] = []
        for table in profile.tables:
            unit_map = self._detect_unit_pairs(table)
            for column in table.columns:
                fields.append(FieldCatalogEntry(
                    semantic_field=f"{table.name}.{column.name}",
                    description=column.semantic_description,
                    table=table.name,
                    column=column.name,
                    is_numeric=column.is_numeric,
                    is_datetime=column.is_datetime,
                    is_identifier_or_code=column.is_identifier_or_code,
                    unit_column=unit_map.get(column.name),
                    included_in_prompt=not any(token in column.storage_type.upper() for token in ("[", "STRUCT", "MAP(", "UNION", "BLOB", "SQLITE_MIXED")),
                ))
        return fields

    def _construction_spec(self, profile: DatabaseProfile) -> ConstructionSpec:
        derived: list[DerivedFieldSpec] = []
        warnings = []
        for table in profile.tables:
            unit_map = self._detect_unit_pairs(table)
            by_name = {column.name: column for column in table.columns}
            reserved = {column.name.casefold() for column in table.columns}

            def target_name(source: str) -> str:
                candidate = f"{source}__norm"
                index = 2
                while candidate.casefold() in reserved:
                    candidate = f"{source}__norm_{index}"
                    index += 1
                reserved.add(candidate.casefold())
                return candidate

            for column in table.columns:
                if column.is_identifier_or_code:
                    continue
                if column.is_numeric:
                    quoted = '"' + column.name.replace('"', '""') + '"'
                    if column.storage_type == "SQLITE_MIXED":
                        quoted = f"CAST({quoted} AS VARCHAR)"
                    # Preserve native numbers. For text use exact integer/decimal storage.
                    # overflow remains NULL and is reported, never rounded to float.
                    numeric_target = "DECIMAL(38, 12)"
                    if column.sample_values and all(re.fullmatch(r"[+-]?\d+", x.strip()) for x in column.sample_values):
                        numeric_target = "HUGEINT"
                    expression = quoted if numeric_storage_type(column.storage_type) else f'TRY_CAST({quoted} AS {numeric_target})'
                    if not numeric_storage_type(column.storage_type):
                        expression = f"CASE WHEN adapter_numeric_exact(CAST({quoted} AS VARCHAR), CAST(({expression}) AS VARCHAR)) THEN {expression} ELSE NULL END"
                    if self._llm is not None and not numeric_storage_type(column.storage_type) and (column.numeric_parse_rate or 0) < 0.98:
                        try:
                            expression = self._llm.propose_conversion(
                                column=column.name,
                                samples=column.sample_values,
                                target_type=numeric_target,
                            )
                        except Exception as exc:
                            warnings.append({"code": "conversion_proposal_failed", "table": table.name, "column": column.name,
                                "message": f"Deterministic fallback used: {type(exc).__name__}"})
                            logging.getLogger(__name__).warning("Conversion proposal unavailable for %s.%s. Using deterministic fallback: %s", table.name, column.name, exc)
                    unit_source = unit_map.get(column.name)
                    observed_units = by_name[unit_source].sample_values if unit_source and unit_source in by_name else []
                    derived.append(DerivedFieldSpec(
                        table=table.name,
                        source_column=column.name,
                        target_column=target_name(column.name),
                        kind="numeric",
                        sql_expression=expression,
                        parse_coverage=column.numeric_parse_rate,
                        unit_source_column=unit_source,
                        unit_target_column=(target_name(unit_source) if unit_source else None),
                        observed_units=observed_units,
                    ))
                elif column.is_datetime:
                    quoted = '"' + column.name.replace('"', '""') + '"'
                    if column.storage_type == "SQLITE_MIXED":
                        quoted = f"CAST({quoted} AS VARCHAR)"
                    expression = (
                        f"COALESCE(TRY_CAST({quoted} AS TIMESTAMP), "
                        f"TRY_STRPTIME(CAST({quoted} AS VARCHAR), "
                        "['%m/%d/%Y', '%m/%d/%Y %H:%M:%S']))"
                    )
                    if self._llm is not None:
                        try:
                            expression = self._llm.propose_conversion(
                                column=column.name,
                                samples=column.sample_values,
                                target_type="TIMESTAMP",
                            )
                        except Exception as exc:
                            warnings.append({"code": "conversion_proposal_failed", "table": table.name, "column": column.name,
                                "message": f"Deterministic fallback used: {type(exc).__name__}"})
                            logging.getLogger(__name__).warning("Conversion proposal unavailable for %s.%s. Using deterministic fallback: %s", table.name, column.name, exc)
                    derived.append(DerivedFieldSpec(
                        table=table.name,
                        source_column=column.name,
                        target_column=target_name(column.name),
                        kind="datetime",
                        sql_expression=expression,
                        parse_coverage=column.datetime_parse_rate,
                    ))
        return ConstructionSpec("csv", profile.source_path, derived, warnings=warnings)

    @staticmethod
    def _detect_unit_pairs(table: TableProfile) -> dict[str, str]:
        names = {column.name.lower(): column.name for column in table.columns}
        pairs: dict[str, str] = {}
        explicit = (
            ("valuenum", "valueuom"),
            ("dose_val_rx", "dose_unit_rx"),
            ("form_val_disp", "form_unit_disp"),
        )
        for value_name, unit_name in explicit:
            if value_name in names and unit_name in names:
                pairs[names[value_name]] = names[unit_name]
        for column in table.columns:
            lower = column.name.lower()
            if column.name in pairs:
                continue
            direct_candidates = (
                f"{lower}_unit",
                f"{lower}_uom",
                f"{lower}unit",
                f"{lower}uom",
            )
            for candidate in direct_candidates:
                if candidate in names:
                    pairs[column.name] = names[candidate]
                    break
            if column.name in pairs:
                continue
            for suffix in ("_value", "_val", "value", "val"):
                if not lower.endswith(suffix):
                    continue
                stem = lower[: -len(suffix)]
                for unit_suffix in ("_unit", "_uom", "unit", "uom"):
                    candidate = f"{stem}{unit_suffix}"
                    if candidate in names:
                        pairs[column.name] = names[candidate]
                        break
        return pairs

    def _write_report(
        self,
        path: Path,
        profile: DatabaseProfile,
        fields: list[FieldCatalogEntry],
        construction: ConstructionSpec,
    ) -> None:
        lines = [
            "# Dataset Adapter Inspection Report",
            "",
            f"Source: `{profile.source_path}`",
            f"Tables: {len(profile.tables)}",
            "",
            "This report describes inferred schema semantics only. Raw source data is not modified.",
            "Numeric/date normalization instructions are proposals recorded in `construction_spec.json` and must be validated before production use.",
            "",
            "## Tables",
            "",
        ]
        derived_by_key = {(item.table, item.source_column): item for item in construction.derived_fields}
        for table in profile.tables:
            lines += [f"### {table.name}", "", f"Rows: {table.row_count:,}", "", "| Field | Description | Storage | Numeric | Datetime | Identifier/code | Unit field | Normalized field |", "|---|---|---|---:|---:|---:|---|---|"]
            for column in table.columns:
                field = next(item for item in fields if item.table == table.name and item.column == column.name)
                derived = derived_by_key.get((table.name, column.name))
                lines.append(
                    f"| `{column.name}` | {column.semantic_description} | `{column.storage_type}` | "
                    f"{'yes' if column.is_numeric else 'no'} | {'yes' if column.is_datetime else 'no'} | "
                    f"{'yes' if column.is_identifier_or_code else 'no'} | "
                    f"{f'`{field.unit_column}`' if field.unit_column else ''} | "
                    f"{f'`{derived.target_column}`' if derived else ''} |"
                )
            lines.append("")
        lines += ["## Inferred relationships", ""]
        if not profile.relationships:
            lines.append("No likely relationships were inferred.")
        else:
            lines += ["| Left | Right | Confidence |", "|---|---|---:|"]
            for relation in profile.relationships:
                lines.append(
                    f"| `{relation.left_table}.{relation.left_column}` | `{relation.right_table}.{relation.right_column}` | {relation.confidence:.2f} |"
                )
        lines += ["", "## Construction summary", "", f"Derived normalized fields: {len(construction.derived_fields)}", ""]
        for item in construction.derived_fields:
            lines.append(f"- `{item.table}.{item.source_column}` → `{item.target_column}` ({item.kind}). SQL: `{item.sql_expression}`")
        path.write_text("\n".join(lines) + "\n")


class ParquetInspector(CSVInspector):
    """Profiles Parquet files through DuckDB without requiring PyArrow."""

    def inspect(self, source: str | Path, output_dir: str | Path) -> DatabaseProfile:
        source_path = Path(source).expanduser().resolve()
        files = self._discover(source_path)
        if not files:
            raise FileNotFoundError(f"no Parquet files found under {source_path}")
        tables = [self._profile_parquet(path) for path in files]
        return self._finalize_profile(
            source_kind="parquet",
            source_path=str(source_path),
            tables=tables,
            output_dir=output_dir,
        )

    @staticmethod
    def _discover(source: Path) -> list[Path]:
        if source.is_file():
            return [source] if source.suffix.lower() == ".parquet" else []
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".parquet")

    def _profile_parquet(self, path: Path) -> TableProfile:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("duckdb is required to inspect Parquet datasets") from exc
        con = duckdb.connect()
        try:
            escaped = str(path).replace("'", "''")
            relation = f"read_parquet('{escaped}')"
            row_count = int(con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])
            schema = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
            columns: list[ColumnProfile] = []
            for row in schema:
                name, storage_type = str(row[0]), str(row[1])
                qname = '"' + name.replace('"', '""') + '"'
                non_null_count = int(con.execute(
                    f"SELECT COUNT({qname}) FROM {relation}"
                ).fetchone()[0])
                values = [r[0] for r in con.execute(
                    f"SELECT {qname} FROM {relation} WHERE {qname} IS NOT NULL LIMIT {self.sample_limit}"
                ).fetchall()]
                columns.append(self._profile_values(
                    path.stem, name, storage_type, values, row_count, non_null_count=non_null_count
                ))
            return TableProfile(path.stem, str(path), row_count, columns)
        finally:
            con.close()

    def _profile_values(
        self,
        table: str,
        name: str,
        storage_type: str,
        values: list[Any],
        row_count: int,
        *,
        non_null_count: int | None = None,
    ) -> ColumnProfile:
        samples: list[str] = []
        distinct: set[str] = set()
        numeric_count = 0
        datetime_count = 0
        numeric_values: list[float] = []
        for raw in values:
            value = str(raw).strip()
            if not value:
                continue
            if len(samples) < 20 and value not in samples:
                samples.append(value)
            if len(distinct) <= self.distinct_sample_limit:
                distinct.add(value)
            if _NUMERIC_RE.fullmatch(value):
                try:
                    number = float(value)
                    if math.isfinite(number):
                        numeric_count += 1
                        numeric_values.append(number)
                except ValueError:
                    pass
            if CSVInspector._parse_datetime(value):
                datetime_count += 1
        denominator = max(1, len(values))
        actual_non_null = len(values) if non_null_count is None else non_null_count
        id_code = identifier_or_code_hint(name, samples)
        numeric_type = numeric_storage_type(storage_type)
        datetime_type = storage_type.upper() in {"DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S"}
        return ColumnProfile(
            name=name,
            storage_type=storage_type,
            null_fraction=(1.0 - actual_non_null / row_count if row_count else 0.0),
            non_null_count=actual_non_null,
            unique_count=(len(distinct) if actual_non_null == len(values) and len(distinct) <= self.distinct_sample_limit else None),
            sample_values=samples,
            min_value=min(numeric_values) if numeric_values else None,
            max_value=max(numeric_values) if numeric_values else None,
            semantic_description=default_semantic_description(table, name),
            is_numeric=((numeric_type or numeric_count / denominator >= 0.98) and not id_code),
            is_datetime=((datetime_type or datetime_count / denominator >= 0.98) and not id_code),
            is_identifier_or_code=id_code,
            numeric_parse_rate=numeric_count / denominator,
            datetime_parse_rate=datetime_count / denominator,
        )


class SQLInspector(CSVInspector):
    """Profiles local DuckDB or SQLite databases."""

    def inspect(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        dialect: str = "auto",
    ) -> DatabaseProfile:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        resolved = self._resolve_dialect(source_path, dialect)
        if resolved == "duckdb":
            tables = self._inspect_duckdb(source_path)
        elif resolved == "sqlite":
            tables = self._inspect_sqlite(source_path)
        else:
            raise ValueError(f"unsupported SQL dialect: {resolved}")
        if not tables:
            raise ValueError(f"no user tables found in {source_path}")
        return self._finalize_profile(
            source_kind=f"sql:{resolved}",
            source_path=str(source_path),
            tables=tables,
            output_dir=output_dir,
        )

    @staticmethod
    def _resolve_dialect(path: Path, dialect: str) -> str:
        value = dialect.lower().strip()
        if value != "auto":
            return value
        if path.suffix.lower() in {".duckdb", ".ddb"}:
            return "duckdb"
        if path.suffix.lower() in {".sqlite", ".sqlite3"}:
            return "sqlite"
        # .db is ambiguous. Prefer SQLite because its header is standardized.
        try:
            with path.open("rb") as stream:
                header = stream.read(16)
        except OSError:
            header = b""
        if header.startswith(b"SQLite format 3"):
            return "sqlite"
        return "duckdb"

    def _inspect_duckdb(self, path: Path) -> list[TableProfile]:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("duckdb is required to inspect DuckDB databases") from exc
        con = duckdb.connect(str(path), read_only=True)
        try:
            rows = con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type='BASE TABLE' AND table_schema NOT IN ('information_schema','pg_catalog') "
                "ORDER BY table_schema, table_name"
            ).fetchall()
            return [self._profile_duckdb_table(con, schema, table, str(path)) for schema, table in rows]
        finally:
            con.close()

    def _profile_duckdb_table(self, con: Any, schema: str, table: str, source_path: str) -> TableProfile:
        qschema = self._quote(schema)
        qtable = self._quote(table)
        qualified = f"{qschema}.{qtable}"
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()[0])
        info = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema=? AND table_name=? ORDER BY ordinal_position",
            [schema, table],
        ).fetchall()
        columns = []
        for name, storage_type in info:
            qcol = self._quote(str(name))
            non_null_count = int(con.execute(f"SELECT COUNT({qcol}) FROM {qualified}").fetchone()[0])
            values = [row[0] for row in con.execute(
                f"SELECT {qcol} FROM {qualified} WHERE {qcol} IS NOT NULL LIMIT {int(self.sample_limit)}"
            ).fetchall()]
            columns.append(ParquetInspector._profile_values(
                self, table, str(name), str(storage_type), values, row_count, non_null_count=non_null_count
            ))
        logical_name = table if schema in {"main", "public"} else f"{schema}.{table}"
        return TableProfile(logical_name, source_path, row_count, columns)

    def _inspect_sqlite(self, path: Path) -> list[TableProfile]:
        import sqlite3
        con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            tables = [row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()]
            result = []
            for table in tables:
                qtable = self._quote(table)
                row_count = int(con.execute(f"SELECT COUNT(*) FROM {qtable}").fetchone()[0])
                info = con.execute(f"PRAGMA table_info({qtable})").fetchall()
                columns = []
                for row in info:
                    name = str(row[1])
                    qcol = self._quote(name)
                    kinds = {r[0] for r in con.execute(f"SELECT DISTINCT typeof({qcol}) FROM {qtable} WHERE {qcol} IS NOT NULL")}
                    storage_type = ("SQLITE_MIXED" if len(kinds) > 1 else
                        {"integer": "BIGINT", "real": "DOUBLE", "blob": "BLOB", "text": "VARCHAR"}.get(next(iter(kinds), "text")))
                    non_null_count = int(con.execute(
                        f"SELECT COUNT({qcol}) FROM {qtable}"
                    ).fetchone()[0])
                    values = [r[0] for r in con.execute(
                        f"SELECT {qcol} FROM {qtable} WHERE {qcol} IS NOT NULL LIMIT ?",
                        (int(self.sample_limit),),
                    ).fetchall()]
                    columns.append(ParquetInspector._profile_values(
                        self, table, name, storage_type, values, row_count, non_null_count=non_null_count
                    ))
                result.append(TableProfile(table, str(path), row_count, columns))
            return result
        finally:
            con.close()

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'


class DatasetInspector:
    """Dispatches inspection based on source format while producing one adapter schema."""

    def __init__(
        self,
        *,
        llm_inspector: AdapterLLMInspector | None = None,
        sample_limit: int = 5000,
        distinct_sample_limit: int = 50,
    ) -> None:
        kwargs = dict(
            llm_inspector=llm_inspector,
            sample_limit=sample_limit,
            distinct_sample_limit=distinct_sample_limit,
        )
        self.csv = CSVInspector(**kwargs)
        self.parquet = ParquetInspector(**kwargs)
        self.sql = SQLInspector(**kwargs)

    def inspect(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        source_format: str = "auto",
        sql_dialect: str = "auto",
    ) -> DatabaseProfile:
        path = Path(source).expanduser().resolve()
        fmt = source_format.lower().strip()
        if fmt == "auto":
            fmt = self._detect_format(path)
        if fmt == "csv":
            return self.csv.inspect(path, output_dir)
        if fmt == "parquet":
            return self.parquet.inspect(path, output_dir)
        if fmt in {"sql", "duckdb", "sqlite"}:
            dialect = sql_dialect if fmt == "sql" else fmt
            return self.sql.inspect(path, output_dir, dialect=dialect)
        raise ValueError(f"unsupported source format: {fmt}")

    @staticmethod
    def _detect_format(path: Path) -> str:
        if path.is_file():
            name = path.name.lower()
            if name.endswith((".csv", ".csv.gz")):
                return "csv"
            if name.endswith(".parquet"):
                return "parquet"
            if path.suffix.lower() in {".duckdb", ".ddb", ".sqlite", ".sqlite3", ".db"}:
                return "sql"
        elif path.is_dir():
            formats = set()
            for item in path.rglob("*"):
                if item.is_file() and item.suffix.lower() == ".parquet":
                    formats.add("parquet")
                if item.is_file() and item.name.lower().endswith((".csv", ".csv.gz")):
                    formats.add("csv")
            if len(formats) > 1:
                raise ValueError("mixed source formats. Pass --format explicitly")
            if formats:
                return formats.pop()
        raise ValueError(f"could not infer source format for {path}. Pass --format explicitly")
