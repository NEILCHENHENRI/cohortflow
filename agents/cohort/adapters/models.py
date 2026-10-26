from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import shutil
import tempfile


@dataclass(slots=True)
class ColumnProfile:
    name: str
    storage_type: str
    null_fraction: float
    non_null_count: int
    unique_count: int | None
    sample_values: list[str] = field(default_factory=list)
    min_value: Any | None = None
    max_value: Any | None = None
    semantic_description: str = ""
    is_numeric: bool = False
    is_datetime: bool = False
    is_identifier_or_code: bool = False
    numeric_parse_rate: float | None = None
    datetime_parse_rate: float | None = None


@dataclass(slots=True)
class TableProfile:
    name: str
    source_path: str
    row_count: int
    columns: list[ColumnProfile]


@dataclass(slots=True)
class RelationshipProfile:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    overlap_fraction: float
    confidence: float


@dataclass(slots=True)
class DatabaseProfile:
    source_kind: str
    source_path: str
    tables: list[TableProfile]
    relationships: list[RelationshipProfile] = field(default_factory=list)


@dataclass(slots=True)
class FieldCatalogEntry:
    semantic_field: str
    description: str
    table: str
    column: str
    is_numeric: bool = False
    is_datetime: bool = False
    is_identifier_or_code: bool = False
    unit_column: str | None = None
    included_in_prompt: bool = True


@dataclass(slots=True)
class DerivedFieldSpec:
    table: str
    source_column: str
    target_column: str
    kind: str
    sql_expression: str
    parse_coverage: float | None = None
    unit_source_column: str | None = None
    unit_target_column: str | None = None
    canonical_unit: str | None = None
    observed_units: list[str] = field(default_factory=list)
    unit_conversions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ConstructionSpec:
    source_kind: str
    source_path: str
    derived_fields: list[DerivedFieldSpec]
    output_format: str = "duckdb"
    warnings: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    semantic_field: str
    description: str
    table: str
    source_column: str
    value_column: str
    value_kind: str
    unit_column: str | None = None
    source_unit_column: str | None = None
    identifier_or_code: bool = False
    included_in_prompt: bool = True
    canonical_unit: str | None = None
    observed_units: tuple[str, ...] = ()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def publish_files(files: list[tuple[Path, Path]]) -> None:
    """Publish a prepared bundle with rollback on ordinary I/O failures.

    Each rename is atomic, but multiple files are not crash-atomic as a group.
    Staged files must live on the same filesystem as their destination.
    """
    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    retain: set[Path] = set()
    try:
        for _, destination in files:
            if destination in backups:
                raise ValueError(f"duplicate publication destination: {destination}")
            backups[destination] = None
            if destination.exists():
                fd, name = tempfile.mkstemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
                os.close(fd)
                backups[destination] = Path(name)
                shutil.copy2(destination, name)
        for staged, destination in files:
            os.replace(staged, destination)
            published.append(destination)
    except Exception as original_error:
        rollback_errors = []
        for destination in reversed(published):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except OSError as exc:
                if backup is not None:
                    retain.add(backup)
                rollback_errors.append(f"{destination}: {exc}. Backup={backup}")
        if rollback_errors:
            original_error.add_note("Publication rollback needs recovery: " + ", ".join(rollback_errors))
        raise
    finally:
        for backup in backups.values():
            if backup is not None and backup not in retain:
                backup.unlink(missing_ok=True)
