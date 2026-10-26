from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def _load_duckdb() -> Any:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DuckDB infrastructure requires the optional 'duckdb' package. "
            "Install project requirements with: pip install -r requirements.txt"
        ) from exc
    return duckdb


@dataclass(frozen=True, slots=True)
class DuckDBConfig:
    database_path: Path
    read_only: bool = True
    threads: int | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.read_only and not self.database_path.exists():
            raise FileNotFoundError(self.database_path)
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive")


class DuckDBConnectionFactory:
    """Creates an independent DuckDB connection for each operation.

    DuckDB recommends separate connections for parallel Python work rather
    than sharing the module-level/global connection.
    """

    def __init__(self, config: DuckDBConfig) -> None:
        self.config = config

    @contextmanager
    def connect(self) -> Iterator[Any]:
        duckdb = _load_duckdb()

        config = dict(self.config.extra_config)
        if self.config.threads is not None:
            config["threads"] = self.config.threads

        if config:
            connection = duckdb.connect(
                str(self.config.database_path),
                read_only=self.config.read_only,
                config=config,
            )
        else:
            connection = duckdb.connect(
                str(self.config.database_path),
                read_only=self.config.read_only,
            )

        try:
            yield connection
        finally:
            connection.close()

    def execute(
        self,
        sql: str,
        parameters: list[Any] | tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]:
        with self.connect() as connection:
            return connection.execute(sql, parameters).fetchall()

    def table_exists(self, table_name: str) -> bool:
        rows = self.execute(
            '''
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE lower(table_name) = lower(?)
            ''',
            [table_name],
        )
        return bool(rows and rows[0][0])

    def qualified_table_name(self, table_name: str) -> str:
        """Resolve a bare table name to a schema-qualified, quoted identifier.

        DuckDB only resolves unqualified table names against its current
        search_path (typically just ``main``). A database that organizes
        tables into named schemas (e.g. a ``vocab`` schema for dictionary
        tables) will silently fail on plain ``FROM some_table`` even though
        :meth:`table_exists` reports the table as present, since that check
        searches ``information_schema.tables`` across every schema. Adapters
        should resolve each table they query through this method once (for
        example in ``__init__``) and use the returned identifier in their SQL,
        rather than hardcoding an unqualified or assumed-schema name.
        """
        rows = self.execute(
            '''
            SELECT table_schema
            FROM information_schema.tables
            WHERE lower(table_name) = lower(?)
            ''',
            [table_name],
        )
        if not rows:
            raise ValueError(f"table not found in any schema: {table_name}")
        schemas = sorted({str(row[0]) for row in rows})
        if len(schemas) > 1:
            raise ValueError(
                f"table name {table_name!r} exists in multiple schemas "
                f"({', '.join(schemas)}). Qualify it explicitly"
            )
        return f'"{schemas[0]}"."{table_name}"'
