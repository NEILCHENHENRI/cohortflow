#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.cohort.adapters.inspector import DatasetInspector
from agents.cohort.adapters.llm_inspector import AdapterLLMInspector
from infrastructure.llm.ollama import OllamaClient, OllamaConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect CSV, Parquet, DuckDB, or SQLite data and generate an auditable adapter specification."
    )
    parser.add_argument("source", help="Dataset path: CSV directory/file, Parquet directory/file, DuckDB, or SQLite database")
    parser.add_argument("--output", default="data/adapters/custom", help="Adapter artifact directory")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "parquet", "sql", "duckdb", "sqlite"),
        default="auto",
        help="Source format (default: infer from path)",
    )
    parser.add_argument(
        "--sql-dialect",
        choices=("auto", "duckdb", "sqlite"),
        default="auto",
        help="SQL database dialect when --format=sql",
    )
    parser.add_argument("--llm", action="store_true", help="Use Ollama to describe fields and classify numeric/datetime semantics")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model used for schema inspection")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--build", action="store_true", help="Run the full adapter: publish dataset.duckdb, catalog, mapping and adaptation_log.json")
    parser.add_argument("--mapping", help="Reviewed cohort mapping JSON (used with --build)")
    args = parser.parse_args()
    if args.mapping and not args.build:
        parser.error("--mapping requires --build")

    llm_inspector = None
    if args.llm:
        llm_inspector = AdapterLLMInspector(OllamaClient(OllamaConfig(
            model=args.model,
            base_url=args.ollama_url,
            think=False,
            temperature=0.0,
        )))
    if args.build:
        from agents.cohort.adapters.constructor import adapt_dataset
        result = adapt_dataset(args.source, args.output, source_format=args.format,
            sql_dialect=args.sql_dialect, llm_inspector=llm_inspector,
            mapping=json.loads(Path(args.mapping).read_text()) if args.mapping else None)
        print(f"Database: {Path(args.output).resolve() / 'dataset.duckdb'}")
        print(f"Log: {Path(args.output).resolve() / 'adaptation_log.json'}")
        print(f"Cohort readiness: {result['status']}")
        return
    profile = DatasetInspector(llm_inspector=llm_inspector).inspect(
        args.source,
        args.output,
        source_format=args.format,
        sql_dialect=args.sql_dialect,
    )
    print(f"Inspected {len(profile.tables)} table(s) from {profile.source_kind}.")
    print(f"Adapter artifacts: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
