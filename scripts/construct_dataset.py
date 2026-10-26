#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agents.cohort.adapters.constructor import DatasetConstructor


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct a normalized dataset from an audited adapter construction spec.")
    parser.add_argument("spec", help="Path to construction_spec.json")
    parser.add_argument("--output", required=True, help="Output DuckDB path")
    parser.add_argument("--report", help="Optional construction report JSON path")
    args = parser.parse_args()
    result = DatasetConstructor().construct(args.spec, args.output, report_path=args.report)
    print(f"Constructed {result.output_path}")
    print(f"Tables: {', '.join(result.tables)}")
    print(f"Derived normalized fields: {result.derived_fields}")
    if result.report_path:
        print(f"Construction report: {result.report_path}")
    for item in result.field_stats:
        print(
            f"  {item.table}.{item.source_column} -> {item.target_column}: "
            f"{item.normalized_non_null}/{item.source_non_null} normalized "
            f"({item.normalized_fraction:.1%}). Unsupported-unit rows={item.unsupported_unit_rows}"
        )


if __name__ == "__main__":
    main()
