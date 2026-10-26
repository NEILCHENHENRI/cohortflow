#!/usr/bin/env python3
"""Backward-compatible wrapper around the universal CSV adapter inspector.

Prefer::

    python scripts/inspect_dataset.py <csv-dir> --output data/adapters/mimic_hosp

This wrapper remains so existing commands do not break. It generates the full
adapter artifact set and copies the human-readable report to ``-o`` when that
legacy option is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from agents.cohort.adapters.inspector import CSVInspector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir")
    parser.add_argument("-o", "--output", default="mimic_content_report.md")
    parser.add_argument("--sample-rows", type=int, default=5, help="Accepted for compatibility. The universal inspector profiles bounded samples internally.")
    parser.add_argument("--chunksize", type=int, default=250000, help="Accepted for compatibility. The universal inspector manages its own chunk size.")
    args = parser.parse_args()

    legacy_report = Path(args.output).expanduser().resolve()
    adapter_dir = legacy_report.parent / "data" / "adapters" / "mimic_hosp"
    CSVInspector().inspect(args.csv_dir, adapter_dir)
    generated = adapter_dir / "report.md"
    if generated != legacy_report:
        legacy_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, legacy_report)
    print(f"Adapter artifacts: {adapter_dir}")
    print(f"Report: {legacy_report}")


if __name__ == "__main__":
    main()
