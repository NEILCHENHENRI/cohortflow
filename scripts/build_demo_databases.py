from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.demo import build_demo_databases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the synthetic databases used by demo mode."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demo"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    clinical, vocabulary = build_demo_databases(
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Created {clinical}")
    print(f"Created {vocabulary}")


if __name__ == "__main__":
    main()
