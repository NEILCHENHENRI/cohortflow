from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit app.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate or reuse the local synthetic demo databases.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional Streamlit server port.",
    )
    args, streamlit_args = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root)
        if not current_pythonpath
        else os.pathsep.join((str(root), current_pythonpath))
    )

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "interfaces" / "streamlit_app.py"),
    ]
    if args.port is not None:
        command.extend(("--server.port", str(args.port)))
    command.extend(streamlit_args)
    command.append("--")
    command.extend(("--mode", "demo" if args.demo else "local"))
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
