"""Interpolated course-wide wind field (IDW grid quiver from mark sensors)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "dataExploration" / "race_replay.py"
EXPORT = ROOT / "dataExploration" / "exported" / "wind_field_interp_Bermuda_Race_5.html"


def main() -> None:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--venue",
        "Bermuda",
        "--race",
        "Race_5",
        "--wind-only",
        "--interpolated",
        "--grid-size",
        "18",
        "--step",
        "2",
        "--output",
        str(EXPORT),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
