"""Load SailGP boat telemetry, marks, and race metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from sailgp_analysis.config import DATA_ROOT, VENUES


def iter_boat_csvs(data_root: Path = DATA_ROOT) -> Iterator[tuple[str, str, str, Path]]:
    for venue in VENUES:
        boats_dir = data_root / venue / "boats"
        if not boats_dir.exists():
            continue
        for path in sorted(boats_dir.rglob("*.csv")):
            yield venue, path.parent.name, path.stem, path


def load_all_boats(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    frames = []
    for venue, race_label, team, path in iter_boat_csvs(data_root):
        df = pd.read_csv(path, parse_dates=["DATETIME"])
        if "DATETIME" in df.columns:
            df = df.set_index("DATETIME")
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        df["venue"] = venue
        df["race_label"] = race_label
        df["team"] = team
        df["source_file"] = str(path.relative_to(data_root))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def load_metadata(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    parts = []
    for venue in VENUES:
        p = data_root / venue / "race_metadata.csv"
        if p.exists():
            parts.append(pd.read_csv(p).assign(venue=venue))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_marks(data_root: Path = DATA_ROOT) -> dict[tuple[str, str], pd.DataFrame]:
    marks: dict[tuple[str, str], pd.DataFrame] = {}
    for venue in VENUES:
        marks_dir = data_root / venue / "marks"
        if not marks_dir.exists():
            continue
        for path in sorted(marks_dir.rglob("marks.csv")):
            marks[(venue, path.parent.name)] = pd.read_csv(path, parse_dates=["DATETIME"])
    return marks


def scan_data_inventory(data_root: Path = DATA_ROOT) -> list[dict]:
    """File inventory with size and mtime for ingest agent."""
    inv = []
    for venue, race_label, team, path in iter_boat_csvs(data_root):
        st = path.stat()
        inv.append({
            "venue": venue,
            "race_label": race_label,
            "team": team,
            "path": str(path),
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
        })
    return inv
