"""Compute aggregates and build JSON snapshot for the web UI."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sailgp_analysis.config import DATA_ROOT, SNAPSHOT_FILE, WEB_DATA_DIR
from sailgp_analysis.data_loader import load_all_boats, load_metadata, load_marks


def add_foiling(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["foiling"] = (
        (out["LENGTH_RH_P_mm"] > 100)
        & (out["LENGTH_RH_S_mm"] > 100)
        & (out["LENGTH_RH_BOW_mm"] > 100)
        & (out["BOAT_SPEED_km_h_1"] > 40)
    )
    return out


def race_speed_series(df: pd.DataFrame, venue: str, race_label: str, team: str, max_points: int = 120) -> list[dict]:
    sub = df[(df["venue"] == venue) & (df["race_label"] == race_label) & (df["team"] == team)]
    sub = sub[sub["TRK_BOAT_RACE_STATUS_unk"] == 2].dropna(subset=["BOAT_SPEED_km_h_1"])
    if sub.empty:
        return []
    if len(sub) > max_points:
        sub = sub.iloc[:: max(1, len(sub) // max_points)]
    return [
        {"t": ts.isoformat(), "speed": float(row.BOAT_SPEED_km_h_1), "rank": int(row.TRK_RACE_RANK_unk) if pd.notna(row.TRK_RACE_RANK_unk) else None}
        for ts, row in sub.iterrows()
    ]


def build_snapshot(data_root: Path = DATA_ROOT) -> dict:
    meta = load_metadata(data_root)
    df = load_all_boats(data_root)
    marks = load_marks(data_root)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "venues": {},
        "status_counts": {},
        "team_colors": {},
        "marks_summary": {},
        "eda_artifacts": [],
    }

    if df.empty:
        return snapshot

    status = df["TRK_BOAT_RACE_STATUS_unk"].value_counts().to_dict()
    snapshot["status_counts"] = {str(int(k)): int(v) for k, v in status.items()}

    df_racing = add_foiling(df[df["TRK_BOAT_RACE_STATUS_unk"] == 2])

    for venue in df["venue"].unique():
        vdf = df[df["venue"] == venue]
        vr = df_racing[df_racing["venue"] == venue]
        races = []
        for race_label in sorted(vdf["race_label"].unique()):
            rdf = vdf[vdf["race_label"] == race_label]
            rmeta = meta[(meta["venue"] == venue) & (meta["race_label"] == race_label)]
            meta_row = rmeta.iloc[0].to_dict() if len(rmeta) else {}
            teams_stats = []
            for team in sorted(rdf["team"].unique()):
                t = rdf[rdf["team"] == team]
                tr = vr[(vr["race_label"] == race_label) & (vr["team"] == team)]
                teams_stats.append({
                    "team": team,
                    "rows": int(len(t)),
                    "mean_speed": float(tr["BOAT_SPEED_km_h_1"].mean()) if len(tr) else None,
                    "mean_vmg": float(tr["VMG_km_h_1"].mean()) if len(tr) and "VMG_km_h_1" in tr else None,
                    "foiling_pct": float(tr["foiling"].mean() * 100) if len(tr) else None,
                    "speed_series": race_speed_series(df, venue, race_label, team),
                })
            races.append({
                "race_label": race_label,
                "metadata": {k: (v if not isinstance(v, float) or not np.isnan(v) else None) for k, v in meta_row.items()},
                "teams": teams_stats,
            })
        snapshot["venues"][venue] = {
            "races": races,
            "teams": sorted(vdf["team"].unique().tolist()),
            "total_rows": int(len(vdf)),
        }

    for (venue, race_label), mdf in marks.items():
        wg = mdf[mdf["MARK"].astype(str).str.startswith("WG", na=False)]
        lg = mdf[mdf["MARK"].astype(str).str.startswith("LG", na=False)]
        snapshot["marks_summary"][f"{venue}/{race_label}"] = {
            "wg_mean_tws": float(wg["TWS_km_h_1"].mean()) if len(wg) else None,
            "lg_mean_tws": float(lg["TWS_km_h_1"].mean()) if len(lg) else None,
            "wg_mean_twd": float(wg["TWD_deg"].mean()) if len(wg) else None,
            "lg_mean_twd": float(lg["TWD_deg"].mean()) if len(lg) else None,
        }

    repo = data_root.parent
    for name in ["sailgp_dashboard.html", "map_bermuda.html", "map_halifax.html", "sailgp_profile.html"]:
        p = repo / name
        if p.exists():
            snapshot["eda_artifacts"].append({"name": name, "path": f"/artifacts/{name}", "size_kb": round(p.stat().st_size / 1024, 1)})

    teams = sorted(df["team"].unique())
    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    snapshot["team_colors"] = {t: palette[i % len(palette)] for i, t in enumerate(teams)}

    return snapshot


def write_snapshot(data_root: Path = DATA_ROOT, out_path: Path = SNAPSHOT_FILE) -> Path:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot(data_root)
    out_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return out_path
