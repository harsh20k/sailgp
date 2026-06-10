"""Experiment 1b — Flight quality extensions: ghost overlay, splashdown gaps, manoeuvre recovery."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_LEG,
    COL_RH_BOW,
    COL_RH_P,
    COL_RH_S,
    COL_STATUS,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.fleet import COL_HEADING, COL_LAT, COL_LON
from dataExploration.next_experiments import exp_1_flight_quality as exp1
from dataExploration.next_experiments.exp_4_ghost_boat import (
    COL_TIME_LEG,
    build_polar_table,
    extract_legs,
    infer_mark_positions,
    simulate_ghost_leg,
    spatial_regret_track,
)
from sailgp_analysis.config import DATA_ROOT

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
HEADING_COL = "HEADING_deg"
COG_COL = "GPS_COG_deg"
MANOEUVRE_DEG = 60.0
MANOEUVRE_WINDOW_S = 10
RECOVERY_THRESHOLD = 0.7
GHOST_RHO_THRESHOLD = -0.3
RECOVERY_CV_THRESHOLD = 0.15


def _heading_col(df: pd.DataFrame) -> str:
    if HEADING_COL in df.columns and df[HEADING_COL].notna().sum() > 0:
        return HEADING_COL
    return COG_COL


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def splashdown_mask(df: pd.DataFrame) -> pd.Series:
    """All three ride heights below 100 mm simultaneously."""
    cols = [COL_RH_P, COL_RH_S, COL_RH_BOW]
    if not all(c in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    return (df[COL_RH_P] < 100) & (df[COL_RH_S] < 100) & (df[COL_RH_BOW] < 100)


def leader_mid_gap_with_splashdowns(
    scores: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extend leader–mid-fleet gap with splashdown (crash) seconds per team."""
    gaps = exp1.leader_mid_gap(scores)
    if gaps.empty:
        return gaps

    racing = source_df[source_df[COL_STATUS] == 2].copy()
    racing = racing.reset_index().rename(columns={"index": "timestamp"})
    racing["splashdown"] = splashdown_mask(racing)

    crash_stats = []
    for (venue, race, team), gdf in racing.groupby(["venue", "race_label", "team"], sort=False):
        n = len(gdf)
        crash_stats.append(
            {
                "venue": venue,
                "race_label": race,
                "team": team,
                "splashdown_seconds": int(gdf["splashdown"].sum()),
                "splashdown_frac": float(gdf["splashdown"].mean()) if n else 0.0,
            }
        )
    crash_df = pd.DataFrame(crash_stats)

    rows = []
    for row in gaps.itertuples(index=False):
        leader = crash_df[
            (crash_df["venue"] == row.venue)
            & (crash_df["race_label"] == row.race_label)
            & (crash_df["team"] == row.leader_team)
        ]
        mid = crash_df[
            (crash_df["venue"] == row.venue)
            & (crash_df["race_label"] == row.race_label)
            & (crash_df["team"].isin(
                scores[
                    (scores["venue"] == row.venue)
                    & (scores["race_label"] == row.race_label)
                ]
                .sort_values("timestamp")
                .groupby("team")["rank"]
                .last()
                .loc[lambda s: (s >= 4) & (s <= 6)]
                .index
            ))
        ]
        rows.append(
            {
                **row._asdict(),
                "leader_splashdown_s": int(leader["splashdown_seconds"].iloc[0]) if len(leader) else 0,
                "mid_fleet_splashdown_s": float(mid["splashdown_seconds"].mean()) if len(mid) else 0.0,
                "mid_fleet_splashdown_frac": float(mid["splashdown_frac"].mean()) if len(mid) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_regret_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Per-second cumulative regret and regret rate with timestamps."""
    polar, global_speed = build_polar_table(df)
    simulate_ghost_leg.polar = (polar, global_speed)  # type: ignore[attr-defined]
    marks = infer_mark_positions(df)
    leg_results = extract_legs(df, marks)
    if leg_results.empty:
        return pd.DataFrame()

    frames = []
    for row in leg_results.itertuples(index=False):
        sub = df[
            (df["venue"] == row.venue)
            & (df["race_label"] == row.race_label)
            & (df["team"] == row.team)
            & (df[COL_LEG] == row.leg)
        ].sort_index()
        if sub.empty:
            continue
        track = spatial_regret_track(sub, row.ghost_track, row.mark_lat, row.mark_lon)
        if track.empty:
            continue
        track["timestamp"] = sub.index[: len(track)]
        track["venue"] = row.venue
        track["race_label"] = row.race_label
        track["team"] = row.team
        track["leg"] = row.leg
        track["regret_rate_s"] = track["cum_regret_s"].diff().fillna(0.0)
        frames.append(track)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ghost_flight_overlay(
    scores: pd.DataFrame,
    regret_ts: pd.DataFrame,
) -> tuple[pd.DataFrame, float, Path]:
    """Join flight quality with regret rate; export scatter HTML."""
    if regret_ts.empty or scores.empty:
        path = EXPORT_DIR / "flight_quality_ghost_scatter.html"
        px.scatter(title="No ghost overlay data").write_html(path, include_plotlyjs="cdn")
        return pd.DataFrame(), 0.0, path

    fq = scores[
        ["timestamp", "venue", "race_label", "team", "flight_quality"]
    ].copy()
    fq["timestamp"] = pd.to_datetime(fq["timestamp"])

    reg = regret_ts[
        ["timestamp", "venue", "race_label", "team", "cum_regret_s", "regret_rate_s"]
    ].copy()
    reg["timestamp"] = pd.to_datetime(reg["timestamp"])

    merged = fq.merge(reg, on=["timestamp", "venue", "race_label", "team"], how="inner")
    merged = merged.dropna(subset=["flight_quality", "regret_rate_s"])
    if merged.empty:
        rho = 0.0
    else:
        rho, _ = stats.spearmanr(merged["flight_quality"], merged["regret_rate_s"])

    fig = px.scatter(
        merged.sample(min(8000, len(merged)), random_state=42) if len(merged) > 8000 else merged,
        x="flight_quality",
        y="regret_rate_s",
        color="team",
        opacity=0.35,
        title=f"Flight Quality vs Ghost Regret Rate (Spearman ρ={rho:.3f})",
        labels={"flight_quality": "Flight Quality", "regret_rate_s": "Regret rate (s/s)"},
        height=600,
        width=900,
    )
    fig.update_layout(template="plotly_white")
    path = EXPORT_DIR / "flight_quality_ghost_scatter.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return merged, float(rho), path


def detect_manoeuvres(grp: pd.DataFrame, hcol: str) -> list[pd.Timestamp]:
    """Tack/gybe: course change > 60° within 10 s."""
    grp = grp.sort_index()
    if hcol not in grp.columns or len(grp) < MANOEUVRE_WINDOW_S + 1:
        return []
    headings = grp[hcol].ffill().to_numpy()
    times = grp.index
    events = []
    for i in range(MANOEUVRE_WINDOW_S, len(grp)):
        delta = abs(_angle_diff_deg(float(headings[i]), float(headings[i - MANOEUVRE_WINDOW_S])))
        if delta >= MANOEUVRE_DEG:
            events.append(times[i])
    return events


def manoeuvre_recovery(
    scores: pd.DataFrame,
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Recovery time to flight_quality > 0.7 after each tack/gybe."""
    hcol = _heading_col(source_df)
    fq = scores.copy()
    fq["timestamp"] = pd.to_datetime(fq["timestamp"])
    fq = fq.set_index("timestamp")

    rows = []
    racing = source_df[source_df[COL_STATUS] == 2]
    for (venue, race, team), boat_df in racing.groupby(["venue", "race_label", "team"], sort=False):
        events = detect_manoeuvres(boat_df, hcol)
        team_fq = fq[
            (fq["venue"] == venue) & (fq["race_label"] == race) & (fq["team"] == team)
        ]["flight_quality"]
        if team_fq.empty:
            continue
        for evt in events:
            evt = pd.Timestamp(evt)
            post = team_fq[team_fq.index > evt]
            recovered = post[post > RECOVERY_THRESHOLD]
            recovery_s = float((recovered.index[0] - evt).total_seconds()) if len(recovered) else np.nan
            fq_at = float(team_fq.asof(evt)) if evt in team_fq.index or team_fq.index.min() <= evt <= team_fq.index.max() else np.nan
            rows.append(
                {
                    "venue": venue,
                    "race_label": race,
                    "team": team,
                    "manoeuvre_time": evt,
                    "fq_at_manoeuvre": fq_at,
                    "recovery_s": recovery_s,
                    "recovered": not np.isnan(recovery_s),
                }
            )

    out = pd.DataFrame(rows)
    team_means = out.dropna(subset=["recovery_s"]).groupby("team")["recovery_s"].mean()
    cv = float(team_means.std() / team_means.mean()) if len(team_means) > 1 and team_means.mean() > 0 else 0.0
    return out, cv


def run(export_core: bool = True) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    core_payload = exp1.run() if export_core else {}
    scores = pd.read_csv(EXPORT_DIR / "flight_quality.csv", parse_dates=["timestamp"])
    source_df = load_racing_boats(DATA_ROOT)
    bermuda = source_df[
        (source_df["venue"] == "Bermuda")
        & (source_df["race_label"].isin(exp1.BERMUDA_RACES))
    ]

    gaps_ext = leader_mid_gap_with_splashdowns(scores, bermuda)
    regret_ts = build_regret_timeseries(bermuda)
    overlay_df, ghost_rho, scatter_path = ghost_flight_overlay(scores, regret_ts)
    manoeuvre_df, recovery_cv = manoeuvre_recovery(scores, bermuda)

    manoeuvre_path = EXPORT_DIR / "flight_quality_manoeuvre_recovery.csv"
    manoeuvre_df.to_csv(manoeuvre_path, index=False)

    # Update core JSON with extension metrics
    json_path = EXPORT_DIR / "flight_quality_results.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text())
    else:
        payload = core_payload

    ext_criteria = {
        "ghost_overlay_spearman": {
            "passed": ghost_rho > GHOST_RHO_THRESHOLD,
            "value": round(ghost_rho, 4),
            "threshold": f"> {GHOST_RHO_THRESHOLD}",
        },
        "tack_recovery_cv": {
            "passed": recovery_cv > RECOVERY_CV_THRESHOLD,
            "value": round(recovery_cv, 4),
            "threshold": f"> {RECOVERY_CV_THRESHOLD}",
        },
    }
    payload["extensions"] = {
        "leader_mid_gaps_splashdown": gaps_ext.to_dict(orient="records"),
        "ghost_overlay": {"spearman_rho": ghost_rho, "n_points": len(overlay_df)},
        "manoeuvre_recovery": {
            "n_events": len(manoeuvre_df),
            "n_recovered": int(manoeuvre_df["recovered"].sum()) if len(manoeuvre_df) else 0,
            "team_recovery_cv": recovery_cv,
        },
        "extension_criteria": ext_criteria,
        "outputs": {
            "ghost_scatter_html": str(scatter_path),
            "manoeuvre_recovery_csv": str(manoeuvre_path),
        },
    }
    payload["all_passed_with_extensions"] = bool(
        payload.get("all_passed", False)
        and all(c["passed"] for c in ext_criteria.values())
    )
    json_path.write_text(json.dumps(payload, indent=2))

    print("\n=== Flight Quality Extensions ===", flush=True)
    for name, c in ext_criteria.items():
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {name}: {c['value']} ({c['threshold']})", flush=True)
    print(f"\nWrote {scatter_path}, {manoeuvre_path}", flush=True)

    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
