#!/usr/bin/env python3
"""C3 — Mark rounding regret: where within a leg regret accumulates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import COL_LEG, COL_TWA, load_racing_boats
from dataExploration.next_experiments.exp_1b_flight_quality_extensions import build_regret_timeseries
from dataExploration.next_experiments.exp_4b_ghost_boat_breakdown import TWA_UPWIND_MAX, leg_twa_avg
from sailgp_analysis.config import DATA_ROOT, VENUES

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
MARK_ENTRY_S = 30
MARK_ROUNDING_HALF_S = 10
MARK_EXIT_S = 30
MARK_ZONE_FRAC_THRESHOLD = 0.25
TEAM_SPREAD_THRESHOLD = 1.5
ZONES = ("mark_entry", "mark_rounding", "mark_exit", "open_water")


def _leg_boundaries(grp: pd.DataFrame) -> list[dict]:
    """Return leg boundary events: first timestamp of each leg > 1."""
    grp = grp.sort_values("timestamp")
    events = []
    for leg, leg_df in grp.groupby("leg", sort=True):
        if leg <= 1 or leg_df.empty:
            continue
        boundary = pd.Timestamp(leg_df["timestamp"].iloc[0])
        events.append({"leg": int(leg), "boundary_ts": boundary})
    return events


def _zone_masks(
    grp: pd.DataFrame,
    boundary_ts: pd.Timestamp,
    leg: int,
) -> dict[str, pd.Series]:
    """Non-overlapping zone masks for one leg transition (priority: rounding > entry/exit)."""
    ts = pd.to_datetime(grp["timestamp"])
    leg_col = grp["leg"]
    b = boundary_ts
    rounding = (ts >= b - pd.Timedelta(seconds=MARK_ROUNDING_HALF_S)) & (
        ts <= b + pd.Timedelta(seconds=MARK_ROUNDING_HALF_S)
    )
    mark_entry = (
        (leg_col == leg - 1)
        & (ts >= b - pd.Timedelta(seconds=MARK_ENTRY_S))
        & (ts < b - pd.Timedelta(seconds=MARK_ROUNDING_HALF_S))
    )
    mark_exit = (
        (leg_col == leg)
        & (ts > b + pd.Timedelta(seconds=MARK_ROUNDING_HALF_S))
        & (ts <= b + pd.Timedelta(seconds=MARK_EXIT_S))
    )
    open_water = (leg_col == leg) & (ts > b + pd.Timedelta(seconds=MARK_EXIT_S))
    return {
        "mark_entry": mark_entry,
        "mark_rounding": rounding,
        "mark_exit": mark_exit,
        "open_water": open_water,
    }


def aggregate_zone_regret(regret_ts: pd.DataFrame, twa_legs: pd.DataFrame) -> pd.DataFrame:
    """Per team×race×leg: regret sum and fraction by zone."""
    if regret_ts.empty:
        return pd.DataFrame()

    regret_ts = regret_ts.copy()
    regret_ts["timestamp"] = pd.to_datetime(regret_ts["timestamp"])
    if "regret_rate_s" not in regret_ts.columns:
        regret_ts["regret_rate_s"] = regret_ts.groupby(
            ["venue", "race_label", "team", "leg"]
        )["cum_regret_s"].diff().fillna(0.0)

    rows = []
    for (venue, race, team), grp in regret_ts.groupby(["venue", "race_label", "team"], sort=False):
        for evt in _leg_boundaries(grp):
            leg = evt["leg"]
            masks = _zone_masks(grp, evt["boundary_ts"], leg)
            zone_sums = {z: float(grp.loc[masks[z], "regret_rate_s"].sum()) for z in ZONES}
            total = sum(zone_sums.values())
            twa_row = twa_legs[
                (twa_legs["venue"] == venue)
                & (twa_legs["race_label"] == race)
                & (twa_legs["team"] == team)
                & (twa_legs["leg"] == leg)
            ]
            twa_avg = float(twa_row["twa_avg_deg"].iloc[0]) if len(twa_row) else np.nan
            leg_type = "upwind" if twa_avg < TWA_UPWIND_MAX else "downwind" if not np.isnan(twa_avg) else "unknown"
            mark_zone = zone_sums["mark_entry"] + zone_sums["mark_rounding"] + zone_sums["mark_exit"]
            rows.append(
                {
                    "venue": venue,
                    "race_label": race,
                    "team": team,
                    "leg": leg,
                    "leg_type": leg_type,
                    "boundary_ts": evt["boundary_ts"].isoformat(),
                    **{f"regret_{z}_s": zone_sums[z] for z in ZONES},
                    "regret_total_zones_s": total,
                    "regret_mark_zone_s": mark_zone,
                    "mark_zone_frac": mark_zone / total if total > 1e-6 else np.nan,
                    "open_water_frac": zone_sums["open_water"] / total if total > 1e-6 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def team_summary(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Per-team mark-zone fraction (upwind legs only for hypothesis check)."""
    up = zone_df[zone_df["leg_type"] == "upwind"].copy()
    if up.empty:
        up = zone_df.copy()
    return (
        up.groupby("team", as_index=False)
        .agg(
            n_legs=("leg", "count"),
            mean_mark_zone_frac=("mark_zone_frac", "mean"),
            mean_mark_zone_regret_s=("regret_mark_zone_s", "mean"),
            mean_total_regret_s=("regret_total_zones_s", "mean"),
        )
        .sort_values("mean_mark_zone_frac", ascending=False)
    )


def evaluate_criteria(zone_df: pd.DataFrame, team_df: pd.DataFrame) -> dict:
    up = zone_df[zone_df["leg_type"] == "upwind"]
    mark_frac_mean = float(up["mark_zone_frac"].mean()) if len(up) else float(zone_df["mark_zone_frac"].mean())
    team_fracs = team_df["mean_mark_zone_frac"].dropna()
    spread = float(team_fracs.max() / team_fracs.min()) if len(team_fracs) > 1 and team_fracs.min() > 0 else 0.0
    fleet_median = float(team_fracs.median()) if len(team_fracs) else 0.0
    ita_frac = float(team_df.loc[team_df["team"] == "ITA", "mean_mark_zone_frac"].iloc[0]) if "ITA" in team_df["team"].values else np.nan
    ita_vs_median = bool(ita_frac >= fleet_median) if not np.isnan(ita_frac) else False

    return {
        "mark_zone_frac_mean_upwind": {
            "value": mark_frac_mean,
            "threshold": MARK_ZONE_FRAC_THRESHOLD,
            "pass": mark_frac_mean >= MARK_ZONE_FRAC_THRESHOLD,
        },
        "team_mark_frac_spread": {
            "value": spread,
            "threshold": TEAM_SPREAD_THRESHOLD,
            "pass": spread >= TEAM_SPREAD_THRESHOLD,
        },
        "ita_mark_frac_vs_median": {
            "value": ita_frac,
            "fleet_median": fleet_median,
            "pass": ita_vs_median,
        },
        "overall_pass": (
            mark_frac_mean >= MARK_ZONE_FRAC_THRESHOLD
            and spread >= TEAM_SPREAD_THRESHOLD
            and ita_vs_median
        ),
    }


def build_stacked_bar(team_df: pd.DataFrame, zone_df: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar: zone regret fractions per team (upwind legs)."""
    up = zone_df[zone_df["leg_type"] == "upwind"]
    if up.empty:
        up = zone_df
    agg = up.groupby("team", as_index=False).agg(
        **{f"regret_{z}_s": (f"regret_{z}_s", "sum") for z in ZONES}
    )
    totals = agg[[f"regret_{z}_s" for z in ZONES]].sum(axis=1).replace(0, np.nan)
    fig = go.Figure()
    colors = {
        "mark_entry": "#2563eb",
        "mark_rounding": "#dc2626",
        "mark_exit": "#ca8a04",
        "open_water": "#6b7280",
    }
    for z in ZONES:
        frac = agg[f"regret_{z}_s"] / totals
        fig.add_trace(
            go.Bar(
                name=z.replace("_", " ").title(),
                x=agg["team"],
                y=frac,
                marker_color=colors[z],
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Upwind Regret by Zone (fraction of leg regret)",
        yaxis_title="Fraction of regret",
        xaxis_title="Team",
        template="plotly_white",
        height=600,
        width=1000,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    boats = load_racing_boats(DATA_ROOT)
    regret_ts = build_regret_timeseries(boats)
    if regret_ts.empty:
        raise RuntimeError("No regret timeseries built — run exp_4 first")

    twa_legs = leg_twa_avg(boats).rename(columns={COL_LEG: "leg"})
    zone_df = aggregate_zone_regret(regret_ts, twa_legs)
    if zone_df.empty:
        raise RuntimeError("No zone aggregates")

    team_df = team_summary(zone_df)
    criteria = evaluate_criteria(zone_df, team_df)

    csv_path = EXPORT_DIR / "ghost_boat_mark_rounding.csv"
    zone_df.to_csv(csv_path, index=False)
    html_path = EXPORT_DIR / "ghost_boat_mark_zones.html"
    build_stacked_bar(team_df, zone_df, html_path)

    results = {
        "experiment": "c3_mark_rounding_regret",
        "venues": VENUES,
        "n_leg_transitions": int(len(zone_df)),
        "n_teams": int(zone_df["team"].nunique()),
        "zone_seconds": {
            "mark_entry": MARK_ENTRY_S,
            "mark_rounding": MARK_ROUNDING_HALF_S * 2,
            "mark_exit": MARK_EXIT_S,
        },
        "summary": {
            "mean_mark_zone_frac_upwind": float(
                zone_df.loc[zone_df["leg_type"] == "upwind", "mark_zone_frac"].mean()
            ),
            "mean_mark_zone_frac_all": float(zone_df["mark_zone_frac"].mean()),
            "ita_mean_mark_zone_frac": float(
                team_df.loc[team_df["team"] == "ITA", "mean_mark_zone_frac"].iloc[0]
            )
            if "ITA" in team_df["team"].values
            else None,
            "fleet_median_mark_zone_frac": float(team_df["mean_mark_zone_frac"].median()),
        },
        "team_summary": team_df.to_dict(orient="records"),
        "success_criteria": criteria,
        "overall_pass": bool(criteria["overall_pass"]),
        "outputs": {
            "csv": str(csv_path),
            "html": str(html_path),
            "json": str(EXPORT_DIR / "ghost_boat_mark_rounding_results.json"),
        },
    }
    json_path = EXPORT_DIR / "ghost_boat_mark_rounding_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    res = run()
    crit = res["success_criteria"]
    print(f"Overall: {'PASS' if res['overall_pass'] else 'FAIL'}")
    print(f"Mark zone frac (upwind mean): {res['summary']['mean_mark_zone_frac_upwind']:.3f}")
    print(f"ITA mark zone frac: {res['summary']['ita_mean_mark_zone_frac']}")
    print(f"Fleet median: {res['summary']['fleet_median_mark_zone_frac']:.3f}")
    for name, c in crit.items():
        if not isinstance(c, dict) or "pass" not in c:
            continue
        print(f"  {name}: {c.get('value', c)} -> {'PASS' if c['pass'] else 'FAIL'}")
    print(f"Outputs: {res['outputs']}")
