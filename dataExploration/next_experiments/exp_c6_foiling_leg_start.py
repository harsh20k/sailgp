#!/usr/bin/env python3
"""C6 — Foiling leg-start quality after mark roundings."""
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
    COL_STATUS,
    COL_TWA,
    load_racing_boats,
)
from dataExploration.next_experiments.exp_1b_flight_quality_extensions import (
    MANOEUVRE_DEG,
    MANOEUVRE_WINDOW_S,
    _angle_diff_deg,
    _heading_col,
    detect_manoeuvres,
)
from dataExploration.next_experiments.exp_4b_ghost_boat_breakdown import TWA_UPWIND_MAX, leg_twa_avg
from sailgp_analysis.config import DATA_ROOT, VENUES

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
RECOVERY_THRESHOLD = 0.7
BOUNDARY_WINDOW_S = 10
FQ_OFFSETS_S = (0, 5, 10, 20)
RHO_THRESHOLD = 0.4
R2_THRESHOLD = 0.20


def _leg_boundary_events(grp: pd.DataFrame) -> list[dict]:
    grp = grp.sort_values("timestamp")
    events = []
    for leg, leg_df in grp.groupby("leg", sort=True):
        if leg <= 1 or leg_df.empty:
            continue
        events.append({"leg": int(leg), "boundary_ts": pd.Timestamp(leg_df["timestamp"].iloc[0])})
    return events


def _fq_at_offset(fq: pd.Series, t0: pd.Timestamp, offset_s: int) -> float:
    target = t0 + pd.Timedelta(seconds=offset_s)
    if fq.empty:
        return np.nan
    if target in fq.index:
        return float(fq.loc[target])
    return float(fq.asof(target)) if fq.index.min() <= target <= fq.index.max() else np.nan


def _reestablishment_s(fq: pd.Series, t0: pd.Timestamp) -> float:
    post = fq[fq.index > t0]
    recovered = post[post > RECOVERY_THRESHOLD]
    if recovered.empty:
        return np.nan
    return float((recovered.index[0] - t0).total_seconds())


def leg_start_events(
    scores: pd.DataFrame,
    source_df: pd.DataFrame,
    twa_legs: pd.DataFrame,
) -> pd.DataFrame:
    """Per leg-boundary rounding: re-establishment time and FQ snapshots."""
    fq_all = scores.copy()
    fq_all["timestamp"] = pd.to_datetime(fq_all["timestamp"])
    fq_all = fq_all.set_index("timestamp")

    rows = []
    racing = source_df[source_df[COL_STATUS] == 2]
    hcol = _heading_col(source_df)

    for (venue, race, team), boat_df in racing.groupby(["venue", "race_label", "team"], sort=False):
        boat_ts = boat_df.copy()
        boat_ts["timestamp"] = pd.to_datetime(boat_ts.index)
        boat_ts["leg"] = boat_ts[COL_LEG]
        manoeuvres = set(pd.to_datetime(detect_manoeuvres(boat_df, hcol)))

        team_fq = fq_all[
            (fq_all["venue"] == venue) & (fq_all["race_label"] == race) & (fq_all["team"] == team)
        ]["flight_quality"]

        for evt in _leg_boundary_events(boat_ts):
            leg = evt["leg"]
            bnd = evt["boundary_ts"]
            twa_row = twa_legs[
                (twa_legs["venue"] == venue)
                & (twa_legs["race_label"] == race)
                & (twa_legs["team"] == team)
                & (twa_legs["leg"] == leg)
            ]
            twa_avg = float(twa_row["twa_avg_deg"].iloc[0]) if len(twa_row) else np.nan
            leg_type = "upwind" if twa_avg < TWA_UPWIND_MAX else "downwind" if not np.isnan(twa_avg) else "unknown"

            near_manoeuvre = any(
                abs((m - bnd).total_seconds()) <= BOUNDARY_WINDOW_S for m in manoeuvres
            )
            event_type = "leg_boundary_rounding"

            row = {
                "venue": venue,
                "race_label": race,
                "team": team,
                "leg": leg,
                "leg_type": leg_type,
                "boundary_ts": bnd.isoformat(),
                "event_type": event_type,
                "has_heading_manoeuvre": near_manoeuvre,
                "reestablishment_s": _reestablishment_s(team_fq, bnd),
                "recovered": False,
            }
            for off in FQ_OFFSETS_S:
                row[f"fq_t{off}s"] = _fq_at_offset(team_fq, bnd, off)
            row["recovered"] = not np.isnan(row["reestablishment_s"])
            rows.append(row)

    return pd.DataFrame(rows)


def tag_mid_leg_manoeuvres(
    scores: pd.DataFrame,
    source_df: pd.DataFrame,
    boundary_events: pd.DataFrame,
) -> pd.DataFrame:
    """Classify manoeuvres as mid-leg vs leg-boundary (for reporting)."""
    hcol = _heading_col(source_df)
    fq_all = scores.copy()
    fq_all["timestamp"] = pd.to_datetime(fq_all["timestamp"])
    fq_all = fq_all.set_index("timestamp")

    bnd_set = set()
    for row in boundary_events.itertuples(index=False):
        bnd_set.add((row.venue, row.race_label, row.team, pd.Timestamp(row.boundary_ts)))

    rows = []
    racing = source_df[source_df[COL_STATUS] == 2]
    for (venue, race, team), boat_df in racing.groupby(["venue", "race_label", "team"], sort=False):
        for evt in detect_manoeuvres(boat_df, hcol):
            evt = pd.Timestamp(evt)
            is_boundary = any(
                abs((evt - b).total_seconds()) <= BOUNDARY_WINDOW_S
                for v, r, t, b in bnd_set
                if v == venue and r == race and t == team
            )
            team_fq = fq_all[
                (fq_all["venue"] == venue) & (fq_all["race_label"] == race) & (fq_all["team"] == team)
            ]["flight_quality"]
            rows.append(
                {
                    "venue": venue,
                    "race_label": race,
                    "team": team,
                    "manoeuvre_time": evt.isoformat(),
                    "event_type": "leg_boundary_rounding" if is_boundary else "mid_leg_manoeuvre",
                    "reestablishment_s": _reestablishment_s(team_fq, evt),
                }
            )
    return pd.DataFrame(rows)


def team_reestablishment_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (team, leg_type), g in events.dropna(subset=["reestablishment_s"]).groupby(["team", "leg_type"]):
        rows.append(
            {
                "team": team,
                "leg_type": leg_type,
                "n_events": len(g),
                "mean_reestablishment_s": float(g["reestablishment_s"].mean()),
                "median_reestablishment_s": float(g["reestablishment_s"].median()),
            }
        )
    return pd.DataFrame(rows)


def load_upwind_regret() -> pd.DataFrame:
    path = EXPORT_DIR / "ghost_boat_regret_by_leg_type.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run exp_4b first: {path}")
    regret = pd.read_csv(path)
    return regret[regret["leg_type"] == "upwind"][["team", "mean_regret_s"]].rename(
        columns={"mean_regret_s": "mean_upwind_regret_s"}
    )


def evaluate_criteria(
    team_upwind: pd.DataFrame,
    regret_df: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict:
    valid = merged.dropna(subset=["mean_reestablishment_s", "mean_upwind_regret_s"])
    rho = 0.0
    p_val = 1.0
    r2 = 0.0
    if len(valid) >= 4:
        rho, p_val = stats.spearmanr(valid["mean_reestablishment_s"], valid["mean_upwind_regret_s"])
        slope, intercept, _, _, _ = stats.linregress(
            valid["mean_reestablishment_s"], valid["mean_upwind_regret_s"]
        )
        pred = slope * valid["mean_reestablishment_s"] + intercept
        ss_res = float(((valid["mean_upwind_regret_s"] - pred) ** 2).sum())
        ss_tot = float(((valid["mean_upwind_regret_s"] - valid["mean_upwind_regret_s"].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    upwind_reest = team_upwind[team_upwind["leg_type"] == "upwind"]
    fleet_mean = float(upwind_reest["mean_reestablishment_s"].mean()) if len(upwind_reest) else np.nan
    fleet_std = float(upwind_reest["mean_reestablishment_s"].std()) if len(upwind_reest) > 1 else 0.0
    ita_reest = float(
        upwind_reest.loc[upwind_reest["team"] == "ITA", "mean_reestablishment_s"].iloc[0]
    ) if "ITA" in upwind_reest["team"].values else np.nan
    ita_pass = bool(ita_reest >= fleet_mean + fleet_std) if not np.isnan(ita_reest) else False

    return {
        "spearman_rho": {
            "value": float(rho),
            "p_value": float(p_val),
            "threshold": RHO_THRESHOLD,
            "pass": float(rho) >= RHO_THRESHOLD,
        },
        "r2_upwind_regret": {
            "value": float(r2),
            "threshold": R2_THRESHOLD,
            "pass": float(r2) >= R2_THRESHOLD,
        },
        "ita_reestablishment_vs_fleet": {
            "value": ita_reest,
            "fleet_mean": fleet_mean,
            "fleet_std": fleet_std,
            "threshold": "median + 1 std",
            "pass": ita_pass,
        },
        "overall_pass": (
            float(rho) >= RHO_THRESHOLD
            and float(r2) >= R2_THRESHOLD
            and ita_pass
        ),
    }


def build_scatter(merged: pd.DataFrame, rho: float, out_path: Path) -> None:
    fig = px.scatter(
        merged,
        x="mean_reestablishment_s",
        y="mean_upwind_regret_s",
        text="team",
        title=f"Leg-Start Re-establishment vs Mean Upwind Regret (Spearman ρ={rho:.3f})",
        labels={
            "mean_reestablishment_s": "Mean re-establishment time (s)",
            "mean_upwind_regret_s": "Mean upwind regret (s)",
        },
        height=600,
        width=900,
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(template="plotly_white")
    fig.write_html(out_path, include_plotlyjs="cdn")


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fq_path = EXPORT_DIR / "flight_quality.csv"
    if not fq_path.exists():
        raise FileNotFoundError(f"Run exp_1 first: {fq_path}")

    scores = pd.read_csv(fq_path, parse_dates=["timestamp"])
    source_df = load_racing_boats(DATA_ROOT)
    twa_legs = leg_twa_avg(source_df).rename(columns={COL_LEG: "leg"})

    events = leg_start_events(scores, source_df, twa_legs)
    if events.empty:
        raise RuntimeError("No leg-start events")

    manoeuvre_tags = tag_mid_leg_manoeuvres(scores, source_df, events)
    team_summary = team_reestablishment_summary(events)
    regret_df = load_upwind_regret()

    merged = team_summary[team_summary["leg_type"] == "upwind"].merge(regret_df, on="team", how="inner")
    criteria = evaluate_criteria(team_summary, regret_df, merged)
    rho = criteria["spearman_rho"]["value"]

    csv_path = EXPORT_DIR / "foiling_leg_start.csv"
    events.to_csv(csv_path, index=False)
    scatter_path = EXPORT_DIR / "foiling_leg_start_scatter.html"
    build_scatter(merged, rho, scatter_path)

    results = {
        "experiment": "c6_foiling_leg_start",
        "venues": VENUES,
        "n_leg_boundary_events": int(len(events)),
        "n_recovered": int(events["recovered"].sum()),
        "recovery_threshold": RECOVERY_THRESHOLD,
        "fq_offsets_s": list(FQ_OFFSETS_S),
        "manoeuvre_classification": {
            "n_mid_leg": int((manoeuvre_tags["event_type"] == "mid_leg_manoeuvre").sum())
            if len(manoeuvre_tags)
            else 0,
            "n_boundary": int((manoeuvre_tags["event_type"] == "leg_boundary_rounding").sum())
            if len(manoeuvre_tags)
            else 0,
        },
        "team_summary": team_summary.to_dict(orient="records"),
        "correlation": {
            "spearman_rho": rho,
            "r2": criteria["r2_upwind_regret"]["value"],
            "n_teams": int(len(merged)),
        },
        "ita": {
            "mean_upwind_reestablishment_s": float(
                team_summary.loc[
                    (team_summary["team"] == "ITA") & (team_summary["leg_type"] == "upwind"),
                    "mean_reestablishment_s",
                ].iloc[0]
            )
            if len(
                team_summary[(team_summary["team"] == "ITA") & (team_summary["leg_type"] == "upwind")]
            )
            else None,
            "mean_upwind_regret_s": float(regret_df.loc[regret_df["team"] == "ITA", "mean_upwind_regret_s"].iloc[0])
            if "ITA" in regret_df["team"].values
            else None,
        },
        "success_criteria": criteria,
        "overall_pass": bool(criteria["overall_pass"]),
        "outputs": {
            "csv": str(csv_path),
            "json": str(EXPORT_DIR / "foiling_leg_start_results.json"),
            "scatter_html": str(scatter_path),
        },
    }
    json_path = EXPORT_DIR / "foiling_leg_start_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    res = run()
    crit = res["success_criteria"]
    print(f"Overall: {'PASS' if res['overall_pass'] else 'FAIL'}")
    print(f"Spearman ρ: {res['correlation']['spearman_rho']:.3f}")
    print(f"R²: {res['correlation']['r2']:.3f}")
    print(f"ITA re-establishment: {res['ita']['mean_upwind_reestablishment_s']}")
    for name, c in crit.items():
        if not isinstance(c, dict) or "pass" not in c:
            continue
        print(f"  {name}: {c.get('value', c)} -> {'PASS' if c['pass'] else 'FAIL'}")
    print(f"Outputs: {res['outputs']}")
