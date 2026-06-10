"""Experiment #3b — Dirty-Air Bearing Filter, POS Split, Exposure, Critical Distance.

Extends exp_3 with:
  1. GPS bearing alignment filter (|bearing| < 20° to boat ahead)
  2. Four point-of-sail bins (close-hauled / beam / broad / run)
  3. Seconds-in-dirty-air per team per race (PC_DTB_m < 60m)
  4. Piecewise-linear breakpoint for critical distance
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_RANK,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.fleet import (
    build_fleet_snapshots,
    ego_relative,
)
from dataExploration.next_experiments.exp_3_dirty_air import (
    BIN_CENTERS,
    DISTANCE_BINS,
    BinStats,
    compute_bin_stats,
    prepare_frame,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"

BEARING_FILTER_DEG = 20.0
EXPOSURE_THRESHOLD_M = 60.0
FORWARD_CONE_DEG = 90.0

POS_BINS = [
    (30, 60, "close_hauled"),
    (60, 90, "beam"),
    (90, 135, "broad"),
    (135, 180, "run"),
]


def compute_ahead_bearing(
    df: pd.DataFrame,
    forward_cone_deg: float = FORWARD_CONE_DEG,
    max_dist_m: float = 500.0,
) -> pd.Series:
    """Ego-frame bearing (deg) to nearest boat in the forward hemisphere."""
    snapshots = build_fleet_snapshots(df)
    out: dict[tuple, float] = {}

    for (venue, race_label), race_df in df.groupby(["venue", "race_label"], sort=False):
        for ts, tick in race_df.groupby(level=0, sort=False):
            snap = snapshots.get((venue, race_label, ts))
            if snap is None:
                continue
            for team, ego_row in tick.set_index("team").iterrows():
                if team not in snap.index:
                    continue
                others = snap.drop(index=team, errors="ignore")
                neighbours = ego_relative(ego_row, others, radius_m=max_dist_m)
                if neighbours.empty:
                    out[(venue, race_label, team, ts)] = np.nan
                    continue
                ahead = neighbours[np.abs(neighbours["bearing_deg"]) <= forward_cone_deg]
                if ahead.empty:
                    out[(venue, race_label, team, ts)] = np.nan
                else:
                    out[(venue, race_label, team, ts)] = float(ahead.iloc[0]["bearing_deg"])

    rows = []
    for (venue, race_label, team, ts), bearing in out.items():
        rows.append(
            {"DATETIME": ts, "venue": venue, "race_label": race_label, "team": team, "ahead_bearing_deg": bearing}
        )
    if not rows:
        return pd.Series(np.nan, index=df.index)
    feat = pd.DataFrame(rows).set_index("DATETIME")
    merged = df.reset_index().merge(feat, on=["DATETIME", "venue", "race_label", "team"], how="left")
    return merged.set_index("DATETIME")["ahead_bearing_deg"]


def assign_pos_bin(abs_twa: float) -> str | None:
    if np.isnan(abs_twa):
        return None
    for lo, hi, label in POS_BINS:
        if lo <= abs_twa < hi:
            return label
    if abs_twa >= 180:
        return "run"
    return None


def is_monotonic_penalty(stats: list[BinStats]) -> tuple[bool, list[float]]:
    """Closer bins should have lower (more negative) mean residual."""
    means = [b.mean_residual_kn for b in stats if b.n > 0 and not np.isnan(b.mean_residual_kn)]
    if len(means) < 2:
        return False, means
    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    return mono, means


def fit_piecewise_breakpoint(
    bin_stats: list[BinStats],
    d_min: float = 20.0,
    d_max: float = 180.0,
) -> dict:
    """One-breakpoint piecewise linear fit on bin-level loss vs control."""
    xs, ys = [], []
    for b, center in zip(bin_stats[:-1], BIN_CENTERS[:-1]):
        if b.n >= 10 and not np.isnan(b.loss_vs_control_kn):
            xs.append(center)
            ys.append(b.loss_vs_control_kn)
    if len(xs) < 3:
        return {"model": "insufficient_bins", "n_bins": len(xs)}

    d = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)

    def pw_residual(params: np.ndarray) -> float:
        bp, m1, m2, y_bp = params
        if bp <= d_min or bp >= d_max:
            return 1e12
        y_pred = np.where(d < bp, m1 * (d - bp) + y_bp, m2 * (d - bp) + y_bp)
        return float(np.sum((y - y_pred) ** 2))

    x0 = np.array([75.0, -0.01, -0.005, float(y[0])])
    bounds = [(40.0, 150.0), (-0.1, 0.0), (-0.05, 0.01), (0.0, 5.0)]
    result = minimize(pw_residual, x0, method="L-BFGS-B", bounds=bounds)

    bp, m1, m2, y_bp = result.x
    y_pred = np.where(d < bp, m1 * (d - bp) + y_bp, m2 * (d - bp) + y_bp)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "model": "piecewise_linear",
        "critical_distance_m": float(bp),
        "slope_close_m1": float(m1),
        "slope_far_m2": float(m2),
        "loss_at_breakpoint_kn": float(y_bp),
        "r2": float(r2),
        "n_bins": int(len(xs)),
        "fit_success": bool(result.success),
        "bin_centers_m": xs,
        "bin_loss_kn": [round(v, 3) for v in ys],
    }


def compute_exposure(venue: str = "Bermuda") -> pd.DataFrame:
    """Seconds at PC_DTB_m < 60m per team per race (following boats only)."""
    df = load_racing_boats()
    if venue:
        df = df[df["venue"] == venue].copy()

    following = df[COL_RANK].to_numpy(dtype=float) > 1 if COL_RANK in df.columns else np.ones(len(df), dtype=bool)
    dirty = df[COL_DTB].to_numpy(dtype=float) < EXPOSURE_THRESHOLD_M
    mask = following & dirty & df[COL_DTB].notna()

    sub = df.loc[mask, ["venue", "race_label", "team"]].copy()
    counts = sub.groupby(["venue", "race_label", "team"], as_index=False).size()
    counts = counts.rename(columns={"size": "seconds_in_dirty_air"})

    race_totals = df.groupby(["venue", "race_label", "team"], as_index=False).size()
    race_totals = race_totals.rename(columns={"size": "racing_seconds"})
    out = counts.merge(race_totals, on=["venue", "race_label", "team"], how="left")
    out["racing_seconds"] = out["racing_seconds"].fillna(0).astype(int)
    out["pct_in_dirty_air"] = out["seconds_in_dirty_air"] / out["racing_seconds"].clip(lower=1)
    return out.sort_values(["venue", "race_label", "team"])


def plot_bearing_filtered(
    all_stats: list[BinStats],
    pos_stats: dict[str, list[BinStats]],
    meta: dict,
    breakpoint: dict,
) -> go.Figure:
    labels = [b.label for b in all_stats]
    means = [b.mean_residual_kn for b in all_stats]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Bearing-filtered penalty curve (all POS)",
            "Loss vs control (kn)",
            "Close-hauled vs run (worst POS check)",
            "Point-of-sail bin means at 30-60m",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=means,
            error_y=dict(type="data", array=[b.std_residual_kn for b in all_stats], visible=True),
            mode="lines+markers",
            name="All POS",
            line=dict(color="#2563eb"),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )

    losses = [b.loss_vs_control_kn for b in all_stats]
    fig.add_trace(
        go.Bar(x=labels, y=losses, marker_color="#dc2626", name="Loss vs control"),
        row=1,
        col=2,
    )

    colors = {"close_hauled": "#059669", "beam": "#0891b2", "broad": "#d97706", "run": "#dc2626"}
    for pos, stats_list in pos_stats.items():
        close_bin = next((b for b in stats_list if b.label == "30-60m"), None)
        if close_bin and close_bin.n > 0:
            fig.add_trace(
                go.Bar(
                    x=[pos],
                    y=[close_bin.mean_residual_kn],
                    name=pos,
                    marker_color=colors.get(pos, "#64748b"),
                    showlegend=True,
                ),
                row=2,
                col=2,
            )

    ch = pos_stats.get("close_hauled", [])
    run = pos_stats.get("run", [])
    for stats_list, name, color in [(ch, "Close-hauled", "#059669"), (run, "Run", "#dc2626")]:
        if stats_list:
            fig.add_trace(
                go.Scatter(
                    x=[b.label for b in stats_list],
                    y=[b.mean_residual_kn for b in stats_list],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color),
                ),
                row=2,
                col=1,
            )

    bp = breakpoint.get("critical_distance_m")
    title = (
        f"Dirty-Air Bearing-Filtered — {meta.get('venue', 'Bermuda')} "
        f"(|bearing|<{BEARING_FILTER_DEG}°, n={meta.get('n_rows_bearing_filtered', 0):,}"
    )
    if bp is not None:
        title += f", critical dist≈{bp:.0f}m"
    title += ")"
    fig.update_layout(title=title, height=750, showlegend=True)
    fig.update_yaxes(title_text="Speed residual (kn)")
    return fig


def evaluate_success(
    bearing_stats: list[BinStats],
    pos_stats: dict[str, list[BinStats]],
    exposure: pd.DataFrame,
) -> dict:
    mono, means = is_monotonic_penalty(bearing_stats)

    pos_penalties = {}
    for pos, stats_list in pos_stats.items():
        close = next((b for b in stats_list if b.label == "30-60m" and b.n >= 5), None)
        if close is not None:
            pos_penalties[pos] = float(close.loss_vs_control_kn)

    run_worst = False
    if pos_penalties:
        run_worst = pos_penalties.get("run", 0.0) == max(pos_penalties.values())

    team_totals = exposure.groupby("team")["seconds_in_dirty_air"].sum()
    exposure_ratio = float(team_totals.max() / max(team_totals.min(), 1))
    exposure_pass = exposure_ratio >= 2.0

    criteria = {
        "bearing_filtered_monotonic": {
            "pass": mono,
            "detail": {"bin_means_kn": [round(m, 3) for m in means]},
        },
        "run_worst_point_of_sail": {
            "pass": run_worst,
            "detail": {k: round(v, 3) for k, v in pos_penalties.items()},
        },
        "team_exposure_2x_spread": {
            "pass": exposure_pass,
            "detail": {
                "min_seconds": int(team_totals.min()),
                "max_seconds": int(team_totals.max()),
                "ratio": round(exposure_ratio, 2),
            },
        },
    }
    return {"overall_pass": all(c["pass"] for c in criteria.values()), "criteria": criteria}


def run(venue: str = "Bermuda") -> dict:
    print(f"Exp #3b Dirty-Air Bearing Filter (venue={venue})", flush=True)

    print("Loading analysis frame...", flush=True)
    frame, meta = prepare_frame(venue=venue)
    meta["bearing_filter_deg"] = BEARING_FILTER_DEG

    print("Computing ahead bearings from GPS...", flush=True)
    frame["ahead_bearing_deg"] = compute_ahead_bearing(frame)
    n_with_bearing = int(frame["ahead_bearing_deg"].notna().sum())
    meta["n_rows_with_bearing"] = n_with_bearing

    filtered = frame[frame["ahead_bearing_deg"].abs() < BEARING_FILTER_DEG].copy()
    meta["n_rows_bearing_filtered"] = len(filtered)
    meta["bearing_filter_retention_pct"] = round(100.0 * len(filtered) / max(len(frame), 1), 1)
    print(
        f"  {len(filtered):,} / {len(frame):,} rows pass bearing filter "
        f"({meta['bearing_filter_retention_pct']}%)",
        flush=True,
    )

    bearing_stats = compute_bin_stats(filtered)

    pos_stats: dict[str, list[BinStats]] = {}
    for lo, hi, label in POS_BINS:
        sub = filtered[(filtered["abs_twa"] >= lo) & (filtered["abs_twa"] < hi)]
        pos_stats[label] = compute_bin_stats(sub)

    print("Fitting piecewise breakpoint...", flush=True)
    breakpoint = fit_piecewise_breakpoint(bearing_stats)

    print("Computing team exposure...", flush=True)
    exposure = compute_exposure(venue=venue)
    exposure_path = EXPORT_DIR / "dirty_air_exposure.csv"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    exposure.to_csv(exposure_path, index=False)

    success = evaluate_success(bearing_stats, pos_stats, exposure)
    fig = plot_bearing_filtered(bearing_stats, pos_stats, meta, breakpoint)

    html_path = EXPORT_DIR / "dirty_air_bearing_filtered.html"
    json_path = EXPORT_DIR / "dirty_air_critical_distance.json"

    payload = {
        "experiment": "dirty_air_bearing_filter",
        "meta": meta,
        "bin_stats_bearing_filtered": [asdict(b) for b in bearing_stats],
        "bin_stats_by_pos": {k: [asdict(b) for b in v] for k, v in pos_stats.items()},
        "critical_distance": breakpoint,
        "exposure_summary": {
            "threshold_m": EXPOSURE_THRESHOLD_M,
            "n_rows": len(exposure),
            "team_totals": exposure.groupby("team")["seconds_in_dirty_air"].sum().to_dict(),
        },
        "success": success,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    fig.write_html(str(html_path))

    print("\nBearing-filtered bin stats (mean residual kn):", flush=True)
    for b in bearing_stats:
        print(f"  {b.label:20s} n={b.n:6d}  {b.mean_residual_kn:+.3f}  loss={b.loss_vs_control_kn:+.3f}kn", flush=True)

    print(f"\nCritical distance: {breakpoint}", flush=True)
    print(f"Exposure: {len(exposure)} team×race rows → {exposure_path}", flush=True)
    print(f"Overall: {'PASS' if success['overall_pass'] else 'FAIL'}", flush=True)
    for name, c in success["criteria"].items():
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'} — {c['detail']}", flush=True)
    print(f"\nWrote {html_path}, {exposure_path}, {json_path}", flush=True)
    return success


def main():
    run(venue="Bermuda")


if __name__ == "__main__":
    main()
