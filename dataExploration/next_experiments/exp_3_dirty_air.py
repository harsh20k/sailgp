"""Experiment #3 — Dirty-Air Speed Penalty Curve.

Quantify speed loss vs following distance (PC_DTB_m) using polar-expected speed
residuals. No ML training.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_RANK,
    COL_SPEED,
    COL_TWA,
    COL_TWS,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.fleet import (
    DEFAULT_DIRTY_AIR_CONE_DEG,
    build_fleet_snapshots,
    ego_relative,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"

KM_TO_KN = 1.0 / 1.852
TWA_BIN = 10.0
TWS_BIN = 2.0
MIN_POLAR_COUNT = 20
DTB_NULL_THRESHOLD = 0.30  # fallback to GPS if >30% null

DISTANCE_BINS = [
    (0, 30, "0-30m"),
    (30, 60, "30-60m"),
    (60, 100, "60-100m"),
    (100, 150, "100-150m"),
    (150, 200, "150-200m"),
    (200, 500, "200-500m (control)"),
]

BIN_CENTERS = [15, 45, 80, 125, 175, 350]


@dataclass
class BinStats:
    label: str
    lo: float
    hi: float
    n: int
    mean_residual_kn: float
    std_residual_kn: float
    mean_residual_kmh: float
    std_residual_kmh: float
    mwu_p_vs_control: float
    cohens_d_vs_control: float
    loss_vs_control_kn: float


def build_polar_speed_table(
    df: pd.DataFrame,
    twa_bin: float = TWA_BIN,
    tws_bin: float = TWS_BIN,
    min_count: int = MIN_POLAR_COUNT,
) -> tuple[dict[tuple[float, float], float], float]:
    """Mean speed per (TWA_bin, TWS_bin); fallback to global mean."""
    sub = df.dropna(subset=[COL_SPEED, COL_TWA, COL_TWS]).copy()
    sub["twa_bin"] = np.floor(np.abs(sub[COL_TWA]) / twa_bin) * twa_bin
    sub["tws_bin"] = np.floor(sub[COL_TWS] / tws_bin) * tws_bin
    grouped = sub.groupby(["twa_bin", "tws_bin"])[COL_SPEED].agg(["mean", "count"])
    table: dict[tuple[float, float], float] = {}
    for (tb, sb), row in grouped.iterrows():
        if row["count"] >= min_count:
            table[(float(tb), float(sb))] = float(row["mean"])
    global_mean = float(sub[COL_SPEED].mean())
    return table, global_mean


def polar_expected_speed(
    twa: float,
    tws: float,
    table: dict[tuple[float, float], float],
    global_mean: float,
    twa_bin: float = TWA_BIN,
    tws_bin: float = TWS_BIN,
) -> float:
    if np.isnan(twa) or np.isnan(tws):
        return np.nan
    tb = float(np.floor(np.abs(twa) / twa_bin) * twa_bin)
    sb = float(np.floor(tws / tws_bin) * tws_bin)
    if (tb, sb) in table:
        return table[(tb, sb)]
    # nearest-bin fallback
    if table:
        best = min(table.keys(), key=lambda k: abs(k[0] - tb) + abs(k[1] - sb))
        return table[best]
    return global_mean


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def gps_ahead_distance(
    df: pd.DataFrame,
    cone_deg: float = DEFAULT_DIRTY_AIR_CONE_DEG,
    max_dist_m: float = 500.0,
) -> pd.Series:
    """Nearest boat in forward cone from fleet snapshots."""
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
                ahead = neighbours[np.abs(neighbours["bearing_deg"]) <= cone_deg]
                if ahead.empty:
                    out[(venue, race_label, team, ts)] = np.nan
                else:
                    out[(venue, race_label, team, ts)] = float(ahead["dist_m"].min())

    rows = []
    for (venue, race_label, team, ts), dist in out.items():
        rows.append({"DATETIME": ts, "venue": venue, "race_label": race_label, "team": team, "gps_ahead_m": dist})
    if not rows:
        return pd.Series(np.nan, index=df.index)
    feat = pd.DataFrame(rows).set_index("DATETIME")
    merged = df.reset_index().merge(feat, on=["DATETIME", "venue", "race_label", "team"], how="left")
    return merged.set_index("DATETIME")["gps_ahead_m"]


def prepare_frame(venue: str = "Bermuda") -> tuple[pd.DataFrame, dict]:
    """Load data, polar residuals, following distance."""
    df = load_racing_boats()
    if venue:
        df = df[df["venue"] == venue].copy()
    meta: dict = {"venue": venue, "n_rows_raw": len(df)}

    # Polar table from all loaded racing data
    polar_table, global_mean = build_polar_speed_table(df)
    meta["polar_bins"] = len(polar_table)
    meta["polar_global_mean_kmh"] = global_mean

    twa = df[COL_TWA].to_numpy(dtype=float)
    tws = df[COL_TWS].to_numpy(dtype=float)
    speed = df[COL_SPEED].to_numpy(dtype=float)
    expected = np.array(
        [polar_expected_speed(t, w, polar_table, global_mean) for t, w in zip(twa, tws)],
        dtype=float,
    )
    df["speed_expected_kmh"] = expected
    df["speed_residual_kmh"] = speed - expected
    df["speed_residual_kn"] = df["speed_residual_kmh"] * KM_TO_KN
    df["abs_twa"] = np.abs(df[COL_TWA])

    # Following distance: PC_DTB with GPS fallback
    dtb_null_rate = float(df[COL_DTB].isna().mean()) if COL_DTB in df.columns else 1.0
    meta["dtb_null_rate"] = dtb_null_rate
    meta["distance_source"] = "PC_DTB_m"

    if COL_DTB in df.columns and dtb_null_rate <= DTB_NULL_THRESHOLD:
        df["follow_dist_m"] = df[COL_DTB]
    else:
        print(f"  PC_DTB_m null rate {dtb_null_rate:.1%} — using GPS ahead distance", flush=True)
        meta["distance_source"] = "gps_ahead_cone"
        df["follow_dist_m"] = gps_ahead_distance(df)

    if COL_DTB in df.columns:
        missing = df["follow_dist_m"].isna()
        if missing.any():
            gps = gps_ahead_distance(df)
            df.loc[missing, "follow_dist_m"] = gps[missing]
            meta["distance_source"] = "PC_DTB_m+gps_fill"

    # Following only (not leading), within 500m
    rank = df[COL_RANK].to_numpy(dtype=float) if COL_RANK in df.columns else np.ones(len(df))
    df["is_following"] = rank > 1
    df = df[df["is_following"]].copy()
    df = df[df["follow_dist_m"].notna() & (df["follow_dist_m"] <= 500)].copy()
    df = df.dropna(subset=["speed_residual_kn", COL_TWA])
    meta["n_rows_analysis"] = len(df)
    return df, meta


def assign_bin(dist: float) -> str | None:
    for lo, hi, label in DISTANCE_BINS:
        if lo <= dist < hi:
            return label
    return None


def compute_bin_stats(frame: pd.DataFrame) -> list[BinStats]:
    frame = frame.copy()
    frame["dist_bin"] = frame["follow_dist_m"].apply(assign_bin)
    frame = frame[frame["dist_bin"].notna()]

    control = frame[frame["dist_bin"] == "200-500m (control)"]["speed_residual_kn"].to_numpy()
    results: list[BinStats] = []

    for lo, hi, label in DISTANCE_BINS:
        sub = frame[frame["dist_bin"] == label]
        vals = sub["speed_residual_kn"].to_numpy()
        vals_kmh = sub["speed_residual_kmh"].to_numpy()
        if len(vals) == 0:
            results.append(
                BinStats(label, lo, hi, 0, np.nan, np.nan, np.nan, np.nan, 1.0, 0.0, np.nan)
            )
            continue

        if len(control) >= 10 and len(vals) >= 10 and label != "200-500m (control)":
            _, p = stats.mannwhitneyu(vals, control, alternative="less")
            d = _cohens_d(vals, control)
            loss = float(np.mean(control) - np.mean(vals))
        elif label == "200-500m (control)":
            p, d, loss = 1.0, 0.0, 0.0
        else:
            p, d, loss = 1.0, 0.0, np.nan

        results.append(
            BinStats(
                label=label,
                lo=lo,
                hi=hi,
                n=len(vals),
                mean_residual_kn=float(np.mean(vals)),
                std_residual_kn=float(np.std(vals)),
                mean_residual_kmh=float(np.mean(vals_kmh)),
                std_residual_kmh=float(np.std(vals_kmh)),
                mwu_p_vs_control=float(p),
                cohens_d_vs_control=d,
                loss_vs_control_kn=loss,
            )
        )
    return results


def fit_decay_curve(bin_stats: list[BinStats]) -> dict:
    """Fit exponential decay to bin means (exclude control for fit)."""
    xs, ys = [], []
    for b, center in zip(bin_stats[:-1], BIN_CENTERS[:-1]):
        if b.n > 0 and not np.isnan(b.mean_residual_kn):
            xs.append(center)
            ys.append(b.mean_residual_kn)

    if len(xs) < 3:
        return {"model": "none", "params": {}, "r2": np.nan}

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    def exp_decay(d, a, b, c):
        return a * np.exp(-b * np.clip(d, 1, 500)) + c

    try:
        popt, _ = curve_fit(
            exp_decay,
            xs_arr,
            ys_arr,
            p0=[-0.5, 0.01, 0.0],
            maxfev=5000,
            bounds=([-5, 0, -2], [5, 1, 2]),
        )
        pred = exp_decay(xs_arr, *popt)
        ss_res = np.sum((ys_arr - pred) ** 2)
        ss_tot = np.sum((ys_arr - np.mean(ys_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "model": "exponential",
            "params": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])},
            "r2": float(r2),
            "formula": "residual_kn = a*exp(-b*dist_m) + c",
        }
    except Exception as e:
        return {"model": "fit_failed", "error": str(e), "params": {}, "r2": np.nan}


def evaluate_success(
    all_stats: list[BinStats],
    upwind_stats: list[BinStats],
    downwind_stats: list[BinStats],
) -> dict:
    """Check pass/fail against experiment success criteria."""
    # Monotonic: means decrease as distance decreases (0-30 lowest)
    means = [b.mean_residual_kn for b in all_stats if b.n > 0 and not np.isnan(b.mean_residual_kn)]
    monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1)) if len(means) >= 2 else False

    # MWU p < 0.05 for bins <= 100m
    close_bins = [b for b in all_stats if b.hi <= 100 and b.label != "200-500m (control)"]
    mwu_pass = all(b.mwu_p_vs_control < 0.05 for b in close_bins if b.n >= 10)

    # >= 0.3 kn loss at < 50m (bins 0-30 and 30-60)
    close50 = [b for b in all_stats if b.hi <= 50]
    effect_pass = any(b.loss_vs_control_kn >= 0.3 for b in close50 if not np.isnan(b.loss_vs_control_kn))

    # Downwind steeper than upwind
    def _slope(stats_list: list[BinStats]) -> float:
        pts = [(BIN_CENTERS[i], b.mean_residual_kn) for i, b in enumerate(stats_list[:-1]) if b.n > 0]
        if len(pts) < 2:
            return 0.0
        x, y = zip(*pts)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    slope_up = _slope(upwind_stats)
    slope_down = _slope(downwind_stats)
    downwind_steeper = abs(slope_down) > abs(slope_up)

    criteria = {
        "monotonic_penalty": {"pass": monotonic, "detail": f"bin means (kn): {[round(m, 3) for m in means]}"},
        "mwu_significance_le100m": {
            "pass": mwu_pass,
            "detail": {b.label: round(b.mwu_p_vs_control, 6) for b in close_bins},
        },
        "effect_size_ge03kn_lt50m": {
            "pass": effect_pass,
            "detail": {b.label: round(b.loss_vs_control_kn, 3) for b in close50},
        },
        "downwind_steeper": {
            "pass": downwind_steeper,
            "detail": {"upwind_slope": round(slope_up, 6), "downwind_slope": round(slope_down, 6)},
        },
    }
    overall = all(c["pass"] for c in criteria.values())
    return {"overall_pass": overall, "criteria": criteria}


def plot_penalty_curve(
    all_stats: list[BinStats],
    upwind_stats: list[BinStats],
    downwind_stats: list[BinStats],
    decay: dict,
    meta: dict,
) -> go.Figure:
    labels = [b.label for b in all_stats]
    means = [b.mean_residual_kn for b in all_stats]
    stds = [b.std_residual_kn for b in all_stats]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "All points of sail",
            "Upwind (|TWA| < 90°)",
            "Downwind (|TWA| > 90°)",
            "Loss vs control (kn)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    for row, col, stats_list, color in [
        (1, 1, all_stats, "#2563eb"),
        (1, 2, upwind_stats, "#059669"),
        (2, 1, downwind_stats, "#d97706"),
    ]:
        m = [b.mean_residual_kn for b in stats_list]
        s = [b.std_residual_kn for b in stats_list]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=m,
                error_y=dict(type="data", array=s, visible=True),
                mode="lines+markers",
                name=["All", "Upwind", "Downwind"][row + col - 2 if row == 1 else 0],
                line=dict(color=color),
                marker=dict(size=8),
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

    losses = [b.loss_vs_control_kn for b in all_stats]
    colors = ["#dc2626" if (l is not None and l >= 0.3) else "#94a3b8" for l in losses]
    fig.add_trace(
        go.Bar(x=labels, y=losses, marker_color=colors, name="Loss vs control"),
        row=2,
        col=2,
    )

    title = (
        f"Dirty-Air Speed Penalty Curve — {meta.get('venue', 'Bermuda')} "
        f"({meta.get('distance_source', 'PC_DTB_m')}, n={meta.get('n_rows_analysis', 0):,})"
    )
    fig.update_layout(title=title, height=700, showlegend=True)
    fig.update_yaxes(title_text="Speed residual (kn)", row=1, col=1)
    fig.update_yaxes(title_text="Speed residual (kn)", row=1, col=2)
    fig.update_yaxes(title_text="Speed residual (kn)", row=2, col=1)
    fig.update_yaxes(title_text="Loss (kn)", row=2, col=2)
    return fig


def export_results(
    all_stats: list[BinStats],
    upwind_stats: list[BinStats],
    downwind_stats: list[BinStats],
    decay: dict,
    success: dict,
    meta: dict,
    fig: go.Figure,
) -> tuple[Path, Path, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = EXPORT_DIR / "dirty_air_penalty.html"
    csv_path = EXPORT_DIR / "dirty_air_stats.csv"
    json_path = EXPORT_DIR / "dirty_air_results.json"

    rows = []
    for sail, stats_list in [("all", all_stats), ("upwind", upwind_stats), ("downwind", downwind_stats)]:
        for b in stats_list:
            rows.append({"point_of_sail": sail, **asdict(b)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "experiment": "dirty_air_penalty_curve",
        "meta": meta,
        "bin_stats": {
            "all": [asdict(b) for b in all_stats],
            "upwind": [asdict(b) for b in upwind_stats],
            "downwind": [asdict(b) for b in downwind_stats],
        },
        "decay_fit": decay,
        "success": success,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig.write_html(str(html_path))
    return html_path, csv_path, json_path


def run(venue: str = "Bermuda") -> dict:
    print(f"Exp #3 Dirty-Air Penalty Curve (venue={venue})", flush=True)
    print("Loading data...", flush=True)
    frame, meta = prepare_frame(venue=venue)
    print(f"  {meta['n_rows_analysis']:,} following timesteps", flush=True)
    print(f"  distance source: {meta['distance_source']}, DTB null rate: {meta.get('dtb_null_rate', 'n/a')}", flush=True)

    all_stats = compute_bin_stats(frame)
    upwind = frame[frame["abs_twa"] < 90]
    downwind = frame[frame["abs_twa"] > 90]
    upwind_stats = compute_bin_stats(upwind)
    downwind_stats = compute_bin_stats(downwind)

    decay = fit_decay_curve(all_stats)
    success = evaluate_success(all_stats, upwind_stats, downwind_stats)
    fig = plot_penalty_curve(all_stats, upwind_stats, downwind_stats, decay, meta)
    html_path, csv_path, json_path = export_results(
        all_stats, upwind_stats, downwind_stats, decay, success, meta, fig
    )

    print("\nBin stats (mean residual kn ± std):", flush=True)
    for b in all_stats:
        sig = "*" if b.mwu_p_vs_control < 0.05 and b.label != "200-500m (control)" else ""
        print(
            f"  {b.label:20s} n={b.n:6d}  {b.mean_residual_kn:+.3f} ± {b.std_residual_kn:.3f}  "
            f"p={b.mwu_p_vs_control:.2e}  loss={b.loss_vs_control_kn:+.3f}kn{sig}",
            flush=True,
        )

    print(f"\nDecay fit: {decay}", flush=True)
    print(f"Overall: {'PASS' if success['overall_pass'] else 'FAIL'}", flush=True)
    for name, c in success["criteria"].items():
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'} — {c['detail']}", flush=True)
    print(f"\nWrote {html_path}, {csv_path}, {json_path}", flush=True)
    return success


def main():
    run(venue="Bermuda")


if __name__ == "__main__":
    main()
