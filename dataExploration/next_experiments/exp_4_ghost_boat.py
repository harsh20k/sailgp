#!/usr/bin/env python3
"""Experiment #4 — Ghost Boat Regret: optimal polar+wind path vs actual leg times."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_LEG,
    COL_RANK,
    COL_SPEED,
    COL_STATUS,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.fleet import COL_LAT, COL_LON, haversine_m
from sailgp_analysis.config import DATA_ROOT, VENUES

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
COL_TWD = "TWD_SGP_deg"
COL_YAW = "RATE_YAW_deg_s_1"
COL_TIME_LEG = "TIME_RACE_LEG_s"

TWA_BIN = 10.0
TWS_BIN = 4.0  # km/h (~2 kn)
MIN_POLAR_COUNT = 20
MARK_ARRIVAL_M = 40.0
MAX_LEG_S = 600.0
CANDIDATE_TWA = np.arange(20.0, 165.0, 5.0)
EARTH_R = 6_371_000.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def offset_latlon(lat: float, lon: float, brng_deg: float, dist_m: float) -> tuple[float, float]:
    brng = math.radians(brng_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    ang = dist_m / EARTH_R
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def build_polar_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=[COL_TWA, COL_TWS, COL_SPEED]).copy()
    sub = sub[sub[COL_TWS] > 2.0]
    sub["twa_abs"] = sub[COL_TWA].abs()
    sub["twa_bin"] = np.floor(sub["twa_abs"] / TWA_BIN) * TWA_BIN
    sub["tws_bin"] = np.floor(sub[COL_TWS] / TWS_BIN) * TWS_BIN
    agg = (
        sub.groupby(["twa_bin", "tws_bin"], as_index=False)
        .agg(speed=(COL_SPEED, "mean"), vmg=(COL_VMG, "mean"), n=(COL_SPEED, "count"))
    )
    global_speed = float(sub[COL_SPEED].mean())
    return agg, global_speed


def polar_speed(twa_abs: float, tws: float, polar: pd.DataFrame, global_speed: float) -> float:
    twa_bin = float(np.floor(twa_abs / TWA_BIN) * TWA_BIN)
    tws_bin = float(np.floor(tws / TWS_BIN) * TWS_BIN)
    hit = polar[(polar["twa_bin"] == twa_bin) & (polar["tws_bin"] == tws_bin)]
    if len(hit) and hit.iloc[0]["n"] >= MIN_POLAR_COUNT:
        return float(hit.iloc[0]["speed"])
    near = polar[(polar["twa_bin"] == twa_bin) | (polar["tws_bin"] == tws_bin)]
    if len(near):
        w = near["n"].to_numpy()
        return float(np.average(near["speed"], weights=w + 1))
    return global_speed


def optimal_heading(twd: float, tws: float, mark_bearing: float, polar: pd.DataFrame, global_speed: float) -> tuple[float, float]:
    best_vmg = -1.0
    best_heading = mark_bearing
    best_speed = 0.0
    for twa in CANDIDATE_TWA:
        speed = polar_speed(twa, tws, polar, global_speed)
        for sign in (1.0, -1.0):
            heading = (twd + sign * twa) % 360.0
            vmg = speed * math.cos(math.radians(angle_diff_deg(heading, mark_bearing)))
            if vmg > best_vmg:
                best_vmg = vmg
                best_heading = heading
                best_speed = speed
    return best_heading, best_speed


def infer_mark_positions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (venue, race), gdf in df.groupby(["venue", "race_label"], sort=False):
        for leg, leg_df in gdf.groupby(COL_LEG, sort=False):
            if leg <= 0:
                continue
            ends = (
                leg_df.groupby("team", sort=False)
                .tail(1)[[COL_LAT, COL_LON]]
                .dropna()
            )
            if ends.empty:
                continue
            rows.append(
                {
                    "venue": venue,
                    "race_label": race,
                    "leg": int(leg),
                    "mark_lat": float(ends[COL_LAT].median()),
                    "mark_lon": float(ends[COL_LON].median()),
                }
            )
    return pd.DataFrame(rows)


def simulate_ghost_leg(
    start_lat: float,
    start_lon: float,
    mark_lat: float,
    mark_lon: float,
    wind_df: pd.DataFrame,
) -> tuple[float, list[tuple[float, float, float]]]:
    """Return ghost leg duration (s) and track [(t, lat, lon), ...]."""
    lat, lon = start_lat, start_lon
    track: list[tuple[float, float, float]] = [(0.0, lat, lon)]
    elapsed = 0.0
    polar, global_speed = simulate_ghost_leg.polar  # type: ignore[attr-defined]

    for _, row in wind_df.iterrows():
        if elapsed >= MAX_LEG_S:
            break
        dist_m = haversine_m(
            np.array([lat]),
            np.array([lon]),
            np.array([mark_lat]),
            np.array([mark_lon]),
        )[0]
        if dist_m <= MARK_ARRIVAL_M:
            break

        tws = float(row.get(COL_TWS, np.nan))
        twd = float(row.get(COL_TWD, np.nan))
        if np.isnan(tws) or np.isnan(twd) or tws < 2.0:
            elapsed += 1.0
            track.append((elapsed, lat, lon))
            continue

        brg = bearing_deg(lat, lon, mark_lat, mark_lon)
        heading, speed = optimal_heading(twd, tws, brg, polar, global_speed)
        step_m = speed * (1000.0 / 3600.0)
        lat, lon = offset_latlon(lat, lon, heading, step_m)
        elapsed += 1.0
        track.append((elapsed, lat, lon))

    return elapsed, track


def extract_legs(df: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mark_lookup = {
        (r.venue, r.race_label, r.leg): (r.mark_lat, r.mark_lon)
        for r in marks.itertuples(index=False)
    }

    for (venue, race, team), gdf in df.groupby(["venue", "race_label", "team"], sort=False):
        gdf = gdf.sort_index()
        finish_rank = float(gdf[COL_RANK].dropna().iloc[-1]) if COL_RANK in gdf.columns and gdf[COL_RANK].notna().any() else np.nan

        for leg, leg_df in gdf.groupby(COL_LEG, sort=False):
            if leg <= 0 or len(leg_df) < 5:
                continue
            key = (venue, race, int(leg))
            if key not in mark_lookup:
                continue

            mark_lat, mark_lon = mark_lookup[key]
            start = leg_df.iloc[0]
            actual_time = float(leg_df[COL_TIME_LEG].max())
            if np.isnan(actual_time) or actual_time <= 0:
                continue

            wind_cols = [c for c in [COL_TWS, COL_TWD] if c in leg_df.columns]
            ghost_time, ghost_track = simulate_ghost_leg(
                float(start[COL_LAT]),
                float(start[COL_LON]),
                mark_lat,
                mark_lon,
                leg_df[wind_cols],
            )
            regret = actual_time - ghost_time

            rows.append(
                {
                    "venue": venue,
                    "race_label": race,
                    "team": team,
                    "leg": int(leg),
                    "actual_leg_s": actual_time,
                    "ghost_leg_s": ghost_time,
                    "regret_s": regret,
                    "finish_rank": finish_rank,
                    "mark_lat": mark_lat,
                    "mark_lon": mark_lon,
                    "start_lat": float(start[COL_LAT]),
                    "start_lon": float(start[COL_LON]),
                    "ghost_track": ghost_track,
                }
            )

    return pd.DataFrame(rows)


def spatial_regret_track(
    leg_df: pd.DataFrame,
    ghost_track: list[tuple[float, float, float]],
    mark_lat: float,
    mark_lon: float,
) -> pd.DataFrame:
    """Cumulative regret: actual elapsed minus ghost time to same distance-to-mark."""
    if not ghost_track:
        return pd.DataFrame()

    ghost_t = np.array([p[0] for p in ghost_track])
    ghost_dist = np.array(
        [
            haversine_m(
                np.array([p[1]]),
                np.array([p[2]]),
                np.array([mark_lat]),
                np.array([mark_lon]),
            )[0]
            for p in ghost_track
        ]
    )

    out_rows = []
    t0 = float(leg_df[COL_TIME_LEG].iloc[0]) if COL_TIME_LEG in leg_df.columns else 0.0

    for _, row in leg_df.iterrows():
        actual_elapsed = float(row.get(COL_TIME_LEG, 0.0)) - t0
        d_actual = haversine_m(
            np.array([float(row[COL_LAT])]),
            np.array([float(row[COL_LON])]),
            np.array([mark_lat]),
            np.array([mark_lon]),
        )[0]
        # Ghost time when it was at least as close to the mark as the actual boat now is.
        closer = ghost_dist <= d_actual
        ghost_at = float(ghost_t[closer][0]) if closer.any() else float(ghost_t[-1])
        out_rows.append(
            {
                "lat": float(row[COL_LAT]),
                "lon": float(row[COL_LON]),
                "cum_regret_s": actual_elapsed - ghost_at,
                "leg_elapsed_s": actual_elapsed,
                "yaw_rate": float(row.get(COL_YAW, 0.0)),
            }
        )
    return pd.DataFrame(out_rows)


def evaluate_success(leg_df: pd.DataFrame, spatial_df: pd.DataFrame) -> dict:
    # Ghost sanity: median ghost faster than median actual per race-leg
    sanity = leg_df.groupby(["venue", "race_label", "leg"]).apply(
        lambda g: float(g["ghost_leg_s"].median()) <= float(g["actual_leg_s"].median())
    )
    ghost_sanity_pct = float(sanity.mean()) if len(sanity) else 0.0

    # Regret spread per leg
    spread = leg_df.groupby(["venue", "race_label", "leg"])["regret_s"].agg(lambda s: float(s.max() - s.min()))
    regret_spread_median = float(spread.median()) if len(spread) else 0.0

    # Spearman: total regret rank vs finish rank per race
    rhos = []
    for (_, _), g in leg_df.groupby(["venue", "race_label"], sort=False):
        totals = g.groupby("team", as_index=False).agg(total_regret=("regret_s", "sum"), finish_rank=("finish_rank", "first"))
        totals = totals.dropna(subset=["finish_rank"])
        if len(totals) < 4:
            continue
        rho, _ = stats.spearmanr(totals["total_regret"], totals["finish_rank"])
        if not np.isnan(rho):
            rhos.append(float(rho))
    spearman_rho = float(np.mean(rhos)) if rhos else 0.0

    # Spatial interpretability: boundary + tack regret concentration
    spatial_score = 0.0
    if not spatial_df.empty and "is_boundary" in spatial_df.columns:
        q75 = spatial_df["cum_regret_s"].quantile(0.75)
        hi = spatial_df["cum_regret_s"] >= q75
        boundary_hi = (spatial_df["is_boundary"] & hi).sum() / max(spatial_df["is_boundary"].sum(), 1)
        tack_hi = (spatial_df["is_tack"] & hi).sum() / max(spatial_df["is_tack"].sum(), 1)
        mid_hi = (spatial_df["is_mid"] & hi).sum() / max(spatial_df["is_mid"].sum(), 1)
        spatial_score = float(boundary_hi + tack_hi - mid_hi)

    criteria = {
        "ghost_sanity_pct": {
            "value": ghost_sanity_pct,
            "threshold": 0.80,
            "pass": ghost_sanity_pct >= 0.80,
        },
        "regret_spread_median_s": {
            "value": regret_spread_median,
            "threshold": 15.0,
            "pass": regret_spread_median >= 15.0,
        },
        "spearman_rho": {
            "value": spearman_rho,
            "threshold": 0.40,
            "pass": spearman_rho >= 0.40,
        },
        "spatial_interpretability": {
            "value": spatial_score,
            "threshold": 0.10,
            "pass": spatial_score >= 0.10,
        },
    }
    criteria["overall_pass"] = all(c["pass"] for c in criteria.values() if isinstance(c, dict))
    return criteria


def build_spatial_frame(df: pd.DataFrame, leg_results: pd.DataFrame) -> pd.DataFrame:
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
        n = len(track)
        track["is_boundary"] = False
        track.loc[track.index[: max(1, n // 10)], "is_boundary"] = True
        track.loc[track.index[-max(1, n // 10) :], "is_boundary"] = True
        track["is_tack"] = track["yaw_rate"].abs() >= track["yaw_rate"].abs().quantile(0.90)
        track["is_mid"] = ~track["is_boundary"] & ~track["is_tack"]
        track["venue"] = row.venue
        track["race_label"] = row.race_label
        track["team"] = row.team
        track["leg"] = row.leg
        frames.append(track)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_map(spatial_df: pd.DataFrame, leg_df: pd.DataFrame, out_path: Path) -> None:
    if spatial_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ghost Boat Regret — no spatial data")
        fig.write_html(out_path, include_plotlyjs="cdn")
        return

    # Pick the race with the most GPS points (one venue only — Bermuda/Halifax are ~700 mi apart).
    best = (
        spatial_df.groupby(["venue", "race_label"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .iloc[0]
    )
    sample = spatial_df[
        (spatial_df["venue"] == best["venue"]) & (spatial_df["race_label"] == best["race_label"])
    ].copy()
    center_lat = float(sample["lat"].median())
    center_lon = float(sample["lon"].median())
    label = f"{best['venue']} {best['race_label']}"

    fig = px.scatter_map(
        sample,
        lat="lat",
        lon="lon",
        color="cum_regret_s",
        hover_data=["venue", "race_label", "team", "leg", "cum_regret_s"],
        color_continuous_scale="RdYlGn_r",
        zoom=12,
        height=800,
        title=f"Ghost Boat Cumulative Regret — {label}",
    )
    fig.update_traces(marker={"size": 7})
    fig.update_layout(
        map_style="open-street-map",
        template="plotly_dark",
        map={"center": {"lat": center_lat, "lon": center_lon}, "zoom": 12},
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_racing_boats(DATA_ROOT)
    if df.empty:
        raise RuntimeError("No racing data loaded")

    needed = [COL_LAT, COL_LON, COL_TWA, COL_TWS, COL_SPEED, COL_LEG, COL_TIME_LEG]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    polar, global_speed = build_polar_table(df)
    simulate_ghost_leg.polar = (polar, global_speed)  # type: ignore[attr-defined]

    marks = infer_mark_positions(df)
    leg_results = extract_legs(df, marks)
    if leg_results.empty:
        raise RuntimeError("No legs extracted")

    spatial_df = build_spatial_frame(df, leg_results)
    criteria = evaluate_success(leg_results, spatial_df)

    out_csv = leg_results.drop(columns=["ghost_track"]).copy()
    out_csv.to_csv(EXPORT_DIR / "ghost_boat_regret.csv", index=False)

    build_map(spatial_df, leg_results, EXPORT_DIR / "ghost_boat_map.html")

    ranking = (
        leg_results.groupby(["venue", "race_label", "team"], as_index=False)
        .agg(total_regret_s=("regret_s", "sum"), finish_rank=("finish_rank", "first"), legs=("leg", "count"))
        .sort_values(["venue", "race_label", "total_regret_s"])
    )

    results = {
        "experiment": "ghost_boat_regret",
        "venues": VENUES,
        "n_legs": int(len(leg_results)),
        "n_races": int(leg_results[["venue", "race_label"]].drop_duplicates().shape[0]),
        "n_teams": int(leg_results["team"].nunique()),
        "polar_bins": int(len(polar)),
        "success_criteria": criteria,
        "overall_pass": bool(criteria["overall_pass"]),
        "summary": {
            "mean_regret_s": float(leg_results["regret_s"].mean()),
            "median_regret_s": float(leg_results["regret_s"].median()),
            "regret_std_s": float(leg_results["regret_s"].std()),
        },
        "ranking_head": ranking.head(20).to_dict(orient="records"),
        "outputs": {
            "csv": str(EXPORT_DIR / "ghost_boat_regret.csv"),
            "map_html": str(EXPORT_DIR / "ghost_boat_map.html"),
            "json": str(EXPORT_DIR / "ghost_boat_results.json"),
        },
    }

    with open(EXPORT_DIR / "ghost_boat_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    crit = res["success_criteria"]
    print(f"Overall: {'PASS' if res['overall_pass'] else 'FAIL'}")
    for name, c in crit.items():
        if not isinstance(c, dict):
            continue
        print(f"  {name}: {c['value']:.3f} (need {c['threshold']}) -> {'PASS' if c['pass'] else 'FAIL'}")
    print(f"Legs: {res['n_legs']} | Outputs: {res['outputs']}")
