#!/usr/bin/env python3
"""Animated map: top-3 finishers vs their ghost boat, 1 Hz playback."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_LEG,
    COL_RANK,
    COL_SPEED,
    COL_TWA,
    COL_TWS,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.fleet import COL_LAT, COL_LON
from dataExploration.next_experiments.exp_4_ghost_boat import (
    COL_TIME_LEG,
    COL_TWD,
    build_polar_table,
    infer_mark_positions,
    simulate_ghost_leg,
)
from sailgp_analysis.config import DATA_ROOT

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
COL_TIME_RACE = "TIME_RACE_s"
TOP_N = 3

TEAM_COLORS = {
    "AUS": "#FFD700",
    "GBR": "#0066CC",
    "USA": "#CC0000",
    "ESP": "#FF6600",
    "FRA": "#003399",
    "NZL": "#000000",
    "CAN": "#FF0000",
    "DEN": "#C8102E",
    "GER": "#000000",
    "SUI": "#FF0000",
    "BRA": "#009C3B",
    "ITA": "#008C45",
    "SWE": "#006AA7",
}


def _color(team: str) -> str:
    return TEAM_COLORS.get(team, "#AAAAAA")


def top_finishers(race_df: pd.DataFrame, n: int = TOP_N) -> list[str]:
    ranks = (
        race_df.dropna(subset=[COL_RANK])
        .sort_index()
        .groupby("team", sort=False)[COL_RANK]
        .last()
        .astype(float)
    )
    return ranks.nsmallest(n).index.tolist()


def pick_race(df: pd.DataFrame, venue: str | None, race_label: str | None) -> tuple[str, str]:
    if venue and race_label:
        return venue, race_label
    counts = df.groupby(["venue", "race_label"], sort=False).size().reset_index(name="n")
    row = counts.sort_values("n", ascending=False).iloc[0]
    return str(row["venue"]), str(row["race_label"])


def ghost_track_race_time(
    team_df: pd.DataFrame,
    mark_lookup: dict[tuple[str, str, int], tuple[float, float]],
    venue: str,
    race_label: str,
) -> pd.DataFrame:
    """Ghost positions on race clock: (race_t, lat, lon)."""
    rows: list[dict] = []
    team_df = team_df.sort_index()

    for leg, leg_df in team_df.groupby(COL_LEG, sort=False):
        leg = int(leg)
        if leg <= 0 or len(leg_df) < 5:
            continue
        key = (venue, race_label, leg)
        if key not in mark_lookup:
            continue

        mark_lat, mark_lon = mark_lookup[key]
        start = leg_df.iloc[0]
        leg_start_t = float(start.get(COL_TIME_RACE, 0.0))
        wind_cols = [c for c in [COL_TWS, COL_TWD] if c in leg_df.columns]
        _, ghost_track = simulate_ghost_leg(
            float(start[COL_LAT]),
            float(start[COL_LON]),
            mark_lat,
            mark_lon,
            leg_df[wind_cols],
        )
        for t_local, lat, lon in ghost_track:
            rows.append({"race_t": leg_start_t + t_local, "lat": lat, "lon": lon})

    return pd.DataFrame(rows)


def actual_track(team_df: pd.DataFrame) -> pd.DataFrame:
    sub = team_df.dropna(subset=[COL_LAT, COL_LON, COL_TIME_RACE]).copy()
    sub["race_t"] = sub[COL_TIME_RACE].astype(float)
    return sub[["race_t", COL_LAT, COL_LON, COL_RANK]].rename(
        columns={COL_LAT: "lat", COL_LON: "lon", COL_RANK: "rank"}
    )


def position_at_time(track: pd.DataFrame, t: float) -> tuple[float, float] | None:
    if track.empty:
        return None
    hit = track[track["race_t"] <= t]
    if hit.empty:
        return None
    row = hit.iloc[-1]
    return float(row["lat"]), float(row["lon"])


def build_animation(
    df: pd.DataFrame,
    venue: str,
    race_label: str,
    top_teams: list[str],
    out_path: Path,
) -> dict:
    race_df = df[(df["venue"] == venue) & (df["race_label"] == race_label)].copy()
    marks = infer_mark_positions(race_df)
    mark_lookup = {
        (r.venue, r.race_label, int(r.leg)): (r.mark_lat, r.mark_lon) for r in marks.itertuples()
    }
    mark_rows = marks[(marks["venue"] == venue) & (marks["race_label"] == race_label)]

    actual: dict[str, pd.DataFrame] = {}
    ghost: dict[str, pd.DataFrame] = {}
    for team in top_teams:
        tdf = race_df[race_df["team"] == team]
        actual[team] = actual_track(tdf)
        ghost[team] = ghost_track_race_time(tdf, mark_lookup, venue, race_label)

    t_min = min(a["race_t"].min() for a in actual.values() if not a.empty)
    t_max = max(a["race_t"].max() for a in actual.values() if not a.empty)
    times = list(range(int(t_min), int(t_max) + 1))

    center_lat = float(race_df[COL_LAT].median())
    center_lon = float(race_df[COL_LON].median())

    # Static full paths (always visible) + animated markers (fixed trace slots for Plotly frames).
    static_traces: list[go.Scattermap] = []
    for team in top_teams:
        color = _color(team)
        act = actual[team]
        gh = ghost[team]
        static_traces.append(
            go.Scattermap(
                lat=act["lat"].tolist(),
                lon=act["lon"].tolist(),
                mode="lines",
                line={"width": 4, "color": color},
                opacity=0.75,
                name=f"{team} actual (full)",
                legendgroup=team,
            )
        )
        static_traces.append(
            go.Scattermap(
                lat=gh["lat"].tolist(),
                lon=gh["lon"].tolist(),
                mode="lines",
                line={"width": 2, "color": "#FFFFFF"},
                opacity=0.4,
                name=f"{team} ghost (full)",
                legendgroup=team,
            )
        )

    if not mark_rows.empty:
        static_traces.append(
            go.Scattermap(
                lat=mark_rows["mark_lat"].tolist(),
                lon=mark_rows["mark_lon"].tolist(),
                mode="markers",
                marker={"size": 12, "symbol": "star", "color": "#FFFFFF"},
                name="Marks",
            )
        )

    n_static = len(static_traces)
    marker_start = n_static
    marker_traces: list[go.Scattermap] = []
    marker_indices: list[int] = []

    for team in top_teams:
        color = _color(team)
        act_pos = position_at_time(actual[team], float(times[0])) or (center_lat, center_lon)
        gh_pos = position_at_time(ghost[team], float(times[0])) or act_pos

        marker_indices.append(len(static_traces) + len(marker_traces))
        marker_traces.append(
            go.Scattermap(
                lat=[act_pos[0]],
                lon=[act_pos[1]],
                mode="markers+text",
                text=[team],
                textfont={"size": 13, "color": "#FFFFFF"},
                textposition="top center",
                marker={"size": 18, "symbol": "circle", "color": color, "allowoverlap": True},
                name=f"{team} actual",
                legendgroup=team,
            )
        )

        marker_indices.append(len(static_traces) + len(marker_traces))
        marker_traces.append(
            go.Scattermap(
                lat=[gh_pos[0]],
                lon=[gh_pos[1]],
                mode="markers",
                marker={"size": 14, "symbol": "diamond", "color": color, "allowoverlap": True},
                name=f"{team} ghost",
                legendgroup=team,
            )
        )

    frames: list[go.Frame] = []
    for t in times:
        frame_data: list[go.Scattermap] = []
        title = f"Top {len(top_teams)} vs Ghost — {venue} {race_label} — {t}s"
        for team in top_teams:
            act_pos = position_at_time(actual[team], float(t))
            gh_pos = position_at_time(ghost[team], float(t))
            if act_pos is None:
                act_pos = (center_lat, center_lon)
            if gh_pos is None:
                gh_pos = act_pos
            frame_data.append(go.Scattermap(lat=[act_pos[0]], lon=[act_pos[1]]))
            frame_data.append(go.Scattermap(lat=[gh_pos[0]], lon=[gh_pos[1]]))
        frames.append(
            go.Frame(
                data=frame_data,
                traces=marker_indices,
                name=str(t),
                layout={"title": {"text": title}},
            )
        )

    fig = go.Figure(data=static_traces + marker_traces, frames=frames)
    duration_ms = max(50, min(500, 120_000 // max(len(times), 1)))
    fig.update_layout(
        title=f"Top {len(top_teams)} vs Ghost — {venue} {race_label} — {times[0]}s",
        template="plotly_dark",
        height=850,
        map={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 13,
        },
        legend={"orientation": "h", "y": 1.04, "x": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration_ms, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.05,
                "len": 0.9,
                "xanchor": "left",
                "y": -0.02,
                "steps": [
                    {
                        "label": str(t),
                        "method": "animate",
                        "args": [
                            [str(t)],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    }
                    for t in times[:: max(1, len(times) // 40)]
                ],
            }
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", auto_play=False)

    return {
        "venue": venue,
        "race_label": race_label,
        "top_teams": top_teams,
        "n_frames": len(times),
        "race_duration_s": int(t_max - t_min),
        "n_static_traces": n_static,
        "n_marker_traces": len(marker_traces),
        "output": str(out_path),
    }


def run(venue: str | None = None, race_label: str | None = None) -> dict:
    df = load_racing_boats(DATA_ROOT)
    needed = [COL_LAT, COL_LON, COL_TWA, COL_TWS, COL_SPEED, COL_LEG, COL_TIME_LEG, COL_TIME_RACE, COL_RANK]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    polar, global_speed = build_polar_table(df)
    simulate_ghost_leg.polar = (polar, global_speed)  # type: ignore[attr-defined]

    venue, race_label = pick_race(df, venue, race_label)
    race_df = df[(df["venue"] == venue) & (df["race_label"] == race_label)]
    top_teams = top_finishers(race_df, TOP_N)
    if len(top_teams) < TOP_N:
        raise RuntimeError(f"Only {len(top_teams)} teams with rank in {venue} {race_label}")

    out_path = EXPORT_DIR / "ghost_boat_race_animation.html"
    meta = build_animation(df, venue, race_label, top_teams, out_path)

    results = {
        "experiment": "ghost_boat_race_animation",
        **meta,
        "legend": {
            "circle": "actual boat (top 3 finishers)",
            "diamond": "ghost boat (polar-optimal path using that boat's wind)",
            "star": "inferred mark positions",
        },
    }
    with open(EXPORT_DIR / "ghost_boat_race_animation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-3 vs ghost race animation")
    parser.add_argument("--venue", default=None)
    parser.add_argument("--race", default=None, dest="race_label")
    args = parser.parse_args()
    res = run(args.venue, args.race_label)
    print(f"Wrote {res['output']}")
    print(f"Race: {res['venue']} {res['race_label']} | Top 3: {res['top_teams']}")
    print(f"Frames: {res['n_frames']} ({res['race_duration_s']}s race)")


if __name__ == "__main__":
    main()
