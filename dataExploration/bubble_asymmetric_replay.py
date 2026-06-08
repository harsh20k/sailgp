#!/usr/bin/env python3
"""Animated 200 m asymmetric tactical bubble — boat near stern, front = heading."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse race helpers
from race_replay import (
    PLOTLY_LAYOUT,
    VENUE_CENTER,
    course_limit_traces,
    find_data_root,
    haversine_m,
    load_course_limits,
    load_race_boats,
    offset_latlon,
    team_color_map,
)

# Asymmetric bubble: 200 m radius circle with centre shifted forward so the boat
# sits near the aft edge (front of bubble = boat heading).
BUBBLE_RADIUS_M = 200.0
CENTER_FORWARD_M = 200.0  # centre one radius ahead → boat at rear of circle
STATS_RADII_M = list(range(100, 301, 50))  # 100, 150, 200, 250, 300
BUBBLE_FILL = "rgba(79, 142, 247, 0.18)"
BUBBLE_LINE = "#4f8ef7"
SAMPLE_TEAM = "GBR"
PANELS = [
    ("sample", f"Sample boat ({SAMPLE_TEAM})"),
    ("rank1", "Rank #1 (live)"),
    ("rank2", "Rank #2 (live)"),
    ("rank3", "Rank #3 (live)"),
]


def ego_offset(
    lat: float,
    lon: float,
    heading_deg: float,
    forward_m: float,
    starboard_m: float,
) -> tuple[float, float]:
    """Move in ego frame: +forward along heading, +starboard to the right."""
    lat1, lon1 = offset_latlon(lat, lon, heading_deg, forward_m)
    return offset_latlon(lat1, lon1, (heading_deg + 90.0) % 360.0, starboard_m)


def asymmetric_bubble_ring(
    lat: float,
    lon: float,
    heading_deg: float,
    *,
    radius_m: float = BUBBLE_RADIUS_M,
    center_forward_m: float = CENTER_FORWARD_M,
    n: int = 64,
) -> tuple[list[float], list[float]]:
    """Closed lat/lon ring for the asymmetric bubble polygon."""
    lats: list[float] = []
    lons: list[float] = []
    for phi in np.linspace(0, 2 * math.pi, n, endpoint=False):
        ex = center_forward_m + radius_m * math.cos(phi)
        ey = radius_m * math.sin(phi)
        la, lo = ego_offset(lat, lon, heading_deg, ex, ey)
        lats.append(la)
        lons.append(lo)
    lats.append(lats[0])
    lons.append(lons[0])
    return lats, lons


def bubble_center(
    lat: float,
    lon: float,
    heading_deg: float,
    *,
    radius_m: float = BUBBLE_RADIUS_M,
) -> tuple[float, float]:
    """Centre of the asymmetric bubble (one radius ahead of ego)."""
    return ego_offset(lat, lon, heading_deg, radius_m, 0.0)


def count_boats_in_bubble(
    snap: pd.DataFrame,
    elat: float,
    elon: float,
    heading: float,
    *,
    radius_m: float = BUBBLE_RADIUS_M,
) -> int:
    """Count all boats whose position lies inside the bubble disc."""
    clat, clon = bubble_center(elat, elon, heading, radius_m=radius_m)
    count = 0
    for _, row in snap.iterrows():
        blat = float(row["LATITUDE_GPS_unk"])
        blon = float(row["LONGITUDE_GPS_unk"])
        if haversine_m(clat, clon, blat, blon) <= radius_m:
            count += 1
    return count


def resolve_panel_team(panel_key: str, snap: pd.DataFrame, sample_team: str) -> str | None:
    if panel_key == "sample":
        row = snap[snap["team"] == sample_team]
        return sample_team if not row.empty else None
    rank = {"rank1": 1, "rank2": 2, "rank3": 3}[panel_key]
    ranked = snap[snap["TRK_RACE_RANK_unk"] == rank]
    if ranked.empty:
        return None
    return str(ranked.iloc[0]["team"])


def panel_traces(
    boats: pd.DataFrame,
    timestamp,
    panel_key: str,
    colors: dict[str, str],
    sample_team: str,
    boundary_traces: list,
    row: int,
    col: int,
) -> list[go.Scattermap]:
    snap = boats[boats["DATETIME"] == timestamp]
    traces: list[go.Scattermap] = list(boundary_traces)

    ego_team = resolve_panel_team(panel_key, snap, sample_team)
    if ego_team is None:
        clat = float(snap["LATITUDE_GPS_unk"].mean())
        clon = float(snap["LONGITUDE_GPS_unk"].mean())
        for _ in range(5):  # bubble, outside, inside, ego, count label
            traces.append(
                go.Scattermap(
                    lat=[clat],
                    lon=[clon],
                    mode="markers",
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                )
            )
        return traces

    ego = snap[snap["team"] == ego_team].iloc[0]
    elat = float(ego["LATITUDE_GPS_unk"])
    elon = float(ego["LONGITUDE_GPS_unk"])
    heading = float(ego.get("HEADING_deg", ego.get("GPS_COG_deg", 0.0)))
    rank = int(ego["TRK_RACE_RANK_unk"]) if pd.notna(ego.get("TRK_RACE_RANK_unk")) else "?"
    n_in_bubble = count_boats_in_bubble(snap, elat, elon, heading, radius_m=BUBBLE_RADIUS_M)
    clat, clon = bubble_center(elat, elon, heading, radius_m=BUBBLE_RADIUS_M)

    # Bubble polygon
    blats, blons = asymmetric_bubble_ring(elat, elon, heading)
    traces.append(
        go.Scattermap(
            lat=blats,
            lon=blons,
            mode="lines",
            fill="toself",
            fillcolor=BUBBLE_FILL,
            line=dict(width=2, color=BUBBLE_LINE),
            name="Bubble",
            hovertemplate=(
                f"<b>{ego_team}</b> bubble<br>"
                f"200 m asymmetric · rank {rank}<br>"
                f"{n_in_bubble} boats in bubble<br>"
                f"heading {heading:.0f}°<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Boats inside vs outside bubble
    others = snap[snap["team"] != ego_team]
    inside_lats, inside_lons, inside_teams = [], [], []
    outside_lats, outside_lons, outside_teams = [], [], []
    for _, row in others.iterrows():
        blat = float(row["LATITUDE_GPS_unk"])
        blon = float(row["LONGITUDE_GPS_unk"])
        team = str(row["team"])
        if haversine_m(clat, clon, blat, blon) <= BUBBLE_RADIUS_M:
            inside_lats.append(blat)
            inside_lons.append(blon)
            inside_teams.append(team)
        else:
            outside_lats.append(blat)
            outside_lons.append(blon)
            outside_teams.append(team)

    traces.append(
        go.Scattermap(
            lat=outside_lats if outside_lats else [elat],
            lon=outside_lons if outside_lons else [elon],
            mode="markers",
            marker=dict(
                size=[7] * len(outside_lats) if outside_lats else [0],
                color="#666666",
                opacity=0.55 if outside_lats else 0,
            ),
            text=outside_teams if outside_teams else None,
            hovertemplate="%{text}<extra></extra>" if outside_teams else None,
            showlegend=False,
        )
    )

    traces.append(
        go.Scattermap(
            lat=inside_lats if inside_lats else [elat],
            lon=inside_lons if inside_lons else [elon],
            mode="markers",
            marker=dict(
                size=[10] * len(inside_lats) if inside_lats else [0],
                color="#f47b1f",
                opacity=0.9 if inside_lats else 0,
            ),
            text=inside_teams if inside_teams else None,
            hovertemplate="%{text} (in bubble)<extra></extra>" if inside_teams else None,
            showlegend=False,
        )
    )

    # Ego boat + heading tick
    tip_lat, tip_lon = offset_latlon(elat, elon, heading, 18.0)
    traces.append(
        go.Scattermap(
            lat=[elat, tip_lat],
            lon=[elon, tip_lon],
            mode="lines+markers",
            line=dict(width=3, color=colors.get(ego_team, "#ffffff")),
            marker=dict(size=14, color=colors.get(ego_team, "#4f8ef7"), symbol="circle"),
            name=ego_team,
            text=[f"{ego_team} (P{rank})", ""],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # Boat count label at bubble centre
    count_label = f"{n_in_bubble} boat{'s' if n_in_bubble != 1 else ''}"
    traces.append(
        go.Scattermap(
            lat=[clat],
            lon=[clon],
            mode="markers+text",
            marker=dict(size=22, color="rgba(15,23,42,0.75)", opacity=0.9),
            text=[count_label],
            textfont=dict(size=13, color="#ffffff", family="IBM Plex Mono, monospace"),
            textposition="middle center",
            hovertemplate=f"<b>{count_label}</b> in bubble<extra></extra>",
            showlegend=False,
        )
    )
    return traces


def build_figure(
    boats: pd.DataFrame,
    course_limits: list[dict],
    *,
    venue: str,
    race: str,
    step: int,
    sample_team: str,
) -> go.Figure:
    times = sorted(boats["DATETIME"].unique())[::step]
    if len(times) < 2:
        raise ValueError("Need at least two timestamps.")

    teams = sorted(boats["team"].unique())
    colors = team_color_map(teams)
    boundary = course_limit_traces(course_limits)
    center = VENUE_CENTER.get(
        venue,
        {
            "lat": float(boats["LATITUDE_GPS_unk"].mean()),
            "lon": float(boats["LONGITUDE_GPS_unk"].mean()),
            "zoom": 15,
        },
    )
    center = {**center, "zoom": 14}

    subplot_titles = [label for _, label in PANELS]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scattermap"}, {"type": "scattermap"}],
               [{"type": "scattermap"}, {"type": "scattermap"}]],
        subplot_titles=subplot_titles,
        vertical_spacing=0.07,
    )

    def traces_for_time(ts):
        result: list[tuple[go.Scattermap, int, int]] = []
        for pi, (pkey, _) in enumerate(PANELS):
            row, col = positions[pi]
            for tr in panel_traces(boats, ts, pkey, colors, sample_team, boundary, row, col):
                result.append((tr, row, col))
        return result

    for tr, row, col in traces_for_time(times[0]):
        fig.add_trace(tr, row=row, col=col)

    fig.frames = [
        go.Frame(data=[tr for tr, _, _ in traces_for_time(ts)], name=str(ts))
        for ts in times
    ]

    slider_steps = []
    for i, t in enumerate(times):
        label = pd.Timestamp(t).strftime("%H:%M:%S")
        slider_steps.append(
            dict(
                method="animate",
                args=[[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=label if i % max(1, len(times) // 10) == 0 else "",
            )
        )

    frame_ms = max(150, int(1000 / max(1, step)))
    map_cfg = dict(
        style="carto-darkmatter",
        center=dict(lat=center["lat"], lon=center["lon"]),
        zoom=center["zoom"],
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(
            f"{venue} {race} — 200 m asymmetric bubble "
            f"(boat at stern · front = heading)"
        ),
        height=980,
        margin=dict(l=0, r=0, t=56, b=0),
        map=map_cfg,
        map2=map_cfg,
        map3=map_cfg,
        map4=map_cfg,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.1,
                y=-0.02,
                len=0.85,
                xanchor="left",
                pad=dict(t=20),
                currentvalue=dict(prefix="Race time: ", visible=True),
                steps=slider_steps,
            )
        ],
        annotations=[
            dict(
                text=(
                    "Blue polygon = 200 m radius · orange dots = rivals inside bubble · "
                    "centre label = boat count · Rank panels follow live P1/P2/P3"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.08,
                showarrow=False,
                font=dict(size=11, color="#888"),
            )
        ],
    )
    return fig


def build_occupancy_timeline(
    boats: pd.DataFrame,
    sample_team: str,
    *,
    radius_m: float = BUBBLE_RADIUS_M,
) -> pd.DataFrame:
    """Per-second boat counts in each ego bubble across the race."""
    rows: list[dict] = []
    for ts, snap in boats.groupby("DATETIME", sort=True):
        snap = snap.copy()
        for pkey, plabel in PANELS:
            ego_team = resolve_panel_team(pkey, snap, sample_team)
            if ego_team is None:
                continue
            ego = snap[snap["team"] == ego_team].iloc[0]
            elat = float(ego["LATITUDE_GPS_unk"])
            elon = float(ego["LONGITUDE_GPS_unk"])
            heading = float(ego.get("HEADING_deg", ego.get("GPS_COG_deg", 0.0)))
            n_boats = count_boats_in_bubble(snap, elat, elon, heading, radius_m=radius_m)
            rows.append(
                {
                    "DATETIME": ts,
                    "panel": pkey,
                    "panel_label": plabel,
                    "ego_team": ego_team,
                    "radius_m": radius_m,
                    "n_boats_in_bubble": n_boats,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
    t0 = df["DATETIME"].min()
    df["race_minute"] = ((df["DATETIME"] - t0).dt.total_seconds() // 60).astype(int)
    df["race_minute_label"] = df["race_minute"].apply(lambda m: f"T+{m:02d}:00")
    return df


def build_radius_comparison(summaries: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Wide table: ego panel × avg boats for each radius."""
    base = summaries[STATS_RADII_M[0]][["panel_label"]].copy()
    for r in STATS_RADII_M:
        col = f"{r} m avg"
        base[col] = summaries[r]["avg_boats"].values
    return base


def build_summary_stats(timeline: pd.DataFrame) -> pd.DataFrame:
    """Overall average / min / max boats in bubble per ego panel."""
    agg = (
        timeline.groupby(["panel", "panel_label"], as_index=False)
        .agg(
            avg_boats=("n_boats_in_bubble", "mean"),
            min_boats=("n_boats_in_bubble", "min"),
            max_boats=("n_boats_in_bubble", "max"),
            median_boats=("n_boats_in_bubble", "median"),
            samples=("n_boats_in_bubble", "count"),
        )
        .sort_values("panel")
    )
    agg["avg_boats"] = agg["avg_boats"].round(2)
    agg["median_boats"] = agg["median_boats"].round(1)
    return agg


def build_minute_averages(timeline: pd.DataFrame) -> pd.DataFrame:
    """Average boats in bubble per race minute, one column per panel."""
    per_min = (
        timeline.groupby(["race_minute", "race_minute_label", "panel_label"], as_index=False)
        ["n_boats_in_bubble"]
        .mean()
        .round(2)
    )
    wide = per_min.pivot(index="race_minute_label", columns="panel_label", values="n_boats_in_bubble")
    wide = wide.reset_index().rename(columns={"race_minute_label": "Race minute"})
    # preserve panel order
    col_order = ["Race minute"] + [label for _, label in PANELS if label in wide.columns]
    return wide[[c for c in col_order if c in wide.columns]]


def build_count_distribution(timeline: pd.DataFrame) -> pd.DataFrame:
    """How often each boat count occurs, per panel (% of seconds)."""
    rows: list[dict] = []
    for (pkey, plabel), grp in timeline.groupby(["panel", "panel_label"]):
        total = len(grp)
        for n_boats, cnt in grp["n_boats_in_bubble"].value_counts().sort_index().items():
            rows.append(
                {
                    "panel_label": plabel,
                    "n_boats_in_bubble": int(n_boats),
                    "seconds": int(cnt),
                    "pct_of_race": round(100.0 * cnt / total, 1),
                }
            )
    return pd.DataFrame(rows)


def _table_trace(headers: list, cells: list[list], *, header_size: int = 11, cell_size: int = 10) -> go.Table:
    return go.Table(
        header=dict(
            values=headers,
            fill_color="#1e293b",
            font=dict(color="white", size=header_size),
        ),
        cells=dict(
            values=cells,
            fill_color="#0f172a",
            font=dict(color="#e2e8f0", size=cell_size),
        ),
    )


def export_stats_html(
    boats: pd.DataFrame,
    output_path: Path,
    *,
    venue: str,
    race: str,
    sample_team: str,
) -> None:
    timelines: dict[int, pd.DataFrame] = {}
    summaries: dict[int, pd.DataFrame] = {}
    minute_tables: dict[int, pd.DataFrame] = {}
    distributions: dict[int, pd.DataFrame] = {}

    for radius_m in STATS_RADII_M:
        tl = build_occupancy_timeline(boats, sample_team, radius_m=radius_m)
        timelines[radius_m] = tl
        summaries[radius_m] = build_summary_stats(tl)
        minute_tables[radius_m] = build_minute_averages(tl)
        distributions[radius_m] = build_count_distribution(tl)

    comparison = build_radius_comparison(summaries)

    n_radius_sections = len(STATS_RADII_M)
    n_rows = 1 + n_radius_sections * 3
    subplot_titles: list[str] = [
        "Cross-radius comparison — average boats in bubble (full race)",
    ]
    for r in STATS_RADII_M:
        subplot_titles.extend([
            f"{r} m — overall average",
            f"{r} m — average per race minute",
            f"{r} m — count distribution",
        ])

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        specs=[[{"type": "table"}] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.025,
    )

    row = 1
    fig.add_trace(
        _table_trace(
            list(comparison.columns),
            [comparison[c].astype(str).tolist() for c in comparison.columns],
        ),
        row=row,
        col=1,
    )
    row += 1

    for radius_m in STATS_RADII_M:
        summary = summaries[radius_m]
        minute_avg = minute_tables[radius_m]
        distribution = distributions[radius_m]

        fig.add_trace(
            _table_trace(
                ["Ego bubble", "Avg boats", "Median", "Min", "Max", "Samples (1 Hz)"],
                [
                    summary["panel_label"].astype(str).tolist(),
                    summary["avg_boats"].astype(str).tolist(),
                    summary["median_boats"].astype(str).tolist(),
                    summary["min_boats"].astype(str).tolist(),
                    summary["max_boats"].astype(str).tolist(),
                    summary["samples"].astype(str).tolist(),
                ],
            ),
            row=row,
            col=1,
        )
        row += 1

        fig.add_trace(
            _table_trace(
                list(minute_avg.columns),
                [minute_avg[c].astype(str).tolist() for c in minute_avg.columns],
                cell_size=9,
            ),
            row=row,
            col=1,
        )
        row += 1

        fig.add_trace(
            _table_trace(
                ["Ego bubble", "Boats in bubble", "Seconds", "% of race"],
                [
                    distribution["panel_label"].astype(str).tolist(),
                    distribution["n_boats_in_bubble"].astype(str).tolist(),
                    distribution["seconds"].astype(str).tolist(),
                    distribution["pct_of_race"].astype(str).tolist(),
                ],
            ),
            row=row,
            col=1,
        )
        row += 1

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(
            f"{venue} {race} — bubble occupancy by radius "
            f"({STATS_RADII_M[0]}–{STATS_RADII_M[-1]} m, +50 m steps, asymmetric)"
        ),
        height=max(900, 320 * n_rows),
        margin=dict(l=20, r=20, t=90, b=20),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, auto_open=False, include_plotlyjs="cdn")

    comparison.to_csv(output_path.with_name(output_path.stem + "_comparison.csv"), index=False)
    for radius_m in STATS_RADII_M:
        rtag = f"_r{radius_m}"
        summaries[radius_m].to_csv(
            output_path.with_name(output_path.stem + rtag + "_summary.csv"),
            index=False,
        )
        minute_tables[radius_m].to_csv(
            output_path.with_name(output_path.stem + rtag + "_minute.csv"),
            index=False,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="200 m asymmetric bubble race replay")
    p.add_argument("--venue", default="Bermuda")
    p.add_argument("--race", default="Race_2")
    p.add_argument("--sample-team", default=SAMPLE_TEAM)
    p.add_argument("--step", type=int, default=2, help="Use every Nth second")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "exported" / "bubble_asymmetric_replay.html",
    )
    p.add_argument(
        "--stats-output",
        type=Path,
        default=Path(__file__).resolve().parent / "exported" / "bubble_asymmetric_stats.html",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = find_data_root()
    boats = load_race_boats(base, args.venue, args.race, racing_only=True)
    if "TRK_RACE_RANK_unk" not in boats.columns:
        raise ValueError("TRK_RACE_RANK_unk required for rank panels")

    course_limits, _ = load_course_limits(base, args.venue, args.race)

    export_stats_html(
        boats,
        args.stats_output,
        venue=args.venue,
        race=args.race,
        sample_team=args.sample_team,
    )
    print(f"Wrote {args.stats_output.resolve()} (radii {STATS_RADII_M}, + CSV exports)")

    fig = build_figure(
        boats,
        course_limits,
        venue=args.venue,
        race=args.race,
        step=max(1, args.step),
        sample_team=args.sample_team,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output,
        auto_open=False,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True},
    )
    print(f"Wrote {args.output.resolve()} ({len(fig.frames)} frames, 4 panels)")


if __name__ == "__main__":
    main()
