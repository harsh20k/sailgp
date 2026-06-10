#!/usr/bin/env python3
"""Build an animated GPS replay for one SailGP race (Plotly mapbox)."""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataExploration.wind_field import course_bbox, grid_quiver_traces, interpolate_wind_grid

TEAM_COLORS = px.colors.qualitative.Plotly
PLOTLY_LAYOUT = dict(template="plotly_dark", font=dict(family="IBM Plex Mono, monospace"))
LIMIT_STYLE = {
    "Boundary": {"color": "#ffcc00", "width": 3, "fillcolor": "rgba(255,204,0,0.12)"},
    "Exclusion Zone": {"color": "#6688cc", "width": 2, "fillcolor": "rgba(102,136,204,0.08)"},
    "Shallow": {"color": "#4488ff", "width": 2, "fillcolor": "rgba(68,136,255,0.12)"},
    "VIP": {"color": "#66ff66", "width": 1, "fillcolor": "rgba(102,255,102,0.08)"},
    "BYOB": {"color": "#ff6666", "width": 1, "fillcolor": "rgba(255,102,102,0.08)"},
}
VENUE_CENTER = {
    "Bermuda": {"lat": 32.27, "lon": -64.85, "zoom": 12},
    "Halifax": {"lat": 44.65, "lon": -63.57, "zoom": 12},
}


def find_data_root() -> Path:
    for path in (Path.cwd(), *Path.cwd().parents):
        candidate = path / "DataChallenge_Export"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "DataChallenge_Export/ not found. Run from the sailgp repo root or dataExploration/."
    )


def load_race_boats(base: Path, venue: str, race: str, *, racing_only: bool) -> pd.DataFrame:
    race_dir = base / venue / "boats" / race
    if not race_dir.is_dir():
        raise FileNotFoundError(f"No boat data at {race_dir}")

    frames = []
    for csv_path in sorted(race_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True, errors="coerce")
        df = df[df["DATETIME"].notna()]
        df["team"] = csv_path.stem
        if racing_only and "TRK_BOAT_RACE_STATUS_unk" in df.columns:
            df = df[df["TRK_BOAT_RACE_STATUS_unk"] == 2]
        frames.append(df)

    if not frames:
        raise ValueError(f"No boat CSV rows loaded for {venue} {race}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["LATITUDE_GPS_unk", "LONGITUDE_GPS_unk"])
    return out.sort_values("DATETIME")


def load_marks_timeseries(base: Path, venue: str, race: str) -> pd.DataFrame:
    marks_path = base / venue / "marks" / race / "marks.csv"
    if not marks_path.exists():
        return pd.DataFrame()

    marks = pd.read_csv(marks_path)
    marks["DATETIME"] = pd.to_datetime(marks["DATETIME"], utc=True, errors="coerce")
    marks = marks[marks["DATETIME"].notna()]
    marks = marks.rename(
        columns={
            "LATITUDE_deg": "lat",
            "LONGITUDE_deg": "lon",
            "TWS_km_h_1": "tws",
            "TWD_deg": "twd",
        }
    )
    return marks.dropna(subset=["lat", "lon", "tws", "twd"])


def pick_race_xml(base: Path, venue: str, race: str) -> Path | None:
    xml_dir = base / venue / "xmls" / race
    if not xml_dir.is_dir():
        return None

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        return None

    meta_path = base / venue / "race_metadata.csv"
    if not meta_path.exists():
        return xml_files[-1]

    meta = pd.read_csv(meta_path, parse_dates=["race_start_utc"])
    row = meta.loc[meta["race_label"] == race]
    if row.empty:
        return xml_files[-1]

    race_start = pd.Timestamp(row.iloc[0]["race_start_utc"])
    if race_start.tzinfo is None:
        race_start = race_start.tz_localize("UTC")

    best_path = xml_files[-1]
    best_delta: float | None = None
    for xml_path in xml_files:
        start_el = ET.parse(xml_path).getroot().find("RaceStartTime")
        if start_el is None:
            continue
        xml_start = pd.Timestamp(start_el.get("Start"))
        if xml_start.tzinfo is None:
            xml_start = xml_start.tz_localize("UTC")
        else:
            xml_start = xml_start.tz_convert("UTC")
        delta = abs((xml_start - race_start).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = xml_path

    return best_path


def load_course_limits(base: Path, venue: str, race: str) -> tuple[list[dict], Path | None]:
    xml_path = pick_race_xml(base, venue, race)
    if xml_path is None:
        return [], None

    limits: list[dict] = []
    for course_limit in ET.parse(xml_path).getroot().findall("CourseLimit"):
        points: list[tuple[float, float]] = []
        for vertex in course_limit.findall("Limit"):
            lat = vertex.get("Lat")
            lon = vertex.get("Lon")
            if lat is None or lon is None:
                continue
            points.append((float(lat), float(lon)))
        if len(points) >= 3:
            limits.append({"name": course_limit.get("name", "Limit"), "points": points})

    return limits, xml_path


def course_limit_traces(limits: list[dict]) -> list[go.Scattermap]:
    traces: list[go.Scattermap] = []
    for limit in limits:
        lats = [p[0] for p in limit["points"]] + [limit["points"][0][0]]
        lons = [p[1] for p in limit["points"]] + [limit["points"][0][1]]
        style = LIMIT_STYLE.get(limit["name"], {"color": "#aaaaaa", "width": 2, "fillcolor": "rgba(170,170,170,0.08)"})
        traces.append(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=style["width"], color=style["color"]),
                fill="toself",
                fillcolor=style["fillcolor"],
                name=limit["name"],
                hovertemplate=f"<b>{limit['name']}</b><extra></extra>",
                showlegend=False,
            )
        )
    return traces


def offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Move distance_m along bearing_deg (0° = north, 90° = east)."""
    earth_r = 6_371_000.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular = distance_m / earth_r
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def tws_to_color(tws_kmh: float, tws_min: float, tws_max: float) -> str:
    """Map TWS to a blue → yellow → red hex color."""
    if tws_max <= tws_min:
        norm = 0.5
    else:
        norm = (tws_kmh - tws_min) / (tws_max - tws_min)
    norm = max(0.0, min(1.0, norm))
    stops = [(0.0, (94, 200, 255)), (0.5, (255, 215, 0)), (1.0, (248, 81, 73))]
    for i in range(len(stops) - 1):
        lo, hi = stops[i], stops[i + 1]
        if norm <= hi[0]:
            t = (norm - lo[0]) / (hi[0] - lo[0]) if hi[0] > lo[0] else 0.0
            r = int(lo[1][0] + t * (hi[1][0] - lo[1][0]))
            g = int(lo[1][1] + t * (hi[1][1] - lo[1][1]))
            b = int(lo[1][2] + t * (hi[1][2] - lo[1][2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#f85149"


def snap_marks_at_time(marks: pd.DataFrame, timestamp) -> pd.DataFrame:
    """Nearest mark reading per MARK within ±2 s of the frame time."""
    if marks.empty:
        return marks
    ts = pd.Timestamp(timestamp)
    tolerance = pd.Timedelta(seconds=2)
    rows: list[pd.Series] = []
    for _, grp in marks.groupby("MARK", sort=False):
        grp = grp.sort_values("DATETIME")
        target = pd.DataFrame({"DATETIME": [ts]})
        hit = pd.merge_asof(
            target,
            grp,
            on="DATETIME",
            direction="nearest",
            tolerance=tolerance,
        )
        if hit["tws"].notna().any():
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows) if rows else marks.iloc[0:0]


def wind_arrow_segments(
    lat: float,
    lon: float,
    twd_deg: float,
    tws_kmh: float,
    *,
    tws_min: float,
    tws_max: float,
) -> tuple[list[float], list[float]]:
    """Build line segments for one wind arrow (shaft + head). TWD = wind FROM; arrow points downwind."""
    if tws_max <= tws_min:
        norm = 0.5
    else:
        norm = (tws_kmh - tws_min) / (tws_max - tws_min)
    norm = max(0.0, min(1.0, norm))

    length_m = 35 + norm * 200
    downwind_bearing = (twd_deg + 180.0) % 360.0
    tip_lat, tip_lon = offset_latlon(lat, lon, downwind_bearing, length_m)

    lats = [lat, tip_lat, None]
    lons = [lon, tip_lon, None]

    head_len = max(12.0, length_m * 0.25)
    for wing_bearing in (downwind_bearing + 150.0, downwind_bearing - 150.0):
        wing_lat, wing_lon = offset_latlon(tip_lat, tip_lon, wing_bearing, head_len)
        lats.extend([wing_lat, tip_lat, None])
        lons.extend([wing_lon, tip_lon, None])

    return lats, lons


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_r * math.asin(math.sqrt(a))


def compute_cumulative_distances(boats: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for team, tdf in boats.groupby("team", sort=True):
        tdf = tdf.sort_values("DATETIME")
        cum_m = 0.0
        prev_lat: float | None = None
        prev_lon: float | None = None
        for row in tdf.itertuples(index=False):
            if prev_lat is not None:
                cum_m += haversine_m(prev_lat, prev_lon, row.LATITUDE_GPS_unk, row.LONGITUDE_GPS_unk)
            records.append({"DATETIME": row.DATETIME, "team": team, "cum_distance_m": cum_m})
            prev_lat, prev_lon = row.LATITUDE_GPS_unk, row.LONGITUDE_GPS_unk
    return pd.DataFrame(records)


def distances_at_time(cum_dist: pd.DataFrame, timestamp, teams: list[str]) -> dict[str, float]:
    ts = pd.Timestamp(timestamp)
    sub = cum_dist[cum_dist["DATETIME"] <= ts]
    out = {team: 0.0 for team in teams}
    if sub.empty:
        return out
    latest = sub.sort_values("DATETIME").groupby("team").last()
    for team in teams:
        if team in latest.index:
            out[team] = float(latest.loc[team, "cum_distance_m"])
    return out


def distance_bar_trace(team_distances: dict[str, float], colors: dict[str, str]) -> go.Bar:
    ordered = sorted(team_distances, key=team_distances.get, reverse=True)
    distances_km = [team_distances[team] / 1000.0 for team in ordered]
    return go.Bar(
        x=ordered,
        y=distances_km,
        marker_color=[colors[team] for team in ordered],
        text=[f"{d:.2f}" for d in distances_km],
        textposition="outside",
        textfont=dict(size=9, color="#dddddd"),
        hovertemplate="<b>%{x}</b><br>Distance: %{y:.2f} km<extra></extra>",
        name="Distance",
        showlegend=False,
    )


def team_color_map(teams: list[str]) -> dict[str, str]:
    return {team: TEAM_COLORS[i % len(TEAM_COLORS)] for i, team in enumerate(sorted(teams))}


def with_alpha(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def trail_trace(
    boats: pd.DataFrame,
    timestamp,
    colors: dict[str, str],
    *,
    trail_seconds: int,
) -> go.Scattermap:
    ts = pd.Timestamp(timestamp)
    window_start = ts - pd.Timedelta(seconds=trail_seconds - 1)
    trail = boats[(boats["DATETIME"] >= window_start) & (boats["DATETIME"] < ts)]
    if trail.empty:
        return go.Scattermap(
            lat=[],
            lon=[],
            mode="markers",
            marker=dict(size=0),
            hoverinfo="skip",
            name="Trail",
            showlegend=False,
        )

    age_s = (ts - trail["DATETIME"]).dt.total_seconds()
    max_age = max(float(age_s.max()), 1.0)
    freshness = 1.0 - age_s / max_age
    sizes = (4 + 7 * freshness).tolist()
    marker_colors = [with_alpha(colors[team], 0.25 + 0.55 * f) for team, f in zip(trail["team"], freshness)]

    return go.Scattermap(
        lat=trail["LATITUDE_GPS_unk"],
        lon=trail["LONGITUDE_GPS_unk"],
        mode="markers",
        marker=dict(size=sizes, color=marker_colors),
        customdata=trail[["team"]].values,
        hovertemplate="<b>%{customdata[0]}</b> trail<extra></extra>",
        name="Trail",
        showlegend=False,
    )


def format_boat_label(team: str, rank_val) -> str:
    if pd.notna(rank_val):
        return f"#{int(rank_val)}<br>{team}"
    return team


def race_rank_series(snapshot: pd.DataFrame) -> pd.Series:
    if "TRK_RACE_RANK_unk" in snapshot.columns and snapshot["TRK_RACE_RANK_unk"].notna().any():
        return snapshot["TRK_RACE_RANK_unk"]
    if "PC_DTL_m" in snapshot.columns and snapshot["PC_DTL_m"].notna().any():
        return snapshot["PC_DTL_m"].rank(method="min")
    return pd.Series([None] * len(snapshot), index=snapshot.index)


def boat_trace(snapshot: pd.DataFrame, colors: dict[str, str]) -> go.Scattermap:
    marker_colors = [colors[team] for team in snapshot["team"]]
    rank = race_rank_series(snapshot)
    speed = snapshot.get("GPS_SOG_km_h_1", pd.Series([None] * len(snapshot)))
    labels = [format_boat_label(team, r) for team, r in zip(snapshot["team"], rank)]
    custom = pd.DataFrame({"team": snapshot["team"], "speed": speed, "rank": rank})

    return go.Scattermap(
        lat=snapshot["LATITUDE_GPS_unk"],
        lon=snapshot["LONGITUDE_GPS_unk"],
        mode="markers+text",
        marker=dict(size=13, color=marker_colors),
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        customdata=custom.values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Speed: %{customdata[1]:.1f} km/h<br>"
            "Rank: %{customdata[2]}<br>"
            "<extra></extra>"
        ),
        name="Boats",
        showlegend=False,
    )


def marks_trace(mark_snap: pd.DataFrame) -> go.Scattermap:
    return go.Scattermap(
        lat=mark_snap["lat"],
        lon=mark_snap["lon"],
        mode="markers+text",
        marker=dict(size=9, color="white"),
        text=mark_snap["MARK"],
        textposition="bottom center",
        textfont=dict(size=9, color="#cccccc"),
        customdata=mark_snap[["tws", "twd"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "TWS: %{customdata[0]:.1f} km/h<br>"
            "TWD: %{customdata[1]:.0f}° (from)<br>"
            "<extra></extra>"
        ),
        name="Marks",
        showlegend=False,
    )


def wind_arrow_traces(mark_snap: pd.DataFrame, *, tws_min: float, tws_max: float) -> list[go.Scattermap]:
    traces: list[go.Scattermap] = []
    for row in mark_snap.itertuples(index=False):
        seg_lats, seg_lons = wind_arrow_segments(
            row.lat,
            row.lon,
            row.twd,
            row.tws,
            tws_min=tws_min,
            tws_max=tws_max,
        )
        color = tws_to_color(row.tws, tws_min, tws_max)
        traces.append(
            go.Scattermap(
                lat=seg_lats,
                lon=seg_lons,
                mode="lines",
                line=dict(width=3, color=color),
                name=f"Wind {row.MARK}",
                hovertemplate=(
                    f"<b>{row.MARK}</b><br>"
                    f"TWS: {row.tws:.1f} km/h<br>"
                    f"TWD: {row.twd:.0f}° (from)<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    return traces


def _wind_legend_annotation(tws_min: float, tws_max: float, *, interpolated: bool = False) -> dict:
    if interpolated:
        text = (
            f"Grid arrows interpolated from mark sensors (IDW); "
            f"length &amp; color = TWS ({tws_min:.0f}–{tws_max:.0f} km/h); direction = downwind"
        )
    else:
        text = (
            f"Arrow length &amp; color = TWS ({tws_min:.0f}–{tws_max:.0f} km/h); "
            "direction = downwind"
        )
    return dict(
        text=text,
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11, color="#cccccc"),
        bgcolor="rgba(0,0,0,0.45)",
        borderpad=4,
    )


def _animation_controls(times: list, step: int, *, y_pos: float) -> tuple[list, list]:
    slider_steps = []
    for i, t in enumerate(times):
        label = pd.Timestamp(t).strftime("%H:%M:%S")
        slider_steps.append(
            dict(
                method="animate",
                args=[[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=label if i % max(1, len(times) // 12) == 0 else "",
            )
        )
    frame_ms = max(100, int(1000 / step))
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=y_pos,
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
                    args=[
                        [None],
                        {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                    ],
                ),
            ],
        )
    ]
    sliders = [
        dict(
            active=0,
            x=0.12,
            y=y_pos,
            len=0.84,
            xanchor="left",
            yanchor="bottom",
            pad=dict(t=40),
            currentvalue=dict(prefix="Time: ", visible=True),
            steps=slider_steps,
        )
    ]
    return updatemenus, sliders


def build_wind_field_figure(
    marks: pd.DataFrame,
    course_limits: list[dict],
    *,
    venue: str,
    race: str,
    step: int,
    show_boundaries: bool,
    interpolated: bool = False,
    grid_size: int = 18,
) -> go.Figure:
    if marks.empty:
        raise ValueError(f"No mark wind data for {venue} {race}")

    times = sorted(marks["DATETIME"].unique())[::step]
    if len(times) < 2:
        raise ValueError("Need at least two mark timestamps after filtering/step.")

    tws_min = float(marks["tws"].min())
    tws_max = float(marks["tws"].max())
    boundary_traces = course_limit_traces(course_limits) if show_boundaries else []
    bbox = course_bbox(course_limits if show_boundaries else [], marks)
    center = VENUE_CENTER.get(
        venue,
        {"lat": float(marks["lat"].mean()), "lon": float(marks["lon"].mean()), "zoom": 12},
    )

    def frame_traces(timestamp) -> list:
        traces: list = list(boundary_traces)
        mark_snap = snap_marks_at_time(marks, timestamp)
        if not mark_snap.empty:
            if interpolated:
                grid = interpolate_wind_grid(mark_snap, bbox, n=grid_size)
                traces.extend(
                    grid_quiver_traces(grid, tws_min=tws_min, tws_max=tws_max, color_fn=tws_to_color)
                )
            else:
                traces.extend(wind_arrow_traces(mark_snap, tws_min=tws_min, tws_max=tws_max))
            traces.append(marks_trace(mark_snap))
        return traces

    title_suffix = "interpolated mark wind field" if interpolated else "mark wind field"
    fig = go.Figure(data=frame_traces(times[0]))
    fig.frames = [go.Frame(data=frame_traces(t), name=str(t)) for t in times]

    updatemenus, sliders = _animation_controls(times, step, y_pos=0.02)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{venue} {race} — {title_suffix}",
        height=720,
        margin=dict(l=0, r=0, t=48, b=0),
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=center["zoom"],
        ),
        annotations=[_wind_legend_annotation(tws_min, tws_max, interpolated=interpolated)],
        updatemenus=updatemenus,
        sliders=sliders,
    )
    return fig


def build_replay_figure(
    boats: pd.DataFrame,
    marks: pd.DataFrame,
    course_limits: list[dict],
    *,
    venue: str,
    race: str,
    step: int,
    trail_seconds: int,
) -> go.Figure:
    times = sorted(boats["DATETIME"].unique())[::step]
    if len(times) < 2:
        raise ValueError("Need at least two timestamps after filtering/step.")

    teams = sorted(boats["team"].unique())
    colors = team_color_map(teams)
    center = VENUE_CENTER.get(venue, {"lat": boats["LATITUDE_GPS_unk"].mean(), "lon": boats["LONGITUDE_GPS_unk"].mean(), "zoom": 12})

    tws_min = float(marks["tws"].min()) if not marks.empty else 0.0
    tws_max = float(marks["tws"].max()) if not marks.empty else 1.0
    boundary_traces = course_limit_traces(course_limits)
    cum_dist = compute_cumulative_distances(boats)

    def frame_traces(timestamp) -> list:
        snap = boats[boats["DATETIME"] == timestamp]
        traces: list = list(boundary_traces)
        mark_snap = snap_marks_at_time(marks, timestamp)
        if not mark_snap.empty:
            traces.extend(wind_arrow_traces(mark_snap, tws_min=tws_min, tws_max=tws_max))
            traces.append(marks_trace(mark_snap))
        if trail_seconds > 0:
            traces.append(trail_trace(boats, timestamp, colors, trail_seconds=trail_seconds))
        traces.append(boat_trace(snap, colors))
        traces.append(distance_bar_trace(distances_at_time(cum_dist, timestamp, teams), colors))
        return traces

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        specs=[[{"type": "scattermap"}], [{"type": "bar"}]],
        vertical_spacing=0.06,
    )

    initial = frame_traces(times[0])
    for trace in initial[:-1]:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(initial[-1], row=2, col=1)
    fig.frames = [go.Frame(data=frame_traces(t), name=str(t)) for t in times]

    updatemenus, sliders = _animation_controls(times, step, y_pos=0.34)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{venue} {race} — boat replay + wind + boundaries",
        height=960,
        margin=dict(l=0, r=0, t=48, b=0),
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=center["zoom"],
        ),
        annotations=[_wind_legend_annotation(tws_min, tws_max)] if not marks.empty else [],
        updatemenus=updatemenus,
        sliders=sliders,
    )
    fig.update_yaxes(title_text="Distance sailed (km)", row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an animated SailGP race replay HTML.")
    parser.add_argument("--venue", default="Bermuda")
    parser.add_argument("--race", default="Race_1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step", type=int, default=1, help="Use every Nth second (2 = half speed/file size).")
    parser.add_argument(
        "--trail-seconds",
        type=int,
        default=10,
        help="Trailing history per boat in seconds (0 to disable).",
    )
    parser.add_argument(
        "--include-prestart",
        action="store_true",
        help="Include pre-start rows (default: racing status only).",
    )
    parser.add_argument(
        "--wind-only",
        action="store_true",
        help="Map-only animated mark wind field (no boats or distance chart).",
    )
    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Omit course limit polygons from the map.",
    )
    parser.add_argument(
        "--interpolated",
        action="store_true",
        help="With --wind-only: IDW grid quiver across course (marks still shown).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=18,
        help="Grid resolution per axis for --interpolated (default 18).",
    )
    return parser.parse_args()


def default_output_path(venue: str, race: str, *, wind_only: bool, interpolated: bool = False) -> Path:
    if wind_only and interpolated:
        return Path(f"dataExploration/exported/wind_field_interp_{venue}_{race}.html")
    if wind_only:
        return Path(f"dataExploration/exported/wind_field_{venue}_{race}.html")
    return Path("race_replay.html")


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = default_output_path(
            args.venue, args.race, wind_only=args.wind_only, interpolated=args.interpolated
        )

    base = find_data_root()
    marks = load_marks_timeseries(base, args.venue, args.race)
    course_limits, xml_path = load_course_limits(base, args.venue, args.race)
    show_boundaries = not args.no_boundaries

    if args.interpolated and not args.wind_only:
        raise SystemExit("--interpolated requires --wind-only")

    if args.wind_only:
        fig = build_wind_field_figure(
            marks,
            course_limits,
            venue=args.venue,
            race=args.race,
            step=max(1, args.step),
            show_boundaries=show_boundaries,
            interpolated=args.interpolated,
            grid_size=max(4, args.grid_size),
        )
    else:
        boats = load_race_boats(base, args.venue, args.race, racing_only=not args.include_prestart)
        if not show_boundaries:
            course_limits = []
        fig = build_replay_figure(
            boats,
            marks,
            course_limits,
            venue=args.venue,
            race=args.race,
            step=max(1, args.step),
            trail_seconds=max(0, args.trail_seconds),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output,
        auto_open=False,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True},
    )
    n_frames = len(fig.frames)
    limit_names = ", ".join(l["name"] for l in course_limits) or "none"
    xml_note = f" from {xml_path.name}" if xml_path else ""
    if args.wind_only:
        n_marks = marks["MARK"].nunique() if not marks.empty else 0
        mode = "interpolated grid" if args.interpolated else "mark arrows"
        print(f"Wrote {args.output.resolve()} ({n_frames} frames, {n_marks} marks, {mode})")
    else:
        n_teams = boats["team"].nunique()
        print(f"Wrote {args.output.resolve()} ({n_frames} frames, {n_teams} teams)")
    if show_boundaries:
        print(f"Course limits{xml_note}: {limit_names}")
    else:
        print("Course limits: omitted (--no-boundaries)")


if __name__ == "__main__":
    main()
