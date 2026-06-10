"""Interpolate mark wind onto a course grid and build Plotly quiver traces."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

Bbox = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
EARTH_R_M = 6_371_000.0
COLOR_BUCKETS = 5


def course_bbox(
    limits: list[dict],
    marks: pd.DataFrame,
    *,
    padding_deg: float = 0.003,
) -> Bbox:
    """Grid extent from Boundary polygon if present, else mark lat/lon bounds."""
    lats: list[float] = []
    lons: list[float] = []

    for limit in limits:
        if limit.get("name") == "Boundary":
            for lat, lon in limit["points"]:
                lats.append(lat)
                lons.append(lon)

    if not lats and not marks.empty:
        lats = marks["lat"].tolist()
        lons = marks["lon"].tolist()

    if not lats:
        raise ValueError("Need course boundary or mark positions to define grid bounds.")

    lat_min, lat_max = min(lats) - padding_deg, max(lats) + padding_deg
    lon_min, lon_max = min(lons) - padding_deg, max(lons) + padding_deg
    return lat_min, lat_max, lon_min, lon_max


def make_grid(bbox: Bbox, n: int = 18) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_centers = np.linspace(lat_min, lat_max, n)
    lon_centers = np.linspace(lon_min, lon_max, n)
    return np.meshgrid(lat_centers, lon_centers, indexing="ij")


def twd_tws_to_uv(twd_deg: float | np.ndarray, tws_kmh: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downwind u (east) and v (north) components in km/h. TWD = wind FROM."""
    downwind = np.radians(np.asarray(twd_deg, dtype=float) + 180.0)
    tws = np.asarray(tws_kmh, dtype=float)
    u = tws * np.sin(downwind)
    v = tws * np.cos(downwind)
    return u, v


def _equirect_dist_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    mean_lat = np.radians((lat1 + lat2) / 2.0)
    dlat = np.radians(lat2 - lat1) * EARTH_R_M
    dlon = np.radians(lon2 - lon1) * EARTH_R_M * np.cos(mean_lat)
    return np.hypot(dlat, dlon)


def _median_mark_spacing_m(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 2:
        return 500.0
    dists: list[float] = []
    for i in range(len(lat)):
        for j in range(i + 1, len(lat)):
            dists.append(float(_equirect_dist_m(np.array([lat[i]]), np.array([lon[i]]), lat[j], lon[j])[0]))
    return float(np.median(dists))


def idw(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    *,
    power: float = 2.0,
    eps: float = 1e-6,
    max_dist_m: float | None = None,
) -> np.ndarray:
    """Inverse-distance weights in local equirectangular space."""
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()
    out = np.full(flat_lat.shape, np.nan, dtype=float)

    for idx, (glat, glon) in enumerate(zip(flat_lat, flat_lon)):
        dist = _equirect_dist_m(lat, lon, float(glat), float(glon))
        if max_dist_m is not None and dist.min() > max_dist_m:
            continue
        w = 1.0 / np.power(dist + eps, power)
        out[idx] = float(np.sum(w * values) / np.sum(w))

    return out.reshape(grid_lat.shape)


def interpolate_wind_grid(
    mark_snap: pd.DataFrame,
    bbox: Bbox,
    n: int = 18,
) -> pd.DataFrame:
    """IDW-interpolate mark wind onto a regular lat/lon grid."""
    if mark_snap.empty:
        return pd.DataFrame(columns=["lat", "lon", "u", "v", "tws"])

    lat = mark_snap["lat"].to_numpy(dtype=float)
    lon = mark_snap["lon"].to_numpy(dtype=float)
    u_src, v_src = twd_tws_to_uv(mark_snap["twd"].to_numpy(), mark_snap["tws"].to_numpy())

    grid_lat, grid_lon = make_grid(bbox, n)
    max_dist = 2.0 * _median_mark_spacing_m(lat, lon)

    u_grid = idw(u_src, lat, lon, grid_lat, grid_lon, max_dist_m=max_dist)
    v_grid = idw(v_src, lat, lon, grid_lat, grid_lon, max_dist_m=max_dist)
    tws_grid = np.hypot(u_grid, v_grid)

    return pd.DataFrame(
        {
            "lat": grid_lat.ravel(),
            "lon": grid_lon.ravel(),
            "u": u_grid.ravel(),
            "v": v_grid.ravel(),
            "tws": tws_grid.ravel(),
        }
    ).dropna(subset=["u", "v"])


def _offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular = distance_m / EARTH_R_M
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _grid_arrow_segments(
    lat: float,
    lon: float,
    u: float,
    v: float,
    tws: float,
    *,
    tws_min: float,
    tws_max: float,
) -> tuple[list[float | None], list[float | None]]:
    if tws <= 0 or not np.isfinite(tws):
        return [], []

    if tws_max <= tws_min:
        norm = 0.5
    else:
        norm = (tws - tws_min) / (tws_max - tws_min)
    norm = max(0.0, min(1.0, norm))

    length_m = 20 + norm * 80
    downwind_bearing = math.degrees(math.atan2(u, v)) % 360.0
    tip_lat, tip_lon = _offset_latlon(lat, lon, downwind_bearing, length_m)

    lats: list[float | None] = [lat, tip_lat, None]
    lons: list[float | None] = [lon, tip_lon, None]

    head_len = max(8.0, length_m * 0.25)
    for wing_bearing in (downwind_bearing + 150.0, downwind_bearing - 150.0):
        wing_lat, wing_lon = _offset_latlon(tip_lat, tip_lon, wing_bearing, head_len)
        lats.extend([wing_lat, tip_lat, None])
        lons.extend([wing_lon, tip_lon, None])

    return lats, lons


def grid_quiver_traces(
    grid: pd.DataFrame,
    *,
    tws_min: float,
    tws_max: float,
    color_fn: Callable[[float, float, float], str],
) -> list[go.Scattermap]:
    """Bucket grid arrows into color traces for Plotly."""
    if grid.empty:
        return []

    buckets: list[tuple[list[float | None], list[float | None], str]] = [
        ([], [], "") for _ in range(COLOR_BUCKETS)
    ]

    for row in grid.itertuples(index=False):
        color = color_fn(row.tws, tws_min, tws_max)
        seg_lats, seg_lons = _grid_arrow_segments(
            row.lat, row.lon, row.u, row.v, row.tws, tws_min=tws_min, tws_max=tws_max
        )
        if not seg_lats:
            continue
        bucket_idx = min(
            COLOR_BUCKETS - 1,
            int((row.tws - tws_min) / max(tws_max - tws_min, 1e-6) * COLOR_BUCKETS),
        )
        blats, blons, bcolor = buckets[bucket_idx]
        if not bcolor:
            bcolor = color
        blats.extend(seg_lats)
        blons.extend(seg_lons)
        buckets[bucket_idx] = (blats, blons, bcolor)

    traces: list[go.Scattermap] = []
    for blats, blons, color in buckets:
        if not blats or not color:
            continue
        traces.append(
            go.Scattermap(
                lat=blats,
                lon=blons,
                mode="lines",
                line=dict(width=2, color=color),
                hoverinfo="skip",
                name="Wind grid",
                showlegend=False,
            )
        )
    return traces
