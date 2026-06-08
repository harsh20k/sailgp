"""Fleet join, ego-frame transforms, and tactical bubble features."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data_prep import (
    COL_HEEL,
    COL_RANK,
    COL_SPEED,
    COL_VMG,
    COL_WING_ROT,
    foiling_label,
    load_racing_boats,
)

# GPS / spatial columns
COL_LAT = "LATITUDE_GPS_unk"
COL_LON = "LONGITUDE_GPS_unk"
COL_HEADING = "HEADING_deg"

# Bubble feature column names (7 summary features)
COL_N_BOATS = "n_boats_in_200m"
COL_NEAREST_DIST = "nearest_boat_dist_m"
COL_NEAREST_BEARING = "nearest_boat_bearing_deg"
COL_NEAREST_FASTER_DIST = "nearest_faster_dist_m"
COL_N_AHEAD = "n_boats_ahead_180deg"
COL_N_FOILING = "n_boats_foiling"
COL_SPEED_DELTA = "speed_delta_nearest_kmh"

BUBBLE_FEATURE_COLS = [
    COL_N_BOATS,
    COL_NEAREST_DIST,
    COL_NEAREST_BEARING,
    COL_NEAREST_FASTER_DIST,
    COL_N_AHEAD,
    COL_N_FOILING,
    COL_SPEED_DELTA,
]

# Per-neighbour token features for attention model
NEIGHBOUR_TOKEN_COLS = [
    "dist_m",
    "bearing_deg",
    COL_SPEED,
    COL_VMG,
    "foiling",
    COL_WING_ROT,
    COL_HEEL,
    "speed_delta",
]

DEFAULT_BUBBLE_RADIUS_M = 200.0
DEFAULT_DIRTY_AIR_DIST_M = 80.0
DEFAULT_DIRTY_AIR_CONE_DEG = 45.0

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class AlignmentReport:
    venue: str
    race_label: str
    n_boats: int
    n_timestamps: int
    max_drift_s: float
    coverage_pct: float


def _to_rad(deg: np.ndarray | float) -> np.ndarray | float:
    return np.deg2rad(deg)


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in metres."""
    lat1r, lon1r = _to_rad(lat1), _to_rad(lon1)
    lat2r, lon2r = _to_rad(lat2), _to_rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def latlon_to_local_m(
    lat_ego: float,
    lon_ego: float,
    lat_other: np.ndarray,
    lon_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-earth approximation: (dx east, dy north) in metres."""
    lat_ref = math.radians(lat_ego)
    dlat = _to_rad(lat_other - lat_ego)
    dlon = _to_rad(lon_other - lon_ego)
    dx = dlon * EARTH_RADIUS_M * math.cos(lat_ref)
    dy = dlat * EARTH_RADIUS_M
    return dx, dy


def rotate_to_ego_frame(dx: np.ndarray, dy: np.ndarray, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world offsets so ego forward is +x."""
    theta = _to_rad(-heading_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x = dx * cos_t - dy * sin_t
    y = dx * sin_t + dy * cos_t
    return x, y


def polar_bearing_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bearing in ego frame: 0=ahead, 90=starboard, 180=behind."""
    return np.degrees(np.arctan2(y, x))


def validate_fleet_alignment(
    df: pd.DataFrame,
    max_drift_s: float = 1.0,
    strict: bool = False,
) -> list[AlignmentReport]:
    """
    Check that boats share a common clock per race on overlapping timestamps.

    Different boats may start/end at different seconds; we only compare drift on
    timestamps present in every boat's series (intersection), not on row-count mismatch.
    """
    reports: list[AlignmentReport] = []
    group_cols = ["venue", "race_label"]
    for (venue, race_label), race_df in df.groupby(group_cols, sort=False):
        teams = sorted(race_df["team"].unique())
        n_boats = len(teams)
        ts_sets = {
            team: set(pd.to_datetime(race_df.loc[race_df["team"] == team].index))
            for team in teams
        }
        if not ts_sets:
            continue
        common = set.intersection(*ts_sets.values()) if len(ts_sets) > 1 else next(iter(ts_sets.values()))
        union = set.union(*ts_sets.values()) if len(ts_sets) > 1 else common
        coverage = 100.0 * len(common) / max(len(union), 1)

        max_drift = 0.0
        if common:
            ref_ts = sorted(common)
            for team in teams:
                team_ts = sorted(ts_sets[team] & common)
                for a, b in zip(ref_ts, team_ts):
                    drift = abs((a - b).total_seconds())
                    max_drift = max(max_drift, drift)

        reports.append(
            AlignmentReport(
                venue=venue,
                race_label=race_label,
                n_boats=n_boats,
                n_timestamps=len(common),
                max_drift_s=max_drift,
                coverage_pct=coverage,
            )
        )
        if strict and max_drift > max_drift_s:
            raise ValueError(
                f"Fleet alignment drift {max_drift:.1f}s exceeds {max_drift_s}s "
                f"for {venue}/{race_label}"
            )
    return reports


def build_fleet_snapshots(
    df: pd.DataFrame,
    state_cols: Iterable[str] | None = None,
) -> dict[tuple[str, str, pd.Timestamp], pd.DataFrame]:
    """
    Build per-timestamp fleet snapshots keyed by (venue, race_label, datetime).

    Each snapshot is a DataFrame indexed by team with spatial/state columns.
    """
    cols = list(state_cols or [COL_LAT, COL_LON, COL_HEADING, COL_SPEED, COL_VMG, COL_RANK, COL_WING_ROT, COL_HEEL])
    cols = [c for c in cols if c in df.columns]
    snapshots: dict[tuple[str, str, pd.Timestamp], pd.DataFrame] = {}

    for (venue, race_label), race_df in df.groupby(["venue", "race_label"], sort=False):
        for ts, tick in race_df.groupby(level=0, sort=False):
            snap = tick.set_index("team")[cols].copy()
            if COL_LAT in snap.columns and COL_LON in snap.columns:
                snap = snap.dropna(subset=[COL_LAT, COL_LON])
            if len(snap) < 2:
                continue
            if "foiling" not in snap.columns:
                snap["foiling"] = tick.set_index("team").apply(foiling_label, axis=1).astype(float)
            snapshots[(venue, race_label, ts)] = snap
    return snapshots


def ego_relative(
    ego_row: pd.Series,
    others: pd.DataFrame,
    radius_m: float = DEFAULT_BUBBLE_RADIUS_M,
) -> pd.DataFrame:
    """
    Compute ego-frame distance/bearing for each neighbour within radius.

    Returns DataFrame indexed by team with dist_m, bearing_deg, and state cols.
    """
    if others.empty:
        return pd.DataFrame()

    lat_e = float(ego_row[COL_LAT])
    lon_e = float(ego_row[COL_LON])
    heading = float(ego_row.get(COL_HEADING, 0.0))

    dx, dy = latlon_to_local_m(lat_e, lon_e, others[COL_LAT].to_numpy(), others[COL_LON].to_numpy())
    x, y = rotate_to_ego_frame(dx, dy, heading)
    dist = np.sqrt(x ** 2 + y ** 2)
    bearing = polar_bearing_deg(x, y)

    out = others.copy()
    out["dist_m"] = dist
    out["bearing_deg"] = bearing
    out = out[dist <= radius_m].copy()
    return out.sort_values("dist_m")


def compute_bubble_summary(
    ego_row: pd.Series,
    neighbours: pd.DataFrame,
) -> dict[str, float]:
    """Compute the 7 summary bubble features for one ego boat at one timestep."""
    ego_speed = float(ego_row.get(COL_SPEED, np.nan))
    ego_foiling = float(foiling_label(ego_row))

    if neighbours.empty:
        return {
            COL_N_BOATS: 0.0,
            COL_NEAREST_DIST: DEFAULT_BUBBLE_RADIUS_M,
            COL_NEAREST_BEARING: 0.0,
            COL_NEAREST_FASTER_DIST: DEFAULT_BUBBLE_RADIUS_M,
            COL_N_AHEAD: 0.0,
            COL_N_FOILING: 0.0,
            COL_SPEED_DELTA: 0.0,
        }

    dist = neighbours["dist_m"].to_numpy()
    bearing = neighbours["bearing_deg"].to_numpy()
    speeds = neighbours[COL_SPEED].to_numpy(dtype=float)
    foiling = neighbours.get("foiling", pd.Series(0, index=neighbours.index)).to_numpy(dtype=float)

    nearest_idx = int(np.argmin(dist))
    nearest_dist = float(dist[nearest_idx])
    nearest_bearing = float(bearing[nearest_idx])
    nearest_speed = float(speeds[nearest_idx])

    faster_mask = speeds > ego_speed
    if faster_mask.any():
        nearest_faster_dist = float(dist[faster_mask].min())
    else:
        nearest_faster_dist = DEFAULT_BUBBLE_RADIUS_M

    # Forward hemisphere: bearing within ±90° of ahead (0°)
    ahead_mask = np.abs(bearing) <= 90.0
    n_ahead = int(ahead_mask.sum())

    return {
        COL_N_BOATS: float(len(neighbours)),
        COL_NEAREST_DIST: nearest_dist,
        COL_NEAREST_BEARING: nearest_bearing,
        COL_NEAREST_FASTER_DIST: nearest_faster_dist,
        COL_N_AHEAD: float(n_ahead),
        COL_N_FOILING: float(foiling.sum()),
        COL_SPEED_DELTA: ego_speed - nearest_speed if not np.isnan(ego_speed) else 0.0,
    }


def compute_neighbour_tokens(
    ego_row: pd.Series,
    neighbours: pd.DataFrame,
    max_k: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build padded neighbour token matrix (max_k, F) and boolean mask (max_k,).

    Token order: [dist_m, bearing_deg, speed, vmg, foiling, wing_rot, heel, speed_delta]
    """
    n_feats = len(NEIGHBOUR_TOKEN_COLS)
    tokens = np.zeros((max_k, n_feats), dtype=np.float32)
    mask = np.zeros(max_k, dtype=bool)

    if neighbours.empty:
        return tokens, mask

    ego_speed = float(ego_row.get(COL_SPEED, 0.0))
    for i, (_, nb) in enumerate(neighbours.head(max_k).iterrows()):
        mask[i] = True
        tokens[i, 0] = float(nb["dist_m"])
        tokens[i, 1] = float(nb["bearing_deg"])
        tokens[i, 2] = float(nb.get(COL_SPEED, 0.0))
        tokens[i, 3] = float(nb.get(COL_VMG, 0.0))
        tokens[i, 4] = float(nb.get("foiling", foiling_label(nb)))
        tokens[i, 5] = float(nb.get(COL_WING_ROT, 0.0))
        tokens[i, 6] = float(nb.get(COL_HEEL, 0.0))
        tokens[i, 7] = ego_speed - float(nb.get(COL_SPEED, 0.0))
    return tokens, mask


def dirty_air_flag(
    neighbours: pd.DataFrame,
    dist_m: float = DEFAULT_DIRTY_AIR_DIST_M,
    cone_deg: float = DEFAULT_DIRTY_AIR_CONE_DEG,
) -> bool:
    """True if a boat is within dist_m in the forward cone."""
    if neighbours.empty:
        return False
    mask = (neighbours["dist_m"] <= dist_m) & (np.abs(neighbours["bearing_deg"]) <= cone_deg)
    return bool(mask.any())


def add_bubble_features(
    df: pd.DataFrame,
    radius_m: float = DEFAULT_BUBBLE_RADIUS_M,
    validate_alignment: bool = True,
) -> pd.DataFrame:
    """
    Augment per-boat dataframe with bubble summary columns.

    Returns a copy with BUBBLE_FEATURE_COLS appended per row.
    """
    if validate_alignment:
        validate_fleet_alignment(df, strict=True)

    snapshots = build_fleet_snapshots(df)
    defaults = {
        COL_N_BOATS: 0.0,
        COL_NEAREST_DIST: radius_m,
        COL_NEAREST_BEARING: 0.0,
        COL_NEAREST_FASTER_DIST: radius_m,
        COL_N_AHEAD: 0.0,
        COL_N_FOILING: 0.0,
        COL_SPEED_DELTA: 0.0,
    }

    rows: list[dict] = []
    for (venue, race_label), race_df in df.groupby(["venue", "race_label"], sort=False):
        for ts, tick in race_df.groupby(level=0, sort=False):
            key = (venue, race_label, ts)
            snap = snapshots.get(key)
            if snap is None:
                continue
            for team, ego_row in tick.set_index("team").iterrows():
                if team not in snap.index:
                    continue
                others = snap.drop(index=team, errors="ignore")
                neighbours = ego_relative(ego_row, others, radius_m=radius_m)
                summary = compute_bubble_summary(ego_row, neighbours)
                rows.append(
                    {
                        "DATETIME": ts,
                        "venue": venue,
                        "race_label": race_label,
                        "team": team,
                        **summary,
                    }
                )

    if not rows:
        out = df.copy()
        for col, val in defaults.items():
            out[col] = val
        return out

    feat_df = pd.DataFrame(rows)
    out = df.reset_index().merge(
        feat_df,
        on=["DATETIME", "venue", "race_label", "team"],
        how="left",
    )
    out = out.set_index("DATETIME")
    for col, val in defaults.items():
        out[col] = out[col].fillna(val)
    return out


def load_racing_boats_with_bubble(
    radius_m: float = DEFAULT_BUBBLE_RADIUS_M,
    validate_alignment: bool = True,
) -> pd.DataFrame:
    """Convenience: load racing boats and attach bubble features."""
    df = load_racing_boats()
    if df.empty:
        return df
    return add_bubble_features(df, radius_m=radius_m, validate_alignment=validate_alignment)


def build_bubble_windows(
    df: pd.DataFrame,
    spec,
    max_k: int = 9,
    radius_m: float = DEFAULT_BUBBLE_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build ego windows plus neighbour token tensors at the last timestep of each window.

    Returns X_ego (N,T,F), X_nb (N,K,Fnb), mask (N,K), y (N,), meta.
    """
    from .data_prep import impute_series

    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    snapshots = build_fleet_snapshots(df)

    ego_xs: list[np.ndarray] = []
    nb_xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for (venue, race_label, team), gdf in df.groupby(group_cols, sort=False):
        gdf = gdf.sort_index()
        cols = [c for c in spec.feature_cols if c in gdf.columns]
        if len(cols) < len(spec.feature_cols):
            continue
        feats = impute_series(gdf[cols].to_numpy())
        n = len(feats)
        end = n - spec.horizon if spec.horizon > 0 else n
        if end <= spec.seq_len:
            continue

        for start in range(0, end - spec.seq_len, spec.stride):
            end_idx = start + spec.seq_len
            if spec.target_fn is not None:
                y_val = spec.target_fn(gdf, end_idx)
            else:
                y_val = 0.0
            if isinstance(y_val, float) and np.isnan(y_val):
                continue

            ts = gdf.index[end_idx - 1]
            ego_row = gdf.iloc[end_idx - 1]
            snap = snapshots.get((venue, race_label, ts))
            if snap is None or team not in snap.index:
                tokens = np.zeros((max_k, len(NEIGHBOUR_TOKEN_COLS)), dtype=np.float32)
                mask = np.zeros(max_k, dtype=bool)
            else:
                others = snap.drop(index=team, errors="ignore")
                neighbours = ego_relative(ego_row, others, radius_m=radius_m)
                tokens, mask = compute_neighbour_tokens(ego_row, neighbours, max_k=max_k)

            ego_xs.append(feats[start:end_idx])
            nb_xs.append(tokens)
            masks.append(mask)
            ys.append(np.atleast_1d(y_val))

    if not ego_xs:
        f_ego = len(spec.feature_cols)
        f_nb = len(NEIGHBOUR_TOKEN_COLS)
        return (
            np.empty((0, spec.seq_len, f_ego)),
            np.empty((0, max_k, f_nb)),
            np.empty((0, max_k), dtype=bool),
            np.empty((0,)),
            {},
        )

    X = np.stack(ego_xs).astype(np.float32)
    X_nb = np.stack(nb_xs).astype(np.float32)
    M = np.stack(masks)
    y = np.stack(ys).astype(np.float32).squeeze(-1)

    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std

    nb_flat = X_nb.reshape(-1, X_nb.shape[-1])
    nb_mean = nb_flat.mean(axis=0)
    nb_std = nb_flat.std(axis=0)
    nb_std[nb_std < 1e-6] = 1.0
    X_nb = (X_nb - nb_mean) / nb_std

    return X, X_nb, M, y, {"mean": mean, "std": std, "nb_mean": nb_mean, "nb_std": nb_std}
