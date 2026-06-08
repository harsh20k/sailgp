"""Windowed dataset builder, NaN handling, and train/val/test splits."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sailgp_analysis.config import DATA_ROOT
from sailgp_analysis.data_loader import load_all_boats

# Column name constants
COL_SPEED = "BOAT_SPEED_km_h_1"
COL_VMG = "VMG_km_h_1"
COL_TWA = "TWA_SGP_deg"
COL_TWS = "TWS_SGP_km_h_1"
COL_AWA = "AWA_SGP_deg"
COL_AWS = "AWS_SGP_km_h_1"
COL_HEEL = "HEEL_deg"
COL_PITCH = "PITCH_deg"
COL_YAW = "RATE_YAW_deg_s_1"
COL_RH_P = "LENGTH_RH_P_mm"
COL_RH_S = "LENGTH_RH_S_mm"
COL_RH_BOW = "LENGTH_RH_BOW_mm"
COL_WING_ROT = "ANGLE_WING_ROT_deg"
COL_WING_TWIST = "ANGLE_WING_TWIST_deg"
COL_DB_CANT_P = "ANGLE_DB_CANT_P_deg"
COL_DB_RAKE_P = "ANGLE_DB_RAKE_P_deg"
COL_JIB_LEAD = "PER_JIB_LEAD_pct"
COL_JIB_SHEET = "PER_JIB_SHEET_pct"
COL_RUDDER = "ANGLE_RUDDER_deg"
COL_RANK = "TRK_RACE_RANK_unk"
COL_LEG = "TRK_LEG_NUM_unk"
COL_STATUS = "TRK_BOAT_RACE_STATUS_unk"
COL_DTB = "PC_DTB_m"
COL_DTL = "PC_DTL_m"
COL_DIST = "DISTANCE_RACE_m"

COLS = {
    "speed": COL_SPEED,
    "vmg": COL_VMG,
    "twa": COL_TWA,
    "tws": COL_TWS,
    "awa": COL_AWA,
    "aws": COL_AWS,
    "heel": COL_HEEL,
    "pitch": COL_PITCH,
    "yaw": COL_YAW,
    "rh_p": COL_RH_P,
    "rh_s": COL_RH_S,
    "rh_bow": COL_RH_BOW,
    "wing_rot": COL_WING_ROT,
    "wing_twist": COL_WING_TWIST,
    "db_cant_p": COL_DB_CANT_P,
    "db_rake_p": COL_DB_RAKE_P,
    "jib_lead": COL_JIB_LEAD,
    "jib_sheet": COL_JIB_SHEET,
    "rank": COL_RANK,
    "leg": COL_LEG,
    "dtb": COL_DTB,
    "dtl": COL_DTL,
    "dist": COL_DIST,
}

CA_COLS = [f"ANGLE_CA{i}_deg" for i in range(1, 7)]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def filter_racing(df: pd.DataFrame) -> pd.DataFrame:
    if COL_STATUS not in df.columns:
        return df
    return df[df[COL_STATUS] == 2].copy()


def load_racing_boats(data_root=DATA_ROOT) -> pd.DataFrame:
    df = load_all_boats(data_root)
    if df.empty:
        return df
    return filter_racing(df)


def _race_number(race_label: str) -> int:
    m = re.search(r"(\d+)", race_label)
    return int(m.group(1)) if m else 0


def split_venue_races(
    df: pd.DataFrame,
    train_venue: str = "Bermuda",
    train_races: Iterable[str] | None = None,
    val_venue: str = "Bermuda",
    val_races: Iterable[str] | None = None,
    test_venue: str = "Halifax",
    test_races: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by venue and optional race labels."""
    train_races = set(train_races or [])
    val_races = set(val_races or [])
    test_races = set(test_races or [])

    def _mask(venue: str, races: set[str]) -> pd.Series:
        m = df["venue"] == venue
        if races:
            m &= df["race_label"].isin(races)
        return m

    train = df[_mask(train_venue, train_races)]
    val = df[_mask(val_venue, val_races)]
    test = df[_mask(test_venue, test_races)]
    return train, val, test


def split_by_races(
    df: pd.DataFrame,
    held_out_race: str,
    venue: str = "Bermuda",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["venue"] == venue]
    test = sub[sub["race_label"] == held_out_race]
    train = sub[sub["race_label"] != held_out_race]
    return train, test


def impute_series(values: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaNs along axis 0."""
    out = values.astype(np.float32, copy=True)
    if out.ndim == 1:
        s = pd.Series(out).ffill().bfill().fillna(0.0)
        return s.to_numpy(dtype=np.float32)
    frame = pd.DataFrame(out).ffill().bfill().fillna(0.0)
    return frame.to_numpy(dtype=np.float32)


def foiling_label(row: pd.Series) -> bool:
    return bool(
        row.get(COL_RH_BOW, 0) > 100
        and row.get(COL_SPEED, 0) > 40
    )


def foiling_rule(row: pd.Series) -> bool:
    """Rule-based baseline from plan / analytics."""
    return bool(
        row.get(COL_RH_BOW, 0) > 80
        and row.get(COL_SPEED, 0) > 35
    )


def rank_delta_label(rank_now: float, rank_future: float) -> int:
    """0=gain (lower rank number), 1=hold, 2=lose."""
    if np.isnan(rank_now) or np.isnan(rank_future):
        return 1
    if rank_future < rank_now:
        return 0
    if rank_future > rank_now:
        return 2
    return 1


@dataclass
class WindowSpec:
    feature_cols: list[str]
    seq_len: int
    horizon: int = 0
    stride: int = 1
    target_col: str | None = None
    target_fn: Callable[[pd.DataFrame, int], np.ndarray | float] | None = None
    group_cols: list[str] | None = None


def build_windows(
    df: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build sliding windows from grouped time series.

    Returns X (N, seq_len, F), y (N, ...) or (N,), meta dict with scaler stats.
    """
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _, gdf in df.groupby(group_cols, sort=False):
        gdf = gdf.sort_index()
        cols = [c for c in spec.feature_cols if c in gdf.columns]
        if spec.target_col and spec.target_col not in gdf.columns:
            continue
        if len(cols) < len(spec.feature_cols):
            continue

        feats = impute_series(gdf[cols].to_numpy())
        n = len(feats)
        end = n - spec.horizon if spec.horizon > 0 else n
        if end <= spec.seq_len:
            continue

        for start in range(0, end - spec.seq_len, spec.stride):
            end_idx = start + spec.seq_len
            x_win = feats[start:end_idx]
            if spec.target_fn is not None:
                y_val = spec.target_fn(gdf, end_idx)
            elif spec.target_col:
                if spec.horizon > 0:
                    y_val = float(gdf[spec.target_col].iloc[end_idx + spec.horizon - 1])
                else:
                    y_val = float(gdf[spec.target_col].iloc[end_idx - 1])
            else:
                y_val = 0.0

            if isinstance(y_val, float) and np.isnan(y_val):
                continue
            if isinstance(y_val, np.ndarray) and np.isnan(y_val).any():
                continue

            xs.append(x_win)
            ys.append(np.atleast_1d(y_val))

    if not xs:
        return np.empty((0, spec.seq_len, len(spec.feature_cols))), np.empty((0,)), {}

    X = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(-1)

    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std

    return X, y, {"mean": mean, "std": std}


class WindowedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if y.ndim == 0:
            y = y.unsqueeze(0)
        return x, y


def normalize_per_race(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Z-score features within each race to remove wind-regime advantage."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(["venue", "race_label"], group_keys=False)[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-6)
        )
    return out


def teams_in_both_venues(df: pd.DataFrame) -> list[str]:
    bermuda = set(df.loc[df["venue"] == "Bermuda", "team"].unique())
    halifax = set(df.loc[df["venue"] == "Halifax", "team"].unique())
    return sorted(bermuda & halifax)
