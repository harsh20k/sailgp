"""Sliding-window dataset builders for TCM variations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sailgp_analysis.analytics import add_foiling
from sailgp_analysis.config import DATA_ROOT
from sailgp_analysis.data_loader import load_all_boats, load_marks, load_metadata
from sailgp_analysis.tcm.xml_parser import load_race_start_lines, start_line_bias

VariationName = Literal["foiling", "vmg", "rank", "prestart", "transfer"]


@dataclass
class VariationConfig:
    name: VariationName
    features: list[str]
    window_size: int
    horizon: int
    task: Literal["classification", "regression"]
    target_col: str
    filter_racing: bool = True
    filter_prestart: bool = False
    venues: list[str] = field(default_factory=lambda: ["Bermuda", "Halifax"])
    train_races: list[str] | None = None
    test_races: list[str] | None = None
    target_transform: str | None = None  # e.g. "top3", "foiling", "line_bias"


VARIATION_CONFIGS: dict[VariationName, VariationConfig] = {
    "foiling": VariationConfig(
        name="foiling",
        features=[
            "BOAT_SPEED_km_h_1",
            "LENGTH_RH_P_mm",
            "LENGTH_RH_S_mm",
            "LENGTH_RH_BOW_mm",
            "PITCH_deg",
            "HEEL_deg",
            "RATE_PITCH_deg_s_1",
            "TWA_SGP_deg",
            "TWS_SGP_km_h_1",
        ],
        window_size=30,
        horizon=1,
        task="classification",
        target_col="foiling",
        target_transform="foiling",
    ),
    "vmg": VariationConfig(
        name="vmg",
        features=[
            "TWA_SGP_deg",
            "TWS_SGP_km_h_1",
            "ANGLE_WING_ROT_deg",
            "ANGLE_WING_TWIST_deg",
            "ANGLE_DB_RAKE_P_deg",
            "ANGLE_DB_RAKE_S_deg",
            "ANGLE_DB_CANT_P_deg",
            "ANGLE_DB_CANT_S_deg",
            "LENGTH_RH_P_mm",
            "HEEL_deg",
        ],
        window_size=60,
        horizon=5,
        task="regression",
        target_col="VMG_km_h_1",
        venues=["Bermuda"],
    ),
    "rank": VariationConfig(
        name="rank",
        features=[
            "TRK_RACE_RANK_unk",
            "DISTANCE_RACE_m",
            "PC_DTL_m",
            "PC_DTB_m",
            "VMG_km_h_1",
            "BOAT_SPEED_km_h_1",
            "TRK_PENALTY_COUNT_unk",
            "mark_twd",
            "mark_tws",
        ],
        window_size=120,
        horizon=60,
        task="classification",
        target_col="top3",
        filter_racing=False,
        venues=["Bermuda"],
        train_races=[f"Race_{i}" for i in range(1, 7)],
        test_races=["Race_7", "Race_8"],
        target_transform="top3",
    ),
    "prestart": VariationConfig(
        name="prestart",
        features=[
            "LATITUDE_GPS_unk",
            "LONGITUDE_GPS_unk",
            "HEADING_deg",
            "GPS_SOG_km_h_1",
            "TWD_SGP_deg",
            "TWS_SGP_km_h_1",
            "PC_DTB_m",
        ],
        window_size=30,
        horizon=1,
        task="classification",
        target_col="line_bias",
        filter_racing=False,
        filter_prestart=True,
        target_transform="line_bias",
    ),
    "transfer": VariationConfig(
        name="transfer",
        features=[
            "BOAT_SPEED_km_h_1",
            "LENGTH_RH_P_mm",
            "LENGTH_RH_S_mm",
            "LENGTH_RH_BOW_mm",
            "PITCH_deg",
            "HEEL_deg",
            "RATE_PITCH_deg_s_1",
            "TWA_SGP_deg",
            "TWS_SGP_km_h_1",
        ],
        window_size=30,
        horizon=1,
        task="classification",
        target_col="foiling",
        target_transform="foiling",
    ),
}


def load_prepared_frames(data_root=DATA_ROOT) -> pd.DataFrame:
    """Load all boat data with foiling label and mark wind joined."""
    df = load_all_boats(data_root)
    if df.empty:
        return df
    df = add_foiling(df)
    df = _join_mark_wind(df, load_marks(data_root))
    return df


def _join_mark_wind(df: pd.DataFrame, marks: dict) -> pd.DataFrame:
    """Add mean mark TWD/TWS per race timestamp."""
    parts = []
    for (venue, race_label), group in df.groupby(["venue", "race_label"], sort=False):
        mdf = marks.get((venue, race_label))
        g = group.copy()
        if mdf is None or mdf.empty:
            g["mark_twd"] = np.nan
            g["mark_tws"] = np.nan
        else:
            m = mdf.copy()
            m["DATETIME"] = pd.to_datetime(m["DATETIME"], utc=True)
            m = m.set_index("DATETIME").sort_index()
            agg = m.groupby(m.index).agg({"TWD_deg": "mean", "TWS_km_h_1": "mean"})
            agg = agg.rename(columns={"TWD_deg": "mark_twd", "TWS_km_h_1": "mark_tws"})
            g = g.join(agg, how="left")
            g["mark_twd"] = g["mark_twd"].ffill().bfill()
            g["mark_tws"] = g["mark_tws"].ffill().bfill()
        parts.append(g)
    return pd.concat(parts).sort_index()


def _add_line_bias(df: pd.DataFrame, start_lines: dict) -> pd.DataFrame:
    out = df.copy()
    biases = []
    for idx, row in out.iterrows():
        key = (row["venue"], row["race_label"])
        sl = start_lines.get(key)
        if sl is None:
            biases.append(np.nan)
            continue
        b = start_line_bias(
            row.get("HEADING_deg", np.nan),
            row.get("LATITUDE_GPS_unk", np.nan),
            row.get("LONGITUDE_GPS_unk", np.nan),
            sl,
        )
        biases.append(b)
    out["line_bias"] = biases
    return out


def _filter_df(df: pd.DataFrame, cfg: VariationConfig, meta: pd.DataFrame) -> pd.DataFrame:
    out = df[df["venue"].isin(cfg.venues)].copy()
    if cfg.filter_racing:
        out = out[out["TRK_BOAT_RACE_STATUS_unk"] == 2]
    if cfg.filter_prestart:
        out = _filter_prestart(out, meta)
    return out


def _filter_prestart(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Keep rows in 6 min window before race start."""
    parts = []
    for (venue, race_label), group in df.groupby(["venue", "race_label"], sort=False):
        m = meta[(meta["venue"] == venue) & (meta["race_label"] == race_label)]
        if m.empty:
            continue
        start = pd.to_datetime(m.iloc[0]["race_start_utc"], utc=True)
        prestart_start = start - pd.Timedelta(minutes=6)
        mask = (group.index >= prestart_start) & (group.index < start)
        parts.append(group[mask])
    return pd.concat(parts) if parts else pd.DataFrame()


def _compute_target(series: pd.Series, cfg: VariationConfig, horizon: int) -> pd.Series:
    if cfg.target_transform == "foiling":
        return series.shift(-horizon).astype(float)
    if cfg.target_transform == "top3":
        future_rank = series.shift(-horizon)
        return (future_rank <= 3).astype(float)
    if cfg.target_transform == "line_bias":
        return series.shift(-horizon).astype(float)
    if cfg.task == "regression":
        return series.shift(-horizon)
    return series.shift(-horizon)


class WindowDataset(Dataset):
    """PyTorch dataset of (window, target, metadata) tuples."""

    def __init__(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        meta: list[dict] | None = None,
    ):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.meta = meta or []

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        y = self.targets[idx]
        if self.meta:
            return x, y, self.meta[idx]
        return x, y


def build_windows_for_group(
    group: pd.DataFrame,
    cfg: VariationConfig,
    feature_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Build windows from a single boat-race group."""
    feats = cfg.features
    missing = [f for f in feats if f not in group.columns]
    if missing:
        return np.empty((0, cfg.window_size, len(feats))), np.empty(0), [], feature_stats or {}

    g = group.dropna(subset=feats + [cfg.target_col] if cfg.target_col in group.columns else feats)
    if len(g) < cfg.window_size + cfg.horizon:
        return np.empty((0, cfg.window_size, len(feats))), np.empty(0), [], feature_stats or {}

    # Target column
    if cfg.target_transform == "foiling":
        target_series = g["foiling"].astype(float)
    elif cfg.target_transform == "top3":
        target_series = (g["TRK_RACE_RANK_unk"] <= 3).astype(float)
    elif cfg.target_transform == "line_bias":
        target_series = g["line_bias"].astype(float)
    else:
        target_series = g[cfg.target_col]

    target_series = _compute_target(target_series, cfg, cfg.horizon)

    X_raw = g[feats].values.astype(np.float32)
    y_raw = target_series.values.astype(np.float32)

    # Normalize features
    stats = feature_stats or {}
    if not stats:
        stats = {"mean": X_raw.mean(axis=0), "std": X_raw.std(axis=0) + 1e-6}
    X_norm = (X_raw - stats["mean"]) / stats["std"]

    windows, targets, meta = [], [], []
    ws, h = cfg.window_size, cfg.horizon
    for i in range(len(X_norm) - ws - h + 1):
        tgt = y_raw[i + ws - 1 + h - 1] if cfg.horizon >= 1 else y_raw[i + ws - 1]
        if np.isnan(tgt):
            continue
        if np.isnan(X_norm[i : i + ws]).any():
            continue
        windows.append(X_norm[i : i + ws])
        targets.append(tgt)
        meta.append({
            "team": g.iloc[i]["team"] if "team" in g.columns else "",
            "venue": g.iloc[i]["venue"] if "venue" in g.columns else "",
            "race_label": g.iloc[i]["race_label"] if "race_label" in g.columns else "",
            "idx": i,
        })

    if not windows:
        return np.empty((0, ws, len(feats))), np.empty(0), [], stats

    return np.stack(windows), np.array(targets), meta, stats


def build_variation_dataset(
    variation: VariationName,
    data_root=DATA_ROOT,
    split: Literal["train", "test", "all"] = "all",
    train_fraction: float = 1.0,
    train_venue: str | None = None,
    test_venue: str | None = None,
) -> tuple[WindowDataset, WindowDataset | None, dict]:
    """
    Build train/test WindowDataset for a variation.
    Returns (train_ds, test_ds, info_dict).
    """
    cfg = VARIATION_CONFIGS[variation]
    df = load_prepared_frames(data_root)
    meta = load_metadata(data_root)

    if variation == "prestart":
        start_lines = load_race_start_lines(data_root)
        df = _add_line_bias(df, start_lines)

    df = _filter_df(df, cfg, meta)
    if df.empty:
        empty = WindowDataset(np.empty((0, cfg.window_size, len(cfg.features))), np.empty(0))
        return empty, empty, {"error": "no data"}

    # Venue-based split (transfer learning)
    if train_venue and test_venue:
        df_train_src = df[df["venue"] == train_venue]
        df_test_src = df[df["venue"] == test_venue]
    else:
        df_train_src = df
        df_test_src = pd.DataFrame()

    def _build_from_df(sub: pd.DataFrame, races: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list, dict]:
        if races:
            sub = sub[sub["race_label"].isin(races)]
        all_w, all_t, all_m, stats = [], [], [], {}
        groups = list(sub.groupby(["venue", "race_label", "team"], sort=False))
        n_use = max(1, int(len(groups) * train_fraction)) if train_fraction < 1.0 and split == "train" else len(groups)
        for key, grp in groups[:n_use] if split == "train" and train_fraction < 1.0 else groups:
            w, t, m, stats = build_windows_for_group(grp, cfg, stats if stats else None)
            if len(w):
                all_w.append(w)
                all_t.append(t)
                all_m.extend(m)
        if not all_w:
            return np.empty((0, cfg.window_size, len(cfg.features))), np.empty(0), [], stats
        return np.concatenate(all_w), np.concatenate(all_t), all_m, stats

    def _build_from_df_with_stats(sub: pd.DataFrame, races: list[str] | None, stats: dict | None = None):
        if races:
            sub = sub[sub["race_label"].isin(races)]
        all_w, all_t, all_m = [], [], []
        use_stats = stats
        for _, grp in sub.groupby(["venue", "race_label", "team"], sort=False):
            w, t, m, use_stats = build_windows_for_group(grp, cfg, use_stats)
            if len(w):
                all_w.append(w)
                all_t.append(t)
                all_m.extend(m)
        if not all_w:
            return np.empty((0, cfg.window_size, len(cfg.features))), np.empty(0), [], use_stats or {}
        return np.concatenate(all_w), np.concatenate(all_t), all_m, use_stats or {}

    if train_venue and test_venue:
        X_tr, y_tr, m_tr, stats = _build_from_df_with_stats(df_train_src, None)
        X_te, y_te, m_te, _ = _build_from_df_with_stats(df_test_src, None, stats)
    elif cfg.train_races and cfg.test_races:
        X_tr, y_tr, m_tr, stats = _build_from_df_with_stats(df, cfg.train_races)
        X_te, y_te, m_te, _ = _build_from_df_with_stats(df, cfg.test_races, stats)
    else:
        races = sorted(df["race_label"].unique())
        n_train = max(1, int(len(races) * 0.8))
        train_races = list(races[:n_train])
        test_races = list(races[n_train:])
        if split == "all":
            X_tr, y_tr, m_tr, stats = _build_from_df(df)
            X_te, y_te, m_te = np.empty((0, cfg.window_size, len(cfg.features))), np.empty(0), []
        else:
            X_tr, y_tr, m_tr, stats = _build_from_df(df, train_races)
            X_te, y_te, m_te, _ = _build_from_df_with_stats(df, test_races, stats)

    train_ds = WindowDataset(X_tr, y_tr, m_tr)
    test_ds = WindowDataset(X_te, y_te, m_te) if len(X_te) else None
    info = {
        "variation": variation,
        "n_train": len(train_ds),
        "n_test": len(test_ds) if test_ds else 0,
        "n_features": len(cfg.features),
        "window_size": cfg.window_size,
        "task": cfg.task,
        "feature_stats": stats,
    }
    return train_ds, test_ds, info
