"""Feature engineering and race-based splits for TabNet variations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sailgp_analysis.analytics import add_foiling
from sailgp_analysis.config import DATA_ROOT
from sailgp_analysis.data_loader import load_all_boats, load_metadata
from sailgp_analysis.tabnet.config import (
    BERMUDA_TRAIN_RACES,
    BERMUDA_VAL_RACES,
    HALIFAX_TEST_RACES,
    V4_TEST_RACES,
    V4_TRAIN_RACES,
    V4_VAL_RACES,
    V1_FEATURES,
    V2_FEATURES,
    V3_FEATURES,
    V4_FEATURES,
    V5_FEATURES,
)

SplitName = Literal["train", "val", "test"]


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    task: Literal["regression", "classification"]
    meta: dict


def _race_key(df: pd.DataFrame) -> pd.Series:
    return df["venue"].astype(str) + "::" + df["race_label"].astype(str)


def _split_mask(df: pd.DataFrame, split: SplitName) -> pd.Series:
    rk = _race_key(df)
    if split == "train":
        return rk.isin([f"Bermuda::{r}" for r in BERMUDA_TRAIN_RACES])
    if split == "val":
        return rk.isin([f"Bermuda::{r}" for r in BERMUDA_VAL_RACES])
    return rk.isin([f"Halifax::{r}" for r in HALIFAX_TEST_RACES])


def _v4_split_mask(df: pd.DataFrame, split: SplitName) -> pd.Series:
    rk = _race_key(df)
    if split == "train":
        return rk.isin([f"Bermuda::{r}" for r in V4_TRAIN_RACES])
    if split == "val":
        return rk.isin([f"Bermuda::{r}" for r in V4_VAL_RACES])
    return rk.isin([f"Bermuda::{r}" for r in V4_TEST_RACES])


def _venue_split_mask(df: pd.DataFrame, train_venue: str, test_venue: str) -> tuple[pd.Series, pd.Series]:
    return df["venue"] == train_venue, df["venue"] == test_venue


def load_base_frames(data_root=DATA_ROOT) -> pd.DataFrame:
    df = load_all_boats(data_root)
    if df.empty:
        return df
    df = add_foiling(df)
    df = _add_derived_features(df)
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_vmg_ratio"] = out["BOAT_SPEED_km_h_1"] / (out["VMG_km_h_1"].abs() + 1e-3)
    out["twa_abs_deg"] = out["TWA_SGP_deg"].abs()
    out["twa_bin"] = pd.cut(
        out["TWA_SGP_deg"].abs(),
        bins=[0, 45, 90, 135, 180],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)
    # Plan V2 target: bow ride height + speed (exclude RH from features)
    out["foiling_v2"] = (out["LENGTH_RH_BOW_mm"] > 100) & (out["BOAT_SPEED_km_h_1"] > 40)
    return out


def _encode_categoricals(
    df: pd.DataFrame,
    team_enc: LabelEncoder | None = None,
    venue_enc: LabelEncoder | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    out = df.copy()
    team_enc = team_enc or LabelEncoder()
    venue_enc = venue_enc or LabelEncoder()
    if fit:
        out["team_enc"] = team_enc.fit_transform(out["team"].astype(str))
        out["venue_enc"] = venue_enc.fit_transform(out["venue"].astype(str))
    else:
        out["team_enc"] = _safe_transform(team_enc, out["team"].astype(str))
        out["venue_enc"] = _safe_transform(venue_enc, out["venue"].astype(str))
    return out, team_enc, venue_enc


def _safe_transform(enc: LabelEncoder, values: pd.Series) -> np.ndarray:
    known = set(enc.classes_)
    mapped = values.map(lambda v: v if v in known else enc.classes_[0])
    return enc.transform(mapped.astype(str))


def _prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in features if c in df.columns]
    sub = df.dropna(subset=cols + [target])
    X = sub[cols].values.astype(np.float32)
    y = sub[target].values
    if task == "classification":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)
    return X, y, cols


def _scale_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_val
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test
    return X_train_s, X_val_s, X_test_s, scaler


def build_row_level_dataset(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str,
    racing_only: bool = True,
    prestart_only: bool = False,
) -> PreparedDataset:
    sub = df.copy()
    if racing_only:
        sub = sub[sub["TRK_BOAT_RACE_STATUS_unk"] == 2]
    elif prestart_only:
        sub = sub[sub["TRK_BOAT_RACE_STATUS_unk"] == 1]

    sub, team_enc, venue_enc = _encode_categoricals(sub, fit=True)

    train_df = sub[_split_mask(sub, "train")]
    val_df = sub[_split_mask(sub, "val")]
    test_df = sub[_split_mask(sub, "test")]

    X_train, y_train, feat_names = _prepare_xy(train_df, features, target, task)
    X_val, y_val, _ = _prepare_xy(val_df, features, target, task)
    X_test, y_test, _ = _prepare_xy(test_df, features, target, task)

    X_train, X_val, X_test, scaler = _scale_splits(X_train, X_val, X_test)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
        task=task,
        meta={
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "team_encoder": team_enc,
            "venue_encoder": venue_enc,
            "scaler": scaler,
        },
    )


def build_v1_dataset(data_root=DATA_ROOT) -> PreparedDataset:
    df = load_base_frames(data_root)
    return build_row_level_dataset(
        df, V1_FEATURES, "BOAT_SPEED_km_h_1", "regression", racing_only=True
    )


def build_v2_dataset(data_root=DATA_ROOT) -> PreparedDataset:
    df = load_base_frames(data_root)
    return build_row_level_dataset(
        df, V2_FEATURES, "foiling_v2", "classification", racing_only=True
    )


def build_leg_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team-race-leg with summary stats."""
    racing = df[df["TRK_BOAT_RACE_STATUS_unk"] == 2].copy()
    rows = []
    meta = load_metadata()
    for (venue, race_label, team, leg), grp in racing.groupby(
        ["venue", "race_label", "team", "TRK_LEG_NUM_unk"], sort=False
    ):
        if grp.empty or pd.isna(leg):
            continue
        m = meta[(meta["venue"] == venue) & (meta["race_label"] == race_label)]
        num_boats = int(m.iloc[0]["num_boats"]) if len(m) else grp["TRK_RACE_RANK_unk"].max()
        rank_end = grp["TRK_RACE_RANK_unk"].iloc[-1]
        rows.append(
            {
                "venue": venue,
                "race_label": race_label,
                "team": team,
                "leg_num": leg,
                "rank_end": rank_end,
                "mean_vmg": grp["VMG_km_h_1"].mean(),
                "foiling_pct": grp["foiling"].mean(),
                "mean_wing_rot": grp["ANGLE_WING_ROT_deg"].mean(),
                "pc_dtl_start": grp["PC_DTL_m"].iloc[0],
                "mean_tws": grp["TWS_SGP_km_h_1"].mean(),
                "mean_twd": grp["TWD_SGP_deg"].mean(),
                "penalty_count": grp["TRK_PENALTY_COUNT_unk"].max(),
                "num_boats": num_boats,
            }
        )
    return pd.DataFrame(rows)


def build_v3_dataset(data_root=DATA_ROOT) -> PreparedDataset:
    df = load_base_frames(data_root)
    leg_df = build_leg_aggregates(df)
    leg_df = leg_df[leg_df["venue"] == "Bermuda"].dropna(subset=["rank_end"])
    leg_df, team_enc, venue_enc = _encode_categoricals(leg_df, fit=True)

    train_df = leg_df[leg_df["race_label"].isin(BERMUDA_TRAIN_RACES)]
    val_df = leg_df[leg_df["race_label"] == "Race_7"]
    test_df = leg_df[leg_df["race_label"] == "Race_8"]

    X_train, y_train, feat_names = _prepare_xy(train_df, V3_FEATURES, "rank_end", "regression")
    X_val, y_val, _ = _prepare_xy(val_df, V3_FEATURES, "rank_end", "regression")
    X_test, y_test, _ = _prepare_xy(test_df, V3_FEATURES, "rank_end", "regression")
    X_train, X_val, X_test, scaler = _scale_splits(X_train, X_val, X_test)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
        task="regression",
        meta={"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test), "scaler": scaler},
    )


def _first_leg_top_half(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-race: did they finish the first racing leg in top half?"""
    racing = df[df["TRK_BOAT_RACE_STATUS_unk"] == 2].copy()
    rows = []
    for (venue, race_label, team), grp in racing.groupby(["venue", "race_label", "team"], sort=False):
        legs = grp["TRK_LEG_NUM_unk"].dropna()
        if legs.empty:
            continue
        first_leg = legs.min()
        leg1 = grp[grp["TRK_LEG_NUM_unk"] == first_leg]
        if leg1.empty:
            continue
        rank = leg1["TRK_RACE_RANK_unk"].iloc[-1]
        n_boats = grp["TRK_RACE_RANK_unk"].max()
        if pd.isna(rank) or pd.isna(n_boats):
            continue
        rows.append(
            {
                "venue": venue,
                "race_label": race_label,
                "team": team,
                "top_half_leg1": int(rank <= n_boats / 2),
            }
        )
    return pd.DataFrame(rows)


def build_prestart_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["TRK_BOAT_RACE_STATUS_unk"] == 1].copy()
    outcomes = _first_leg_top_half(df)
    rows = []
    for (venue, race_label, team), grp in pre.groupby(["venue", "race_label", "team"], sort=False):
        oc = outcomes[
            (outcomes["venue"] == venue)
            & (outcomes["race_label"] == race_label)
            & (outcomes["team"] == team)
        ]
        if oc.empty:
            continue
        rows.append(
            {
                "venue": venue,
                "race_label": race_label,
                "team": team,
                "mean_pc_tts": grp["PC_TTS_s"].mean(),
                "mean_pc_dtl": grp["PC_DTL_m"].mean(),
                "mean_pc_dto": grp["PC_DTO_m"].mean(),
                "mean_pc_dtb": grp["PC_DTB_m"].mean(),
                "mean_speed": grp["BOAT_SPEED_km_h_1"].mean(),
                "mean_vmg": grp["VMG_km_h_1"].mean(),
                "mean_twa": grp["TWA_SGP_deg"].mean(),
                "penalty_count": grp["TRK_PENALTY_COUNT_unk"].max(),
                "late_entry": int(grp["PC_TTS_s"].min() < 0),
                "top_half_leg1": oc.iloc[0]["top_half_leg1"],
            }
        )
    return pd.DataFrame(rows)


def build_v4_dataset(data_root=DATA_ROOT) -> PreparedDataset:
    df = load_base_frames(data_root)
    pre_df = build_prestart_aggregates(df)
    pre_df = pre_df[pre_df["venue"] == "Bermuda"]
    pre_df, team_enc, venue_enc = _encode_categoricals(pre_df, fit=True)

    train_df = pre_df[_v4_split_mask(pre_df, "train")]
    val_df = pre_df[_v4_split_mask(pre_df, "val")]
    test_df = pre_df[_v4_split_mask(pre_df, "test")]

    X_train, y_train, feat_names = _prepare_xy(train_df, V4_FEATURES, "top_half_leg1", "classification")
    X_val, y_val, _ = _prepare_xy(val_df, V4_FEATURES, "top_half_leg1", "classification")
    X_test, y_test, _ = _prepare_xy(test_df, V4_FEATURES, "top_half_leg1", "classification")
    X_train, X_val, X_test, scaler = _scale_splits(X_train, X_val, X_test)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
        task="classification",
        meta={"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test), "scaler": scaler},
    )


def build_cross_venue_dataset(
    train_venue: str,
    test_venue: str,
    data_root=DATA_ROOT,
) -> PreparedDataset:
    df = load_base_frames(data_root)
    sub = df[df["TRK_BOAT_RACE_STATUS_unk"] == 2].copy()
    sub, team_enc, venue_enc = _encode_categoricals(sub, fit=True)

    train_df = sub[sub["venue"] == train_venue]
    test_df = sub[sub["venue"] == test_venue]
    # Use 15% of train as val
    races = sorted(train_df["race_label"].unique())
    n_val = max(1, len(races) // 6)
    val_races = set(races[-n_val:])
    val_df = train_df[train_df["race_label"].isin(val_races)]
    train_df = train_df[~train_df["race_label"].isin(val_races)]

    X_train, y_train, feat_names = _prepare_xy(train_df, V5_FEATURES, "BOAT_SPEED_km_h_1", "regression")
    X_val, y_val, _ = _prepare_xy(val_df, V5_FEATURES, "BOAT_SPEED_km_h_1", "regression")
    X_test, y_test, _ = _prepare_xy(test_df, V5_FEATURES, "BOAT_SPEED_km_h_1", "regression")
    X_train, X_val, X_test, scaler = _scale_splits(X_train, X_val, X_test)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
        task="regression",
        meta={
            "train_venue": train_venue,
            "test_venue": test_venue,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "scaler": scaler,
        },
    )


def build_v3_loocv_folds(data_root=DATA_ROOT) -> list[tuple[str, PreparedDataset]]:
    """Leave-one-race-out folds for V3 (Bermuda only — rank available)."""
    df = load_base_frames(data_root)
    leg_df = build_leg_aggregates(df)
    leg_df = leg_df[leg_df["venue"] == "Bermuda"].dropna(subset=["rank_end"])
    leg_df, _, _ = _encode_categoricals(leg_df, fit=True)
    races = sorted(leg_df["race_label"].unique())
    folds = []
    for held_out in races:
        train_df = leg_df[leg_df["race_label"] != held_out]
        test_df = leg_df[leg_df["race_label"] == held_out]
        if test_df.empty or train_df.empty:
            continue
        X_train, y_train, feat_names = _prepare_xy(train_df, V3_FEATURES, "rank_end", "regression")
        X_test, y_test, _ = _prepare_xy(test_df, V3_FEATURES, "rank_end", "regression")
        X_val = X_test[: max(1, len(X_test) // 5)]
        y_val = y_test[: len(X_val)]
        X_train_s, X_val_s, X_test_s, scaler = _scale_splits(X_train, X_val, X_test)
        ds = PreparedDataset(
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s,
            y_val=y_val,
            X_test=X_test_s,
            y_test=y_test,
            feature_names=feat_names,
            task="regression",
            meta={"held_out_race": held_out, "scaler": scaler},
        )
        folds.append((held_out, ds))
    return folds
