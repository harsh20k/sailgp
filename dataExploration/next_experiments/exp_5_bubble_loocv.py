"""Experiment 5 — Bubble Attention Multi-Race LOOCV (next-experiments #5)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_DTL,
    COL_DIST,
    COL_HEEL,
    COL_LEG,
    COL_RANK,
    COL_SPEED,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    COL_WING_ROT,
    WindowSpec,
    foiling_label,
    get_device,
    impute_series,
    rank_delta_label,
    split_by_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    evaluate_classification,
    predict_lstm,
    run_training,
)
from dataExploration.lstm_experiments.shared.fleet import (
    DEFAULT_BUBBLE_RADIUS_M,
    NEIGHBOUR_TOKEN_COLS,
    build_fleet_snapshots,
    ego_relative,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.models import (
    BiLSTMClassifier,
    BubbleAttentionClassifier,
    BubbleAttentionLSTMTokenClassifier,
)
from dataExploration.lstm_experiments.shared.data_prep import WindowedDataset

EGO_FEATURES = [
    COL_SPEED,
    COL_VMG,
    COL_DTB,
    COL_DTL,
    COL_TWA,
    COL_TWS,
    COL_LEG,
    COL_HEEL,
    COL_WING_ROT,
    COL_DIST,
]
NB_SEQ_FEATURES = [
    COL_SPEED,
    COL_VMG,
    "foiling",
    COL_WING_ROT,
    COL_HEEL,
    "dist_m",
    "bearing_deg",
    "speed_delta",
]
SEQ_LEN = 30
HORIZON = 30
NB_SEQ_LEN = 5
MAX_K = 9
BERMUDA_RACES = [f"Race_{i}" for i in range(1, 9)]
MAX_EPOCHS = 30
PATIENCE = 8
EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"


def _rank_delta_target(gdf, end_idx: int) -> float:
    rank_now = float(gdf[COL_RANK].iloc[end_idx - 1])
    future_idx = end_idx + HORIZON - 1
    if future_idx >= len(gdf):
        return np.nan
    rank_future = float(gdf[COL_RANK].iloc[future_idx])
    return float(rank_delta_label(rank_now, rank_future))


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=3)
    raw = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor((raw / raw.sum() * 3).astype(np.float32))


def _normalize_ego(X: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    flat = X.reshape(-1, X.shape[-1])
    if stats is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean, "std": std}
    Xn = (X - stats["mean"]) / stats["std"]
    return Xn.astype(np.float32), stats


def _normalize_nb_snapshot(X_nb: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    flat = X_nb.reshape(-1, X_nb.shape[-1])
    if stats is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean, "std": std}
    Xn = (X_nb - stats["mean"]) / stats["std"]
    return Xn.astype(np.float32), stats


def _normalize_nb_seq(X_nb: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    flat = X_nb.reshape(-1, X_nb.shape[-1])
    if stats is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean, "std": std}
    Xn = (X_nb - stats["mean"]) / stats["std"]
    return Xn.astype(np.float32), stats


def _neighbour_seq_at_ts(
    ego_gdf: pd.DataFrame,
    nb_gdf: pd.DataFrame,
    ts: pd.Timestamp,
    ego_speed: float,
) -> np.ndarray:
    """Last NB_SEQ_LEN timesteps of neighbour features in ego frame."""
    n_f = len(NB_SEQ_FEATURES)
    seq = np.zeros((NB_SEQ_LEN, n_f), dtype=np.float32)
    hist = nb_gdf.loc[:ts].tail(NB_SEQ_LEN)
    if hist.empty:
        return seq
    offset = NB_SEQ_LEN - len(hist)
    for j, (t, nb_row) in enumerate(hist.iterrows()):
        idx = offset + j
        if t not in ego_gdf.index:
            continue
        ego_row = ego_gdf.loc[t]
        lat_e, lon_e = float(ego_row["LATITUDE_GPS_unk"]), float(ego_row["LONGITUDE_GPS_unk"])
        heading = float(ego_row.get("HEADING_deg", 0.0))
        from dataExploration.lstm_experiments.shared.fleet import (
            latlon_to_local_m,
            rotate_to_ego_frame,
            polar_bearing_deg,
        )

        dx, dy = latlon_to_local_m(lat_e, lon_e, np.array([nb_row["LATITUDE_GPS_unk"]]), np.array([nb_row["LONGITUDE_GPS_unk"]]))
        x, y = rotate_to_ego_frame(dx, dy, heading)
        dist = float(np.sqrt(x[0] ** 2 + y[0] ** 2))
        bearing = float(polar_bearing_deg(x, y)[0])
        nb_speed = float(nb_row.get(COL_SPEED, 0.0))
        seq[idx, 0] = nb_speed
        seq[idx, 1] = float(nb_row.get(COL_VMG, 0.0))
        seq[idx, 2] = float(foiling_label(nb_row))
        seq[idx, 3] = float(nb_row.get(COL_WING_ROT, 0.0))
        seq[idx, 4] = float(nb_row.get(COL_HEEL, 0.0))
        seq[idx, 5] = dist
        seq[idx, 6] = bearing
        seq[idx, 7] = ego_speed - nb_speed
    return seq


def build_bubble_windows_loocv(
    df: pd.DataFrame,
    spec: WindowSpec,
    team_gdfs: dict,
    max_k: int = MAX_K,
    radius_m: float = DEFAULT_BUBBLE_RADIUS_M,
    ego_stats: dict | None = None,
    nb_stats: dict | None = None,
    lstm_tokens: bool = False,
    nb_seq_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build bubble windows; normalize with train stats when provided."""
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    snapshots = build_fleet_snapshots(df)

    ego_xs: list[np.ndarray] = []
    nb_xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    raw_dists: list[np.ndarray] = []
    raw_speed_deltas: list[np.ndarray] = []

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
            y_val = spec.target_fn(gdf, end_idx) if spec.target_fn else 0.0
            if isinstance(y_val, float) and np.isnan(y_val):
                continue

            ts = gdf.index[end_idx - 1]
            ego_row = gdf.iloc[end_idx - 1]
            ego_speed = float(ego_row.get(COL_SPEED, 0.0))
            snap = snapshots.get((venue, race_label, ts))
            dist_row = np.full(max_k, radius_m, dtype=np.float32)
            sd_row = np.zeros(max_k, dtype=np.float32)

            if snap is None or team not in snap.index:
                if lstm_tokens:
                    tokens = np.zeros((max_k, NB_SEQ_LEN, len(NB_SEQ_FEATURES)), dtype=np.float32)
                else:
                    tokens = np.zeros((max_k, len(NEIGHBOUR_TOKEN_COLS)), dtype=np.float32)
                mask = np.zeros(max_k, dtype=bool)
            else:
                others = snap.drop(index=team, errors="ignore")
                neighbours = ego_relative(ego_row, others, radius_m=radius_m)
                mask = np.zeros(max_k, dtype=bool)
                if lstm_tokens:
                    tokens = np.zeros((max_k, NB_SEQ_LEN, len(NB_SEQ_FEATURES)), dtype=np.float32)
                else:
                    tokens = np.zeros((max_k, len(NEIGHBOUR_TOKEN_COLS)), dtype=np.float32)

                for i, (nb_team, nb) in enumerate(neighbours.head(max_k).iterrows()):
                    mask[i] = True
                    dist_row[i] = float(nb["dist_m"])
                    sd_row[i] = abs(ego_speed - float(nb.get(COL_SPEED, 0.0)))
                    if lstm_tokens:
                        nb_gdf = team_gdfs.get((venue, race_label, nb_team))
                        if nb_gdf is not None:
                            tokens[i] = _neighbour_seq_at_ts(gdf, nb_gdf, ts, ego_speed)
                    else:
                        tokens[i, 0] = float(nb["dist_m"])
                        tokens[i, 1] = float(nb["bearing_deg"])
                        tokens[i, 2] = float(nb.get(COL_SPEED, 0.0))
                        tokens[i, 3] = float(nb.get(COL_VMG, 0.0))
                        tokens[i, 4] = float(nb.get("foiling", foiling_label(nb)))
                        tokens[i, 5] = float(nb.get(COL_WING_ROT, 0.0))
                        tokens[i, 6] = float(nb.get(COL_HEEL, 0.0))
                        tokens[i, 7] = ego_speed - float(nb.get(COL_SPEED, 0.0))

            ego_xs.append(feats[start:end_idx])
            nb_xs.append(tokens)
            masks.append(mask)
            ys.append(np.atleast_1d(y_val))
            raw_dists.append(dist_row)
            raw_speed_deltas.append(sd_row)

    if not ego_xs:
        if lstm_tokens:
            empty_nb = np.empty((0, max_k, NB_SEQ_LEN, len(NB_SEQ_FEATURES)))
        else:
            empty_nb = np.empty((0, max_k, len(NEIGHBOUR_TOKEN_COLS)))
        return (
            np.empty((0, spec.seq_len, len(spec.feature_cols))),
            empty_nb,
            np.empty((0, max_k), dtype=bool),
            np.empty((0,)),
            {},
        )

    X = np.stack(ego_xs).astype(np.float32)
    X_nb = np.stack(nb_xs).astype(np.float32)
    M = np.stack(masks)
    y = np.stack(ys).astype(np.float32).squeeze(-1)

    X, ego_stats = _normalize_ego(X, ego_stats)
    if lstm_tokens:
        X_nb, nb_seq_stats = _normalize_nb_seq(X_nb, nb_seq_stats)
    else:
        X_nb, nb_stats = _normalize_nb_snapshot(X_nb, nb_stats)

    meta = {
        "ego_stats": ego_stats,
        "nb_stats": nb_stats if not lstm_tokens else None,
        "nb_seq_stats": nb_seq_stats if lstm_tokens else None,
        "raw_dist": np.stack(raw_dists),
        "raw_abs_speed_delta": np.stack(raw_speed_deltas),
    }
    return X, X_nb, M, y, meta


def _build_team_gdfs(df: pd.DataFrame) -> dict:
    out = {}
    for key, gdf in df.groupby(["venue", "race_label", "team"], sort=False):
        out[key] = gdf.sort_index()
    return out


def _build_windows_loocv(
    df: pd.DataFrame,
    spec: WindowSpec,
    ego_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """build_windows with train-only normalization for LOOCV."""
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _, gdf in df.groupby(group_cols, sort=False):
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
            y_val = spec.target_fn(gdf, end_idx) if spec.target_fn else 0.0
            if isinstance(y_val, float) and np.isnan(y_val):
                continue
            xs.append(feats[start:end_idx])
            ys.append(np.atleast_1d(y_val))

    if not xs:
        return np.empty((0, spec.seq_len, len(spec.feature_cols))), np.empty((0,)), {}

    X = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32).squeeze(-1)
    X, ego_stats = _normalize_ego(X, ego_stats)
    return X, y, ego_stats


class BubbleDataset(Dataset):
    def __init__(self, X, X_nb, mask, y):
        self.X = torch.from_numpy(X)
        self.X_nb = torch.from_numpy(X_nb)
        self.mask = torch.from_numpy(mask)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = self.y[idx]
        if y.ndim == 0:
            y = y.unsqueeze(0)
        return self.X[idx], self.X_nb[idx], self.mask[idx], y


def _train_bubble(
    model: nn.Module,
    X_tr, X_nb_tr, m_tr, y_tr,
    X_va, X_nb_va, m_va, y_va,
    class_weights: torch.Tensor,
    label: str,
    lstm_tokens: bool = False,
) -> tuple[nn.Module, np.ndarray]:
    device = get_device()
    train_ds = BubbleDataset(X_tr, X_nb_tr, m_tr, y_tr)
    val_ds = BubbleDataset(X_va, X_nb_va, m_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    stale = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, xnb, mb, yb in train_loader:
            xb, xnb, mb, yb = xb.to(device), xnb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, xnb, mb)
            loss = criterion(out, yb.long().squeeze(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, xnb, mb, yb in val_loader:
                xb, xnb, mb, yb = xb.to(device), xnb.to(device), mb.to(device), yb.to(device)
                out = model(xb, xnb, mb)
                val_losses.append(float(criterion(out, yb.long().squeeze(-1)).item()))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, xnb, mb, _ in val_loader:
            xb, xnb, mb = xb.to(device), xnb.to(device), mb.to(device)
            out = model(xb, xnb, mb)
            preds.append(out.argmax(dim=-1).cpu().numpy())
    return model, np.concatenate(preds)


def _predict_bubble(
    model: nn.Module,
    X: np.ndarray,
    X_nb: np.ndarray,
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    device = get_device()
    ds = BubbleDataset(X, X_nb, M, np.zeros(len(X), dtype=np.float32))
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds: list[np.ndarray] = []
    attn_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, xnb, mb, _ in loader:
            xb, xnb, mb = xb.to(device), xnb.to(device), mb.to(device)
            out = model(xb, xnb, mb)
            preds.append(out.argmax(dim=-1).cpu().numpy())
            if model.last_attn_weights is not None:
                w = model.last_attn_weights.cpu().numpy()
                if w.ndim == 4:
                    w = w.mean(axis=1)
                if w.ndim == 3:
                    w = w[:, 0, :]
                attn_chunks.append(w)
    attn = np.concatenate(attn_chunks) if attn_chunks else None
    return np.concatenate(preds), attn


def _attention_correlations(
    attn: np.ndarray | None,
    mask: np.ndarray,
    raw_dist: np.ndarray,
    raw_abs_sd: np.ndarray,
) -> dict[str, float]:
    if attn is None:
        return {"dist_m": np.nan, "abs_speed_delta": np.nan, "abs_relative_speed": np.nan}

    weights, dists, sds = [], [], []
    for i in range(len(attn)):
        for k in range(attn.shape[1]):
            if not mask[i, k]:
                continue
            weights.append(float(attn[i, k]))
            dists.append(float(raw_dist[i, k]))
            sds.append(float(raw_abs_sd[i, k]))

    if len(weights) < 10:
        return {"dist_m": np.nan, "abs_speed_delta": np.nan, "abs_relative_speed": np.nan}

    w_arr = np.array(weights)
    d_arr = np.array(dists)
    s_arr = np.array(sds)
    rho_dist, _ = spearmanr(w_arr, -d_arr)
    rho_sd, _ = spearmanr(w_arr, s_arr)
    return {
        "dist_m": float(rho_dist) if not np.isnan(rho_dist) else 0.0,
        "abs_speed_delta": float(rho_sd) if not np.isnan(rho_sd) else 0.0,
        "abs_relative_speed": float(rho_sd) if not np.isnan(rho_sd) else 0.0,
    }


@dataclass
class FoldResult:
    race: str
    n_train: int
    n_test: int
    majority_f1: float
    ego_f1: float
    snapshot_f1: float
    lstm_token_f1: float
    snapshot_attn: dict[str, float]
    lstm_attn: dict[str, float]


def run_fold(
    df: pd.DataFrame,
    team_gdfs: dict,
    held_out: str,
    spec: WindowSpec,
) -> FoldResult:
    print(f"\n=== LOOCV fold: test={held_out} ===", flush=True)
    train_df, test_df = split_by_races(df, held_out, venue="Bermuda")

    X_tr, y_tr, ego_stats = _build_windows_loocv(train_df, spec)
    X_te, y_te, _ = _build_windows_loocv(test_df, spec, ego_stats=ego_stats)
    if len(X_tr) == 0 or len(X_te) == 0:
        raise RuntimeError(f"Empty windows for fold {held_out}")

    cw = _class_weights(y_tr)
    majority = float(np.bincount(y_tr.astype(int)).argmax())
    maj_pred = np.full(len(y_te), majority)
    majority_f1 = evaluate_classification(y_te.astype(int), maj_pred.astype(int), average="macro")["f1"]

    tr_i, va_i = train_test_split(np.arange(len(X_tr)), test_size=0.1, random_state=42, stratify=y_tr.astype(int))
    ego_model, _, _, _ = run_training(
        X_tr[tr_i], y_tr[tr_i], X_tr[va_i], y_tr[va_i],
        lambda: BiLSTMClassifier(len(EGO_FEATURES), num_classes=3),
        task="classification",
        class_weights=cw,
        label=f"ego-{held_out}",
        verbose=False,
    )
    ego_loader = DataLoader(WindowedDataset(X_te, y_te), batch_size=64)
    ego_pred, _ = predict_lstm(ego_model, ego_loader, task="classification")
    ego_f1 = evaluate_classification(y_te.astype(int), ego_pred.astype(int), average="macro")["f1"]
    print(f"  majority={majority_f1:.3f} ego={ego_f1:.3f}", flush=True)

    X_tr_b, X_nb_tr, m_tr, y_tr_b, meta_tr = build_bubble_windows_loocv(
        train_df, spec, team_gdfs, lstm_tokens=False,
    )
    X_te_b, X_nb_te, m_te, y_te_b, meta_te = build_bubble_windows_loocv(
        test_df, spec, team_gdfs, lstm_tokens=False,
        ego_stats=meta_tr["ego_stats"],
        nb_stats=meta_tr["nb_stats"],
    )
    tr_i, va_i = train_test_split(np.arange(len(X_tr_b)), test_size=0.1, random_state=42, stratify=y_tr_b.astype(int))
    snap_model = BubbleAttentionClassifier(len(EGO_FEATURES), X_nb_tr.shape[-1], num_classes=3)
    snap_model, _ = _train_bubble(
        snap_model,
        X_tr_b[tr_i], X_nb_tr[tr_i], m_tr[tr_i], y_tr_b[tr_i],
        X_tr_b[va_i], X_nb_tr[va_i], m_tr[va_i], y_tr_b[va_i],
        cw, f"snapshot-{held_out}",
    )
    snap_pred, snap_attn = _predict_bubble(snap_model, X_te_b, X_nb_te, m_te)
    snapshot_f1 = evaluate_classification(y_te_b.astype(int), snap_pred.astype(int), average="macro")["f1"]
    snap_corr = _attention_correlations(snap_attn, m_te, meta_te["raw_dist"], meta_te["raw_abs_speed_delta"])

    X_tr_l, X_nb_tr_l, m_tr_l, y_tr_l, meta_tr_l = build_bubble_windows_loocv(
        train_df, spec, team_gdfs, lstm_tokens=True,
    )
    X_te_l, X_nb_te_l, m_te_l, y_te_l, meta_te_l = build_bubble_windows_loocv(
        test_df, spec, team_gdfs, lstm_tokens=True,
        ego_stats=meta_tr_l["ego_stats"],
        nb_seq_stats=meta_tr_l["nb_seq_stats"],
    )
    tr_i, va_i = train_test_split(np.arange(len(X_tr_l)), test_size=0.1, random_state=42, stratify=y_tr_l.astype(int))
    lstm_model = BubbleAttentionLSTMTokenClassifier(
        len(EGO_FEATURES), len(NB_SEQ_FEATURES), neighbour_seq_len=NB_SEQ_LEN, neighbour_hidden=32, num_classes=3,
    )
    lstm_model, _ = _train_bubble(
        lstm_model,
        X_tr_l[tr_i], X_nb_tr_l[tr_i], m_tr_l[tr_i], y_tr_l[tr_i],
        X_tr_l[va_i], X_nb_tr_l[va_i], m_tr_l[va_i], y_tr_l[va_i],
        cw, f"lstm-token-{held_out}", lstm_tokens=True,
    )
    lstm_pred, lstm_attn = _predict_bubble(lstm_model, X_te_l, X_nb_te_l, m_te_l)
    lstm_f1 = evaluate_classification(y_te_l.astype(int), lstm_pred.astype(int), average="macro")["f1"]
    lstm_corr = _attention_correlations(lstm_attn, m_te_l, meta_te_l["raw_dist"], meta_te_l["raw_abs_speed_delta"])

    print(
        f"  snapshot={snapshot_f1:.3f} lstm_token={lstm_f1:.3f} "
        f"lift={lstm_f1 - snapshot_f1:+.3f} attn_rho_speed={snap_corr['abs_relative_speed']:.3f}",
        flush=True,
    )

    return FoldResult(
        race=held_out,
        n_train=len(X_tr),
        n_test=len(X_te),
        majority_f1=majority_f1,
        ego_f1=ego_f1,
        snapshot_f1=snapshot_f1,
        lstm_token_f1=lstm_f1,
        snapshot_attn=snap_corr,
        lstm_attn=lstm_corr,
    )


def _export_attention_heatmap(folds: list[FoldResult], path: Path) -> None:
    metrics = ["dist_m", "abs_speed_delta", "abs_relative_speed"]
    variants = ["snapshot", "lstm_token"]
    z = []
    text = []
    for var in variants:
        row, trow = [], []
        for m in metrics:
            vals = [(f.snapshot_attn if var == "snapshot" else f.lstm_attn)[m] for f in folds]
            mean_v = float(np.nanmean(vals))
            row.append(mean_v)
            trow.append(f"{mean_v:.3f}")
        z.append(row)
        text.append(trow)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=variants,
        text=text,
        texttemplate="%{text}",
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Spearman ρ"),
    ))
    fig.update_layout(
        title="Attention weight correlations (LOOCV mean across folds)",
        xaxis_title="Neighbour attribute",
        yaxis_title="Model variant",
        height=400,
    )
    fig.write_html(str(path))


def run() -> dict:
    print("Loading Bermuda racing data...", flush=True)
    df = load_racing_boats()
    df = df[df["venue"] == "Bermuda"].copy()
    team_gdfs = _build_team_gdfs(df)

    spec = WindowSpec(
        feature_cols=EGO_FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=5,
        target_fn=_rank_delta_target,
    )

    folds: list[FoldResult] = []
    for race in BERMUDA_RACES:
        if race not in df["race_label"].unique():
            print(f"Skipping {race} (not in data)", flush=True)
            continue
        folds.append(run_fold(df, team_gdfs, race, spec))

    rows = []
    for f in folds:
        rows.append({
            "race": f.race,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "majority_macro_f1": round(f.majority_f1, 4),
            "ego_macro_f1": round(f.ego_f1, 4),
            "snapshot_macro_f1": round(f.snapshot_f1, 4),
            "lstm_token_macro_f1": round(f.lstm_token_f1, 4),
            "snapshot_vs_majority": round(f.snapshot_f1 - f.majority_f1, 4),
            "lstm_vs_snapshot": round(f.lstm_token_f1 - f.snapshot_f1, 4),
            "snapshot_attn_rho_speed": round(f.snapshot_attn["abs_relative_speed"], 4),
            "lstm_attn_rho_speed": round(f.lstm_attn["abs_relative_speed"], 4),
        })

    fold_df = pd.DataFrame(rows)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXPORT_DIR / "bubble_loocv_folds.csv"
    fold_df.to_csv(csv_path, index=False)

    def _summary(col: str) -> dict[str, float]:
        v = fold_df[col].to_numpy()
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v.tolist()}

    snapshot_sum = _summary("snapshot_macro_f1")
    lstm_sum = _summary("lstm_token_macro_f1")
    ego_sum = _summary("ego_macro_f1")
    maj_sum = _summary("majority_macro_f1")
    lstm_lift = float(np.mean(fold_df["lstm_vs_snapshot"]))
    attn_rho = float(np.nanmean([f.snapshot_attn["abs_relative_speed"] for f in folds] +
                                 [f.lstm_attn["abs_relative_speed"] for f in folds]))

    criteria = {
        "loocv_mean_macro_f1_gt_0.35": snapshot_sum["mean"] > 0.35,
        "loocv_std_lt_0.05": snapshot_sum["std"] < 0.05,
        "lstm_token_lift_ge_0.02": lstm_lift >= 0.02,
        "attention_rho_speed_gt_0.2": attn_rho > 0.2,
    }
    passed = all(criteria.values())

    results = {
        "experiment": "exp5_bubble_loocv",
        "n_folds": len(folds),
        "pass": passed,
        "criteria": criteria,
        "summary": {
            "majority": maj_sum,
            "ego_ref": ego_sum,
            "snapshot_attention": snapshot_sum,
            "lstm_token_attention": lstm_sum,
            "lstm_vs_snapshot_mean_lift": lstm_lift,
            "attention_rho_abs_relative_speed_mean": attn_rho,
        },
        "folds": rows,
        "outputs": {
            "folds_csv": str(csv_path),
            "results_json": str(EXPORT_DIR / "bubble_loocv_results.json"),
            "attention_heatmap_html": str(EXPORT_DIR / "bubble_loocv_attention_corr.html"),
        },
    }

    json_path = EXPORT_DIR / "bubble_loocv_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _export_attention_heatmap(folds, EXPORT_DIR / "bubble_loocv_attention_corr.html")

    print("\n=== LOOCV SUMMARY ===", flush=True)
    print(f"majority:  {maj_sum['mean']:.3f} ± {maj_sum['std']:.3f}", flush=True)
    print(f"ego_ref:   {ego_sum['mean']:.3f} ± {ego_sum['std']:.3f}", flush=True)
    print(f"snapshot:  {snapshot_sum['mean']:.3f} ± {snapshot_sum['std']:.3f}", flush=True)
    print(f"lstm_tok:  {lstm_sum['mean']:.3f} ± {lstm_sum['std']:.3f} (lift {lstm_lift:+.3f})", flush=True)
    print(f"attn ρ:    {attn_rho:.3f}", flush=True)
    print(f"PASS: {passed} | criteria={criteria}", flush=True)
    print(f"Outputs: {csv_path}", flush=True)

    return results


if __name__ == "__main__":
    run()
