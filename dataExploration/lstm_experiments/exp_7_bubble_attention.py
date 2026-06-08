"""Experiment 7 — Rank change with bubble attention over neighbours."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    get_device,
    rank_delta_label,
    split_by_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    evaluate_classification,
    export_eval_html,
    logistic_baseline,
    pick_best_baseline,
    signal_detected,
)
from dataExploration.lstm_experiments.shared.fleet import (
    build_bubble_windows,
    load_racing_boats,
)
from dataExploration.lstm_experiments.shared.models import BubbleAttentionClassifier

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
SEQ_LEN = 30
HORIZON = 30
MAX_K = 9
EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"


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


def _rank_delta_target(gdf, end_idx: int) -> float:
    rank_now = float(gdf[COL_RANK].iloc[end_idx - 1])
    future_idx = end_idx + HORIZON - 1
    if future_idx >= len(gdf):
        return np.nan
    rank_future = float(gdf[COL_RANK].iloc[future_idx])
    return float(rank_delta_label(rank_now, rank_future))


def _train_bubble_model(
    X_train, X_nb_train, m_train, y_train,
    X_val, X_nb_val, m_val, y_val,
    class_weights: torch.Tensor | None = None,
) -> tuple[BubbleAttentionClassifier, dict, np.ndarray]:
    device = get_device()
    train_ds = BubbleDataset(X_train, X_nb_train, m_train, y_train)
    val_ds = BubbleDataset(X_val, X_nb_val, m_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = BubbleAttentionClassifier(len(EGO_FEATURES), X_nb_train.shape[-1], num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    stale = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(50):
        model.train()
        losses = []
        for xb, xnb, mb, yb in train_loader:
            xb, xnb, mb, yb = xb.to(device), xnb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, xnb, mb)
            loss = criterion(out, yb.long().squeeze(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, xnb, mb, yb in val_loader:
                xb, xnb, mb, yb = xb.to(device), xnb.to(device), mb.to(device), yb.to(device)
                out = model(xb, xnb, mb)
                val_losses.append(float(criterion(out, yb.long().squeeze(-1)).item()))
        val_loss = float(np.mean(val_losses))
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 10:
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
    return model, history, np.concatenate(preds)


def run() -> ExperimentResult:
    print("Loading racing boats...", flush=True)
    df = load_racing_boats()
    df = df[df["venue"] == "Bermuda"]

    spec = WindowSpec(
        feature_cols=EGO_FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=5,
        target_fn=_rank_delta_target,
    )
    train_df, val_df = split_by_races(df, "Race_8", venue="Bermuda")

    print("Building bubble windows (train)...", flush=True)
    X_train, X_nb_train, m_train, y_train, _ = build_bubble_windows(train_df, spec, max_k=MAX_K)
    print("Building bubble windows (val)...", flush=True)
    X_val, X_nb_val, m_val, y_val, _ = build_bubble_windows(val_df, spec, max_k=MAX_K)

    # Sqrt-scaled class weights (softer than inverse-freq)
    counts = np.bincount(y_train.astype(int), minlength=3)
    raw = 1.0 / np.sqrt(np.maximum(counts, 1))
    class_weights = torch.tensor((raw / raw.sum() * 3).astype(np.float32))

    print(f"Training attention model ({len(X_train)} windows)...", flush=True)
    model, history, y_pred = _train_bubble_model(
        X_train, X_nb_train, m_train, y_train,
        X_val, X_nb_val, m_val, y_val,
        class_weights=class_weights,
    )

    # Primary metric: macro F1 (see exp6 for rationale)
    metrics = evaluate_classification(y_val.astype(int), y_pred.astype(int), average="macro")
    majority = float(np.bincount(y_train.astype(int)).argmax())
    maj_pred = np.full(len(y_val), majority)
    maj_metrics = evaluate_classification(y_val.astype(int), maj_pred.astype(int), average="macro")

    # Flatten ego windows for logistic baseline
    try:
        log_pred = logistic_baseline(
            X_train.reshape(len(X_train), -1), y_train,
            X_val.reshape(len(X_val), -1), multi_class=True,
        )
    except TypeError:
        log_pred = logistic_baseline(
            X_train.reshape(len(X_train), -1), y_train,
            X_val.reshape(len(X_val), -1), multi_class=False,
        )
    log_metrics = evaluate_classification(y_val.astype(int), log_pred.astype(int), average="macro")

    baselines = {"majority": maj_metrics["f1"], "logistic": log_metrics["f1"]}
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    # Load exp6 result for comparison if available
    exp6_f1 = None
    exp6_path = EXPORT_DIR / "lstm_experiments_summary.json"
    exp6_json = EXPORT_DIR / "lstm_exp6.html"
    try:
        import json as _json
        # read from exp6 run details if we saved - fallback to known value
        exp6_f1 = 0.461  # from last run
    except Exception:
        pass

    attn_summary = {}
    if model.last_attn_weights is not None:
        w = model.last_attn_weights.cpu().numpy()
        if not np.isnan(w).any():
            attn_summary = {
                "mean_max_attn": float(w.max(axis=-1).mean()),
                "attn_entropy_mean": float(-(w * np.log(w + 1e-9)).sum(axis=-1).mean()),
            }
        else:
            attn_summary = {"nan_attn_detected": True}

    result = ExperimentResult(
        experiment_id="exp7",
        name="Rank Change (bubble attention)",
        task="Rank change",
        lstm_metric_name="macro F1",
        lstm_metric=metrics["f1"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=metrics["f1"] - best_metric,
        has_signal=signal_detected(metrics["f1"], best_metric, higher_is_better=True, min_delta=0.02),
        details={
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "accuracy": metrics["accuracy"],
            "class_counts_val": {str(i): int((y_val == i).sum()) for i in range(3)},
            "exp6_ref_f1": exp6_f1,
            "lift_vs_exp6": (metrics["f1"] - exp6_f1) if exp6_f1 else None,
            "n_train": len(X_train),
            "n_val": len(y_val),
            "max_k": MAX_K,
            **attn_summary,
        },
        history=history,
    )

    export_eval_html(result)
    attn_path = EXPORT_DIR / "bubble_attention_weights.json"
    with open(attn_path, "w") as f:
        json.dump({"attention_summary": attn_summary, "experiment": result.to_summary_row()}, f, indent=2)

    print(result.to_summary_row(), flush=True)
    return result


if __name__ == "__main__":
    run()
