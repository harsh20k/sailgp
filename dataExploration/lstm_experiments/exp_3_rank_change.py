"""Experiment 3 — Race rank change prediction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

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
    build_windows,
    load_racing_boats,
    rank_delta_label,
    split_by_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    evaluate_classification,
    export_eval_html,
    logistic_baseline,
    pick_best_baseline,
    run_training,
    signal_detected,
    xgboost_baseline,
)
from dataExploration.lstm_experiments.shared.models import BiLSTMClassifier

FEATURES = [
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


def _rank_delta_target(gdf, end_idx: int) -> float:
    rank_now = float(gdf[COL_RANK].iloc[end_idx - 1])
    future_idx = end_idx + HORIZON - 1
    if future_idx >= len(gdf):
        return np.nan
    rank_future = float(gdf[COL_RANK].iloc[future_idx])
    return float(rank_delta_label(rank_now, rank_future))


def run() -> ExperimentResult:
    df = load_racing_boats()
    df = df[df["venue"] == "Bermuda"]

    spec = WindowSpec(
        feature_cols=FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=5,
        target_fn=_rank_delta_target,
    )

    # Leave-one-race-out: hold out Race_8 for validation metrics
    train_df, val_df = split_by_races(df, "Race_8", venue="Bermuda")

    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp3")

    counts = np.bincount(y_train.astype(int), minlength=3)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * 3
    class_weights = torch.tensor(weights, dtype=torch.float32)

    _, history, y_pred_val, _ = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: BiLSTMClassifier(len(FEATURES), num_classes=3),
        task="classification",
        class_weights=class_weights,
        label="exp3-rank",
    )

    lstm_metrics = evaluate_classification(y_val.astype(int), y_pred_val.astype(int))

    majority = float(np.bincount(y_train.astype(int)).argmax())
    maj_pred = np.full(len(y_val), majority)
    maj_metrics = evaluate_classification(y_val.astype(int), maj_pred.astype(int))

    xgb_pred = xgboost_baseline(X_train, y_train, X_val)
    xgb_metrics = evaluate_classification(y_val.astype(int), xgb_pred.astype(int))

    log_pred = logistic_baseline(X_train, y_train, X_val, multi_class=True)
    log_metrics = evaluate_classification(y_val.astype(int), log_pred.astype(int))

    baselines = {
        "majority": maj_metrics["f1"],
        "xgboost": xgb_metrics["f1"],
        "logistic": log_metrics["f1"],
    }
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    result = ExperimentResult(
        experiment_id="exp3",
        name="Rank Change Prediction",
        task="Rank change",
        lstm_metric_name="weighted F1",
        lstm_metric=lstm_metrics["f1"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=lstm_metrics["f1"] - best_metric,
        has_signal=signal_detected(lstm_metrics["f1"], best_metric, higher_is_better=True, min_delta=0.02),
        details={
            "f1_macro": lstm_metrics["f1_macro"],
            "accuracy": lstm_metrics["accuracy"],
            "class_counts_val": {str(i): int((y_val == i).sum()) for i in range(3)},
            "n_train": len(X_train),
            "n_val": len(X_val),
        },
        history=history,
    )
    export_eval_html(result)
    return result


if __name__ == "__main__":
    print(run().to_summary_row())
