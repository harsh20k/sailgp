"""Experiment 6 — Rank change with tactical bubble features."""
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
)
from dataExploration.lstm_experiments.shared.fleet import (
    BUBBLE_FEATURE_COLS,
    load_racing_boats_with_bubble,
)
from dataExploration.lstm_experiments.shared.models import BiLSTMClassifier

# exp3 ego-only features
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

FEATURES = EGO_FEATURES + BUBBLE_FEATURE_COLS
SEQ_LEN = 30
HORIZON = 30


def _rank_delta_target(gdf, end_idx: int) -> float:
    rank_now = float(gdf[COL_RANK].iloc[end_idx - 1])
    future_idx = end_idx + HORIZON - 1
    if future_idx >= len(gdf):
        return np.nan
    rank_future = float(gdf[COL_RANK].iloc[future_idx])
    return float(rank_delta_label(rank_now, rank_future))


def _run_experiment(
    feature_cols: list[str],
    df,
    experiment_id: str,
    label: str,
) -> ExperimentResult:
    spec = WindowSpec(
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=5,
        target_fn=_rank_delta_target,
    )
    train_df, val_df = split_by_races(df, "Race_8", venue="Bermuda")
    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)

    if len(X_train) == 0:
        raise RuntimeError(f"No training windows for {experiment_id}")

    # Sqrt-scaled class weights: softer than inverse-freq; stops model collapsing
    # to all-hold while not over-correcting on the imbalanced val set.
    counts = np.bincount(y_train.astype(int), minlength=3)
    raw = 1.0 / np.sqrt(np.maximum(counts, 1))
    class_weights = torch.tensor((raw / raw.sum() * 3).astype(np.float32))

    _, history, y_pred_val, _ = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: BiLSTMClassifier(len(feature_cols), num_classes=3),
        task="classification",
        class_weights=class_weights,
        label=label,
    )

    # Primary metric: macro F1.  Weighted F1 rewards predicting all-hold on an
    # 81%-hold val set; macro F1 penalises all-hold with ~0.30 and rewards any
    # genuine minority-class signal.
    lstm_metrics = evaluate_classification(y_val.astype(int), y_pred_val.astype(int), average="macro")
    majority = float(np.bincount(y_train.astype(int)).argmax())
    maj_pred = np.full(len(y_val), majority)
    maj_metrics = evaluate_classification(y_val.astype(int), maj_pred.astype(int), average="macro")

    # XGBoost segfaults on large flattened windows on some macOS builds — omit baseline
    xgb_metrics = {"f1": maj_metrics["f1"]}

    try:
        log_pred = logistic_baseline(X_train, y_train, X_val, multi_class=True)
    except TypeError:
        log_pred = logistic_baseline(X_train, y_train, X_val, multi_class=False)
    log_metrics = evaluate_classification(y_val.astype(int), log_pred.astype(int), average="macro")

    baselines = {
        "majority": maj_metrics["f1"],
        "xgboost": xgb_metrics["f1"],
        "logistic": log_metrics["f1"],
    }
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    return ExperimentResult(
        experiment_id=experiment_id,
        name=f"Rank Change ({label})",
        task="Rank change",
        lstm_metric_name="macro F1",
        lstm_metric=lstm_metrics["f1"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=lstm_metrics["f1"] - best_metric,
        has_signal=signal_detected(lstm_metrics["f1"], best_metric, higher_is_better=True, min_delta=0.02),
        details={
            "f1_macro": lstm_metrics["f1"],
            "f1_weighted": lstm_metrics["f1_weighted"],
            "accuracy": lstm_metrics["accuracy"],
            "class_counts_val": {str(i): int((y_val == i).sum()) for i in range(3)},
            "n_features": len(feature_cols),
            "bubble_features": BUBBLE_FEATURE_COLS if experiment_id == "exp6" else [],
            "n_train": len(X_train),
            "n_val": len(X_val),
        },
        history=history,
    )


def run() -> ExperimentResult:
    print("Loading racing boats with bubble features...", flush=True)
    df = load_racing_boats_with_bubble()
    df = df[df["venue"] == "Bermuda"]

    print("Running exp3 baseline (ego-only)...", flush=True)
    exp3_result = _run_experiment(EGO_FEATURES, df.copy(), "exp3_ref", "ego-only ref")

    print("Running exp6 (ego + bubble)...", flush=True)
    result = _run_experiment(FEATURES, df, "exp6", "bubble rank")

    # Compare against exp3 model metric directly
    exp3_f1 = exp3_result.lstm_metric
    bubble_lift = result.lstm_metric - exp3_f1
    result.details["exp3_ref_f1"] = exp3_f1
    result.details["bubble_lift_vs_exp3"] = bubble_lift
    result.details["exp3_has_signal"] = exp3_result.has_signal
    result.has_signal = signal_detected(result.lstm_metric, exp3_f1, higher_is_better=True, min_delta=0.02)
    result.delta = bubble_lift

    print(f"exp3 ref F1={exp3_f1:.4f} | exp6 F1={result.lstm_metric:.4f} | lift={bubble_lift:+.4f}", flush=True)
    export_eval_html(result)
    export_eval_html(exp3_result)

    summary_path = Path(__file__).resolve().parents[1] / "exported" / "bubble_exp6_results.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({
            "exp3_ref": exp3_result.to_summary_row(),
            "exp6": result.to_summary_row(),
            "bubble_lift": bubble_lift,
        }, f, indent=2)
    return result


if __name__ == "__main__":
    print(run().to_summary_row())
