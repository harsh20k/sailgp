"""Experiment 2 — Foiling state transition detection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_AWA,
    COL_DB_RAKE_P,
    COL_HEEL,
    COL_PITCH,
    COL_RH_BOW,
    COL_RH_P,
    COL_RH_S,
    COL_SPEED,
    COL_TWA,
    COL_YAW,
    WindowSpec,
    build_windows,
    foiling_label,
    foiling_rule,
    load_racing_boats,
    split_venue_races,
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
from dataExploration.lstm_experiments.shared.models import LSTMFutureBinaryClassifier

FEATURES = [
    COL_PITCH,
    COL_HEEL,
    COL_YAW,
    COL_RH_P,
    COL_RH_S,
    COL_RH_BOW,
    COL_SPEED,
    COL_TWA,
    COL_AWA,
    COL_DB_RAKE_P,
]
SEQ_LEN = 30
HORIZON = 10


def _foiling_target(gdf, end_idx: int) -> np.ndarray:
    labels = []
    for h in range(1, HORIZON + 1):
        idx = end_idx + h - 1
        if idx >= len(gdf):
            return np.full(HORIZON, np.nan)
        labels.append(float(foiling_label(gdf.iloc[idx])))
    return np.array(labels, dtype=np.float32)


def _rule_target(gdf, end_idx: int) -> np.ndarray:
    labels = []
    for h in range(1, HORIZON + 1):
        idx = end_idx + h - 1
        if idx >= len(gdf):
            return np.full(HORIZON, np.nan)
        labels.append(float(foiling_rule(gdf.iloc[idx])))
    return np.array(labels, dtype=np.float32)


def _collect_rule_preds(df, spec: WindowSpec) -> np.ndarray:
    ys = []
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    for _, gdf in df.groupby(group_cols, sort=False):
        gdf = gdf.sort_index()
        n = len(gdf)
        end = n - spec.horizon
        if end <= spec.seq_len:
            continue
        for start in range(0, end - spec.seq_len, spec.stride):
            end_idx = start + spec.seq_len
            y = _rule_target(gdf, end_idx)
            if np.isnan(y).any():
                continue
            ys.append(y)
    return np.stack(ys).astype(np.float32) if ys else np.empty((0, HORIZON))


def run() -> ExperimentResult:
    df = load_racing_boats()
    train_df, _, _ = split_venue_races(
        df,
        train_venue="Bermuda",
        train_races={f"Race_{i}" for i in range(1, 8)},
        val_venue="Bermuda",
        val_races={"Race_8"},
        test_venue="Halifax",
    )
    val_df = df[(df["venue"] == "Bermuda") & (df["race_label"] == "Race_8")]
    test_df = df[df["venue"] == "Halifax"]

    spec = WindowSpec(
        feature_cols=FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=2,
        target_fn=_foiling_target,
    )

    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)
    X_test, y_test, _ = build_windows(test_df, spec)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp2")

    print(f"[exp2] building windows: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}", flush=True)

    _, history, y_pred_val, y_prob_val = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: LSTMFutureBinaryClassifier(len(FEATURES), horizon=HORIZON),
        task="binary_future",
        label="exp2-foiling",
    )

    y_flat = y_val.reshape(-1)
    pred_flat = y_pred_val.reshape(-1)
    prob_flat = y_prob_val.reshape(-1) if y_prob_val is not None else None
    lstm_metrics = evaluate_classification(y_flat, pred_flat, prob_flat)

    y_train_bin = y_train[:, 0]
    y_val_bin = y_val[:, 0]
    log_pred = logistic_baseline(X_train, y_train_bin, X_val)
    log_metrics = evaluate_classification(y_val_bin, log_pred)

    rule_pred = _collect_rule_preds(val_df, spec)
    if len(rule_pred) == len(y_val):
        rule_metrics = evaluate_classification(y_flat, rule_pred.reshape(-1))
    else:
        rule_metrics = {"f1": 0.0}

    baselines = {"logistic_t+1": log_metrics["f1"], "rule_based": rule_metrics["f1"]}
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    result = ExperimentResult(
        experiment_id="exp2",
        name="Foiling Transition Detection",
        task="Foiling transition",
        lstm_metric_name="F1",
        lstm_metric=lstm_metrics["f1"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=lstm_metrics["f1"] - best_metric,
        has_signal=signal_detected(lstm_metrics["f1"], best_metric, higher_is_better=True, min_delta=0.02),
        details={
            "roc_auc": lstm_metrics.get("roc_auc", 0),
            "f1_macro": lstm_metrics["f1_macro"],
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        },
        history=history,
    )
    export_eval_html(result)
    return result


if __name__ == "__main__":
    print(run().to_summary_row())
