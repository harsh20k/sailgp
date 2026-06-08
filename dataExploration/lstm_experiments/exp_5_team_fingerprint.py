"""Experiment 5 — Cross-boat team fingerprinting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_AWA,
    COL_DB_CANT_P,
    COL_HEEL,
    COL_PITCH,
    COL_RH_P,
    COL_SPEED,
    COL_TWA,
    COL_VMG,
    COL_WING_ROT,
    COL_YAW,
    WindowSpec,
    build_windows,
    load_racing_boats,
    normalize_per_race,
    teams_in_both_venues,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    evaluate_classification,
    export_eval_html,
    logistic_baseline,
    pick_best_baseline,
    run_training,
    signal_detected,
    top_k_accuracy,
)
from dataExploration.lstm_experiments.shared.models import LSTMClassifier

FEATURES = [
    COL_SPEED,
    COL_VMG,
    COL_TWA,
    COL_HEEL,
    COL_PITCH,
    COL_WING_ROT,
    COL_DB_CANT_P,
    COL_RH_P,
    COL_AWA,
    COL_YAW,
]
SEQ_LEN = 60


def _encode_teams(df, encoder: LabelEncoder) -> np.ndarray:
    return encoder.transform(df["team"].astype(str))


def run() -> ExperimentResult:
    df = load_racing_boats()
    shared = teams_in_both_venues(df)
    if not shared:
        shared = sorted(df["team"].unique())

    df = df[df["team"].isin(shared)]
    df = normalize_per_race(df, FEATURES)

    train_df = df[df["venue"] == "Bermuda"]
    test_df = df[df["venue"] == "Halifax"]

    encoder = LabelEncoder()
    encoder.fit(sorted(df["team"].unique()))
    n_classes = len(encoder.classes_)

    # 80/20 split within Bermuda for val
    races = sorted(train_df["race_label"].unique())
    val_races = set(races[-2:])
    val_df = train_df[train_df["race_label"].isin(val_races)]
    train_df = train_df[~train_df["race_label"].isin(val_races)]

    def team_target(gdf, end_idx: int) -> float:
        return float(encoder.transform([str(gdf["team"].iloc[0])])[0])

    spec = WindowSpec(
        feature_cols=FEATURES,
        seq_len=SEQ_LEN,
        horizon=0,
        stride=10,
        target_fn=team_target,
    )

    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)
    X_test, y_test, _ = build_windows(test_df, spec)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp5")

    _, history, y_pred_val, y_prob_val = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: LSTMClassifier(len(FEATURES), num_classes=n_classes),
        task="classification",
        label="exp5-team",
    )

    lstm_metrics = evaluate_classification(y_val.astype(int), y_pred_val.astype(int))
    top3 = top_k_accuracy(y_val.astype(int), y_prob_val, k=3) if y_prob_val is not None else 0.0

    majority = float(np.bincount(y_train.astype(int), minlength=n_classes).argmax())
    maj_pred = np.full(len(y_val), majority)
    maj_acc = float((maj_pred == y_val).mean())

    random_acc = 1.0 / n_classes
    log_pred = logistic_baseline(X_train, y_train, X_val, multi_class=True)
    log_metrics = evaluate_classification(y_val.astype(int), log_pred.astype(int))

    baselines = {"majority": maj_acc, "random": random_acc, "logistic": log_metrics["accuracy"]}
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    result = ExperimentResult(
        experiment_id="exp5",
        name="Team Fingerprinting",
        task="Team fingerprint",
        lstm_metric_name="Top-1 accuracy",
        lstm_metric=lstm_metrics["accuracy"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=lstm_metrics["accuracy"] - best_metric,
        has_signal=signal_detected(lstm_metrics["accuracy"], best_metric, higher_is_better=True, min_delta=0.03),
        details={
            "top3_accuracy": top3,
            "f1_weighted": lstm_metrics["f1"],
            "shared_teams": shared,
            "n_classes": n_classes,
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
