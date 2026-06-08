"""Experiment 1 — Boat speed forecasting (next 5 s)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_AWA,
    COL_AWS,
    COL_DB_CANT_P,
    COL_HEEL,
    COL_PITCH,
    COL_RH_P,
    COL_RH_S,
    COL_SPEED,
    COL_TWA,
    COL_TWS,
    COL_WING_ROT,
    WindowSpec,
    WindowedDataset,
    build_windows,
    get_device,
    load_racing_boats,
    split_venue_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    evaluate_regression,
    export_eval_html,
    pick_best_baseline,
    predict_lstm,
    ridge_baseline,
    rmse,
    run_training,
    signal_detected,
)
from dataExploration.lstm_experiments.shared.models import LSTMRegressor

FEATURES = [
    COL_TWA,
    COL_TWS,
    COL_AWA,
    COL_AWS,
    COL_HEEL,
    COL_PITCH,
    COL_RH_P,
    COL_RH_S,
    COL_WING_ROT,
    COL_DB_CANT_P,
]
SEQ_LEN = 30
HORIZON = 5


def _build_persist_y(df, spec: WindowSpec) -> np.ndarray:
    """Speed at last input timestep (persistence baseline)."""
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
            val = float(gdf[COL_SPEED].iloc[end_idx - 1])
            if np.isnan(val):
                continue
            ys.append(val)
    return np.array(ys, dtype=np.float32)


def run() -> ExperimentResult:
    df = load_racing_boats()
    train_df, val_df, test_df = split_venue_races(
        df,
        train_venue="Bermuda",
        train_races={f"Race_{i}" for i in range(1, 7)},
        val_venue="Bermuda",
        val_races={f"Race_{i}" for i in range(7, 9)},
        test_venue="Halifax",
    )

    spec = WindowSpec(
        feature_cols=FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=5,
        target_col=COL_SPEED,
    )

    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)
    X_test, y_test, _ = build_windows(test_df, spec)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp1 — check DataChallenge_Export paths")

    print(f"[exp1] building windows: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}", flush=True)

    model, history, y_pred_val, _ = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: LSTMRegressor(len(FEATURES)),
        task="regression",
        label="exp1-speed",
    )
    lstm_metrics = evaluate_regression(y_val, y_pred_val)

    y_persist_val = _build_persist_y(val_df, spec)[: len(y_val)]
    y_persist_test = _build_persist_y(test_df, spec)[: len(y_test)]

    persist_val_rmse = rmse(y_val, y_persist_val) if len(y_persist_val) == len(y_val) else float("inf")
    ridge_val = ridge_baseline(X_train, y_train, X_val)
    ridge_val_rmse = rmse(y_val, ridge_val)

    baselines = {"persistence": persist_val_rmse, "ridge": ridge_val_rmse}
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=False)

    test_loader = DataLoader(WindowedDataset(X_test, y_test), batch_size=64)
    y_pred_test, _ = predict_lstm(model, test_loader, task="regression", device=get_device())
    test_metrics = evaluate_regression(y_test, y_pred_test)
    persist_test_rmse = rmse(y_test, y_persist_test) if len(y_persist_test) == len(y_test) else float("inf")

    delta = best_metric - lstm_metrics["rmse"]
    result = ExperimentResult(
        experiment_id="exp1",
        name="Boat Speed Forecast (t+5s)",
        task="Speed forecast",
        lstm_metric_name="RMSE",
        lstm_metric=lstm_metrics["rmse"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=delta,
        has_signal=signal_detected(lstm_metrics["rmse"], best_metric, higher_is_better=False, min_delta=0.5),
        details={
            "val_mae": lstm_metrics["mae"],
            "val_r2": lstm_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_persistence_rmse": persist_test_rmse,
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
