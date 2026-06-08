"""Experiment 4 — Wing trim / VMG efficiency (VMG/TWS ratio)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    CA_COLS,
    COL_DB_CANT_P,
    COL_HEEL,
    COL_JIB_LEAD,
    COL_JIB_SHEET,
    COL_PITCH,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    COL_WING_ROT,
    COL_WING_TWIST,
    WindowSpec,
    build_windows,
    load_racing_boats,
    split_venue_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    evaluate_regression,
    export_eval_html,
    pick_best_baseline,
    polar_bin_baseline,
    ridge_baseline,
    rmse,
    run_training,
    signal_detected,
)
from dataExploration.lstm_experiments.shared.models import AttentionLSTMRegressor

FEATURES = (
    [COL_WING_ROT, COL_WING_TWIST]
    + CA_COLS
    + [COL_JIB_LEAD, COL_JIB_SHEET, COL_DB_CANT_P, COL_TWA, COL_TWS, COL_HEEL, COL_PITCH]
)
SEQ_LEN = 20
HORIZON = 1


def _vmg_ratio_target(gdf, end_idx: int) -> float:
    row = gdf.iloc[end_idx - 1]
    tws = float(row.get(COL_TWS, np.nan))
    vmg = float(row.get(COL_VMG, np.nan))
    if np.isnan(tws) or np.isnan(vmg) or tws < 1.0:
        return np.nan
    return vmg / tws


def _collect_twa(df, spec: WindowSpec) -> np.ndarray:
    twas = []
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    for _, gdf in df.groupby(group_cols, sort=False):
        gdf = gdf.sort_index()
        if COL_TWA not in gdf.columns:
            continue
        n = len(gdf)
        end = n - spec.horizon
        if end <= spec.seq_len:
            continue
        for start in range(0, end - spec.seq_len, spec.stride):
            end_idx = start + spec.seq_len
            val = float(gdf[COL_TWA].iloc[end_idx - 1])
            if np.isnan(val):
                continue
            twas.append(val)
    return np.array(twas, dtype=np.float32)


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
        stride=2,
        target_fn=_vmg_ratio_target,
    )

    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)
    X_test, y_test, _ = build_windows(test_df, spec)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp4")

    _, history, y_pred_val, _ = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: AttentionLSTMRegressor(len(FEATURES)),
        task="regression",
        label="exp4-vmg",
    )

    lstm_metrics = evaluate_regression(y_val, y_pred_val)

    twa_train = _collect_twa(train_df, spec)[: len(y_train)]
    twa_val = _collect_twa(val_df, spec)[: len(y_val)]
    polar_pred = polar_bin_baseline(twa_train, y_train, twa_val)
    polar_metrics = evaluate_regression(y_val, polar_pred)

    ridge_pred = ridge_baseline(X_train, y_train, X_val)
    ridge_metrics = evaluate_regression(y_val, ridge_pred)

    baselines = {"polar_bin": polar_metrics["r2"], "ridge": ridge_metrics["r2"]}
    best_name, best_metric = pick_best_baseline(baselines, higher_is_better=True)

    result = ExperimentResult(
        experiment_id="exp4",
        name="VMG Efficiency (VMG/TWS)",
        task="VMG efficiency",
        lstm_metric_name="R²",
        lstm_metric=lstm_metrics["r2"],
        baseline_metrics=baselines,
        best_baseline_name=best_name,
        best_baseline_metric=best_metric,
        delta=lstm_metrics["r2"] - best_metric,
        has_signal=signal_detected(lstm_metrics["r2"], best_metric, higher_is_better=True, min_delta=0.02),
        details={
            "rmse": lstm_metrics["rmse"],
            "polar_rmse": polar_metrics["rmse"],
            "ridge_r2": ridge_metrics["r2"],
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
