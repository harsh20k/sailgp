"""V3 — Race rank prediction with LOOCV."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sailgp_analysis.tabnet.config import VARIATION_PARAMS
from sailgp_analysis.tabnet.data_prep import build_v3_dataset, build_v3_loocv_folds
from sailgp_analysis.tabnet.evaluate import (
    EvalResult,
    compute_verdict,
    default_output_dir,
    extract_feature_importance,
    fit_baselines,
    plot_attention_heatmap,
    predict_tabnet,
    rank_metrics,
    regression_metrics,
    save_result,
    train_tabnet,
)
from sailgp_analysis.tabnet.runner import run_variation


def run(data_root=None, output_dir: Path | None = None) -> EvalResult:
    output_dir = output_dir or default_output_dir()
    ds = build_v3_dataset(data_root)
    result = run_variation("v3", ds, output_dir)

    # LOOCV supplement
    folds = build_v3_loocv_folds(data_root)
    rhos = []
    params = VARIATION_PARAMS["v3"]
    for race, fold_ds in folds:
        X_val, y_val = fold_ds.X_val, fold_ds.y_val
        if len(X_val) == 0:
            X_val, y_val = fold_ds.X_train[-max(1, len(fold_ds.X_train) // 10):], fold_ds.y_train[-max(1, len(fold_ds.y_train) // 10):]
        model = train_tabnet(fold_ds.X_train, fold_ds.y_train, X_val, y_val, "regression", params)
        preds, _ = predict_tabnet(model, fold_ds.X_test, "regression")
        m = rank_metrics(fold_ds.y_test, preds)
        rhos.append(m["spearman_rho"])

    if rhos:
        result.loocv_spearman_mean = float(np.mean(rhos))
        result.loocv_spearman_std = float(np.std(rhos))
        _, meaningful, reason = compute_verdict(
            "v3",
            "regression",
            result.tabnet_metrics,
            result.baseline_metrics,
            result.feature_importance,
            loocv_std=result.loocv_spearman_std,
        )
        result.meaningful = meaningful
        result.verdict_reason = reason + f"; LOOCV rho mean={result.loocv_spearman_mean:.3f} std={result.loocv_spearman_std:.3f}"
        result.extra["loocv_rhos"] = rhos
        save_result(result, output_dir)

    return result
