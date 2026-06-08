"""Shared TabNet variation runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sailgp_analysis.tabnet.config import VARIATION_PARAMS, VariationName
from sailgp_analysis.tabnet.data_prep import PreparedDataset
from sailgp_analysis.tabnet.evaluate import (
    EvalResult,
    check_attention_interpretable,
    classification_metrics,
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


def _eval_set(ds: PreparedDataset) -> tuple[np.ndarray, np.ndarray]:
    if len(ds.X_val) > 0:
        return ds.X_val, ds.y_val
    # Fallback: last 10% of train
    n = max(1, len(ds.X_train) // 10)
    return ds.X_train[-n:], ds.y_train[-n:]


def run_variation(
    variation: VariationName,
    ds: PreparedDataset,
    output_dir: Path | None = None,
    extra_meta: dict | None = None,
) -> EvalResult:
    output_dir = output_dir or default_output_dir()
    params = VARIATION_PARAMS[variation]
    X_val, y_val = _eval_set(ds)

    if len(ds.y_test) == 0:
        # Fallback: use val as test when venue has no held-out rows (e.g. v4 Halifax-only test)
        X_test, y_test = ds.X_val, ds.y_val
    else:
        X_test, y_test = ds.X_test, ds.y_test

    model = train_tabnet(ds.X_train, ds.y_train, X_val, y_val, ds.task, params)
    preds, prob = predict_tabnet(model, X_test, ds.task)

    if ds.task == "regression":
        tabnet_metrics = regression_metrics(y_test, preds)
        if variation == "v3":
            tabnet_metrics.update(rank_metrics(y_test, preds))
    else:
        tabnet_metrics = classification_metrics(y_test, preds, prob)

    rf_metrics, lin_metrics, rf_imp, _ = fit_baselines(
        ds.X_train, ds.y_train, X_test, y_test, ds.task, ds.feature_names
    )
    importance = extract_feature_importance(model, ds.feature_names)
    beats, meaningful, reason = compute_verdict(
        variation, ds.task, tabnet_metrics, rf_metrics, importance
    )
    interpretable, hits = check_attention_interpretable(importance, variation)

    result = EvalResult(
        variation=variation,
        task=ds.task,
        n_train=len(ds.y_train),
        n_val=len(ds.y_val),
        n_test=len(y_test),
        tabnet_metrics=tabnet_metrics,
        baseline_metrics=rf_metrics,
        linear_metrics=lin_metrics,
        beats_baseline=beats,
        attention_interpretable=interpretable,
        meaningful=meaningful,
        verdict_reason=reason,
        feature_importance=importance,
        baseline_importance=rf_imp,
        expected_dominant_hits=hits,
        extra=extra_meta or {},
    )

    plot_attention_heatmap(
        importance,
        variation,
        output_dir / f"{variation}_attention.png",
    )
    save_result(result, output_dir)
    return result
