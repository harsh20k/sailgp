"""V5 — Cross-venue speed regression generalization."""
from __future__ import annotations

from pathlib import Path

from sailgp_analysis.tabnet.data_prep import build_cross_venue_dataset
from sailgp_analysis.tabnet.evaluate import (
    EvalResult,
    compute_verdict,
    default_output_dir,
    extract_feature_importance,
    fit_baselines,
    plot_attention_heatmap,
    predict_tabnet,
    regression_metrics,
    save_result,
    train_tabnet,
)
from sailgp_analysis.tabnet.config import VARIATION_PARAMS
from sailgp_analysis.tabnet.runner import _eval_set


def run(data_root=None, output_dir: Path | None = None) -> EvalResult:
    output_dir = output_dir or default_output_dir()
    params = VARIATION_PARAMS["v5"]
    directions = {}

    for train_v, test_v in [("Bermuda", "Halifax"), ("Halifax", "Bermuda")]:
        ds = build_cross_venue_dataset(train_v, test_v, data_root)
        X_val, y_val = _eval_set(ds)
        model = train_tabnet(ds.X_train, ds.y_train, X_val, y_val, ds.task, params)
        preds, _ = predict_tabnet(model, ds.X_test, ds.task)
        tabnet_m = regression_metrics(ds.y_test, preds)
        rf_m, lin_m, rf_imp, _ = fit_baselines(
            ds.X_train, ds.y_train, ds.X_test, ds.y_test, ds.task, ds.feature_names
        )
        imp = extract_feature_importance(model, ds.feature_names)
        key = f"{train_v}_to_{test_v}"
        directions[key] = {
            "tabnet": tabnet_m,
            "rf": rf_m,
            "linear": lin_m,
            "importance": imp,
            "n_train": len(ds.y_train),
            "n_test": len(ds.y_test),
        }
        plot_attention_heatmap(imp, f"v5_{key}", output_dir / f"v5_{key}_attention.png")

    # Primary result: Bermuda -> Halifax
    primary = build_cross_venue_dataset("Bermuda", "Halifax", data_root)
    X_val, y_val = _eval_set(primary)
    model = train_tabnet(primary.X_train, primary.y_train, X_val, y_val, primary.task, params)
    preds, _ = predict_tabnet(model, primary.X_test, primary.task)
    tabnet_metrics = regression_metrics(primary.y_test, preds)
    rf_metrics, lin_metrics, rf_imp, _ = fit_baselines(
        primary.X_train, primary.y_train, primary.X_test, primary.y_test, primary.task, primary.feature_names
    )
    importance = extract_feature_importance(model, primary.feature_names)
    beats, meaningful, reason = compute_verdict("v5", "regression", tabnet_metrics, rf_metrics, importance)

    result = EvalResult(
        variation="v5",
        task="regression",
        n_train=len(primary.y_train),
        n_val=len(primary.y_val),
        n_test=len(primary.y_test),
        tabnet_metrics=tabnet_metrics,
        baseline_metrics=rf_metrics,
        linear_metrics=lin_metrics,
        beats_baseline=beats,
        attention_interpretable=bool(importance),
        meaningful=meaningful,
        verdict_reason=reason,
        feature_importance=importance,
        baseline_importance=rf_imp,
        per_direction=directions,
    )
    save_result(result, output_dir)
    return result
