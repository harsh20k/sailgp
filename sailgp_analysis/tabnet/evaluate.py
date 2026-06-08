"""Shared evaluation: metrics, baselines, attention analysis, verdicts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from sailgp_analysis.config import OUTPUT_DIR
from sailgp_analysis.tabnet.config import EXPECTED_DOMINANT, TabNetParams, VariationName


@dataclass
class EvalResult:
    variation: str
    task: str
    n_train: int
    n_val: int
    n_test: int
    tabnet_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    linear_metrics: dict[str, float] | None = None
    beats_baseline: bool = False
    attention_interpretable: bool = False
    meaningful: bool = False
    verdict_reason: str = ""
    feature_importance: dict[str, float] = field(default_factory=dict)
    baseline_importance: dict[str, float] = field(default_factory=dict)
    expected_dominant_hits: list[str] = field(default_factory=list)
    loocv_spearman_mean: float | None = None
    loocv_spearman_std: float | None = None
    per_direction: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    yt = y_true.astype(int)
    yh = y_pred.astype(int)
    out = {
        "accuracy": float(accuracy_score(yt, yh)),
        "f1": float(f1_score(yt, yh, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(yt)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(yt, y_prob))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rho, _ = spearmanr(y_true, y_pred)
    return {"spearman_rho": float(rho) if not np.isnan(rho) else 0.0, **regression_metrics(y_true, y_pred)}


def fit_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    if task == "regression":
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_metrics = regression_metrics(y_test, rf_pred)
        rf_imp = {feature_names[i]: float(v) for i, v in enumerate(rf.feature_importances_)}

        lin = Ridge(alpha=1.0)
        lin.fit(X_train, y_train)
        lin_pred = lin.predict(X_test)
        lin_metrics = regression_metrics(y_test, lin_pred)
        return rf_metrics, lin_metrics, rf_imp, {}
    else:
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1] if len(rf.classes_) > 1 else None
        rf_metrics = classification_metrics(y_test, rf_pred, rf_prob)
        rf_imp = {feature_names[i]: float(v) for i, v in enumerate(rf.feature_importances_)}

        lin = LogisticRegression(max_iter=500, random_state=42)
        lin.fit(X_train, y_train)
        lin_pred = lin.predict(X_test)
        lin_prob = lin.predict_proba(X_test)[:, 1] if len(lin.classes_) > 1 else None
        lin_metrics = classification_metrics(y_test, lin_pred, lin_prob)
        return rf_metrics, lin_metrics, rf_imp, {}


def tabnet_params_to_dict(params: TabNetParams) -> dict:
    import torch.optim as optim

    sched_map = {"step": optim.lr_scheduler.StepLR}
    return {
        "n_d": params.n_d,
        "n_a": params.n_a,
        "n_steps": params.n_steps,
        "gamma": params.gamma,
        "lambda_sparse": params.lambda_sparse,
        "optimizer_fn": optim.Adam,
        "optimizer_params": params.optimizer_params,
        "scheduler_params": params.scheduler_params,
        "scheduler_fn": sched_map.get(params.scheduler_fn, optim.lr_scheduler.StepLR),
        "mask_type": params.mask_type,
        "verbose": params.verbose,
        "seed": params.seed,
    }


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    params: TabNetParams,
):
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    fit_kwargs = tabnet_params_to_dict(params)
    eval_set = [(X_val, y_val)] if len(X_val) > 0 else None

    if task == "regression":
        model = TabNetRegressor(**fit_kwargs)
        y_tr = y_train.reshape(-1, 1)
        eval_set_fixed = [(X_val, y_val.reshape(-1, 1))] if eval_set else None
        model.fit(
            X_train,
            y_tr,
            eval_set=eval_set_fixed,
            eval_metric=["mae"],
            max_epochs=params.max_epochs,
            patience=params.patience,
            batch_size=params.batch_size,
            virtual_batch_size=params.virtual_batch_size,
        )
    else:
        model = TabNetClassifier(**fit_kwargs)
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=["accuracy"],
            max_epochs=params.max_epochs,
            patience=params.patience,
            batch_size=params.batch_size,
            virtual_batch_size=params.virtual_batch_size,
        )
    return model


def predict_tabnet(model, X: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    if len(X) == 0:
        return np.array([]), None
    if task == "regression":
        preds = model.predict(X).ravel()
        return preds, None
    preds = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return preds, prob


def extract_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    return {feature_names[i]: float(v) for i, v in enumerate(imp)}


def top_k_features(importance: dict[str, float], k: int = 5) -> list[str]:
    return [f for f, _ in sorted(importance.items(), key=lambda x: -x[1])[:k]]


def check_attention_interpretable(
    importance: dict[str, float],
    variation: VariationName,
    k: int = 5,
) -> tuple[bool, list[str]]:
    expected = EXPECTED_DOMINANT.get(variation, [])
    top = top_k_features(importance, k)
    hits = [f for f in expected if f in top]
    # Meaningful if at least one expected feature in top-5
    return len(hits) >= 1, hits


def compute_verdict(
    variation: VariationName,
    task: str,
    tabnet_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    importance: dict[str, float],
    loocv_std: float | None = None,
) -> tuple[bool, bool, str]:
    interpretable, hits = check_attention_interpretable(importance, variation)

    if task == "regression":
        tabnet_mae = tabnet_metrics.get("mae", float("inf"))
        base_mae = baseline_metrics.get("mae", float("inf"))
        beats = tabnet_mae <= base_mae * 1.05  # within 5% of RF
        r2 = tabnet_metrics.get("r2", -999)
        meaningful = beats and r2 > 0.05 and interpretable
        reason = f"MAE {tabnet_mae:.3f} vs RF {base_mae:.3f}, R2={r2:.3f}, hits={hits}"
    else:
        tabnet_f1 = tabnet_metrics.get("f1", 0)
        base_f1 = baseline_metrics.get("f1", 0)
        beats = tabnet_f1 >= base_f1 - 0.05
        meaningful = beats and tabnet_f1 > 0.5 and interpretable
        reason = f"F1 {tabnet_f1:.3f} vs RF {base_f1:.3f}, hits={hits}"

    if loocv_std is not None and loocv_std > 0.4:
        meaningful = False
        reason += "; LOOCV variance too high"

    return beats, meaningful and interpretable, reason


def plot_attention_heatmap(
    importance: dict[str, float],
    variation: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    if not importance:
        return
    items = sorted(importance.items(), key=lambda x: -x[1])[:15]
    names = [x[0] for x in items]
    vals = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    ax.barh(names[::-1], vals[::-1], color="#2563eb")
    ax.set_xlabel("Feature importance")
    ax.set_title(title or f"TabNet feature importance — {variation}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_learning_curve(
    fractions: list[float],
    metrics: list[dict[str, float]],
    metric_key: str,
    output_path: Path,
    title: str,
) -> None:
    xs = [m.get("n_train", f) for f, m in zip(fractions, metrics)]
    ys = [m.get(metric_key, 0) for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#2563eb")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def save_result(result: EvalResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.variation}_results.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path


def default_output_dir() -> Path:
    return OUTPUT_DIR / "tabnet"
