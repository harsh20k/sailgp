"""Evaluation: baselines, metrics, learning curves, ablation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from sailgp_analysis.analytics import add_foiling
from sailgp_analysis.config import DATA_ROOT, OUTPUT_DIR
from sailgp_analysis.data_loader import load_all_boats
from sailgp_analysis.tcm.dataset import (
    VARIATION_CONFIGS,
    VariationName,
    WindowDataset,
    build_variation_dataset,
)
from sailgp_analysis.tcm.model import SimpleLSTM, TCNModel
from sailgp_analysis.tcm.train import TrainConfig, default_device, train_model


@dataclass
class EvalResult:
    variation: str
    task: str
    n_train: int
    n_test: int
    model_metrics: dict
    baseline_metrics: dict
    lstm_metrics: dict | None
    beats_baseline: bool
    meaningful: bool
    per_team: dict | None = None
    learning_curve: list | None = None
    ablation: dict | None = None


def _predict(
    model: torch.nn.Module,
    ds: WindowDataset,
    task: str,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = device or default_device()
    model.eval()
    model.to(device)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> dict:
    y_prob = 1 / (1 + np.exp(-y_pred_logits))
    y_hat = (y_prob >= 0.5).astype(int)
    yt = y_true.astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yt, y_hat)),
        "f1": float(f1_score(yt, y_hat, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yt, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def foiling_threshold_baseline(ds: WindowDataset) -> dict:
    """Naive foiling rule: predict foiling if ride heights > 100 and speed > 40 at last timestep."""
    # Features: speed=0, rh_p=1, rh_s=2, rh_bow=3
    windows = ds.windows.numpy()
    if windows.shape[-1] < 4:
        return {"f1": 0.0, "accuracy": 0.0}
    last = windows[:, -1, :]
    speed = last[:, 0]  # normalized - use relative: if all RH features high and speed high
    # Denormalized thresholds don't apply; use percentile-based proxy on last step
    rh_mean = (last[:, 1] + last[:, 2] + last[:, 3]) / 3
    pred = ((rh_mean > rh_mean.mean()) & (speed > speed.mean())).astype(int)
    y = ds.targets.numpy().astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def persistence_baseline(ds: WindowDataset, target_idx: int = -1) -> dict:
    """Regression baseline: predict target at t+horizon = value at last window step."""
    windows = ds.windows.numpy()
    y_true = ds.targets.numpy()
    # Use VMG proxy from features if available - for VMG variation TWA is idx 0, use mean as proxy
    y_pred = windows[:, -1, 0]  # crude persistence on first feature
    # Scale pred to target range
    if y_true.std() > 0:
        y_pred = y_pred * (y_true.std() / (y_pred.std() + 1e-6)) + (y_true.mean() - y_pred.mean())
    return _regression_metrics(y_true, y_pred)


def rank_persistence_baseline(ds: WindowDataset) -> dict:
    """Predict top-3 status unchanged (rank feature idx 0)."""
    windows = ds.windows.numpy()
    y_true = ds.targets.numpy().astype(int)
    current_rank = windows[:, -1, 0]
    pred = (current_rank <= 0).astype(int)  # normalized rank - use median split
    median = np.median(current_rank)
    pred = (current_rank <= median).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def majority_class_baseline(ds: WindowDataset) -> dict:
    y = ds.targets.numpy().astype(int)
    majority = int(y.mean() >= 0.5)
    pred = np.full_like(y, majority)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def per_team_residuals(ds: WindowDataset, y_pred: np.ndarray, task: str) -> dict:
    if not ds.meta:
        return {}
    by_team: dict[str, list[float]] = {}
    y_true = ds.targets.numpy()
    for i, m in enumerate(ds.meta):
        team = m.get("team", "unknown")
        if task == "regression":
            err = abs(y_true[i] - y_pred[i])
        else:
            err = abs(y_true[i] - (1 / (1 + np.exp(-y_pred[i]))))
        by_team.setdefault(team, []).append(err)
    return {t: float(np.mean(v)) for t, v in sorted(by_team.items())}


def learning_curve(
    variation: VariationName,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    config: TrainConfig | None = None,
) -> list[dict]:
    cfg = VARIATION_CONFIGS[variation]
    config = config or TrainConfig(epochs=20)
    curve = []
    for frac in fractions:
        train_ds, test_ds, info = build_variation_dataset(variation, split="train", train_fraction=frac)
        if test_ds is None or len(train_ds) < 10:
            train_ds, test_ds, info = build_variation_dataset(variation)
        if len(train_ds) < 10:
            continue
        model, _ = train_model(
            train_ds,
            task=cfg.task,
            n_features=info["n_features"],
            config=config,
            label=f"{variation}/curve@{frac:.0%}",
        )
        preds, targets = _predict(model, test_ds or train_ds, cfg.task)
        if cfg.task == "classification":
            m = _classification_metrics(targets, preds)
        else:
            m = _regression_metrics(targets, preds)
        curve.append({"fraction": frac, "n_train": len(train_ds), **m})
    return curve


def evaluate_variation(
    variation: VariationName,
    config: TrainConfig | None = None,
    train_venue: str | None = None,
    test_venue: str | None = None,
) -> EvalResult:
    cfg = VARIATION_CONFIGS[variation]
    config = config or TrainConfig()

    if variation == "transfer":
        train_ds, test_ds, info = build_variation_dataset(
            "foiling", train_venue=train_venue or "Halifax", test_venue=test_venue or "Bermuda"
        )
    else:
        train_ds, test_ds, info = build_variation_dataset(variation)

    if test_ds is None or len(test_ds) == 0:
        # fallback split
        train_ds, test_ds, info = build_variation_dataset(variation, split="all")
        n = len(train_ds)
        split = int(n * 0.8)
        if split < 1:
            split = max(1, n - 1)
        test_ds = WindowDataset(
            train_ds.windows[split:].numpy(),
            train_ds.targets[split:].numpy(),
            train_ds.meta[split:] if train_ds.meta else [],
        )
        train_ds = WindowDataset(
            train_ds.windows[:split].numpy(),
            train_ds.targets[:split].numpy(),
            train_ds.meta[:split] if train_ds.meta else [],
        )

    run_label = variation
    if variation == "transfer" and train_venue and test_venue:
        run_label = f"transfer_{train_venue}_to_{test_venue}"

    print(f"\n=== {run_label} ===", flush=True)
    print(f"  train={len(train_ds):,} test={len(test_ds):,}", flush=True)

    model, history = train_model(
        train_ds,
        task=cfg.task,
        n_features=info["n_features"],
        config=config,
        label=f"{run_label}/TCN",
    )
    preds, targets = _predict(model, test_ds, cfg.task)

    if cfg.task == "classification":
        model_metrics = _classification_metrics(targets, preds)
    else:
        model_metrics = _regression_metrics(targets, preds)

    # Baselines
    if variation in ("foiling", "transfer"):
        baseline_metrics = foiling_threshold_baseline(test_ds)
        lstm = SimpleLSTM(info["n_features"], task=cfg.task)
        lstm, _ = train_model(
            train_ds,
            task=cfg.task,
            n_features=info["n_features"],
            config=config,
            model=lstm,
            label=f"{run_label}/LSTM",
        )
        lstm_preds, _ = _predict(lstm, test_ds, cfg.task)
        lstm_metrics = _classification_metrics(targets, lstm_preds)
    elif variation == "vmg":
        baseline_metrics = persistence_baseline(test_ds)
        lstm_metrics = None
    elif variation == "rank":
        baseline_metrics = rank_persistence_baseline(test_ds)
        lstm_metrics = None
    elif variation == "prestart":
        baseline_metrics = majority_class_baseline(test_ds)
        lstm_metrics = None
    else:
        baseline_metrics = {}
        lstm_metrics = None

    # Meaningful check
    if cfg.task == "classification":
        beats = model_metrics.get("f1", 0) > baseline_metrics.get("f1", 0)
        meaningful = beats and model_metrics.get("f1", 0) > 0.55
    else:
        beats = model_metrics.get("mae", 999) < baseline_metrics.get("mae", 999)
        meaningful = beats and model_metrics.get("r2", -999) > 0.1

    team_residuals = per_team_residuals(test_ds, preds, cfg.task)

    return EvalResult(
        variation=variation if variation != "transfer" else f"transfer_{train_venue}_to_{test_venue}",
        task=cfg.task,
        n_train=len(train_ds),
        n_test=len(test_ds),
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        lstm_metrics=lstm_metrics,
        beats_baseline=beats,
        meaningful=meaningful,
        per_team=team_residuals,
        learning_curve=None,
    )


def _results_path(output_dir: Path) -> Path:
    return output_dir / "tcm_results.json"


def _load_results(output_dir: Path) -> dict:
    path = _results_path(output_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_results(results: dict, output_dir: Path, variation_key: str | None = None) -> Path:
    """Persist results after each variation completes."""
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _results_path(output_dir)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": list(results.keys()),
        "variations": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if variation_key:
        print(f"  Saved → {path} ({variation_key})", flush=True)
    return path


def run_all_variations(
    output_dir: Path | None = None,
    config: TrainConfig | None = None,
    include_learning_curves: bool = True,
    resume: bool = True,
) -> dict:
    output_dir = output_dir or (OUTPUT_DIR / "tcm")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or TrainConfig(epochs=25)

    existing = _load_results(output_dir) if resume else {}
    results: dict = existing.get("variations", {}) if isinstance(existing, dict) else {}

    all_variations = [
        ("foiling", lambda: evaluate_variation("foiling", config=config)),
        ("vmg", lambda: evaluate_variation("vmg", config=config)),
        ("rank", lambda: evaluate_variation("rank", config=config)),
        ("prestart", lambda: evaluate_variation("prestart", config=config)),
        ("transfer_Halifax_to_Bermuda", lambda: evaluate_variation(
            "transfer", config=config, train_venue="Halifax", test_venue="Bermuda"
        )),
        ("transfer_Bermuda_to_Halifax", lambda: evaluate_variation(
            "transfer", config=config, train_venue="Bermuda", test_venue="Halifax"
        )),
    ]

    total = len(all_variations)
    for i, (key, run_fn) in enumerate(all_variations, start=1):
        if resume and key in results:
            print(f"\n[{i}/{total}] Skipping {key} (already saved)", flush=True)
            continue

        print(f"\n[{i}/{total}] Running variation: {key}", flush=True)
        print("-" * 60, flush=True)
        res = run_fn()
        if include_learning_curves and not key.startswith("transfer"):
            print(f"  Learning curve for {key}...", flush=True)
            res.learning_curve = learning_curve(key, config=config)  # type: ignore[arg-type]
        results[key] = asdict(res)
        _save_results(results, output_dir, variation_key=key)
        print(
            f"  Result: meaningful={res.meaningful} | model={res.model_metrics} | baseline={res.baseline_metrics}",
            flush=True,
        )

    out_path = _results_path(output_dir)
    print(f"\nAll results in {out_path} ({len(results)} variations)", flush=True)
    return results
