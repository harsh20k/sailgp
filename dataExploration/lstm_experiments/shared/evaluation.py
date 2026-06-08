"""Metrics, baseline runners, training loop, and Plotly exports."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split

from .data_prep import WindowedDataset, get_device

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


EXPORT_DIR = Path(__file__).resolve().parents[2] / "exported"
SUMMARY_JSON = EXPORT_DIR / "lstm_experiments_summary.json"


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    task: str
    lstm_metric_name: str
    lstm_metric: float
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    best_baseline_name: str = ""
    best_baseline_metric: float = 0.0
    delta: float = 0.0
    has_signal: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)

    def to_summary_row(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_id,
            "task": self.task,
            "lstm_metric": self.lstm_metric,
            "metric_name": self.lstm_metric_name,
            "best_baseline": self.best_baseline_name,
            "best_baseline_metric": self.best_baseline_metric,
            "delta": self.delta,
            "signal": self.has_signal,
        }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def persistence_baseline(X: np.ndarray, y: np.ndarray, target_idx: int = 0) -> np.ndarray:
    """Last timestep value of target feature column (denormalized proxy: use last col if unknown)."""
    # For speed forecast, target is speed at t+h; persistence = last speed in window
    return X[:, -1, target_idx]


def ridge_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(flatten_windows(X_train), y_train)
    return model.predict(flatten_windows(X_test))


def logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    multi_class: bool = False,
) -> np.ndarray:
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    if multi_class:
        model = LogisticRegression(max_iter=500, class_weight="balanced", multi_class="multinomial")
    model.fit(flatten_windows(X_train), y_train.astype(int))
    return model.predict(flatten_windows(X_test))


def xgboost_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if XGBClassifier is None:
        return np.full(len(X_test), np.bincount(y_train.astype(int)).argmax())
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(flatten_windows(X_train), y_train.astype(int))
    return model.predict(flatten_windows(X_test))


def polar_bin_baseline(
    twa_train: np.ndarray,
    y_train: np.ndarray,
    twa_test: np.ndarray,
    bin_width: float = 10.0,
) -> np.ndarray:
    """Mean VMG/TWS per TWA bin from training set."""
    bins = np.floor(twa_train / bin_width) * bin_width
    table: dict[float, float] = {}
    for b in np.unique(bins):
        mask = bins == b
        table[float(b)] = float(np.mean(y_train[mask]))
    global_mean = float(np.mean(y_train))
    preds = []
    for twa in twa_test:
        b = float(np.floor(twa / bin_width) * bin_width)
        preds.append(table.get(b, global_mean))
    return np.array(preds, dtype=np.float32)


def rule_foiling_baseline(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Rule: RH_BOW > 80 and speed > 35 (on denormalized approx — use raw indices)."""
    try:
        rh_idx = feature_names.index("LENGTH_RH_BOW_mm")
        spd_idx = feature_names.index("BOAT_SPEED_km_h_1")
    except ValueError:
        return np.zeros(len(X))
    rh = X[:, -1, rh_idx]
    spd = X[:, -1, spd_idx]
    return ((rh > 80) & (spd > 35)).astype(np.float32)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    if average == "macro":
        metrics["f1"] = metrics["f1_macro"]
    else:
        metrics["f1"] = metrics["f1_weighted"]
    return metrics


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    hits = sum(int(true in row) for true, row in zip(y_true, top_k))
    return hits / len(y_true) if len(y_true) else 0.0


def _log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    task: str = "regression",
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
    label: str = "LSTM",
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, list[float]]]:
    device = device or get_device()
    model = model.to(device)
    _log(f"[{label}] device={device} | train_batches={len(train_loader)} | val_batches={len(val_loader) if val_loader else 0}", verbose=verbose)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if task == "regression":
        criterion: nn.Module = nn.MSELoss()
    elif task == "binary_seq" or task == "binary_future":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    stopped_early = False

    epoch_bar = tqdm(
        range(epochs),
        desc=f"{label} train",
        unit="epoch",
        disable=not verbose,
        file=sys.stdout,
    )
    for epoch in epoch_bar:
        model.train()
        train_losses = []
        n_batches = len(train_loader)
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if task == "binary_seq" or task == "binary_future":
                loss = criterion(out, yb)
            elif task == "regression":
                loss = criterion(out, yb.squeeze(-1) if yb.ndim > 1 else yb)
            else:
                loss = criterion(out, yb.long().squeeze(-1) if yb.ndim > 1 else yb.long())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if verbose and n_batches >= 20 and (batch_idx + 1) % max(1, n_batches // 5) == 0:
                epoch_bar.write(
                    f"  [{label}] epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{n_batches} loss={loss.item():.4f}"
                )
        scheduler.step()
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    if task == "binary_seq" or task == "binary_future":
                        loss = criterion(out, yb)
                    elif task == "regression":
                        loss = criterion(out, yb.squeeze(-1) if yb.ndim > 1 else yb)
                    else:
                        loss = criterion(out, yb.long().squeeze(-1) if yb.ndim > 1 else yb.long())
                    val_losses.append(float(loss.item()))
            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)
            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", stale=stale)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    stopped_early = True
                    _log(f"[{label}] early stop at epoch {epoch + 1} (best val={best_val:.4f})", verbose=verbose)
                    break
        else:
            epoch_bar.set_postfix(train=f"{train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not stopped_early and verbose:
        _log(f"[{label}] finished {len(history['train_loss'])} epochs", verbose=verbose)
    return model, history


def predict_lstm(
    model: nn.Module,
    loader: DataLoader,
    task: str = "regression",
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    device = device or get_device()
    model.eval()
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            if task == "regression":
                preds.append(out.cpu().numpy())
            elif task == "binary_seq" or task == "binary_future":
                p = torch.sigmoid(out).cpu().numpy()
                preds.append((p >= 0.5).astype(np.float32))
                probs.append(p.reshape(-1))
            else:
                prob = torch.softmax(out, dim=-1).cpu().numpy()
                preds.append(prob.argmax(axis=-1))
                probs.append(prob)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs) if probs else None
    return y_pred, y_prob


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_factory: Callable[[], nn.Module],
    task: str = "regression",
    batch_size: int = 64,
    class_weights: torch.Tensor | None = None,
    label: str = "LSTM",
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], np.ndarray, np.ndarray | None]:
    device = get_device()
    _log(
        f"[{label}] windows train={len(X_train):,} val={len(X_val):,} "
        f"features={X_train.shape[-1] if len(X_train) else 0} seq={X_train.shape[1] if len(X_train) else 0}",
        verbose=verbose,
    )
    train_ds = WindowedDataset(X_train, y_train)
    val_ds = WindowedDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = model_factory()
    model, history = train_lstm(
        model,
        train_loader,
        val_loader,
        task=task,
        class_weights=class_weights,
        device=device,
        label=label,
        verbose=verbose,
    )
    _log(f"[{label}] evaluating validation set...", verbose=verbose)
    y_pred, y_prob = predict_lstm(model, val_loader, task=task, device=device)
    return model, history, y_pred, y_prob


def pick_best_baseline(
    baselines: dict[str, float],
    higher_is_better: bool,
) -> tuple[str, float]:
    if not baselines:
        return "", 0.0
    if higher_is_better:
        name = max(baselines, key=baselines.get)  # type: ignore[arg-type]
    else:
        name = min(baselines, key=baselines.get)  # type: ignore[arg-type]
    return name, baselines[name]


def signal_detected(
    lstm_metric: float,
    baseline_metric: float,
    higher_is_better: bool,
    min_delta: float = 0.01,
) -> bool:
    if higher_is_better:
        return lstm_metric > baseline_metric + min_delta
    return lstm_metric < baseline_metric - min_delta


def export_eval_html(result: ExperimentResult, output_path: Path | None = None) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or (EXPORT_DIR / f"lstm_{result.experiment_id}.html")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training loss",
            "Metrics comparison",
            "LSTM vs baseline",
            "Details",
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "bar"}, {"type": "table"}]],
    )

    if result.history.get("train_loss"):
        fig.add_trace(
            go.Scatter(y=result.history["train_loss"], name="train", mode="lines"),
            row=1,
            col=1,
        )
    if result.history.get("val_loss"):
        fig.add_trace(
            go.Scatter(y=result.history["val_loss"], name="val", mode="lines"),
            row=1,
            col=1,
        )

    names = ["LSTM"] + list(result.baseline_metrics.keys())
    values = [result.lstm_metric] + list(result.baseline_metrics.values())
    fig.add_trace(go.Bar(x=names, y=values, name=result.lstm_metric_name), row=1, col=2)

    fig.add_trace(
        go.Bar(
            x=["LSTM", result.best_baseline_name or "baseline"],
            y=[result.lstm_metric, result.best_baseline_metric],
            marker_color=["#2563eb", "#94a3b8"],
        ),
        row=2,
        col=1,
    )

    detail_rows = [[k, str(v)] for k, v in result.details.items()]
    if detail_rows:
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=list(zip(*detail_rows))),
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=f"{result.name} — signal={'YES' if result.has_signal else 'NO'} (Δ={result.delta:.4f})",
        height=700,
        showlegend=True,
    )
    fig.write_html(str(path))
    return path


def export_summary_dashboard(results: list[ExperimentResult], output_path: Path | None = None) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or (EXPORT_DIR / "lstm_experiments_dashboard.html")
    summary_path = SUMMARY_JSON

    rows = [r.to_summary_row() for r in results]
    with open(summary_path, "w") as f:
        json.dump({"experiments": rows, "full": [asdict(r) for r in results]}, f, indent=2)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LSTM vs best baseline", "Signal detected"),
        specs=[[{"type": "bar"}, {"type": "table"}]],
    )

    fig.add_trace(
        go.Bar(
            name="LSTM",
            x=[r.experiment_id for r in results],
            y=[r.lstm_metric for r in results],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Best baseline",
            x=[r.experiment_id for r in results],
            y=[r.best_baseline_metric for r in results],
        ),
        row=1,
        col=1,
    )

    table_rows = [
        [r.experiment_id, r.task, f"{r.lstm_metric:.4f}", r.best_baseline_name, f"{r.best_baseline_metric:.4f}", f"{r.delta:.4f}", "✓" if r.has_signal else "✗"]
        for r in results
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=["Exp", "Task", "LSTM", "Baseline", "Base val", "Delta", "Signal"]),
            cells=dict(values=list(zip(*table_rows)) if table_rows else [[], [], [], [], [], [], []]),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title="SailGP LSTM Experiments Summary", height=500, barmode="group")
    fig.write_html(str(path))
    return path
