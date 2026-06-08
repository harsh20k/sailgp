"""Training loop with early stopping for TCM models."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sailgp_analysis.tcm.dataset import WindowDataset
from sailgp_analysis.tcm.model import TCNModel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def default_device() -> str:
    """Pick best available accelerator: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    val_fraction: float = 0.15
    device: str = field(default_factory=default_device)
    verbose: bool = True
    show_batch_progress: bool = True


def _make_loader(ds: WindowDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _iter_batches(loader: DataLoader, desc: str, verbose: bool, show_batch_progress: bool):
    if not verbose or not show_batch_progress or tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, file=sys.stdout, dynamic_ncols=True)


def train_model(
    train_ds: WindowDataset,
    task: str = "classification",
    n_features: int | None = None,
    config: TrainConfig | None = None,
    model: nn.Module | None = None,
    label: str = "TCN",
) -> tuple[nn.Module, dict]:
    """Train TCN model; returns (model, history)."""
    config = config or TrainConfig()
    device = torch.device(config.device)

    if n_features is None:
        n_features = train_ds.windows.shape[-1]

    if model is None:
        model = TCNModel(n_features, task=task)
    model = model.to(device)

    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Val split from train
    n_val = max(1, int(len(train_ds) * config.val_fraction))
    n_train = len(train_ds) - n_val
    if n_train < 1:
        train_subset = train_ds
        val_subset = train_ds
    else:
        train_subset, val_subset = random_split(train_ds, [n_train, n_val])

    train_loader = _make_loader(train_subset, config.batch_size, shuffle=True)
    val_loader = _make_loader(val_subset, config.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    t0 = time.perf_counter()

    _log(
        f"  [{label}] train={len(train_subset):,} val={len(val_subset):,} "
        f"task={task} device={config.device} epochs<={config.epochs}",
        config.verbose,
    )

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        batch_iter = _iter_batches(
            train_loader,
            desc=f"  [{label}] epoch {epoch + 1}/{config.epochs} train",
            verbose=config.verbose,
            show_batch_progress=config.show_batch_progress,
        )
        for batch in batch_iter:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if tqdm is not None and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            val_iter = _iter_batches(
                val_loader,
                desc=f"  [{label}] epoch {epoch + 1}/{config.epochs} val",
                verbose=config.verbose,
                show_batch_progress=config.show_batch_progress,
            )
            for batch in val_iter:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        va_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            marker = "*"
        else:
            stale += 1
            marker = f" ({stale}/{config.patience})"

        _log(
            f"  [{label}] epoch {epoch + 1:>2}/{config.epochs} "
            f"train={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f}{marker}",
            config.verbose,
        )

        if stale >= config.patience:
            _log(f"  [{label}] early stop at epoch {epoch + 1}", config.verbose)
            break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.perf_counter() - t0
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    history["elapsed_s"] = elapsed
    _log(
        f"  [{label}] finished in {elapsed:.1f}s — {history['epochs_run']} epochs, best val={best_val:.4f}",
        config.verbose,
    )
    return model, history


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, n_features: int, task: str = "classification") -> TCNModel:
    model = TCNModel(n_features, task=task)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
