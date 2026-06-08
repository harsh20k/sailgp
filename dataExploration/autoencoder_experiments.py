"""Autoencoder experiments for SailGP telemetry analysis."""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sailgp_analysis.analytics import add_foiling
from sailgp_analysis.config import DATA_ROOT
from sailgp_analysis.data_loader import load_all_boats

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

warnings.filterwarnings("ignore", category=UserWarning)


def _log(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)

# ---------------------------------------------------------------------------
# Feature column groups
# ---------------------------------------------------------------------------

CONTROL_SURFACE_COLS = [
    "ANGLE_WING_ROT_deg",
    "ANGLE_WING_TWIST_deg",
    "ANGLE_CA1_deg",
    "ANGLE_CA2_deg",
    "ANGLE_CA3_deg",
    "ANGLE_CA4_deg",
    "ANGLE_CA5_deg",
    "ANGLE_CA6_deg",
    "PER_JIB_LEAD_pct",
    "PER_JIB_SHEET_pct",
    "ANGLE_DB_RAKE_P_deg",
    "ANGLE_DB_RAKE_S_deg",
    "ANGLE_DB_CANT_P_deg",
    "ANGLE_DB_CANT_S_deg",
    "LENGTH_DB_H_P_mm",
    "LENGTH_DB_H_S_mm",
    "ANGLE_RUDDER_deg",
    "ANGLE_RUD_RAKE_P_deg",
    "ANGLE_RUD_RAKE_S_deg",
]

PLATFORM_COLS = [
    "PITCH_deg",
    "HEEL_deg",
    "LEEWAY_deg",
    "RATE_YAW_deg_s_1",
    "RATE_PITCH_deg_s_1",
    "RATE_ROLL_deg_s_1",
]

RIDE_HEIGHT_COLS = ["LENGTH_RH_P_mm", "LENGTH_RH_S_mm", "LENGTH_RH_BOW_mm"]

SPEED_COLS = ["BOAT_SPEED_km_h_1", "VMG_km_h_1", "GPS_SOG_km_h_1"]

WIND_COLS = ["TWA_SGP_deg", "TWS_SGP_km_h_1", "AWA_SGP_deg", "AWS_SGP_km_h_1"]

TACTICAL_COLS = ["PC_DTL_m", "PC_DTB_m", "PC_TTS_s", "TRK_RACE_RANK_unk"]

BASELINE_22_COLS = [
    "BOAT_SPEED_km_h_1",
    "VMG_km_h_1",
    "TWA_SGP_deg",
    "TWS_SGP_km_h_1",
    *RIDE_HEIGHT_COLS,
    "HEEL_deg",
    "PITCH_deg",
    "RATE_YAW_deg_s_1",
    "ANGLE_WING_ROT_deg",
    "ANGLE_WING_TWIST_deg",
    *[f"ANGLE_CA{i}_deg" for i in range(1, 7)],
    "ANGLE_DB_CANT_P_deg",
    "ANGLE_DB_CANT_S_deg",
]

EXP1_COLS = CONTROL_SURFACE_COLS + RIDE_HEIGHT_COLS + PLATFORM_COLS

EXP2_COLS = WIND_COLS + CONTROL_SURFACE_COLS

EXP3_COLS = (
    SPEED_COLS
    + RIDE_HEIGHT_COLS
    + WIND_COLS
    + PLATFORM_COLS
    + [
        "ANGLE_WING_ROT_deg",
        "ANGLE_WING_TWIST_deg",
        *[f"ANGLE_CA{i}_deg" for i in range(1, 7)],
        "PER_JIB_LEAD_pct",
        "PER_JIB_SHEET_pct",
        "ANGLE_DB_CANT_P_deg",
        "ANGLE_DB_CANT_S_deg",
        "ANGLE_RUDDER_deg",
    ]
)

EXP4_COLS = TACTICAL_COLS + PLATFORM_COLS

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def available_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_prepared_data(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    df = load_all_boats(data_root)
    if df.empty:
        raise ValueError("No boat data found.")
    return add_foiling(df)


def filter_status(df: pd.DataFrame, status: int | list[int]) -> pd.DataFrame:
    if isinstance(status, int):
        status = [status]
    return df[df["TRK_BOAT_RACE_STATUS_unk"].isin(status)].copy()


def drop_na_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = available_columns(df, cols)
    return df.dropna(subset=cols)


def time_split(
    X: np.ndarray,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    split = int(n * train_frac)
    return X[:split], X[split:]


def stratified_subsample(
    df: pd.DataFrame,
    group_col: str,
    n_per_group: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    counts = df.groupby(group_col).size()
    if n_per_group is None:
        n_per_group = int(counts.min())
    for group, gdf in df.groupby(group_col):
        n = min(n_per_group, len(gdf))
        idx = rng.choice(len(gdf), size=n, replace=False)
        parts.append(gdf.iloc[idx])
    return pd.concat(parts).sort_index()


def twa_bin(twa: float) -> str:
    if pd.isna(twa):
        return "unknown"
    a = abs(twa)
    if a <= 60:
        return "0-60"
    if a <= 120:
        return "60-120"
    return "120-180"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class DenseAE(nn.Module):
    """Fully-connected autoencoder with configurable layer sizes."""

    def __init__(self, n_in: int, encoder_dims: list[int], use_batchnorm: bool = True):
        super().__init__()
        self.latent_dim = encoder_dims[-1]
        enc_layers: list[nn.Module] = []
        prev = n_in
        for dim in encoder_dims:
            enc_layers.append(nn.Linear(prev, dim))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(dim))
            enc_layers.append(nn.ReLU())
            prev = dim
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = list(reversed(encoder_dims[:-1])) + [n_in]
        dec_layers: list[nn.Module] = []
        prev = encoder_dims[-1]
        for i, dim in enumerate(dec_dims):
            dec_layers.append(nn.Linear(prev, dim))
            if i < len(dec_dims) - 1:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(dim))
                dec_layers.append(nn.ReLU())
            prev = dim
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


class SparseAE(nn.Module):
    """Dense AE with L1 penalty on latent activations."""

    def __init__(self, n_in: int, encoder_dims: list[int]):
        super().__init__()
        self.ae = DenseAE(n_in, encoder_dims, use_batchnorm=False)
        self.latent_dim = encoder_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ae(x)


class VAE(nn.Module):
    """Variational autoencoder."""

    def __init__(self, n_in: int, hidden: int = 64, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, latent_dim)
        self.logvar = nn.Linear(hidden // 2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_in),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), z, mu, logvar


class LSTMAE(nn.Module):
    """Sequence autoencoder using LSTM encoder/decoder."""

    def __init__(self, n_features: int, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.latent_dim = latent
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.enc_to_latent(h[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.shape
        z = self.encode(x)
        h0 = self.latent_to_dec(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_in = h0.transpose(0, 1).repeat(1, seq_len, 1)
        out, _ = self.decoder(dec_in, (h0, c0))
        recon = self.output(out)
        return recon, z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_epoch_batches(
    model: nn.Module,
    Xt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    epoch: int,
    epochs: int,
    epoch_bar: Any,
    label: str,
    verbose: bool,
) -> float:
    model.train()
    perm = torch.randperm(len(Xt))
    epoch_loss = 0.0
    n_batches = max(1, (len(Xt) + batch_size - 1) // batch_size)
    batch_report = max(1, n_batches // 5) if n_batches >= 10 else n_batches + 1

    for batch_idx, i in enumerate(range(0, len(Xt), batch_size)):
        idx = perm[i : i + batch_size]
        batch = Xt[idx]
        optimizer.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if verbose and n_batches >= 10 and (batch_idx + 1) % batch_report == 0:
            epoch_bar.write(
                f"  [{label}] epoch {epoch + 1}/{epochs} "
                f"batch {batch_idx + 1}/{n_batches} loss={loss.item():.4f}"
            )

    return epoch_loss / n_batches


@dataclass
class TrainResult:
    model: nn.Module
    losses: list[float]
    scaler: StandardScaler | MinMaxScaler
    feature_cols: list[str]
    device: torch.device = field(default_factory=lambda: DEVICE)


def train_dense_ae(
    X_train: np.ndarray,
    encoder_dims: list[int],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    l1_latent: float = 0.0,
    feature_cols: list[str] | None = None,
    use_sparse: bool = False,
    label: str = "Dense AE",
    verbose: bool = True,
) -> TrainResult:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)
    n_in = Xs.shape[1]
    n_batches = max(1, (len(Xs) + batch_size - 1) // batch_size)

    if use_sparse:
        model: nn.Module = SparseAE(n_in, encoder_dims).to(DEVICE)
        arch = f"SparseAE {n_in}→{'→'.join(map(str, encoder_dims))}"
    else:
        model = DenseAE(n_in, encoder_dims).to(DEVICE)
        arch = f"DenseAE {n_in}→{'→'.join(map(str, encoder_dims))}"

    _log(
        f"[{label}] {arch} | device={DEVICE} | samples={len(Xs):,} | batches/epoch={n_batches}",
        verbose=verbose,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    Xt = torch.tensor(Xs, device=DEVICE)
    losses: list[float] = []

    def batch_loss(batch: torch.Tensor) -> torch.Tensor:
        recon, z = model(batch)
        loss = mse(recon, batch)
        if l1_latent > 0:
            loss = loss + l1_latent * z.abs().mean()
        return loss

    epoch_bar = tqdm(
        range(epochs),
        desc=f"{label} train",
        unit="epoch",
        disable=not verbose,
        file=sys.stdout,
    )
    for epoch in epoch_bar:
        epoch_loss = _train_epoch_batches(
            model, Xt, opt, batch_size, batch_loss,
            epoch=epoch, epochs=epochs, epoch_bar=epoch_bar, label=label, verbose=verbose,
        )
        losses.append(epoch_loss)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

    if verbose:
        _log(f"[{label}] finished {len(losses)} epochs | final loss={losses[-1]:.4f}", verbose=verbose)

    return TrainResult(model=model, losses=losses, scaler=scaler, feature_cols=feature_cols or [])


def train_vae(
    X_train: np.ndarray,
    hidden: int = 64,
    latent_dim: int = 8,
    epochs: int = 150,
    lr: float = 1e-3,
    beta: float = 0.5,
    batch_size: int = 512,
    feature_cols: list[str] | None = None,
    label: str = "VAE",
    verbose: bool = True,
) -> TrainResult:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)
    n_in = Xs.shape[1]
    n_batches = max(1, (len(Xs) + batch_size - 1) // batch_size)
    model = VAE(n_in, hidden=hidden, latent_dim=latent_dim).to(DEVICE)

    _log(
        f"[{label}] VAE {n_in}→{hidden}→{latent_dim} | device={DEVICE} | "
        f"samples={len(Xs):,} | batches/epoch={n_batches} | β={beta}",
        verbose=verbose,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xs, device=DEVICE)
    losses: list[float] = []

    def batch_loss(batch: torch.Tensor) -> torch.Tensor:
        recon, _, mu, logvar = model(batch)
        recon_loss = nn.functional.mse_loss(recon, batch)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl

    epoch_bar = tqdm(
        range(epochs),
        desc=f"{label} train",
        unit="epoch",
        disable=not verbose,
        file=sys.stdout,
    )
    for epoch in epoch_bar:
        epoch_loss = _train_epoch_batches(
            model, Xt, opt, batch_size, batch_loss,
            epoch=epoch, epochs=epochs, epoch_bar=epoch_bar, label=label, verbose=verbose,
        )
        losses.append(epoch_loss)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

    if verbose:
        _log(f"[{label}] finished {len(losses)} epochs | final loss={losses[-1]:.4f}", verbose=verbose)

    return TrainResult(model=model, losses=losses, scaler=scaler, feature_cols=feature_cols or [])


def train_lstm_ae(
    sequences: np.ndarray,
    hidden: int = 32,
    latent: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 128,
    feature_cols: list[str] | None = None,
    label: str = "LSTM AE",
    verbose: bool = True,
) -> tuple[TrainResult, MinMaxScaler]:
    """sequences shape: (n_seq, seq_len, n_features)."""
    n_feat = sequences.shape[2]
    seq_len = sequences.shape[1]
    flat = sequences.reshape(-1, n_feat)
    scaler = MinMaxScaler()
    flat_s = scaler.fit_transform(flat)
    Xs = flat_s.reshape(sequences.shape).astype(np.float32)
    n_batches = max(1, (len(Xs) + batch_size - 1) // batch_size)

    model = LSTMAE(n_feat, hidden=hidden, latent=latent).to(DEVICE)
    _log(
        f"[{label}] LSTM-AE seq={seq_len}×{n_feat}→{latent} | device={DEVICE} | "
        f"sequences={len(Xs):,} | batches/epoch={n_batches}",
        verbose=verbose,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    Xt = torch.tensor(Xs, device=DEVICE)
    losses: list[float] = []

    def batch_loss(batch: torch.Tensor) -> torch.Tensor:
        recon, _ = model(batch)
        return mse(recon, batch)

    epoch_bar = tqdm(
        range(epochs),
        desc=f"{label} train",
        unit="epoch",
        disable=not verbose,
        file=sys.stdout,
    )
    for epoch in epoch_bar:
        epoch_loss = _train_epoch_batches(
            model, Xt, opt, batch_size, batch_loss,
            epoch=epoch, epochs=epochs, epoch_bar=epoch_bar, label=label, verbose=verbose,
        )
        losses.append(epoch_loss)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

    if verbose:
        _log(f"[{label}] finished {len(losses)} epochs | final loss={epoch_loss:.4f}", verbose=verbose)

    result = TrainResult(model=model, losses=losses, scaler=scaler, feature_cols=feature_cols or [])
    return result, scaler


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def reconstruct_dense(
    result: TrainResult,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = result.model
    model.eval()
    Xs = result.scaler.transform(X).astype(np.float32)
    Xt = torch.tensor(Xs, device=DEVICE)
    recon, z = model(Xt)
    recon_np = recon.cpu().numpy()
    z_np = z.cpu().numpy()
    err = ((recon_np - Xs) ** 2).mean(axis=1)
    return recon_np, z_np, err


@torch.no_grad()
def reconstruct_vae(
    result: TrainResult,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = result.model
    model.eval()
    Xs = result.scaler.transform(X).astype(np.float32)
    Xt = torch.tensor(Xs, device=DEVICE)
    recon, z, _, _ = model(Xt)
    recon_np = recon.cpu().numpy()
    z_np = z.cpu().numpy()
    err = ((recon_np - Xs) ** 2).mean(axis=1)
    return recon_np, z_np, err


@torch.no_grad()
def encode_vae(
    result: TrainResult,
    X: np.ndarray,
) -> np.ndarray:
    model = result.model
    model.eval()
    Xs = result.scaler.transform(X).astype(np.float32)
    Xt = torch.tensor(Xs, device=DEVICE)
    mu, _ = model.encode(Xt)
    return mu.cpu().numpy()


@torch.no_grad()
def encode_lstm(
    result: TrainResult,
    sequences: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    model = result.model
    model.eval()
    n_feat = sequences.shape[2]
    flat = sequences.reshape(-1, n_feat)
    flat_s = scaler.transform(flat)
    Xs = flat_s.reshape(sequences.shape).astype(np.float32)
    Xt = torch.tensor(Xs, device=DEVICE)
    recon, z = model(Xt)
    recon_np = recon.cpu().numpy()
    err = ((recon_np - Xs) ** 2).mean(axis=(1, 2))
    return z.cpu().numpy(), err


def per_feature_error(
    result: TrainResult,
    X: np.ndarray,
    use_vae: bool = False,
) -> np.ndarray:
    if use_vae:
        recon, _, _ = reconstruct_vae(result, X)
    else:
        recon, _, _ = reconstruct_dense(result, X)
    Xs = result.scaler.transform(X)
    return ((recon - Xs) ** 2).mean(axis=0)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 30,
    group_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract fixed-length sequences per boat/leg group."""
    group_cols = group_cols or ["venue", "race_label", "team", "TRK_LEG_NUM_unk"]
    group_cols = [c for c in group_cols if c in df.columns]
    feature_cols = available_columns(df, feature_cols)

    seqs: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for key, gdf in df.groupby(group_cols, dropna=False):
        gdf = gdf.sort_index()
        vals = gdf[feature_cols].values.astype(np.float32)
        if len(vals) < seq_len:
            continue
        for start in range(0, len(vals) - seq_len + 1, seq_len // 2):
            seqs.append(vals[start : start + seq_len])
            row = {c: v for c, v in zip(group_cols, key if isinstance(key, tuple) else (key,))}
            row["seq_start"] = gdf.index[start]
            row["twa_mean"] = float(gdf["TWA_SGP_deg"].iloc[start : start + seq_len].mean()) if "TWA_SGP_deg" in gdf else np.nan
            row["leg"] = row.get("TRK_LEG_NUM_unk", np.nan)
            meta_rows.append(row)

    if not seqs:
        return np.empty((0, seq_len, len(feature_cols))), pd.DataFrame()

    return np.stack(seqs), pd.DataFrame(meta_rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    name: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


def reconstruction_stats(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }


def null_check_auc(y_true: np.ndarray, scores: np.ndarray, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(y_true)
    try:
        return float(roc_auc_score(shuffled, scores))
    except ValueError:
        return 0.5


def cluster_metrics(latent: np.ndarray, k_values: list[int] | None = None) -> dict[str, Any]:
    k_values = k_values or [4, 6, 8]
    pca = PCA(n_components=min(10, latent.shape[1], latent.shape[0] - 1))
    pca_emb = pca.fit_transform(latent)

    out: dict[str, Any] = {"pca_silhouette": {}, "ae_silhouette": {}, "davies_bouldin": {}}
    for k in k_values:
        if k >= len(latent):
            continue
        km_ae = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_ae = km_ae.fit_predict(latent)
        km_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_pca = km_pca.fit_predict(pca_emb)
        out["ae_silhouette"][k] = float(silhouette_score(latent, labels_ae))
        out["pca_silhouette"][k] = float(silhouette_score(pca_emb, labels_pca))
        out["davies_bouldin"][k] = float(davies_bouldin_score(latent, labels_ae))
    return out


def leg_alignment_score(labels: np.ndarray, legs: np.ndarray) -> float:
    """Fraction of sequences where cluster mode matches majority leg in cluster."""
    df = pd.DataFrame({"cluster": labels, "leg": legs})
    df = df.dropna(subset=["leg"])
    if df.empty:
        return 0.0
    leg_modes = df.groupby("cluster")["leg"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    mapped = df["cluster"].map(leg_modes)
    return float((df["leg"] == mapped).mean())


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_experiment_1(
    df: pd.DataFrame | None = None,
    epochs: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Foiling-mode AE: lower recon error on foiling rows."""
    _log("=== Experiment 1: Foiling-mode AE ===", verbose=verbose)
    if df is None:
        df = load_prepared_data()
    cols = available_columns(df, EXP1_COLS)
    racing = drop_na_features(filter_status(df, 2), cols)
    racing = racing.sort_index()

    X = racing[cols].values.astype(np.float32)
    X_train, X_test = time_split(X, 0.8)
    train_df = racing.iloc[: len(X_train)]
    test_df = racing.iloc[len(X_train) :]
    _log(f"[Exp1] train={len(X_train):,} test={len(X_test):,} features={len(cols)}", verbose=verbose)

    result = train_dense_ae(
        X_train,
        encoder_dims=[16, 8, 4],
        epochs=epochs,
        feature_cols=cols,
        label="Exp1 Foiling AE",
        verbose=verbose,
    )

    _, _, err_train = reconstruct_dense(result, X_train)
    _, _, err_test = reconstruct_dense(result, X_test)

    y_train = train_df["foiling"].values.astype(int)
    y_test = test_df["foiling"].values.astype(int)

    # Lower error = more "normal"; foiling should be lower if model learns foiling manifold
    auc_train = roc_auc_score(y_train, -err_train) if len(np.unique(y_train)) > 1 else 0.5
    auc_test = roc_auc_score(y_test, -err_test) if len(np.unique(y_test)) > 1 else 0.5
    null_auc = null_check_auc(y_test, -err_test)

    team_stats = (
        test_df.assign(recon_error=err_test)
        .groupby("team")["recon_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    foiling_err = test_df.assign(recon_error=err_test).groupby("foiling")["recon_error"].mean()

    return {
        "experiment": "exp1_foiling",
        "feature_cols": cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "losses": result.losses,
        "metrics": {
            "auc_train": float(auc_train),
            "auc_test": float(auc_test),
            "null_auc": float(null_auc),
            "foiling_mean_error": float(foiling_err.get(True, np.nan)),
            "non_foiling_mean_error": float(foiling_err.get(False, np.nan)),
            **{f"recon_{k}": v for k, v in reconstruction_stats(err_test).items()},
        },
        "team_stats": team_stats,
        "test_df": test_df.assign(recon_error=err_test),
        "result": result,
        "meaningful": auc_test > 0.6 and auc_test > null_auc + 0.05,
    }


def run_experiment_2(
    df: pd.DataFrame | None = None,
    epochs: int = 200,
    seq_len: int = 30,
    verbose: bool = True,
) -> dict[str, Any]:
    """Sailing-mode discovery via LSTM AE sequences."""
    _log("=== Experiment 2: Sailing-mode LSTM AE ===", verbose=verbose)
    if df is None:
        df = load_prepared_data()
    cols = available_columns(df, EXP2_COLS)
    racing = drop_na_features(filter_status(df, 2), cols)

    _log(f"[Exp2] building {seq_len}s sequences...", verbose=verbose)
    sequences, meta = build_sequences(racing, cols, seq_len=seq_len)
    if len(sequences) < 50:
        raise ValueError("Not enough sequences for experiment 2.")

    n_train = int(len(sequences) * 0.8)
    train_seq, test_seq = sequences[:n_train], sequences[n_train:]
    meta_train, meta_test = meta.iloc[:n_train], meta.iloc[n_train:]
    _log(f"[Exp2] sequences={len(sequences):,} train={n_train:,} test={len(test_seq):,}", verbose=verbose)

    result, mm_scaler = train_lstm_ae(
        train_seq, epochs=epochs, feature_cols=cols, label="Exp2 LSTM AE", verbose=verbose,
    )
    z_train, _ = encode_lstm(result, train_seq, mm_scaler)
    z_test, err_test = encode_lstm(result, test_seq, mm_scaler)
    z_all = np.vstack([z_train, z_test])

    cluster_res = cluster_metrics(z_test, k_values=[4, 6, 8])
    best_k = max(cluster_res["ae_silhouette"], key=cluster_res["ae_silhouette"].get)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(z_test)

    legs = meta_test["leg"].values if "leg" in meta_test else np.full(len(labels), np.nan)
    leg_align = leg_alignment_score(labels, legs)

    meta_test = meta_test.copy()
    meta_test["cluster"] = labels
    meta_test["recon_error"] = err_test
    if "twa_mean" in meta_test:
        meta_test["twa_bin"] = meta_test["twa_mean"].apply(twa_bin)

    return {
        "experiment": "exp2_sailing_modes",
        "feature_cols": cols,
        "n_sequences": len(sequences),
        "seq_len": seq_len,
        "losses": result.losses,
        "latent_test": z_test,
        "meta_test": meta_test,
        "metrics": {
            "best_k": int(best_k),
            "ae_silhouette": cluster_res["ae_silhouette"].get(best_k, 0.0),
            "pca_silhouette": cluster_res["pca_silhouette"].get(best_k, 0.0),
            "davies_bouldin": cluster_res["davies_bouldin"].get(best_k, 0.0),
            "leg_alignment": float(leg_align),
            **{f"recon_{k}": v for k, v in reconstruction_stats(err_test).items()},
        },
        "cluster_results": cluster_res,
        "result": result,
        "mm_scaler": mm_scaler,
        "meaningful": (
            cluster_res["ae_silhouette"].get(best_k, 0) > cluster_res["pca_silhouette"].get(best_k, 0)
            and leg_align > 0.35
        ),
    }


def run_experiment_3(
    df: pd.DataFrame | None = None,
    epochs: int = 150,
    verbose: bool = True,
) -> dict[str, Any]:
    """Team DNA via VAE latent fingerprints."""
    _log("=== Experiment 3: Team DNA VAE ===", verbose=verbose)
    if df is None:
        df = load_prepared_data()
    cols = available_columns(df, EXP3_COLS)
    racing = drop_na_features(filter_status(df, 2), cols)
    balanced = stratified_subsample(racing, "team")
    _log(f"[Exp3] balanced samples={len(balanced):,} teams={balanced['team'].nunique()}", verbose=verbose)

    X = balanced[cols].values.astype(np.float32)
    result = train_vae(X, epochs=epochs, feature_cols=cols, label="Exp3 Team VAE", verbose=verbose)
    latent = encode_vae(result, X)
    balanced = balanced.copy()
    balanced["latent_idx"] = np.arange(len(balanced))

    centroids: dict[str, np.ndarray] = {}
    variances: dict[str, float] = {}
    for team, gdf in balanced.groupby("team"):
        idx = gdf["latent_idx"].values
        z = latent[idx]
        centroids[team] = z.mean(axis=0)
        variances[team] = float(z.var(axis=0).mean())

    teams = sorted(centroids.keys())
    dist = np.zeros((len(teams), len(teams)))
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            dist[i, j] = np.linalg.norm(centroids[t1] - centroids[t2])

    # Rank correlation: distance to best team vs mean rank
    top_team = (
        balanced.groupby("team")["TRK_RACE_RANK_unk"]
        .mean()
        .idxmin()
    )
    mean_ranks = balanced.groupby("team")["TRK_RACE_RANK_unk"].mean()
    dist_to_top = {t: np.linalg.norm(centroids[t] - centroids[top_team]) for t in teams}
    rho, pval = stats.spearmanr(
        [dist_to_top[t] for t in teams],
        [mean_ranks[t] for t in teams],
    )

    # Cross-venue transfer: train Halifax, embed Bermuda
    halifax = drop_na_features(filter_status(df[df["venue"] == "Halifax"], 2), cols)
    bermuda = drop_na_features(filter_status(df[df["venue"] == "Bermuda"], 2), cols)
    transfer_result = None
    transfer_metrics: dict[str, float] = {}
    if len(halifax) > 100 and len(bermuda) > 100:
        halifax_bal = stratified_subsample(halifax, "team")
        X_h = halifax_bal[cols].values.astype(np.float32)
        _log("[Exp3] cross-venue transfer: train Halifax → embed Bermuda", verbose=verbose)
        transfer_result = train_vae(
            X_h,
            epochs=max(50, epochs // 2),
            feature_cols=cols,
            label="Exp3 Transfer VAE",
            verbose=verbose,
        )
        _, _, err_bermuda = reconstruct_vae(
            transfer_result,
            bermuda[cols].values.astype(np.float32),
        )
        _, _, err_halifax = reconstruct_vae(transfer_result, X_h)
        transfer_metrics = {
            "halifax_recon_mean": float(np.mean(err_halifax)),
            "bermuda_recon_mean": float(np.mean(err_bermuda)),
            "transfer_ratio": float(np.mean(err_bermuda) / (np.mean(err_halifax) + 1e-9)),
        }

    return {
        "experiment": "exp3_team_dna",
        "feature_cols": cols,
        "n_samples": len(balanced),
        "losses": result.losses,
        "centroids": centroids,
        "variances": variances,
        "distance_matrix": dist,
        "teams": teams,
        "top_team": top_team,
        "metrics": {
            "rank_spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "rank_spearman_p": float(pval) if not np.isnan(pval) else 1.0,
            **transfer_metrics,
        },
        "balanced_df": balanced,
        "latent": latent,
        "result": result,
        "transfer_result": transfer_result,
        "meaningful": (rho < -0.3 and pval < 0.1) if not np.isnan(rho) else False,
    }


def run_experiment_4(
    df: pd.DataFrame | None = None,
    epochs: int = 80,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tactical anomaly: train on racing, score pre-start and penalties."""
    _log("=== Experiment 4: Tactical Anomaly AE ===", verbose=verbose)
    if df is None:
        df = load_prepared_data()
    cols = available_columns(df, EXP4_COLS)
    racing = drop_na_features(filter_status(df, 2), cols).sort_index()
    prestart = drop_na_features(filter_status(df, 1), cols).sort_index()
    _log(f"[Exp4] racing={len(racing):,} prestart={len(prestart):,}", verbose=verbose)

    X_race = racing[cols].values.astype(np.float32)
    result = train_dense_ae(
        X_race,
        encoder_dims=[6, 3],
        epochs=epochs,
        feature_cols=cols,
        label="Exp4 Tactical AE",
        verbose=verbose,
    )

    _, _, err_race = reconstruct_dense(result, X_race)
    racing_scored = racing.copy()
    racing_scored["recon_error"] = err_race

    if len(prestart):
        X_pre = prestart[cols].values.astype(np.float32)
        _, _, err_pre = reconstruct_dense(result, X_pre)
        prestart_scored = prestart.copy()
        prestart_scored["recon_error"] = err_pre
    else:
        prestart_scored = pd.DataFrame()
        err_pre = np.array([])

    penalty_mask = racing_scored["TRK_PENALTY_COUNT_unk"].fillna(0) > 0
    clean_err = racing_scored.loc[~penalty_mask, "recon_error"].values
    penalty_err = racing_scored.loc[penalty_mask, "recon_error"].values

    mwu_stat, mwu_p = stats.mannwhitneyu(penalty_err, clean_err, alternative="greater") if len(penalty_err) > 5 and len(clean_err) > 5 else (np.nan, 1.0)

    top5 = racing_scored.nlargest(5, "recon_error")[
        ["team", "race_label", "venue", "recon_error", "LATITUDE_GPS_unk", "LONGITUDE_GPS_unk", "TRK_LEG_NUM_unk"]
    ].copy() if "LATITUDE_GPS_unk" in racing_scored else racing_scored.nlargest(5, "recon_error")

    # Pre-start spike: last 120s mean error vs earlier
    prestart_summary: dict[str, float] = {}
    if len(prestart_scored) and "PC_TTS_s" in prestart_scored:
        last60 = prestart_scored[prestart_scored["PC_TTS_s"] <= 60]["recon_error"].mean()
        first60 = prestart_scored[prestart_scored["PC_TTS_s"] > 60]["recon_error"].mean()
        prestart_summary = {"last60s_mean": float(last60), "earlier_mean": float(first60)}

    return {
        "experiment": "exp4_tactical_anomaly",
        "feature_cols": cols,
        "n_racing": len(racing),
        "n_prestart": len(prestart),
        "losses": result.losses,
        "metrics": {
            "penalty_mean_error": float(np.mean(penalty_err)) if len(penalty_err) else np.nan,
            "clean_mean_error": float(np.mean(clean_err)),
            "mannwhitney_p": float(mwu_p) if not np.isnan(mwu_p) else 1.0,
            "mannwhitney_stat": float(mwu_stat) if not np.isnan(mwu_stat) else 0.0,
            **prestart_summary,
            **{f"race_recon_{k}": v for k, v in reconstruction_stats(err_race).items()},
        },
        "racing_scored": racing_scored,
        "prestart_scored": prestart_scored,
        "top5_moments": top5,
        "result": result,
        "meaningful": float(mwu_p) < 0.05 if not np.isnan(mwu_p) else False,
    }


def run_experiment_5(
    df: pd.DataFrame | None = None,
    epochs: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Cross-venue sparse AE generalization."""
    _log("=== Experiment 5: Cross-Venue Sparse AE ===", verbose=verbose)
    if df is None:
        df = load_prepared_data()
    cols = available_columns(df, BASELINE_22_COLS)
    bermuda = drop_na_features(filter_status(df[df["venue"] == "Bermuda"], 2), cols).sort_index()
    halifax = drop_na_features(filter_status(df[df["venue"] == "Halifax"], 2), cols).sort_index()
    _log(f"[Exp5] Bermuda={len(bermuda):,} Halifax={len(halifax):,}", verbose=verbose)

    conditions: dict[str, Any] = {}
    train_specs = [
        ("bermuda_to_halifax", bermuda, {"Bermuda": bermuda, "Halifax": halifax}),
        ("halifax_to_bermuda", halifax, {"Bermuda": bermuda, "Halifax": halifax}),
        ("joint", pd.concat([bermuda, halifax]).sort_index(), {"Bermuda": bermuda, "Halifax": halifax}),
    ]
    cond_bar = tqdm(
        train_specs,
        desc="Exp5 conditions",
        unit="cond",
        disable=not verbose,
        file=sys.stdout,
    )
    for name, train_df, eval_dfs in cond_bar:
        cond_bar.set_postfix(condition=name)
        X_train = train_df[cols].values.astype(np.float32)
        result = train_dense_ae(
            X_train,
            encoder_dims=[12, 6, 3],
            epochs=epochs,
            l1_latent=1e-4,
            use_sparse=True,
            feature_cols=cols,
            label=f"Exp5 {name}",
            verbose=verbose,
        )
        cond_metrics: dict[str, Any] = {"losses": result.losses}
        for venue_name, eval_df in eval_dfs.items():
            X_eval = eval_df[cols].values.astype(np.float32)
            _, z, err = reconstruct_dense(result, X_eval)
            cond_metrics[venue_name] = {
                "recon_stats": reconstruction_stats(err),
                "latent": z,
                "n": len(eval_df),
            }
            if "avg_tws_km_h" in eval_df.columns or True:
                tws = eval_df["TWS_SGP_km_h_1"] if "TWS_SGP_km_h_1" in eval_df else pd.Series(np.nan, index=eval_df.index)
                median_tws = tws.median()
                high = err[tws >= median_tws]
                low = err[tws < median_tws]
                cond_metrics[venue_name]["high_tws_mean"] = float(np.mean(high)) if len(high) else np.nan
                cond_metrics[venue_name]["low_tws_mean"] = float(np.mean(low)) if len(low) else np.nan

        feat_err_bermuda = per_feature_error(result, bermuda[cols].values.astype(np.float32))
        feat_err_halifax = per_feature_error(result, halifax[cols].values.astype(np.float32))
        cond_metrics["feature_error_delta"] = {
            c: float(feat_err_halifax[i] - feat_err_bermuda[i])
            for i, c in enumerate(cols)
        }
        conditions[name] = {"result": result, "metrics": cond_metrics}

    # Compare joint latent overlap via venue label mixing (higher = better overlap)
    joint_z = conditions["joint"]["metrics"]
    z_berm = joint_z["Bermuda"]["latent"]
    z_hal = joint_z["Halifax"]["latent"]
    labels = np.array([0] * len(z_berm) + [1] * len(z_hal))
    z_all = np.vstack([z_berm, z_hal])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(z_all)
    # venue mixing: avg fraction of minority venue per cluster
    mixing = []
    for c in range(4):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        frac_hal = labels[mask].mean()
        mixing.append(min(frac_hal, 1 - frac_hal))
    venue_mixing = float(np.mean(mixing)) if mixing else 0.0

    return {
        "experiment": "exp5_cross_venue",
        "feature_cols": cols,
        "conditions": conditions,
        "metrics": {
            "venue_mixing_score": venue_mixing,
            "bermuda_to_halifax_ratio": (
                conditions["bermuda_to_halifax"]["metrics"]["Halifax"]["recon_stats"]["mean"]
                / (conditions["bermuda_to_halifax"]["metrics"]["Bermuda"]["recon_stats"]["mean"] + 1e-9)
            ),
            "halifax_to_bermuda_ratio": (
                conditions["halifax_to_bermuda"]["metrics"]["Bermuda"]["recon_stats"]["mean"]
                / (conditions["halifax_to_bermuda"]["metrics"]["Halifax"]["recon_stats"]["mean"] + 1e-9)
            ),
        },
        "meaningful": venue_mixing > 0.25,
    }


def run_all_experiments(
    data_root: Path = DATA_ROOT,
    fast: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all five experiments; use fast=True for reduced epochs in CI/smoke tests."""
    _log("Loading SailGP telemetry...", verbose=verbose)
    df = load_prepared_data(data_root)
    _log(f"Loaded {len(df):,} rows from {df['venue'].nunique()} venues", verbose=verbose)
    epochs = {"e1": 30, "e2": 50, "e3": 40, "e4": 30, "e5": 40} if fast else {"e1": 100, "e2": 200, "e3": 150, "e4": 80, "e5": 100}

    runners = [
        ("exp1", lambda: run_experiment_1(df, epochs=epochs["e1"], verbose=verbose)),
        ("exp2", lambda: run_experiment_2(df, epochs=epochs["e2"], verbose=verbose)),
        ("exp3", lambda: run_experiment_3(df, epochs=epochs["e3"], verbose=verbose)),
        ("exp4", lambda: run_experiment_4(df, epochs=epochs["e4"], verbose=verbose)),
        ("exp5", lambda: run_experiment_5(df, epochs=epochs["e5"], verbose=verbose)),
    ]

    results: dict[str, Any] = {}
    exp_bar = tqdm(
        runners,
        desc="Autoencoder experiments",
        unit="exp",
        disable=not verbose,
        file=sys.stdout,
    )
    for key, runner in exp_bar:
        exp_bar.set_postfix(experiment=key)
        results[key] = runner()

    summary = []
    for key, res in results.items():
        summary.append({
            "experiment": res["experiment"],
            "meaningful": res.get("meaningful", False),
            "key_metric": _key_metric(res),
        })
    results["summary"] = pd.DataFrame(summary)
    return results


def _key_metric(res: dict[str, Any]) -> str:
    m = res.get("metrics", {})
    exp = res.get("experiment", "")
    if "foiling" in exp:
        return f"auc_test={m.get('auc_test', 0):.3f}"
    if "sailing" in exp:
        return f"silhouette={m.get('ae_silhouette', 0):.3f}, leg_align={m.get('leg_alignment', 0):.3f}"
    if "team" in exp:
        return f"rank_rho={m.get('rank_spearman_rho', 0):.3f}"
    if "tactical" in exp:
        return f"mwu_p={m.get('mannwhitney_p', 1):.4f}"
    if "cross" in exp:
        return f"venue_mixing={m.get('venue_mixing_score', 0):.3f}"
    return ""
