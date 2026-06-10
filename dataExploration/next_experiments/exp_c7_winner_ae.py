#!/usr/bin/env python3
"""C7 — Winner Profile Autoencoder: distance from winning behaviour."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.autoencoder_experiments import (
    BASELINE_22_COLS,
    DenseAE,
    TrainResult,
    _train_epoch_batches,
    available_columns,
    drop_na_features,
    filter_status,
    load_prepared_data,
    time_split,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
EXPORT_PREFIX = "winner_ae"

ENCODER_DIMS = [64, 32, 16]
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 512
LR = 1e-3
WINNER_RANK_MAX = 3

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@dataclass
class TrainResultLocal(TrainResult):
    device: torch.device = field(default_factory=lambda: device)
    train_time_s: float = 0.0


def train_winner_ae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> TrainResultLocal:
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train).astype(np.float32)
    Xs_val = scaler.transform(X_val).astype(np.float32)
    n_in = Xs_train.shape[1]

    model = DenseAE(n_in, ENCODER_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    Xt = torch.tensor(Xs_train, device=device)
    Xv = torch.tensor(Xs_val, device=device)

    losses: list[float] = []
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = _train_epoch_batches(
            model, Xt, opt, BATCH_SIZE,
            lambda b: mse(model(b)[0], b),
            epoch=epoch, epochs=epochs, epoch_bar=None, label="C7 Winner AE", verbose=False,
        )
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(Xv)
            val_loss = float(mse(val_recon, Xv).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[C7] early stop epoch {epoch + 1}, val_loss={val_loss:.4f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    elapsed = time.perf_counter() - t0
    print(
        f"[C7] trained {len(losses)} epochs | val_loss={best_val:.4f} | "
        f"samples={len(Xs_train):,} | device={device} | {elapsed:.1f}s",
        flush=True,
    )
    return TrainResultLocal(
        model=model, losses=losses, scaler=scaler,
        feature_cols=feature_cols, device=device, train_time_s=elapsed,
    )


@torch.no_grad()
def reconstruct_all(result: TrainResultLocal, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result.model.eval()
    Xs = result.scaler.transform(X).astype(np.float32)
    Xt = torch.tensor(Xs, device=result.device)
    recon, _ = result.model(Xt)
    recon_np = recon.cpu().numpy()
    err = ((recon_np - Xs) ** 2).mean(axis=1)
    per_feat = (recon_np - Xs) ** 2
    return recon_np, per_feat, err


def norm_similarity(df: pd.DataFrame, err_col: str = "recon_error") -> pd.Series:
    def _norm(e: pd.Series) -> pd.Series:
        lo, hi = float(e.min()), float(e.max())
        if hi - lo < 1e-12:
            return pd.Series(0.5, index=e.index)
        return 1.0 - (e - lo) / (hi - lo)

    return df.groupby(["venue", "race_label"], group_keys=False)[err_col].transform(_norm)


def feature_importance(
    per_feat: np.ndarray,
    is_winner: np.ndarray,
    cols: list[str],
) -> pd.DataFrame:
    winner_mask = is_winner.astype(bool)
    rows = []
    for i, col in enumerate(cols):
        w_err = float(per_feat[winner_mask, i].mean())
        nw_err = float(per_feat[~winner_mask, i].mean())
        rows.append({
            "feature": col,
            "mean_error_winner": w_err,
            "mean_error_nonwinner": nw_err,
            "gap": nw_err - w_err,
        })
    imp = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def team_race_scores(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["venue", "race_label", "team"], as_index=False)
        .agg(
            mean_winner_similarity=("winner_similarity", "mean"),
            mean_recon_error=("recon_error", "mean"),
            n_seconds=("recon_error", "count"),
        )
    )
    finish = (
        df.sort_index()
        .groupby(["venue", "race_label", "team"])["TRK_RACE_RANK_unk"]
        .last()
        .reset_index()
        .rename(columns={"TRK_RACE_RANK_unk": "actual_rank"})
    )
    return agg.merge(finish, on=["venue", "race_label", "team"])


def gap_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    feat_cols = [f"feat_err_{c}" for c in cols]
    rows = []
    for (venue, race, team), g in df.groupby(["venue", "race_label", "team"]):
        means = g[feat_cols].mean()
        top3 = means.nlargest(3)
        for rank, (feat_col, val) in enumerate(top3.items(), 1):
            feat = feat_col.replace("feat_err_", "")
            rows.append({
                "venue": venue,
                "race_label": race,
                "team": team,
                "gap_rank": rank,
                "feature": feat,
                "mean_feature_error": float(val),
            })
    return pd.DataFrame(rows)


def export_timeline(df: pd.DataFrame, path: Path) -> None:
    races = (
        df.groupby(["venue", "race_label"], as_index=False)
        .size()
        .sort_values(["venue", "race_label"])
    )
    n = len(races)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    titles = [f"{r['venue']} {r['race_label']}" for _, r in races.iterrows()]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.1)

    teams = sorted(df["team"].unique())
    palette = [
        "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
        "#0891b2", "#ea580c", "#4f46e5", "#be123c", "#059669",
        "#7c3aed", "#0d9488", "#b45309",
    ]
    team_color = {t: palette[i % len(palette)] for i, t in enumerate(teams)}

    for i, (_, race_row) in enumerate(races.iterrows()):
        r, c = divmod(i, cols)
        sub = df[
            (df["venue"] == race_row["venue"]) & (df["race_label"] == race_row["race_label"])
        ]
        for team in teams:
            ts = sub[sub["team"] == team].sort_index()
            if ts.empty:
                continue
            x = ts["elapsed_s"] if "elapsed_s" in ts else np.arange(len(ts))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ts["winner_similarity"],
                    name=team,
                    legendgroup=team,
                    showlegend=(i == 0),
                    line=dict(color=team_color[team], width=1.5),
                    mode="lines",
                ),
                row=r + 1,
                col=c + 1,
            )

    fig.update_layout(
        title="Winner Similarity Timeline (1 = most winner-like)",
        height=max(400, 220 * rows),
        width=1200,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_yaxes(title_text="winner_similarity", range=[0, 1])
    fig.write_html(str(path), include_plotlyjs="cdn")


def export_gap_html(gap_df: pd.DataFrame, path: Path) -> None:
    display = gap_df.copy()
    display["label"] = (
        display["venue"] + " " + display["race_label"] + " — " + display["team"]
        + " #" + display["gap_rank"].astype(str) + ": " + display["feature"]
        + f" ({display['mean_feature_error'].map(lambda x: f'{x:.4f}')})"
    )
    pivot_rows = []
    for (venue, race, team), g in display.groupby(["venue", "race_label", "team"]):
        feats = ", ".join(
            f"{r['feature']} ({r['mean_feature_error']:.4f})"
            for _, r in g.sort_values("gap_rank").iterrows()
        )
        pivot_rows.append({
            "venue": venue,
            "race_label": race,
            "team": team,
            "top_3_gap_features": feats,
        })
    table_df = pd.DataFrame(pivot_rows).sort_values(["venue", "race_label", "team"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Venue", "Race", "Team", "Top 3 Gap Features"],
                    fill_color="#1e293b",
                    font=dict(color="white"),
                ),
                cells=dict(
                    values=[
                        table_df["venue"],
                        table_df["race_label"],
                        table_df["team"],
                        table_df["top_3_gap_features"],
                    ],
                    fill_color="#f8fafc",
                ),
            )
        ]
    )
    fig.update_layout(title="Per Team × Race — Top 3 Gap Features", height=600)
    fig.write_html(str(path), include_plotlyjs="cdn")


def race_winner_identification(team_scores: pd.DataFrame) -> dict:
    """Races where highest mean_winner_similarity team matches lowest actual_rank."""
    correct = 0
    total = 0
    details = []
    for (venue, race), g in team_scores.groupby(["venue", "race_label"]):
        g = g.dropna(subset=["actual_rank", "mean_winner_similarity"])
        if len(g) < 4:
            continue
        total += 1
        pred = g.loc[g["mean_winner_similarity"].idxmax(), "team"]
        actual = g.loc[g["actual_rank"].idxmin(), "team"]
        ok = pred == actual
        if ok:
            correct += 1
        details.append({
            "venue": venue,
            "race_label": race,
            "predicted_winner": pred,
            "actual_winner": actual,
            "correct": ok,
        })
    return {"correct": correct, "total": total, "details": details}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def run() -> dict:
    print(f"[C7] device={device} | mps_available={torch.backends.mps.is_available()}", flush=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_prepared_data()
    cols = available_columns(df, BASELINE_22_COLS)
    racing = drop_na_features(filter_status(df, 2), cols).sort_index()
    racing = racing.copy()
    racing["is_winner_tier"] = racing["TRK_RACE_RANK_unk"] <= WINNER_RANK_MAX

    winner_df = racing[racing["is_winner_tier"]]
    if len(winner_df) < 500:
        raise ValueError(f"Too few winner-tier rows: {len(winner_df)}")

    X_winner = winner_df[cols].values.astype(np.float32)
    X_train, X_val = time_split(X_winner, 0.8)
    print(
        f"[C7] racing={len(racing):,} winner_tier={len(winner_df):,} "
        f"train={len(X_train):,} val={len(X_val):,} features={len(cols)}",
        flush=True,
    )

    result = train_winner_ae(X_train, X_val, cols)

    X_all = racing[cols].values.astype(np.float32)
    _, per_feat, err = reconstruct_all(result, X_all)
    racing["recon_error"] = err
    racing["winner_similarity"] = norm_similarity(racing)

    # elapsed seconds within each team×race for timeline x-axis
    racing = racing.reset_index().rename(columns={"index": "timestamp"})
    racing["elapsed_s"] = racing.groupby(["venue", "race_label", "team"]).cumcount()

    for i, col in enumerate(cols):
        racing[f"feat_err_{col}"] = per_feat[:, i]

    is_winner = racing["is_winner_tier"].to_numpy()
    imp = feature_importance(per_feat, is_winner, cols)

    winner_err = racing.loc[racing["is_winner_tier"], "recon_error"].to_numpy()
    nonwinner_err = racing.loc[~racing["is_winner_tier"], "recon_error"].to_numpy()
    mwu_stat, mwu_p = stats.mannwhitneyu(nonwinner_err, winner_err, alternative="greater")

    team_scores = team_race_scores(racing)
    gap_df = gap_table(racing, cols)
    race_check = race_winner_identification(team_scores)

    wind_feats = {"TWA_SGP_deg", "TWS_SGP_km_h_1"}
    top10 = imp.head(10)
    controllable_top10 = int((~top10["feature"].isin(wind_feats)).sum())

    imp_path = EXPORT_DIR / f"{EXPORT_PREFIX}_feature_importance.csv"
    scores_path = EXPORT_DIR / f"{EXPORT_PREFIX}_team_scores.csv"
    timeline_path = EXPORT_DIR / f"{EXPORT_PREFIX}_timeline.html"
    gap_path = EXPORT_DIR / f"{EXPORT_PREFIX}_gap_table.html"
    json_path = EXPORT_DIR / f"{EXPORT_PREFIX}_results.json"

    imp.to_csv(imp_path, index=False)
    team_scores.to_csv(scores_path, index=False)
    export_timeline(racing, timeline_path)
    export_gap_html(gap_df, gap_path)

    results = {
        "experiment": "c7_winner_profile_ae",
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "training_time_s": round(result.train_time_s, 2),
        "epochs_trained": len(result.losses),
        "encoder_dims": ENCODER_DIMS,
        "feature_cols": cols,
        "n_racing": len(racing),
        "n_winner_tier": int(racing["is_winner_tier"].sum()),
        "metrics": {
            "mean_error_winner": float(winner_err.mean()),
            "mean_error_nonwinner": float(nonwinner_err.mean()),
            "mannwhitney_stat": float(mwu_stat),
            "mannwhitney_p": float(mwu_p),
            "winner_err_lower": bool(winner_err.mean() < nonwinner_err.mean()),
        },
        "top_5_features": imp.head(5).to_dict(orient="records"),
        "success_criteria": {
            "mwu_p_lt_0.01": bool(mwu_p < 0.01),
            "controllable_in_top10": controllable_top10 >= 8,
            "races_winner_identified_ge_3": race_check["correct"] >= 3,
        },
        "race_winner_check": race_check,
        "outputs": {
            "feature_importance_csv": str(imp_path),
            "team_scores_csv": str(scores_path),
            "timeline_html": str(timeline_path),
            "gap_table_html": str(gap_path),
            "results_json": str(json_path),
        },
    }

    with open(json_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)

    print(f"[C7] MWU p={mwu_p:.2e} | top feature gap={imp.iloc[0]['feature']}", flush=True)
    print(f"[C7] outputs written to {EXPORT_DIR}", flush=True)
    return results


if __name__ == "__main__":
    run()
