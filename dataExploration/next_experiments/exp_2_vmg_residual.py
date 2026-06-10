"""Experiment #2 — VMG Residual from Polar (skill minus physics)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    CA_COLS,
    COL_DB_CANT_P,
    COL_HEEL,
    COL_JIB_LEAD,
    COL_JIB_SHEET,
    COL_LEG,
    COL_PITCH,
    COL_RANK,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    COL_WING_ROT,
    COL_WING_TWIST,
    WindowSpec,
    WindowedDataset,
    build_windows,
    get_device,
    load_racing_boats,
    split_venue_races,
)
from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    EXPORT_DIR,
    evaluate_regression,
    export_eval_html,
    predict_lstm,
    ridge_baseline,
    rmse,
    run_training,
)
from dataExploration.lstm_experiments.shared.models import LSTMRegressor

# ── config ────────────────────────────────────────────────────────────────────
TWA_BIN_WIDTH = 10.0
TWS_BIN_WIDTH_KN = 2.0
KMH_PER_KN = 1.852
MIN_BIN_COUNT = 20
SEQ_LEN = 30
HORIZON = 5
STRIDE = 5

FEATURES = (
    [COL_WING_ROT, COL_WING_TWIST]
    + CA_COLS
    + [COL_JIB_LEAD, COL_JIB_SHEET, COL_DB_CANT_P, COL_TWA, COL_TWS, COL_HEEL, COL_PITCH]
)

RESIDUAL_COL = "vmg_residual"
EXPORT_PREFIX = "exp2_vmg_residual"


@dataclass
class PolarTable:
    table: dict[tuple[float, float], float]
    counts: dict[tuple[float, float], int]
    global_mean: float
    twa_bin_width: float = TWA_BIN_WIDTH
    tws_bin_width_kn: float = TWS_BIN_WIDTH_KN

    def lookup(self, twa: float, tws_kmh: float) -> float:
        if np.isnan(twa) or np.isnan(tws_kmh):
            return self.global_mean
        twa_bin = float(np.floor(twa / self.twa_bin_width) * self.twa_bin_width)
        tws_kn = tws_kmh / KMH_PER_KN
        tws_bin = float(np.floor(tws_kn / self.tws_bin_width_kn) * self.tws_bin_width_kn)
        key = (twa_bin, tws_bin)
        if key in self.table and self.counts.get(key, 0) >= MIN_BIN_COUNT:
            return self.table[key]
        return self._nearest_fallback(twa_bin, tws_bin)

    def _nearest_fallback(self, twa_bin: float, tws_bin: float) -> float:
        valid = [(k, v) for k, v in self.table.items() if self.counts.get(k, 0) >= MIN_BIN_COUNT]
        if not valid:
            return self.global_mean
        dists = [
            (np.hypot((k[0] - twa_bin) / self.twa_bin_width, (k[1] - tws_bin) / self.tws_bin_width_kn), v)
            for k, v in valid
        ]
        return min(dists, key=lambda x: x[0])[1]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (twa_b, tws_b), mean_vmg in sorted(self.table.items()):
            rows.append(
                {
                    "twa_bin_deg": twa_b,
                    "tws_bin_kn": tws_b,
                    "mean_vmg_kmh": mean_vmg,
                    "n": self.counts.get((twa_b, tws_b), 0),
                    "valid": self.counts.get((twa_b, tws_b), 0) >= MIN_BIN_COUNT,
                }
            )
        return pd.DataFrame(rows)


def _twa_bin(twa: float) -> float:
    return float(np.floor(twa / TWA_BIN_WIDTH) * TWA_BIN_WIDTH)


def _tws_bin_kn(tws_kmh: float) -> float:
    return float(np.floor((tws_kmh / KMH_PER_KN) / TWS_BIN_WIDTH_KN) * TWS_BIN_WIDTH_KN)


def build_polar_table(train_df: pd.DataFrame) -> PolarTable:
    sub = train_df.dropna(subset=[COL_VMG, COL_TWA, COL_TWS]).copy()
    sub["twa_bin"] = sub[COL_TWA].map(_twa_bin)
    sub["tws_bin_kn"] = sub[COL_TWS].map(_tws_bin_kn)
    agg = sub.groupby(["twa_bin", "tws_bin_kn"], as_index=False).agg(
        mean_vmg=(COL_VMG, "mean"),
        n=(COL_VMG, "size"),
    )
    table = {
        (float(r["twa_bin"]), float(r["tws_bin_kn"])): float(r["mean_vmg"])
        for _, r in agg.iterrows()
    }
    counts_d = {
        (float(r["twa_bin"]), float(r["tws_bin_kn"])): int(r["n"])
        for _, r in agg.iterrows()
    }
    return PolarTable(table=table, counts=counts_d, global_mean=float(sub[COL_VMG].mean()))


def attach_residuals(df: pd.DataFrame, polar: PolarTable) -> pd.DataFrame:
    out = df.copy()
    valid = out[COL_VMG].notna() & out[COL_TWA].notna() & out[COL_TWS].notna()
    out["polar_expected_vmg"] = np.nan
    out[RESIDUAL_COL] = np.nan
    out.loc[valid, "polar_expected_vmg"] = [
        polar.lookup(twa, tws)
        for twa, tws in zip(out.loc[valid, COL_TWA], out.loc[valid, COL_TWS])
    ]
    out.loc[valid, RESIDUAL_COL] = out.loc[valid, COL_VMG] - out.loc[valid, "polar_expected_vmg"]
    out["sail_mode"] = np.where(out[COL_TWA].abs() < 90, "upwind", "downwind")
    return out


def _residual_target(gdf: pd.DataFrame, end_idx: int) -> float:
    idx = end_idx + HORIZON - 1
    if idx >= len(gdf):
        return np.nan
    val = float(gdf[RESIDUAL_COL].iloc[idx])
    return val if not np.isnan(val) else np.nan


def _collect_persist_residual(df: pd.DataFrame, spec: WindowSpec) -> np.ndarray:
    ys = []
    group_cols = spec.group_cols or ["venue", "race_label", "team"]
    for _, gdf in df.groupby(group_cols, sort=False):
        gdf = gdf.sort_index()
        if RESIDUAL_COL not in gdf.columns:
            continue
        n = len(gdf)
        end = n - spec.horizon
        if end <= spec.seq_len:
            continue
        for start in range(0, end - spec.seq_len, spec.stride):
            end_idx = start + spec.seq_len
            val = float(gdf[RESIDUAL_COL].iloc[end_idx - 1])
            if np.isnan(val):
                continue
            ys.append(val)
    return np.array(ys, dtype=np.float32)


def _finishing_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Final race rank per team (lower = better)."""
    rows = []
    for (venue, race, team), gdf in df.groupby(["venue", "race_label", "team"], sort=False):
        gdf = gdf.sort_index()
        final_rank = gdf[COL_RANK].iloc[-1] if COL_RANK in gdf.columns else np.nan
        mean_res = gdf[RESIDUAL_COL].mean() if RESIDUAL_COL in gdf.columns else np.nan
        rows.append(
            {
                "venue": venue,
                "race_label": race,
                "team": team,
                "final_rank": final_rank,
                "mean_residual": mean_res,
            }
        )
    return pd.DataFrame(rows)


def _spearman_rank_correlation(skill_df: pd.DataFrame) -> tuple[float, float]:
    sub = skill_df.dropna(subset=["final_rank", "mean_residual"])
    if len(sub) < 3:
        return 0.0, 1.0
    residual_rank = sub["mean_residual"].rank(ascending=False)  # higher residual = better
    rho, p = spearmanr(residual_rank, sub["final_rank"])
    return float(rho) if not np.isnan(rho) else 0.0, float(p) if not np.isnan(p) else 1.0


def _collect_window_twa(df: pd.DataFrame, spec: WindowSpec) -> np.ndarray:
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
            twas.append(val)
    return np.array(twas, dtype=np.float32)


def _leg_breakdown(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, twas: np.ndarray) -> dict[str, Any]:
    n = min(len(y_true), len(y_pred), len(twas))
    if n == 0:
        return {}
    y_true, y_pred, twas = y_true[:n], y_pred[:n], twas[:n]
    up = np.abs(twas) < 90
    down = ~up
    out: dict[str, Any] = {}
    for label, mask in [("upwind", up), ("downwind", down)]:
        if mask.sum() < 10:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[label] = {
            "n": int(mask.sum()),
            "residual_std": float(np.std(yt)),
            "rmse": rmse(yt, yp),
            "mean_residual": float(np.mean(yt)),
        }
    return out


def _venue_transfer_direction(val_skill: pd.DataFrame, test_skill: pd.DataFrame) -> dict[str, Any]:
    """Check if residual-based team ordering direction holds across venues."""
    val_agg = val_skill.groupby("team")["mean_residual"].mean()
    test_agg = test_skill.groupby("team")["mean_residual"].mean()
    common = sorted(set(val_agg.index) & set(test_agg.index))
    if len(common) < 3:
        return {"rho": 0.0, "pass": False, "n_teams": len(common)}
    rho, _ = spearmanr(val_agg.loc[common], test_agg.loc[common])
    rho = float(rho) if not np.isnan(rho) else 0.0
    return {"rho": rho, "pass": rho > 0, "n_teams": len(common)}


@dataclass
class Exp2Result:
    experiment_id: str = "exp2_vmg_residual"
    polar_r2_train: float = 0.0
    lstm_rmse_val: float = 0.0
    ridge_rmse_val: float = 0.0
    persist_rmse_val: float = 0.0
    spearman_rho_val: float = 0.0
    spearman_rho_test: float = 0.0
    venue_transfer: dict[str, Any] = field(default_factory=dict)
    leg_breakdown_val: dict[str, Any] = field(default_factory=dict)
    leg_breakdown_test: dict[str, Any] = field(default_factory=dict)
    success: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)


def export_plots(
    polar: PolarTable,
    train_df: pd.DataFrame,
    skill_val: pd.DataFrame,
    skill_test: pd.DataFrame,
    result: Exp2Result,
    history: dict[str, list[float]],
) -> list[Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Residual histogram per team (val)
    fig1 = make_subplots(rows=1, cols=1)
    for team in sorted(skill_val["team"].unique())[:8]:
        sub = train_df[(train_df["team"] == team) & train_df[RESIDUAL_COL].notna()]
        if len(sub) < 50:
            continue
        fig1.add_trace(go.Histogram(x=sub[RESIDUAL_COL], name=team, opacity=0.6))
    fig1.update_layout(title="VMG Residual Distribution by Team (Bermuda train)", barmode="overlay")
    p1 = EXPORT_DIR / f"{EXPORT_PREFIX}_residual_hist.html"
    fig1.write_html(str(p1))
    paths.append(p1)

    # Skill rank vs finishing rank
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Bermuda val", "Halifax test"))
    for col, skill, title in [(1, skill_val, "val"), (2, skill_test, "test")]:
        sub = skill.dropna(subset=["final_rank", "mean_residual"])
        fig2.add_trace(
            go.Scatter(
                x=sub["mean_residual"],
                y=sub["final_rank"],
                mode="markers+text",
                text=sub["team"],
                textposition="top center",
                name=title,
            ),
            row=1,
            col=col,
        )
    fig2.update_layout(title="Mean Residual vs Finishing Rank")
    fig2.update_yaxes(autorange="reversed")
    p2 = EXPORT_DIR / f"{EXPORT_PREFIX}_skill_rank.html"
    fig2.write_html(str(p2))
    paths.append(p2)

    # Polar heatmap
    pdf = polar.to_dataframe()
    if not pdf.empty:
        pivot = pdf.pivot_table(index="twa_bin_deg", columns="tws_bin_kn", values="mean_vmg_kmh", aggfunc="first")
        fig3 = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale="Viridis",
            )
        )
        fig3.update_layout(title="Polar VMG Lookup (km/h)", xaxis_title="TWS bin (kn)", yaxis_title="TWA bin (deg)")
        p3 = EXPORT_DIR / f"{EXPORT_PREFIX}_polar_heatmap.html"
        fig3.write_html(str(p3))
        paths.append(p3)

    # Training + metrics dashboard
    exp_result = ExperimentResult(
        experiment_id="exp2",
        name="VMG Residual from Polar",
        task="VMG residual forecast",
        lstm_metric_name="RMSE",
        lstm_metric=result.lstm_rmse_val,
        baseline_metrics={"ridge": result.ridge_rmse_val, "persistence": result.persist_rmse_val},
        best_baseline_name="ridge" if result.ridge_rmse_val <= result.persist_rmse_val else "persistence",
        best_baseline_metric=min(result.ridge_rmse_val, result.persist_rmse_val),
        delta=min(result.ridge_rmse_val, result.persist_rmse_val) - result.lstm_rmse_val,
        has_signal=result.success.get("lstm_beats_ridge", False),
        details=result.details,
        history=history,
    )
    p4 = export_eval_html(exp_result, EXPORT_DIR / f"{EXPORT_PREFIX}_eval.html")
    paths.append(p4)

    return paths


def run() -> Exp2Result:
    print("[exp2] loading data...", flush=True)
    df = load_racing_boats()
    train_df, val_df, test_df = split_venue_races(
        df,
        train_venue="Bermuda",
        train_races={f"Race_{i}" for i in range(1, 7)},
        val_venue="Bermuda",
        val_races={f"Race_{i}" for i in range(7, 9)},
        test_venue="Halifax",
    )

    print("[exp2] building polar table from Bermuda train...", flush=True)
    polar = build_polar_table(train_df)
    polar_df = polar.to_dataframe()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    polar_path = EXPORT_DIR / f"{EXPORT_PREFIX}_polar_table.csv"
    polar_df.to_csv(polar_path, index=False)
    print(f"[exp2] polar table: {len(polar_df)} bins -> {polar_path}", flush=True)

    train_df = attach_residuals(train_df, polar)
    val_df = attach_residuals(val_df, polar)
    test_df = attach_residuals(test_df, polar)

    # Polar R² on training data
    valid_train = train_df.dropna(subset=[COL_VMG, "polar_expected_vmg"])
    polar_r2 = float(r2_score(valid_train[COL_VMG], valid_train["polar_expected_vmg"]))
    print(f"[exp2] polar R² (train) = {polar_r2:.4f}", flush=True)

    # Residual time series export
    ts_cols = ["venue", "race_label", "team", COL_TWA, COL_TWS, COL_VMG, "polar_expected_vmg", RESIDUAL_COL, "sail_mode"]
    ts_cols = [c for c in ts_cols if c in pd.concat([train_df, val_df, test_df]).columns]
    residual_ts = pd.concat([train_df, val_df, test_df])[ts_cols].dropna(subset=[RESIDUAL_COL])
    ts_path = EXPORT_DIR / f"{EXPORT_PREFIX}_residual_timeseries.csv"
    residual_ts.to_csv(ts_path, index=False)
    print(f"[exp2] residual time series: {len(residual_ts):,} rows -> {ts_path}", flush=True)

    spec = WindowSpec(
        feature_cols=FEATURES,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=STRIDE,
        target_fn=_residual_target,
    )

    print("[exp2] building LSTM windows...", flush=True)
    X_train, y_train, _ = build_windows(train_df, spec)
    X_val, y_val, _ = build_windows(val_df, spec)
    X_test, y_test, _ = build_windows(test_df, spec)
    print(f"[exp2] windows train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}", flush=True)

    if len(X_train) == 0:
        raise RuntimeError("No training windows for exp2")

    model, history, y_pred_val, _ = run_training(
        X_train,
        y_train,
        X_val,
        y_val,
        lambda: LSTMRegressor(len(FEATURES), hidden_size=128, num_layers=2),
        task="regression",
        label="exp2-vmg-residual",
    )
    lstm_metrics = evaluate_regression(y_val, y_pred_val)

    y_persist_val = _collect_persist_residual(val_df, spec)[: len(y_val)]
    persist_metrics = evaluate_regression(y_val, y_persist_val) if len(y_persist_val) == len(y_val) else {"rmse": float("inf")}

    ridge_pred_val = ridge_baseline(X_train, y_train, X_val)
    ridge_metrics = evaluate_regression(y_val, ridge_pred_val)

    test_loader = DataLoader(WindowedDataset(X_test, y_test), batch_size=64)
    y_pred_test, _ = predict_lstm(model, test_loader, task="regression", device=get_device())
    lstm_test_metrics = evaluate_regression(y_test, y_pred_test)
    ridge_pred_test = ridge_baseline(X_train, y_train, X_test)
    ridge_test_metrics = evaluate_regression(y_test, ridge_pred_test)

    twa_val = _collect_window_twa(val_df, spec)[: len(y_val)]
    twa_test = _collect_window_twa(test_df, spec)[: len(y_test)]
    leg_val = _leg_breakdown(val_df, y_val, y_pred_val, twa_val)
    leg_test = _leg_breakdown(test_df, y_test, y_pred_test, twa_test)

    # Skill ranking
    skill_val = _finishing_ranks(val_df)
    skill_test = _finishing_ranks(test_df)
    skill_all = pd.concat([_finishing_ranks(train_df), skill_val, skill_test], ignore_index=True)
    skill_val["residual_rank"] = skill_val["mean_residual"].rank(ascending=False)
    skill_test["residual_rank"] = skill_test["mean_residual"].rank(ascending=False)
    skill_path = EXPORT_DIR / f"{EXPORT_PREFIX}_skill_ranking.csv"
    skill_all.to_csv(skill_path, index=False)

    rho_val, p_val = _spearman_rank_correlation(skill_val)
    rho_test, p_test = _spearman_rank_correlation(skill_test)
    venue_transfer = _venue_transfer_direction(skill_val, skill_test)

    success = {
        "polar_r2": polar_r2 >= 0.75,
        "lstm_beats_ridge": lstm_metrics["rmse"] < ridge_metrics["rmse"],
        "spearman_rank_val": rho_val >= 0.3,
        "venue_transfer": venue_transfer.get("pass", False),
    }

    result = Exp2Result(
        polar_r2_train=polar_r2,
        lstm_rmse_val=lstm_metrics["rmse"],
        ridge_rmse_val=ridge_metrics["rmse"],
        persist_rmse_val=persist_metrics["rmse"],
        spearman_rho_val=rho_val,
        spearman_rho_test=rho_test,
        venue_transfer=venue_transfer,
        leg_breakdown_val=leg_val,
        leg_breakdown_test=leg_test,
        success=success,
        details={
            "polar_r2_train": polar_r2,
            "lstm_rmse_val": lstm_metrics["rmse"],
            "lstm_rmse_test": lstm_test_metrics["rmse"],
            "ridge_rmse_val": ridge_metrics["rmse"],
            "ridge_rmse_test": ridge_test_metrics["rmse"],
            "persist_rmse_val": persist_metrics["rmse"],
            "spearman_rho_val": rho_val,
            "spearman_p_val": p_val,
            "spearman_rho_test": rho_test,
            "spearman_p_test": p_test,
            "n_train_windows": len(X_train),
            "n_val_windows": len(X_val),
            "n_test_windows": len(X_test),
            "n_polar_bins": len(polar_df),
            "n_valid_bins": int(polar_df["valid"].sum()) if "valid" in polar_df.columns else 0,
        },
        history=history,
    )

    json_path = EXPORT_DIR / f"{EXPORT_PREFIX}_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                **asdict(result),
                "output_paths": {
                    "polar_table": str(polar_path),
                    "residual_timeseries": str(ts_path),
                    "skill_ranking": str(skill_path),
                    "results_json": str(json_path),
                },
            },
            f,
            indent=2,
        )

    plot_paths = export_plots(polar, train_df, skill_val, skill_test, result, history)
    result.details["plot_paths"] = [str(p) for p in plot_paths]

    print("\n[exp2] SUCCESS CRITERIA:", flush=True)
    for k, v in success.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"\n[exp2] key metrics:", flush=True)
    print(f"  polar R²={polar_r2:.4f}  LSTM RMSE={lstm_metrics['rmse']:.4f}  Ridge RMSE={ridge_metrics['rmse']:.4f}", flush=True)
    print(f"  Spearman ρ (val)={rho_val:.4f}  venue transfer ρ={venue_transfer.get('rho', 0):.4f}", flush=True)

    return result


if __name__ == "__main__":
    run()
