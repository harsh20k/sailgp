"""Experiment 1 — Flight Quality Score (AE recon error + TCN foiling prob fusion)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.autoencoder_experiments import (
    EXP1_COLS,
    available_columns,
    drop_na_features,
    filter_status,
    load_prepared_data,
    reconstruct_dense,
    time_split,
    train_dense_ae,
)
from dataExploration.lstm_experiments.shared.data_prep import COL_LEG, COL_RANK, COL_STATUS, foiling_label
from sailgp_analysis.tcm.dataset import (
    VARIATION_CONFIGS,
    WindowDataset,
    build_windows_for_group,
    load_prepared_frames,
)
from sailgp_analysis.tcm.train import TrainConfig, train_model

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
AE_EPOCHS = 30
TCN_EPOCHS = 30
WINDOW_SIZE = 30
ROLLING_S = 5
FUSION_WEIGHTS = (0.5, 0.5)
BERMUDA_RACES = [f"Race_{i}" for i in range(1, 9)]
TRAIN_RACES = BERMUDA_RACES[:7]
AUC_THRESHOLD = 0.87
AE_BASELINE_AUC = 0.865


@dataclass
class CriterionResult:
    criterion: str
    passed: bool
    value: float | str
    threshold: str
    details: dict


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _minmax_invert(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5, dtype=np.float64)
    norm = (values - lo) / (hi - lo)
    return 1.0 - norm


def train_foiling_ae(df: pd.DataFrame, epochs: int = AE_EPOCHS) -> tuple[object, pd.Series]:
    """Train Exp1 foiling AE; return model result and per-row recon errors (racing rows)."""
    cols = available_columns(df, EXP1_COLS)
    racing = drop_na_features(filter_status(df, 2), cols).sort_index()
    X = racing[cols].values.astype(np.float32)
    X_train, _ = time_split(X, 0.8)
    print(f"[exp1-fq] training AE on {len(X_train):,} rows, {epochs} epochs", flush=True)
    result = train_dense_ae(
        X_train,
        encoder_dims=[16, 8, 4],
        epochs=epochs,
        feature_cols=cols,
        label="FlightQuality AE",
        verbose=True,
    )
    _, _, err = reconstruct_dense(result, X)
    return result, pd.Series(err, index=racing.index, name="ae_error")


def row_ae_errors(result, df: pd.DataFrame) -> pd.Series:
    """Infer AE recon error for any rows with Exp1 features."""
    cols = result.feature_cols
    sub = drop_na_features(df, cols).sort_index()
    if sub.empty:
        return pd.Series(dtype=float)
    _, _, err = reconstruct_dense(result, sub[cols].values.astype(np.float32))
    return pd.Series(err, index=sub.index, name="ae_error")


def train_tcn_foiling(train_df: pd.DataFrame) -> tuple[torch.nn.Module, dict, list[str]]:
    """Train TCN foiling classifier on Bermuda train races."""
    cfg = VARIATION_CONFIGS["foiling"]
    all_w, all_t, all_m, stats = [], [], [], {}
    for _, grp in train_df.groupby(["venue", "race_label", "team"], sort=False):
        w, t, m, stats = build_windows_for_group(grp, cfg, stats if stats else None)
        if len(w):
            all_w.append(w)
            all_t.append(t)
            all_m.extend(m)
    if not all_w:
        raise RuntimeError("No TCN training windows")
    train_ds = WindowDataset(np.concatenate(all_w), np.concatenate(all_t), all_m)
    print(f"[exp1-fq] training TCN on {len(train_ds):,} windows", flush=True)
    model, history = train_model(
        train_ds,
        task="classification",
        n_features=len(cfg.features),
        config=TrainConfig(epochs=TCN_EPOCHS, verbose=True, show_batch_progress=False),
        label="FlightQuality/TCN",
    )
    return model, stats, cfg.features


@torch.no_grad()
def tcn_probs(model: torch.nn.Module, windows: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    ds = WindowDataset(windows, np.zeros(len(windows)))
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    probs = []
    dev = torch.device(device)
    model.to(dev)
    for batch in loader:
        x = batch[0].to(dev)
        logits = model(x).cpu().numpy()
        probs.append(_sigmoid(logits))
    return np.concatenate(probs)


def build_fused_frame(
    df: pd.DataFrame,
    ae_series: pd.Series,
    tcn_model: torch.nn.Module,
    feature_stats: dict,
    device: str,
) -> pd.DataFrame:
    """Build per-window fused scores with timestamps."""
    cfg = VARIATION_CONFIGS["foiling"]
    ae_series = ae_series[~ae_series.index.duplicated(keep="last")]
    rows: list[dict] = []

    for (venue, race, team), grp in df.groupby(["venue", "race_label", "team"], sort=False):
        grp = grp.sort_index().assign(ae_error=ae_series.reindex(grp.index).to_numpy())
        feats = cfg.features
        need = feats + (["foiling"] if "foiling" in grp.columns else [])
        g = grp.dropna(subset=[c for c in need if c in grp.columns] + ["ae_error"])
        if len(g) < cfg.window_size + cfg.horizon:
            continue

        w, _, meta, _ = build_windows_for_group(g, cfg, feature_stats)
        if len(w) == 0:
            continue
        probs = tcn_probs(tcn_model, w, device)
        ws, h = cfg.window_size, cfg.horizon

        for j, m in enumerate(meta):
            idx = m["idx"]
            end_idx = idx + ws - 1 + (h - 1 if h >= 1 else 0)
            if end_idx >= len(g):
                continue
            ae_mean = float(g["ae_error"].iloc[idx : idx + ws].mean())
            ts_end = g.index[end_idx]
            row_end = g.iloc[end_idx]
            rows.append(
                {
                    "timestamp": ts_end,
                    "venue": venue,
                    "race_label": race,
                    "team": team,
                    "ae_error": ae_mean,
                    "tcn_prob": float(probs[j]),
                    "foiling": foiling_label(row_end),
                    "rank": float(row_end.get(COL_RANK, np.nan)),
                    "leg": float(row_end.get(COL_LEG, np.nan)),
                    "race_status": int(row_end.get(COL_STATUS, 2)),
                }
            )

    return pd.DataFrame(rows)


def finalize_scores(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize, fuse, expand to per-second timeline with rolling mean."""
    if windows_df.empty:
        return windows_df

    ae_quality = _minmax_invert(windows_df["ae_error"].to_numpy())
    w_ae, w_tcn = FUSION_WEIGHTS
    raw = w_ae * ae_quality + w_tcn * windows_df["tcn_prob"].to_numpy()

    out = windows_df.copy()
    out["ae_quality"] = ae_quality
    out["flight_quality_raw"] = raw
    score_cols = ["flight_quality_raw", "ae_quality", "tcn_prob", "ae_error"]
    meta_cols = ["rank", "leg", "race_status"]

    parts = []
    for keys, gdf in out.groupby(["venue", "race_label", "team"], sort=False):
        gdf = gdf.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        gdf = gdf.set_index("timestamp")
        full_idx = pd.date_range(gdf.index.min(), gdf.index.max(), freq="1s")
        scores = gdf[score_cols].reindex(full_idx).ffill()
        meta = gdf[meta_cols].reindex(full_idx).ffill()
        merged = scores.join(meta)
        merged["flight_quality"] = merged["flight_quality_raw"].rolling(ROLLING_S, min_periods=1).mean()
        merged = merged.reset_index().rename(columns={"index": "timestamp"})
        for k, v in zip(["venue", "race_label", "team"], keys):
            merged[k] = v
        parts.append(merged)

    return pd.concat(parts, ignore_index=True)


def attach_foiling_labels(scores: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Join foiling labels onto per-second scores by timestamp."""
    label_rows = []
    for (venue, race, team), grp in source_df.groupby(["venue", "race_label", "team"], sort=False):
        grp = grp.sort_index()
        tmp = pd.DataFrame(
            {
                "timestamp": grp.index,
                "foiling": [foiling_label(row) for _, row in grp.iterrows()],
                "venue": venue,
                "race_label": race,
                "team": team,
            }
        )
        label_rows.append(tmp)
    labels = pd.concat(label_rows, ignore_index=True)
    out = scores.drop(columns=["foiling"], errors="ignore")
    return out.merge(labels, on=["timestamp", "venue", "race_label", "team"], how="left")


def leader_mid_gap(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean flight_quality gap: race leader vs mid-fleet (ranks 4–6)."""
    rows = []
    racing = scores[scores["race_status"] == 2].copy()
    for (venue, race), gdf in racing.groupby(["venue", "race_label"], sort=False):
        last_rank = gdf.sort_values("timestamp").groupby("team")["rank"].last()
        leader_team = last_rank.idxmin() if last_rank.notna().any() else None
        mid_teams = last_rank[(last_rank >= 4) & (last_rank <= 6)].index.tolist()
        if leader_team is None or not mid_teams:
            continue
        leader_mean = gdf[gdf["team"] == leader_team]["flight_quality"].mean()
        mid_mean = gdf[gdf["team"].isin(mid_teams)]["flight_quality"].mean()
        rows.append(
            {
                "venue": venue,
                "race_label": race,
                "leader_team": leader_team,
                "leader_mean_fq": leader_mean,
                "mid_fleet_mean_fq": mid_mean,
                "gap": leader_mean - mid_mean,
                "n_mid_teams": len(mid_teams),
            }
        )
    return pd.DataFrame(rows)


def evaluate_criteria(scores: pd.DataFrame, prestart: pd.DataFrame) -> list[CriterionResult]:
    racing = scores[scores["race_status"] == 2].dropna(subset=["flight_quality", "foiling"])
    racing = racing.copy()
    racing["foiling"] = racing["foiling"].astype(bool)
    results: list[CriterionResult] = []

    # Temporal alignment
    foil_mean = racing.loc[racing["foiling"], "flight_quality"].mean()
    non_foil_mean = racing.loc[~racing["foiling"], "flight_quality"].mean()
    align_gap = float(foil_mean - non_foil_mean)
    t_stat, t_p = stats.ttest_ind(
        racing.loc[racing["foiling"], "flight_quality"],
        racing.loc[~racing["foiling"], "flight_quality"],
        equal_var=False,
    )
    results.append(
        CriterionResult(
            criterion="Temporal alignment",
            passed=align_gap > 0.05 and t_p < 0.05,
            value=round(align_gap, 4),
            threshold="Score higher when foiling; gap>0.05, p<0.05",
            details={"foiling_mean": foil_mean, "non_foiling_mean": non_foil_mean, "p_value": float(t_p)},
        )
    )

    # AUC vs foiling labels
    y_true = racing["foiling"].astype(int).to_numpy()
    y_score = racing["flight_quality"].to_numpy()
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    ae_only = float(roc_auc_score(y_true, racing["ae_quality"].to_numpy())) if len(np.unique(y_true)) > 1 else 0.5
    results.append(
        CriterionResult(
            criterion="AUC when thresholded",
            passed=auc > AUC_THRESHOLD,
            value=round(auc, 4),
            threshold=f"> {AUC_THRESHOLD} (AE baseline {AE_BASELINE_AUC})",
            details={"ae_only_auc": round(ae_only, 4), "improvement_vs_ae": round(auc - ae_only, 4)},
        )
    )

    # Team spread across Bermuda races
    race_rhos = []
    for race in BERMUDA_RACES:
        sub = racing[(racing["venue"] == "Bermuda") & (racing["race_label"] == race)]
        if sub.empty:
            continue
        team_fq = sub.groupby("team")["flight_quality"].mean()
        finish_rank = sub.sort_values("timestamp").groupby("team")["rank"].last()
        common = team_fq.index.intersection(finish_rank.dropna().index)
        if len(common) < 4:
            continue
        rho, _ = stats.spearmanr(team_fq[common], finish_rank[common])
        race_rhos.append({"race": race, "spearman_rho": float(rho), "passed": float(rho) < -0.3})

    n_pass = sum(1 for r in race_rhos if r["passed"])
    results.append(
        CriterionResult(
            criterion="Team spread",
            passed=n_pass >= 4,
            value=n_pass,
            threshold="Clear rank ordering on ≥4/8 races (Spearman ρ<-0.3 vs finish rank)",
            details={"per_race": race_rhos, "races_passed": n_pass},
        )
    )

    # Sanity check — pre-race must be excluded or score < 0.3
    pre_mean = float(prestart["flight_quality"].mean()) if len(prestart) else np.nan
    pre_excluded = len(prestart) == 0 or prestart["flight_quality"].isna().all()
    pre_ok = pre_excluded or (not np.isnan(pre_mean) and pre_mean < 0.3)
    clean_flying = racing[racing["foiling"] & (racing["flight_quality"] > 0)]
    clean_mean = float(clean_flying["flight_quality"].mean()) if len(clean_flying) else 0.0
    high_foil_frac = float((racing.loc[racing["foiling"], "flight_quality"] > 0.7).mean()) if racing["foiling"].any() else 0.0
    sanity_pass = pre_ok and high_foil_frac > 0.5
    pre_label = "excluded" if pre_excluded else f"{pre_mean:.3f}"
    results.append(
        CriterionResult(
            criterion="Sanity check",
            passed=sanity_pass,
            value=f"pre={pre_label}, foiling>0.7={high_foil_frac:.3f}",
            threshold="Pre-race <0.3 or excluded; >50% foiling seconds >0.7",
            details={
                "pre_race_mean": pre_mean if not pre_excluded else None,
                "pre_race_excluded": pre_excluded,
                "clean_flying_mean": clean_mean,
                "foiling_above_0.7_frac": high_foil_frac,
            },
        )
    )

    return results


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
    return obj


def export_html(scores: pd.DataFrame, gaps: pd.DataFrame, criteria: list[CriterionResult]) -> Path:
    bermuda = scores[(scores["venue"] == "Bermuda") & (scores["race_status"] == 2)]
    races = sorted(bermuda["race_label"].unique())
    n = len(races)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=races, vertical_spacing=0.08)

    teams = sorted(bermuda["team"].unique())
    palette = [
        "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
        "#0891b2", "#ea580c", "#4f46e5", "#be123c", "#059669",
    ]
    team_color = {t: palette[i % len(palette)] for i, t in enumerate(teams)}

    for i, race in enumerate(races):
        r, c = divmod(i, cols)
        sub = bermuda[bermuda["race_label"] == race]
        for team in teams:
            ts = sub[sub["team"] == team]
            if ts.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ts["timestamp"],
                    y=ts["flight_quality"],
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
        title="Flight Quality Score — Bermuda races (per team)",
        height=max(400, 220 * rows),
        width=1200,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_yaxes(title_text="flight_quality", range=[0, 1])

    path = EXPORT_DIR / "flight_quality.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = TrainConfig().device

    df = load_prepared_frames()
    bermuda = df[(df["venue"] == "Bermuda") & (df["race_label"].isin(BERMUDA_RACES))].copy()
    bermuda_racing = bermuda[bermuda[COL_STATUS] == 2].copy()
    train_mask = bermuda_racing["race_label"].isin(TRAIN_RACES)
    train_racing = bermuda_racing[train_mask]

    ae_result, _ = train_foiling_ae(df)
    ae_all = row_ae_errors(ae_result, bermuda_racing)
    tcn_model, feat_stats, _ = train_tcn_foiling(train_racing)

    # Gate on racing status before scoring — pre-start rows excluded from output
    windows = build_fused_frame(bermuda_racing, ae_all, tcn_model, feat_stats, device)
    scores = attach_foiling_labels(finalize_scores(windows), bermuda_racing)
    scores = scores[scores["race_status"] == 2].copy()
    pre_scores = pd.DataFrame(columns=["flight_quality"])

    gaps = leader_mid_gap(scores)
    criteria = evaluate_criteria(scores, pre_scores)

    csv_path = EXPORT_DIR / "flight_quality.csv"
    out_cols = [
        "timestamp", "venue", "race_label", "team", "flight_quality",
        "flight_quality_raw", "ae_quality", "tcn_prob", "ae_error", "foiling", "rank", "leg", "race_status",
    ]
    scores[out_cols].to_csv(csv_path, index=False)

    json_path = EXPORT_DIR / "flight_quality_results.json"
    payload = {
        "experiment": "exp1_flight_quality",
        "fusion_weights": {"ae_quality": FUSION_WEIGHTS[0], "tcn_prob": FUSION_WEIGHTS[1]},
        "rolling_seconds": ROLLING_S,
        "n_rows": len(scores),
        "criteria": {c.criterion: {"passed": c.passed, "value": c.value, "threshold": c.threshold, "details": c.details} for c in criteria},
        "all_passed": all(c.passed for c in criteria),
        "leader_mid_gaps": gaps.to_dict(orient="records"),
        "outputs": {
            "csv": str(csv_path),
            "html": str(EXPORT_DIR / "flight_quality.html"),
        },
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2))

    html_path = export_html(scores, gaps, criteria)

    print("\n=== Flight Quality Results ===", flush=True)
    for c in criteria:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.criterion}: {c.value} ({c.threshold})", flush=True)
    print(f"\nWrote {csv_path}, {html_path}, {json_path}", flush=True)

    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
