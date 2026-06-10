"""Experiment C8 — First-Minute Race Fingerprint.

Which sensor channels in the first 60s after the start gun predict final race rank?
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.autoencoder_experiments import (
    BASELINE_22_COLS,
    available_columns,
    load_prepared_data,
)
from dataExploration.lstm_experiments.shared.data_prep import COL_RANK, COL_STATUS

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
EXPORT_PREFIX = "first_minute"
WINDOW_S = 60
SUB_EARLY_S = 20
WINNER_RANK_MAX = 4
BOTTOM_RANK_MIN = 8
RHO_THRESHOLD = 0.3
WIND_COLS = ["TWS_SGP_km_h_1", "TWA_SGP_deg"]


@dataclass
class TeamRaceRecord:
    venue: str
    race_label: str
    team: str
    final_rank: float
    tier: str
    n_rows_60: int
    n_rows_0_20: int
    n_rows_20_60: int
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_means_0_20: dict[str, float]
    feature_means_20_60: dict[str, float]
    wind_tws_mean: float
    wind_twa_mean: float


def _tier(rank: float) -> str:
    if np.isnan(rank):
        return "unknown"
    if rank <= WINNER_RANK_MAX:
        return "winner"
    if rank >= BOTTOM_RANK_MIN:
        return "bottom"
    return "middle"


def _window_stats(df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {c: float(df[c].mean()) for c in cols}
    stds = {c: float(df[c].std(ddof=0)) if len(df) > 1 else 0.0 for c in cols}
    return means, stds


def extract_team_race_records(df: pd.DataFrame, cols: list[str]) -> list[TeamRaceRecord]:
    records: list[TeamRaceRecord] = []
    group_cols = ["venue", "race_label", "team"]
    for (venue, race, team), grp in df.groupby(group_cols, sort=False):
        grp = grp.sort_index()
        racing = grp[grp[COL_STATUS] == 2]
        if racing.empty or COL_RANK not in grp.columns:
            continue
        final_rank = float(racing[COL_RANK].iloc[-1])
        w60 = racing.iloc[:WINDOW_S]
        if len(w60) < SUB_EARLY_S:
            continue
        w0_20 = w60.iloc[:SUB_EARLY_S]
        w20_60 = w60.iloc[SUB_EARLY_S:WINDOW_S] if len(w60) >= WINDOW_S else w60.iloc[SUB_EARLY_S:]

        means, stds = _window_stats(w60, cols)
        means_0_20, _ = _window_stats(w0_20, cols)
        means_20_60, _ = _window_stats(w20_60, cols) if len(w20_60) else ({c: np.nan for c in cols}, {})

        tws = float(w60["TWS_SGP_km_h_1"].mean()) if "TWS_SGP_km_h_1" in w60.columns else np.nan
        twa = float(w60["TWA_SGP_deg"].mean()) if "TWA_SGP_deg" in w60.columns else np.nan

        records.append(
            TeamRaceRecord(
                venue=venue,
                race_label=race,
                team=team,
                final_rank=final_rank,
                tier=_tier(final_rank),
                n_rows_60=len(w60),
                n_rows_0_20=len(w0_20),
                n_rows_20_60=len(w20_60),
                feature_means=means,
                feature_stds=stds,
                feature_means_0_20=means_0_20,
                feature_means_20_60=means_20_60,
                wind_tws_mean=tws,
                wind_twa_mean=twa,
            )
        )
    return records


def records_to_frame(records: list[TeamRaceRecord], cols: list[str]) -> pd.DataFrame:
    rows = []
    for r in records:
        row: dict[str, Any] = {
            "venue": r.venue,
            "race_label": r.race_label,
            "team": r.team,
            "final_rank": r.final_rank,
            "tier": r.tier,
            "n_rows_60": r.n_rows_60,
            "wind_tws_mean": r.wind_tws_mean,
            "wind_twa_mean": r.wind_twa_mean,
        }
        for c in cols:
            row[f"mean_{c}"] = r.feature_means.get(c, np.nan)
            row[f"std_{c}"] = r.feature_stds.get(c, np.nan)
            row[f"mean0_20_{c}"] = r.feature_means_0_20.get(c, np.nan)
            row[f"mean20_60_{c}"] = r.feature_means_20_60.get(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _spearman(feature: np.ndarray, rank: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(feature) & np.isfinite(rank)
    if mask.sum() < 5:
        return np.nan, np.nan
    rho, p = spearmanr(feature[mask], rank[mask])
    return float(rho), float(p)


def wind_residualise(df: pd.DataFrame, feature_col: str) -> np.ndarray:
    """OLS residual of feature mean vs TWS + TWA means (per team×race)."""
    y = df[feature_col].values.astype(float)
    tws = df["wind_tws_mean"].values.astype(float)
    twa = df["wind_twa_mean"].values.astype(float)
    mask = np.isfinite(y) & np.isfinite(tws) & np.isfinite(twa)
    resid = np.full(len(df), np.nan)
    if mask.sum() < 5:
        return resid
    X = np.column_stack([np.ones(mask.sum()), tws[mask], twa[mask]])
    beta, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
    resid[mask] = y[mask] - X @ beta
    return resid


def compute_feature_correlations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rank = df["final_rank"].values
    rows = []
    for c in cols:
        col60 = f"mean_{c}"
        col0 = f"mean0_20_{c}"
        col20 = f"mean20_60_{c}"
        rho_raw, p_raw = _spearman(df[col60].values, rank)
        resid = wind_residualise(df, col60)
        rho_wind, p_wind = _spearman(resid, rank)
        rho_0_20, p_0_20 = _spearman(df[col0].values, rank)
        rho_20_60, p_20_60 = _spearman(df[col20].values, rank)
        rows.append(
            {
                "feature": c,
                "rho_raw": rho_raw,
                "p_raw": p_raw,
                "rho_wind_residualised": rho_wind,
                "p_wind_residualised": p_wind,
                "rho_0_20s": rho_0_20,
                "p_0_20s": p_0_20,
                "rho_20_60s": rho_20_60,
                "p_20_60s": p_20_60,
                "abs_rho_raw": abs(rho_raw) if np.isfinite(rho_raw) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_rho_raw", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def winner_vs_bottom(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    winners = df[df["tier"] == "winner"]
    bottoms = df[df["tier"] == "bottom"]
    rows = []
    for c in cols:
        col = f"mean_{c}"
        w_vals = winners[col].dropna().values
        b_vals = bottoms[col].dropna().values
        if len(w_vals) < 2 or len(b_vals) < 2:
            p_val = np.nan
            stat = np.nan
        else:
            stat, p_val = mannwhitneyu(w_vals, b_vals, alternative="two-sided")
        rows.append(
            {
                "feature": c,
                "winner_mean": float(np.nanmean(w_vals)) if len(w_vals) else np.nan,
                "bottom_mean": float(np.nanmean(b_vals)) if len(b_vals) else np.nan,
                "delta": float(np.nanmean(w_vals) - np.nanmean(b_vals)) if len(w_vals) and len(b_vals) else np.nan,
                "n_winner": len(w_vals),
                "n_bottom": len(b_vals),
                "mwu_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                "significant_005": bool(p_val < 0.05) if np.isfinite(p_val) else False,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)


def cluster_features(df: pd.DataFrame, cols: list[str], n_clusters: int = 4) -> dict[str, Any]:
    """Cluster features by cross-observation correlation of 60s means."""
    mat = df[[f"mean_{c}" for c in cols]].dropna()
    if len(mat) < 5 or len(cols) < 2:
        return {"n_clusters": 0, "clusters": {}}
    corr = mat.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - np.abs(corr)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters: dict[int, list[str]] = {}
    for feat, lab in zip(cols, labels):
        clusters.setdefault(int(lab), []).append(feat)
    return {
        "n_clusters": n_clusters,
        "clusters": {str(k): v for k, v in sorted(clusters.items())},
    }


def plot_correlations(corr_df: pd.DataFrame) -> go.Figure:
    sub = corr_df.dropna(subset=["rho_raw"]).sort_values("rho_raw", ascending=True)
    colors = ["#059669" if r < 0 else "#dc2626" for r in sub["rho_raw"]]
    fig = go.Figure(
        go.Bar(
            x=sub["rho_raw"],
            y=sub["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"ρ={r:.2f}" for r in sub["rho_raw"]],
            textposition="outside",
            name="ρ (60s mean vs rank)",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_layout(
        title="First-Minute Feature Correlations with Final Rank (Spearman ρ)",
        xaxis_title="Spearman ρ (negative = higher value → better finish)",
        yaxis_title="Feature",
        height=max(500, 28 * len(sub)),
        margin=dict(l=200),
    )
    return fig


def plot_timeline(
    records: list[TeamRaceRecord],
    raw_df: pd.DataFrame,
    top_features: list[str],
    cols: list[str],
) -> go.Figure:
    """Per-second mean for winner-tier vs bottom-tier for top features."""
    n_feat = len(top_features)
    fig = make_subplots(
        rows=n_feat,
        cols=1,
        subplot_titles=[f"{f} — winner (1–4) vs bottom (8+)" for f in top_features],
        vertical_spacing=0.08,
    )
    group_cols = ["venue", "race_label", "team"]
    record_lookup = {(r.venue, r.race_label, r.team): r.tier for r in records}

    for i, feat in enumerate(top_features, start=1):
        if feat not in cols:
            continue
        winner_sec: list[list[float]] = [[] for _ in range(WINDOW_S)]
        bottom_sec: list[list[float]] = [[] for _ in range(WINDOW_S)]

        for (venue, race, team), grp in raw_df.groupby(group_cols, sort=False):
            tier = record_lookup.get((venue, race, team))
            if tier not in ("winner", "bottom"):
                continue
            racing = grp[grp[COL_STATUS] == 2].sort_index().iloc[:WINDOW_S]
            if feat not in racing.columns:
                continue
            vals = racing[feat].values
            bucket = winner_sec if tier == "winner" else bottom_sec
            for s in range(min(len(vals), WINDOW_S)):
                if np.isfinite(vals[s]):
                    bucket[s].append(float(vals[s]))

        seconds = list(range(WINDOW_S))
        w_mean = [float(np.mean(bucket)) if bucket else np.nan for bucket in winner_sec]
        b_mean = [float(np.mean(bucket)) if bucket else np.nan for bucket in bottom_sec]

        fig.add_trace(
            go.Scatter(x=seconds, y=w_mean, mode="lines", name="Winner (1–4)", line=dict(color="#059669")),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=seconds, y=b_mean, mode="lines", name="Bottom (8+)", line=dict(color="#dc2626")),
            row=i,
            col=1,
        )
        fig.update_xaxes(title_text="Seconds after start gun", row=i, col=1)
        fig.update_yaxes(title_text=feat, row=i, col=1)

    fig.update_layout(
        title="First-Minute Timeline: Winner vs Bottom Tier",
        height=280 * n_feat,
        showlegend=True,
    )
    return fig


def evaluate_success(corr_df: pd.DataFrame, wvb_df: pd.DataFrame) -> dict[str, Any]:
    strong = corr_df[corr_df["abs_rho_raw"] > RHO_THRESHOLD]
    wind_hits = corr_df[
        (corr_df["rho_wind_residualised"].abs() > RHO_THRESHOLD)
        & (corr_df["p_wind_residualised"] < 0.05)
    ]
    controllable = [
        c
        for c in wind_hits["feature"]
        if c not in WIND_COLS and not c.startswith("LENGTH_RH")  # ride height is outcome-ish but keep
    ]
    # re-include ride height as controllable-ish foiling proxy
    controllable = [c for c in wind_hits["feature"] if c not in WIND_COLS]

    sub_diff = []
    for _, row in corr_df.iterrows():
        r0 = row.get("rho_0_20s", np.nan)
        r1 = row.get("rho_20_60s", np.nan)
        if np.isfinite(r0) and np.isfinite(r1) and abs(r0 - r1) > 0.1:
            sub_diff.append(row["feature"])

    sig_wvb = wvb_df[wvb_df["significant_005"] == True]  # noqa: E712

    criteria = {
        "n_features_rho_gt_03": {
            "pass": len(strong) >= 5,
            "value": int(len(strong)),
            "threshold": ">= 5",
        },
        "wind_residual_controllable_significant": {
            "pass": len(controllable) >= 3,
            "value": int(len(controllable)),
            "features": controllable[:10],
            "threshold": ">= 3",
        },
        "subwindow_temporal_dynamics": {
            "pass": len(sub_diff) >= 3,
            "value": int(len(sub_diff)),
            "features": sub_diff[:10],
            "threshold": ">= 3 features with |ρ_0-20 − ρ_20-60| > 0.1",
        },
        "winner_bottom_significant": {
            "pass": len(sig_wvb) >= 3,
            "value": int(len(sig_wvb)),
            "threshold": ">= 3 significant MWU deltas",
        },
    }
    return {"overall_pass": all(c["pass"] for c in criteria.values()), "criteria": criteria}


def main() -> None:
    print("[c8] loading data...", flush=True)
    raw = load_prepared_data()
    cols = available_columns(raw, BASELINE_22_COLS)
    print(f"[c8] {len(cols)} baseline features available", flush=True)

    records = extract_team_race_records(raw, cols)
    print(f"[c8] {len(records)} team×race first-minute windows", flush=True)
    if len(records) < 10:
        raise RuntimeError(f"Too few team×race records: {len(records)}")

    obs_df = records_to_frame(records, cols)
    corr_df = compute_feature_correlations(obs_df, cols)
    wvb_df = winner_vs_bottom(obs_df, cols)
    clusters = cluster_features(obs_df, cols)

    top5 = corr_df.head(5)[["feature", "rho_raw", "rho_wind_residualised"]].to_dict("records")
    wind_hits = corr_df[
        (corr_df["rho_wind_residualised"].abs() > RHO_THRESHOLD)
        & (corr_df["p_wind_residualised"] < 0.05)
    ]["feature"].tolist()

    success = evaluate_success(corr_df, wvb_df)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    corr_path = EXPORT_DIR / f"{EXPORT_PREFIX}_feature_correlations.csv"
    wvb_path = EXPORT_DIR / f"{EXPORT_PREFIX}_winner_vs_bottom.csv"
    json_path = EXPORT_DIR / f"{EXPORT_PREFIX}_results.json"
    corr_html = EXPORT_DIR / f"{EXPORT_PREFIX}_correlations.html"
    timeline_html = EXPORT_DIR / f"{EXPORT_PREFIX}_timeline.html"

    corr_df.to_csv(corr_path, index=False)
    wvb_df.to_csv(wvb_path, index=False)

    top3_feats = corr_df.head(3)["feature"].tolist()
    fig_corr = plot_correlations(corr_df)
    fig_timeline = plot_timeline(records, raw, top3_feats, cols)
    fig_corr.write_html(corr_html, include_plotlyjs="cdn")
    fig_timeline.write_html(timeline_html, include_plotlyjs="cdn")

    payload = {
        "experiment": "c8_first_minute_fingerprint",
        "n_team_races": len(records),
        "n_features": len(cols),
        "top5_by_abs_rho": top5,
        "wind_residualised_hits": wind_hits,
        "feature_clusters": clusters,
        "success": success,
        "winner_vs_bottom_top_deltas": wvb_df.nlargest(5, "delta", keep="all").to_dict("records"),
        "outputs": {
            "correlations_csv": str(corr_path),
            "winner_vs_bottom_csv": str(wvb_path),
            "results_json": str(json_path),
            "correlations_html": str(corr_html),
            "timeline_html": str(timeline_html),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[c8] top 5 features by |ρ|:", flush=True)
    for row in top5:
        print(f"  {row['feature']}: ρ={row['rho_raw']:.3f}  wind-resid ρ={row['rho_wind_residualised']:.3f}", flush=True)
    print(f"[c8] wind-residualised hits (|ρ|>{RHO_THRESHOLD}, p<0.05): {wind_hits}", flush=True)
    print(f"[c8] success overall={success['overall_pass']}", flush=True)
    print(f"[c8] wrote {corr_path}", flush=True)
    print(f"[c8] wrote {wvb_path}", flush=True)
    print(f"[c8] wrote {json_path}", flush=True)
    print(f"[c8] wrote {corr_html}", flush=True)
    print(f"[c8] wrote {timeline_html}", flush=True)


if __name__ == "__main__":
    main()
