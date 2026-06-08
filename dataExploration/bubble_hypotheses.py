"""Statistical hypothesis tests for tactical bubble features."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_RANK,
    COL_SPEED,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    foiling_label,
    load_racing_boats,
    rank_delta_label,
)
from dataExploration.lstm_experiments.shared.fleet import (
    BUBBLE_FEATURE_COLS,
    COL_N_AHEAD,
    COL_N_BOATS,
    COL_N_FOILING,
    DEFAULT_DIRTY_AIR_CONE_DEG,
    DEFAULT_DIRTY_AIR_DIST_M,
    add_bubble_features,
    build_fleet_snapshots,
    dirty_air_flag,
    ego_relative,
)

EXPORT_DIR = Path(__file__).resolve().parent / "exported"
HORIZON_S = 60
BUBBLE_RADIUS_M = 200.0


@dataclass
class HypothesisResult:
    id: str
    name: str
    description: str
    test: str
    p_value: float
    effect_size: float
    effect_label: str
    n_samples: int
    verdict: str
    enter_model: bool
    details: dict


def build_analysis_frame(venue: str | None = None) -> pd.DataFrame:
    df = load_racing_boats()
    if venue:
        df = df[df["venue"] == venue]
    df = add_bubble_features(df, radius_m=BUBBLE_RADIUS_M, validate_alignment=True)

    rows: list[dict] = []
    snapshots = build_fleet_snapshots(df)

    for (v, race, team), gdf in df.groupby(["venue", "race_label", "team"], sort=False):
        gdf = gdf.sort_index()
        ranks = gdf[COL_RANK].to_numpy(dtype=float)
        for i, (ts, row) in enumerate(gdf.iterrows()):
            future_i = i + HORIZON_S
            if future_i >= len(gdf):
                continue
            rank_now = ranks[i]
            rank_future = ranks[future_i]
            if np.isnan(rank_now) or np.isnan(rank_future):
                continue

            label = rank_delta_label(rank_now, rank_future)
            rank_loss = label == 2
            rank_gain = label == 0

            snap = snapshots.get((v, race, ts))
            in_dirty_air = False
            if snap is not None and team in snap.index:
                others = snap.drop(index=team, errors="ignore")
                neighbours = ego_relative(row, others)
                in_dirty_air = dirty_air_flag(
                    neighbours,
                    dist_m=DEFAULT_DIRTY_AIR_DIST_M,
                    cone_deg=DEFAULT_DIRTY_AIR_CONE_DEG,
                )

            twa = float(row.get(COL_TWA, np.nan))
            tws = float(row.get(COL_TWS, np.nan))
            vmg = float(row.get(COL_VMG, np.nan))
            speed = float(row.get(COL_SPEED, np.nan))
            vmg_ratio = vmg / tws if tws > 1.0 and not np.isnan(vmg) else np.nan

            ego_foiling = float(foiling_label(row))
            n_foiling = float(row.get(COL_N_FOILING, 0))
            foil_deficit = max(0.0, n_foiling - ego_foiling) if ego_foiling else n_foiling

            rows.append(
                {
                    "venue": v,
                    "race_label": race,
                    "team": team,
                    "ts": ts,
                    "rank_now": rank_now,
                    "rank_future": rank_future,
                    "rank_delta": label,
                    "rank_loss": rank_loss,
                    "rank_gain": rank_gain,
                    "in_dirty_air": in_dirty_air,
                    "speed": speed,
                    "vmg_ratio": vmg_ratio,
                    "twa": twa,
                    "n_boats_ahead": float(row.get(COL_N_AHEAD, 0)),
                    "n_boats_in_bubble": float(row.get(COL_N_BOATS, 0)),
                    "clear_air": float(row.get(COL_N_BOATS, 0)) == 0,
                    "foil_deficit": foil_deficit,
                    **{c: float(row.get(c, 0)) for c in BUBBLE_FEATURE_COLS},
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Polar-bin VMG residual (global lookup)
    valid = frame.dropna(subset=["twa", "vmg_ratio"])
    if len(valid) > 0:
        bins = np.floor(valid["twa"] / 10.0) * 10.0
        polar_table = valid.groupby(bins)["vmg_ratio"].mean().to_dict()
        global_mean = float(valid["vmg_ratio"].mean())

        def _residual(row: pd.Series) -> float:
            if np.isnan(row["twa"]) or np.isnan(row["vmg_ratio"]):
                return np.nan
            b = float(np.floor(row["twa"] / 10.0) * 10.0)
            expected = polar_table.get(b, global_mean)
            return row["vmg_ratio"] - expected

        frame["vmg_residual"] = frame.apply(_residual, axis=1)
        frame["speed_residual"] = frame["speed"] - frame["speed"].groupby(
            [frame["venue"], frame["race_label"]]
        ).transform("mean")
    return frame


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def test_h1_dirty_air(frame: pd.DataFrame) -> HypothesisResult:
    """Dirty air → lower VMG/speed residual."""
    sub = frame.dropna(subset=["vmg_residual"])
    clean = sub[~sub["in_dirty_air"]]["vmg_residual"].to_numpy()
    dirty = sub[sub["in_dirty_air"]]["vmg_residual"].to_numpy()
    if len(clean) < 10 or len(dirty) < 10:
        return HypothesisResult(
            id="H1",
            name="Dirty air penalty",
            description="Rival within 80m forward cone → lower VMG residual vs polar",
            test="Mann-Whitney U",
            p_value=1.0,
            effect_size=0.0,
            effect_label="Cohen's d",
            n_samples=len(sub),
            verdict="Insufficient samples",
            enter_model=False,
            details={"n_clean": len(clean), "n_dirty": len(dirty)},
        )
    stat, p = stats.mannwhitneyu(dirty, clean, alternative="less")
    d = _cohens_d(dirty, clean)
    significant = p < 0.05 and d < -0.05
    return HypothesisResult(
        id="H1",
        name="Dirty air penalty",
        description="Rival within 80m forward cone → lower VMG residual vs polar",
        test="Mann-Whitney U",
        p_value=float(p),
        effect_size=d,
        effect_label="Cohen's d (dirty - clean)",
        n_samples=len(sub),
        verdict="Supported" if significant else "Weak / not supported",
        enter_model=significant,
        details={
            "n_clean": len(clean),
            "n_dirty": len(dirty),
            "mean_clean": float(np.mean(clean)),
            "mean_dirty": float(np.mean(dirty)),
            "mwu_stat": float(stat),
        },
    )


def test_h2_traffic_rank(frame: pd.DataFrame) -> HypothesisResult:
    """More boats ahead → higher rank-loss probability."""
    sub = frame.dropna(subset=["n_boats_ahead", "rank_loss"])
    if len(sub) < 50:
        return HypothesisResult(
            id="H2",
            name="Traffic → rank loss",
            description="Higher n_boats_ahead_180deg predicts rank loss in next 60s",
            test="Logistic regression",
            p_value=1.0,
            effect_size=0.0,
            effect_label="AUC",
            n_samples=len(sub),
            verdict="Insufficient samples",
            enter_model=False,
            details={},
        )
    X = sub[["n_boats_ahead"]].to_numpy()
    y = sub["rank_loss"].astype(int).to_numpy()
    try:
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X, y)
        prob = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else 0.5
        coef = float(model.coef_[0, 0])
        # Wald-style p via correlation proxy
        r, p_corr = stats.pointbiserialr(y, sub["n_boats_ahead"])
    except Exception as e:
        return HypothesisResult(
            id="H2",
            name="Traffic → rank loss",
            description="Higher n_boats_ahead_180deg predicts rank loss in next 60s",
            test="Logistic regression",
            p_value=1.0,
            effect_size=0.0,
            effect_label="AUC",
            n_samples=len(sub),
            verdict=f"Failed: {e}",
            enter_model=False,
            details={},
        )
    significant = p_corr < 0.05 and coef > 0 and auc > 0.52
    return HypothesisResult(
        id="H2",
        name="Traffic → rank loss",
        description="Higher n_boats_ahead_180deg predicts rank loss in next 60s",
        test="Logistic regression + point-biserial",
        p_value=float(p_corr),
        effect_size=auc,
        effect_label="AUC / coef",
        n_samples=len(sub),
        verdict="Supported" if significant else "Weak / not supported",
        enter_model=significant or p_corr < 0.1,
        details={"coef_n_ahead": coef, "auc": auc, "point_biserial_r": float(r)},
    )


def test_h3_clear_air(frame: pd.DataFrame) -> HypothesisResult:
    """Clear air (no boats in 200m) → rank gain/hold vs loss."""
    sub = frame.copy()
    clear = sub[sub["clear_air"]]
    traffic = sub[~sub["clear_air"]]
    if len(clear) < 10 or len(traffic) < 10:
        return HypothesisResult(
            id="H3",
            name="Clear-air leader",
            description="Zero boats in 200m bubble correlates with rank gain/hold",
            test="Chi-square",
            p_value=1.0,
            effect_size=0.0,
            effect_label="Cramér's V",
            n_samples=len(sub),
            verdict="Insufficient samples",
            enter_model=False,
            details={},
        )
    clear_gain_rate = float((clear["rank_gain"] | ~clear["rank_loss"]).mean())
    traffic_gain_rate = float((traffic["rank_gain"] | ~traffic["rank_loss"]).mean())
    contingency = pd.crosstab(sub["clear_air"], sub["rank_loss"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    v = math.sqrt(chi2 / (n * min(contingency.shape[0] - 1, contingency.shape[1] - 1))) if n > 0 else 0.0
    significant = p < 0.05 and clear_gain_rate > traffic_gain_rate
    return HypothesisResult(
        id="H3",
        name="Clear-air leader",
        description="Zero boats in 200m bubble correlates with rank gain/hold",
        test="Chi-square",
        p_value=float(p),
        effect_size=float(v),
        effect_label="Cramér's V",
        n_samples=len(sub),
        verdict="Supported" if significant else "Weak / not supported",
        enter_model=significant or p < 0.1,
        details={
            "clear_gain_hold_rate": clear_gain_rate,
            "traffic_gain_hold_rate": traffic_gain_rate,
            "n_clear": len(clear),
            "n_traffic": len(traffic),
        },
    )


def test_h4_foil_advantage(frame: pd.DataFrame) -> HypothesisResult:
    """Foiling when rivals aren't → rank gain."""
    sub = frame.dropna(subset=["foil_deficit"])
    sub = sub[sub["foil_deficit"] >= 0]
    high_deficit = sub[sub["foil_deficit"] >= 2]["rank_loss"].astype(int)
    low_deficit = sub[sub["foil_deficit"] < 1]["rank_loss"].astype(int)
    if len(high_deficit) < 10 or len(low_deficit) < 10:
        return HypothesisResult(
            id="H4",
            name="Foiling advantage",
            description="More rivals foiling in bubble while ego is not → rank loss",
            test="Mann-Whitney U",
            p_value=1.0,
            effect_size=0.0,
            effect_label="Cohen's d",
            n_samples=len(sub),
            verdict="Insufficient samples",
            enter_model=False,
            details={},
        )
    stat, p = stats.mannwhitneyu(high_deficit, low_deficit, alternative="greater")
    d = _cohens_d(high_deficit.to_numpy(), low_deficit.to_numpy())
    significant = p < 0.05 and d > 0.05
    return HypothesisResult(
        id="H4",
        name="Foiling advantage",
        description="More rivals foiling in bubble while ego is not → rank loss",
        test="Mann-Whitney U on rank_loss",
        p_value=float(p),
        effect_size=d,
        effect_label="Cohen's d",
        n_samples=len(sub),
        verdict="Supported" if significant else "Weak / not supported",
        enter_model=significant or COL_N_FOILING in BUBBLE_FEATURE_COLS,
        details={
            "loss_rate_high_deficit": float(high_deficit.mean()),
            "loss_rate_low_deficit": float(low_deficit.mean()),
            "n_high": len(high_deficit),
            "n_low": len(low_deficit),
        },
    )




def run_all_hypotheses(venue: str | None = "Bermuda") -> list[HypothesisResult]:
    print(f"Building analysis frame (venue={venue})...", flush=True)
    frame = build_analysis_frame(venue=venue)
    print(f"  {len(frame):,} labelled timesteps", flush=True)
    results = [
        test_h1_dirty_air(frame),
        test_h2_traffic_rank(frame),
        test_h3_clear_air(frame),
        test_h4_foil_advantage(frame),
    ]
    return results


def export_results(results: list[HypothesisResult], frame_stats: dict | None = None) -> tuple[Path, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPORT_DIR / "bubble_hypotheses.json"
    html_path = EXPORT_DIR / "bubble_hypotheses.html"

    payload = {
        "hypotheses": [asdict(r) for r in results],
        "frame_stats": frame_stats or {},
        "features_for_model": [c for c in BUBBLE_FEATURE_COLS],
        "go_count": sum(1 for r in results if r.enter_model),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[r.name for r in results],
    )
    for i, r in enumerate(results):
        row, col = (i // 2) + 1, (i % 2) + 1
        fig.add_trace(
            go.Bar(
                x=["-log10(p)", "|effect|"],
                y=[-np.log10(max(r.p_value, 1e-300)), abs(r.effect_size)],
                name=r.id,
                marker_color="#2563eb" if r.enter_model else "#94a3b8",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        title=f"Tactical Bubble — Hypothesis Tests ({BUBBLE_RADIUS_M:.0f} m asymmetric, Bermuda)",
        height=600,
    )
    fig.write_html(str(html_path))

    # Separate summary table HTML snippet appended
    table_html = "<h2>Summary</h2><table border='1'><tr><th>ID</th><th>Hypothesis</th><th>Verdict</th><th>p</th><th>Effect</th></tr>"
    for r in results:
        table_html += f"<tr><td>{r.id}</td><td>{r.name}</td><td>{r.verdict}</td><td>{r.p_value:.2e}</td><td>{r.effect_size:.3f}</td></tr>"
    table_html += "</table>"
    with open(html_path, "a") as f:
        f.write(table_html)
    return json_path, html_path


def main():
    results = run_all_hypotheses(venue="Bermuda")
    for r in results:
        print(f"{r.id} {r.name}: {r.verdict} (p={r.p_value:.2e}, effect={r.effect_size:.3f})")
    json_path, html_path = export_results(results)
    print(f"Wrote {json_path} and {html_path}")


if __name__ == "__main__":
    main()
