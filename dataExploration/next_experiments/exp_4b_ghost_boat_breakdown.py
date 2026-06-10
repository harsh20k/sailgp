#!/usr/bin/env python3
"""Experiment #4b — Ghost Boat per-leg breakdown, overlays, and composite metric."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_LEG,
    COL_TWA,
    load_racing_boats,
)
from sailgp_analysis.config import DATA_ROOT

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
DIRTY_AIR_THRESHOLD_M = 60.0
TWA_UPWIND_MAX = 90.0


def ols_pvalues(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, standard errors, two-sided p-values (intercept + features)."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = max(n - k, 1)
    mse = float(np.sum(resid**2) / df)
    try:
        cov = mse * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = mse * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    p = 2.0 * stats.t.sf(np.abs(t), df)
    return beta, se, p


def load_regret() -> pd.DataFrame:
    path = EXPORT_DIR / "ghost_boat_regret.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run exp_4 first: {path}")
    return pd.read_csv(path)


def leg_twa_avg(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=[COL_TWA, COL_LEG]).copy()
    sub["twa_abs"] = sub[COL_TWA].abs()
    return (
        sub.groupby(["venue", "race_label", "team", COL_LEG], as_index=False)
        .agg(twa_avg_deg=("twa_abs", "mean"))
        .rename(columns={COL_LEG: "leg"})
    )


def dirty_air_per_leg(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds with PC_DTB_m < threshold per leg."""
    sub = df.dropna(subset=[COL_LEG]).copy()
    if COL_DTB not in sub.columns:
        sub["dirty_air_seconds"] = 0
        sub["leg_length_s"] = 1
        return (
            sub.groupby(["venue", "race_label", "team", COL_LEG], as_index=False)
            .agg(dirty_air_seconds=("dirty_air_seconds", "sum"), leg_length_s=("leg_length_s", "count"))
            .rename(columns={COL_LEG: "leg"})
        )

    sub["in_dirty_air"] = sub[COL_DTB].fillna(9999.0) < DIRTY_AIR_THRESHOLD_M
    return (
        sub.groupby(["venue", "race_label", "team", COL_LEG], as_index=False)
        .agg(
            dirty_air_seconds=("in_dirty_air", "sum"),
            leg_length_s=(COL_DTB, "count"),
        )
        .rename(columns={COL_LEG: "leg"})
    )


def load_dirty_air_exposure() -> pd.DataFrame | None:
    path = EXPORT_DIR / "dirty_air_exposure.csv"
    if not path.exists():
        return None
    exp = pd.read_csv(path)
    if "dirty_air_seconds" in exp.columns and "leg" in exp.columns:
        return exp
    if "seconds_in_dirty_air" in exp.columns:
        exp = exp.rename(columns={"seconds_in_dirty_air": "dirty_air_seconds"})
    return exp if "dirty_air_seconds" in exp.columns else None


def flight_quality_per_leg() -> pd.DataFrame | None:
    path = EXPORT_DIR / "flight_quality.csv"
    if not path.exists():
        return None
    fq = pd.read_csv(path, usecols=["venue", "race_label", "team", "leg", "flight_quality"])
    fq = fq.dropna(subset=["leg"])
    fq = fq[fq["leg"] > 0]
    return (
        fq.groupby(["venue", "race_label", "team", "leg"], as_index=False)
        .agg(mean_flight_quality=("flight_quality", "mean"))
    )


def enrich_legs(regret: pd.DataFrame, boats: pd.DataFrame) -> pd.DataFrame:
    legs = regret.copy()
    legs = legs.merge(leg_twa_avg(boats), on=["venue", "race_label", "team", "leg"], how="left")
    legs["leg_type"] = np.where(legs["twa_avg_deg"] < TWA_UPWIND_MAX, "upwind", "downwind")

    exposure = load_dirty_air_exposure()
    if exposure is not None and "leg" in exposure.columns:
        legs = legs.merge(
            exposure[["venue", "race_label", "team", "leg", "dirty_air_seconds"]],
            on=["venue", "race_label", "team", "leg"],
            how="left",
        )
        legs["dirty_air_source"] = "dirty_air_exposure.csv"
    else:
        da = dirty_air_per_leg(boats)
        legs = legs.merge(da, on=["venue", "race_label", "team", "leg"], how="left")
        legs["dirty_air_source"] = f"computed_PC_DTB_lt_{int(DIRTY_AIR_THRESHOLD_M)}m"

    legs["leg_length_s"] = legs["actual_leg_s"]

    fq = flight_quality_per_leg()
    if fq is not None:
        legs = legs.merge(fq, on=["venue", "race_label", "team", "leg"], how="left")
        legs["flight_quality_source"] = "flight_quality.csv"
    else:
        legs["mean_flight_quality"] = np.nan
        legs["flight_quality_source"] = "unavailable"

    legs["performance_index"] = 1.0 - (legs["regret_s"] / legs["ghost_leg_s"].clip(lower=1e-6))
    legs["performance_index"] = legs["performance_index"].clip(0.0, 1.0)
    return legs


def leg_type_summary(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (team, leg_type), g in legs.groupby(["team", "leg_type"], sort=True):
        rows.append(
            {
                "team": team,
                "leg_type": leg_type,
                "n_legs": len(g),
                "mean_regret_s": float(g["regret_s"].mean()),
                "std_regret_s": float(g["regret_s"].std(ddof=1)) if len(g) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_waterfall(legs: pd.DataFrame, out_path: Path) -> None:
    races = legs[["venue", "race_label"]].drop_duplicates().sort_values(["venue", "race_label"])
    n = len(races)
    if n == 0:
        go.Figure().write_html(out_path, include_plotlyjs="cdn")
        return

    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    titles = [f"{r.venue} {r.race_label}" for r in races.itertuples(index=False)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, race in enumerate(races.itertuples(index=False)):
        r = idx // cols + 1
        c = idx % cols + 1
        sub = legs[(legs["venue"] == race.venue) & (legs["race_label"] == race.race_label)].copy()
        sub = sub.sort_values(["team", "leg"])
        sub["cum_regret_s"] = sub.groupby("team")["regret_s"].cumsum()

        for team, g in sub.groupby("team", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g["leg"],
                    y=g["cum_regret_s"],
                    mode="lines+markers",
                    name=team,
                    legendgroup=team,
                    showlegend=(idx == 0),
                    hovertemplate=(
                        f"team={team}<br>leg=%{{x}}<br>"
                        "cum_regret=%{y:.1f}s<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        height=max(400, 320 * rows),
        title="Ghost Boat Cumulative Regret by Leg (waterfall)",
        template="plotly_dark",
    )
    fig.update_xaxes(title_text="Leg")
    fig.update_yaxes(title_text="Cumulative regret (s)")
    fig.write_html(out_path, include_plotlyjs="cdn")


def dirty_air_regression(legs: pd.DataFrame) -> dict:
    sub = legs.dropna(subset=["regret_s", "dirty_air_seconds", "leg_length_s"]).copy()
    sub = sub[sub["leg_length_s"] > 0]
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    y = sub["regret_s"].to_numpy(dtype=float)
    x1 = sub["dirty_air_seconds"].to_numpy(dtype=float)
    x2 = sub["leg_length_s"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), x1, x2])
    beta, se, p = ols_pvalues(X, y)

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": "regret_s ~ dirty_air_seconds + leg_length_s",
        "n_obs": int(len(sub)),
        "dirty_air_source": str(sub["dirty_air_source"].iloc[0]),
        "intercept": {"coef": float(beta[0]), "se": float(se[0]), "p": float(p[0])},
        "dirty_air_seconds": {
            "coef": float(beta[1]),
            "se": float(se[1]),
            "p": float(p[1]),
            "significant_positive": bool(beta[1] > 0 and p[1] < 0.05),
        },
        "leg_length_s": {"coef": float(beta[2]), "se": float(se[2]), "p": float(p[2])},
        "r2": float(r2),
    }


def leg_vs_cumulative_rank_diff(legs: pd.DataFrame) -> dict:
    """Find races where per-leg regret ranking differs from cumulative total ranking."""
    races_with_diff = []
    for (venue, race), g in legs.groupby(["venue", "race_label"], sort=False):
        totals = (
            g.groupby("team", as_index=False)
            .agg(total_regret_s=("regret_s", "sum"))
            .sort_values("total_regret_s")
        )
        totals["cumulative_rank"] = range(1, len(totals) + 1)
        cum_order = totals.set_index("team")["cumulative_rank"]

        leg_diffs = []
        for leg, lg in g.groupby("leg", sort=True):
            lr = lg.sort_values("regret_s")[["team", "regret_s"]].copy()
            lr["leg_rank"] = range(1, len(lr) + 1)
            merged = lr.set_index("team")["leg_rank"].to_frame().join(cum_order, how="inner")
            if merged.empty:
                continue
            diff_teams = merged[merged["leg_rank"] != merged["cumulative_rank"]]
            if len(diff_teams):
                leg_diffs.append(
                    {
                        "leg": int(leg),
                        "n_teams_misranked": int(len(diff_teams)),
                        "examples": diff_teams.head(3).reset_index().to_dict(orient="records"),
                    }
                )
        if leg_diffs:
            races_with_diff.append(
                {"venue": venue, "race_label": race, "legs_with_rank_diff": leg_diffs}
            )

    return {
        "n_races_with_diff": len(races_with_diff),
        "pass": len(races_with_diff) >= 1,
        "races": races_with_diff[:5],
    }


def evaluate_criteria(legs: pd.DataFrame, reg: dict, rank_diff: dict) -> dict:
    up = legs.loc[legs["leg_type"] == "upwind", "regret_s"].dropna()
    down = legs.loc[legs["leg_type"] == "downwind", "regret_s"].dropna()
    if len(up) >= 2 and len(down) >= 2:
        _, mwu_p = stats.mannwhitneyu(up, down, alternative="two-sided")
    else:
        mwu_p = 1.0

    da = reg.get("dirty_air_seconds", {})
    criteria = {
        "upwind_vs_downwind_mwu": {
            "value": float(mwu_p),
            "threshold": 0.05,
            "pass": float(mwu_p) < 0.05,
            "n_upwind": int(len(up)),
            "n_downwind": int(len(down)),
            "mean_upwind_regret_s": float(up.mean()) if len(up) else np.nan,
            "mean_downwind_regret_s": float(down.mean()) if len(down) else np.nan,
        },
        "dirty_air_coefficient": {
            "value": da.get("coef", np.nan),
            "p_value": da.get("p", np.nan),
            "threshold": "coef>0, p<0.05",
            "pass": bool(da.get("significant_positive", False)),
        },
        "leg_rank_differs_from_cumulative": {
            "value": rank_diff.get("n_races_with_diff", 0),
            "threshold": ">=1 race",
            "pass": bool(rank_diff.get("pass", False)),
        },
    }
    criteria["overall_pass"] = all(c["pass"] for c in criteria.values() if isinstance(c, dict))
    return criteria


def composite_index_table(legs: pd.DataFrame) -> pd.DataFrame:
    leg_cols = [
        "venue",
        "race_label",
        "team",
        "leg",
        "leg_type",
        "regret_s",
        "ghost_leg_s",
        "performance_index",
        "dirty_air_seconds",
        "mean_flight_quality",
    ]
    leg_out = legs[[c for c in leg_cols if c in legs.columns]].copy()

    race_agg = (
        legs.groupby(["venue", "race_label", "team"], as_index=False)
        .agg(
            mean_performance_index=("performance_index", "mean"),
            total_regret_s=("regret_s", "sum"),
            total_ghost_s=("ghost_leg_s", "sum"),
            n_legs=("leg", "count"),
        )
    )
    race_agg["race_performance_index"] = 1.0 - (
        race_agg["total_regret_s"] / race_agg["total_ghost_s"].clip(lower=1e-6)
    )
    race_agg["race_performance_index"] = race_agg["race_performance_index"].clip(0.0, 1.0)
    return leg_out, race_agg


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    regret = load_regret()
    boats = load_racing_boats(DATA_ROOT)
    legs = enrich_legs(regret, boats)

    # 1. Per-leg type split
    type_summary = leg_type_summary(legs)
    type_path = EXPORT_DIR / "ghost_boat_regret_by_leg_type.csv"
    type_summary.to_csv(type_path, index=False)

    # 2. Waterfall
    waterfall_path = EXPORT_DIR / "ghost_boat_leg_waterfall.html"
    build_waterfall(legs, waterfall_path)

    # 3. Dirty air regression
    reg = dirty_air_regression(legs)
    reg_path = EXPORT_DIR / "ghost_boat_dirty_air_regression.json"
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)

    # 4. Rank diff criterion
    rank_diff = leg_vs_cumulative_rank_diff(legs)

    # 5. Composite index
    leg_index, race_index = composite_index_table(legs)
    composite_path = EXPORT_DIR / "ghost_boat_composite_index.csv"
    leg_index.to_csv(composite_path, index=False)
    race_index.to_csv(EXPORT_DIR / "ghost_boat_composite_index_race.csv", index=False)

    criteria = evaluate_criteria(legs, reg, rank_diff)

    fq_overlay = {}
    fq_sub = legs.dropna(subset=["mean_flight_quality", "regret_s"])
    if len(fq_sub) >= 10:
        rho, fq_p = stats.spearmanr(fq_sub["mean_flight_quality"], fq_sub["regret_s"])
        fq_overlay = {
            "n_legs": int(len(fq_sub)),
            "spearman_rho_regret_vs_flight_quality": float(rho),
            "p_value": float(fq_p),
            "source": str(legs["flight_quality_source"].iloc[0]),
        }

    results = {
        "experiment": "ghost_boat_breakdown",
        "n_legs": int(len(legs)),
        "n_races": int(legs[["venue", "race_label"]].drop_duplicates().shape[0]),
        "success_criteria": criteria,
        "overall_pass": bool(criteria["overall_pass"]),
        "dirty_air_regression": reg,
        "leg_rank_analysis": rank_diff,
        "flight_quality_overlay": fq_overlay,
        "leg_type_summary": type_summary.to_dict(orient="records"),
        "performance_index_summary": {
            "mean_leg_index": float(legs["performance_index"].mean()),
            "mean_race_index": float(race_index["race_performance_index"].mean()),
        },
        "outputs": {
            "regret_by_leg_type_csv": str(type_path),
            "leg_waterfall_html": str(waterfall_path),
            "dirty_air_regression_json": str(reg_path),
            "composite_index_csv": str(composite_path),
        },
    }

    with open(EXPORT_DIR / "ghost_boat_breakdown_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    print(f"Overall: {'PASS' if res['overall_pass'] else 'FAIL'}")
    for name, c in res["success_criteria"].items():
        if not isinstance(c, dict):
            continue
        print(f"  {name}: {c.get('value')} -> {'PASS' if c['pass'] else 'FAIL'}")
    print("Outputs:", res["outputs"])
