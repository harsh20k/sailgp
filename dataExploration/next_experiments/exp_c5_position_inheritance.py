#!/usr/bin/env python3
"""Experiment C5 — Position inheritance: entering rank vs leg regret."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_LEG,
    COL_RANK,
    COL_TWA,
    load_racing_boats,
)
from dataExploration.next_experiments.exp_2b_vmg_decomposition import ols_regression

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
TWA_UPWIND_MAX = 90.0
DIRTY_AIR_THRESHOLD_M = 60.0
LEG_KEYS = ["venue", "race_label", "team", "leg"]


def _with_leg_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "leg" not in out.columns and COL_LEG in out.columns:
        out = out.rename(columns={COL_LEG: "leg"})
    return out


def leg_twa_avg(df: pd.DataFrame) -> pd.DataFrame:
    sub = _with_leg_col(df).dropna(subset=[COL_TWA, "leg"]).copy()
    sub["twa_abs"] = sub[COL_TWA].abs()
    return sub.groupby(LEG_KEYS, as_index=False).agg(twa_avg_deg=("twa_abs", "mean"))


def leg_end_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank at the last timestep of each leg (TRK_RACE_RANK_unk)."""
    sub = _with_leg_col(df).dropna(subset=[COL_RANK, "leg"]).copy()
    sub = sub[sub["leg"] > 0]
    return sub.groupby(LEG_KEYS, as_index=False).agg(leg_end_rank=(COL_RANK, "last"))


def dirty_air_per_leg(df: pd.DataFrame) -> pd.DataFrame:
    sub = _with_leg_col(df).dropna(subset=["leg"]).copy()
    if COL_DTB not in sub.columns:
        return pd.DataFrame(columns=LEG_KEYS + ["dirty_air_seconds", "leg_length_s"])

    sub["in_dirty_air"] = sub[COL_DTB].fillna(9999.0) < DIRTY_AIR_THRESHOLD_M
    return sub.groupby(LEG_KEYS, as_index=False).agg(
        dirty_air_seconds=("in_dirty_air", "sum"),
        leg_length_s=(COL_DTB, "count"),
    )


def load_regret() -> pd.DataFrame:
    path = EXPORT_DIR / "ghost_boat_regret.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run exp_4 first: {path}")
    return pd.read_csv(path)


def attach_entering_rank(regret: pd.DataFrame, boats: pd.DataFrame) -> pd.DataFrame:
    ranks = leg_end_ranks(boats)
    entering = ranks.rename(columns={"leg": "prev_leg", "leg_end_rank": "entering_rank"})
    entering["leg"] = entering["prev_leg"] + 1
    entering = entering.drop(columns=["prev_leg"])

    legs = regret.copy()
    legs = legs.merge(entering, on=LEG_KEYS, how="left")
    legs["leg_regret_s"] = legs["regret_s"]
    legs["leg_length_s"] = legs["actual_leg_s"]
    return legs


def enrich_legs(legs: pd.DataFrame, boats: pd.DataFrame) -> pd.DataFrame:
    legs = legs.merge(leg_twa_avg(boats), on=LEG_KEYS, how="left")
    legs["leg_type"] = np.where(legs["twa_avg_deg"] < TWA_UPWIND_MAX, "upwind", "downwind")

    decomp_path = EXPORT_DIR / "regret_decomposition.csv"
    if decomp_path.exists():
        da = pd.read_csv(decomp_path, usecols=LEG_KEYS + ["dirty_air_seconds", "leg_length_s"])
        da = da.rename(columns={"leg_length_s": "leg_length_telemetry"})
        legs = legs.merge(da, on=LEG_KEYS, how="left")
        legs["dirty_air_source"] = "regret_decomposition.csv"
    else:
        da = dirty_air_per_leg(boats)
        legs = legs.merge(da, on=LEG_KEYS, how="left")
        legs["dirty_air_source"] = f"computed_PC_DTB_lt_{int(DIRTY_AIR_THRESHOLD_M)}m"

    legs["dirty_air_exposure_fraction"] = (
        legs["dirty_air_seconds"] / legs["leg_length_s"].clip(lower=1e-6)
    )
    return legs


def regression_subset(legs: pd.DataFrame) -> pd.DataFrame:
    """Exclude first leg and rows with undefined entering rank."""
    sub = legs[legs["leg"] > 1].copy()
    sub = sub.dropna(subset=["leg_regret_s", "entering_rank", "leg_length_s", "leg_type"])
    sub = sub[sub["entering_rank"] > 0]
    return sub


def fit_position_model(sub: pd.DataFrame) -> dict:
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    sub = sub.copy()
    sub["is_downwind"] = (sub["leg_type"] == "downwind").astype(float)
    y = sub["leg_regret_s"].to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(len(sub)),
            sub["entering_rank"].to_numpy(dtype=float),
            sub["leg_length_s"].to_numpy(dtype=float),
            sub["is_downwind"].to_numpy(dtype=float),
        ]
    )
    model = ols_regression(
        y,
        X,
        ["intercept", "entering_rank", "leg_length_s", "is_downwind"],
    )
    er = model["coefficients"]["entering_rank"]
    model["rank_penalty_s_per_position"] = er["coefficient"]
    model["success_criteria"] = {
        "entering_rank_positive": er["coefficient"] > 0,
        "entering_rank_significant": er.get("significant_005", False),
        "effect_at_least_1s": er["coefficient"] >= 1.0,
    }
    model["overall_pass"] = all(model["success_criteria"].values())
    return model


def fit_by_leg_type(sub: pd.DataFrame) -> dict:
    out: dict = {}
    for leg_type in ("upwind", "downwind"):
        g = sub[sub["leg_type"] == leg_type].copy()
        if len(g) < 10:
            out[leg_type] = {"error": "insufficient rows", "n": len(g)}
            continue
        y = g["leg_regret_s"].to_numpy(dtype=float)
        X = np.column_stack(
            [
                np.ones(len(g)),
                g["entering_rank"].to_numpy(dtype=float),
                g["leg_length_s"].to_numpy(dtype=float),
            ]
        )
        model = ols_regression(y, X, ["intercept", "entering_rank", "leg_length_s"])
        er = model["coefficients"]["entering_rank"]
        model["rank_penalty_s_per_position"] = er["coefficient"]
        out[leg_type] = model

    up_coef = out.get("upwind", {}).get("rank_penalty_s_per_position")
    down_coef = out.get("downwind", {}).get("rank_penalty_s_per_position")
    if up_coef is not None and down_coef is not None:
        out["upwind_ge_downwind"] = bool(up_coef >= down_coef)
    return out


def crosscheck_dirty_air(sub: pd.DataFrame) -> dict:
    g = sub.dropna(subset=["entering_rank", "dirty_air_exposure_fraction"]).copy()
    if len(g) < 4:
        return {"error": "insufficient rows", "n": len(g)}
    rho, p = stats.spearmanr(g["entering_rank"], g["dirty_air_exposure_fraction"])
    return {
        "n": int(len(g)),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "interpretation": "positive rho => worse entering rank correlates with more dirty air",
    }


def build_scatter(sub: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure()
    colors = {"upwind": "#1f77b4", "downwind": "#ff7f0e"}
    for leg_type, color in colors.items():
        g = sub[sub["leg_type"] == leg_type]
        fig.add_trace(
            go.Scatter(
                x=g["entering_rank"],
                y=g["leg_regret_s"],
                mode="markers",
                name=leg_type,
                marker=dict(color=color, opacity=0.55),
                hovertemplate=(
                    "entering_rank=%{x}<br>regret=%{y:.1f}s<extra></extra>"
                ),
            )
        )
        if len(g) >= 2:
            x = g["entering_rank"].to_numpy(dtype=float)
            y = g["leg_regret_s"].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(x)), x])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x_line = np.linspace(float(x.min()), float(x.max()), 50)
            y_line = beta[0] + beta[1] * x_line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"{leg_type} fit",
                    line=dict(color=color, dash="dash"),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Position Inheritance: Entering Rank vs Leg Regret",
        template="plotly_dark",
        height=520,
        width=820,
        xaxis_title="Entering rank (end of prior leg)",
        yaxis_title="Leg regret (s)",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[c5] loading ghost_boat_regret.csv...", flush=True)
    regret = load_regret()

    print("[c5] loading telemetry for ranks + leg type...", flush=True)
    boats = load_racing_boats()

    legs = attach_entering_rank(regret, boats)
    legs = enrich_legs(legs, boats)
    sub = regression_subset(legs)

    print(f"[c5] regression rows (leg>1, valid entering_rank): {len(sub)}", flush=True)

    full_model = fit_position_model(sub)
    by_type = fit_by_leg_type(sub)
    dirty_check = crosscheck_dirty_air(sub)

    csv_path = EXPORT_DIR / "position_inheritance.csv"
    export_cols = [
        *LEG_KEYS,
        "entering_rank",
        "leg_regret_s",
        "leg_length_s",
        "leg_type",
        "dirty_air_exposure_fraction",
        "dirty_air_seconds",
        "twa_avg_deg",
    ]
    sub[export_cols].to_csv(csv_path, index=False)

    html_path = EXPORT_DIR / "position_inheritance.html"
    if len(sub) >= 4:
        build_scatter(sub, html_path)

    excluded = {
        "first_leg": int((legs["leg"] == 1).sum()),
        "undefined_entering_rank": int(legs["leg"].gt(1).sum() - len(sub)),
        "total_legs": int(len(legs)),
        "regression_legs": int(len(sub)),
    }

    results = {
        "experiment": "exp_c5_position_inheritance",
        "inputs": {
            "ghost_boat_regret_csv": str(EXPORT_DIR / "ghost_boat_regret.csv"),
            "ghost_boat_regret_by_leg_type_csv": str(
                EXPORT_DIR / "ghost_boat_regret_by_leg_type.csv"
            ),
        },
        "excluded": excluded,
        "full_regression": full_model,
        "by_leg_type": by_type,
        "dirty_air_crosscheck": dirty_check,
        "outputs": {
            "position_inheritance_csv": str(csv_path),
            "position_inheritance_results_json": str(
                EXPORT_DIR / "position_inheritance_results.json"
            ),
            "position_inheritance_html": str(html_path),
        },
    }

    json_path = EXPORT_DIR / "position_inheritance_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    full = res["full_regression"]
    by_type = res["by_leg_type"]
    dirty = res["dirty_air_crosscheck"]

    if "error" in full:
        print(f"[c5] FULL MODEL ERROR: {full['error']}")
    else:
        er = full["coefficients"]["entering_rank"]
        print(
            f"\n[c5] FULL MODEL R²={full['r2']:.4f}  n={full['n']}"
            f"\n  entering_rank: β={er['coefficient']:.3f} s/rank"
            f"  p={er['p_value']:.4g}"
            f"  PASS={full.get('overall_pass')}"
        )

    for lt in ("upwind", "downwind"):
        m = by_type.get(lt, {})
        if "error" in m:
            print(f"[c5] {lt}: {m['error']}")
        else:
            er = m["coefficients"]["entering_rank"]
            print(
                f"[c5] {lt}: entering_rank β={er['coefficient']:.3f} s/rank"
                f"  p={er['p_value']:.4g}  n={m['n']}"
            )

    if "error" not in dirty:
        print(
            f"\n[c5] dirty_air crosscheck: ρ={dirty['spearman_rho']:.3f}"
            f"  p={dirty['p_value']:.4g}  n={dirty['n']}"
        )

    print(f"\nOutputs: {res['outputs']}")
