#!/usr/bin/env python3
"""Experiment C2 — Regret residual analysis after removing leg-length effect."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import COL_LEG, COL_TWA, load_racing_boats
from dataExploration.next_experiments.exp_2b_vmg_decomposition import ols_regression

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
TWA_UPWIND_MAX = 90.0
LEG_KEYS = ["venue", "race_label", "team", "leg"]


def leg_twa_avg(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=[COL_TWA, COL_LEG]).copy()
    sub["twa_abs"] = sub[COL_TWA].abs()
    return (
        sub.groupby(["venue", "race_label", "team", COL_LEG], as_index=False)
        .agg(twa_avg_deg=("twa_abs", "mean"))
        .rename(columns={COL_LEG: "leg"})
    )


def load_leg_table() -> pd.DataFrame:
    path = EXPORT_DIR / "regret_decomposition.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp_2b or exp_c1 first")
    df = pd.read_csv(path)
    if "dirty_air_fraction" not in df.columns:
        df["dirty_air_fraction"] = df["dirty_air_seconds"] / df["leg_length_s"].clip(lower=1e-6)

    boats = load_racing_boats()
    twa = leg_twa_avg(boats)
    df = df.merge(twa, on=LEG_KEYS, how="left")
    df["leg_type"] = np.where(df["twa_avg_deg"] < TWA_UPWIND_MAX, "upwind", "downwind")
    return df


def fit_leg_length_baseline(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sub = df.dropna(subset=["ghost_regret_s", "leg_length_s"]).copy()
    y = sub["ghost_regret_s"].to_numpy(dtype=float)
    x = sub["leg_length_s"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    model = ols_regression(y, X, ["intercept", "leg_length_s"])

    sub = sub.copy()
    sub["regret_predicted"] = model["predictions"]
    sub["regret_residual"] = sub["ghost_regret_s"] - sub["regret_predicted"]
    return model, sub


def standardize_columns(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    stats = {}
    for c in cols:
        mu = float(out[c].mean())
        sd = float(out[c].std())
        if sd < 1e-12:
            sd = 1.0
        out[f"z_{c}"] = (out[c] - mu) / sd
        stats[c] = {"mean": mu, "std": sd}
    return out, stats


def fit_residual_model(df: pd.DataFrame) -> dict:
    features = ["dirty_air_fraction", "mean_flight_quality", "mean_vmg_residual"]
    sub = df.dropna(subset=["regret_residual"] + features).copy()
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    sub, z_stats = standardize_columns(sub, features)
    z_features = [f"z_{f}" for f in features]
    y = sub["regret_residual"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub[z_features].to_numpy(dtype=float)])
    model = ols_regression(y, X, ["intercept"] + features)
    model["standardization"] = z_stats
    model["note"] = "Coefficients are on standardized predictors (comparable beta weights)"

    fq = model["coefficients"].get("mean_flight_quality", {})
    dirty = model["coefficients"].get("dirty_air_fraction", {})
    model["success_criteria"] = {
        "r2_above_01": model["r2"] > 0.10,
        "flight_quality_significant": fq.get("significant_005", False),
        "dirty_air_positive_significant": (
            dirty.get("coefficient", 0) > 0 and dirty.get("significant_005", False)
        ),
    }
    model["overall_pass"] = model["success_criteria"]["r2_above_01"] and model["success_criteria"]["flight_quality_significant"]
    return model


def fit_leg_type_model(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["regret_residual", "leg_type"]).copy()
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    sub["is_downwind"] = (sub["leg_type"] == "downwind").astype(float)
    y = sub["regret_residual"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub["is_downwind"].to_numpy(dtype=float)])
    model = ols_regression(y, X, ["intercept", "is_downwind"])
    model["leg_type_counts"] = sub["leg_type"].value_counts().to_dict()
    model["mean_residual_by_leg_type"] = (
        sub.groupby("leg_type")["regret_residual"].mean().round(4).to_dict()
    )
    return model


def build_scatter_html(df: pd.DataFrame, regressions: dict, out_path: Path) -> None:
    predictors = [
        ("dirty_air_fraction", "Dirty Air Fraction"),
        ("mean_flight_quality", "Mean Flight Quality"),
        ("mean_vmg_residual", "Mean VMG Residual"),
        ("leg_type", "Leg Type"),
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[p[1] for p in predictors],
    )

    for idx, (col, title) in enumerate(predictors):
        row, col_idx = idx // 2 + 1, idx % 2 + 1
        sub = df.dropna(subset=["regret_residual", col]).copy()
        if col == "leg_type":
            for lt, color in [("upwind", "#1f77b4"), ("downwind", "#ff7f0e")]:
                g = sub[sub["leg_type"] == lt]
                fig.add_trace(
                    go.Scatter(
                        x=g["leg_type"],
                        y=g["regret_residual"],
                        mode="markers",
                        name=lt,
                        marker=dict(color=color, opacity=0.5),
                        showlegend=idx == 0,
                    ),
                    row=row,
                    col=col_idx,
                )
            continue

        x = sub[col].to_numpy(dtype=float)
        y = sub["regret_residual"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", marker=dict(opacity=0.4), name=title, showlegend=False),
            row=row,
            col=col_idx,
        )
        if len(x) >= 2:
            coef = regressions["coefficients"].get(col, {})
            beta = coef.get("coefficient", 0.0)
            z_stats = regressions.get("standardization", {}).get(col, {})
            mu = z_stats.get("mean", float(np.mean(x)))
            sd = z_stats.get("std", float(np.std(x)) or 1.0)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 50)
            intercept = regressions["coefficients"]["intercept"]["coefficient"]
            y_line = intercept + beta * ((x_line - mu) / sd)
            fig.add_trace(
                go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="red", width=2), showlegend=False),
                row=row,
                col=col_idx,
            )

    fig.update_layout(
        title="Regret Residual vs Predictors (after leg-length removal)",
        height=700,
        width=900,
    )
    fig.update_yaxes(title_text="regret_residual")
    fig.write_html(str(out_path))


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[c2] loading leg table...", flush=True)
    leg_df = load_leg_table()

    print("[c2] fitting regret ~ leg_length baseline...", flush=True)
    baseline, analysis_df = fit_leg_length_baseline(leg_df)

    print("[c2] fitting residual model (standardized predictors)...", flush=True)
    residual_model = fit_residual_model(analysis_df)

    print("[c2] fitting leg_type secondary model...", flush=True)
    leg_type_model = fit_leg_type_model(analysis_df)

    csv_path = EXPORT_DIR / "regret_residual_analysis.csv"
    export_cols = [
        *LEG_KEYS,
        "ghost_regret_s",
        "leg_length_s",
        "regret_predicted",
        "regret_residual",
        "dirty_air_fraction",
        "mean_flight_quality",
        "mean_vmg_residual",
        "leg_type",
        "twa_avg_deg",
    ]
    analysis_df[export_cols].to_csv(csv_path, index=False)

    scatter_path = EXPORT_DIR / "regret_residual_scatter.html"
    if "error" not in residual_model:
        build_scatter_html(analysis_df, residual_model, scatter_path)

    results = {
        "experiment": "exp_c2_regret_residual",
        "n_legs": int(len(leg_df)),
        "n_analysis": int(len(analysis_df)),
        "leg_length_baseline": baseline,
        "residual_regression": residual_model,
        "leg_type_regression": leg_type_model,
        "outputs": {
            "regret_residual_analysis_csv": str(csv_path),
            "regret_residual_results_json": str(EXPORT_DIR / "regret_residual_results.json"),
            "regret_residual_scatter_html": str(scatter_path),
        },
    }

    json_path = EXPORT_DIR / "regret_residual_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    baseline = res["leg_length_baseline"]
    residual = res["residual_regression"]
    leg_type = res["leg_type_regression"]

    print(f"\n[c2] BASELINE (regret ~ leg_length) R² = {baseline['r2']:.4f}  n = {baseline['n']}")
    if "error" in residual:
        print(f"[c2] residual model error: {residual['error']}")
    else:
        dirty = residual["coefficients"]["dirty_air_fraction"]
        fq = residual["coefficients"]["mean_flight_quality"]
        print(f"\n[c2] RESIDUAL MODEL R² = {residual['r2']:.4f}  n = {residual['n']}")
        print(f"  dirty_air_fraction: β={dirty['coefficient']:.4f} p={dirty['p_value']:.4g}")
        print(f"  mean_flight_quality: β={fq['coefficient']:.4f} p={fq['p_value']:.4g}")
        for name, c in residual["coefficients"].items():
            if name in ("dirty_air_fraction", "mean_flight_quality"):
                continue
            sig = "*" if c.get("significant_005") else ""
            print(f"  {name}: {c['coefficient']:.4f} (p={c['p_value']:.4g}){sig}")
        print(f"  variance_contribution_pct: {residual.get('variance_contribution_pct')}")
        print(f"  PASS: {residual.get('overall_pass')}")

    if "error" not in leg_type:
        print(f"\n[c2] LEG TYPE R² = {leg_type['r2']:.4f}")
        print(f"  mean_residual_by_leg_type: {leg_type.get('mean_residual_by_leg_type')}")

    print(f"\nOutputs: {res['outputs']}")
