#!/usr/bin/env python3
"""Experiment C1 — Dirty air fraction regression (decoupled from leg length)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.next_experiments.exp_2b_vmg_decomposition import ols_regression

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"


def load_leg_table() -> pd.DataFrame:
    path = EXPORT_DIR / "regret_decomposition.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp_2b_vmg_decomposition.py first")
    df = pd.read_csv(path)
    df["dirty_air_fraction"] = df["dirty_air_seconds"] / df["leg_length_s"].clip(lower=1e-6)
    df["regret_normalized"] = df["ghost_regret_s"] / df["leg_length_s"].clip(lower=1e-6)
    return df


def fit_primary(leg_df: pd.DataFrame) -> dict:
    features = ["mean_vmg_residual", "dirty_air_fraction", "mean_flight_quality", "leg_length_s"]
    sub = leg_df.dropna(subset=["ghost_regret_s"] + features).copy()
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    y = sub["ghost_regret_s"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub[features].to_numpy(dtype=float)])
    model = ols_regression(y, X, ["intercept"] + features)

    dirty = model["coefficients"].get("dirty_air_fraction", {})
    model["success_criteria"] = {
        "dirty_air_positive": dirty.get("coefficient", 0) > 0,
        "dirty_air_significant": dirty.get("significant_005", False),
    }
    model["overall_pass"] = all(model["success_criteria"].values())
    model["notes"] = {
        "dirty_air_fraction_leg_length_r": float(sub["dirty_air_fraction"].corr(sub["leg_length_s"])),
        "dirty_air_fraction_regret_r": float(sub["dirty_air_fraction"].corr(sub["ghost_regret_s"])),
    }
    return model


def fit_secondary(leg_df: pd.DataFrame) -> dict:
    features = ["dirty_air_fraction", "mean_flight_quality"]
    sub = leg_df.dropna(subset=["regret_normalized"] + features).copy()
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    y = sub["regret_normalized"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub[features].to_numpy(dtype=float)])
    model = ols_regression(y, X, ["intercept"] + features)
    model["success_criteria"] = {"r2_above_005": model["r2"] > 0.05}
    model["overall_pass"] = model["success_criteria"]["r2_above_005"]
    return model


def load_baseline_regression() -> dict | None:
    path = EXPORT_DIR / "regret_decomposition_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("regression")


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[c1] loading regret_decomposition.csv...", flush=True)
    leg_df = load_leg_table()
    leg_df.to_csv(EXPORT_DIR / "regret_decomposition.csv", index=False)

    print("[c1] fitting primary model (fraction)...", flush=True)
    primary = fit_primary(leg_df)
    print("[c1] fitting secondary model (normalized regret)...", flush=True)
    secondary = fit_secondary(leg_df)

    baseline = load_baseline_regression()
    comparison = None
    if baseline and "coefficients" in baseline:
        comparison = {
            "before_dirty_air_seconds": baseline["coefficients"].get("dirty_air_seconds"),
            "after_dirty_air_fraction": primary.get("coefficients", {}).get("dirty_air_fraction"),
            "before_r2": baseline.get("r2"),
            "after_r2": primary.get("r2"),
            "before_variance_contribution_pct": baseline.get("variance_contribution_pct"),
            "after_variance_contribution_pct": primary.get("variance_contribution_pct"),
        }

    results = {
        "experiment": "exp_c1_dirty_air_fraction",
        "n_legs": int(len(leg_df)),
        "n_primary": primary.get("n", 0),
        "n_secondary": secondary.get("n", 0),
        "primary_regression": primary,
        "secondary_regression": secondary,
        "comparison_vs_dirty_air_seconds": comparison,
        "outputs": {
            "regret_decomposition_fraction_json": str(EXPORT_DIR / "regret_decomposition_fraction.json"),
        },
    }

    out_path = EXPORT_DIR / "regret_decomposition_fraction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    primary = res["primary_regression"]
    secondary = res["secondary_regression"]

    if "error" in primary:
        print(f"[c1] primary error: {primary['error']}")
    else:
        dirty = primary["coefficients"]["dirty_air_fraction"]
        print(f"\n[c1] PRIMARY R² = {primary['r2']:.4f}  n = {primary['n']}")
        print(f"  dirty_air_fraction: β={dirty['coefficient']:.4f} p={dirty['p_value']:.4g}")
        for name, c in primary["coefficients"].items():
            if name == "dirty_air_fraction":
                continue
            sig = "*" if c.get("significant_005") else ""
            print(f"  {name}: {c['coefficient']:.4f} (p={c['p_value']:.4g}){sig}")
        print(f"  PASS: {primary.get('overall_pass')}")

    if "error" not in secondary:
        print(f"\n[c1] SECONDARY (normalized) R² = {secondary['r2']:.4f}  n = {secondary['n']}")
        print(f"  PASS: {secondary.get('overall_pass')}")

    print(f"\nOutputs: {res['outputs']}")
