#!/usr/bin/env python3
"""Experiment 2b — VMG residual per-leg features + ghost regret decomposition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.data_prep import (
    COL_DTB,
    COL_LEG,
    COL_TWA,
    COL_TWS,
    COL_VMG,
    load_racing_boats,
)
from dataExploration.next_experiments.shared.polar import (
    PolarTable,
    load_polar_table,
    test_polar_lookup,
)

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"
RESIDUAL_COL = "vmg_residual"
DIRTY_AIR_THRESHOLD_M = 60.0
LEG_KEYS = ["venue", "race_label", "team", "leg"]


def _with_leg_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "leg" not in out.columns and COL_LEG in out.columns:
        out = out.rename(columns={COL_LEG: "leg"})
    return out


def attach_residuals(df: pd.DataFrame, polar: PolarTable) -> pd.DataFrame:
    out = df.copy()
    valid = out[COL_VMG].notna() & out[COL_TWA].notna() & out[COL_TWS].notna()
    out["polar_expected_vmg"] = np.nan
    out[RESIDUAL_COL] = np.nan
    twa = out.loc[valid, COL_TWA].to_numpy()
    tws = out.loc[valid, COL_TWS].to_numpy()
    out.loc[valid, "polar_expected_vmg"] = [polar.lookup(t, w) for t, w in zip(twa, tws)]
    out.loc[valid, RESIDUAL_COL] = out.loc[valid, COL_VMG] - out.loc[valid, "polar_expected_vmg"]
    return out


def aggregate_residuals_per_leg(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_leg_col(df)
    sub = df.dropna(subset=[RESIDUAL_COL, "leg"])
    sub = sub[sub["leg"] > 0]
    agg = (
        sub.groupby(LEG_KEYS, as_index=False)[RESIDUAL_COL]
        .agg(
            mean_vmg_residual="mean",
            std_vmg_residual="std",
            frac_positive_residual=lambda s: float((s > 0).mean()),
            n_residual_rows="count",
        )
    )
    return agg


def load_dirty_air_per_leg(df: pd.DataFrame, exposure_path: Path) -> pd.DataFrame:
    if exposure_path.exists():
        exp = pd.read_csv(exposure_path)
        if "leg" in exp.columns:
            if "dirty_air_seconds" in exp.columns:
                return exp[LEG_KEYS + ["dirty_air_seconds"]].drop_duplicates(LEG_KEYS)
            if "seconds_in_dirty_air" in exp.columns:
                exp = exp.rename(columns={"seconds_in_dirty_air": "dirty_air_seconds"})
                return exp[LEG_KEYS + ["dirty_air_seconds"]].drop_duplicates(LEG_KEYS)

    if COL_DTB not in df.columns:
        return pd.DataFrame(columns=LEG_KEYS + ["dirty_air_seconds"])

    df = _with_leg_col(df)
    sub = df[df["leg"] > 0].copy()
    sub["in_dirty_air"] = sub[COL_DTB].notna() & (sub[COL_DTB] < DIRTY_AIR_THRESHOLD_M)
    return (
        sub.groupby(LEG_KEYS, as_index=False)["in_dirty_air"]
        .sum()
        .rename(columns={"in_dirty_air": "dirty_air_seconds"})
    )


def load_flight_quality_per_leg(fq_path: Path) -> pd.DataFrame:
    if not fq_path.exists():
        return pd.DataFrame(columns=LEG_KEYS + ["mean_flight_quality"])

    fq = pd.read_csv(fq_path)
    if "race_status" in fq.columns:
        fq = fq[fq["race_status"] == 2]
    fq = fq.dropna(subset=["flight_quality", "leg"])
    fq = fq[fq["leg"] > 0]
    return (
        fq.groupby(LEG_KEYS, as_index=False)["flight_quality"]
        .mean()
        .rename(columns={"flight_quality": "mean_flight_quality"})
    )


def build_leg_table(
    regret_path: Path,
    polar: PolarTable,
    racing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    regret = pd.read_csv(regret_path)
    regret = regret.rename(columns={"regret_s": "ghost_regret_s"})
    regret["leg_length_s"] = regret["actual_leg_s"]

    if racing_df is None:
        racing_df = load_racing_boats()
    racing_df = attach_residuals(racing_df, polar)

    residual_legs = aggregate_residuals_per_leg(racing_df)
    dirty_legs = load_dirty_air_per_leg(racing_df, EXPORT_DIR / "dirty_air_exposure.csv")
    fq_legs = load_flight_quality_per_leg(EXPORT_DIR / "flight_quality.csv")

    out = regret.merge(residual_legs, on=LEG_KEYS, how="left")
    out = out.merge(dirty_legs, on=LEG_KEYS, how="left")
    out = out.merge(fq_legs, on=LEG_KEYS, how="left")
    out["dirty_air_seconds"] = out["dirty_air_seconds"].fillna(0.0)
    out["dirty_air_fraction"] = out["dirty_air_seconds"] / out["leg_length_s"].clip(lower=1e-6)
    return out


def ols_regression(y: np.ndarray, X: np.ndarray, feature_names: list[str]) -> dict:
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    r2 = float(r2_score(y, y_pred))
    ss_res = float(np.sum((y - y_pred) ** 2))
    mse = ss_res / max(n - k, 1)
    try:
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), max(n - k, 1)))

    coefs = {}
    for i, name in enumerate(feature_names):
        coefs[name] = {
            "coefficient": float(beta[i]),
            "std_error": float(se[i]) if not np.isnan(se[i]) else None,
            "p_value": float(p_values[i]) if not np.isnan(p_values[i]) else None,
            "significant_005": bool(p_values[i] < 0.05) if not np.isnan(p_values[i]) else False,
        }

    # Relative contribution via squared standardized coefficients (excluding intercept)
    contrib = {}
    y_std = float(np.std(y))
    for i, name in enumerate(feature_names):
        if name == "intercept":
            continue
        x_std = float(np.std(X[:, i]))
        if x_std < 1e-12 or y_std < 1e-12:
            contrib[name] = 0.0
        else:
            contrib[name] = float((beta[i] * x_std / y_std) ** 2)
    total = sum(contrib.values())
    if total > 0:
        contrib = {k: round(100.0 * v / total, 1) for k, v in contrib.items()}

    return {
        "r2": r2,
        "n": int(n),
        "coefficients": coefs,
        "variance_contribution_pct": contrib,
        "predictions": y_pred.tolist(),
    }


def fit_regret_decomposition(leg_df: pd.DataFrame) -> dict:
    features = ["mean_vmg_residual", "dirty_air_seconds", "mean_flight_quality", "leg_length_s"]
    sub = leg_df.dropna(subset=["ghost_regret_s"] + features).copy()
    if len(sub) < 10:
        return {"error": "insufficient rows", "n": len(sub)}

    y = sub["ghost_regret_s"].to_numpy(dtype=float)
    X_raw = sub[features].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), X_raw])
    names = ["intercept"] + features
    model = ols_regression(y, X, names)

    dirty = model["coefficients"].get("dirty_air_seconds", {})
    success = {
        "r2_above_01": model["r2"] > 0.1,
        "dirty_air_positive": dirty.get("coefficient", 0) > 0,
        "dirty_air_significant": dirty.get("significant_005", False),
    }
    model["success_criteria"] = success
    model["overall_pass"] = all(success.values())
    model["feature_coverage"] = {
        f: int(sub[f].notna().sum()) for f in ["mean_vmg_residual", "dirty_air_seconds", "mean_flight_quality"]
    }
    model["notes"] = {
        "dirty_air_definition": f"PC_DTB_m < {DIRTY_AIR_THRESHOLD_M:.0f}m (1 Hz row count per leg)",
        "dirty_air_univariate_r": float(sub["dirty_air_seconds"].corr(sub["ghost_regret_s"])),
        "dirty_air_leg_length_r": float(sub["dirty_air_seconds"].corr(sub["leg_length_s"])),
        "multicollinearity_warning": (
            "dirty_air_seconds correlates strongly with leg_length_s; "
            "partial coefficient can differ from univariate sign"
        ),
    }
    return model


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    polar_test = test_polar_lookup()
    polar = load_polar_table()

    regret_path = EXPORT_DIR / "ghost_boat_regret.csv"
    if not regret_path.exists():
        raise FileNotFoundError(f"Missing {regret_path}; run exp_4_ghost_boat.py first")

    print("[exp2b] building per-leg feature table...", flush=True)
    leg_df = build_leg_table(regret_path, polar)
    csv_path = EXPORT_DIR / "regret_decomposition.csv"
    leg_df.to_csv(csv_path, index=False)
    print(f"[exp2b] leg table: {len(leg_df)} rows -> {csv_path}", flush=True)

    print("[exp2b] fitting regret decomposition...", flush=True)
    model = fit_regret_decomposition(leg_df)

    results = {
        "experiment": "exp2b_vmg_decomposition",
        "polar_test": polar_test,
        "n_legs": int(len(leg_df)),
        "n_regression": model.get("n", 0),
        "regression": model,
        "outputs": {
            "regret_decomposition_csv": str(csv_path),
            "regret_decomposition_json": str(EXPORT_DIR / "regret_decomposition_results.json"),
            "polar_module": str(Path(__file__).resolve().parent / "shared" / "polar.py"),
        },
    }

    json_path = EXPORT_DIR / "regret_decomposition_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    pt = res["polar_test"]
    reg = res["regression"]
    print(f"\n[exp2b] polar test: {'PASS' if pt['pass'] else 'FAIL'}")
    print(f"  polar_expected(45°, 20kn) = {pt.get('lookup_kmh'):.4f} km/h (csv={pt.get('csv_kmh')})")
    if "error" in reg:
        print(f"[exp2b] regression error: {reg['error']}")
    else:
        sc = reg.get("success_criteria", {})
        print(f"\n[exp2b] R² = {reg['r2']:.4f}  n = {reg['n']}")
        for name, c in reg.get("coefficients", {}).items():
            sig = "*" if c.get("significant_005") else ""
            print(f"  {name}: {c['coefficient']:.4f} (p={c['p_value']:.4g}){sig}")
        print(f"\n[exp2b] SUCCESS:")
        for k, v in sc.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
        print(f"  overall: {'PASS' if reg.get('overall_pass') else 'FAIL'}")
    print(f"\nOutputs: {res['outputs']}")
