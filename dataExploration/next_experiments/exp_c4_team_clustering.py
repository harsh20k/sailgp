#!/usr/bin/env python3
"""Experiment C4 — Team style clustering (broadcast-ready archetypes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPORT_DIR = Path(__file__).resolve().parents[1] / "exported"

FEATURES = [
    "mean_upwind_regret",
    "std_upwind_regret",
    "mean_downwind_regret",
    "std_downwind_regret",
    "upwind_downwind_ratio",
    "dirty_air_exposure_fraction",
    "mean_flight_quality",
    "tack_recovery_time",
    "performance_index",
]

FEATURE_LABELS = {
    "mean_upwind_regret": "Upwind regret (mean)",
    "std_upwind_regret": "Upwind variance",
    "mean_downwind_regret": "Downwind regret (mean)",
    "std_downwind_regret": "Downwind variance",
    "upwind_downwind_ratio": "Upwind/downwind ratio",
    "dirty_air_exposure_fraction": "Dirty air exposure",
    "mean_flight_quality": "Flight quality",
    "tack_recovery_time": "Tack recovery time",
    "performance_index": "Performance index",
}


def load_regret_by_leg_type() -> pd.DataFrame:
    path = EXPORT_DIR / "ghost_boat_regret_by_leg_type.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp_4b first")
    return pd.read_csv(path)


def load_composite_index() -> pd.DataFrame:
    path = EXPORT_DIR / "ghost_boat_composite_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp_4b first")
    return pd.read_csv(path)


def load_dirty_air_exposure(composite: pd.DataFrame) -> pd.Series:
    """Team-level fraction of leg time in dirty air."""
    path = EXPORT_DIR / "dirty_air_exposure.csv"
    if path.exists():
        exp = pd.read_csv(path)
        if "pct_in_dirty_air" in exp.columns:
            w = exp["racing_seconds"] if "racing_seconds" in exp.columns else 1.0
            exp = exp.assign(weight=w)
            return (
                exp.groupby("team")
                .apply(lambda g: np.average(g["pct_in_dirty_air"], weights=g["weight"]), include_groups=False)
                .rename("dirty_air_exposure_fraction")
            )

    legs = composite.copy()
    legs["leg_length_s"] = legs["ghost_leg_s"].clip(lower=1.0)
    legs["dirty_frac"] = legs["dirty_air_seconds"].fillna(0) / legs["leg_length_s"]
    return legs.groupby("team")["dirty_frac"].mean().rename("dirty_air_exposure_fraction")


def load_flight_quality() -> pd.Series:
    path = EXPORT_DIR / "flight_quality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp_1 first")
    fq = pd.read_csv(path)
    bermuda = fq[fq["venue"] == "Bermuda"]
    return bermuda.groupby("team")["flight_quality"].mean().rename("mean_flight_quality")


def load_tack_recovery() -> pd.Series:
    path = EXPORT_DIR / "flight_quality_manoeuvre_recovery.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="tack_recovery_time")
    rec = pd.read_csv(path)
    return (
        rec.dropna(subset=["recovery_s"])
        .groupby("team")["recovery_s"]
        .mean()
        .rename("tack_recovery_time")
    )


def load_season_points() -> pd.Series:
    regret = pd.read_csv(EXPORT_DIR / "ghost_boat_regret.csv")
    finishes = regret.groupby(["venue", "race_label", "team"], as_index=False)["finish_rank"].first()
    finishes = finishes.dropna(subset=["finish_rank"])
    finishes["points"] = 11 - finishes["finish_rank"]
    return finishes.groupby("team")["points"].sum().rename("season_points")


def assemble_feature_matrix() -> pd.DataFrame:
    leg_type = load_regret_by_leg_type()
    composite = load_composite_index()

    up = leg_type[leg_type["leg_type"] == "upwind"].set_index("team")
    down = leg_type[leg_type["leg_type"] == "downwind"].set_index("team")
    teams = sorted(leg_type["team"].unique())

    rows = []
    dirty = load_dirty_air_exposure(composite)
    fq = load_flight_quality()
    tack = load_tack_recovery()
    perf = composite.groupby("team")["performance_index"].mean()

    for team in teams:
        mu_u = float(up.loc[team, "mean_regret_s"]) if team in up.index else np.nan
        sd_u = float(up.loc[team, "std_regret_s"]) if team in up.index else np.nan
        mu_d = float(down.loc[team, "mean_regret_s"]) if team in down.index else np.nan
        sd_d = float(down.loc[team, "std_regret_s"]) if team in down.index else np.nan
        ratio = mu_u / mu_d if mu_d and mu_d > 0 else np.nan
        rows.append(
            {
                "team": team,
                "mean_upwind_regret": mu_u,
                "std_upwind_regret": sd_u,
                "mean_downwind_regret": mu_d,
                "std_downwind_regret": sd_d,
                "upwind_downwind_ratio": ratio,
                "dirty_air_exposure_fraction": float(dirty.get(team, np.nan)),
                "mean_flight_quality": float(fq.get(team, np.nan)),
                "tack_recovery_time": float(tack.get(team, np.nan)),
                "performance_index": float(perf.get(team, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    filled = df.copy()
    imputed: dict[str, int] = {}
    for col in FEATURES:
        n_miss = int(filled[col].isna().sum())
        if n_miss:
            filled[col] = filled[col].fillna(filled[col].median())
            imputed[col] = n_miss
    return filled, imputed


def pick_k(X: np.ndarray) -> tuple[int, dict[int, float], KMeans]:
    scores: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in (2, 3, 4):
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
        models[k] = km
    best_k = max(scores, key=scores.get)
    return best_k, scores, models[best_k]


def label_archetype(cluster_df: pd.DataFrame, fleet_df: pd.DataFrame) -> str:
    """Rule-based archetype from cluster vs fleet profile."""
    prof = cluster_df[FEATURES].mean()
    fleet = fleet_df[FEATURES].mean()
    z = (prof - fleet) / fleet_df[FEATURES].std().replace(0, 1)

    high_up = z["mean_upwind_regret"] > 0.5
    high_var = z["std_upwind_regret"] > 0.5 or z["std_downwind_regret"] > 0.5
    high_da = z["dirty_air_exposure_fraction"] > 0.5
    low_fq = z["mean_flight_quality"] < -0.5
    high_fq = z["mean_flight_quality"] > 0.5
    low_regret = z["mean_upwind_regret"] < -0.5 and z["mean_downwind_regret"] < -0.5
    slow_tack = z["tack_recovery_time"] > 0.5
    high_pi = z["performance_index"] > 0.5

    if low_regret and high_fq:
        return "Precision flyers"
    if high_var and not high_up:
        return "High-variance competitors"
    if high_up and high_da:
        return "Structural underperformers"
    if high_up and slow_tack:
        return "Manoeuvre-limited upwind"
    if high_pi and low_regret:
        return "Consistent performers"
    if high_var:
        return "Volatile racers"
    if high_up:
        return "Upwind strugglers"
    return "Balanced midfield"


def build_radar(cluster_stats: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure()
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    mins = cluster_stats[FEATURES].min()
    maxs = cluster_stats[FEATURES].max()
    span = (maxs - mins).replace(0, 1)

    for _, row in cluster_stats.iterrows():
        vals = [(row[f] - mins[f]) / span[f] for f in FEATURES]
        vals.append(vals[0])
        theta = labels + [labels[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=theta,
                name=f"Cluster {int(row['cluster'])}: {row['archetype']}",
                fill="toself",
                opacity=0.55,
            )
        )

    fig.update_layout(
        title="Team style clusters — feature radar (0–1 scaled per feature)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=620,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def validate_clusters(
    teams: pd.DataFrame,
    season_points: pd.Series,
) -> dict:
    merged = teams.merge(season_points.reset_index(), on="team", how="left")
    merged = merged.dropna(subset=["season_points"])

    rho_pi, p_pi = stats.spearmanr(merged["performance_index"], merged["season_points"])

    cluster_pi = (
        teams.groupby("cluster", as_index=False)
        .agg(cluster_centroid_pi=("performance_index", "mean"))
        .sort_values("cluster_centroid_pi", ascending=False)
    )
    cluster_pi["cluster_rank"] = range(1, len(cluster_pi) + 1)
    merged2 = merged.merge(
        teams[["team", "cluster"]].merge(cluster_pi[["cluster", "cluster_centroid_pi", "cluster_rank"]], on="cluster"),
        on="team",
        suffixes=("", "_dup"),
    )
    rho_cluster, p_cluster = stats.spearmanr(merged2["cluster_rank"], merged2["season_points"])

    return {
        "spearman_performance_index_vs_season_points": {
            "rho": float(rho_pi),
            "p_value": float(p_pi),
            "n_teams": int(len(merged)),
        },
        "spearman_cluster_rank_vs_season_points": {
            "rho": float(rho_cluster),
            "p_value": float(p_cluster),
            "n_teams": int(len(merged2)),
        },
        "season_points": merged.set_index("team")["season_points"].sort_values(ascending=False).to_dict(),
        "cluster_centroids": cluster_pi.to_dict(orient="records"),
    }


def run() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = assemble_feature_matrix()
    filled, imputed = impute_missing(raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(filled[FEATURES])
    best_k, silhouette_by_k, model = pick_k(X)
    labels = model.fit_predict(X)

    teams = filled.copy()
    teams["cluster"] = labels

    archetypes: dict[int, str] = {}
    for cid in sorted(teams["cluster"].unique()):
        archetypes[int(cid)] = label_archetype(teams[teams["cluster"] == cid], teams)
    teams["archetype"] = teams["cluster"].map(archetypes)

    cluster_stats = (
        teams.groupby("cluster", as_index=False)
        .agg(
            n_teams=("team", "count"),
            teams=("team", lambda s: ", ".join(sorted(s))),
            archetype=("archetype", "first"),
            **{f: (f, "mean") for f in FEATURES},
        )
    )

    season_points = load_season_points()
    validation = validate_clusters(teams, season_points)

    silhouette = silhouette_by_k[best_k]
    criteria = {
        "silhouette": {
            "value": silhouette,
            "threshold": 0.3,
            "pass": silhouette > 0.3,
            "by_k": silhouette_by_k,
        },
        "spearman_performance_index": {
            "value": validation["spearman_performance_index_vs_season_points"]["rho"],
            "threshold": 0.4,
            "pass": validation["spearman_performance_index_vs_season_points"]["rho"] > 0.4,
        },
    }
    overall_pass = all(c["pass"] for c in criteria.values())

    csv_path = EXPORT_DIR / "team_style_clusters.csv"
    radar_path = EXPORT_DIR / "team_style_radar.html"
    json_path = EXPORT_DIR / "team_style_results.json"

    teams.to_csv(csv_path, index=False)
    build_radar(cluster_stats, radar_path)

    results = {
        "experiment": "c4_team_style_clustering",
        "n_teams": int(len(teams)),
        "chosen_k": int(best_k),
        "silhouette_score": float(silhouette),
        "silhouette_by_k": silhouette_by_k,
        "imputed_features": imputed,
        "archetypes": archetypes,
        "cluster_summary": cluster_stats.to_dict(orient="records"),
        "team_assignments": teams[["team", "cluster", "archetype"] + FEATURES].to_dict(orient="records"),
        "validation": validation,
        "success_criteria": criteria,
        "overall_pass": overall_pass,
        "outputs": {
            "clusters_csv": str(csv_path),
            "radar_html": str(radar_path),
            "results_json": str(json_path),
        },
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run()
    print(f"Status: {'PASS' if res['overall_pass'] else 'FAIL'}")
    print(f"Chosen k={res['chosen_k']}, silhouette={res['silhouette_score']:.3f}")
    for cid, name in res["archetypes"].items():
        members = [t["team"] for t in res["team_assignments"] if t["cluster"] == cid]
        print(f"  Cluster {cid} ({name}): {', '.join(members)}")
    val = res["validation"]["spearman_performance_index_vs_season_points"]
    print(f"Spearman ρ (performance_index vs season points) = {val['rho']:.3f} (p={val['p_value']:.4f})")
    print("Outputs:")
    for p in res["outputs"].values():
        print(f"  {p}")
