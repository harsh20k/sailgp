# Next Experiments — Set A, B, C
*SailGP Ocean of Data Challenge · Foil Forward · Deadline Jun 10, 2026*

All scripts output to `../exported/`. Run from repo root with `.venv` active.

---

## Set A + B — Core Metrics (9 scripts)

| Script | What it builds | Key output |
|--------|---------------|-----------|
| `exp_1_flight_quality.py` | Flight quality score (0–1) per second | `flight_quality.csv` — AUC 0.978 |
| `exp_1b_flight_quality_extensions.py` | Tack/gybe recovery time per team | `flight_quality_manoeuvre_recovery.csv` |
| `exp_2_vmg_residual.py` | VMG vs polar table per leg | `exp2_vmg_residual_*.html` |
| `exp_2b_vmg_decomposition.py` | Master 763-leg table + regression | `regret_decomposition.csv` |
| `exp_3_dirty_air.py` | Speed penalty inside 200m following cone | `dirty_air_results.json` |
| `exp_3b_dirty_air_bearing.py` | Bearing-filtered penalty + 75m threshold | `dirty_air_critical_distance.json` |
| `exp_4_ghost_boat.py` | Ghost boat regret per team × leg | `ghost_boat_regret.csv` — ρ=0.915 |
| `exp_4b_ghost_boat_breakdown.py` | Upwind/downwind split + performance index | `ghost_boat_regret_by_leg_type.csv` |
| `exp_5_bubble_loocv.py` | LOOCV validation of fleet-context models | `bubble_loocv_results.json` — dead end |

**Run order:** 1 → 1b → 2 → 2b → 3 → 3b → 4 → 4b → 5 (each builds on previous)

---

## Set C — New Avenues (8 scripts, all run Jun 8)

| Script | What it tests | Status | Key output |
|--------|--------------|--------|-----------|
| `exp_c1_dirty_air_fraction.py` | Dirty air fraction (decoupled from leg length) | ⚠️ Partial | `regret_decomposition_fraction.json` |
| `exp_c2_regret_residual.py` | Regret after removing leg-length confound | ✅ Pass | `regret_residual_results.json` |
| `exp_c3_mark_rounding.py` | Regret by zone (marks vs open water) | ⚠️ Partial | `ghost_boat_mark_rounding_results.json` |
| `exp_c4_team_clustering.py` | Team archetypes (k-means on 9 features) | ✅ Pass | `team_style_results.json` |
| `exp_c5_position_inheritance.py` | Does rank at end of leg N predict leg N+1? | ⚠️ Partial | `position_inheritance_results.json` |
| `exp_c6_foiling_leg_start.py` | Foil re-establishment time post-rounding | ✅ Pass | `foiling_leg_start_results.json` |
| `exp_c7_winner_ae.py` | Neural net trained on winner moments only | ✅ Pass | `winner_ae_results.json` |
| `exp_c8_first_minute.py` | Which sensors predict rank in first 60s | ✅ Pass | `first_minute_results.json` |

**C7 uses MPS GPU automatically if available (Apple Silicon).**

---

## Shared Utilities

`shared/polar.py` — polar table lookup: given wind angle + speed, returns expected VMG. Used by exp_2, exp_2b, and Set C regressions.

---

## Full Index

See `notes/experiment-index.md` for plain-English descriptions, results, and broadcast numbers for every experiment.
