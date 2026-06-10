# LSTM Experiments — Set 0
*SailGP Ocean of Data Challenge · Early architecture exploration*

These are the original neural network experiments. Most were superseded by the more interpretable
physics-based approaches in `next_experiments/`. They provided valuable negative results and
pointed the project towards what actually works.

---

## Scripts

| Script | What it tried | Result |
|--------|--------------|--------|
| `exp_1_speed_forecast.py` | Predict boat speed 5s ahead | ❌ Not interesting beyond "speed predicts speed" |
| `exp_2_foiling_transition.py` | Detect foiling lift-off/splash-down | ⚠️ Works but simple threshold rules are equally good |
| `exp_3_rank_change.py` | Predict rank improvement on next leg | ❌ Could not reliably predict — too many confounds |
| `exp_4_vmg_efficiency.py` | Score VMG vs theoretical max | ⚠️ Superseded by polar table approach in exp_2 |
| `exp_5_team_fingerprint.py` | Identify team from anonymous telemetry | ⚠️ AUC ~0.7 — better than random, not reliable enough |
| `exp_6_bubble_rank.py` | Rank prediction with fleet neighbours | ⚠️ LOOCV F1=0.385 (+0.017 over solo) — marginal |
| `exp_7_bubble_attention.py` | Attention over fleet neighbours | ❌ No meaningful improvement — dead end |

---

## Key Lesson

Adding fleet-context (bubble) features to LSTM models produced marginal, inconsistent gains
(+0.017 F1 in best case). The architecture is at ceiling. Downstream projects should include
bubble features as flat inputs to gradient boosting (TabNet/XGBoost), not as LSTM tokens.

---

## Shared Utilities

`shared/data_prep.py` — data loading, feature engineering, device detection (MPS/CUDA/CPU)  
`shared/models.py` — LSTM, TCN, and AE architecture definitions  
`shared/evaluation.py` — LOOCV, F1, rank correlation utilities  
`shared/fleet.py` — bubble/fleet-context feature construction

---

## Full Index

See `notes/experiment-index.md` for plain-English descriptions of all experiments across all sets.
