# Exported Outputs
*All experiment outputs land here. Do not edit manually.*

## Key files by experiment

| File | Source | What it contains |
|------|--------|-----------------|
| `ghost_boat_regret.csv` | A4 | Regret per team × race × leg (763 rows) |
| `ghost_boat_regret_by_leg_type.csv` | A4b | Same, split by upwind/downwind |
| `ghost_boat_composite_index.csv` | A4b | Per-leg performance index (0–1) |
| `ghost_boat_map.html` | A4 | Interactive map of regret by position |
| `ghost_boat_leg_waterfall.html` | A4b | Leg-by-leg regret waterfall per team |
| `flight_quality.csv` | A1 | Flight quality score per second (racing only) |
| `flight_quality_manoeuvre_recovery.csv` | A1b | Per-team tack/gybe recovery times |
| `dirty_air_critical_distance.json` | A3b | 75m threshold + penalty curve |
| `dirty_air_bearing_filtered.html` | A3b | Monotonic penalty curve chart |
| `regret_decomposition.csv` | A2b | Master 763-leg table (all features joined) |
| `regret_decomposition_fraction.json` | C1 | Updated regression with dirty_air_fraction |
| `regret_residual_results.json` | C2 | FQ=86% of residual variance |
| `ghost_boat_mark_rounding_results.json` | C3 | Mark zone fraction per team |
| `team_style_results.json` | C4 | 2 clusters, ρ=0.70 vs season points |
| `position_inheritance_results.json` | C5 | ~1s/rank inheritance coefficient |
| `foiling_leg_start_results.json` | C6 | ITA 61s vs AUS 2s re-establishment |
| `winner_ae_results.json` | C7 | Daggerboard cant = #1 differentiator |
| `first_minute_results.json` | C8 | +425mm, +15 km/h within 60s of start |
| `wind_field_Bermuda_Race_5.html` | race_replay | Animated mark wind arrows on 2D map (TWS color/length) |
| `wind_field_interp_Bermuda_Race_5.html` | race_replay | IDW-interpolated grid quiver across full course bbox |

## Broadcast-ready HTML charts

`dirty_air_bearing_filtered.html` · `ghost_boat_map.html` · `ghost_boat_leg_waterfall.html`  
`ghost_boat_mark_zones.html` · `team_style_radar.html` · `winner_ae_timeline.html`  
`winner_ae_gap_table.html` · `first_minute_correlations.html` · `first_minute_timeline.html`  
`foiling_leg_start_scatter.html` · `position_inheritance.html` · `wind_field_Bermuda_Race_5.html`  
`wind_field_interp_Bermuda_Race_5.html`

### Wind field map

**Mark arrows only** (sparse sensors):

```bash
python dataExploration/race_replay.py \
  --venue Bermuda --race Race_5 \
  --wind-only --step 2
```

**Interpolated grid** (IDW from marks, covers whole course):

```bash
python dataExploration/race_replay.py \
  --venue Bermuda --race Race_5 \
  --wind-only --interpolated --grid-size 18 --step 2

# Or via experiment wrapper:
python dataExploration/next_experiments/exp_wind_field_interp.py
```

Field is interpolated from ~9 mark sensors, not measured everywhere. IDW smooths spatial shifts.

```bash
# View (Plotly CDN + map tiles need a local server)
cd dataExploration/exported && python -m http.server 8765
# open http://127.0.0.1:8765/wind_field_interp_Bermuda_Race_5.html
```
