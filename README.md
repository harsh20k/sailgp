# The race is decided in the first 60 seconds

**Foil Forward · Stream #2: On the Water** · Ocean of Data Challenge · June 2026

Four experiments, one causal chain — built from F50 telemetry (Bermuda 2026, 8 races + Halifax 2024, 6 races).  
**Full write-up:** open [`race-decided-in-60-seconds.html`](race-decided-in-60-seconds.html) in a browser.

---

## C8 · The first 60 seconds

Within 20 s of the gun, eventual top-4 boats fly ~40 cm higher and 15 km/h faster — before any tactical choices. **11/12** sensor gaps survive wind-speed control.

| +15.3 km/h | +425 mm port foil | 11/12 gaps | 116 start windows |
|------------|-------------------|------------|-------------------|

<p align="center">
  <img src="assets/race-decided/img1_first60seconds.png" alt="Two foiling catamarans at 20 seconds after the start" width="48%" />
</p>
<p align="center">
  <img src="assets/race-decided/charts/chart-c8.png" alt="Mean ride height in first 60 seconds by finish group" width="48%" /><br/>
  <sub><em>Source: SailGP race data</em></sub>
</p>

---

## C2 · Flight quality drives the gap

After leg-length control (74.8% of raw variance), flight quality explains **86.3%** of the controllable gap. Dirty air + VMG efficiency ≈ 14%.

<p align="center">
  <img src="assets/race-decided/img2_flight_quality.png" alt="Catamaran flying cleanly on hydrofoils" width="48%" />
</p>
<p align="center">
  <img src="assets/race-decided/charts/chart-c2.png" alt="Share of residual variance by factor" width="48%" /><br/>
  <sub><em>Source: SailGP race data</em></sub>
</p>

---

## A4b · Ghost boat — fair benchmark

Per-team optimal path in each boat's wind. Total regret vs finish rank: **Spearman ρ = 0.915**. Upwind regret **2.2×** downwind. Best upwind: AUS 28.8 s.

<p align="center">
  <img src="assets/race-decided/img3_ghost_boat.png" alt="Real boat chasing a ghost boat benchmark" width="48%" />
</p>
<p align="center">
  <img src="assets/race-decided/charts/chart-a4b.png" alt="Mean regret by leg type" width="48%" /><br/>
  <sub><em>Source: SailGP race data</em></sub>
</p>

---

## C6 · Slow restarts after turns

Slow re-foiling after mark roundings: ITA **61 s** vs AUS **1.5 s** (~8 min lost/race). Explains **52.7%** of upwind-regret variance.

<p align="center">
  <img src="assets/race-decided/img4_restart_failure.png" alt="Fast vs slow re-foiling after a mark rounding" width="48%" />
</p>
<p align="center">
  <img src="assets/race-decided/charts/chart-c6.png" alt="Seconds to re-fly by team" width="48%" /><br/>
  <sub><em>Source: SailGP race data</em></sub>
</p>

**For coaches:** start ride height, mid-leg flight stability, restart speed after turns.

---

## Wind · Course wind field

Mark sensors interpolated across the full course (IDW) — wind speed and direction for every manoeuvre.

<p align="center">
  <img src="assets/race-decided/img5_wind_field.png" alt="Race course with colour-coded wind arrows" width="720" /><br/>
  <sub><em>Source: SailGP race data</em></sub>
</p>

Interactive: `dataExploration/exported/wind_field_interp_Bermuda_Race_5.html`

---

## Glossary

| Term | Meaning |
|------|---------|
| **Flight quality** | 0–1 score from ride height + foiling sensors |
| **Ghost boat** | Per-team optimal path using fleet polar + that boat's wind |
| **Regret** | Seconds lost vs ghost over a leg (actual − ghost time) |
| **Re-establishment** | Seconds until flight quality > 0.7 after a turn |

---

## Data & repo

- **Source:** [SailGP challenge dataset](https://drive.google.com/file/d/1yXLn4tzXRdJ2C-udGX4OavMSsRIeAblx/view) (local: `DataChallenge_Export/`)
- **Interactive charts:** `dataExploration/exported/` (ghost boat animation, wind field, first-minute timelines)
- **Reproduce:** `pip install -r requirements.txt` → `python scripts/build_snapshot.py`
