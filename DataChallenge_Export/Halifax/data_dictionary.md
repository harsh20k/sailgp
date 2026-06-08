# SailGP Data Challenge — Data Dictionary

## 1. Introduction to SailGP

SailGP is the world's most exciting sail racing league, featuring identical high-performance
F50 catamarans racing in iconic harbours around the globe. National teams compete across a
multi-event season, earning points that determine who qualifies for the winner-takes-all
Grand Final.

The F50 is a foiling catamaran — a twin-hulled boat that rises above the water on hydrofoils
(underwater wings), reducing drag and enabling speeds exceeding 100 km/h. The
boat is powered by a rigid wing sail (not a traditional soft sail) that works like an
aeroplane wing turned on its side, generating enormous aerodynamic force.

Key components of the F50:
- **Wing sail**: A rigid, articulated wing (it has 4 configurations, 18m, 24m, 27.5m and 28m tall) with adjustable camber at up to 6 stations
  (CA1 at the bottom to CA6 at the top), plus overall twist and rotation controls.
- **Jib**: A smaller headsail in front of the wing that helps with upwind performance.
- **Daggerboards (foils)**: Two retractable hydrofoils — one on each hull (port and starboard).
  When deployed, they generate vertical lift to raise the boat out of the water. Their rake
  (fore-aft angle) and cant (side-to-side angle) are adjustable.
- **Rudders**: Two rudders at the stern, each with adjustable rake. They also have small
  horizontal foils (elevators) that help control pitch.
- **Platform**: The overall catamaran structure. Key attitudes are pitch (bow up/down),
  heel (leaning port/starboard), and ride height (how far above the water).

## 2. How Racing Works

### The SailGP course

SailGP uses a distinctive course layout. The wind blows roughly from top to bottom.

### Race sequence

1. **Pre-start**: Boats manoeuvre near the start line. The critical phase is the final
   60–90 seconds as boats jockey for position and time their run to the line.
2. **Reaching start**: The fleet crosses the start line on a reach — sailing roughly
   perpendicular to the wind (~90° TWA) at high speed towards the mid-course mark (M1),
   which is positioned in the axis of the main course axis.
3. **M1 rounding**: Boats round M1 and bear away onto the first downwind leg towards
   the leeward gate (LG).
4. **Downwind leg**: Boats sail downwind towards the leeward gate (LG1/LG2), gybing
   to optimise their angle.
5. **Leeward gate**: Boats choose which mark of the gate to round (LG1 or LG2) based
   on tactical advantage, then turn upwind.
6. **Upwind leg**: Boats tack upwind towards the windward gate (WG1/WG2). This is the
   slowest leg, but tactically rich — choosing the right side of the course based on
   wind shifts and current is critical.
7. **Windward gate**: Boats choose which mark to round (WG1 or WG2), then bear away
   downwind.
8. **Laps**: Steps 4–7 repeat for several laps.
9. **Finish**: After the final leg, boats cross the finish line near the start line.
   Shortest elapsed time wins.

### Race outcomes
Not every boat finishes every race normally. Possible outcomes include:
- **Finished** — crossed the finish line normally
- **OCS** (On Course Side) — crossed the start line early; the boat must drop behind the rest of the fleet before resuming racing
- **DNF** (Did Not Finish) — started but could not finish (e.g. equipment failure)
- **DNS** (Did Not Start) — was scheduled but did not start
- **DSQ** (Disqualified) — removed from results due to a rule infringement
- **DNC** (Did Not Compete) — did not take part in the race at all

## 3. Dataset Overview

| Property | Value |
|----------|-------|
| Event | Halifax |
| Date range | 2024-06-01 to 2024-06-02 |
| Number of races | 6 |
| Data frequency | 1 Hz (1 sample per second) |
| Pre-start included | Yes (6 minutes before race start) |
| Teams | AUS, CAN, DEN, ESP, FRA, GBR, GER, NZL, SUI, USA |

### Data sources
- **Boat telemetry**: Collected by onboard sensors at up to 100 Hz, aggregated to 1 Hz.
  Sourced from InfluxDB time-series database.
- **Mark data**: GPS positions and wind readings from sensors mounted on course marks.
  Provides a distributed wind field across the racecourse.
- **Race XMLs**: Race configuration files from the race management system (Oracle database).
  Contain course layout, mark positions, and race timing data.

## 4. Folder Structure

```
Halifax/
  boats/                     Boat telemetry CSVs
    Race_1/
      AUS.csv                One CSV per team per race
      GBR.csv
      ...
    Race_2/
      ...
  marks/                     Course mark positions and wind
    Race_1/
      marks.csv
    Race_2/
      ...
  xmls/                      Race configuration XMLs
    Race_1/
      ...
  race_metadata.csv          Summary: race times, wind, participants
  data_dictionary.md         This file
```

## 5. Boat Data — Variable Descriptions

Each boat CSV has a `DATETIME` index (UTC timestamps) and a `TEAM` column, plus the
following telemetry channels:

### Position & Navigation

| Column | Description | Unit |
|--------|-------------|------|
| LATITUDE_GPS_unk | GPS latitude | degrees (decimal) |
| LONGITUDE_GPS_unk | GPS longitude | degrees (decimal) |
| HEADING_deg | Compass heading (where the bow points) | degrees (0–360) |
| GPS_COG_deg | Course over ground (actual direction of travel) | degrees (0–360) |
| GPS_SOG_km_h_1 | Speed over ground (GPS-derived) | km/h |

### Speed

| Column | Description | Unit |
|--------|-------------|------|
| BOAT_SPEED_km_h_1 | Boat speed through the water | km/h |
| VMG_km_h_1 | Velocity Made Good — speed component towards/away from the next mark | km/h |

### Wind

| Column | Description | Unit |
|--------|-------------|------|
| TWA_SGP_deg | True Wind Angle — angle between boat heading and true wind | degrees (0=head to wind, 180=downwind) |
| TWS_SGP_km_h_1 | True Wind Speed — wind speed corrected for boat motion | km/h |
| TWD_SGP_deg | True Wind Direction — compass direction the wind blows FROM | degrees (0–360) |
| AWA_SGP_deg | Apparent Wind Angle — wind as felt on the moving boat | degrees |
| AWS_SGP_km_h_1 | Apparent Wind Speed — wind speed as measured on the boat | km/h |

### Platform Orientation

| Column | Description | Unit |
|--------|-------------|------|
| PITCH_deg | Pitch angle — positive = bow up | degrees |
| HEEL_deg | Heel angle — lean to port (positive) or starboard (negative) | degrees |
| RATE_YAW_deg_s_1 | Yaw rate (rate of turn) | degrees/second |
| RATE_PITCH_deg_s_1 | Pitch rate | degrees/second |
| RATE_ROLL_deg_s_1 | Roll rate | degrees/second |
| LEEWAY_deg | Leeway — sideways slip angle between heading and actual track | degrees |

### Wing Sail

| Column | Description | Unit |
|--------|-------------|------|
| ANGLE_WING_TWIST_deg | Wing twist — differential angle between top and bottom of wing | degrees |
| ANGLE_WING_ROT_deg | Wing rotation — angle of the entire wing relative to the boat | degrees |
| ANGLE_CA1_deg | Camber angle at station 1 (lowest) | degrees |
| ANGLE_CA2_deg | Camber angle at station 2 | degrees |
| ANGLE_CA3_deg | Camber angle at station 3 | degrees |
| ANGLE_CA4_deg | Camber angle at station 4 | degrees |
| ANGLE_CA5_deg | Camber angle at station 5 | degrees |
| ANGLE_CA6_deg | Camber angle at station 6 (highest) | degrees |

Camber angles control the curvature (power) of the wing at each station. More camber =
more power but also more drag. Teams adjust these dynamically throughout the race.

### Jib

| Column | Description | Unit |
|--------|-------------|------|
| PER_JIB_LEAD_pct | Jib lead position — controls the vertical trim of the jib | percent |
| PER_JIB_SHEET_pct | Jib sheet position — controls how tightly the jib is pulled in | percent |

### Ride Height

| Column | Description | Unit |
|--------|-------------|------|
| LENGTH_RH_P_mm | Port hull ride height above water | millimetres |
| LENGTH_RH_S_mm | Starboard hull ride height above water | millimetres |
| LENGTH_RH_BOW_mm | Bow ride height above water | millimetres |

When foiling, ride heights are typically 500–2000 mm. Near zero means the hull is
touching the water (displacement sailing). Negative values can indicate sensor noise
or wave effects.

### Daggerboard (Foil)

| Column | Description | Unit |
|--------|-------------|------|
| ANGLE_DB_RAKE_P_deg | Port daggerboard rake — fore-aft tilt of the foil | degrees |
| ANGLE_DB_RAKE_S_deg | Starboard daggerboard rake | degrees |
| ANGLE_DB_CANT_P_deg | Port daggerboard cant — side-to-side tilt of the foil | degrees |
| ANGLE_DB_CANT_S_deg | Starboard daggerboard cant | degrees |
| LENGTH_DB_H_P_mm | Port daggerboard deploy length — how far the foil extends below the hull | millimetres |
| LENGTH_DB_H_S_mm | Starboard daggerboard deploy length | millimetres |

Only one daggerboard is deployed at a time (the leeward one — the side the boat leans
towards). The deploy length tells you which board is down — a large value means the foil
is extended and generating lift, while a small value means it is retracted (stowed).
Rake angle controls the boat's ride height and pitch. Cant angle affects the direction
of the lift force.

### Rudder

| Column | Description | Unit |
|--------|-------------|------|
| ANGLE_RUDDER_deg | Rudder angle — controls boat steering | degrees |
| ANGLE_RUD_RAKE_P_deg | Port rudder rake — fore-aft tilt of rudder foil | degrees |
| ANGLE_RUD_RAKE_S_deg | Starboard rudder rake | degrees |

### Race State

| Column | Description | Unit / Values |
|--------|-------------|---------------|
| TRK_RACE_NUM_unk | Race number in YYMMDDRR format (e.g., 25031501 = 2025-03-15 Race 01) | integer |
| TRK_LEG_NUM_unk | Current leg number within the race | integer |
| TRK_LEG_NUM_TOT_unk | Total number of legs in the race | integer |
| TRK_BOAT_RACE_STATUS_unk | Boat's racing status (see table below) | integer (0–8) |
| TRK_RACE_RANK_unk | Boat's current position in the race (1 = leading) | integer |
| PC_BEACON_NUMBER_unk | Current mark/gate the boat is heading towards (0–10) | integer |
| BROADCAST_MODE_unk | Whether the boat is in broadcast mode (1 = active race broadcast) | 0 or 1 |

#### TRK_BOAT_RACE_STATUS_unk values

| Value | Code | Meaning |
|-------|------|---------|
| 0 | bsNone | No race state — boat is not in a race |
| 1 | bsPrestart | Pre-start — boat is manoeuvring before the start signal |
| 2 | bsRacing | Racing — boat is actively racing on the course |
| 3 | bsFinished | Finished — boat has crossed the finish line |
| 4 | bsDNS | Did Not Start — boat was scheduled but did not start |
| 5 | bsDNF | Did Not Finish — boat started but could not finish |
| 6 | bsDSQ | Disqualified — boat removed from results for rule infringement |
| 7 | bsOCS | On Course Side — boat crossed start line early; must drop behind the fleet |
| 8 | bsDNC | Did Not Compete — boat did not take part in the race |

During normal racing, you will typically see the status transition: 0 → 1 (pre-start) → 2
(racing) → 3 (finished). A boat that crosses the start line early will show status 7 (OCS). To clear the
penalty, it must drop behind all other boats before resuming racing (status reverts to 2). Status values 4–8
appear at the end of a race when a boat has an abnormal result.

### Race Progress

| Column | Description | Unit |
|--------|-------------|------|
| DISTANCE_RACE_m | Total distance sailed since race start | metres |
| DISTANCE_RACE_LEG_m | Distance sailed in current leg | metres |
| TIME_RACE_s | Elapsed time since race start | seconds |
| TIME_RACE_LEG_s | Elapsed time in current leg | seconds |

### Tactical

| Column | Description | Unit |
|--------|-------------|------|
| PC_DTB_m | Distance to boat ahead | metres |
| PC_TTS_s | Time to start line (pre-start phase) | seconds |
| PC_DTO_m | Distance to next mark/opponent | metres |
| PC_DTL_m | Distance to leader | metres |

### Penalty

| Column | Description | Unit |
|--------|-------------|------|
| TRK_PENALTY_COUNT_unk | Number of penalties accumulated | integer |
| TRK_PENALTY_DISTANCE_m | Penalty distance to be served | metres |

### Ocean Current

| Column | Description | Unit |
|--------|-------------|------|
| CURRENT_DIRECTION_deg | Direction the ocean current flows TOWARDS | degrees (0–360) |
| CURRENT_MAGNITUDE_km_h_1 | Speed of the ocean current | km/h |

## 6. Mark Data — Variable Descriptions

Mark CSVs contain GPS positions and wind readings from sensors on course buoys.

| Column | Description | Unit |
|--------|-------------|------|
| DATETIME | Timestamp (UTC) | ISO 8601 |
| MARK | Mark identifier (WG1, WG2, LG1, LG2, SL1, SL2, M1, FL1, FL2) | string |
| LATITUDE_deg | Mark latitude | degrees |
| LONGITUDE_deg | Mark longitude | degrees |
| TWD_deg | True wind direction measured at this mark | degrees |
| TWS_km_h_1 | True wind speed measured at this mark | km/h |

### Mark abbreviations
- **SL1, SL2**: Start / Finish Line marks
- **M1**: Mid-course mark — first mark after the reaching start
- **LG1, LG2**: Leeward Gate (downwind marks) — boats pass between these at the bottom of the course
- **WG1, WG2**: Windward Gate (upwind marks) — boats pass between these at the top of the course
- **FL1, FL2**: Finish Line marks (when separate from start line)

Wind data from marks is valuable because it gives a spatially distributed picture of
the wind field across the racecourse, not just what each boat experiences locally.

## 7. Race Metadata (`race_metadata.csv`)

| Column | Description |
|--------|-------------|
| race_label | Human-readable label (Race_1, Race_2, ...) |
| race_number | System race number in YYMMDDRR format |
| prestart_start_utc | When the pre-start period begins (UTC) |
| race_start_utc | Official race start time (UTC) |
| race_end_utc | Race end time (UTC) |
| race_start_local | Race start time in local timezone |
| timezone | Local timezone identifier |
| avg_tws_km_h | Average true wind speed during the race (km/h) |
| avg_twd_deg | Average true wind direction during the race (degrees) |
| num_boats | Number of boats in the race |
| teams | Comma-separated list of team names |

## 8. XML Data

The `xmls/` folders contain race configuration XML files from the race management system.
These include:
- Course layout (mark positions, course axis)
- Race timing (scheduled start, sequence)
- Boat-to-team assignments

XMLs are primarily useful for understanding the intended course layout and race scheduling.

## 9. Key Concepts & Glossary

| Term | Definition |
|------|------------|
| **Foiling** | Sailing above the water surface on hydrofoils, dramatically reducing drag |
| **Reaching** | Sailing roughly perpendicular to the wind (~90° TWA) — the fastest point of sail |
| **Tack** | A manoeuvre turning the bow through the wind (used when sailing upwind) |
| **Gybe** | A manoeuvre turning the stern through the wind (used when sailing downwind) |
| **Bear away** | Turning away from the wind (e.g. from upwind to downwind at a windward gate) |
| **Port** | Left side of the boat (when looking forward) |
| **Starboard** | Right side of the boat (when looking forward) |
| **Leeward** | The side away from the wind (downwind side) |
| **Windward** | The side towards the wind (upwind side) |
| **VMG** | Velocity Made Good — the component of boat speed towards the upwind or downwind mark |
| **TWA** | True Wind Angle — the angle between the boat's heading and the true wind direction |
| **TWS** | True Wind Speed — the actual wind speed, corrected for boat movement |
| **AWA/AWS** | Apparent wind — what the crew feels; a combination of true wind and boat speed |
| **Camber** | The curvature of the wing sail — more camber generates more power |
| **Rake** | The fore-aft angle of a foil or rudder — controls vertical lift and pitch |
| **Cant** | The side-to-side angle of a daggerboard — affects lift direction |
| **Ride height** | Vertical distance between hull and water surface when foiling |
| **OCS** | On Course Side — crossed the start line early; must drop behind the fleet to clear the penalty |
| **DNS** | Did Not Start — boat was entered but did not begin the race |
| **DNF** | Did Not Finish — boat started but retired before finishing |
| **DSQ** | Disqualified — boat removed from results for a rule violation |
| **DNC** | Did Not Compete — boat was absent from the race entirely |

## 10. Sign Conventions

- **Heel**: Positive = leaning to port, Negative = leaning to starboard
- **Pitch**: Positive = bow up, Negative = bow down
- **TWA**: 0° = head to wind, 180° = directly downwind. Values 0–180° indicate port tack,
  180–360° indicate starboard tack.
- **Wind direction** (TWD): Compass bearing the wind blows FROM (e.g., 270° = westerly wind)
- **Current direction**: Compass bearing the current flows TOWARDS
- **Latitude**: Positive = North, Negative = South
- **Longitude**: Positive = East, Negative = West

## 11. Getting Started

```python
import pandas as pd
from pathlib import Path

# Load one boat's data for Race 1
race_dir = Path("Halifax/boats/Race_1")
aus = pd.read_csv(race_dir / "AUS.csv", parse_dates=["DATETIME"], index_col="DATETIME")

# Load all boats for a race
boats = {}
for csv_file in race_dir.glob("*.csv"):
    team = csv_file.stem
    boats[team] = pd.read_csv(csv_file, parse_dates=["DATETIME"], index_col="DATETIME")

# Load mark positions
marks = pd.read_csv("Halifax/marks/Race_1/marks.csv",
                     parse_dates=["DATETIME"])

# Load race metadata
metadata = pd.read_csv("Halifax/race_metadata.csv")

# Quick plot: boat speed over the race
import matplotlib.pyplot as plt
for team, df in boats.items():
    plt.plot(df.index, df["BOAT_SPEED_km_h_1"], label=team, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Boat Speed (km/h)")
plt.legend()
plt.title("Race 1 — Boat Speed")
plt.show()

# Plot the racecourse from mark positions
for mark_name, group in marks.groupby("MARK"):
    plt.scatter(group["LONGITUDE_deg"].mean(), group["LATITUDE_deg"].mean(),
                label=mark_name, s=100)
for team, df in boats.items():
    plt.plot(df["LONGITUDE_GPS_unk"], df["LATITUDE_GPS_unk"], alpha=0.5, label=team)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.title("Race 1 — Course Map")
plt.axis("equal")
plt.show()
```
