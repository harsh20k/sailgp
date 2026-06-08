#!/usr/bin/env python3
"""Generate dataExploration.ipynb from plan."""
import json
from pathlib import Path

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True) if isinstance(s, str) else s}

def code(s):
    return {"cell_type": "code", "metadata": {}, "source": s.splitlines(keepends=True), "outputs": [], "execution_count": None}

cells = []

cells.append(md("""# SailGP Exploratory Visual Data Analysis

F50 catamaran telemetry — Bermuda (2026) & Halifax (2024).

**Data:** `DataChallenge_Export/` · **Schema:** `DataChallenge_Export/Bermuda/data_dictionary.md`
"""))

cells.append(code(r'''## Section 0 — Environment Setup
# Install all packages (pinned versions in comments for reproducibility)
# pandas==2.2.* numpy==1.26.* matplotlib==3.8.* seaborn==0.13.* plotly==5.22.*
%pip install -q pandas numpy matplotlib seaborn plotly kaleido scipy scikit-learn
%pip install -q ydata-profiling folium pyarrow fastparquet
%pip install -q torch torchvision umap-learn dtaidistance tqdm ipywidgets nbformat>=4.2.0
try:
    %pip install -q windrose
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, IFrame
from tqdm.notebook import tqdm

def find_data_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        candidate = p / "DataChallenge_Export"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "DataChallenge_Export/ not found. Open this notebook from the sailgp repo "
        "(repo root or dataExploration/)."
    )

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BASE = find_data_root()
REPO_ROOT = BASE.parent
VENUES = ["Bermuda", "Halifax"]
PLOTLY_LAYOUT = dict(template="plotly_dark", font=dict(family="IBM Plex Mono, monospace"))
FIGSIZE = (16, 8)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = FIGSIZE
sns.set_theme(style="darkgrid")

def apply_plotly(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

print("Environment ready.")
'''))

cells.append(code(r'''## Section 1 — Data Loading & Schema Audit
# Load all boat CSVs into df_all; load metadata and marks per venue/race.
import pandas as pd
from tqdm.notebook import tqdm

if "BASE" not in globals() or not BASE.exists():
    BASE = find_data_root()
    VENUES = ["Bermuda", "Halifax"]

dfs, meta_parts, marks = [], [], {}
for venue in VENUES:
    meta_parts.append(pd.read_csv(BASE / venue / "race_metadata.csv").assign(venue=venue))
    for mpath in sorted((BASE / venue / "marks").rglob("marks.csv")):
        race_label = mpath.parent.name
        marks[(venue, race_label)] = pd.read_csv(mpath, parse_dates=["DATETIME"])
    for boat_csv in tqdm(sorted((BASE / venue / "boats").rglob("*.csv")), desc=venue):
        race_label = boat_csv.parent.name
        team = boat_csv.stem
        tmp = pd.read_csv(boat_csv, parse_dates=["DATETIME"], index_col="DATETIME")
        tmp.index = pd.to_datetime(tmp.index, utc=True, errors="coerce")
        tmp = tmp[~tmp.index.isna()]
        tmp["venue"] = venue
        tmp["race_label"] = race_label
        tmp["team"] = team
        dfs.append(tmp)

df_all = pd.concat(dfs).sort_index()
meta_all = pd.concat(meta_parts, ignore_index=True)

# Schema audit
numeric = df_all.select_dtypes(include=[np.number])
audit = []
for col in df_all.columns:
    s = df_all[col]
    audit.append({
        "column": col,
        "dtype": str(s.dtype),
        "pct_missing": round(100 * s.isna().mean(), 2),
        "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
        "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
        "nunique": s.nunique(),
    })
audit_df = pd.DataFrame(audit).set_index("column")

styled = (
    audit_df.style.background_gradient(subset=["pct_missing"], cmap="YlOrRd")
    .format({"pct_missing": "{:.1f}%"})
)
display(styled)

print(f"Total rows: {len(df_all):,}")
print(f"Date range: {df_all.index.min()} → {df_all.index.max()}")
print(f"Memory: {df_all.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print("\nTeams per venue:")
print(df_all.groupby("venue")["team"].nunique())
print("\nRaces per venue:")
print(df_all.groupby(["venue", "race_label"]).ngroups)
print(f"Loaded {len(marks)} mark files")
'''))

cells.append(code(r'''%%time
## Section 2 — Automated Data Profiling (ydata-profiling)
# 10% stratified sample by venue + race_label; save sailgp_profile.html
from ydata_profiling import ProfileReport

df_sample = (
    df_all.groupby(["venue", "race_label"], group_keys=False)
    .apply(lambda g: g.sample(n=max(1, int(len(g) * 0.10)), random_state=RANDOM_STATE))
)
if len(df_sample) < 3000:
    df_sample = df_all.sample(min(10000, len(df_all)), random_state=RANDOM_STATE)

profile = ProfileReport(
    df_sample.reset_index(),
    title="SailGP Full Dataset Profile",
    explorative=True,
    minimal=False,
    correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
    missing_diagrams={"heatmap": True, "dendrogram": True},
)
profile.to_file("sailgp_profile.html")
display(HTML('<a href="sailgp_profile.html" target="_blank">Open sailgp_profile.html</a>'))

num_cols = df_sample.select_dtypes(include=[np.number]).columns
if len(num_cols) > 1:
    corr = df_sample[num_cols].corr(method="pearson")
    pairs = []
    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if j <= i:
                continue
            pairs.append((a, b, corr.loc[a, b]))
    top_pairs = pd.DataFrame(pairs, columns=["col_a", "col_b", "pearson"]).assign(abs_r=lambda d: d.pearson.abs()).nlargest(10, "abs_r")
    display(top_pairs.drop(columns="abs_r"))
'''))

cells.append(code(r'''## Section 3 — Racing-Only Filter & Status Distribution
# Filter TRK_BOAT_RACE_STATUS_unk == 2; status bar charts and fleet presence
STATUS_LABELS = {0: "bsNone", 1: "bsPrestart", 2: "bsRacing", 3: "bsFinished",
                 4: "bsDNS", 5: "bsDNF", 6: "bsDSQ", 7: "bsOCS", 8: "bsDNC"}

status_counts = df_all["TRK_BOAT_RACE_STATUS_unk"].value_counts().sort_index()
status_df = pd.DataFrame({
    "status": [STATUS_LABELS.get(int(k), str(k)) for k in status_counts.index],
    "rows": status_counts.values,
})
fig = px.bar(status_df, x="rows", y="status", orientation="h", title="Rows by race status")
apply_plotly(fig).show()

df_racing = df_all[df_all["TRK_BOAT_RACE_STATUS_unk"] == 2].copy()
dropped = len(df_all) - len(df_racing)
print(f"df_racing: {len(df_racing):,} rows ({dropped:,} dropped)")

fleet = df_all.groupby(["venue", "race_label", "team"]).size().reset_index(name="rows")
fig2 = px.bar(fleet, x="race_label", y="team", color="team", facet_col="venue",
              title="Boats present per race", barmode="stack")
fig2.update_layout(showlegend=False)
if "Race_6" in fleet["race_label"].values and (fleet["venue"] == "Halifax").any():
    fig2.add_annotation(text="Halifax Race 6: only 3 boats", xref="paper", yref="paper",
                        x=0.85, y=0.15, showarrow=True, arrowhead=2)
apply_plotly(fig2).show()
'''))

cells.append(code(r'''## Section 4 — GPS Track Visualization
# Folium maps per venue + Plotly mapbox for Bermuda Race 1
import folium
from folium import plugins

TEAM_COLORS = px.colors.qualitative.Plotly

def build_folium_map(venue, out_path):
    m = folium.Map(location=[32.27, -64.85] if venue == "Bermuda" else [44.65, -63.57], zoom_start=13)
    races = sorted(df_all[df_all["venue"] == venue]["race_label"].unique())
    for ri, race in enumerate(races):
        fg = folium.FeatureGroup(name=race, show=(ri == 0))
        sub = df_all[(df_all["venue"] == venue) & (df_all["race_label"] == race)]
        teams = sorted(sub["team"].unique())
        for ti, team in enumerate(teams):
            tdf = sub[sub["team"] == team]
            if "LATITUDE_GPS_unk" not in tdf.columns:
                continue
            coords = tdf[["LATITUDE_GPS_unk", "LONGITUDE_GPS_unk"]].dropna().values.tolist()
            if len(coords) < 2:
                continue
            color = TEAM_COLORS[ti % len(TEAM_COLORS)]
            folium.PolyLine(coords, color=color, weight=2, opacity=0.7, popup=team).add_to(fg)
        fg.add_to(m)
        key = (venue, race)
        if key in marks:
            md = marks[key].groupby("MARK").agg(
                LATITUDE_deg=("LATITUDE_deg", "mean"),
                LONGITUDE_deg=("LONGITUDE_deg", "mean"),
                TWS_km_h_1=("TWS_km_h_1", "mean"),
                TWD_deg=("TWD_deg", "mean"),
            ).reset_index()
            for _, row in md.iterrows():
                folium.CircleMarker(
                    [row.LATITUDE_deg, row.LONGITUDE_deg], radius=6, color="white", fill=True,
                    popup=f"{row.MARK}<br>TWS={row.TWS_km_h_1:.1f}<br>TWD={row.TWD_deg:.0f}",
                ).add_to(fg)
    folium.LayerControl().add_to(m)
    m.save(out_path)
    return out_path

build_folium_map("Bermuda", "map_bermuda.html")
build_folium_map("Halifax", "map_halifax.html")
display(IFrame("map_bermuda.html", width="100%", height=450))
display(IFrame("map_halifax.html", width="100%", height=450))

sub = df_all[(df_all["venue"] == "Bermuda") & (df_all["race_label"] == "Race_1")].reset_index()
fig = px.scatter_mapbox(sub, lat="LATITUDE_GPS_unk", lon="LONGITUDE_GPS_unk", color="BOAT_SPEED_km_h_1",
                        hover_name="team", mapbox_style="carto-darkmatter", zoom=12, height=600,
                        title="Bermuda Race 1 — speed coloured tracks")
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
apply_plotly(fig).show()
'''))

cells.append(code(r'''## Section 5 — Speed & VMG Analysis
# Box, VMG scatter, polar speed diagrams, violin by leg
import statsmodels.api as sm

# 1. Box plot speed by team
med = df_racing.groupby(["venue", "team"])["BOAT_SPEED_km_h_1"].median().reset_index()
order = med.sort_values("BOAT_SPEED_km_h_1", ascending=False)["team"].tolist()
fig1 = px.box(df_racing, x="team", y="BOAT_SPEED_km_h_1", color="venue", category_orders={"team": order},
              points=False, title="Boat speed by team")
apply_plotly(fig1).show()

# 2. VMG vs TWA with LOWESS
df_w = df_racing.dropna(subset=["TWA_SGP_deg", "VMG_km_h_1", "BOAT_SPEED_km_h_1"]).copy()
df_w["mode"] = np.where(df_w["TWA_SGP_deg"] < 100, "upwind", "downwind")
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, mode in zip(axes, ["upwind", "downwind"]):
    d = df_w[df_w["mode"] == mode].sample(min(8000, len(df_w)), random_state=RANDOM_STATE)
    sns.scatterplot(data=d, x="TWA_SGP_deg", y="VMG_km_h_1", hue="BOAT_SPEED_km_h_1",
                    palette="viridis", alpha=0.3, ax=ax, legend=False)
    if len(d) > 100:
        lowess = sm.nonparametric.lowess(d["VMG_km_h_1"], d["TWA_SGP_deg"], frac=0.2)
        ax.plot(lowess[:, 0], lowess[:, 1], color="red", lw=2)
    ax.set_title(mode)
plt.suptitle("VMG vs TWA (LOWESS)")
plt.tight_layout()
plt.show()

# 3. Polar speed diagrams (sample teams)
teams_polar = sorted(df_racing["team"].unique())[:15]
fig3, axes = plt.subplots(3, 5, figsize=(20, 12), subplot_kw=dict(projection="polar"))
for ax, team in zip(axes.flat, teams_polar):
    d = df_racing[df_racing["team"] == team].dropna(subset=["TWA_SGP_deg", "BOAT_SPEED_km_h_1"])
    d = d.assign(twa_bin=(d["TWA_SGP_deg"] // 10) * 10)
    g = d.groupby("twa_bin")["BOAT_SPEED_km_h_1"].mean()
    theta = np.deg2rad(g.index.values)
    ax.bar(theta, g.values, width=np.deg2rad(8), alpha=0.7)
    ax.set_title(team, fontsize=8)
plt.suptitle("Mean speed vs TWA (10° bins)")
plt.tight_layout()
plt.show()

# 4. Violin GPS SOG by leg
fig4 = px.violin(df_racing, x="TRK_LEG_NUM_unk", y="GPS_SOG_km_h_1", color="venue", box=True,
                 title="GPS SOG by leg")
apply_plotly(fig4).show()
'''))

cells.append(code(r'''## Section 6 — Wind Analysis
# Wind rose, mark vs boat TWS, TWD heatmap, TWA histogram
def wind_rose_manual(ax, twd, tws, n_sectors=16):
    sectors = np.linspace(0, 360, n_sectors + 1)
    twd = twd % 360
    hist, _ = np.histogram(twd, bins=sectors, weights=tws)
    theta = np.deg2rad((sectors[:-1] + sectors[1:]) / 2)
    ax.bar(theta, hist, width=2 * np.pi / n_sectors, alpha=0.8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection="polar"))
for ax, venue in zip(axes, VENUES):
    d = df_racing[df_racing["venue"] == venue].dropna(subset=["TWD_SGP_deg", "TWS_SGP_km_h_1"])
    wind_rose_manual(ax, d["TWD_SGP_deg"].values, d["TWS_SGP_km_h_1"].values)
    ax.set_title(venue)
plt.suptitle("Wind rose by venue")
plt.tight_layout()
plt.show()

# Mark vs boat TWS — Bermuda Race 3
race_key = ("Bermuda", "Race_3")
if race_key in marks:
    mk = marks[race_key]
    wg = mk[mk["MARK"].str.startswith("WG", na=False)].groupby("DATETIME")["TWS_km_h_1"].mean()
    boats = df_all[(df_all["venue"] == "Bermuda") & (df_all["race_label"] == "Race_3")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wg.index, y=wg.values, name="Mark WG mean TWS", line=dict(width=3)))
    for team in sorted(boats["team"].unique()):
        t = boats[boats["team"] == team]["TWS_SGP_km_h_1"].dropna()
        fig.add_trace(go.Scatter(x=t.index, y=t.values, name=team, opacity=0.5))
    apply_plotly(fig.update_layout(title="Bermuda Race 3: mark vs boat TWS")).show()

# TWD heatmap by race and minute
df_tmp = df_racing.reset_index().merge(
    meta_all[["venue", "race_label", "race_start_utc"]], on=["venue", "race_label"], how="left"
)
df_tmp["race_start_utc"] = pd.to_datetime(df_tmp["race_start_utc"], utc=True)
df_tmp["minute"] = ((df_tmp["DATETIME"] - df_tmp["race_start_utc"]).dt.total_seconds() // 60).astype(int)
pivot = df_tmp.pivot_table(index="race_label", columns="minute", values="TWD_SGP_deg", aggfunc="mean")
plt.figure(figsize=(16, 6))
sns.heatmap(pivot, cmap="hsv", center=180)
plt.title("TWD by race and minute of race")
plt.show()

fig_hist = px.histogram(
    df_racing[df_racing["venue"] == "Bermuda"], x="TWA_SGP_deg", color="team",
    barmode="overlay", opacity=0.5, nbins=40, title="TWA distribution — Bermuda"
)
apply_plotly(fig_hist).show()
'''))

cells.append(code(r'''## Section 7 — Foiling State Detection
# Foiling flag from ride height + speed; heatmaps and time series
df_racing["foiling"] = (
    (df_racing["LENGTH_RH_P_mm"] > 100) &
    (df_racing["LENGTH_RH_S_mm"] > 100) &
    (df_racing["LENGTH_RH_BOW_mm"] > 100) &
    (df_racing["BOAT_SPEED_km_h_1"] > 40)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, venue in zip(axes, VENUES):
    sub = df_racing[df_racing["venue"] == venue].groupby(["race_label", "team"])["foiling"].mean().unstack("team") * 100
    sns.heatmap(sub, ax=ax, cmap="Greens", annot=False)
    ax.set_title(f"Foiling % — {venue}")
plt.tight_layout()
plt.show()

# Ride height time series GBR Bermuda Race 1
boat = df_all[(df_all["venue"] == "Bermuda") & (df_all["race_label"] == "Race_1") & (df_all["team"] == "GBR")].copy()
boat["foiling"] = (
    (boat["LENGTH_RH_P_mm"] > 100) & (boat["LENGTH_RH_S_mm"] > 100) &
    (boat["LENGTH_RH_BOW_mm"] > 100) & (boat["BOAT_SPEED_km_h_1"] > 40)
)
fig = go.Figure()
for col, name in [("LENGTH_RH_P_mm", "Port"), ("LENGTH_RH_S_mm", "Stbd"), ("LENGTH_RH_BOW_mm", "Bow")]:
    fig.add_trace(go.Scatter(x=boat.index, y=boat[col], name=name))
apply_plotly(fig.update_layout(title="GBR Bermuda R1 ride heights")).show()

# 3D scatter downsampled
d3 = df_racing[df_racing["venue"] == "Bermuda"].dropna(
    subset=["LENGTH_RH_P_mm", "LENGTH_RH_S_mm", "BOAT_SPEED_km_h_1", "PITCH_deg"]
).sample(min(5000, len(df_racing)), random_state=RANDOM_STATE)
fig3d = px.scatter_3d(d3, x="LENGTH_RH_P_mm", y="LENGTH_RH_S_mm", z="BOAT_SPEED_km_h_1", color="PITCH_deg",
                      opacity=0.5, title="Ride height vs speed (Bermuda)")
apply_plotly(fig3d).show()
'''))

cells.append(code(r'''## Section 8 — Wing & Foil Control Surface Analysis
# Wing rot heatmap, parallel coordinates, scatter matrix
race_sub = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == "Race_1") &
                     (df_racing["TRK_LEG_NUM_unk"] == 2)]
if len(race_sub) > 0:
    wing_rows = []
    for team in sorted(race_sub["team"].unique()):
        t = race_sub[race_sub["team"] == team].sort_index().reset_index()
        t["t_sec"] = (t["DATETIME"] - t["DATETIME"].min()).dt.total_seconds().astype(int)
        for _, row in t.iterrows():
            wing_rows.append({"team": team, "t_sec": row["t_sec"], "wing": row["ANGLE_WING_ROT_deg"]})
    wing_pivot = pd.DataFrame(wing_rows).pivot_table(index="team", columns="t_sec", values="wing", aggfunc="mean")
    plt.figure(figsize=(16, 6))
    sns.heatmap(wing_pivot.iloc[:, :min(200, wing_pivot.shape[1])], cmap="coolwarm")
    plt.title("Wing rotation over time — Bermuda R1 leg 2")
    plt.show()

ca_cols = [f"ANGLE_CA{i}_deg" for i in range(1, 7)]
pcols = ca_cols + ["BOAT_SPEED_km_h_1", "VMG_km_h_1"]
leg2 = df_racing[(df_racing["TRK_LEG_NUM_unk"] == 2)].dropna(subset=pcols + ["foiling"]).sample(
    min(2000, len(df_racing)), random_state=RANDOM_STATE)
fig_par = px.parallel_coordinates(leg2, dimensions=pcols, color="foiling",
                                color_continuous_scale="Viridis", title="Camber + speed (leg 2)")
apply_plotly(fig_par).show()

scatter_cols = ["ANGLE_DB_CANT_P_deg", "ANGLE_DB_CANT_S_deg", "HEEL_deg", "PITCH_deg", "RATE_YAW_deg_s_1"]
fig_sm = px.scatter_matrix(
    df_racing.dropna(subset=scatter_cols).sample(min(3000, len(df_racing)), random_state=RANDOM_STATE),
    dimensions=scatter_cols, color="team", opacity=0.1, title="DB cant vs platform attitudes"
)
apply_plotly(fig_sm.update_layout(height=900)).show()
'''))

cells.append(code(r'''## Section 9 — Race Rank Dynamics
# Rank timelines, DTL heatmap, overtake counts
bermuda_races = sorted(df_racing[df_racing["venue"] == "Bermuda"]["race_label"].unique())
fig = make_subplots(rows=2, cols=4, subplot_titles=bermuda_races[:8])
for i, race in enumerate(bermuda_races[:8]):
    r, c = i // 4 + 1, i % 4 + 1
    sub = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == race)]
    for team in sub["team"].unique():
        t = sub[sub["team"] == team].sort_index()
        fig.add_trace(go.Scatter(x=t["TIME_RACE_s"], y=t["TRK_RACE_RANK_unk"], name=team,
                                 legendgroup=team, showlegend=(i == 0)), row=r, col=c)
apply_plotly(fig.update_layout(height=800, title="Rank vs time — Bermuda")).show()

r1 = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == "Race_1")].reset_index()
pivot_dtl = r1.pivot_table(index="DATETIME", columns="team", values="PC_DTL_m")
fig_hm = px.imshow(pivot_dtl.T, aspect="auto", color_continuous_scale="RdYlGn_r",
                   title="PC_DTL_m — Bermuda Race 1")
apply_plotly(fig_hm).show()

def count_overtakes(g):
    g = g.sort_index()
    rank = g["TRK_RACE_RANK_unk"]
    return (rank.diff() < -1).sum()

ot = df_racing.groupby(["venue", "race_label", "team"]).apply(count_overtakes).reset_index(name="overtakes")
fig_ot = px.bar(ot.groupby(["venue", "team"])["overtakes"].sum().reset_index(),
                x="team", y="overtakes", color="venue", barmode="group", title="Overtake events by team")
apply_plotly(fig_ot).show()
'''))

cells.append(code(r'''## Section 10 — Pre-Start Analysis
df_prestart = df_all[df_all["TRK_BOAT_RACE_STATUS_unk"] == 1].copy()
meta_ts = meta_all.copy()
meta_ts["race_start_utc"] = pd.to_datetime(meta_ts["race_start_utc"], utc=True)

offsets = [-60, -30, -10, 0]
fig = go.Figure()
for off in offsets:
    for _, row in meta_ts[meta_ts["venue"] == "Bermuda"].head(3).iterrows():
        t0 = row["race_start_utc"] + pd.Timedelta(seconds=off)
        snap = df_prestart[(df_prestart["venue"] == row["venue"]) &
                           (df_prestart["race_label"] == row["race_label"])]
        near = snap.loc[(snap.index >= t0 - pd.Timedelta(seconds=1)) &
                        (snap.index <= t0 + pd.Timedelta(seconds=1))]
        if len(near):
            fig.add_trace(go.Scatter(x=near["LONGITUDE_GPS_unk"], y=near["LATITUDE_GPS_unk"],
                                     mode="markers", name=f"{row['race_label']} T{off}s",
                                     marker=dict(size=near["BOAT_SPEED_km_h_1"] / 5)))
apply_plotly(fig.update_layout(title="Pre-start positions (sample races)")).show()

fig_tts = px.histogram(df_prestart[df_prestart["venue"] == "Bermuda"], x="PC_TTS_s", color="team",
                       nbins=40, title="Time to start — Bermuda pre-start")
apply_plotly(fig_tts).show()

last60 = df_prestart.reset_index().merge(
    meta_ts[["venue", "race_label", "race_start_utc"]], on=["venue", "race_label"]
)
last60["secs_to_start"] = (last60["race_start_utc"] - last60["DATETIME"]).dt.total_seconds()
last60 = last60[(last60["secs_to_start"] >= 0) & (last60["secs_to_start"] <= 60)]
fig_box = px.box(last60, x="team", y="GPS_SOG_km_h_1", title="Pre-start SOG last 60s — Bermuda")
apply_plotly(fig_box).show()
'''))

cells.append(code(r'''## Section 11 — Leg-by-Leg Performance Decomposition
leg_agg = df_racing.groupby(["venue", "race_label", "team", "TRK_LEG_NUM_unk"]).agg(
    mean_speed=("BOAT_SPEED_km_h_1", "mean"),
    mean_vmg=("VMG_km_h_1", "mean"),
    mean_twa=("TWA_SGP_deg", "mean"),
    foiling_pct=("foiling", "mean"),
    heel_std=("HEEL_deg", "std"),
    mean_wing=("ANGLE_WING_ROT_deg", "mean"),
).reset_index()

metrics = ["mean_speed", "mean_vmg", "foiling_pct", "heel_std", "mean_wing"]
legs = sorted(leg_agg["TRK_LEG_NUM_unk"].dropna().unique())[:6]
fig = make_subplots(rows=2, cols=3, specs=[[{"type": "polar"}] * 3] * 2,
                    subplot_titles=[f"Leg {int(l)}" for l in legs[:6]])
for i, leg in enumerate(legs[:6]):
    r, c = i // 3 + 1, i % 3 + 1
    d = leg_agg[leg_agg["TRK_LEG_NUM_unk"] == leg]
    for team in d["team"].unique()[:8]:
        row = d[d["team"] == team]
        vals = []
        for m in metrics:
            v = row[m].values[0] if len(row) else 0
            vals.append(v)
        vals_norm = np.array(vals)
        if vals_norm.max() > vals_norm.min():
            vals_norm = (vals_norm - vals_norm.min()) / (vals_norm.max() - vals_norm.min())
        fig.add_trace(go.Scatterpolar(r=list(vals_norm) + [vals_norm[0]],
                                      theta=metrics + [metrics[0]], name=team), row=r, col=c)
apply_plotly(fig.update_layout(height=700, title="Leg radar charts (normalized)")).show()

mat = leg_agg.groupby("team")[metrics].mean()
g = sns.clustermap(mat, z_score=0, cmap="vlag", figsize=(10, 10), method="ward")
plt.suptitle("Team performance clustermap")
plt.show()

leg_dur = df_racing.groupby(["venue", "race_label", "team", "TRK_LEG_NUM_unk"])["TIME_RACE_LEG_s"].max().reset_index()
fig_strip = px.strip(leg_dur, x="TRK_LEG_NUM_unk", y="TIME_RACE_LEG_s", color="team", facet_col="venue",
                     title="Leg duration by team")
apply_plotly(fig_strip).show()
'''))

cells.append(code(r'''## Section 12 — Correlation & Feature Importance
from sklearn.feature_selection import mutual_info_regression

drop_cols = {"venue", "race_label", "team", "TEAM", "BOAT", "TRK_BOAT_RACE_STATUS_unk",
             "TRK_RACE_NUM_unk", "PC_BEACON_NUMBER_unk", "BROADCAST_MODE_unk"}
num_racing = df_racing.select_dtypes(include=[np.number]).drop(
    columns=[c for c in df_racing.columns if c in drop_cols], errors="ignore"
)
corr = num_racing.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(14, 12))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True)
plt.title("Pearson correlation (racing numeric)")
plt.tight_layout()
plt.show()

pairs = []
cols = corr.columns
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if j > i:
            pairs.append((a, b, corr.loc[a, b]))
top10 = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
print("Top 10 correlations:", top10)

X = num_racing.drop(columns=["BOAT_SPEED_km_h_1"], errors="ignore").fillna(0)
y = num_racing["BOAT_SPEED_km_h_1"].fillna(0)
mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({"feature": X.columns, "mi": mi}).sort_values("mi", ascending=False).head(25)
groups = []
for f in mi_df["feature"]:
    if "TWS" in f or "TWA" in f or "TWD" in f or "AW" in f:
        groups.append("wind")
    elif "DB" in f or "RH" in f or "RUD" in f:
        groups.append("foil")
    elif "CA" in f or "WING" in f or "JIB" in f:
        groups.append("wing")
    else:
        groups.append("platform")
mi_df["group"] = groups
fig_mi = px.bar(mi_df, x="mi", y="feature", color="group", orientation="h", title="MI vs boat speed")
apply_plotly(fig_mi).show()

# Lagged cross-correlation Bermuda Race 1
sub = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == "Race_1")]
fig_lag = go.Figure()
lags = list(range(-30, 31))
for team in sorted(sub["team"].unique())[:6]:
    t = sub[sub["team"] == team][["TWA_SGP_deg", "BOAT_SPEED_km_h_1"]].dropna()
    if len(t) < 60:
        continue
    twa = (t["TWA_SGP_deg"] - t["TWA_SGP_deg"].mean()).values
    spd = (t["BOAT_SPEED_km_h_1"] - t["BOAT_SPEED_km_h_1"].mean()).values
    ccs = []
    for lag in lags:
        if lag >= 0:
            a, b = twa[lag:], spd[: len(spd) - lag]
        else:
            a, b = twa[: len(twa) + lag], spd[-lag:]
        ccs.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 else 0.0)
    best = lags[int(np.nanargmax(np.abs(ccs)))]
    fig_lag.add_trace(go.Scatter(x=lags, y=ccs, name=f"{team} (max@{best})"))
apply_plotly(fig_lag.update_layout(title="TWA vs speed cross-correlation")).show()
'''))

cells.append(code(r'''%%time
## Section 13 — Anomaly Detection (PyTorch Autoencoder)
import torch
import torch.nn as nn

AE_FEATURES = [
    "BOAT_SPEED_km_h_1", "VMG_km_h_1", "TWA_SGP_deg", "TWS_SGP_km_h_1",
    "LENGTH_RH_P_mm", "LENGTH_RH_S_mm", "LENGTH_RH_BOW_mm",
    "HEEL_deg", "PITCH_deg", "RATE_YAW_deg_s_1",
    "ANGLE_WING_ROT_deg", "ANGLE_WING_TWIST_deg",
] + [f"ANGLE_CA{i}_deg" for i in range(1, 7)] + ["ANGLE_DB_CANT_P_deg", "ANGLE_DB_CANT_S_deg"]

train_df = df_racing[df_racing["venue"] == "Bermuda"].dropna(subset=AE_FEATURES)
X = train_df[AE_FEATURES].values.astype(np.float32)
X_mean, X_std = X.mean(0), X.std(0) + 1e-6
Xn = (X - X_mean) / X_std

class Autoencoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, n_in))
    def forward(self, x):
        return self.dec(self.enc(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(len(AE_FEATURES)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
Xt = torch.tensor(Xn, device=device)
losses = []
for epoch in range(50):
    model.train()
    opt.zero_grad()
    recon = model(Xt)
    loss = loss_fn(recon, Xt)
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Autoencoder training loss")
plt.xlabel("epoch")
plt.show()

model.eval()
with torch.no_grad():
    err = ((model(Xt) - Xt) ** 2).mean(dim=1).cpu().numpy()
train_df = train_df.copy()
train_df["recon_error"] = err
thresh = np.percentile(err, 95)
train_df["anomaly"] = err > thresh

r1 = train_df[(train_df["race_label"] == "Race_1")].reset_index()
fig_map = px.scatter_mapbox(
    r1, lat="LATITUDE_GPS_unk", lon="LONGITUDE_GPS_unk", color="anomaly",
    color_discrete_map={True: "red", False: "blue"}, mapbox_style="carto-darkmatter", zoom=12,
    title="Bermuda R1 anomalies (red)"
)
apply_plotly(fig_map).show()

display(train_df.nlargest(20, "recon_error")[AE_FEATURES + ["recon_error", "team", "race_label"]].style.background_gradient(subset=["recon_error"]))
'''))

cells.append(code(r'''%%time
## Section 14 — Dimensionality Reduction & Clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

Xdf = df_racing[df_racing["venue"] == "Bermuda"].dropna(subset=AE_FEATURES).sample(
    min(15000, len(df_racing)), random_state=RANDOM_STATE
)
Xs = StandardScaler().fit_transform(Xdf[AE_FEATURES])

pca = PCA(n_components=10, random_state=RANDOM_STATE)
Xp = pca.fit_transform(Xs)
fig_scree = px.bar(x=list(range(1, 11)), y=pca.explained_variance_ratio_, title="PCA scree")
apply_plotly(fig_scree).show()

for color_col, title in [("team", "team"), ("foiling", "foiling"), ("TRK_LEG_NUM_unk", "leg")]:
    fig_p = px.scatter(x=Xp[:, 0], y=Xp[:, 1], color=Xdf[color_col].astype(str), title=f"PCA — {title}",
                       labels={"x": "PC1", "y": "PC2"})
    apply_plotly(fig_p).show()

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
Xu = reducer.fit_transform(Xs)
fig_u = make_subplots(rows=1, cols=3, subplot_titles=["team", "rank", "leg"])
for i, col in enumerate(["team", "TRK_RACE_RANK_unk", "TRK_LEG_NUM_unk"]):
    fig_u.add_trace(go.Scatter(x=Xu[:, 0], y=Xu[:, 1], mode="markers",
                               marker=dict(color=pd.Categorical(Xdf[col]).codes, colorscale="Viridis", showscale=False),
                               text=Xdf[col].astype(str)), row=1, col=i+1)
apply_plotly(fig_u.update_layout(height=400, title="UMAP embeddings")).show()

km = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
Xdf["cluster"] = km.fit_predict(Xu)
summary = Xdf.groupby("cluster")[AE_FEATURES].mean()
display(summary.style.format("{:.2f}"))

fig_cl = px.scatter_mapbox(
    Xdf.dropna(subset=["LATITUDE_GPS_unk", "LONGITUDE_GPS_unk"]),
    lat="LATITUDE_GPS_unk", lon="LONGITUDE_GPS_unk", color="cluster",
    mapbox_style="carto-darkmatter", zoom=11, title="GPS coloured by UMAP cluster"
)
apply_plotly(fig_cl).show()
'''))

cells.append(code(r'''## Section 15 — Venue Comparison: Bermuda vs. Halifax
metrics_cmp = ["BOAT_SPEED_km_h_1", "VMG_km_h_1", "TWS_SGP_km_h_1", "HEEL_deg", "LENGTH_RH_P_mm"]
for m in metrics_cmp:
    fig = px.violin(df_racing, x="venue", y=m, color="venue", box=True, points="outliers", title=m)
    apply_plotly(fig).show()

fig_foil = px.violin(df_racing, x="venue", y="foiling", color="venue", title="Foiling state (0/1)")
apply_plotly(fig_foil).show()

perf = df_racing.groupby(["venue", "race_label", "team"])["BOAT_SPEED_km_h_1"].mean().reset_index()
pivot_perf = perf.pivot_table(index=["venue", "team"], columns="race_label", values="BOAT_SPEED_km_h_1")
display(pivot_perf.style.background_gradient(cmap="RdYlGn", axis=None))

fig_wind = px.scatter(meta_all, x="avg_tws_km_h", y="avg_twd_deg", size="num_boats", color="venue",
                      text="race_label", title="Race wind regimes")
fig_wind.update_traces(textposition="top center")
apply_plotly(fig_wind).show()
'''))

cells.append(code(r'''## Section 16 — Penalty & Rule Events
pen = df_racing.groupby(["venue", "race_label", "team"])["TRK_PENALTY_COUNT_unk"].max().reset_index()
for venue in VENUES:
    sub = pen[pen["venue"] == venue].pivot(index="team", columns="race_label", values="TRK_PENALTY_COUNT_unk")
    plt.figure(figsize=(12, 5))
    sns.heatmap(sub.fillna(0), annot=True, fmt=".0f", cmap="Reds")
    plt.title(f"Max penalty count — {venue}")
    plt.show()

final_rank = df_racing.sort_index().groupby(["venue", "race_label", "team"]).last().reset_index()
fig_p = px.scatter(final_rank, x="TRK_PENALTY_DISTANCE_m", y="TRK_RACE_RANK_unk", color="team",
                   facet_col="venue", title="Penalty distance vs final rank")
apply_plotly(fig_p).show()

ocs = df_all[df_all["TRK_BOAT_RACE_STATUS_unk"] == 7].reset_index()
ocs_tbl = ocs.groupby(["venue", "race_label", "team"]).agg(
    ocs_time=("DATETIME", "first"),
    lat=("LATITUDE_GPS_unk", "first"),
    lon=("LONGITUDE_GPS_unk", "first"),
).reset_index()
display(ocs_tbl if len(ocs_tbl) else pd.DataFrame({"note": ["No OCS rows found"]}))
'''))

cells.append(code(r'''%%time
## Section 17 — Time-Series Distance & Similarity (dtaidistance)
from dtaidistance import dtw

sub = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == "Race_1")]
series = {}
for team in sorted(sub["team"].unique()):
    s = sub[sub["team"] == team]["BOAT_SPEED_km_h_1"].dropna().values.astype(np.double)
    if len(s) > 30:
        series[team] = s

teams = list(series.keys())
n = len(teams)
dm = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = dtw.distance(series[teams[i]], series[teams[j]])
        dm[i, j] = dm[j, i] = d

dm_df = pd.DataFrame(dm, index=teams, columns=teams)
g = sns.clustermap(dm_df, cmap="mako", figsize=(10, 10))
plt.suptitle("DTW distance — BOAT_SPEED Bermuda R1")
plt.show()

# Most similar / dissimilar pairs
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((teams[i], teams[j], dm[i, j]))
pairs.sort(key=lambda x: x[2])
print("Most similar:", pairs[:2])
print("Most dissimilar:", pairs[-2:])

fig_dtw = go.Figure()
for t1, t2, _ in pairs[:2] + pairs[-2:]:
    fig_dtw.add_trace(go.Scatter(y=series[t1], name=t1, mode="lines"))
    fig_dtw.add_trace(go.Scatter(y=series[t2], name=t2, mode="lines", line=dict(dash="dash")))
apply_plotly(fig_dtw.update_layout(title="DTW pair speed overlays")).show()
'''))

cells.append(code(r'''## Section 18 — Dashboard Summary (Plotly make_subplots)
# 4x3 executive dashboard — reuse aggregates from prior sections
teams = df_racing.groupby("team").agg(
    speed=("BOAT_SPEED_km_h_1", "mean"),
    foil=("foiling", "mean"),
    vmg=("VMG_km_h_1", "mean"),
).reset_index().sort_values("speed", ascending=False)

fig = make_subplots(rows=4, cols=3, subplot_titles=[
    "Mean speed", "Foiling %", "Mean VMG",
    "Speed vs TWA", "Rank R1", "Wind rose",
    "Ride height", "Penalties", "Pre-start SOG",
    "UMAP team", "PCA foiling", "Recon error",
], specs=[[{}, {}, {}], [{"type": "polar"}, {}, {"type": "polar"}],
          [{}, {}, {}], [{}, {}, {}]], vertical_spacing=0.08)

fig.add_trace(go.Bar(x=teams["team"], y=teams["speed"]), row=1, col=1)
fig.add_trace(go.Bar(x=teams["team"], y=teams["foil"] * 100), row=1, col=2)
fig.add_trace(go.Bar(x=teams["team"], y=teams["vmg"]), row=1, col=3)

d_pol = df_racing[df_racing["venue"] == "Bermuda"].dropna(subset=["TWA_SGP_deg", "BOAT_SPEED_km_h_1"]).sample(3000, random_state=RANDOM_STATE)
fig.add_trace(go.Scatterpolar(r=d_pol["BOAT_SPEED_km_h_1"], theta=d_pol["TWA_SGP_deg"], mode="markers"), row=2, col=1)

r1 = df_racing[(df_racing["venue"] == "Bermuda") & (df_racing["race_label"] == "Race_1")]
for team in r1["team"].unique()[:6]:
    t = r1[r1["team"] == team]
    fig.add_trace(go.Scatter(x=t["TIME_RACE_s"], y=t["TRK_RACE_RANK_unk"], name=team), row=2, col=2)

berm = df_racing[df_racing["venue"] == "Bermuda"].dropna(subset=["TWD_SGP_deg", "TWS_SGP_km_h_1"])
sectors = np.linspace(0, 360, 17)
hist, _ = np.histogram(berm["TWD_SGP_deg"] % 360, bins=sectors, weights=berm["TWS_SGP_km_h_1"])
fig.add_trace(go.Barpolar(r=hist, theta=(sectors[:-1] + sectors[1:]) / 2), row=2, col=3)

fig.add_trace(go.Box(y=df_racing["LENGTH_RH_P_mm"], x=df_racing["team"], name="RH"), row=3, col=1)
pen_pivot = pen[pen["venue"] == "Bermuda"].pivot(index="team", columns="race_label", values="TRK_PENALTY_COUNT_unk").fillna(0)
if len(pen_pivot):
    fig.add_trace(go.Heatmap(z=pen_pivot.values, x=list(pen_pivot.columns), y=list(pen_pivot.index)), row=3, col=2)
if "last60" in dir() and len(last60):
    fig.add_trace(go.Violin(y=last60["GPS_SOG_km_h_1"], x=last60["team"]), row=3, col=3)

if "Xu" in dir() and "Xdf" in dir():
    fig.add_trace(go.Scatter(x=Xu[:, 0], y=Xu[:, 1], mode="markers", marker=dict(color=pd.Categorical(Xdf["team"]).codes)), row=4, col=1)
    fig.add_trace(go.Scatter(x=Xp[:, 0], y=Xp[:, 1], mode="markers", marker=dict(color=Xdf["foiling"].astype(int))), row=4, col=2)
if "train_df" in dir():
    fig.add_trace(go.Histogram(x=train_df["recon_error"], nbinsx=40), row=4, col=3)

apply_plotly(fig.update_layout(height=1600, title="SailGP EDA Summary Dashboard")).write_html("sailgp_dashboard.html")
display(HTML('<a href="sailgp_dashboard.html">sailgp_dashboard.html</a>'))
'''))

cells.append(code(r'''## Output Files Summary
from pathlib import Path

outputs = [
    "sailgp_profile.html",
    "map_bermuda.html",
    "map_halifax.html",
    "sailgp_dashboard.html",
    "dataExploration/dataExploration.ipynb",
]
print("Saved artefacts:")
for p in outputs:
    path = Path(p)
    if path.exists():
        print(f"  ✓ {p} — {path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"  ✗ {p} — not found")
'''))

# Split ## Section headers into markdown cells
processed = [cells[0]]
for cell in cells[1:]:
    if cell["cell_type"] != "code":
        processed.append(cell)
        continue
    src = cell["source"]
    if not src or not src[0].startswith("## Section"):
        processed.append(cell)
        continue
    header = src[0].strip()
    desc = ""
    new_src = []
    for line in src[1:]:
        if line.startswith("%%time"):
            new_src.append(line)
        elif (
            not desc
            and line.startswith("# ")
            and not line.startswith("%")
            and "pip install" not in line
            and line.strip() != "#"
        ):
            desc = line[2:].strip()
        else:
            new_src.append(line)
    processed.append(md(f"{header}\n\n{desc}\n"))
    processed.append({**cell, "source": new_src})

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": processed,
}

out = Path(__file__).parent / "dataExploration.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} with {len(processed)} cells")
