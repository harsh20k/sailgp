const API = ""; // same origin when served via FastAPI

let snapshot = null;
let teamColors = {};

async function fetchJson(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

async function loadSnapshot() {
  snapshot = await fetchJson("/api/snapshot");
  teamColors = snapshot.team_colors || {};
  renderOverview();
  populateSelectors();
  renderRaceView();
  renderWindGateChart();
  renderArtifacts();
  document.getElementById("generated-at").textContent =
    `Snapshot: ${snapshot.generated_at || "—"}`;
}

function renderOverview() {
  const el = document.getElementById("kpi-cards");
  el.innerHTML = "";
  let totalRows = 0;
  let totalRaces = 0;
  const venues = Object.keys(snapshot.venues || {});
  venues.forEach((v) => {
    totalRows += snapshot.venues[v].total_rows || 0;
    totalRaces += (snapshot.venues[v].races || []).length;
  });
  const kpis = [
    { label: "Venues", value: venues.length },
    { label: "Races", value: totalRaces },
    { label: "Telemetry rows", value: totalRows.toLocaleString() },
    { label: "Teams", value: Object.keys(teamColors).length },
  ];
  const status = snapshot.status_counts || {};
  const racing = status["2"] || 0;
  kpis.push({ label: "Racing rows", value: Number(racing).toLocaleString() });
  kpis.forEach((k) => {
    el.innerHTML += `<div class="kpi"><div class="value">${k.value}</div><div class="label">${k.label}</div></div>`;
  });
}

function populateSelectors() {
  const vSel = document.getElementById("sel-venue");
  const venues = Object.keys(snapshot.venues || {});
  vSel.innerHTML = venues.map((v) => `<option value="${v}">${v}</option>`).join("");
  vSel.onchange = () => { populateRaces(); renderRaceView(); };
  populateRaces();
  document.getElementById("sel-race").onchange = () => { populateTeams(); renderRaceView(); };
  document.getElementById("sel-team").onchange = renderRaceView;
}

function populateRaces() {
  const venue = document.getElementById("sel-venue").value;
  const races = (snapshot.venues[venue]?.races || []).map((r) => r.race_label);
  const rSel = document.getElementById("sel-race");
  rSel.innerHTML = races.map((r) => `<option value="${r}">${r}</option>`).join("");
  populateTeams();
}

function populateTeams() {
  const venue = document.getElementById("sel-venue").value;
  const race = document.getElementById("sel-race").value;
  const raceData = (snapshot.venues[venue]?.races || []).find((r) => r.race_label === race);
  const teams = (raceData?.teams || []).map((t) => t.team);
  const tSel = document.getElementById("sel-team");
  tSel.innerHTML = teams.map((t) => `<option value="${t}">${t}</option>`).join("");
}

function getRaceData() {
  const venue = document.getElementById("sel-venue").value;
  const race = document.getElementById("sel-race").value;
  return (snapshot.venues[venue]?.races || []).find((r) => r.race_label === race);
}

function renderRaceView() {
  const raceData = getRaceData();
  if (!raceData) return;
  const meta = raceData.metadata || {};
  document.getElementById("race-meta").innerHTML = `
    <strong>${meta.venue || document.getElementById("sel-venue").value} · ${raceData.race_label}</strong><br>
    Wind: ${meta.avg_tws_km_h ?? "—"} km/h · TWD ${meta.avg_twd_deg ?? "—"}° ·
    Boats: ${meta.num_boats ?? raceData.teams.length} ·
    Start: ${meta.race_start_utc ?? "—"}
  `;
  const rows = raceData.teams
    .map(
      (t) => `<tr>
        <td>${t.team}</td>
        <td>${t.mean_speed?.toFixed(1) ?? "—"}</td>
        <td>${t.mean_vmg?.toFixed(1) ?? "—"}</td>
        <td>${t.foiling_pct?.toFixed(0) ?? "—"}%</td>
        <td>${t.rows?.toLocaleString() ?? "—"}</td>
      </tr>`
    )
    .join("");
  document.getElementById("team-table-wrap").innerHTML = `
    <table class="data"><thead><tr>
      <th>Team</th><th>Mean speed</th><th>Mean VMG</th><th>Foiling %</th><th>Rows</th>
    </tr></thead><tbody>${rows}</tbody></table>`;

  renderSpeedChart();
  renderTeamsBar();
}

function renderSpeedChart() {
  const venue = document.getElementById("sel-venue").value;
  const race = document.getElementById("sel-race").value;
  const team = document.getElementById("sel-team").value;
  const raceData = getRaceData();
  const t = raceData?.teams.find((x) => x.team === team);
  const series = t?.speed_series || [];
  const trace = {
    x: series.map((p) => p.t),
    y: series.map((p) => p.speed),
    type: "scatter",
    mode: "lines",
    name: `${team} speed`,
    line: { color: teamColors[team] || "#636EFA" },
  };
  const rankTrace = {
    x: series.map((p) => p.t),
    y: series.map((p) => p.rank),
    type: "scatter",
    mode: "lines",
    name: "Rank",
    yaxis: "y2",
    line: { color: "#ffa15a", dash: "dot" },
  };
  Plotly.newPlot(
    "chart-speed",
    [trace, rankTrace],
    {
      template: "plotly_dark",
      title: `${venue} ${race} — ${team}`,
      xaxis: { title: "Time (UTC)" },
      yaxis: { title: "Speed (km/h)" },
      yaxis2: { title: "Rank", overlaying: "y", side: "right", autorange: "reversed" },
      font: { family: "IBM Plex Mono, monospace" },
      margin: { t: 40 },
    },
    { responsive: true }
  );
}

function renderTeamsBar() {
  const raceData = getRaceData();
  const teams = raceData?.teams || [];
  Plotly.newPlot(
    "chart-teams-bar",
    [
      {
        x: teams.map((t) => t.team),
        y: teams.map((t) => t.mean_speed || 0),
        type: "bar",
        marker: { color: teams.map((t) => teamColors[t.team] || "#636EFA") },
        name: "Mean speed",
      },
    ],
    {
      template: "plotly_dark",
      title: "Mean racing speed by team",
      yaxis: { title: "km/h" },
      font: { family: "IBM Plex Mono, monospace" },
    },
    { responsive: true }
  );
}

function renderWindGateChart() {
  const keys = Object.keys(snapshot.marks_summary || {});
  if (!keys.length) return;
  const labels = keys.map((k) => k.replace("/", " "));
  const wg = keys.map((k) => snapshot.marks_summary[k].wg_mean_tws);
  const lg = keys.map((k) => snapshot.marks_summary[k].lg_mean_tws);
  Plotly.newPlot(
    "chart-wind-gate",
    [
      { x: labels, y: wg, name: "WG TWS", type: "bar" },
      { x: labels, y: lg, name: "LG TWS", type: "bar" },
    ],
    {
      template: "plotly_dark",
      title: "Windward vs leeward gate TWS (marks)",
      barmode: "group",
      font: { family: "IBM Plex Mono, monospace" },
    },
    { responsive: true }
  );
}

function renderArtifacts() {
  const wrap = document.getElementById("artifact-links");
  const arts = snapshot.eda_artifacts || [];
  const defaults = [
    { name: "sailgp_dashboard.html", path: "/artifacts/sailgp_dashboard.html" },
    { name: "map_bermuda.html", path: "/artifacts/map_bermuda.html" },
    { name: "map_halifax.html", path: "/artifacts/map_halifax.html" },
    { name: "sailgp_profile.html", path: "/artifacts/sailgp_profile.html" },
  ];
  const merged = arts.length ? arts : defaults;
  wrap.innerHTML = merged
    .map(
      (a) =>
        `<button type="button" class="link-btn" data-src="${a.path}">${a.name}${a.size_kb ? ` (${a.size_kb} KB)` : ""}</button>`
    )
    .join("");
  wrap.querySelectorAll(".link-btn").forEach((btn) => {
    btn.onclick = () => {
      const src = btn.dataset.src;
      const fw = document.getElementById("artifact-frame-wrap");
      const fr = document.getElementById("artifact-frame");
      fw.classList.remove("hidden");
      fr.src = src;
    };
  });
}

async function loadAgents() {
  try {
    const status = await fetchJson("/api/agents/status");
    const grid = document.getElementById("agent-status");
    const agents = [
      { id: "ingest", title: "Ingest Agent", desc: "Scans CSV hashes for new/changed boat files" },
      { id: "analytics", title: "Analytics Agent", desc: "Rebuilds metrics.json & web snapshot" },
      { id: "report", title: "Report Agent", desc: "Writes latest_report.md & insights" },
    ];
    grid.innerHTML = agents
      .map(
        (a) => `<div class="agent-card ${a.id}">
          <h4>${a.title}</h4>
          <p class="muted">${a.desc}</p>
          <p>Runs: <strong>${status.runs ?? 0}</strong></p>
        </div>`
      )
      .join("");
    document.getElementById("agent-log").textContent = JSON.stringify(
      status.messages || [],
      null,
      2
    );
    const insights = await fetchJson("/api/insights");
    const ul = document.getElementById("insights-list");
    ul.innerHTML = (insights || [])
      .slice(0, 15)
      .map((i) => `<li><code>${i.type}</code> — ${JSON.stringify(i)}</li>`)
      .join("") || "<li class='muted'>No insights yet — run agents.</li>";
  } catch (e) {
    document.getElementById("agent-log").textContent = String(e);
  }
}

document.getElementById("btn-refresh").onclick = async () => {
  await loadSnapshot();
  await loadAgents();
};

document.getElementById("btn-run-agents").onclick = async () => {
  const btn = document.getElementById("btn-run-agents");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    await fetchJson("/api/agents/run?force=true", { method: "POST" });
    await loadSnapshot();
    await loadAgents();
  } catch (e) {
    alert(e.message);
  }
  btn.disabled = false;
  btn.textContent = "Run agents";
};

// Fallback: load static snapshot if API unavailable (file:// or static server)
async function init() {
  try {
    await loadSnapshot();
    await loadAgents();
  } catch {
    try {
      const res = await fetch("data/snapshot.json");
      if (res.ok) {
        snapshot = await res.json();
        teamColors = snapshot.team_colors || {};
        renderOverview();
        populateSelectors();
        renderRaceView();
        renderWindGateChart();
        renderArtifacts();
        document.getElementById("generated-at").textContent =
          "Static snapshot (run API or scripts/build_snapshot.py for live data)";
        document.getElementById("agent-log").textContent =
          "Start API: uvicorn api.main:app --reload";
      }
    } catch (e2) {
      document.getElementById("kpi-cards").innerHTML =
        `<p class="muted">Start server: <code>uvicorn api.main:app --reload</code> then open http://127.0.0.1:8000</p>`;
    }
  }
}

init();
setInterval(loadAgents, 30000);
