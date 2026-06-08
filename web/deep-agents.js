const API = "";

async function fetchJson(path, opts = {}) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

function statusClass(s) {
  if (s === "converged") return "status-converged";
  if (s === "supported") return "status-supported";
  return "";
}

function renderHypotheses(hypotheses) {
  const el = document.getElementById("hypothesis-board");
  if (!hypotheses?.length) {
    el.innerHTML = "<p class='muted'>No hypotheses yet. Run a research cycle.</p>";
    return;
  }
  el.innerHTML = hypotheses
    .slice(0, 25)
    .map(
      (h) => `
    <div class="hyp-card ${h.stream}">
      <div class="hyp-meta">
        <span>${h.stream}</span>
        <span class="${statusClass(h.status)}">${h.status}</span>
        <span>conf ${(h.confidence * 100).toFixed(0)}%</span>
        <span>novel ${(h.novelty * 100).toFixed(0)}%</span>
        <span>${h.evidence_count} evidence</span>
      </div>
      <p class="stmt">${h.statement}</p>
    </div>`
    )
    .join("");
}

function renderInsights(insights) {
  const el = document.getElementById("insight-board");
  if (!insights?.length) {
    el.innerHTML = "<p class='muted'>Coordinator insights appear after multiple cycles with cross-stream agreement.</p>";
    return;
  }
  el.innerHTML = insights
    .map(
      (i) => `
    <div class="insight-card">
      <h4>${i.title}</h4>
      <p>${i.narrative}</p>
      <div class="insight-scores">
        streams: ${(i.streams || []).join(", ")} ·
        convergence ${(i.convergence * 100).toFixed(0)}% ·
        novelty ${(i.novelty * 100).toFixed(0)}%
      </div>
    </div>`
    )
    .join("");
}

function renderConvergence(conv) {
  const total = conv?.total_hypotheses || 0;
  const converged = conv?.converged_count || 0;
  const pct = total ? Math.min(100, (converged / Math.max(total * 0.15, 1)) * 100) : 0;
  document.getElementById("convergence-fill").style.width = `${pct}%`;
  document.getElementById("convergence-label").textContent =
    `${converged} converged · ${total} hypotheses · ${conv?.runs_completed || 0} cycles completed`;
}

function renderMessages(msgs) {
  document.getElementById("message-bus").textContent = JSON.stringify(msgs || [], null, 2);
}

function renderRunHistory(status) {
  const el = document.getElementById("run-history");
  const cycle = status?.last_cycle;
  if (!cycle) {
    el.innerHTML = "<span class='muted'>—</span>";
    return;
  }
  const streams = Object.entries(cycle.streams || {})
    .map(([k, v]) => `${k}: ${v.findings} findings`)
    .join(" · ");
  el.innerHTML = `
    <span class="run-chip">Run #${cycle.run_id}</span>
    <span class="run-chip">${cycle.started || ""}</span>
    <span class="run-chip">${streams}</span>
    <span class="run-chip">coordinator: ${cycle.coordinator?.new_insights ?? 0} new insights</span>
  `;
  document.getElementById("cycle-meta").textContent = `Last cycle: ${cycle.finished || "—"}`;
}

async function loadDeepState() {
  try {
    const status = await fetchJson("/api/deep/status");
    renderHypotheses(status.hypotheses);
    renderInsights(status.insights);
    renderConvergence(status.convergence);
    renderMessages(status.recent_messages);
    renderRunHistory(status);
  } catch (e) {
    document.getElementById("hypothesis-board").innerHTML =
      `<p class="muted">Start API: uvicorn api.main:app --reload — then run <code>python scripts/run_deep_agents.py</code></p>`;
    console.error(e);
  }
}

document.getElementById("btn-deep-refresh").onclick = loadDeepState;

document.getElementById("btn-deep-cycle").onclick = async () => {
  const btn = document.getElementById("btn-deep-cycle");
  btn.disabled = true;
  btn.textContent = "Running cycle…";
  try {
    await fetchJson("/api/deep/run-cycle", { method: "POST" });
    await loadDeepState();
  } catch (e) {
    alert(e.message);
  }
  btn.disabled = false;
  btn.textContent = "Run one research cycle";
};

loadDeepState();
setInterval(loadDeepState, 45000);
