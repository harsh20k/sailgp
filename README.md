# SailGP

Ocean of Data Challenge — **Foil Forward**. Telemetry analysis for F50 catamaran races (Bermuda 2026, Halifax 2024).

## Stream

**#2: On the Water** — boat telemetry & performance science

## Deadline

June 10, 2026 · info@deepsense.ca

## Stack

Python · Pandas · FastAPI · HTML dashboard · multi-agent analysis

## Quick start

```bash
cd /path/to/sailgp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build web data snapshot + run agents once
python scripts/build_snapshot.py
python scripts/run_agents.py --force

# Web UI + API (open http://127.0.0.1:8000)
uvicorn api.main:app --reload --app-dir .
```

## Project layout

| Path | Purpose |
|------|---------|
| `DataChallenge_Export/` | SailGP boat CSVs, marks, metadata |
| `challengeDetails/` | Challenge docs, data summary, project ideas |
| `dataExploration/` | Jupyter EDA notebook |
| `web/` | HTML dashboard (`index.html`, `app.js`) |
| `api/main.py` | FastAPI — `/api/snapshot`, `/api/agents/run` |
| `sailgp_analysis/` | Data loader, analytics, multi-agent system |
| `scripts/run_agents.py` | One-shot or `--watch` continuous analysis |

## Multi-agent system

Three agents run in sequence (or on a timer when new CSVs appear):

1. **Ingest** — SHA256 scan of `DataChallenge_Export/**/boats/**/*.csv`
2. **Analytics** — Rebuilds `analysis_output/metrics.json` and `web/data/snapshot.json`
3. **Report** — Writes `analysis_output/latest_report.md` and `insights.json`

```bash
# Continuous watch (every 60s)
python scripts/run_agents.py --watch --interval 60
```

Outputs: `analysis_output/agent_state.json`, `agent_status.json`, `run_log.jsonl`

## Deep agentic research (3 challenge streams)

Three **stream research agents** (Above / On / Around) + **coordinator** with shared memory for multi-day convergence.

```bash
python scripts/run_deep_agents.py              # one cycle
python scripts/run_deep_agents.py --watch --interval 600
python scripts/run_deep_agents.py --until-converged 5
```

**Website:** [web/deep-agents.html](web/deep-agents.html) — architecture, live hypothesis board, API hooks.

Memory: `analysis_output/deep_research/shared_memory.json`

## EDA notebook

```bash
jupyter notebook dataExploration/dataExploration.ipynb
```

Generates: `sailgp_profile.html`, `map_bermuda.html`, `map_halifax.html`, `sailgp_dashboard.html` (embedded in the web UI when present).
