# SailGP — Ocean of Data Challenge: Foil Forward

**Stream #2: On the Water** · June 10, 2026 · *Source: SailGP data*

## Summary

I analyzed 1 Hz F50 telemetry from 14 races (Bermuda 2026 + Halifax 2024) to answer one question: what actually separates winning teams from the rest?

The answer is the first 60 seconds. Within 20 seconds of the start gun — before any tactical choices — eventual top-4 boats already fly ~40 cm higher on their foils and travel 15 km/h faster. 11 of 12 sensor gaps survive a wind-speed control, so the early advantage is skill, not a lucky gust.

I traced that gap through a four-step causal chain:

1. **First-minute fingerprint (C8)** — ride height, speed, and wing/foil settings in the opening 60 s already predict finishing position.
2. **Flight quality (C2)** — after removing leg-length effects, how stably the boat stays on its foils explains 86% of the controllable performance gap.
3. **Ghost boat benchmark (A4b)** — a virtual perfect sailor built from the fleet's own polar gives every team a fair "seconds lost" score. Total regret tracks race rank at ρ = 0.915.
4. **Restart time (C6)** — the most fixable failure: slow re-foiling after mark roundings. The slowest team takes 61 s to get flying again vs 2 s for the best, losing ~8 minutes per race.

Coaches get three concrete targets from the telemetry SailGP already collects: start ride height, mid-leg flight stability, and restart speed after turns.

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
