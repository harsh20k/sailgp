"""FastAPI backend: SailGP data API, static site, agent control."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sailgp_analysis.agents.orchestrator import AnalysisOrchestrator
from sailgp_analysis.analytics import build_snapshot, write_snapshot
from sailgp_analysis.config import DATA_ROOT, OUTPUT_DIR, REPO_ROOT, SNAPSHOT_FILE, WEB_DATA_DIR
from sailgp_analysis.deep_agents.orchestrator import DeepResearchOrchestrator, STATUS_FILE

deep_orchestrator = DeepResearchOrchestrator()

app = FastAPI(title="SailGP Data Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = REPO_ROOT / "web"
orchestrator = AnalysisOrchestrator()


@app.on_event("startup")
def startup_snapshot():
    if not SNAPSHOT_FILE.exists() or DATA_ROOT.exists():
        try:
            write_snapshot()
            orchestrator.run_once(force_rebuild=True)
        except Exception:
            pass


@app.get("/api/health")
def health():
    return {"status": "ok", "data_root": str(DATA_ROOT), "data_exists": DATA_ROOT.exists()}


@app.get("/api/snapshot")
def get_snapshot():
    if SNAPSHOT_FILE.exists():
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    return build_snapshot()


@app.get("/api/metrics")
def get_metrics():
    p = OUTPUT_DIR / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return get_snapshot()


@app.get("/api/insights")
def get_insights():
    p = OUTPUT_DIR / "insights.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


@app.get("/api/agents/status")
def agent_status():
    p = OUTPUT_DIR / "agent_status.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"runs": 0, "messages": []}


@app.post("/api/agents/run")
def agent_run(force: bool = Query(False)):
    return orchestrator.run_once(force_rebuild=force)


@app.get("/api/report")
def get_report():
    p = OUTPUT_DIR / "latest_report.md"
    if p.exists():
        return {"markdown": p.read_text(encoding="utf-8")}
    return {"markdown": "# No report yet\n\nPOST `/api/agents/run` to generate."}


@app.get("/api/deep/status")
def deep_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {
        "convergence": {"total_hypotheses": 0, "converged_count": 0, "top_insights": []},
        "hypotheses": [],
        "insights": [],
        "recent_messages": [],
    }


@app.post("/api/deep/run-cycle")
def deep_run_cycle():
    return deep_orchestrator.run_cycle()


@app.get("/api/deep/architecture")
def deep_architecture():
    return {
        "streams": [
            {"id": "above", "name": "Above the Water", "focus": "Wind, atmosphere, forecasts, dirty air"},
            {"id": "on_water", "name": "On the Water", "focus": "Telemetry, foil/wing, winning runs"},
            {"id": "around", "name": "Around the Water", "focus": "Fan narrative, rank drama, engagement"},
        ],
        "coordinator": "Cross-stream synthesis, novelty scoring, convergence promotion",
        "memory_path": str(STATUS_FILE.parent / "shared_memory.json"),
        "cycle_interval_recommended_seconds": 600,
    }


# EDA HTML artifacts from repo root
@app.get("/artifacts/{name}")
def artifact(name: str):
    allowed = {"sailgp_dashboard.html", "map_bermuda.html", "map_halifax.html", "sailgp_profile.html"}
    if name not in allowed:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = REPO_ROOT / name
    if not path.exists():
        return JSONResponse({"error": "file missing — run dataExploration notebook"}, status_code=404)
    return FileResponse(path)


# Static website
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
