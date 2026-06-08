"""Coordinate ingest → analytics → report on a schedule or single pass."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sailgp_analysis.agents.analytics_agent import AnalyticsAgent
from sailgp_analysis.agents.base import AgentContext
from sailgp_analysis.agents.ingest_agent import IngestAgent
from sailgp_analysis.agents.report_agent import ReportAgent
from sailgp_analysis.config import DATA_ROOT, OUTPUT_DIR, STATE_FILE


class AnalysisOrchestrator:
    def __init__(
        self,
        data_root: Path = DATA_ROOT,
        output_dir: Path = OUTPUT_DIR,
        state_path: Path = STATE_FILE,
    ):
        self.data_root = data_root
        self.output_dir = output_dir
        self.state_path = state_path
        self.agents = [IngestAgent(), AnalyticsAgent(), ReportAgent()]

    def _ctx(self) -> AgentContext:
        return AgentContext(
            data_root=self.data_root,
            output_dir=self.output_dir,
            state_path=self.state_path,
        )

    def run_once(self, force_rebuild: bool = False) -> dict:
        ctx = self._ctx()
        state = ctx.load_state()
        state["runs"] = state.get("runs", 0) + 1
        if force_rebuild:
            state["force_full_rebuild"] = True

        for agent in self.agents:
            state = agent.run(ctx, state)

        state["force_full_rebuild"] = False
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        ctx.save_state(state)

        status_path = self.output_dir / "agent_status.json"
        status = {
            "last_run": state["last_run"],
            "runs": state["runs"],
            "pending": state.get("pending_events", {}),
            "last_snapshot": state.get("last_snapshot"),
            "last_report": state.get("last_report"),
            "messages": [
                {"agent": m.agent, "event": m.event, "payload": m.payload, "timestamp": m.timestamp}
                for m in ctx.messages[-20:]
            ],
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    def watch(self, interval_seconds: int = 60, max_iterations: int | None = None) -> None:
        """Continuous loop: re-analyze when new/changed CSVs appear."""
        n = 0
        while max_iterations is None or n < max_iterations:
            self.run_once()
            n += 1
            time.sleep(interval_seconds)
