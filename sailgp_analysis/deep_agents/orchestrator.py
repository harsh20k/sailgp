"""Long-running deep research: three stream agents + coordinator each cycle."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sailgp_analysis.config import DATA_ROOT, OUTPUT_DIR
from sailgp_analysis.deep_agents.coordinator import CoordinatorAgent
from sailgp_analysis.deep_agents.shared_memory import SharedMemory
from sailgp_analysis.deep_agents.stream_agents import STREAM_AGENTS

DEEP_OUTPUT = OUTPUT_DIR / "deep_research"
MEMORY_FILE = DEEP_OUTPUT / "shared_memory.json"
STATUS_FILE = DEEP_OUTPUT / "deep_agent_status.json"


class DeepResearchOrchestrator:
    """
    Runs for days: each cycle all three stream agents research independently,
    then coordinator synthesizes cross-stream insights. Convergence emerges
    as hypotheses accumulate evidence across runs.
    """

    def __init__(self, data_root: Path = DATA_ROOT, output_dir: Path = DEEP_OUTPUT):
        self.data_root = data_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory = SharedMemory(MEMORY_FILE)

    def run_cycle(self) -> dict:
        run_id = len(self.memory.run_history) + 1
        started = datetime.now(timezone.utc).isoformat()
        cycle_log = {"run_id": run_id, "started": started, "streams": {}}

        for stream_id, AgentCls in STREAM_AGENTS.items():
            agent = AgentCls(self.data_root, self.memory, run_id)
            agent.load()
            findings = agent.research_cycle()
            cycle_log["streams"][stream_id] = {
                "findings": len(findings),
                "agent": AgentCls.stream_name,
            }

        coordinator = CoordinatorAgent(self.memory, run_id)
        new_insights = coordinator.synthesize()
        cycle_log["coordinator"] = {"new_insights": len(new_insights)}
        cycle_log["finished"] = datetime.now(timezone.utc).isoformat()
        cycle_log["convergence"] = coordinator.convergence_report()

        self.memory.run_history.append(cycle_log)
        self.memory.meta["last_cycle"] = cycle_log["finished"]
        self.memory.meta["total_cycles"] = run_id
        self.memory.save()

        status = {
            "last_cycle": cycle_log,
            "convergence": cycle_log["convergence"],
            "hypotheses": [
                {
                    "id": h.id,
                    "stream": h.stream,
                    "statement": h.statement,
                    "status": h.status,
                    "confidence": round(h.confidence, 3),
                    "novelty": round(h.novelty_score, 3),
                    "evidence_count": len(h.evidence),
                }
                for h in sorted(self.memory.hypotheses, key=lambda x: -x.confidence)[:40]
            ],
            "insights": [
                {
                    "id": i.id,
                    "title": i.title,
                    "narrative": i.narrative,
                    "streams": i.streams,
                    "convergence": i.convergence_score,
                    "novelty": i.novelty_score,
                }
                for i in self.memory.top_insights(15)
            ],
            "recent_messages": self.memory.messages[-30:],
        }
        STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    def run_until_converged(
        self,
        min_converged: int = 5,
        max_cycles: int = 200,
        interval_seconds: int = 300,
    ) -> None:
        """Run for days until N hypotheses converge or max_cycles hit."""
        for _ in range(max_cycles):
            status = self.run_cycle()
            n = status["convergence"].get("converged_count", 0)
            if n >= min_converged:
                break
            time.sleep(interval_seconds)

    def watch(self, interval_seconds: int = 600, max_cycles: int | None = None) -> None:
        n = 0
        while max_cycles is None or n < max_cycles:
            self.run_cycle()
            n += 1
            time.sleep(interval_seconds)
