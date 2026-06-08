"""Cross-stream synthesis, novelty detection, and convergence."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sailgp_analysis.deep_agents.shared_memory import CoordinatorInsight, Hypothesis, SharedMemory


class CoordinatorAgent:
    name = "coordinator"

    def __init__(self, memory: SharedMemory, run_id: int):
        self.memory = memory
        self.run_id = run_id

    def synthesize(self) -> list[CoordinatorInsight]:
        new_insights: list[CoordinatorInsight] = []
        converged = self.memory.converged_hypotheses()
        supported = [h for h in self.memory.hypotheses if h.status == "supported"]

        # Cross-stream links: same venue/race mentioned in multiple streams
        by_venue_race: dict[str, list[Hypothesis]] = {}
        for h in supported + converged:
            for token in h.statement.split():
                if "/" in token and "Race" in token:
                    by_venue_race.setdefault(token.rstrip(","), []).append(h)

        for key, group in by_venue_race.items():
            streams = list({h.stream for h in group})
            if len(streams) >= 2 and len(group) >= 2:
                insight = self._make_insight(
                    title=f"Multi-stream signal: {key}",
                    narrative=(
                        f"Independent {', '.join(streams)} agents corroborate patterns for {key}. "
                        f"Hypotheses: {'; '.join(h.statement[:80] for h in group[:3])}"
                    ),
                    streams=streams,
                    hypothesis_ids=[h.id for h in group],
                    convergence_score=min(0.95, 0.5 + 0.1 * len(group)),
                    novelty_score=min(0.95, 0.6 + 0.05 * len(streams)),
                )
                if not self._insight_exists(insight.title):
                    new_insights.append(insight)

        # Promote converged hypotheses to top-level insights
        for h in converged:
            if h.confidence >= 0.88:
                insight = self._make_insight(
                    title=f"[{h.stream}] Converged: {h.statement[:70]}",
                    narrative=(
                        f"After {len(h.evidence)} evidence passes (runs {sorted({e.run_id for e in h.evidence})}), "
                        f"confidence {h.confidence:.0%}. {h.statement}"
                    ),
                    streams=[h.stream],
                    hypothesis_ids=[h.id],
                    convergence_score=h.confidence,
                    novelty_score=h.novelty_score,
                )
                if not self._insight_exists(insight.title):
                    new_insights.append(insight)

        # Novel cross-stream: wind gradient (above) + leader foil (on_water)
        above_h = [h for h in supported if h.stream == "above" and "gradient" in h.statement.lower()]
        on_h = [h for h in supported if h.stream == "on_water" and "foiling" in h.statement.lower()]
        if above_h and on_h:
            insight = self._make_insight(
                title="Coupled atmosphere–performance hypothesis",
                narrative=(
                    "Above-water wind structure may drive on-water foiling strategy: "
                    f"({above_h[0].statement[:60]}…) aligns with ({on_h[0].statement[:60]}…). "
                    "Worth joint visualization for submission."
                ),
                streams=["above", "on_water"],
                hypothesis_ids=[above_h[0].id, on_h[0].id],
                convergence_score=0.82,
                novelty_score=0.92,
            )
            if not self._insight_exists(insight.title):
                new_insights.append(insight)
                for h in [above_h[0], on_h[0]]:
                    h.cross_stream_links.extend([x.id for x in [above_h[0], on_h[0]] if x.id != h.id])

        for ins in new_insights:
            self.memory.add_insight(ins)
            self.memory.post_message("coordinator", "all", "insight", {"id": ins.id, "title": ins.title})

        return new_insights

    def _make_insight(self, **kwargs) -> CoordinatorInsight:
        return CoordinatorInsight(id=str(uuid.uuid4())[:8], **kwargs)

    def _insight_exists(self, title: str) -> bool:
        return any(i.title == title for i in self.memory.insights)

    def convergence_report(self) -> dict:
        total_h = len(self.memory.hypotheses)
        by_status = {}
        for h in self.memory.hypotheses:
            by_status[h.status] = by_status.get(h.status, 0) + 1
        return {
            "total_hypotheses": total_h,
            "by_status": by_status,
            "converged_count": len(self.memory.converged_hypotheses()),
            "top_insights": [
                {"title": i.title, "convergence": i.convergence_score, "novelty": i.novelty_score, "streams": i.streams}
                for i in self.memory.top_insights(8)
            ],
            "runs_completed": len(self.memory.run_history),
        }
