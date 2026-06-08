"""Shared knowledge store for cross-stream agent coordination."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Evidence:
    stream: str
    source: str
    metric: str
    value: Any
    strength: float  # 0-1
    run_id: int


@dataclass
class Hypothesis:
    id: str
    stream: str
    statement: str
    status: str  # proposed | testing | supported | refuted | converged
    confidence: float
    novelty_score: float
    evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cross_stream_links: list[str] = field(default_factory=list)


@dataclass
class CoordinatorInsight:
    id: str
    title: str
    narrative: str
    streams: list[str]
    hypothesis_ids: list[str]
    convergence_score: float
    novelty_score: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SharedMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            raw = {"hypotheses": [], "insights": [], "messages": [], "run_history": []}
        self.hypotheses = [Hypothesis(**{**h, "evidence": [Evidence(**e) for e in h.get("evidence", [])]}) for h in raw.get("hypotheses", [])]
        self.insights = [CoordinatorInsight(**i) for i in raw.get("insights", [])]
        self.messages: list[dict] = raw.get("messages", [])
        self.run_history: list[dict] = raw.get("run_history", [])
        self.meta: dict = raw.get("meta", {})

    def save(self) -> None:
        data = {
            "meta": self.meta,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "insights": [asdict(i) for i in self.insights],
            "messages": self.messages[-500:],
            "run_history": self.run_history[-200:],
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def post_message(self, from_agent: str, to_agent: str, topic: str, body: dict) -> None:
        self.messages.append({
            "from": from_agent,
            "to": to_agent,
            "topic": topic,
            "body": body,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_hypothesis(self, stream: str, statement: str, novelty: float = 0.5) -> Hypothesis:
        h = Hypothesis(
            id=str(uuid.uuid4())[:8],
            stream=stream,
            statement=statement,
            status="proposed",
            confidence=0.2,
            novelty_score=novelty,
        )
        self.hypotheses.append(h)
        return h

    def find_hypothesis(self, statement_substr: str, stream: str | None = None) -> Hypothesis | None:
        for h in reversed(self.hypotheses):
            if statement_substr.lower() in h.statement.lower():
                if stream is None or h.stream == stream:
                    return h
        return None

    def add_evidence(self, hypothesis_id: str, evidence: Evidence) -> None:
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                h.evidence.append(evidence)
                h.updated_at = datetime.now(timezone.utc).isoformat()
                strengths = [e.strength for e in h.evidence]
                h.confidence = min(0.99, sum(strengths) / max(len(strengths), 1) * (1 + 0.1 * (len(strengths) - 1)))
                if h.confidence >= 0.75 and len(h.evidence) >= 2:
                    h.status = "supported"
                if h.confidence >= 0.88 and len(h.evidence) >= 4:
                    h.status = "converged"
                return

    def add_insight(self, insight: CoordinatorInsight) -> None:
        self.insights.append(insight)
        self.insights.sort(key=lambda x: x.convergence_score * x.novelty_score, reverse=True)

    def converged_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status == "converged"]

    def top_insights(self, n: int = 10) -> list[CoordinatorInsight]:
        return self.insights[:n]
