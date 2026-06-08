"""Base types for the SailGP analysis agent system."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AgentMessage:
    agent: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentContext:
    data_root: Path
    output_dir: Path
    state_path: Path
    messages: list[AgentMessage] = field(default_factory=list)

    def log(self, agent: str, event: str, **payload: Any) -> None:
        self.messages.append(AgentMessage(agent=agent, event=event, payload=payload))

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"file_hashes": {}, "last_run": None, "runs": 0}

    def save_state(self, state: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: AgentContext, state: dict) -> dict:
        """Run agent pass; return updated state slice."""
