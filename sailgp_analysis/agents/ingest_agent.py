"""Detect new or changed boat CSV files in DataChallenge_Export."""
from __future__ import annotations

import hashlib
from pathlib import Path

from sailgp_analysis.agents.base import Agent, AgentContext
from sailgp_analysis.data_loader import iter_boat_csvs


class IngestAgent(Agent):
    name = "ingest"

    def _file_hash(self, path: Path, chunk: int = 65536) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()[:16]

    def run(self, ctx: AgentContext, state: dict) -> dict:
        known: dict[str, str] = state.get("file_hashes", {})
        new_files: list[dict] = []
        changed_files: list[dict] = []
        current: dict[str, str] = {}

        for venue, race_label, team, path in iter_boat_csvs(ctx.data_root):
            key = str(path)
            digest = self._file_hash(path)
            current[key] = digest
            info = {"path": key, "venue": venue, "race_label": race_label, "team": team}
            if key not in known:
                new_files.append(info)
            elif known[key] != digest:
                changed_files.append(info)

        state["file_hashes"] = current
        state["pending_events"] = {
            "new_files": new_files,
            "changed_files": changed_files,
            "has_changes": bool(new_files or changed_files),
        }
        ctx.log(self.name, "scan_complete", new=len(new_files), changed=len(changed_files), total=len(current))
        return state
