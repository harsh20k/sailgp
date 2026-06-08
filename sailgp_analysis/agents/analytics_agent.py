"""Recompute metrics when ingest detects changes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sailgp_analysis.agents.base import Agent, AgentContext
from sailgp_analysis.analytics import build_snapshot, write_snapshot


class AnalyticsAgent(Agent):
    name = "analytics"

    def run(self, ctx: AgentContext, state: dict) -> dict:
        pending = state.get("pending_events", {})
        force = state.get("force_full_rebuild", False)
        if not pending.get("has_changes") and not force and state.get("last_snapshot"):
            ctx.log(self.name, "skipped", reason="no_changes")
            return state

        snapshot = build_snapshot(ctx.data_root)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = ctx.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        write_snapshot(ctx.data_root)

        insights = self._derive_insights(snapshot)
        insights_path = ctx.output_dir / "insights.json"
        insights_path.write_text(json.dumps(insights, indent=2), encoding="utf-8")

        state["last_snapshot"] = datetime.now(timezone.utc).isoformat()
        state["pending_events"] = {"has_changes": False, "new_files": [], "changed_files": []}
        ctx.log(self.name, "snapshot_built", venues=list(snapshot.get("venues", {}).keys()), insights=len(insights))
        return state

    def _derive_insights(self, snapshot: dict) -> list[dict]:
        insights: list[dict] = []
        for venue, vdata in snapshot.get("venues", {}).items():
            for race in vdata.get("races", []):
                teams = [t for t in race.get("teams", []) if t.get("mean_speed")]
                if not teams:
                    continue
                fastest = max(teams, key=lambda t: t["mean_speed"])
                most_foil = max(teams, key=lambda t: t.get("foiling_pct") or 0)
                insights.append({
                    "venue": venue,
                    "race_label": race["race_label"],
                    "type": "fastest_team",
                    "team": fastest["team"],
                    "mean_speed_kmh": round(fastest["mean_speed"], 2),
                })
                if most_foil.get("foiling_pct"):
                    insights.append({
                        "venue": venue,
                        "race_label": race["race_label"],
                        "type": "highest_foiling_pct",
                        "team": most_foil["team"],
                        "foiling_pct": round(most_foil["foiling_pct"], 1),
                    })
        for key, mw in snapshot.get("marks_summary", {}).items():
            if mw.get("wg_mean_tws") and mw.get("lg_mean_tws"):
                delta = mw["wg_mean_tws"] - mw["lg_mean_tws"]
                if abs(delta) > 2:
                    insights.append({
                        "venue_race": key,
                        "type": "wind_gradient",
                        "wg_minus_lg_tws_kmh": round(delta, 2),
                    })
        return insights
