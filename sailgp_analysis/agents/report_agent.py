"""Write human-readable reports from analytics output."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sailgp_analysis.agents.base import Agent, AgentContext


class ReportAgent(Agent):
    name = "report"

    def run(self, ctx: AgentContext, state: dict) -> dict:
        metrics_path = ctx.output_dir / "metrics.json"
        insights_path = ctx.output_dir / "insights.json"
        if not metrics_path.exists():
            ctx.log(self.name, "skipped", reason="no_metrics")
            return state

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        insights = json.loads(insights_path.read_text(encoding="utf-8")) if insights_path.exists() else []

        lines = [
            "# SailGP Analysis Report",
            f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
            "## Dataset overview\n",
        ]
        for venue, v in metrics.get("venues", {}).items():
            lines.append(f"- **{venue}**: {v.get('total_rows', 0):,} rows, {len(v.get('races', []))} races, teams: {', '.join(v.get('teams', []))}")

        lines.append("\n## Insights\n")
        if not insights:
            lines.append("_No insights this run._\n")
        for ins in insights[:30]:
            lines.append(f"- `{ins.get('type')}` — {json.dumps(ins)}")

        report_path = ctx.output_dir / "latest_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        history = ctx.output_dir / "run_log.jsonl"
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "insight_count": len(insights),
                "venues": list(metrics.get("venues", {}).keys()),
            }) + "\n")

        state["last_report"] = datetime.now(timezone.utc).isoformat()
        ctx.log(self.name, "report_written", path=str(report_path))
        return state
