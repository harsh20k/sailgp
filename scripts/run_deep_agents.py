#!/usr/bin/env python3
"""Run deep three-stream research agents (single cycle or long watch)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailgp_analysis.deep_agents.orchestrator import DeepResearchOrchestrator


def main():
    p = argparse.ArgumentParser(description="Deep SailGP stream agents + coordinator")
    p.add_argument("--watch", action="store_true", help="Run continuously")
    p.add_argument("--interval", type=int, default=600, help="Seconds between cycles (default 10 min)")
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--until-converged", type=int, default=None, metavar="N",
                   help="Stop after N converged hypotheses")
    args = p.parse_args()

    orch = DeepResearchOrchestrator()
    if args.until_converged:
        orch.run_until_converged(min_converged=args.until_converged, max_cycles=args.max_cycles or 500,
                                 interval_seconds=args.interval)
    elif args.watch:
        print(f"Deep agents watching every {args.interval}s — Ctrl+C to stop")
        orch.watch(interval_seconds=args.interval, max_cycles=args.max_cycles)
    else:
        status = orch.run_cycle()
        print(json.dumps(status["convergence"], indent=2))


if __name__ == "__main__":
    main()
