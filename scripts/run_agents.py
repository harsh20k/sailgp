#!/usr/bin/env python3
"""Run SailGP multi-agent analysis once or in watch mode."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailgp_analysis.agents.orchestrator import AnalysisOrchestrator


def main():
    p = argparse.ArgumentParser(description="SailGP continuous analysis agents")
    p.add_argument("--watch", action="store_true", help="Poll for new data every N seconds")
    p.add_argument("--interval", type=int, default=60, help="Watch interval (seconds)")
    p.add_argument("--force", action="store_true", help="Force full rebuild")
    p.add_argument("--max-iter", type=int, default=None, help="Max watch iterations")
    args = p.parse_args()

    orch = AnalysisOrchestrator()
    if args.watch:
        print(f"Watching data folder every {args.interval}s (Ctrl+C to stop)")
        orch.watch(interval_seconds=args.interval, max_iterations=args.max_iter)
    else:
        status = orch.run_once(force_rebuild=args.force)
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
