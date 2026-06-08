#!/usr/bin/env python3
"""Run TabNet SailGP analysis variations."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sailgp_analysis.config import DATA_ROOT, OUTPUT_DIR
from sailgp_analysis.tabnet.config import VARIATION_NAMES
from sailgp_analysis.tabnet.variations.v1_speed_regression import run as run_v1
from sailgp_analysis.tabnet.variations.v2_foiling_classification import run as run_v2
from sailgp_analysis.tabnet.variations.v3_rank_prediction import run as run_v3
from sailgp_analysis.tabnet.variations.v4_prestart import run as run_v4
from sailgp_analysis.tabnet.variations.v5_cross_venue import run as run_v5

RUNNERS = {
    "v1": run_v1,
    "v2": run_v2,
    "v3": run_v3,
    "v4": run_v4,
    "v5": run_v5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="TabNet SailGP analysis")
    parser.add_argument(
        "--variation",
        choices=list(VARIATION_NAMES) + ["all"],
        default="all",
        help="Which variation to run (default: all)",
    )
    parser.add_argument(
        "--venue",
        choices=["bermuda", "halifax", "both"],
        default="both",
        help="Venue filter (v5 uses both; others use race splits from data_prep)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to DataChallenge_Export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "tabnet",
        help="Directory for JSON results and plots",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Data root not found: {args.data_root}")

    to_run = list(VARIATION_NAMES) if args.variation == "all" else [args.variation]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for var in to_run:
        print(f"Running {var}...")
        result = RUNNERS[var](data_root=args.data_root, output_dir=args.output_dir)
        summary[var] = {
            "meaningful": result.meaningful,
            "beats_baseline": result.beats_baseline,
            "tabnet_metrics": result.tabnet_metrics,
            "baseline_metrics": result.baseline_metrics,
            "verdict_reason": result.verdict_reason,
        }
        print(f"  meaningful={result.meaningful} | {result.verdict_reason}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
