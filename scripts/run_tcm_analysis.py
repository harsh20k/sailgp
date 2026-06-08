#!/usr/bin/env python3
"""Run all TCM variation experiments and write results JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sailgp_analysis.tcm.evaluate import run_all_variations
from sailgp_analysis.tcm.train import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Run SailGP TCM analysis variations")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-learning-curves", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress batch-level progress bars")
    parser.add_argument("--no-resume", action="store_true", help="Re-run all variations from scratch")
    parser.add_argument("--device", default=None, help="Force device: mps, cuda, or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or default_device()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=not args.quiet,
        show_batch_progress=not args.quiet,
    )
    print(f"SailGP TCM suite — epochs={config.epochs} batch={config.batch_size} device={config.device}\n{'=' * 60}", flush=True)
    run_all_variations(
        config=config,
        include_learning_curves=not args.no_learning_curves,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
