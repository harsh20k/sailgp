#!/usr/bin/env python3
"""Run all SailGP autoencoder experiments and save summary JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataExploration.autoencoder_experiments import run_all_experiments


def main() -> int:
    fast = "--fast" in sys.argv
    out = ROOT / "dataExploration" / "exported" / "autoencoder_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    r = run_all_experiments(fast=fast, verbose=True)

    payload = {
        "fast_mode": fast,
        "summary": r["summary"].to_dict(orient="records"),
        "experiments": {
            k: {
                "meaningful": r[k]["meaningful"],
                "metrics": {
                    mk: float(mv) if isinstance(mv, (int, float)) else str(mv)
                    for mk, mv in r[k]["metrics"].items()
                },
            }
            for k in ["exp1", "exp2", "exp3", "exp4", "exp5"]
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(r["summary"].to_string(index=False))
    for k in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
        print(f"{k}: meaningful={r[k]['meaningful']}")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
