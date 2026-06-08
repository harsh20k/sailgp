#!/usr/bin/env python3
"""Build web/data/snapshot.json from DataChallenge_Export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailgp_analysis.analytics import write_snapshot

if __name__ == "__main__":
    path = write_snapshot()
    print(f"Wrote {path}")
