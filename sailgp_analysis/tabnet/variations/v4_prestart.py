"""V4 — Pre-start performance classification."""
from __future__ import annotations

from pathlib import Path

from sailgp_analysis.tabnet.data_prep import build_v4_dataset
from sailgp_analysis.tabnet.evaluate import EvalResult, default_output_dir
from sailgp_analysis.tabnet.runner import run_variation


def run(data_root=None, output_dir: Path | None = None) -> EvalResult:
    ds = build_v4_dataset(data_root)
    return run_variation("v4", ds, output_dir or default_output_dir())
