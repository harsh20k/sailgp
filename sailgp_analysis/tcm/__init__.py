"""Temporal Convolutional Model analysis for SailGP telemetry."""

from sailgp_analysis.tcm.dataset import (
    VARIATION_CONFIGS,
    build_variation_dataset,
    load_prepared_frames,
)
from sailgp_analysis.tcm.evaluate import evaluate_variation, run_all_variations
from sailgp_analysis.tcm.model import TCNModel
from sailgp_analysis.tcm.train import TrainConfig, default_device, train_model

__all__ = [
    "VARIATION_CONFIGS",
    "TCNModel",
    "build_variation_dataset",
    "default_device",
    "evaluate_variation",
    "load_prepared_frames",
    "run_all_variations",
    "train_model",
]
