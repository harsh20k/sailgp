"""TabNet hyperparameters and feature definitions per variation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VariationName = Literal["v1", "v2", "v3", "v4", "v5"]

VARIATION_NAMES: tuple[VariationName, ...] = ("v1", "v2", "v3", "v4", "v5")

# Race-based splits (no row-level leakage)
BERMUDA_TRAIN_RACES = [f"Race_{i}" for i in range(1, 7)]
BERMUDA_VAL_RACES = ["Race_7", "Race_8"]
HALIFAX_TEST_RACES = [f"Race_{i}" for i in range(1, 7)]
# Rank/pre-start targets only available in Bermuda (Halifax rank column is all NaN)
V4_TRAIN_RACES = [f"Race_{i}" for i in range(1, 6)]
V4_VAL_RACES = ["Race_6"]
V4_TEST_RACES = ["Race_7", "Race_8"]

V1_FEATURES = [
    "TWS_SGP_km_h_1",
    "TWA_SGP_deg",
    "AWA_SGP_deg",
    "AWS_SGP_km_h_1",
    "ANGLE_WING_ROT_deg",
    "ANGLE_WING_TWIST_deg",
    "ANGLE_CA1_deg",
    "ANGLE_CA2_deg",
    "ANGLE_CA3_deg",
    "ANGLE_CA4_deg",
    "ANGLE_CA5_deg",
    "ANGLE_CA6_deg",
    "LENGTH_RH_P_mm",
    "LENGTH_RH_S_mm",
    "LENGTH_RH_BOW_mm",
    "ANGLE_DB_RAKE_P_deg",
    "ANGLE_DB_RAKE_S_deg",
    "ANGLE_DB_CANT_P_deg",
    "ANGLE_DB_CANT_S_deg",
    "LENGTH_DB_H_P_mm",
    "LENGTH_DB_H_S_mm",
    "PITCH_deg",
    "HEEL_deg",
    "LEEWAY_deg",
    "RATE_YAW_deg_s_1",
    "PER_JIB_LEAD_pct",
    "PER_JIB_SHEET_pct",
    "TRK_LEG_NUM_unk",
    "speed_vmg_ratio",
    "twa_abs_deg",
    "team_enc",
    "venue_enc",
]

V2_FEATURES = [
    "TWS_SGP_km_h_1",
    "TWA_SGP_deg",
    "AWA_SGP_deg",
    "AWS_SGP_km_h_1",
    "BOAT_SPEED_km_h_1",
    "VMG_km_h_1",
    "ANGLE_WING_ROT_deg",
    "ANGLE_WING_TWIST_deg",
    "ANGLE_CA1_deg",
    "ANGLE_CA2_deg",
    "ANGLE_CA3_deg",
    "ANGLE_CA4_deg",
    "ANGLE_CA5_deg",
    "ANGLE_CA6_deg",
    "ANGLE_DB_RAKE_P_deg",
    "ANGLE_DB_RAKE_S_deg",
    "ANGLE_DB_CANT_P_deg",
    "ANGLE_DB_CANT_S_deg",
    "LENGTH_DB_H_P_mm",
    "LENGTH_DB_H_S_mm",
    "PITCH_deg",
    "HEEL_deg",
    "LEEWAY_deg",
    "RATE_YAW_deg_s_1",
    "PER_JIB_LEAD_pct",
    "PER_JIB_SHEET_pct",
    "TRK_LEG_NUM_unk",
    "speed_vmg_ratio",
    "twa_abs_deg",
    "team_enc",
    "venue_enc",
]

V3_FEATURES = [
    "mean_vmg",
    "foiling_pct",
    "mean_wing_rot",
    "pc_dtl_start",
    "mean_tws",
    "mean_twd",
    "penalty_count",
    "leg_num",
    "venue_enc",
    "team_enc",
    "num_boats",
]

V4_FEATURES = [
    "mean_pc_tts",
    "mean_pc_dtl",
    "mean_pc_dto",
    "mean_pc_dtb",
    "mean_speed",
    "mean_vmg",
    "mean_twa",
    "penalty_count",
    "late_entry",
    "team_enc",
    "venue_enc",
]

V5_FEATURES = V1_FEATURES  # same as V1, different split


@dataclass
class TabNetParams:
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 3
    gamma: float = 1.3
    lambda_sparse: float = 1e-3
    optimizer_fn: str = "adam"
    optimizer_params: dict = field(default_factory=lambda: {"lr": 2e-2})
    scheduler_params: dict = field(default_factory=lambda: {"step_size": 10, "gamma": 0.9})
    scheduler_fn: str = "step"
    mask_type: str = "sparsemax"
    verbose: int = 0
    seed: int = 42
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 1024
    virtual_batch_size: int = 128


VARIATION_PARAMS: dict[VariationName, TabNetParams] = {
    "v1": TabNetParams(max_epochs=80, batch_size=2048),
    "v2": TabNetParams(max_epochs=80, batch_size=2048),
    "v3": TabNetParams(max_epochs=60, batch_size=64, n_d=8, n_a=8),
    "v4": TabNetParams(max_epochs=60, batch_size=32, n_d=8, n_a=8),
    "v5": TabNetParams(max_epochs=80, batch_size=2048),
}

# Expected dominant features for interpretability checks
EXPECTED_DOMINANT: dict[VariationName, list[str]] = {
    "v1": ["TWA_SGP_deg", "ANGLE_DB_RAKE_P_deg", "LENGTH_RH_BOW_mm", "ANGLE_WING_ROT_deg"],
    "v2": ["ANGLE_DB_RAKE_P_deg", "ANGLE_WING_ROT_deg", "BOAT_SPEED_km_h_1", "PITCH_deg"],
    "v3": ["pc_dtl_start", "mean_vmg", "foiling_pct"],
    "v4": ["mean_pc_tts", "mean_pc_dtb", "mean_pc_dtl"],
    "v5": ["TWA_SGP_deg", "AWS_SGP_km_h_1", "AWA_SGP_deg"],
}
