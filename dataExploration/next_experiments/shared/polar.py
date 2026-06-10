"""Canonical VMG polar lookup from exp2 polar table CSV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KMH_PER_KN = 1.852
TWA_BIN_WIDTH = 10.0
TWS_BIN_WIDTH_KN = 2.0
MIN_BIN_COUNT = 20

DEFAULT_POLAR_CSV = (
    Path(__file__).resolve().parents[2] / "exported" / "exp2_vmg_residual_polar_table.csv"
)


@dataclass
class PolarTable:
    table: dict[tuple[float, float], float]
    counts: dict[tuple[float, float], int]
    global_mean: float
    twa_bin_width: float = TWA_BIN_WIDTH
    tws_bin_width_kn: float = TWS_BIN_WIDTH_KN

    def lookup(self, twa: float, tws_kmh: float) -> float:
        if np.isnan(twa) or np.isnan(tws_kmh):
            return self.global_mean
        twa_bin = float(np.floor(twa / self.twa_bin_width) * self.twa_bin_width)
        tws_kn = tws_kmh / KMH_PER_KN
        tws_bin = float(np.floor(tws_kn / self.tws_bin_width_kn) * self.tws_bin_width_kn)
        key = (twa_bin, tws_bin)
        if key in self.table and self.counts.get(key, 0) >= MIN_BIN_COUNT:
            return self.table[key]
        return self._nearest_fallback(twa_bin, tws_bin)

    def _nearest_fallback(self, twa_bin: float, tws_bin: float) -> float:
        valid = [(k, v) for k, v in self.table.items() if self.counts.get(k, 0) >= MIN_BIN_COUNT]
        if not valid:
            return self.global_mean
        dists = [
            (np.hypot((k[0] - twa_bin) / self.twa_bin_width, (k[1] - tws_bin) / self.tws_bin_width_kn), v)
            for k, v in valid
        ]
        return min(dists, key=lambda x: x[0])[1]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (twa_b, tws_b), mean_vmg in sorted(self.table.items()):
            rows.append(
                {
                    "twa_bin_deg": twa_b,
                    "tws_bin_kn": tws_b,
                    "mean_vmg_kmh": mean_vmg,
                    "n": self.counts.get((twa_b, tws_b), 0),
                    "valid": self.counts.get((twa_b, tws_b), 0) >= MIN_BIN_COUNT,
                }
            )
        return pd.DataFrame(rows)


def _twa_bin(twa: float) -> float:
    return float(np.floor(twa / TWA_BIN_WIDTH) * TWA_BIN_WIDTH)


def _tws_bin_kn(tws_kmh: float) -> float:
    return float(np.floor((tws_kmh / KMH_PER_KN) / TWS_BIN_WIDTH_KN) * TWS_BIN_WIDTH_KN)


def load_polar_table(csv_path: Path | str | None = None) -> PolarTable:
    path = Path(csv_path) if csv_path else DEFAULT_POLAR_CSV
    df = pd.read_csv(path)
    table = {
        (float(r["twa_bin_deg"]), float(r["tws_bin_kn"])): float(r["mean_vmg_kmh"])
        for _, r in df.iterrows()
    }
    counts = {
        (float(r["twa_bin_deg"]), float(r["tws_bin_kn"])): int(r["n"])
        for _, r in df.iterrows()
    }
    valid = df[df["n"] >= MIN_BIN_COUNT]
    global_mean = float(valid["mean_vmg_kmh"].mean()) if len(valid) else float(df["mean_vmg_kmh"].mean())
    return PolarTable(table=table, counts=counts, global_mean=global_mean)


def polar_expected(twa_deg: float, tws_kn: float, polar: PolarTable | None = None) -> float:
    """Expected VMG (km/h) for signed TWA (deg) and TWS (knots)."""
    p = polar or load_polar_table()
    return p.lookup(twa_deg, tws_kn * KMH_PER_KN)


def csv_lookup_value(twa_deg: float, tws_kn: float, csv_path: Path | str | None = None) -> float | None:
    """Direct bin value from CSV (valid bins only), for consistency checks."""
    path = Path(csv_path) if csv_path else DEFAULT_POLAR_CSV
    df = pd.read_csv(path)
    twa_bin = _twa_bin(twa_deg)
    tws_bin = _tws_bin_kn(tws_kn * KMH_PER_KN)
    hit = df[(df["twa_bin_deg"] == twa_bin) & (df["tws_bin_kn"] == tws_bin)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    if not bool(row.get("valid", row["n"] >= MIN_BIN_COUNT)):
        return None
    return float(row["mean_vmg_kmh"])


def test_polar_lookup(twa_deg: float = 45.0, tws_kn: float = 20.0, rtol: float = 1e-6) -> dict:
    polar = load_polar_table()
    got = polar_expected(twa_deg, tws_kn, polar)
    csv_val = csv_lookup_value(twa_deg, tws_kn)
    if csv_val is None:
        return {"pass": False, "reason": "no valid CSV bin", "lookup": got}
    ok = bool(np.isclose(got, csv_val, rtol=rtol, atol=1e-4))
    return {
        "pass": ok,
        "twa_deg": twa_deg,
        "tws_kn": tws_kn,
        "lookup_kmh": got,
        "csv_kmh": csv_val,
        "twa_bin": _twa_bin(twa_deg),
        "tws_bin_kn": _tws_bin_kn(tws_kn * KMH_PER_KN),
    }


if __name__ == "__main__":
    result = test_polar_lookup()
    status = "PASS" if result["pass"] else "FAIL"
    print(f"[polar] {status}: TWA={result.get('twa_deg')} TWS={result.get('tws_kn')}kn")
    print(f"  lookup={result.get('lookup_kmh')} csv={result.get('csv_kmh')}")
