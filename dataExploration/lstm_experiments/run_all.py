"""Run all SailGP LSTM experiments and emit summary dashboard."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataExploration.lstm_experiments.shared.evaluation import (
    ExperimentResult,
    export_summary_dashboard,
)


def run_all() -> list[ExperimentResult]:
    from dataExploration.lstm_experiments import (
        exp_1_speed_forecast,
        exp_2_foiling_transition,
        exp_3_rank_change,
        exp_4_vmg_efficiency,
        exp_5_team_fingerprint,
    )

    runners = [
        ("exp1", "Speed forecast", exp_1_speed_forecast.run),
        ("exp2", "Foiling transition", exp_2_foiling_transition.run),
        ("exp3", "Rank change", exp_3_rank_change.run),
        ("exp4", "VMG efficiency", exp_4_vmg_efficiency.run),
        ("exp5", "Team fingerprint", exp_5_team_fingerprint.run),
    ]

    total = len(runners)
    results: list[ExperimentResult] = []
    t0 = time.perf_counter()

    print(f"\nSailGP LSTM suite — {total} experiments\n{'=' * 60}", flush=True)

    for i, (exp_id, title, fn) in enumerate(runners, start=1):
        print(f"\n[{i}/{total}] {exp_id}: {title}", flush=True)
        print("-" * 60, flush=True)
        exp_t0 = time.perf_counter()
        try:
            result = fn()
            results.append(result)
            elapsed = time.perf_counter() - exp_t0
            signal = "YES" if result.has_signal else "no"
            print(
                f"  done in {elapsed:.1f}s | {result.lstm_metric_name}={result.lstm_metric:.4f} "
                f"| best baseline ({result.best_baseline_name})={result.best_baseline_metric:.4f} "
                f"| signal={signal}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - exp_t0
            print(f"  FAILED after {elapsed:.1f}s: {exc}", flush=True)
            traceback.print_exc()
            results.append(
                ExperimentResult(
                    experiment_id=exp_id,
                    name=exp_id,
                    task="failed",
                    lstm_metric_name="error",
                    lstm_metric=0.0,
                    details={"error": str(exc)},
                )
            )

    dashboard = export_summary_dashboard(results)
    total_elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}", flush=True)
    print(f"Summary ({total_elapsed:.1f}s total)", flush=True)
    print(f"{'Exp':<6} {'Task':<18} {'LSTM':>10} {'Baseline':>10} {'Delta':>10} Signal", flush=True)
    for r in results:
        print(
            f"{r.experiment_id:<6} {r.task:<18} {r.lstm_metric:>10.4f} "
            f"{r.best_baseline_metric:>10.4f} {r.delta:>+10.4f} "
            f"{'✓' if r.has_signal else '✗'}",
            flush=True,
        )
    print(f"\nDashboard: {dashboard}", flush=True)
    return results


if __name__ == "__main__":
    run_all()
