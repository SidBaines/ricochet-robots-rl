"""Run the reproducible Phase 2 feed-forward benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import Phase2Config, run_phase2_benchmark  # noqa: E402


def _format_completion_summary(result: Mapping[str, Any]) -> str:
    seed_results = result["seeds"]
    tests_ran = all(seed_result.get("test_selected") is not None for seed_result in seed_results)
    if tests_ran:
        return (
            "Held-out success across seeds: "
            f"deterministic={result['aggregate']['deterministic']['mean']:.1%}, "
            f"stochastic={result['aggregate']['stochastic']['mean']:.1%}"
        )

    selected_points = []
    for seed_result in seed_results:
        selected_timesteps = int(seed_result["selected_checkpoint_timesteps"])
        selected_points.append(next(
            point for point in seed_result["curve"]
            if int(point["actual_timesteps"]) == selected_timesteps
        ))
    deterministic = sum(float(point["deterministic"]["success_rate"]) for point in selected_points)
    stochastic = sum(float(point["stochastic"]["success_rate"]) for point in selected_points)
    return (
        "Final test sealed; selected validation success across seeds: "
        f"deterministic={deterministic / len(selected_points):.1%}, "
        f"stochastic={stochastic / len(selected_points):.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "phase2_benchmark.yaml")
    parser.add_argument("--bank-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "phase2_bank")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "phase2_benchmark")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Phase 2 config must be a mapping")
    config = Phase2Config.from_mapping(raw)
    result = run_phase2_benchmark(config, bank_dir=args.bank_dir, output_dir=args.output_dir)
    print(f"Phase 2 benchmark complete: {args.output_dir / 'report.md'}")
    print(_format_completion_summary(result))


if __name__ == "__main__":
    main()
