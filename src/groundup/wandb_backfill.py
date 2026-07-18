"""Backfill a completed (or in-flight) ground-up run into W&B from its local artifacts.

Usage:
    .venv-phase2/bin/python -m src.groundup.wandb_backfill \
        --run-dir artifacts/groundup/e2_8x8_scratch --project ricochet-robots-rl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import _flatten_eval, _flatten_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project", default="ricochet-robots-rl")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())

    import wandb

    run = wandb.init(
        project=args.project,
        name=args.name or f"groundup-{run_dir.name}",
        config=config,
        dir=str(run_dir),
    )
    run.define_metric("train/timesteps")
    for pattern in ("train/*", "curriculum/*", "val_det/*", "final/*"):
        run.define_metric(pattern, step_metric="train/timesteps")

    curve = run_dir / "curve.jsonl"
    if curve.exists():
        for line in curve.read_text().splitlines():
            entry = json.loads(line)
            payload = {
                "train/timesteps": entry["timesteps"],
                "train/steps_per_s": entry["steps_per_s"],
                **_flatten_eval("val_det", entry["val_det"]),
                **_flatten_sampler(entry.get("sampler", {})),
            }
            run.log(payload, step=int(entry["timesteps"]))

    report_path = run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        final_step = int(report["config"]["total_steps"])
        run.log(
            {
                "train/timesteps": final_step,
                **_flatten_eval("final/val_det", report["final_val_det"]),
                **_flatten_eval("final/val_stoch", report["final_val_stoch"]),
                **_flatten_eval("final/baseline_random", report["baseline_random"]),
                **_flatten_eval("final/baseline_target_only", report["baseline_target_only"]),
            },
            step=final_step,
        )
        test_eval = run_dir / "test_eval.json"
        if test_eval.exists():
            test = json.loads(test_eval.read_text())
            run.summary["test/det_success_rate"] = test["det"]["success_rate"]
            run.summary["test/stoch_success_rate"] = test["stoch"]["success_rate"]
    run.finish()
    print(f"backfilled {run_dir} -> {run.url}")


if __name__ == "__main__":
    main()
