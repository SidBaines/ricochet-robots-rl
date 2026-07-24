"""Watch a DRC run directory and probe each immutable model snapshot once."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from .collate_drc_probes import (
    load_probe_series,
    write_machine_outputs,
    write_outputs,
)
from .probe_drc_landing import probe_checkpoint


def _timesteps(path: Path) -> int:
    match = re.fullmatch(r"(?:probe_)?model_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"Not a retained model snapshot: {path}")
    return int(match.group(1))


def _available_snapshots(run_dir: Path) -> list[Path]:
    """Return one checkpoint per step, preferring registered 5M snapshots."""
    by_step: dict[int, Path] = {}
    for model in run_dir.glob("probe_model_[0-9]*.pt"):
        by_step[_timesteps(model)] = model
    for model in run_dir.glob("model_[0-9]*.pt"):
        by_step[_timesteps(model)] = model
    return [by_step[steps] for steps in sorted(by_step)]


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def _curve_reached(run_dir: Path, target_timesteps: int) -> bool:
    curve = run_dir / "curve.jsonl"
    if not curve.exists():
        return False
    rows = [line for line in curve.read_text().splitlines() if line.strip()]
    if not rows:
        return False
    return int(json.loads(rows[-1])["timesteps"]) >= target_timesteps


def process_available(
    run_dir: str | Path,
    *,
    dataset: str | Path,
    device: str,
    n_states: int,
    probe_epochs: int,
    p1_run_dir: str | Path | None,
) -> int:
    run_dir = Path(run_dir)
    snapshots = _available_snapshots(run_dir)
    completed = 0
    for model in snapshots:
        steps = _timesteps(model)
        output = run_dir / f"probe_t{steps}.json"
        if output.exists():
            continue
        control = run_dir / "probe_control.json"
        if not control.exists():
            result = probe_checkpoint(
                model,
                dataset,
                n_states=n_states,
                device=device,
                random_init=True,
                probe_epochs=probe_epochs,
            )
            _write_json_atomic(control, result)
        result = probe_checkpoint(
            model,
            dataset,
            n_states=n_states,
            device=device,
            probe_epochs=probe_epochs,
        )
        _write_json_atomic(output, result)
        completed += 1
        rows = load_probe_series(run_dir)
        control_payload = json.loads(control.read_text())
        write_outputs(
            run_dir,
            rows,
            control=control_payload,
            p1_run_dir=p1_run_dir,
        )
        write_machine_outputs(run_dir, rows, control=control_payload)
        print(f"[drc-probe] completed t={steps:,}", flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-states", type=int, default=16_000)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--p1-run-dir", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--target-timesteps", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise ValueError("poll-seconds must be positive")

    run_dir = Path(args.run_dir)
    while True:
        completed = process_available(
            run_dir,
            dataset=args.dataset,
            device=args.device,
            n_states=args.n_states,
            probe_epochs=args.probe_epochs,
            p1_run_dir=args.p1_run_dir,
        )
        if args.once:
            break
        if args.target_timesteps is not None and _curve_reached(run_dir, args.target_timesteps) and completed == 0:
            break
        if args.target_timesteps is None and (run_dir / "training_final.pt").exists() and completed == 0:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
