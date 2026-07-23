"""Retain every evaluated DRC model for dense, asynchronous probes.

The trainer atomically rewrites ``model_latest.pt`` after each periodic
evaluation but writes registered immutable model/training snapshots less
often. This lightweight watcher copies a matching ``model_latest.pt`` to a
separate ``probe_model_<steps>.pt`` name as soon as the corresponding
``curve.jsonl`` row appears. Probe training can then lag behind without losing
the next evaluation checkpoint.

The watcher never writes trainer state and does not alter optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .probe_drc_daemon import _timesteps


def _curve_timesteps(path: Path) -> list[int]:
    if not path.exists():
        return []
    steps: list[int] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            steps.append(int(payload["timesteps"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid curve row {line_number} in {path}") from exc
    return steps


def _retained_timesteps(run_dir: Path) -> set[int]:
    retained: set[int] = set()
    for pattern in ("model_[0-9]*.pt", "probe_model_[0-9]*.pt"):
        retained.update(_timesteps(path) for path in run_dir.glob(pattern))
    return retained


def pending_eval_timesteps(run_dir: str | Path) -> list[int]:
    run_dir = Path(run_dir)
    retained = _retained_timesteps(run_dir)
    # A watcher attached after training began cannot reconstruct evaluation
    # checkpoints already overwritten by model_latest. Once the first model is
    # retained, treat it as the explicit beginning of the dense probe series
    # rather than retrying older, permanently unavailable curve rows forever.
    first_retained = min(retained) if retained else None
    return [
        steps
        for steps in _curve_timesteps(run_dir / "curve.jsonl")
        if steps not in retained and (first_retained is None or steps >= first_retained)
    ]


def _model_timesteps(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return int(payload["metadata"]["timesteps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint lacks metadata.timesteps: {path}") from exc


def _copy_atomic(source: Path, target: Path) -> None:
    if target.exists():
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def retain_available(run_dir: str | Path) -> Path | None:
    """Retain the current evaluated model when it matches an unretained row."""
    run_dir = Path(run_dir)
    pending = pending_eval_timesteps(run_dir)
    latest = run_dir / "model_latest.pt"
    if not pending or not latest.exists():
        return None

    latest_steps = _model_timesteps(latest)
    if latest_steps not in pending:
        return None

    target = run_dir / f"probe_model_{latest_steps}.pt"
    _copy_atomic(latest, target)
    copied_steps = _model_timesteps(target)
    if copied_steps != latest_steps:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Retained checkpoint step changed during copy: expected {latest_steps}, found {copied_steps}"
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")

    run_dir = Path(args.run_dir)
    while True:
        retained = retain_available(run_dir)
        if retained is not None:
            print(
                f"[drc-probe-snapshot] retained t={_timesteps(retained):,} as {retained.name}",
                flush=True,
            )
        if args.once:
            break
        if (run_dir / "training_final.pt").exists() and not pending_eval_timesteps(run_dir):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
