"""Benchmark complete DRC collect+update throughput on the real bank factory."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import psutil
import torch

from .curriculum import ReverseFrontierSampler
from .drc_train import DRCTrainConfig, RecurrentPPO, _resolve_device, build_net
from .env import EnvConfig, RicochetEnv
from .puzzles import load_puzzles


def parse_candidate(value: str) -> Tuple[int, int, int, int]:
    try:
        num_envs, n_steps, batch_size, n_epochs = (int(part) for part in value.split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be num_envs x n_steps x batch_size x n_epochs"
        ) from exc
    if min(num_envs, n_steps, batch_size, n_epochs) < 1:
        raise argparse.ArgumentTypeError("candidate values must be positive")
    if batch_size % n_steps:
        raise argparse.ArgumentTypeError("batch_size must be divisible by n_steps")
    if batch_size // n_steps > num_envs:
        raise argparse.ArgumentTypeError("minibatch sequence count exceeds num_envs")
    return num_envs, n_steps, batch_size, n_epochs


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_candidate(
    base: DRCTrainConfig,
    puzzles,
    candidate: Tuple[int, int, int, int],
    *,
    device: str,
    warmup_updates: int,
    measured_updates: int,
) -> Dict[str, object]:
    num_envs, n_steps, batch_size, n_epochs = candidate
    config = dataclasses.replace(
        base,
        num_envs=num_envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device=device,
    )
    env_config = EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
        stop_planes=config.stop_planes,
        distance_plane=config.distance_plane,
    )
    sampler = ReverseFrontierSampler(
        puzzles,
        tau=config.tau,
        ema_beta=config.ema_beta,
        min_episodes=config.min_episodes,
        frontier_p=config.frontier_p,
        retention_p=config.retention_p,
        uniform_p=config.uniform_p,
    )
    envs = [
        RicochetEnv(puzzles, env_config, sampler=sampler, seed=config.seed * 1000 + rank)
        for rank in range(num_envs)
    ]
    torch.manual_seed(config.seed)
    net = build_net(config, envs[0].observation_space.shape, envs[0].action_space.n)
    algo = RecurrentPPO(
        net,
        envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        learning_rate_final=config.learning_rate_final,
        target_kl=config.target_kl,
        adv_std_floor=config.adv_std_floor,
        adv_clip=config.adv_clip,
        clip_range_vf=config.clip_range_vf,
        device=device,
        seed=config.seed,
    )

    metrics: Dict[str, object] = {}
    for _ in range(warmup_updates):
        buffer = algo.collect_rollout()
        metrics = algo.update(buffer, progress_remaining=1.0)
    _sync(device)
    started = time.perf_counter()
    for _ in range(measured_updates):
        buffer = algo.collect_rollout()
        metrics = algo.update(buffer, progress_remaining=1.0)
    _sync(device)
    elapsed = time.perf_counter() - started
    steps = measured_updates * num_envs * n_steps
    process = psutil.Process()
    allocated = None
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        allocated = int(torch.mps.current_allocated_memory())
    return {
        "num_envs": num_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "measured_updates": measured_updates,
        "measured_steps": steps,
        "elapsed_s": round(elapsed, 4),
        "steps_per_s": round(steps / elapsed, 2),
        "rss_bytes": int(process.memory_info().rss),
        "accelerator_allocated_bytes": allocated,
        "last_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=None,
        help="num_envs x n_steps x batch_size x n_epochs; repeatable",
    )
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup_updates < 1 or args.measured_updates < 1:
        raise ValueError("warmup and measured update counts must be positive")
    candidates = args.candidate or [
        (32, 32, 256, 2),
        (64, 32, 512, 1),
        (128, 32, 1024, 1),
        (64, 64, 1024, 1),
    ]
    base = DRCTrainConfig.from_yaml(args.config)
    device = _resolve_device(args.device)
    puzzles = load_puzzles(
        base.bank_path,
        "train",
        eval_fraction=base.eval_fraction,
        split_seed=base.split_seed,
        val_test_seed=base.val_test_seed,
    )
    rows = []
    for candidate in candidates:
        print(f"[drc-benchmark] candidate={candidate}", flush=True)
        row = benchmark_candidate(
            base,
            puzzles,
            candidate,
            device=device,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    best = max(rows, key=lambda row: float(row["steps_per_s"]))
    report = {
        "config": args.config,
        "device": device,
        "bank_path": base.bank_path,
        "train_puzzles": len(puzzles),
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "candidates": rows,
        "selected": {
            key: best[key] for key in ("num_envs", "n_steps", "batch_size", "n_epochs")
        },
        "selected_steps_per_s": best["steps_per_s"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected": report["selected"], "steps_per_s": best["steps_per_s"]}))


if __name__ == "__main__":
    main()
