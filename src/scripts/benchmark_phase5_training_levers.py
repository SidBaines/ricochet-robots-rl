"""Bounded real-policy throughput screen for Phase 5 training levers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from stable_baselines3.common.env_util import make_vec_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.phase2 import Phase2Config, _build_model  # noqa: E402
from src.env.ricochet_env import RicochetRobotsEnv  # noqa: E402


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _clear(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _device_memory(device: str) -> Dict[str, int] | None:
    if device == "mps":
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device == "cuda":
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated()),
            "driver_allocated_bytes": int(torch.cuda.memory_reserved()),
        }
    return None


def _benchmark_condition(
    base_config: Phase2Config,
    observations: np.ndarray,
    *,
    device: str,
    batch_size: int,
    conv_channels: int,
    n_blocks: int,
    precision: str,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> Dict[str, Any]:
    _clear(device)
    torch.manual_seed(seed)
    config = replace(
        base_config,
        conv_channels=conv_channels,
        n_conv_layers=n_blocks,
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        initial_checkpoint_path=None,
        resume_checkpoint_path=None,
        resume_state_path=None,
        checkpoints=(0,),
        save_checkpoint_weights=False,
        select_checkpoint_by_validation=False,
        run_final_test=False,
        wandb_project=None,
        wandb_run_name=None,
    )
    vec_env = make_vec_env(
        lambda: RicochetRobotsEnv(**config.environment_spec(seed).to_env_kwargs()),
        n_envs=1,
        seed=seed,
    )
    model = None
    try:
        model = _build_model(vec_env, config, seed=seed, device=device)
        policy = model.policy
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=base_config.bc_learning_rate)
        scaler = torch.amp.GradScaler(device, enabled=precision == "autocast_fp16")
        tiled = np.resize(observations, (batch_size, *observations.shape[1:]))
        batch_obs = torch.as_tensor(tiled, dtype=torch.float32, device=device)
        batch_actions = torch.arange(batch_size, device=device) % int(model.action_space.n)
        losses = []
        gradient_abs_max = 0.0
        gradients_finite = True
        step_seconds = []
        for step in range(warmup_steps + measured_steps):
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=precision == "autocast_fp16",
            ):
                distribution = policy.get_distribution(batch_obs)
                loss = -distribution.log_prob(batch_actions).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for parameter in policy.parameters():
                if parameter.grad is None:
                    continue
                gradients_finite = gradients_finite and bool(torch.isfinite(parameter.grad).all())
                gradient_abs_max = max(gradient_abs_max, float(parameter.grad.detach().abs().max()))
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            elapsed = time.perf_counter() - started
            losses.append(float(loss.detach()))
            if step >= warmup_steps:
                step_seconds.append(elapsed)
        total = sum(step_seconds)
        parameters_finite = all(
            bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()
        )
        return {
            "status": "ok",
            "batch_size": batch_size,
            "conv_channels": conv_channels,
            "n_blocks": n_blocks,
            "precision": precision,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "total_seconds": total,
            "mean_step_seconds": total / measured_steps,
            "examples_per_second": batch_size * measured_steps / total,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "losses_finite": all(math.isfinite(loss) for loss in losses),
            "gradients_finite": gradients_finite,
            "parameters_finite": parameters_finite,
            "gradient_abs_max": gradient_abs_max,
            "device_memory": _device_memory(device),
        }
    except RuntimeError as error:
        return {
            "status": "runtime_error",
            "batch_size": batch_size,
            "conv_channels": conv_channels,
            "n_blocks": n_blocks,
            "precision": precision,
            "error": str(error),
        }
    finally:
        if model is not None:
            del model
        vec_env.close()
        _clear(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase5_screen_b_stable.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured steps must be positive")

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = Phase2Config.from_mapping(raw)
    probe_env = RicochetRobotsEnv(**config.environment_spec(91_001).to_env_kwargs())
    observations = []
    for seed in range(64):
        observation, _ = probe_env.reset(seed=91_001 + seed)
        observations.append(observation)
    probe_env.close()
    observation_array = np.stack(observations)

    condition_specs = [
        (1024, 128, 6, "fp32"),
        (2048, 128, 6, "fp32"),
        (4096, 128, 6, "fp32"),
        (1024, 128, 6, "autocast_fp16"),
        (1024, 128, 4, "fp32"),
        (1024, 96, 4, "fp32"),
    ]
    conditions = []
    for batch_size, channels, blocks, precision in condition_specs:
        label = f"batch{batch_size}_c{channels}_b{blocks}_{precision}"
        print(f"[training-lever] {label}", flush=True)
        result = _benchmark_condition(
            config,
            observation_array,
            device=args.device,
            batch_size=batch_size,
            conv_channels=channels,
            n_blocks=blocks,
            precision=precision,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            seed=20260717,
        )
        result["label"] = label
        conditions.append(result)
        if result["status"] == "ok":
            print(
                f"[training-lever] {label}: {result['examples_per_second']:.1f} examples/s, "
                f"{result['mean_step_seconds']:.3f}s/step",
                flush=True,
            )
        else:
            print(f"[training-lever] {label}: {result['status']}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "bounded throughput screen; not a model-quality promotion result",
        "device": args.device,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "source_config": asdict(config),
        "observation_shape": list(observation_array.shape[1:]),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
