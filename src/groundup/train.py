"""Training entrypoint: MaskablePPO over the ground-up env.

Usage:
    .venv-phase2/bin/python -m src.groundup.train --config configs/groundup_e2_8x8.yaml

Every run writes to <out_dir>/<run_name>/: config.json, curve.jsonl (periodic
validation evals + curriculum state), report.json (final evals), model.zip.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class TrainConfig:
    bank_path: str = ""
    out_dir: str = "artifacts/groundup"
    run_name: str = "run"
    # splits (must match legacy discipline)
    eval_fraction: float = 0.25
    split_seed: int = 20_260_710
    val_test_seed: int = 20_260_711
    # env
    horizon: int = 32
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    shaping_coef: float = 0.1
    time_channel: bool = True
    canonical_identity: bool = True
    # curriculum: "reverse" | "uniform"
    curriculum: str = "reverse"
    tau: float = 0.75
    ema_beta: float = 0.02
    min_episodes: int = 200
    frontier_p: float = 0.5
    retention_p: float = 0.25
    uniform_p: float = 0.125
    # optional overfit mode: train on a tiny fixed subset, eval on the same
    overfit_n: Optional[int] = None
    # PPO
    seed: int = 11
    num_envs: int = 16
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 4
    total_steps: int = 1_000_000
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-4
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    # model
    channels: int = 64
    n_blocks: int = 4
    reduce_channels: int = 16
    features_dim: int = 256
    pi_head: int = 128
    vf_head: int = 128
    # evaluation
    eval_every: int = 100_000
    eval_per_length: int = 25
    final_eval_per_length: int = 50
    final_eval_episodes: int = 3
    device: str = "auto"
    # W&B (opt-in, mirrors the legacy harness convention)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        values = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)


def _flatten_eval(prefix: str, summary: dict) -> dict:
    out = {f"{prefix}/success_rate": summary["success_rate"]}
    for length, v in summary.get("by_length", {}).items():
        out[f"{prefix}/len_{length}"] = v["success_rate"]
    for robots, v in summary.get("by_robots_moved", {}).items():
        out[f"{prefix}/robots_{robots}"] = v["success_rate"]
    return out


def _flatten_sampler(stats: dict) -> dict:
    out = {}
    if "frontier" in stats:
        out["curriculum/frontier"] = stats["frontier"]
        for d, value in stats.get("ema", {}).items():
            out[f"curriculum/ema_d{d}"] = value
    return out


def _maybe_init_wandb(config: "TrainConfig", out_dir: Path):
    if config.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or config.run_name,
        config=dataclasses.asdict(config),
        dir=str(out_dir),
    )
    run.define_metric("train/timesteps")
    for pattern in ("train/*", "curriculum/*", "val_det/*", "final/*"):
        run.define_metric(pattern, step_metric="train/timesteps")
    return run


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_and_train(config: TrainConfig) -> dict:
    import torch.nn as nn
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from .curriculum import ReverseFrontierSampler, UniformSampler
    from .env import EnvConfig, RicochetEnv
    from .evaluate import evaluate_policy, scripted_baseline
    from .model import GridTrunk
    from .puzzles import load_puzzles, stratified_subset

    out = Path(config.out_dir) / config.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    wandb_run = _maybe_init_wandb(config, out)

    env_config = EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
    )

    split_kwargs = dict(
        eval_fraction=config.eval_fraction,
        split_seed=config.split_seed,
        val_test_seed=config.val_test_seed,
    )
    train_puzzles = load_puzzles(config.bank_path, "train", **split_kwargs)
    if config.overfit_n is not None:
        train_puzzles = stratified_subset(
            train_puzzles, per_length=max(1, config.overfit_n // 4), seed=config.seed
        )[: config.overfit_n]
        val_suite = list(train_puzzles)
        final_suite = list(train_puzzles)
    else:
        val_puzzles = load_puzzles(config.bank_path, "val", **split_kwargs)
        val_suite = stratified_subset(val_puzzles, config.eval_per_length, seed=7)
        final_suite = stratified_subset(val_puzzles, config.final_eval_per_length, seed=8)
    print(
        f"[groundup] train={len(train_puzzles)} puzzles, val_suite={len(val_suite)}, "
        f"final_suite={len(final_suite)}",
        flush=True,
    )

    if config.curriculum == "reverse":
        sampler = ReverseFrontierSampler(
            train_puzzles,
            tau=config.tau,
            ema_beta=config.ema_beta,
            min_episodes=config.min_episodes,
            frontier_p=config.frontier_p,
            retention_p=config.retention_p,
            uniform_p=config.uniform_p,
        )
    elif config.curriculum == "uniform":
        sampler = UniformSampler(train_puzzles)
    else:
        raise ValueError(f"Unknown curriculum: {config.curriculum}")

    def make_env(rank: int):
        def _init():
            return RicochetEnv(train_puzzles, env_config, sampler=sampler, seed=config.seed * 1000 + rank)

        return _init

    vec_env = VecMonitor(DummyVecEnv([make_env(r) for r in range(config.num_envs)]))

    device = _resolve_device(config.device)
    lr0, lr1 = config.learning_rate, config.learning_rate_final

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lambda progress: lr1 + (lr0 - lr1) * progress,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        seed=config.seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=GridTrunk,
            features_extractor_kwargs=dict(
                features_dim=config.features_dim,
                channels=config.channels,
                n_blocks=config.n_blocks,
                reduce_channels=config.reduce_channels,
            ),
            net_arch=dict(pi=[config.pi_head], vf=[config.vf_head]),
            # SB3's default Tanh heads saturate on CNN features (constant
            # value function; see E3 v1 diagnosis and the legacy Phase 5
            # provenance which hit the identical failure).
            activation_fn=nn.ReLU,
        ),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[groundup] device={device} policy_params={n_params:,}", flush=True)

    curve_path = out / "curve.jsonl"
    start_time = time.time()

    class PeriodicEval(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_eval = config.eval_every

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps < self.next_eval:
                return
            self.next_eval += config.eval_every
            det = evaluate_policy(
                self.model, val_suite, env_config, episodes_per_puzzle=1, deterministic=True
            )
            entry = {
                "timesteps": int(self.num_timesteps),
                "elapsed_s": round(time.time() - start_time, 1),
                "steps_per_s": round(self.num_timesteps / max(1e-9, time.time() - start_time), 1),
                "val_det": det,
                "sampler": sampler.stats(),
            }
            with curve_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            if wandb_run is not None:
                payload = {
                    "train/timesteps": int(self.num_timesteps),
                    "train/steps_per_s": entry["steps_per_s"],
                    **_flatten_eval("val_det", det),
                    **_flatten_sampler(sampler.stats()),
                }
                wandb_run.log(payload, step=int(self.num_timesteps))
            print(
                f"[groundup] t={entry['timesteps']:>9,} ({entry['steps_per_s']:.0f} st/s) "
                f"val_det={det['success_rate']:.3f} "
                f"by_len={ {k: v['success_rate'] for k, v in det['by_length'].items()} } "
                f"sampler={sampler.stats().get('frontier', '-')}",
                flush=True,
            )
            self.model.save(out / "model_latest.zip")

    model.learn(total_timesteps=config.total_steps, callback=PeriodicEval(), progress_bar=False)
    model.save(out / "model_final.zip")

    report = {
        "config": dataclasses.asdict(config),
        "policy_params": int(n_params),
        "elapsed_s": round(time.time() - start_time, 1),
        "final_val_det": evaluate_policy(
            model, final_suite, env_config, episodes_per_puzzle=1, deterministic=True
        ),
        "final_val_stoch": evaluate_policy(
            model,
            final_suite,
            env_config,
            episodes_per_puzzle=config.final_eval_episodes,
            deterministic=False,
        ),
        "baseline_random": scripted_baseline(
            final_suite, env_config, policy="random", episodes_per_puzzle=5
        ),
        "baseline_target_only": scripted_baseline(
            final_suite, env_config, policy="target_only", episodes_per_puzzle=5
        ),
        "sampler_final": sampler.stats(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("final_val_det", "final_val_stoch", "baseline_random", "baseline_target_only")}, indent=2), flush=True)
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/timesteps": int(config.total_steps),
                **_flatten_eval("final/val_det", report["final_val_det"]),
                **_flatten_eval("final/val_stoch", report["final_val_stoch"]),
                **_flatten_eval("final/baseline_random", report["baseline_random"]),
                **_flatten_eval("final/baseline_target_only", report["baseline_target_only"]),
            },
            step=int(config.total_steps),
        )
        wandb_run.finish()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    if args.set:
        values = dataclasses.asdict(config)
        for item in args.set:
            key, _, raw = item.partition("=")
            if key not in values:
                raise ValueError(f"Unknown override key: {key}")
            values[key] = yaml.safe_load(raw)
        config = TrainConfig(**values)
    build_and_train(config)


if __name__ == "__main__":
    main()
