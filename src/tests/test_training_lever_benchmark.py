"""Smoke contract for the bounded Phase 5 training-lever benchmark."""

import numpy as np
from pathlib import Path

from src.benchmarking.phase2 import Phase2Config
from src.env.ricochet_env import RicochetRobotsEnv
from src.scripts.benchmark_phase5_training_levers import _benchmark_condition
from src.scripts.run_phase5_bc_lever_screen import build_variant_configs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_training_lever_condition_runs_real_policy_backward_on_cpu() -> None:
    config = Phase2Config(
        obs_mode="image",
        policy="grid_resnet",
        features_dim=32,
        conv_channels=8,
        n_conv_layers=1,
        training_seeds=(1,),
        checkpoints=(0,),
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
    )
    env = RicochetRobotsEnv(**config.environment_spec(123).to_env_kwargs())
    observation, _ = env.reset()
    env.close()
    result = _benchmark_condition(
        config,
        np.stack([observation]),
        device="cpu",
        batch_size=4,
        conv_channels=8,
        n_blocks=1,
        precision="fp32",
        warmup_steps=1,
        measured_steps=1,
        seed=7,
    )
    assert result["status"] == "ok"
    assert result["examples_per_second"] > 0
    assert result["losses_finite"]
    assert result["gradients_finite"]
    assert result["parameters_finite"]


def test_bc_lever_screen_changes_only_predeclared_factors() -> None:
    _, variants = build_variant_configs(
        PROJECT_ROOT / "configs" / "phase5_bc_training_lever_screen.yaml"
    )
    assert set(variants) == {
        "batch2048",
        "batch4096",
        "autocast_fp16",
        "resnet128x4",
        "resnet96x4",
    }
    assert variants["batch2048"].bc_batch_size == 2048
    assert variants["batch4096"].bc_batch_size == 4096
    assert variants["autocast_fp16"].bc_precision == "autocast_fp16"
    assert (variants["resnet128x4"].conv_channels, variants["resnet128x4"].n_conv_layers) == (128, 4)
    assert (variants["resnet96x4"].conv_channels, variants["resnet96x4"].n_conv_layers) == (96, 4)
    assert all(config.bc_pretrain_epochs == 5 for config in variants.values())
    assert all(not config.run_final_test for config in variants.values())
