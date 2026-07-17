from dataclasses import asdict
from pathlib import Path

import yaml

from src.benchmarking.phase2 import Phase2Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config(name: str) -> Phase2Config:
    path = PROJECT_ROOT / "configs" / name
    return Phase2Config.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_canonical_bc_config_is_validation_only_and_saves_step_zero() -> None:
    config = _load_config("phase5_canonical_bc_seed11_retry1.yaml")

    assert config.training_seeds == (11,)
    assert config.checkpoints == (0,)
    assert config.bc_pretrain_epochs == 25
    assert config.bc_select_best_by_validation
    assert config.use_action_masking
    assert not config.bc_apply_action_masks
    assert config.save_checkpoint_weights
    assert config.select_checkpoint_by_validation
    assert not config.run_final_test


def test_phase5_screens_are_matched_except_for_declared_optimizer_bundle() -> None:
    historical = asdict(_load_config("phase5_screen_a_historical.yaml"))
    stable = asdict(_load_config("phase5_screen_b_stable.yaml"))
    allowed_differences = {"learning_rate", "ent_coef", "target_kl", "wandb_run_name"}

    differences = {
        key for key in historical
        if historical[key] != stable[key]
    }

    assert differences == allowed_differences
    assert historical["initial_checkpoint_path"] == stable["initial_checkpoint_path"]
    assert historical["checkpoints"] == (0, 102400, 256000, 512000)
    assert historical["validation_episodes_per_puzzle"] == 5
    assert historical["evaluation_batch_size"] == 64
    assert historical["fixed_validation_seeds"]
    assert historical["use_action_masking"]
    assert not historical["run_final_test"]


def test_phase5_geometry_benchmarks_are_matched_except_for_rollout_shape() -> None:
    names = (
        "phase5_geometry_16x128.yaml",
        "phase5_geometry_32x64.yaml",
        "phase5_geometry_64x32.yaml",
    )
    configs = [asdict(_load_config(name)) for name in names]
    reference = configs[0]
    for candidate in configs[1:]:
        differences = {key for key in reference if reference[key] != candidate[key]}
        assert differences == {"n_envs", "n_steps"}
    assert [(config["n_envs"], config["n_steps"]) for config in configs] == [
        (16, 128),
        (32, 64),
        (64, 32),
    ]
    assert all(config["n_envs"] * config["n_steps"] == 2048 for config in configs)
    assert all(config["checkpoints"] == (0, 102400) for config in configs)
    assert all(config["evaluation_batch_size"] == 64 for config in configs)
    assert all(not config["run_final_test"] for config in configs)


def test_phase5_allocation_rules_are_predeclared_and_conservative() -> None:
    path = PROJECT_ROOT / "configs" / "phase5_allocation_rules.yaml"
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert rules["schema_version"] == 1
    assert rules["selection"]["require_disjoint_confidence_intervals"]
    assert rules["selection"]["consistent_lead_checkpoints"] == 2
    assert rules["selection"]["retain_all_on_ambiguity"]
    assert rules["rungs"][512000]["promotion_checkpoints"] == [512000, 1024000, 2048000]
    full = rules["rungs"][2048000]
    assert full["promotion_checkpoints"][-1] == 10240000
    assert full["plateau"] == {
        "metric": "validation_stochastic_success_rate",
        "minimum_timesteps": 4096000,
        "patience_checkpoints": 3,
        "minimum_delta": 0.01,
    }
