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
    assert historical["checkpoints"] == (0, 102400, 256000, 512000, 1024000, 2048000)
    assert historical["validation_episodes_per_puzzle"] == 5
    assert historical["evaluation_batch_size"] == 64
    assert historical["fixed_validation_seeds"]
    assert historical["use_action_masking"]
    assert not historical["run_final_test"]
