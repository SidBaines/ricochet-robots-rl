import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from src.env.puzzle_bank import (
    PuzzleBank,
    PuzzleMetadata,
    SolverKey,
    SpecKey,
    compute_board_hash,
    compute_layout_hash,
)
from src.env.ricochet_env import generate_board_from_spec_with_seed


@pytest.mark.integration
@pytest.mark.training
def test_phase1_stock_cnn_run_writes_reloadable_model_and_manifest(tmp_path: Path) -> None:
    from stable_baselines3 import PPO

    project_root = Path(__file__).resolve().parents[2]
    log_dir = tmp_path / "logs"
    save_path = tmp_path / "model"
    config = {
        "profile": "none",
        "initialization": {"mode": "fresh"},
        "env": {
            "mode": "random",
            "height": 8,
            "width": 8,
            "num_robots": 2,
            "include_noop": False,
            "step_penalty": -0.01,
            "goal_reward": 1.0,
            "noop_penalty": -0.01,
            "max_steps": 10,
            "seed": 17,
            "ensure_solvable": False,
            "solver_max_depth": 10,
            "solver_max_nodes": 1_000,
            "obs_mode": "rgb_cell_image",
            "cell_obs_pixel_size": 40,
        },
        "model": {"type": "stock_cnn"},
        "algo": {
            "algo": "ppo",
            "timesteps": 64,
            "n_envs": 1,
            "n_steps": 32,
            "batch_size": 32,
            "n_epochs": 1,
            "lr": 0.0003,
            "lr_end": 0.0003,
            "lr_schedule": "none",
            "gamma": 0.99,
            "clip_range": 0.2,
            "clip_end": 0.2,
            "clip_schedule": "none",
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "ent_coef": 0.001,
            "normalize_advantage": True,
            "device": "cpu",
        },
        "curriculum": {"enabled": False},
        "eval": {
            "eval_freq": 0,
            "eval_episodes": 2,
            "fixedset": None,
            "solver_cache": str(tmp_path / "solver.json"),
            "traj_record_freq": 0,
            "traj_record_episodes": 1,
        },
        "monitoring": {
            "log_dir": str(log_dir),
            "save_path": str(save_path),
            "save_freq": 0,
            "tensorboard": False,
            "wandb": False,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "src/scripts/train_agent.py", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    models = list(tmp_path.glob("*_model.zip"))
    assert len(models) == 1
    model = PPO.load(models[0], device="cpu")
    assert type(model.policy.features_extractor).__name__ == "NatureCNN"
    assert model.seed == 17

    manifests = list(log_dir.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["model"]["type"] == "stock_cnn"
    assert manifest["model"]["features_extractor_class"].endswith("NatureCNN")
    assert manifest["environment"]["cell_obs_pixel_size"] == 40
    assert manifest["timesteps"]["requested_additional"] == 64
    assert manifest["timesteps"]["final_total"] == 64


@pytest.mark.integration
@pytest.mark.training
def test_bank_training_uses_heldout_eval_split_end_to_end(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    bank_path = tmp_path / "bank"
    bank = PuzzleBank(bank_path)
    spec = SpecKey(8, 8, 2, 0, 0)
    puzzles = []
    for seed in range(100, 160):
        board = generate_board_from_spec_with_seed(
            height=8,
            width=8,
            num_robots=2,
            edge_t_per_quadrant=0,
            central_l_per_quadrant=0,
            seed=seed,
        )
        puzzles.append(
            PuzzleMetadata(
                seed=seed,
                spec_key=spec,
                solver_key=SolverKey(),
                board_hash=compute_board_hash(board),
                optimal_length=1,
                robots_moved=1,
                nodes_expanded=0,
                solve_time_ms=0,
                solved_within_limits=True,
                hit_depth_limit=False,
                hit_node_limit=False,
                actions_encoded=[],
                created_at=0.0,
                run_id="test",
                board_seed=seed,
                layout_hash=compute_layout_hash(board),
            )
        )
    bank.add_puzzles(puzzles)

    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level": 0,
                        "name": "fixture",
                        "spec_key": {
                            "height": 8,
                            "width": 8,
                            "num_robots": 2,
                            "edge_t_per_quadrant": 0,
                            "central_l_per_quadrant": 0,
                            "generator_rules_version": "1.0",
                        },
                        "min_optimal_length": 1,
                        "max_optimal_length": 1,
                        "min_robots_moved": 1,
                        "max_robots_moved": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    log_dir = tmp_path / "bank_logs"
    config = {
        "profile": "none",
        "initialization": {"mode": "fresh"},
        "env": {
            "mode": "random",
            "height": 8,
            "width": 8,
            "num_robots": 2,
            "max_steps": 10,
            "seed": 23,
            "obs_mode": "rgb_cell_image",
            "cell_obs_pixel_size": 40,
        },
        "model": {"type": "stock_cnn"},
        "algo": {
            "timesteps": 64,
            "n_envs": 1,
            "n_steps": 32,
            "batch_size": 32,
            "n_epochs": 1,
            "device": "cpu",
        },
        "curriculum": {
            "enabled": True,
            "use_bank": True,
            "bank_dir": str(bank_path),
            "config": str(curriculum_path),
            "initial_level": 0,
            "success_threshold": 1.0,
            "min_episodes": 1_000,
            "window_size": 100,
            "check_freq": 100,
            "verbose": False,
            "eval_fraction": 0.25,
            "split_seed": 7,
            "split_key": "layout_hash",
        },
        "eval": {"eval_freq": 32, "eval_episodes": 2, "traj_record_freq": 0},
        "monitoring": {
            "log_dir": str(log_dir),
            "save_path": str(tmp_path / "bank_model"),
            "save_freq": 0,
            "tensorboard": False,
            "wandb": False,
        },
    }
    config_path = tmp_path / "bank_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "src/scripts/train_agent.py", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    run_dirs = [path for path in log_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "evaluations.npz").exists()
    manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
    assert manifest["bank_split"]["training_split"] == "train"
    assert manifest["bank_split"]["evaluation_split"] == "eval"
    assert manifest["bank_split"]["manifest_sha256"]
