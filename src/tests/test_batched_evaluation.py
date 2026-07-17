"""Contracts for deterministic deduplication and batched model evaluation."""

import numpy as np
import torch

from src.benchmarking.phase2 import (
    Phase2Config,
    _build_model,
    _episode,
    _fixture_attempt,
    evaluate_controller,
)
from src.env.ricochet_env import RicochetRobotsEnv


def _model_and_puzzles():
    from stable_baselines3.common.env_util import make_vec_env

    config = Phase2Config(
        height=8,
        width=8,
        num_robots=2,
        obs_mode="image",
        policy="grid_cnn",
        features_dim=32,
        conv_channels=8,
        n_conv_layers=1,
        use_action_masking=True,
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
    )
    spec = config.environment_spec(17)
    vec_env = make_vec_env(lambda: RicochetRobotsEnv(**spec.to_env_kwargs()), n_envs=1, seed=17)
    model = _build_model(vec_env, config, seed=17, device="cpu")
    puzzles = [
        puzzle
        for candidate_seed in range(500)
        if (puzzle := _fixture_attempt((candidate_seed, config))) is not None
    ][:3]
    assert len(puzzles) == 3
    return config, vec_env, model, puzzles


def _summary_without_timing(summary):
    return {
        key: value
        for key, value in summary.items()
        if key not in {"elapsed_seconds", "environment_steps_per_second"}
    }


def test_deterministic_evaluation_runs_once_and_matches_serial_records() -> None:
    config, vec_env, model, puzzles = _model_and_puzzles()
    spec = config.environment_spec(21)
    evaluation = evaluate_controller(
        "model",
        puzzles,
        spec,
        episodes_per_puzzle=5,
        seed=22,
        model=model,
        deterministic=True,
        action_masking=True,
        batch_size=3,
    )
    serial = [
        _episode(
            "model",
            puzzle,
            spec,
            rng=np.random.default_rng(22),
            model=model,
            deterministic=True,
            action_masking=True,
        )
        for puzzle in puzzles
    ]
    assert evaluation["summary"]["episodes"] == len(puzzles)
    assert evaluation["records"] == serial
    vec_env.close()


def test_stochastic_evaluation_is_batch_width_invariant_and_rng_isolated() -> None:
    config, vec_env, model, puzzles = _model_and_puzzles()
    spec = config.environment_spec(31)
    torch_state = torch.get_rng_state().clone()
    single = evaluate_controller(
        "model",
        puzzles,
        spec,
        episodes_per_puzzle=4,
        seed=32,
        model=model,
        deterministic=False,
        action_masking=True,
        batch_size=1,
    )
    assert torch.equal(torch_state, torch.get_rng_state())
    wide = evaluate_controller(
        "model",
        puzzles,
        spec,
        episodes_per_puzzle=4,
        seed=32,
        model=model,
        deterministic=False,
        action_masking=True,
        batch_size=7,
    )
    assert single["records"] == wide["records"]
    assert _summary_without_timing(single["summary"]) == _summary_without_timing(wide["summary"])
    assert single["summary"]["no_movement_rate"] == 0.0
    vec_env.close()


def test_evaluation_batch_size_must_be_positive() -> None:
    config, vec_env, model, puzzles = _model_and_puzzles()
    try:
        evaluate_controller(
            "model",
            puzzles,
            config.environment_spec(41),
            episodes_per_puzzle=1,
            seed=42,
            model=model,
            batch_size=0,
        )
    except ValueError as error:
        assert "batch_size must be positive" in str(error)
    else:
        raise AssertionError("zero evaluation batch size was accepted")
    vec_env.close()
