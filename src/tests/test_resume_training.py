"""Exact continuation contracts for promotable Phase 5 checkpoints."""

import copy
from dataclasses import replace

import numpy as np
import torch

from src.benchmarking.phase2 import (
    Phase2Config,
    _build_model,
    _capture_global_rng_state,
    _curriculum_checkpoint_state,
    _fixture_attempt,
    _restore_resumable_checkpoint,
    _save_resumable_checkpoint,
    _training_factory,
)
from src.env.puzzle_bank import PuzzleBank


def _config(**overrides) -> Phase2Config:
    values = dict(
        height=8,
        width=8,
        num_robots=2,
        obs_mode="image",
        policy="grid_cnn",
        features_dim=32,
        conv_channels=8,
        n_conv_layers=1,
        max_optimal_length=2,
        curriculum_stage_max_lengths=(2,),
        curriculum_min_episodes=10_000,
        use_action_masking=True,
        training_seeds=(5,),
        checkpoints=(0, 8, 16),
        n_envs=2,
        n_steps=4,
        batch_size=8,
        n_epochs=1,
        learning_rate=3e-4,
        save_checkpoint_weights=True,
        select_checkpoint_by_validation=True,
        run_final_test=False,
        device="cpu",
    )
    values.update(overrides)
    return Phase2Config(**values)


def _assert_optimizer_equal(left, right) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id in left_state["state"]:
        assert left_state["state"][parameter_id].keys() == right_state["state"][parameter_id].keys()
        for key, left_value in left_state["state"][parameter_id].items():
            right_value = right_state["state"][parameter_id][key]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value), (parameter_id, key)
            else:
                assert left_value == right_value


def test_resumable_checkpoint_matches_uninterrupted_update(tmp_path) -> None:
    from stable_baselines3.common.env_util import make_vec_env

    config = _config()
    puzzles = [
        puzzle
        for seed in range(1000)
        if (puzzle := _fixture_attempt((seed, config))) is not None
    ][:40]
    assert len(puzzles) == 40
    bank = PuzzleBank(tmp_path / "bank", run_id="resume-exact")
    bank.add_puzzles(puzzles)

    factory = _training_factory(bank, config, seed=5)
    vec_env = make_vec_env(factory, n_envs=config.n_envs, seed=5)
    reference = _build_model(vec_env, config, seed=5, device="cpu")
    reference.learn(total_timesteps=8, reset_num_timesteps=False)
    checkpoint_path = tmp_path / "checkpoint" / "model"
    state_path = _save_resumable_checkpoint(
        reference,
        vec_env,
        factory.curriculum_manager,
        config,
        seed=5,
        model_path=checkpoint_path,
    )
    reference.learn(total_timesteps=8, reset_num_timesteps=False)
    reference_policy = copy.deepcopy(reference.policy.state_dict())
    reference_curriculum = _curriculum_checkpoint_state(factory.curriculum_manager)
    reference_last_obs = np.asarray(reference._last_obs).copy()
    reference_rng = _capture_global_rng_state()

    resume_config = replace(
        config,
        checkpoints=(8, 16),
        resume_checkpoint_path=str(checkpoint_path.with_suffix(".zip")),
        resume_state_path=str(state_path),
    )
    resumed_factory = _training_factory(bank, resume_config, seed=5)
    resumed_vec_env = make_vec_env(resumed_factory, n_envs=resume_config.n_envs, seed=5)
    resumed = type(reference).load(
        checkpoint_path.with_suffix(".zip"),
        env=resumed_vec_env,
        device="cpu",
        seed=5,
        force_reset=False,
        n_steps=resume_config.n_steps,
        batch_size=resume_config.batch_size,
        n_epochs=resume_config.n_epochs,
        learning_rate=resume_config.learning_rate,
        gamma=resume_config.gamma,
        gae_lambda=resume_config.gae_lambda,
        ent_coef=resume_config.ent_coef,
        vf_coef=resume_config.vf_coef,
        max_grad_norm=resume_config.max_grad_norm,
        target_kl=resume_config.target_kl,
    )
    _restore_resumable_checkpoint(
        resumed,
        resumed_vec_env,
        resumed_factory.curriculum_manager,
        resume_config,
        seed=5,
        model_path=checkpoint_path.with_suffix(".zip"),
        state_path=state_path,
    )
    resumed.learn(total_timesteps=8, reset_num_timesteps=False)

    assert resumed.num_timesteps == reference.num_timesteps == 16
    for name, parameter in resumed.policy.state_dict().items():
        assert torch.equal(parameter, reference_policy[name]), name
    _assert_optimizer_equal(reference.policy.optimizer, resumed.policy.optimizer)
    assert _curriculum_checkpoint_state(resumed_factory.curriculum_manager) == reference_curriculum
    assert np.array_equal(resumed._last_obs, reference_last_obs)
    resumed_rng = _capture_global_rng_state()
    assert torch.equal(resumed_rng["torch_cpu"], reference_rng["torch_cpu"])
    assert resumed_rng["python"] == reference_rng["python"]
    assert np.array_equal(resumed_rng["numpy"][1], reference_rng["numpy"][1])

    vec_env.close()
    resumed_vec_env.close()


def test_resume_config_rejects_ambiguous_initialization() -> None:
    try:
        _config(initial_checkpoint_path="warm.zip", resume_checkpoint_path="resume.zip", checkpoints=(8, 16))
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("ambiguous resume and warm-start configuration was accepted")
