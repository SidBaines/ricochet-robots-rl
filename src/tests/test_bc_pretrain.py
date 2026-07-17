"""Tests for the behaviour-cloning warmstart stage."""

import numpy as np
import pytest

from src.benchmarking.phase2 import (
    Phase2Config,
    _bc_dataset,
    _bc_dataset_impl,
    _bc_pretrain,
    _build_model,
    _fixture_attempt,
)


def _small_config(**overrides):
    values = dict(
        height=8,
        width=8,
        num_robots=2,
        edge_t_per_quadrant=0,
        central_l_per_quadrant=0,
        obs_mode="image",
        policy="grid_cnn",
        max_steps=10,
        bc_pretrain_epochs=3,
        bc_batch_size=32,
        bc_learning_rate=1e-3,
    )
    values.update(overrides)
    return Phase2Config(**values)


def _labelled_puzzles(config, count=6):
    puzzles = []
    seed = 0
    while len(puzzles) < count and seed < 500:
        candidate = _fixture_attempt((seed, config))
        if candidate is not None:
            puzzles.append(candidate)
        seed += 1
    assert len(puzzles) == count, "could not label enough small puzzles"
    return puzzles


def test_bc_dataset_replays_optimal_plans() -> None:
    config = _small_config()
    puzzles = _labelled_puzzles(config)
    observations, actions = _bc_dataset(puzzles, config, seed=0)
    assert observations.dtype == np.uint8
    assert observations.shape[1:] == (8, 8, 8)
    assert len(observations) == len(actions) == sum(p.optimal_length for p in puzzles)
    assert set(int(a) for a in actions) <= set(range(8))


def test_bc_dataset_respects_max_puzzles() -> None:
    config = _small_config(bc_max_puzzles=2)
    puzzles = _labelled_puzzles(config)
    observations, actions = _bc_dataset(puzzles, config, seed=0)
    assert len(actions) <= sum(sorted((p.optimal_length for p in puzzles), reverse=True)[:2])


def test_bc_pretrain_reduces_loss_and_learns_actions() -> None:
    from stable_baselines3.common.env_util import make_vec_env

    from src.env.ricochet_env import RicochetRobotsEnv

    config = _small_config()
    puzzles = _labelled_puzzles(config)
    spec = config.environment_spec(0)

    def factory():
        return RicochetRobotsEnv(**spec.to_env_kwargs())

    vec_env = make_vec_env(factory, n_envs=1, seed=0)
    model = _build_model(vec_env, config, seed=0, device="cpu")
    long_config = _small_config(bc_pretrain_epochs=80)
    stats = _bc_pretrain(model, puzzles, long_config, seed=0, device="cpu")
    assert stats["pairs"] > 0
    assert len(stats["epochs"]) == 80
    assert stats["epochs"][-1]["loss"] < stats["epochs"][0]["loss"]
    # A tiny dataset must be memorizable by the policy head.
    assert stats["epochs"][-1]["accuracy"] > 0.6
    vec_env.close()


def test_bc_requires_plane_observations() -> None:
    with pytest.raises(ValueError, match="BC pretraining requires"):
        Phase2Config(bc_pretrain_epochs=1, obs_mode="rgb_cell_image", policy="stock_cnn")


@pytest.mark.parametrize("bc_apply_action_masks", [False, True])
def test_masked_bc_tracks_validation_and_restores_best_epoch(bc_apply_action_masks) -> None:
    from stable_baselines3.common.env_util import make_vec_env

    from src.env.ricochet_env import RicochetRobotsEnv

    config = _small_config(
        bc_pretrain_epochs=5,
        bc_select_best_by_validation=True,
        use_action_masking=True,
        bc_apply_action_masks=bc_apply_action_masks,
    )
    puzzles = _labelled_puzzles(config, count=10)
    train, validation = puzzles[:6], puzzles[6:]
    spec = config.environment_spec(0)
    vec_env = make_vec_env(lambda: RicochetRobotsEnv(**spec.to_env_kwargs()), n_envs=1, seed=0)
    model = _build_model(vec_env, config, seed=0, device="cpu")
    stats = _bc_pretrain(
        model,
        train,
        config,
        seed=0,
        device="cpu",
        validation_puzzles=validation,
    )
    assert stats["selected_epoch"] in range(1, 6)
    assert stats["validation_pairs"] == sum(p.optimal_length for p in validation)
    assert all("validation_loss" in epoch for epoch in stats["epochs"])
    assert stats["loss_action_masking"] is bc_apply_action_masks
    vec_env.close()


def test_bc_action_mask_loss_requires_maskable_policy() -> None:
    with pytest.raises(ValueError, match="bc_apply_action_masks requires"):
        _small_config(bc_apply_action_masks=True)


def test_bc_device_metric_accumulation_preserves_model_update() -> None:
    """The synchronization optimization must not alter the supervised step."""
    import copy

    import torch
    from stable_baselines3.common.env_util import make_vec_env

    from src.env.ricochet_env import RicochetRobotsEnv

    config = _small_config(
        bc_pretrain_epochs=1,
        bc_batch_size=4,
        bc_reinitialize_action_head=False,
    )
    puzzles = _labelled_puzzles(config)
    spec = config.environment_spec(0)
    vec_env = make_vec_env(lambda: RicochetRobotsEnv(**spec.to_env_kwargs()), n_envs=1, seed=0)
    reference = _build_model(vec_env, config, seed=0, device="cpu")
    optimized = _build_model(vec_env, config, seed=1, device="cpu")
    optimized.policy.load_state_dict(copy.deepcopy(reference.policy.state_dict()))

    observations, actions, _ = _bc_dataset_impl(puzzles, config, seed=0, include_masks=False)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions)
    optimizer = torch.optim.Adam(reference.policy.parameters(), lr=config.bc_learning_rate)
    order = torch.randperm(len(action_tensor), generator=torch.Generator().manual_seed(0))
    reference_loss = 0.0
    reference_correct = 0
    for start in range(0, len(order), config.bc_batch_size):
        index = order[start : start + config.bc_batch_size]
        distribution = reference.policy.get_distribution(obs_tensor[index])
        loss = -distribution.log_prob(action_tensor[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reference_loss += float(loss.detach()) * len(index)
        predicted = distribution.distribution.probs.argmax(dim=-1)
        reference_correct += int((predicted == action_tensor[index]).sum())

    stats = _bc_pretrain(optimized, puzzles, config, seed=0, device="cpu")
    assert stats["epochs"][0]["loss"] == pytest.approx(reference_loss / len(actions), abs=1e-6)
    assert stats["epochs"][0]["accuracy"] == pytest.approx(reference_correct / len(actions))
    for name, parameter in reference.policy.state_dict().items():
        assert torch.equal(parameter, optimized.policy.state_dict()[name]), name
    vec_env.close()
