from dataclasses import replace

import numpy as np

from src.benchmarking.phase2 import (
    Phase2Config,
    _bc_dataset_impl,
    _build_model,
    _fixture_attempt,
    _training_factory,
    evaluate_controller,
)
from src.env.ricochet_env import FixedLayout, RicochetRobotsEnv
from src.env.puzzle_bank import PuzzleBank


def _layout() -> FixedLayout:
    height = width = 4
    h_walls = np.zeros((height + 1, width), dtype=bool)
    v_walls = np.zeros((height, width + 1), dtype=bool)
    h_walls[0, :] = h_walls[-1, :] = True
    v_walls[:, 0] = v_walls[:, -1] = True
    return FixedLayout(
        height=height,
        width=width,
        h_walls=h_walls,
        v_walls=v_walls,
        robot_positions={0: (0, 0), 1: (0, 1)},
        goal_position=(3, 3),
        target_robot=0,
    )


def test_action_masks_match_brute_force_and_do_not_mutate() -> None:
    layout = _layout()
    env = RicochetRobotsEnv(fixed_layout=layout, obs_mode="image", include_noop=False)
    env.reset()
    before = dict(env.get_board().robot_positions)
    mask = env.action_masks()
    assert mask.shape == (8,)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, env.action_masks())
    assert env.get_board().robot_positions == before
    for action in range(8):
        candidate = RicochetRobotsEnv(fixed_layout=layout, obs_mode="image", include_noop=False)
        candidate.reset()
        candidate_before = dict(candidate.get_board().robot_positions)
        candidate.step(action)
        moved = candidate.get_board().robot_positions != candidate_before
        assert bool(mask[action]) is moved


def test_expert_actions_are_allowed_on_representative_puzzles() -> None:
    config = Phase2Config(
        height=8,
        width=8,
        num_robots=2,
        obs_mode="image",
        policy="grid_cnn",
        use_action_masking=True,
        fixture_attempts=100,
    )
    puzzles = [puzzle for seed in range(200) if (puzzle := _fixture_attempt((seed, config))) is not None][:8]
    observations, actions, masks = _bc_dataset_impl(puzzles, config, seed=3, include_masks=True)
    assert masks is not None
    assert len(observations) == len(actions) == len(masks)
    assert all(bool(masks[index, action]) for index, action in enumerate(actions))


def test_maskable_ppo_smoke_and_evaluation_have_zero_blocked_actions() -> None:
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
        training_seeds=(3,),
        checkpoints=(0, 16),
        device="cpu",
    )
    vec_env = make_vec_env(
        lambda: RicochetRobotsEnv(**config.environment_spec(7).to_env_kwargs()),
        n_envs=1,
        seed=7,
    )
    model = _build_model(vec_env, config, seed=7, device="cpu")
    model.learn(total_timesteps=16)
    puzzle = next(
        puzzle for seed in range(500) if (puzzle := _fixture_attempt((seed, replace(config, fixture_attempts=500)))) is not None
    )
    evaluation = evaluate_controller(
        "model",
        [puzzle],
        config.environment_spec(9),
        episodes_per_puzzle=5,
        seed=10,
        model=model,
        deterministic=False,
        action_masking=True,
    )
    assert evaluation["summary"]["no_movement_rate"] == 0.0
    vec_env.close()


def test_bank_curriculum_wrapper_forwards_masks(tmp_path) -> None:
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.env_util import make_vec_env

    config = Phase2Config(
        height=8,
        width=8,
        num_robots=2,
        obs_mode="image",
        policy="grid_cnn",
        use_action_masking=True,
        curriculum_stage_max_lengths=(2,),
        training_seeds=(3,),
        checkpoints=(0, 8),
        n_envs=1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
    )
    puzzles = [puzzle for seed in range(500) if (puzzle := _fixture_attempt((seed, config))) is not None][:30]
    bank = PuzzleBank(tmp_path / "bank", run_id="mask-wrapper-test")
    bank.add_puzzles(puzzles)
    factory = _training_factory(bank, config, 3)
    vec_env = make_vec_env(factory, n_envs=1, seed=3)
    vec_env.reset()
    masks = get_action_masks(vec_env)
    assert masks.shape == (1, 8)
    assert masks.any()
    vec_env.close()
