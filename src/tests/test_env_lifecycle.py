import numpy as np
import pytest

from src.env.ricochet_env import RicochetRobotsEnv
from src.profiling.profiler import Profiler


def _board_state(env: RicochetRobotsEnv) -> tuple:
    board = env._board
    assert board is not None
    return (
        board.h_walls.tobytes(),
        board.v_walls.tobytes(),
        tuple(sorted(board.robot_positions.items())),
        board.goal_position,
        board.target_robot,
    )


def test_profiler_is_opt_in() -> None:
    profiler = Profiler(enable_gpu_profiling=False)
    assert profiler.enabled is False

    profiler.enable()
    assert profiler.enabled is True

    profiler.disable()
    assert profiler.enabled is False


def test_unseeded_resets_advance_and_explicit_seed_replays_stream() -> None:
    env = RicochetRobotsEnv(height=8, width=8, num_robots=2, seed=123)

    first_obs, first_info = env.reset()
    first_state = _board_state(env)
    second_obs, second_info = env.reset()
    second_state = _board_state(env)

    assert first_info["board_seed"] != second_info["board_seed"]
    assert first_state != second_state

    replay_first_obs, replay_first_info = env.reset(seed=123)
    replay_first_state = _board_state(env)
    replay_second_obs, replay_second_info = env.reset()
    replay_second_state = _board_state(env)

    assert replay_first_info["board_seed"] == first_info["board_seed"]
    assert replay_second_info["board_seed"] == second_info["board_seed"]
    assert replay_first_state == first_state
    assert replay_second_state == second_state
    np.testing.assert_array_equal(replay_first_obs, first_obs)
    np.testing.assert_array_equal(replay_second_obs, second_obs)


@pytest.mark.parametrize("obs_mode", ["rgb_image", "rgb_cell_image"])
def test_rgb_layout_cache_matches_fresh_environment_after_reset(obs_mode: str) -> None:
    kwargs = {
        "height": 8,
        "width": 8,
        "num_robots": 2,
        "obs_mode": obs_mode,
        "channels_first": False,
    }
    if obs_mode == "rgb_cell_image":
        kwargs["cell_obs_pixel_size"] = 40

    reused = RicochetRobotsEnv(**kwargs)
    reused.reset(seed=101)
    prior_state = _board_state(reused)
    current_obs, _ = reused.reset(seed=202)
    current_state = _board_state(reused)

    fresh = RicochetRobotsEnv(**kwargs)
    reference_obs, _ = fresh.reset(seed=202)

    assert current_state != prior_state
    np.testing.assert_array_equal(current_obs, reference_obs)
