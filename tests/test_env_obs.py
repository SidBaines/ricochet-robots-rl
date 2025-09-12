import numpy as np

from env.ricochet_env import RicochetRobotsEnv, FixedLayout
from env.ricochet_core import UP


def make_layout_two_robots():
    H, W = 4, 4
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    # Add a specific wall: up wall above (2,1), and right wall at (1,2)
    h_walls[2, 1] = True
    v_walls[1, 3] = True
    robots = {0: (2, 1), 1: (1, 2)}
    goal = (0, 0)
    return FixedLayout(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=1)


def test_channels_first_image_obs():
    layout = make_layout_two_robots()
    env = RicochetRobotsEnv(fixed_layout=layout, channels_first=True)
    obs, info = env.reset()
    C, H, W = obs.shape
    # 4 wall channels + goal + target + 2 robots = 8
    assert C == 8
    assert H == layout.height and W == layout.width
    # goal channel at index 4 in channels_first
    assert obs[4, layout.goal_position[0], layout.goal_position[1]] == 1.0
    # up wall channel at index 0 should be 1 above (2,1)
    assert obs[0, 2, 1] == 1.0
    # right wall channel at index 3 should be 1 at (1,2)
    assert obs[3, 1, 2] == 1.0


def test_symbolic_obs_structure():
    layout = make_layout_two_robots()
    env = RicochetRobotsEnv(fixed_layout=layout, obs_mode="symbolic")
    obs, info = env.reset()
    # vector: [goal_r, goal_c, target_id_one_hot(2), robots_flat(4)] length 8
    assert obs.shape == (8,)
    goal_r, goal_c = int(obs[0]), int(obs[1])
    assert (goal_r, goal_c) == layout.goal_position
    # target one hot at index 1
    assert obs[2 + 1] == 1.0


def test_noop_penalty_and_state_unchanged():
    layout = make_layout_two_robots()
    env = RicochetRobotsEnv(fixed_layout=layout, include_noop=True, step_penalty=-0.1, noop_penalty=-0.01)
    obs0, info = env.reset()
    noop_action = env.action_space.n - 1
    obs1, reward, term, trunc, info = env.step(noop_action)
    assert reward == -0.01
    assert np.array_equal(obs0, obs1)


def test_terminated_truncated_exclusive():
    layout = make_layout_two_robots()
    env = RicochetRobotsEnv(fixed_layout=layout, max_steps=1)
    _, _ = env.reset()
    # Take a noop to consume step
    noop_action = env.action_space.n - 1
    _, _, term, trunc, _ = env.step(noop_action)
    assert (term and not trunc) or (trunc and not term) or (not term and not trunc)
