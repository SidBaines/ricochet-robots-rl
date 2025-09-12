import numpy as np

from env.ricochet_env import RicochetRobotsEnv, FixedLayout
from env.ricochet_core import UP, LEFT, RIGHT


def make_simple_layout():
    H, W = 5, 5
    # Edge walls init with borders
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    # Add a horizontal wall above (1,2): between rows 0-1 at col 2
    h_walls[1, 2] = True
    # Add a vertical wall to the right of (0,2): between cols 2-3 at row 0
    v_walls[0, 3] = True
    robots = {0: (3, 2)}
    goal = (0, 2)
    target_robot = 0
    return FixedLayout(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=target_robot)


def test_slide_and_stop():
    layout = make_simple_layout()
    env = RicochetRobotsEnv(fixed_layout=layout, include_noop=False, step_penalty=0.0)
    _, _ = env.reset()
    action_up = 0 * 4 + UP
    _, _, _, _, _ = env.step(action_up)
    _, _, _, _, _ = env.step(action_up)
    action_right = 0 * 4 + RIGHT
    _, _, _, _, _ = env.step(action_right)
    action_left = 0 * 4 + LEFT
    _, _, terminated, _, _ = env.step(action_left)
    assert not terminated


def test_goal_termination_and_reward():
    layout = make_simple_layout()
    env = RicochetRobotsEnv(fixed_layout=layout, include_noop=False, step_penalty=-0.01, goal_reward=1.0)
    _, _ = env.reset()
    # Move up to (2,2) if possible; with h_wall at (1,2), robot at (3,2) moves to (2,2)
    _, r1, term, _, _ = env.step(0 * 4 + UP)
    # Move left to reach column 0
    _, r2, term, _, _ = env.step(0 * 4 + LEFT)
    # Move up to row 0
    _, r3, term, _, _ = env.step(0 * 4 + UP)
    # Move right; v_wall at (0,3) should cause stopping on (0,2) goal
    _, r4, term, _, info = env.step(0 * 4 + RIGHT)
    assert term
    assert info.get("is_success", False)
    total_reward = r1 + r2 + r3 + r4
    assert total_reward > 0.9


def test_multi_robot_blocking_and_truncation():
    H, W = 5, 5
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    robots = {0: (4, 2), 1: (2, 2)}
    goal = (0, 0)
    layout = FixedLayout(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)
    env = RicochetRobotsEnv(fixed_layout=layout, include_noop=False, step_penalty=0.0, max_steps=3)
    _, _ = env.reset()
    # Move robot 0 up; should stop at (3,2) because robot 1 occupies (2,2)
    _, _, _, _, _ = env.step(0 * 4 + UP)
    # Move robot 1 left, then right; episode should truncate due to max_steps
    _, _, _, _, _ = env.step(1 * 4 + LEFT)
    _, _, term, trunc, _ = env.step(1 * 4 + RIGHT)
    assert not term
    assert trunc
