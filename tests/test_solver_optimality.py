import numpy as np

from env.ricochet_core import Board, apply_action, reached_goal, UP, LEFT, RIGHT
from env.solver import solve_bfs, solve_astar, apply_actions


def make_small_board():
    H, W = 3, 3
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    robots = {0: (2, 1)}
    goal = (0, 1)
    board = Board(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)
    return board


def test_bfs_optimality_trivial():
    board = make_small_board()
    actions = solve_bfs(board, max_depth=5, max_nodes=100)
    assert actions is not None
    assert len(actions) == 1
    end = apply_actions(board, actions)
    assert reached_goal(end)


def test_bfs_cutoff_returns_none():
    board = make_small_board()
    actions = solve_bfs(board, max_depth=0, max_nodes=1)
    assert actions is None


def test_multirobot_requires_moving_blocker():
    H, W = 4, 4
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    robots = {0: (3, 1), 1: (1, 1)}
    goal = (0, 1)
    v_walls[0, 2] = True
    board = Board(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)
    actions = solve_bfs(board, max_depth=20, max_nodes=10000)
    assert actions is not None
    assert len(actions) >= 2
    end = apply_actions(board, actions)
    assert reached_goal(end)


def test_astar_admissible_matches_bfs():
    board = make_small_board()
    a_bfs = solve_bfs(board, max_depth=5, max_nodes=100)
    a_a0 = solve_astar(board, max_depth=5, max_nodes=100, h_mode="admissible_zero")
    a_a1 = solve_astar(board, max_depth=5, max_nodes=100, h_mode="admissible_one")
    assert a_bfs == a_a0 == a_a1


def test_zero_length_when_start_on_goal():
    H, W = 3, 3
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    robots = {0: (0, 1)}
    goal = (0, 1)
    board = Board(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)
    a_bfs = solve_bfs(board, max_depth=5, max_nodes=100)
    a_astar = solve_astar(board, max_depth=5, max_nodes=100)
    assert isinstance(a_bfs, list) and len(a_bfs) == 0
    assert isinstance(a_astar, list) and len(a_astar) == 0
