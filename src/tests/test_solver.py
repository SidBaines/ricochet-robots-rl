import numpy as np

from src.env.ricochet_core import Board, apply_action, reached_goal
from src.env.solver import solve_bfs


def test_bfs_finds_solution_on_simple_board():
    H, W = 5, 5
    h_walls = np.zeros((H + 1, W), dtype=bool)
    v_walls = np.zeros((H, W + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[H, :] = True
    v_walls[:, 0] = True
    v_walls[:, W] = True
    # Place a horizontal wall above row 1 at col 2 and vertical wall to right of (0,2)
    h_walls[1, 2] = True
    v_walls[0, 3] = True
    robots = {0: (3, 2)}
    goal = (0, 2)
    board = Board(height=H, width=W, h_walls=h_walls, v_walls=v_walls, robot_positions=robots, goal_position=goal, target_robot=0)

    actions = solve_bfs(board, max_depth=20, max_nodes=10000)
    assert actions is not None

    # Apply actions and verify goal reached
    cur = board
    for rid, d in actions:
        cur = apply_action(cur, rid, d)
    assert reached_goal(cur)
