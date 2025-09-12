import numpy as np
import builtins
import types
import importlib

from env.ricochet_env import RicochetRobotsEnv
from env.ricochet_core import Board
from env.solver import solve_bfs, apply_actions


def test_ensure_solvable_success(monkeypatch):
    # Monkeypatch solver.solve_bfs to always return a dummy solution
    import env.solver as solver
    monkeypatch.setattr(solver, "solve_bfs", lambda *args, **kwargs: [(0, 0)])

    env = RicochetRobotsEnv(height=3, width=3, num_robots=1, ensure_solvable=True)
    obs, info = env.reset()
    assert info.get("level_solvable") is True


def test_ensure_solvable_attempt_limit_failure(monkeypatch):
    # Force generation to a fixed unsolvable board by patching _generate_random_board
    def make_unsolvable(self):
        h, w = self.height, self.width
        h_walls = np.ones((h + 1, w), dtype=bool)
        v_walls = np.ones((h, w + 1), dtype=bool)
        # Fully boxed; no movement possible
        robot_positions = {0: (1 if h > 1 else 0, 1 if w > 1 else 0)}
        goal = (0, 0)
        return Board(height=h, width=w, h_walls=h_walls, v_walls=v_walls, robot_positions=robot_positions, goal_position=goal, target_robot=0)

    import env.ricochet_env as env_mod
    monkeypatch.setattr(env_mod.RicochetRobotsEnv, "_generate_random_board", make_unsolvable, raising=True)

    # Patch solver to always return None (no solution)
    import env.solver as solver
    monkeypatch.setattr(solver, "solve_bfs", lambda *args, **kwargs: None)

    env = env_mod.RicochetRobotsEnv(height=2, width=2, num_robots=1, ensure_solvable=True)
    # Reduce attempt limit by temporarily patching the loop via attribute (simulate few tries)
    # We'll call reset and expect RuntimeError
    raised = False
    try:
        env.reset()
    except RuntimeError:
        raised = True
    assert raised


def test_ensure_solvable_includes_optimal_length_and_limits(monkeypatch):
    # Create a board where optimal path length is known (1 move)
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

    # Patch generator to always produce our board
    import env.ricochet_env as env_mod
    monkeypatch.setattr(env_mod.RicochetRobotsEnv, "_generate_random_board", lambda self: board, raising=True)

    env = env_mod.RicochetRobotsEnv(height=H, width=W, num_robots=1, ensure_solvable=True, solver_max_depth=5, solver_max_nodes=100)
    _, info = env.reset()
    assert info.get("optimal_length") == 1
    limits = info.get("solver_limits")
    assert isinstance(limits, dict)
    assert limits["max_depth"] == 5 and limits["max_nodes"] == 100

    # Consistency: applying BFS actions reaches goal in that many moves
    actions = solve_bfs(board, max_depth=5, max_nodes=100)
    assert actions is not None
    assert len(actions) == info["optimal_length"]
    end = apply_actions(board, actions)
    from env.ricochet_core import reached_goal
    assert reached_goal(end)
