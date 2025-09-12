import numpy as np
import builtins
import types
import importlib

from env.ricochet_env import RicochetRobotsEnv
from env.ricochet_core import Board


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
