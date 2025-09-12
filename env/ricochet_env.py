from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Literal

import numpy as np

from .ricochet_core import Board, apply_action, reached_goal

# Provide a runtime gymnasium binding if available, otherwise a minimal fallback
try:  # pragma: no cover
    import importlib
    _gym = importlib.import_module("gymnasium")  # type: ignore
    _spaces = importlib.import_module("gymnasium.spaces")  # type: ignore
    GymEnvBase = _gym.Env  # type: ignore
    spaces = _spaces  # type: ignore
except Exception:  # pragma: no cover
    class GymEnvBase(object):
        pass
    class _SpacesDummy:  # minimal placeholders to satisfy type usage
        class Box:  # type: ignore
            def __init__(self, low=None, high=None, shape=None, dtype=None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        class Discrete:  # type: ignore
            def __init__(self, n: int):
                self.n = n
    spaces = _SpacesDummy()  # type: ignore


@dataclass
class FixedLayout:
    height: int
    width: int
    # Edge walls
    h_walls: np.ndarray  # (H+1, W) bool
    v_walls: np.ndarray  # (H, W+1) bool
    robot_positions: Dict[int, Tuple[int, int]]
    goal_position: Tuple[int, int]
    target_robot: int


ObsMode = Literal["image", "symbolic"]


class RicochetRobotsEnv(GymEnvBase):
    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        num_robots: int = 2,
        include_noop: bool = True,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        noop_penalty: Optional[float] = None,
        max_steps: int = 100,
        fixed_layout: Optional[FixedLayout] = None,
        seed: Optional[int] = None,
        ensure_solvable: bool = False,
        solver_max_depth: int = 30,
        solver_max_nodes: int = 20000,
        obs_mode: ObsMode = "image",
        channels_first: bool = False,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_robots = num_robots
        self.include_noop = include_noop
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.noop_penalty = step_penalty if noop_penalty is None else noop_penalty
        self.max_steps = max_steps
        self.fixed_layout = fixed_layout
        self.ensure_solvable = ensure_solvable
        self.solver_max_depth = solver_max_depth
        self.solver_max_nodes = solver_max_nodes
        self.obs_mode: ObsMode = obs_mode
        self.channels_first = channels_first

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._board: Optional[Board] = None
        self._num_steps = 0

        # Observation spaces
        if self.obs_mode == "image":
            # channels: up_wall, down_wall, left_wall, right_wall, goal, target_mask, robots...
            self._base_wall_channels = 4
            self._base_misc_channels = 2  # goal, target
            self._num_channels = self._base_wall_channels + self._base_misc_channels + self.num_robots
            if self.channels_first:
                obs_shape = (self._num_channels, self.height, self.width)
            else:
                obs_shape = (self.height, self.width, self._num_channels)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=obs_shape,
                dtype=np.float32,
            )
        elif self.obs_mode == "symbolic":
            vec_len = 2 + self.num_robots + (self.num_robots * 2)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(vec_len,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        # Action space: num_robots * 4 directions (+ 1 for noop)
        self._noop_action = self.num_robots * 4 if include_noop else None
        self.action_space = spaces.Discrete(self.num_robots * 4 + (1 if include_noop else 0))

    # Gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _ = options  # silence unused
        self._num_steps = 0

        if self.fixed_layout is not None:
            self._board = Board(
                height=self.fixed_layout.height,
                width=self.fixed_layout.width,
                h_walls=self.fixed_layout.h_walls.copy(),
                v_walls=self.fixed_layout.v_walls.copy(),
                robot_positions=dict(self.fixed_layout.robot_positions),
                goal_position=self.fixed_layout.goal_position,
                target_robot=self.fixed_layout.target_robot,
            )
            obs = self._make_obs(self._board)
            info = {"level_solvable": None, "target_robot": self._board.target_robot}
            return obs, info

        # Otherwise, generate random board, optionally ensure solvable via solver
        attempts = 0
        while True:
            attempts += 1
            board = self._generate_random_board()
            if not self.ensure_solvable:
                self._board = board
                break
            try:
                from .solver import solve_bfs  # type: ignore
                solution = solve_bfs(board, max_depth=self.solver_max_depth, max_nodes=self.solver_max_nodes)
                if solution is not None:
                    self._board = board
                    optimal_length = len(solution)
                    break
            except ImportError as exc:
                raise RuntimeError("ensure_solvable=True but solver unavailable (ImportError)") from exc
            if attempts >= 50:
                raise RuntimeError("Failed to generate solvable board within attempt limit")

        obs = self._make_obs(self._board)
        # Note: solver None during attempts may be due to cutoffs (depth/nodes), not true unsolvability.
        info = {"level_solvable": True if self.ensure_solvable else None, "target_robot": self._board.target_robot}
        if self.ensure_solvable:
            info["optimal_length"] = optimal_length
            info["solver_limits"] = {"max_depth": self.solver_max_depth, "max_nodes": self.solver_max_nodes}
        return obs, info

    def step(self, action: int):
        assert self._board is not None, "Call reset() first"
        self._num_steps += 1

        terminated = False
        truncated = False
        info: Dict[str, object] = {}

        reward = self.step_penalty
        if self.include_noop and action == self._noop_action:
            reward = self.noop_penalty
        else:
            robot_id, direction = self._decode_action(action)
            self._board = apply_action(self._board, robot_id, direction)

        if reached_goal(self._board):
            reward += self.goal_reward
            terminated = True
            info["success"] = True

        if not terminated and self._num_steps >= self.max_steps:
            truncated = True

        obs = self._make_obs(self._board)
        info["steps"] = self._num_steps
        info["target_robot"] = self._board.target_robot
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._board is None:
            return ""
        from .ricochet_core import render_ascii  # local import to avoid circulars in lints
        return render_ascii(self._board)

    # Helpers
    def _decode_action(self, action: int) -> Tuple[int, int]:
        if not (0 <= action < self.action_space.n):
            raise ValueError(f"Invalid action {action}")
        if self.include_noop and action == self._noop_action:
            raise ValueError("_decode_action called on noop action")
        robot_id = action // 4
        direction = action % 4
        if robot_id >= self.num_robots:
            raise ValueError(f"Invalid robot_id {robot_id} for action {action}")
        return robot_id, direction

    def _make_obs(self, board: Board) -> np.ndarray:
        if self.obs_mode == "image":
            return self._board_to_image_obs(board)
        return self._board_to_symbolic_obs(board)

    def _board_to_image_obs(self, board: Board) -> np.ndarray:
        h, w = board.height, board.width
        C = self._num_channels
        obs = np.zeros((h, w, C), dtype=np.float32)
        # wall channels 0..3
        for r in range(h):
            for c in range(w):
                obs[r, c, 0] = 1.0 if board.has_wall_up(r, c) else 0.0
                obs[r, c, 1] = 1.0 if board.has_wall_down(r, c) else 0.0
                obs[r, c, 2] = 1.0 if board.has_wall_left(r, c) else 0.0
                obs[r, c, 3] = 1.0 if board.has_wall_right(r, c) else 0.0
        # goal channel 4
        gr, gc = board.goal_position
        obs[gr, gc, 4] = 1.0
        # target mask channel 5
        tr = board.target_robot
        tr_r, tr_c = board.robot_positions[tr]
        obs[tr_r, tr_c, 5] = 1.0
        # robot channels start at 6
        base = 6
        for rid, (r, c) in board.robot_positions.items():
            obs[r, c, base + rid] = 1.0
        if self.channels_first:
            obs = np.transpose(obs, (2, 0, 1))
        return obs

    def _board_to_symbolic_obs(self, board: Board) -> np.ndarray:
        gr, gc = board.goal_position
        target_one_hot = np.zeros((self.num_robots,), dtype=np.float32)
        target_one_hot[board.target_robot] = 1.0
        robots_vec: List[float] = []
        for rid in range(self.num_robots):
            r, c = board.robot_positions[rid]
            robots_vec.extend([float(r), float(c)])
        vec = np.concatenate([
            np.array([float(gr), float(gc)], dtype=np.float32),
            target_one_hot.astype(np.float32),
            np.array(robots_vec, dtype=np.float32),
        ])
        return vec

    def _generate_random_board(self) -> Board:
        h, w = self.height, self.width
        # Start with open grid, then add random interior edge walls, and border walls
        h_walls = np.zeros((h + 1, w), dtype=bool)
        v_walls = np.zeros((h, w + 1), dtype=bool)
        # Borders
        h_walls[0, :] = True
        h_walls[h, :] = True
        v_walls[:, 0] = True
        v_walls[:, w] = True
        # Random interior edges
        max_edges = max(1, (h * w) // 6)
        num_edges = int(self._rng.integers(0, max_edges))
        for _ in range(num_edges):
            if self._rng.integers(0, 2) == 0:
                rr = int(self._rng.integers(1, h))
                cc = int(self._rng.integers(0, w))
                h_walls[rr, cc] = True
            else:
                rr = int(self._rng.integers(0, h))
                cc = int(self._rng.integers(1, w))
                v_walls[rr, cc] = True
        # Place goal and robots in free cells
        free_cells: List[Tuple[int, int]] = [(r, c) for r in range(h) for c in range(w)]
        self._rng.shuffle(free_cells)
        goal = free_cells.pop()
        robot_positions: Dict[int, Tuple[int, int]] = {}
        for rid in range(self.num_robots):
            robot_positions[rid] = free_cells.pop()
        target_robot = int(self._rng.integers(0, self.num_robots))
        return Board(height=h, width=w, h_walls=h_walls, v_walls=v_walls, robot_positions=robot_positions, goal_position=goal, target_robot=target_robot)
