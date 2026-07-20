"""Lean fixed-layout Ricochet Robots training environment.

Differences from the legacy stack (see docs/GroundUpRLRedesign.md):

- Puzzles come from a preloaded in-memory table; resets are array ops plus a
  curriculum sample — no env-object reconstruction, no pandas.
- Canonical identity: the target robot is always robot 0 (obs channel 5,
  actions 0-3); non-target robots are sorted by initial position. Removes the
  target-mask indirection and the robot permutation redundancy.
- Potential-based reward shaping on the relaxed slide distance of the target
  robot (optimal policy preserving; Ng et al. 1999).
- Reverse starts: episodes may start part-way along the stored optimal plan.
- Native action masks for MaskablePPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .distance import relaxed_distance_map
from .puzzles import Puzzle

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 32
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    shaping_coef: float = 0.1
    gamma: float = 0.99  # must equal the PPO discount for valid shaping
    time_channel: bool = True
    canonical_identity: bool = True
    # One plane per (robot, direction): one-hot of the cell that robot would
    # stop at if it slid that way now. Injects one-step dynamics the flat
    # trunk otherwise has to learn by ray-tracing (E3 v2 diagnosis).
    stop_planes: bool = False
    # Normalized relaxed-distance-to-goal map (same field the shaping uses).
    distance_plane: bool = False
    # RGB observation mode: the board (walls/goal/robots) is delivered as a
    # rendered image instead of structured planes; helper planes (stop,
    # distance, time) ride alongside at cell resolution in a Dict obs.
    rgb_obs: bool = False
    rgb_cell_px: int = 5


class RicochetEnv(gym.Env):
    """Single-board environment over a fixed puzzle table.

    Observation (channels-first float32, values in [0, 1]):
      0-3  wall_up / wall_down / wall_left / wall_right
      4    goal
      5    target robot (always, via canonical relabelling)
      6+   remaining robots
      last remaining-steps fraction (constant plane), if time_channel
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        puzzles: Sequence[Puzzle],
        config: EnvConfig = EnvConfig(),
        sampler: Optional[object] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not puzzles:
            raise ValueError("Need at least one puzzle")
        self.puzzles = list(puzzles)
        first = self.puzzles[0]
        self.height, self.width = first.height, first.width
        self.num_robots = first.num_robots
        for p in self.puzzles:
            if (p.height, p.width, p.num_robots) != (self.height, self.width, self.num_robots):
                raise ValueError("All puzzles in one env must share geometry")
        self.config = config
        self.sampler = sampler
        self._rng = np.random.default_rng(seed)
        self._dmap_cache: Dict[int, np.ndarray] = {}

        helper_channels = 0
        if config.stop_planes:
            helper_channels += 4 * self.num_robots
        if config.distance_plane:
            helper_channels += 1
        if config.time_channel:
            helper_channels += 1
        if config.rgb_obs:
            if helper_channels == 0:
                raise ValueError("rgb_obs currently requires at least one helper plane")
            k = int(config.rgb_cell_px)
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(3, self.height * k, self.width * k), dtype=np.float32,
                    ),
                    "planes": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(helper_channels, self.height, self.width), dtype=np.float32,
                    ),
                }
            )
        else:
            channels = 5 + self.num_robots + helper_channels
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(channels, self.height, self.width), dtype=np.float32
            )
        self.action_space = spaces.Discrete(4 * self.num_robots)

        self._puzzle: Optional[Puzzle] = None
        self._robots: List[Tuple[int, int]] = []
        self._steps = 0
        self._start_distance = 0
        self._episode_done = True

    # ------------------------------------------------------------- helpers
    def _distance_map(self, puzzle: Puzzle) -> np.ndarray:
        cached = self._dmap_cache.get(puzzle.seed)
        if cached is None:
            cached = relaxed_distance_map(puzzle.h_walls, puzzle.v_walls, puzzle.goal)
            self._dmap_cache[puzzle.seed] = cached
        return cached

    def _static_planes(self, puzzle: Puzzle) -> np.ndarray:
        planes = np.zeros((5, self.height, self.width), dtype=np.float32)
        planes[0] = puzzle.h_walls[:-1, :]
        planes[1] = puzzle.h_walls[1:, :]
        planes[2] = puzzle.v_walls[:, :-1]
        planes[3] = puzzle.v_walls[:, 1:]
        planes[4, puzzle.goal[0], puzzle.goal[1]] = 1.0
        return planes

    def _start_positions(self, puzzle: Puzzle, start_distance: int) -> List[Tuple[int, int]]:
        """Robot positions (original id order) after replaying the plan prefix."""
        positions = list(puzzle.robots)
        prefix = puzzle.actions[: puzzle.optimal_length - start_distance]
        for rid, direction in prefix:
            positions[rid] = self._slide_from(positions, rid, direction, puzzle)
        return positions

    def _canonicalize(self, puzzle: Puzzle, positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not self.config.canonical_identity:
            return positions
        others = [positions[i] for i in range(self.num_robots) if i != puzzle.target]
        others.sort()
        return [positions[puzzle.target]] + others

    def _slide_from(
        self,
        positions: Sequence[Tuple[int, int]],
        rid: int,
        direction: int,
        puzzle: Puzzle,
    ) -> Tuple[int, int]:
        r, c = positions[rid]
        occupied = {pos for i, pos in enumerate(positions) if i != rid}
        h_walls, v_walls = puzzle.h_walls, puzzle.v_walls
        if direction == UP:
            while r > 0 and not h_walls[r, c] and (r - 1, c) not in occupied:
                r -= 1
        elif direction == DOWN:
            while r < self.height - 1 and not h_walls[r + 1, c] and (r + 1, c) not in occupied:
                r += 1
        elif direction == LEFT:
            while c > 0 and not v_walls[r, c] and (r, c - 1) not in occupied:
                c -= 1
        elif direction == RIGHT:
            while c < self.width - 1 and not v_walls[r, c + 1] and (r, c + 1) not in occupied:
                c += 1
        else:
            raise ValueError(f"Invalid direction {direction}")
        return (r, c)

    def _target_index(self) -> int:
        return 0 if self.config.canonical_identity else self._puzzle.target

    def _potential(self) -> float:
        if self.config.shaping_coef == 0.0:
            return 0.0
        dmap = self._distance_map(self._puzzle)
        tr, tc = self._robots[self._target_index()]
        return -self.config.shaping_coef * float(dmap[tr, tc])

    def _helper_planes(self) -> list:
        parts = []
        if self.config.stop_planes:
            stops = np.zeros((4 * self.num_robots, self.height, self.width), dtype=np.float32)
            for rid in range(self.num_robots):
                for direction in range(4):
                    sr, sc = self._slide_from(self._robots, rid, direction, self._puzzle)
                    stops[rid * 4 + direction, sr, sc] = 1.0
            parts.append(stops)
        if self.config.distance_plane:
            dmap = self._distance_map(self._puzzle).astype(np.float32)
            parts.append((dmap / max(1.0, float(dmap.max())))[None])
        if self.config.time_channel:
            remaining = (self.config.horizon - self._steps) / self.config.horizon
            parts.append(np.full((1, self.height, self.width), remaining, dtype=np.float32))
        return parts

    def _obs(self):
        if self.config.rgb_obs:
            from .render import render

            return {
                "image": render(self._puzzle, self._robots, int(self.config.rgb_cell_px)),
                "planes": np.concatenate(self._helper_planes(), axis=0),
            }
        planes = self._static_planes(self._puzzle)
        robot_planes = np.zeros((self.num_robots, self.height, self.width), dtype=np.float32)
        for i, (r, c) in enumerate(self._robots):
            robot_planes[i, r, c] = 1.0
        parts = [planes, robot_planes] + self._helper_planes()
        return np.concatenate(parts, axis=0)

    @property
    def stop_planes_offset(self) -> int:
        """First channel index of the stop planes (requires stop_planes=True)."""
        if not self.config.stop_planes:
            raise ValueError("stop_planes disabled")
        return 5 + self.num_robots

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "puzzle_index" in options:
            puzzle = self.puzzles[int(options["puzzle_index"])]
            start_distance = int(options.get("start_distance", puzzle.optimal_length))
        elif self.sampler is not None:
            puzzle, start_distance = self.sampler.sample(self._rng)
        else:
            puzzle = self.puzzles[int(self._rng.integers(len(self.puzzles)))]
            start_distance = puzzle.optimal_length
        if not 1 <= start_distance <= puzzle.optimal_length:
            raise ValueError(
                f"start_distance {start_distance} out of range for optimal_length {puzzle.optimal_length}"
            )
        self._puzzle = puzzle
        positions = self._start_positions(puzzle, start_distance)
        self._robots = self._canonicalize(puzzle, positions)
        self._steps = 0
        self._start_distance = start_distance
        self._episode_done = False
        info = {
            "optimal_length": puzzle.optimal_length,
            "start_distance": start_distance,
            "robots_moved": puzzle.robots_moved,
        }
        return self._obs(), info

    def step(self, action: int):
        if self._episode_done:
            raise RuntimeError("Episode finished; call reset()")
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"Invalid action {action}")
        rid, direction = action // 4, action % 4
        self._steps += 1

        phi_before = self._potential()
        self._robots[rid] = self._slide_from(self._robots, rid, direction, self._puzzle)
        tr, tc = self._robots[self._target_index()]
        success = (tr, tc) == self._puzzle.goal
        phi_after = self._potential()  # 0 at the goal by construction

        reward = self.config.step_penalty
        reward += self.config.gamma * phi_after - phi_before
        terminated = False
        truncated = False
        if success:
            reward += self.config.goal_reward
            terminated = True
        elif self._steps >= self.config.horizon:
            truncated = True

        info = {
            "is_success": success,
            "steps": self._steps,
            "optimal_length": self._puzzle.optimal_length,
            "start_distance": self._start_distance,
        }
        if terminated or truncated:
            self._episode_done = True
            if self.sampler is not None:
                self.sampler.record(self._start_distance, success, self._puzzle)
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """True where the action moves its robot at least one cell."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        puzzle = self._puzzle
        occupied = set(self._robots)
        for rid in range(self.num_robots):
            r, c = self._robots[rid]
            if r > 0 and not puzzle.h_walls[r, c] and (r - 1, c) not in occupied:
                mask[rid * 4 + UP] = True
            if r < self.height - 1 and not puzzle.h_walls[r + 1, c] and (r + 1, c) not in occupied:
                mask[rid * 4 + DOWN] = True
            if c > 0 and not puzzle.v_walls[r, c] and (r, c - 1) not in occupied:
                mask[rid * 4 + LEFT] = True
            if c < self.width - 1 and not puzzle.v_walls[r, c + 1] and (r, c + 1) not in occupied:
                mask[rid * 4 + RIGHT] = True
        return mask
