from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Literal, Set

import numpy as np

from .ricochet_core import Board, apply_action, reached_goal
FIXED_PIXEL_SIZE = 128 # This needs to be board size * the default cell size, otherwise we get errors caused by a mismatch between the obs/render (with the fixed cache size) TODO fix this: observe the error by changing this or the defult cell size, and then running the bank_curriculum_preview.py

# Provide a runtime gymnasium binding if available, otherwise a minimal fallback
try:  # pragma: no cover
    import importlib
    _gym = importlib.import_module("gymnasium")  # type: ignore
    _spaces = importlib.import_module("gymnasium.spaces")  # type: ignore
    _seeding = importlib.import_module("gymnasium.utils.seeding")  # type: ignore
    _error = importlib.import_module("gymnasium.error")  # type: ignore
    GymEnvBase = _gym.Env  # type: ignore
    spaces = _spaces  # type: ignore
    GymInvalidAction = getattr(_error, "InvalidAction", None)  # type: ignore
except (ModuleNotFoundError, ImportError):  # pragma: no cover
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
    GymInvalidAction = None  # type: ignore


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


ObsMode = Literal["image", "symbolic", "rgb_image"]


class RicochetRobotsEnv(GymEnvBase):
    """Ricochet Robots Gymnasium-compatible environment.

    Determinism and seeding:
    - Deterministic given the seed provided to __init__ or reset(seed=...).
    - Randomness flows through self.np_random (Gymnasium seeding utils); the episode's resolved
      seed is exposed via info["episode_seed"].
    - When ensure_solvable=True, the environment may retry board generation; retries do not add
      extra non-determinism beyond RNG draws. The included solver is deterministic.

    Observations:
    - Image mode: float32 in [0,1]. Channels are named by channel_names; note that the target robot
      is represented both by a dedicated mask (index 5) and its robot channel (duplication by design).
    - Symbolic mode: float32 with absolute grid indices (not normalized). Downstream code may wish to
      normalize to [0,1]. The Box low/high reflect board bounds.

    Rendering:
    - Only render_mode="ascii" is implemented and returns a string frame.
    - Other modes are not implemented and will raise a clear error.
    """
    metadata = {"render_modes": ["ascii", "rgb"], "render_fps": 4}
    # Class-level cache for RGB backgrounds per (H, W, cell_size, grid_color, grid_th, wall_color, wall_th)
    _RGB_BG_CACHE: Dict[Tuple[int, int, int, Tuple[int, int, int], int, Tuple[int, int, int], int], np.ndarray] = {}

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        num_robots: int = 2,
        include_noop: bool = True,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        noop_penalty: Optional[float] = None,
        max_steps: int = 10,
        fixed_layout: Optional[FixedLayout] = None,
        seed: Optional[int] = None,
        ensure_solvable: bool = False,
        solver_max_depth: int = 30,
        solver_max_nodes: int = 20000,
        obs_mode: ObsMode = "image",
        channels_first: bool = False,
        render_mode: Optional[str] = None,
        ensure_attempt_limit: int = 500,
        # Spec-driven generation configuration
        edge_t_per_quadrant: Optional[int] = None,
        central_l_per_quadrant: Optional[int] = None,
        # RGB render configuration
        render_rgb_config: Optional[Dict[str, object]] = None,
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
        # If a fixed layout is provided, align env geometry before building spaces
        if self.fixed_layout is not None:
            self.height = int(self.fixed_layout.height)
            self.width = int(self.fixed_layout.width)
            self.num_robots = int(len(self.fixed_layout.robot_positions))
        # Render API
        self.render_mode = render_mode
        if self.render_mode is not None and self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {self.render_mode}. Supported: {self.metadata['render_modes']}")
        # RGB render defaults
        self._render_rgb_cfg: Dict[str, object] = {
            "cell_size": 7,
            "grid_color": (200, 200, 200),
            "grid_thickness": 1,
            "wall_color": (0, 0, 0),
            "wall_thickness": 3,
            "robot_colors": [
                (220, 50, 50),     # red
                (50, 120, 220),    # blue
                (50, 180, 80),     # green
                (230, 200, 40),    # yellow
                (180, 80, 180),    # purple (extra if more robots)
            ],
            "circle_fill": True,
            "circle_radius_frac": 0.35,
            "target_dark_factor": 0.6,
            "star_thickness": 2,
        }
        if render_rgb_config is not None:
            self._render_rgb_cfg.update(render_rgb_config)
        # Per-instance cache: RGB internal walls overlay built for current layout (excludes boundary walls)
        self._rgb_internal_walls_overlay: Optional[np.ndarray] = None

        # Seeding: adopt Gymnasium convention with self.np_random
        self.np_random: np.random.Generator
        self._episode_seed: Optional[int] = None
        try:  # gymnasium present
            # _seeding.np_random returns (rng, seed)
            self.np_random, used_seed = _seeding.np_random(seed)  # type: ignore[name-defined]
            self._episode_seed = int(used_seed) if used_seed is not None else None
        except NameError:
            self.np_random = np.random.default_rng(seed)
            self._episode_seed = int(seed) if seed is not None else None
        # Preserve a base seed to enforce deterministic resets when no new seed is provided
        self._base_seed: Optional[int] = int(seed) if seed is not None else self._episode_seed
        self._board: Optional[Board] = None
        self._num_steps = 0
        self._cached_wall_obs: Optional[np.ndarray] = None  # (H,W,4) float32 for wall channels
        self._ensure_attempt_limit = int(ensure_attempt_limit)
        # Spec default counts per quadrant (based on smaller board side)
        side_min = min(self.height, self.width)
        default_per_quad = 2 if side_min < 10 else 4
        self._edge_t_per_quadrant = default_per_quad if edge_t_per_quadrant is None else int(edge_t_per_quadrant)
        self._central_l_per_quadrant = default_per_quad if central_l_per_quadrant is None else int(central_l_per_quadrant)
        # Track the exact per-episode board seed used to generate the current board
        self._last_board_seed: Optional[int] = None

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
        elif self.obs_mode == "rgb_image":
            # RGB image observation: uses fixed pixel dimensions for consistent observation space
            # This ensures all curriculum levels have the same observation space size
            # Lowered from 128 to 96 to reduce memory/compute; resize handled in _board_to_rgb_obs
            
            if self.channels_first:
                obs_shape = (3, FIXED_PIXEL_SIZE, FIXED_PIXEL_SIZE)
            else:
                obs_shape = (FIXED_PIXEL_SIZE, FIXED_PIXEL_SIZE, 3)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=obs_shape,
                dtype=np.uint8,
            )
        elif self.obs_mode == "symbolic":
            # [goal_r, goal_c] in [0..H-1], [0..W-1]; one-hot target in [0,1]; robots positions in bounds
            vec_len = 2 + self.num_robots + (self.num_robots * 2)
            low = np.full((vec_len,), 0.0, dtype=np.float32)
            high = np.full((vec_len,), 1.0, dtype=np.float32)
            # goal bounds
            low[0] = 0.0
            high[0] = float(self.height - 1)
            low[1] = 0.0
            high[1] = float(self.width - 1)
            # target one-hot already [0,1]
            # robots positions
            base = 2 + self.num_robots
            for i in range(self.num_robots):
                low[base + 2 * i] = 0.0
                high[base + 2 * i] = float(self.height - 1)
                low[base + 2 * i + 1] = 0.0
                high[base + 2 * i + 1] = float(self.width - 1)
            self.observation_space = spaces.Box(low=low, high=high, shape=(vec_len,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

        # Action space: num_robots * 4 directions (+ 1 for noop)
        self._noop_action = self.num_robots * 4 if include_noop else None
        self.action_space = spaces.Discrete(self.num_robots * 4 + (1 if include_noop else 0))

    # Gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and start a new episode.

        Returns (obs, info) per Gymnasium API. info contains keys: 'target_robot',
        'level_solvable' (optional), 'optimal_length' (if ensured), 'solver_limits' (if ensured),
        and 'episode_seed' with the seed used for this episode.
        """
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()

        # Standard seeding protocol
        if seed is not None:
            try:
                self.np_random, used_seed = _seeding.np_random(seed)  # type: ignore[name-defined]
                self._episode_seed = int(used_seed) if used_seed is not None else None
            except NameError:
                self.np_random = np.random.default_rng(seed)
                self._episode_seed = int(seed)
            self._base_seed = int(self._episode_seed) if self._episode_seed is not None else None
        else:
            # Re-seed with base seed so that resets are deterministic when no new seed is provided
            if self._base_seed is not None:
                try:
                    self.np_random, used_seed = _seeding.np_random(self._base_seed)  # type: ignore[name-defined]
                    self._episode_seed = int(used_seed) if used_seed is not None else int(self._base_seed)
                except NameError:
                    self.np_random = np.random.default_rng(self._base_seed)
                    self._episode_seed = int(self._base_seed)
        _ = options  # silence unused
        self._num_steps = 0
        # clear cached walls in case layout changes
        self._cached_wall_obs = None

        if self.fixed_layout is not None:
            with profile("env_reset_fixed_layout", track_memory=True):
                self._board = Board(
                height=self.fixed_layout.height,
                width=self.fixed_layout.width,
                h_walls=self.fixed_layout.h_walls.copy(),
                v_walls=self.fixed_layout.v_walls.copy(),
                robot_positions=dict(self.fixed_layout.robot_positions),
                goal_position=self.fixed_layout.goal_position,
                target_robot=self.fixed_layout.target_robot,
            )
            # cache static walls for this layout
            with profile("env_build_cached_walls", track_memory=True):
                self._build_cached_walls(self._board)
            # Build per-layout internal walls RGB overlay cache (for rgb render mode)
            try:
                if self.render_mode == "rgb":
                    self._build_rgb_internal_walls_overlay(self._board)
            except Exception:
                self._rgb_internal_walls_overlay = None
            with profile("env_make_obs", track_memory=True):
                obs = self._make_obs(self._board)
            info = {
                "level_solvable": False,
                "ensure_solvable_enabled": False,
                "target_robot": self._board.target_robot,
                "episode_seed": self._episode_seed,
            }
            if self.obs_mode == "image":
                info["channel_names"] = self.channel_names
            return obs, info

        # Otherwise, generate random board, optionally ensure solvable via solver
        attempts = 0
        while True:
            attempts += 1
            with profile("env_generate_board", track_memory=True):
                board = self._generate_random_board()
            if not self.ensure_solvable:
                self._board = board
                break
            try:
                from .solver import solve_bfs  # type: ignore
                with profile("env_solver_check", track_memory=True):
                    solution = solve_bfs(board, max_depth=self.solver_max_depth, max_nodes=self.solver_max_nodes)
                if solution is not None:
                    self._board = board
                    optimal_length = len(solution)
                    break
            except ImportError as exc:
                raise RuntimeError("ensure_solvable=True but solver unavailable (ImportError)") from exc
            if attempts >= self._ensure_attempt_limit:
                raise RuntimeError("Failed to generate solvable board within attempt limit")

        # cache static walls for this layout
        assert self._board is not None
        with profile("env_build_cached_walls", track_memory=True):
            self._build_cached_walls(self._board)
        # Build per-layout internal walls RGB overlay cache (for rgb render mode)
        try:
            if self.render_mode == "rgb":
                self._build_rgb_internal_walls_overlay(self._board)
        except Exception:
            self._rgb_internal_walls_overlay = None
        with profile("env_make_obs", track_memory=True):
            obs = self._make_obs(self._board)
        # Note: solver None during attempts may be due to cutoffs (depth/nodes), not true unsolvability.
        info = {
            "level_solvable": bool(self.ensure_solvable),
            "ensure_solvable_enabled": bool(self.ensure_solvable),
            "target_robot": self._board.target_robot,
            "episode_seed": self._episode_seed,
        }
        # Expose the exact per-episode board seed used to build this board
        if hasattr(self, "_last_board_seed") and self._last_board_seed is not None:
            info["board_seed"] = int(self._last_board_seed)
        if self.obs_mode == "image":
            info["channel_names"] = self.channel_names
        if self.ensure_solvable:
            info["optimal_length"] = optimal_length
            info["solver_limits"] = {"max_depth": self.solver_max_depth, "max_nodes": self.solver_max_nodes}
        return obs, info

    def step(self, action: int):
        """Apply an action and return (obs, reward, terminated, truncated, info).

        info always includes 'steps', 'target_robot', and 'is_success' (bool). When truncated due to
        max steps, info includes 'TimeLimit.truncated'=True.
        """
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()

        assert self._board is not None, "Call reset() first"
        self._num_steps += 1

        terminated = False
        truncated = False
        info: Dict[str, object] = {}

        reward = self.step_penalty
        # Validate action range early
        if not (0 <= action < self.action_space.n):
            if GymInvalidAction is not None:
                raise GymInvalidAction(f"Invalid action {action}")  # type: ignore[misc]
            raise ValueError(f"Invalid action {action}")

        if self.include_noop and action == self._noop_action:
            reward = self.noop_penalty
        else:
            with profile("env_apply_action", track_memory=True):
                robot_id, direction = self._decode_action(action)
                self._board = apply_action(self._board, robot_id, direction)

        if reached_goal(self._board):
            reward += self.goal_reward
            terminated = True
            info["is_success"] = True

        if not terminated and self._num_steps >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True

        with profile("env_make_obs", track_memory=True):
            obs = self._make_obs(self._board)
        info["steps"] = self._num_steps
        info["target_robot"] = self._board.target_robot
        if "is_success" not in info:
            info["is_success"] = False
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render according to render_mode.

        - render_mode=="ascii": returns a string frame.
        - render_mode=="rgb": returns an (H_px, W_px, 3) uint8 array.
        - render_mode is None or unsupported: returns None.
        """
        if self.render_mode is None:
            return None
        if self._board is None:
            return "" if self.render_mode == "ascii" else None
        if self.render_mode == "ascii":
            from .ricochet_core import render_ascii  # local import to avoid circulars in lints
            return render_ascii(self._board)
        if self.render_mode == "rgb":
            return self._render_rgb(self._board)
        if self._board is None:
            return ""
        raise ValueError(f"Unsupported render_mode {self.render_mode}. Supported: {self.metadata['render_modes']}")

    def close(self) -> None:
        """Close environment resources (no-op)."""
        return None

    def seed(self, seed: Optional[int] = None):  # legacy convenience
        """Legacy seeding helper returning [seed]. Prefer reset(seed=...)."""
        if seed is None:
            return [self._episode_seed]
        try:
            self.np_random, used_seed = _seeding.np_random(seed)  # type: ignore[name-defined]
            self._episode_seed = int(used_seed) if used_seed is not None else int(seed)
        except NameError:
            self.np_random = np.random.default_rng(seed)
            self._episode_seed = int(seed)
        return [self._episode_seed]

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
        elif self.obs_mode == "rgb_image":
            return self._board_to_rgb_obs(board)
        return self._board_to_symbolic_obs(board)

    def _board_to_image_obs(self, board: Board) -> np.ndarray:
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()

        with profile("env_image_observation_generation", track_memory=True):
            h, w = board.height, board.width
            C = self._num_channels
            obs = np.zeros((h, w, C), dtype=np.float32)
            # wall channels 0..3 from cache
            if self._cached_wall_obs is not None:
                obs[:, :, 0:4] = self._cached_wall_obs
            else:
                # Fallback if cache not built (should not happen)
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
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()

        with profile("env_symbolic_observation_generation", track_memory=True):
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

    def _board_to_rgb_obs(self, board: Board) -> np.ndarray:
        """Convert board to RGB image observation with fixed pixel dimensions.
        
        This creates an RGB image with fixed pixel dimensions (128x128) by adjusting
        the cell size to fit the board within the fixed pixel space. This ensures
        consistent observation space across all curriculum levels.
        """
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()
        
        with profile("env_rgb_observation_generation", track_memory=True):
            # Padding around the board within the fixed output
            padding = 4
            available_size = FIXED_PIXEL_SIZE - 2 * padding
            cell_size = max(1, available_size // max(board.height, board.width))
            # Compute board pixel dims and canvas
            H_px = int(board.height * cell_size)
            W_px = int(board.width * cell_size)
            # Compose into fixed-size canvas without resizing
            canvas = np.ones((FIXED_PIXEL_SIZE, FIXED_PIXEL_SIZE, 3), dtype=np.uint8) * 255
            y0 = padding + (available_size - H_px) // 2
            x0 = padding + (available_size - W_px) // 2
            # Render board region with caching
            with profile("env_rgb_rendering_cached", track_memory=True):
                region = self._render_rgb_cached(board, cell_size)
            canvas[y0:y0+H_px, x0:x0+W_px] = region
            if self.channels_first:
                canvas = np.transpose(canvas, (2, 0, 1))
            return canvas

    def _generate_random_board(self) -> Board:
        # Draw a per-episode board seed from the env RNG
        seed_used = int(self.np_random.integers(0, 2**31 - 1))
        # Record it for reproducibility/debugging
        self._last_board_seed = seed_used
        # Generate deterministically from the explicit seed only (no external RNG)
        return generate_board_from_spec_with_seed(
            height=self.height,
            width=self.width,
            num_robots=self.num_robots,
            edge_t_per_quadrant=self._edge_t_per_quadrant,
            central_l_per_quadrant=self._central_l_per_quadrant,
            seed=seed_used,
        )

    # Public channel names for image observations
    @property
    def channel_names(self) -> List[str]:
        """Human-readable names for image observation channels."""
        names = ["wall_up", "wall_down", "wall_left", "wall_right", "goal", "target_robot_mask"]
        names.extend([f"robot_{i}" for i in range(self.num_robots)])
        return names

    def _build_cached_walls(self, board: Board) -> None:
        """Precompute static wall indicator channels (H,W,4)."""
        h, w = board.height, board.width
        wall = np.zeros((h, w, 4), dtype=np.float32)
        for r in range(h):
            for c in range(w):
                wall[r, c, 0] = 1.0 if board.has_wall_up(r, c) else 0.0
                wall[r, c, 1] = 1.0 if board.has_wall_down(r, c) else 0.0
                wall[r, c, 2] = 1.0 if board.has_wall_left(r, c) else 0.0
                wall[r, c, 3] = 1.0 if board.has_wall_right(r, c) else 0.0
        self._cached_wall_obs = wall

    # Public accessor for current board (read-only clone)
    def get_board(self) -> Board:
        if self._board is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._board.clone()

    def _render_rgb(self, board: Board) -> np.ndarray:
        # Import profiling here to avoid circular imports
        try:
            from ..profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(name, track_memory=True):
                from contextlib import nullcontext
                return nullcontext()
        
        with profile("env_rgb_render_setup", track_memory=True):
            cfg = self._render_rgb_cfg
            cell_size = int(cfg["cell_size"])  # pixels
            grid_color = tuple(int(x) for x in cfg["grid_color"])  # type: ignore[index]
            grid_th = int(cfg["grid_thickness"])  # px
            wall_color = tuple(int(x) for x in cfg["wall_color"])  # type: ignore[index]
            wall_th = int(cfg["wall_thickness"])  # px
            robot_colors: List[Tuple[int, int, int]] = [tuple(int(a) for a in t) for t in cfg["robot_colors"]]  # type: ignore[index]
            circle_fill: bool = bool(cfg["circle_fill"])  # type: ignore[index]
            circle_r_frac: float = float(cfg["circle_radius_frac"])  # type: ignore[index]
            target_dark_factor: float = float(cfg["target_dark_factor"])  # type: ignore[index]
            star_th: int = int(cfg["star_thickness"])  # type: ignore[index]

            H, W = board.height, board.width
            H_px, W_px = H * cell_size, W * cell_size
            # Start from cached background with grid+boundary walls
            bg = self._get_rgb_background(H, W, cell_size, grid_color, grid_th, wall_color, wall_th)
            img = bg.copy()

        # Helpers to draw
        def draw_hline(y: int, x0: int, x1: int, color: Tuple[int, int, int], thickness: int) -> None:
            y0 = max(0, y - thickness // 2)
            y1 = min(H_px, y + (thickness - thickness // 2))
            x0c = max(0, min(x0, x1))
            x1c = min(W_px, max(x0, x1))
            img[y0:y1, x0c:x1c] = color

        def draw_vline(x: int, y0: int, y1: int, color: Tuple[int, int, int], thickness: int) -> None:
            x0 = max(0, x - thickness // 2)
            x1 = min(W_px, x + (thickness - thickness // 2))
            y0c = max(0, min(y0, y1))
            y1c = min(H_px, max(y0, y1))
            img[y0c:y1c, x0:x1] = color

        def draw_circle(cx: float, cy: float, radius: float, color: Tuple[int, int, int], fill: bool) -> None:
            # Bounding box
            x0 = int(max(0, np.floor(cx - radius)))
            x1 = int(min(W_px - 1, np.ceil(cx + radius)))
            y0 = int(max(0, np.floor(cy - radius)))
            y1 = int(min(H_px - 1, np.ceil(cy + radius)))
            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            if fill:
                img[y0:y1 + 1, x0:x1 + 1][mask] = color
            else:
                # outline 1px
                inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius - 1) ** 2
                ring = np.logical_and(mask, np.logical_not(inner))
                img[y0:y1 + 1, x0:x1 + 1][ring] = color

        def draw_star(cx: float, cy: float, arm_len: float, color: Tuple[int, int, int], thickness: int) -> None:
            # Draw + arms
            draw_hline(int(cy), int(cx - arm_len), int(cx + arm_len), color, thickness)
            draw_vline(int(cx), int(cy - arm_len), int(cy + arm_len), color, thickness)
            # Draw x arms (diagonals) by sampling small line thickness via plotting short lines of squares
            # Approximate diagonals by drawing small rectangles along the diagonal
            steps = int(arm_len)
            for s in range(-steps, steps + 1):
                x = int(cx + s)
                y1 = int(cy + s)
                y2 = int(cy - s)
                y0a = max(0, y1 - thickness // 2)
                y1a = min(H_px, y1 + (thickness - thickness // 2))
                x0a = max(0, x - thickness // 2)
                x1a = min(W_px, x + (thickness - thickness // 2))
                img[y0a:y1a, x0a:x1a] = color
                y0b = max(0, y2 - thickness // 2)
                y1b = min(H_px, y2 + (thickness - thickness // 2))
                img[y0b:y1b, x0a:x1a] = color

        # 1) Internal walls overlay (excluding boundary walls) drawn once per layout
        with profile("env_rgb_draw_internal_walls", track_memory=True):
            if self._rgb_internal_walls_overlay is None:
                self._build_rgb_internal_walls_overlay(board)
            if self._rgb_internal_walls_overlay is not None:
                overlay = self._rgb_internal_walls_overlay
                mask = (overlay.sum(axis=2) < 3 * 255)
                img[mask] = overlay[mask]

        # 3) Draw robots as filled circles
        with profile("env_rgb_draw_robots", track_memory=True):
            radius = cell_size * float(circle_r_frac)
            for rid, (rr, cc) in board.robot_positions.items():
                color = robot_colors[rid % len(robot_colors)]
                cy = rr * cell_size + cell_size * 0.5
                cx = cc * cell_size + cell_size * 0.5
                draw_circle(cx, cy, radius, color, circle_fill)

        # 4) Draw target as a star in darker color of target robot
        with profile("env_rgb_draw_target", track_memory=True):
            tr = board.target_robot
            tr_color = robot_colors[tr % len(robot_colors)]
            dark_color = tuple(int(max(0, min(255, target_dark_factor * v))) for v in tr_color)
            gr, gc = board.goal_position
            cy = gr * cell_size + cell_size * 0.5
            cx = gc * cell_size + cell_size * 0.5
            draw_star(cx, cy, arm_len=cell_size * 0.35, color=dark_color, thickness=star_th)

        return img

    def _render_rgb_cached(self, board: Board, cell_size: int) -> np.ndarray:
        # Compose cached background + internal walls overlay + dynamic robots/target into board-sized image
        cfg = self._render_rgb_cfg
        grid_color = tuple(int(x) for x in cfg["grid_color"])  # type: ignore[index]
        grid_th = int(cfg["grid_thickness"])  # px
        wall_color = tuple(int(x) for x in cfg["wall_color"])  # type: ignore[index]
        wall_th = int(cfg["wall_thickness"])  # px
        robot_colors: List[Tuple[int, int, int]] = [tuple(int(a) for a in t) for t in cfg["robot_colors"]]  # type: ignore[index]
        circle_fill: bool = bool(cfg["circle_fill"])  # type: ignore[index]
        circle_r_frac: float = float(cfg["circle_radius_frac"])  # type: ignore[index]
        target_dark_factor: float = float(cfg["target_dark_factor"])  # type: ignore[index]
        star_th: int = int(cfg["star_thickness"])  # type: ignore[index]

        H, W = board.height, board.width
        H_px, W_px = H * cell_size, W * cell_size
        bg = self._get_rgb_background(H, W, cell_size, grid_color, grid_th, wall_color, wall_th)
        img = bg.copy()
        if self._rgb_internal_walls_overlay is None or self._rgb_internal_walls_overlay.shape[:2] != (H_px, W_px):
            self._build_rgb_internal_walls_overlay(board, cell_size_override=cell_size)
        if self._rgb_internal_walls_overlay is not None:
            overlay = self._rgb_internal_walls_overlay
            mask = (overlay.sum(axis=2) < 3 * 255)
            img[mask] = overlay[mask]

        # Inline draw helpers for robots/target
        def draw_hline(y: int, x0: int, x1: int, color: Tuple[int, int, int], thickness: int) -> None:
            y0 = max(0, y - thickness // 2)
            y1 = min(H_px, y + (thickness - thickness // 2))
            x0c = max(0, min(x0, x1))
            x1c = min(W_px, max(x0, x1))
            img[y0:y1, x0c:x1c] = color

        def draw_vline(x: int, y0: int, y1: int, color: Tuple[int, int, int], thickness: int) -> None:
            x0 = max(0, x - thickness // 2)
            x1 = min(W_px, x + (thickness - thickness // 2))
            y0c = max(0, min(y0, y1))
            y1c = min(H_px, max(y0, y1))
            img[y0c:y1c, x0:x1] = color

        def draw_circle(cx: float, cy: float, radius: float, color: Tuple[int, int, int], fill: bool) -> None:
            x0 = int(max(0, np.floor(cx - radius)))
            x1 = int(min(W_px - 1, np.ceil(cx + radius)))
            y0 = int(max(0, np.floor(cy - radius)))
            y1 = int(min(H_px - 1, np.ceil(cy + radius)))
            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            if fill:
                img[y0:y1 + 1, x0:x1 + 1][mask] = color
            else:
                inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius - 1) ** 2
                ring = np.logical_and(mask, np.logical_not(inner))
                img[y0:y1 + 1, x0:x1 + 1][ring] = color

        def draw_star(cx: float, cy: float, arm_len: float, color: Tuple[int, int, int], thickness: int) -> None:
            draw_hline(int(cy), int(cx - arm_len), int(cx + arm_len), color, thickness)
            draw_vline(int(cx), int(cy - arm_len), int(cy + arm_len), color, thickness)
            steps = int(arm_len)
            for s in range(-steps, steps + 1):
                x = int(cx + s)
                y1 = int(cy + s)
                y2 = int(cy - s)
                y0a = max(0, y1 - thickness // 2)
                y1a = min(H_px, y1 + (thickness - thickness // 2))
                x0a = max(0, x - thickness // 2)
                x1a = min(W_px, x + (thickness - thickness // 2))
                img[y0a:y1a, x0a:x1a] = color
                y0b = max(0, y2 - thickness // 2)
                y1b = min(H_px, y2 + (thickness - thickness // 2))
                img[y0b:y1b, x0a:x1a] = color

        # Draw robots
        radius = cell_size * float(circle_r_frac)
        for rid, (rr, cc) in board.robot_positions.items():
            color = robot_colors[rid % len(robot_colors)]
            cy = rr * cell_size + cell_size * 0.5
            cx = cc * cell_size + cell_size * 0.5
            draw_circle(cx, cy, radius, color, circle_fill)

        # Draw target star
        tr = board.target_robot
        tr_color = robot_colors[tr % len(robot_colors)]
        dark_color = tuple(int(max(0, min(255, target_dark_factor * v))) for v in tr_color)
        gr, gc = board.goal_position
        cy = gr * cell_size + cell_size * 0.5
        cx = gc * cell_size + cell_size * 0.5
        draw_star(cx, cy, arm_len=cell_size * 0.35, color=dark_color, thickness=star_th)
        return img

    def _get_rgb_background(self, H: int, W: int, cell_size: int, grid_color: Tuple[int, int, int], grid_th: int, wall_color: Tuple[int, int, int], wall_th: int) -> np.ndarray:
        key = (int(H), int(W), int(cell_size), tuple(int(x) for x in grid_color), int(grid_th), tuple(int(x) for x in wall_color), int(wall_th))
        cached = RicochetRobotsEnv._RGB_BG_CACHE.get(key)
        if cached is not None:
            return cached
        H_px, W_px = int(H * cell_size), int(W * cell_size)
        img = np.ones((H_px, W_px, 3), dtype=np.uint8) * 255
        # Draw grid
        for r in range(H + 1):
            y = int(r * cell_size)
            y0 = max(0, y - grid_th // 2)
            y1 = min(H_px, y + (grid_th - grid_th // 2))
            img[y0:y1, 0:W_px] = grid_color
        for c in range(W + 1):
            x = int(c * cell_size)
            x0 = max(0, x - grid_th // 2)
            x1 = min(W_px, x + (grid_th - grid_th // 2))
            img[0:H_px, x0:x1] = grid_color
        # Draw boundary walls (top/bottom/left/right)
        # Horizontal top and bottom
        y = 0
        y0 = max(0, y - wall_th // 2)
        y1 = min(H_px, y + (wall_th - wall_th // 2))
        img[y0:y1, 0:W_px] = wall_color
        y = H_px
        y0 = max(0, y - wall_th // 2)
        y1 = min(H_px, y + (wall_th - wall_th // 2))
        img[y0:y1, 0:W_px] = wall_color
        # Vertical left and right
        x = 0
        x0 = max(0, x - wall_th // 2)
        x1 = min(W_px, x + (wall_th - wall_th // 2))
        img[0:H_px, x0:x1] = wall_color
        x = W_px
        x0 = max(0, x - wall_th // 2)
        x1 = min(W_px, x + (wall_th - wall_th // 2))
        img[0:H_px, x0:x1] = wall_color
        RicochetRobotsEnv._RGB_BG_CACHE[key] = img
        return img

    def _build_rgb_internal_walls_overlay(self, board: Board, cell_size_override: Optional[int] = None) -> None:
        cfg = self._render_rgb_cfg
        cell_size = int(cell_size_override if cell_size_override is not None else cfg["cell_size"])  # type: ignore[index]
        wall_color = tuple(int(x) for x in cfg["wall_color"])  # type: ignore[index]
        wall_th = int(cfg["wall_thickness"])  # type: ignore[index]
        H, W = board.height, board.width
        H_px, W_px = H * cell_size, W * cell_size
        overlay = np.ones((H_px, W_px, 3), dtype=np.uint8) * 255
        # Draw only interior walls (exclude boundaries at r=0,r=H,c=0,c=W)
        for r in range(1, H):
            for c in range(W):
                if board.h_walls[r, c]:
                    y = int(r * cell_size)
                    y0 = max(0, y - wall_th // 2)
                    y1 = min(H_px, y + (wall_th - wall_th // 2))
                    x0 = int(c * cell_size)
                    x1 = int((c + 1) * cell_size)
                    overlay[y0:y1, x0:x1] = wall_color
        for r in range(H):
            for c in range(1, W):
                if board.v_walls[r, c]:
                    x = int(c * cell_size)
                    x0 = max(0, x - wall_th // 2)
                    x1 = min(W_px, x + (wall_th - wall_th // 2))
                    y0 = int(r * cell_size)
                    y1 = int((r + 1) * cell_size)
                    overlay[y0:y1, x0:x1] = wall_color
        self._rgb_internal_walls_overlay = overlay


def generate_board_from_spec_with_seed(
    *,
    height: int,
    width: int,
    num_robots: int,
    edge_t_per_quadrant: int,
    central_l_per_quadrant: int,
    seed: int,
    _rng: Optional[np.random.Generator] = None,
) -> Board:
    """Generate a Board deterministically from spec and seed without constructing the env.

    If an RNG is provided, it will be used (and its state advanced deterministically);
    otherwise, a fresh Generator seeded with 'seed' is created.
    """
    h, w = int(height), int(width)
    assert h == w, "Boards should always be square per specification"
    rng = _rng if _rng is not None else np.random.default_rng(int(seed))

    # Initialize empty walls then set borders
    h_walls = np.zeros((h + 1, w), dtype=bool)
    v_walls = np.zeros((h, w + 1), dtype=bool)
    h_walls[0, :] = True
    h_walls[h, :] = True
    v_walls[:, 0] = True
    v_walls[:, w] = True

    reserved_cells: Set[Tuple[int, int]] = set()
    def reserve_cell(cell: Tuple[int, int]) -> None:
        r, c = cell
        if 0 <= r < h and 0 <= c < w:
            reserved_cells.add((r, c))

    # Forbidden central block
    if h % 2 == 1:
        mid = h // 2
        h_walls[mid, mid] = True
        h_walls[mid + 1, mid] = True
        v_walls[mid, mid] = True
        v_walls[mid, mid + 1] = True
        reserve_cell((mid, mid))
        central_rows = [mid]
        central_cols = [mid]
    else:
        mid = h // 2
        h_walls[mid-1, mid - 1] = True
        h_walls[mid-1, mid] = True
        h_walls[mid + 1, mid - 1] = True
        h_walls[mid + 1, mid] = True
        v_walls[mid - 1, mid-1] = True
        v_walls[mid, mid-1] = True
        v_walls[mid - 1, mid + 1] = True
        v_walls[mid, mid + 1] = True
        for rr in (mid - 1, mid):
            for cc in (mid - 1, mid):
                reserve_cell((rr, cc))
        central_rows = [mid - 1, mid]
        central_cols = [mid - 1, mid]

    def ranges_excluding_central(size: int, central_idxs: List[int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        unique = sorted(set(central_idxs))
        if len(unique) == 1:
            mid_idx = unique[0]
            return (0, mid_idx - 1), (mid_idx + 1, size - 1)
        else:
            low_end = unique[0] - 1
            high_start = unique[1] + 1
            return (0, low_end), (high_start, size - 1)

    (r0_start, r0_end), (r1_start, r1_end) = ranges_excluding_central(h, central_rows)
    (c0_start, c0_end), (c1_start, c1_end) = ranges_excluding_central(w, central_cols)
    quadrants = {
        "NW": (r0_start, r0_end, c0_start, c0_end),
        "NE": (r0_start, r0_end, c1_start, c1_end),
        "SW": (r1_start, r1_end, c0_start, c0_end),
        "SE": (r1_start, r1_end, c1_start, c1_end),
    }

    used_interior_h: Set[Tuple[int, int]] = set()
    used_interior_v: Set[Tuple[int, int]] = set()

    def cell_has_nonboundary_wall(r: int, c: int) -> bool:
        if r > 0 and h_walls[r, c]:
            return True
        if r + 1 < h and h_walls[r + 1, c]:
            return True
        if c > 0 and v_walls[r, c]:
            return True
        if c + 1 < w and v_walls[r, c + 1]:
            return True
        return False

    def neighbors4(r: int, c: int) -> List[Tuple[int, int]]:
        nbrs: List[Tuple[int, int]] = []
        if r - 1 >= 0:
            nbrs.append((r - 1, c))
        if r + 1 < h:
            nbrs.append((r + 1, c))
        if c - 1 >= 0:
            nbrs.append((r, c - 1))
        if c + 1 < w:
            nbrs.append((r, c + 1))
        return nbrs

    def can_place_h(row: int, col: int) -> bool:
        if not (1 <= row <= h - 1 and 0 <= col <= w - 1):
            return False
        if h_walls[row, col]:
            return False
        neighbors = [
            (row - 1, col), (row + 1, col)
        ]
        for nr, nc in neighbors:
            if 1 <= nr <= h - 1 and 0 <= nc <= w - 1 and h_walls[nr, nc]:
                return False
        return True

    def can_place_v(row: int, col: int) -> bool:
        if not (0 <= row <= h - 1 and 1 <= col <= w - 1):
            return False
        if v_walls[row, col]:
            return False
        neighbors = [
            (row, col - 1), (row, col + 1)
        ]
        for nr, nc in neighbors:
            if 0 <= nr <= h - 1 and 1 <= nc <= w - 1 and v_walls[nr, nc]:
                return False
        return True

    def place_edge_t_in_quadrant(q_bounds: Tuple[int, int, int, int]) -> bool:
        rs, re, cs, ce = q_bounds
        if rs > re or cs > ce:
            return False
        choices: List[Tuple[str, Tuple[int, int]]] = []
        if rs == 0:
            for c in range(cs + 1, ce + 1):
                choices.append(("TOP", (0, c)))
        if re == h - 1:
            for c in range(cs + 1, ce + 1):
                choices.append(("BOTTOM", (h - 1, c)))
        if cs == 0:
            for r in range(rs + 1, re + 1):
                choices.append(("LEFT", (r, 0)))
        if ce == w - 1:
            for r in range(rs + 1, re + 1):
                choices.append(("RIGHT", (r, w - 1)))
        if not choices:
            return False
        idxs = np.arange(len(choices))
        rng.shuffle(idxs)
        for idx in idxs:
            side, pos = choices[idx]
            if side in ("TOP", "BOTTOM"):
                r0, c = pos
                if c in (1, w - 1):
                    continue
                if can_place_v(r0, c):
                    adj_cell = (1 if side == "TOP" else h - 2, c - 1)
                    blocked = any(cell_has_nonboundary_wall(rr, cc) for rr, cc in neighbors4(adj_cell[0], adj_cell[1]))
                    if not blocked and (0 <= adj_cell[0] < h and 0 <= adj_cell[1] < w) and adj_cell not in reserved_cells:
                        v_walls[r0, c] = True
                        used_interior_v.add((r0, c))
                        reserve_cell(adj_cell)
                        return True
            else:
                r, c0 = pos
                if r in (1, h - 1):
                    continue
                if can_place_h(r, c0):
                    adj_cell = (r - 1, 1 if side == "LEFT" else w - 2)
                    blocked = any(cell_has_nonboundary_wall(rr, cc) for rr, cc in neighbors4(adj_cell[0], adj_cell[1]))
                    if not blocked and (0 <= adj_cell[0] < h and 0 <= adj_cell[1] < w) and adj_cell not in reserved_cells:
                        h_walls[r, c0] = True
                        used_interior_h.add((r, c0))
                        reserve_cell(adj_cell)
                        return True
        return False

    orientations = ["TL", "TR", "BL", "BR"]
    central_l_centers: List[Tuple[int, int]] = []

    def place_central_l_in_quadrant(q_bounds: Tuple[int, int, int, int], desired_orientation: str) -> bool:
        rs, re, cs, ce = q_bounds
        if rs > re or cs > ce:
            return False
        candidates: List[Tuple[int, int]] = []
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    continue
                if (r, c) in reserved_cells:
                    continue
                if any(cell_has_nonboundary_wall(rr, cc) for rr, cc in neighbors4(r, c)):
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        idxs = np.arange(len(candidates))
        rng.shuffle(idxs)
        for idx in idxs:
            r, c = candidates[idx]
            need_h: Optional[Tuple[int, int]] = None
            need_v: Optional[Tuple[int, int]] = None
            if desired_orientation == "TL":
                need_h = (r, c)
                need_v = (r, c)
            elif desired_orientation == "TR":
                need_h = (r, c)
                need_v = (r, c + 1)
            elif desired_orientation == "BL":
                need_h = (r + 1, c)
                need_v = (r, c)
            else:  # BR
                need_h = (r + 1, c)
                need_v = (r, c + 1)
            if need_h is None or need_v is None:
                continue
            hr, hc = need_h
            vr, vc = need_v
            if not can_place_h(hr, hc):
                continue
            if not can_place_v(vr, vc):
                continue
            if hr in (0, h) or vc in (0, w):
                continue
            h_walls[hr, hc] = True
            v_walls[vr, vc] = True
            used_interior_h.add((hr, hc))
            used_interior_v.add((vr, vc))
            reserve_cell((r, c))
            central_l_centers.append((r, c))
            return True
        return False

    # 1) Edge-T per quadrant
    for _, bounds in quadrants.items():
        count = int(edge_t_per_quadrant)
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 10:
            if place_edge_t_in_quadrant(bounds):
                placed += 1
            attempts += 1

    # 2) Central-L per quadrant with even orientation distribution
    for _, bounds in quadrants.items():
        count = int(central_l_per_quadrant)
        if count <= 0:
            continue
        orient_list: List[str] = []
        for i in range(count):
            orient_list.append(orientations[i % 4])
        idxs = np.arange(len(orient_list))
        rng.shuffle(idxs)
        orient_list = [orient_list[i] for i in idxs]
        placed = 0
        attempts = 0
        i = 0
        while placed < count and attempts < count * 20 and i < len(orient_list):
            if place_central_l_in_quadrant(bounds, orient_list[i]):
                placed += 1
            i += 1
            attempts += 1

    # Goal inside central-L
    central_set = set((r, c) for r in central_rows for c in central_cols)
    candidates_goal = [rc for rc in central_l_centers if rc not in central_set]
    if not candidates_goal:
        candidates_goal = [(r, c) for r in range(h) for c in range(w) if (r, c) not in central_set]
    idx = int(rng.integers(0, len(candidates_goal)))
    goal = candidates_goal[idx]

    # Robots
    free_cells: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in central_set or (r, c) == goal:
                continue
            free_cells.append((r, c))
    rng.shuffle(free_cells)
    robot_positions: Dict[int, Tuple[int, int]] = {}
    for rid in range(int(num_robots)):
        robot_positions[rid] = free_cells.pop()
    target_robot = int(rng.integers(0, int(num_robots)))
    return Board(height=h, width=w, h_walls=h_walls, v_walls=v_walls, robot_positions=robot_positions, goal_position=goal, target_robot=target_robot)


# Optional helper for registration with gymnasium
def register_env(env_id: str = "RicochetRobots-v0", *, max_episode_steps: Optional[int] = None) -> None:
    """Register this environment with gymnasium so it can be created via gym.make(env_id).

    Note: This env already enforces a time limit via its max_steps argument. If you also set
    max_episode_steps here and wrap with Gym's TimeLimit implicitly via gym.make, you may see
    duplicate truncation behavior. Prefer one source of time limits.
    """
    try:
        from gymnasium.envs.registration import register  # type: ignore
        from gymnasium.error import Error as GymError  # type: ignore
        try:
            kwargs = {}
            if max_episode_steps is not None:
                kwargs["max_episode_steps"] = int(max_episode_steps)
            register(
                id=env_id,
                entry_point="env.ricochet_env:RicochetRobotsEnv",
                **kwargs,
            )
        except GymError as e:  # pragma: no cover - depends on gym version behavior
            # Swallow duplicate registration errors; re-raise others
            if "already registered" in str(e).lower():
                return
            raise
    except (ModuleNotFoundError, ImportError):
        # Silently ignore if gymnasium not available
        pass
