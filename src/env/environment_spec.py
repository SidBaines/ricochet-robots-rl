"""Validated environment settings shared by every construction path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EnvironmentSpec:
    """The experiment-level environment contract.

    Curriculum levels and fixed layouts may replace geometry and generation
    fields, but must not silently replace rewards, horizon, actions, or the
    observation encoding.
    """

    height: int = 8
    width: int = 8
    num_robots: int = 2
    include_noop: bool = False
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    noop_penalty: Optional[float] = None
    max_steps: int = 20
    seed: Optional[int] = 0
    ensure_solvable: bool = False
    solver_max_depth: int = 30
    solver_max_nodes: int = 20_000
    obs_mode: str = "image"
    channels_first: bool = True
    cell_obs_pixel_size: int = 128

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Environment dimensions must be positive")
        if self.num_robots <= 0:
            raise ValueError("num_robots must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.solver_max_depth <= 0 or self.solver_max_nodes <= 0:
            raise ValueError("Solver limits must be positive")
        if self.cell_obs_pixel_size <= 0:
            raise ValueError("cell_obs_pixel_size must be positive")
        if self.obs_mode not in {"image", "symbolic", "rgb_image", "rgb_cell_image"}:
            raise ValueError(f"Unsupported observation mode: {self.obs_mode}")

    @classmethod
    def from_namespace(cls, args: Any) -> "EnvironmentSpec":
        """Build the validated contract from parsed training arguments."""
        return cls(
            height=int(args.height),
            width=int(args.width),
            num_robots=int(args.num_robots),
            include_noop=bool(args.include_noop),
            step_penalty=float(args.step_penalty),
            goal_reward=float(args.goal_reward),
            noop_penalty=float(args.noop_penalty) if args.noop_penalty is not None else None,
            max_steps=int(args.max_steps),
            seed=int(args.seed) if args.seed is not None else None,
            ensure_solvable=bool(args.ensure_solvable),
            solver_max_depth=int(args.solver_max_depth),
            solver_max_nodes=int(args.solver_max_nodes),
            obs_mode=str(args.obs_mode),
            channels_first=True,
            cell_obs_pixel_size=int(getattr(args, "cell_obs_pixel_size", 128)),
        )

    def to_env_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """Return kwargs for ``RicochetRobotsEnv`` with explicit overrides."""
        values = asdict(self)
        values.update(overrides)
        return values

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for run manifests."""
        return asdict(self)
