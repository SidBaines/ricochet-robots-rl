"""Exact trajectory-state labels and recurrent-carry capture for DRC policies.

This module contains no probe fitting and no causal intervention.  Its job is
to make an observational dataset whose rows are real decision points from a
masked deterministic policy rollout.  Exact graph labels are allowed to be
censored; a resource cap is always represented as unknown, never as a numeric
distance or an unsolvability claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .drc import DRCNet, DRCState
from .env import DOWN, LEFT, RIGHT, UP, EnvConfig, RicochetEnv
from .puzzles import Puzzle


DATASET_SCHEMA_VERSION = 2
CAPTURE_PROTOCOL_VERSION = "drc-carry-trajectory-v2"
SOLVER_PROTOCOL_VERSION = "canonical-unit-slide-bfs-first-action-union-v1"


def BOS_ACTION(n_actions: int) -> int:
    """Class used when a requested action lag precedes episode start."""

    return int(n_actions)


def MISSING_ACTION(n_actions: int) -> int:
    """Reserved class for unavailable action provenance (not used in full capture)."""

    return int(n_actions) + 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SolverContract:
    """Resource and semantic contract for an exact state query."""

    max_depth: int = 64
    max_nodes: int = 2_000_000
    protocol_version: str = SOLVER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {self.max_depth}")
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be at least 1, got {self.max_nodes}")
        if self.protocol_version != SOLVER_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported solver protocol {self.protocol_version!r}; expected {SOLVER_PROTOCOL_VERSION!r}"
            )

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ExactSolveResult:
    """One exact BFS result.

    ``status='known', solvable=False`` is possible only after exhausting the
    reachable graph.  A depth/node cap instead has ``status='unknown'`` and
    ``solvable=None``.  Unknown results deliberately carry no partial action
    mask, even if a goal was observed before a node cap interrupted the rest
    of that layer.
    """

    status: str
    solvable: Optional[bool]
    distance: Optional[int]
    optimal_action_bits: int
    nodes_discovered: int
    nodes_expanded: int
    censor_reason: Optional[str] = None

    @property
    def known(self) -> bool:
        return self.status == "known"


@dataclass(frozen=True)
class ExactStateLabels:
    solve: ExactSolveResult
    optimal_action_mask: np.ndarray
    reducing_action_mask: np.ndarray
    reducing_known: bool
    reducing_censor_reason: Optional[str]
    reducing_verification_nodes: int


@dataclass(frozen=True)
class _BoundedReachability:
    reachable: Optional[bool]
    nodes_discovered: int
    nodes_expanded: int
    censor_reason: Optional[str]


def _normalise_positions(
    positions: Sequence[Tuple[int, int]], *, height: int, width: int
) -> Tuple[Tuple[int, int], ...]:
    result = tuple((int(row), int(col)) for row, col in positions)
    if not result:
        raise ValueError("At least one canonical robot position is required")
    if len(set(result)) != len(result):
        raise ValueError(f"Robot positions must be distinct, got {result}")
    if any(not (0 <= row < height and 0 <= col < width) for row, col in result):
        raise ValueError(f"Robot position outside {height}x{width} board: {result}")
    return result


class ExactStateOracle:
    """Exact unit-slide BFS over canonical robot identities.

    Canonical robot 0 is always the target.  Other robot identities retain the
    order supplied by :class:`RicochetEnv`; states are never re-sorted after a
    move.  The public cache key includes wall geometry, goal, canonical
    positions, robot count, and the complete solver contract.
    """

    def __init__(
        self,
        puzzle: Puzzle,
        contract: SolverContract = SolverContract(),
        *,
        cache: Optional[MutableMapping[str, ExactSolveResult]] = None,
    ) -> None:
        self.puzzle = puzzle
        self.contract = contract
        self.height = int(puzzle.height)
        self.width = int(puzzle.width)
        self.n_actions = 4 * puzzle.num_robots
        self._cache: MutableMapping[str, ExactSolveResult] = {} if cache is None else cache
        self._bounded_cache: Dict[str, _BoundedReachability] = {}
        geometry = hashlib.sha256()
        geometry.update(np.asarray([self.height, self.width, puzzle.num_robots], dtype="<i4").tobytes())
        geometry.update(np.ascontiguousarray(puzzle.h_walls, dtype=np.uint8).tobytes())
        geometry.update(np.ascontiguousarray(puzzle.v_walls, dtype=np.uint8).tobytes())
        geometry.update(np.asarray(puzzle.goal, dtype="<i4").tobytes())
        geometry.update(b"canonical-target-robot-0")
        self.geometry_goal_fingerprint = geometry.hexdigest()
        self._stats = Counter()

    def cache_key(self, positions: Sequence[Tuple[int, int]]) -> str:
        canonical = _normalise_positions(positions, height=self.height, width=self.width)
        return _sha256_json(
            {
                "geometry_goal": self.geometry_goal_fingerprint,
                "canonical_positions": canonical,
                "solver_contract": self.contract.as_dict(),
            }
        )

    def transition(self, positions: Sequence[Tuple[int, int]], action: int) -> Tuple[Tuple[int, int], ...]:
        canonical = _normalise_positions(positions, height=self.height, width=self.width)
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} outside [0, {self.n_actions})")
        return self._transition_canonical(canonical, action)

    def _transition_canonical(self, canonical: Tuple[Tuple[int, int], ...], action: int) -> Tuple[Tuple[int, int], ...]:
        """Transition an already validated state (the solver's hot path)."""

        rid, direction = divmod(action, 4)
        row, col = canonical[rid]
        occupied = set(canonical)
        occupied.remove((row, col))
        h_walls, v_walls = self.puzzle.h_walls, self.puzzle.v_walls
        if direction == UP:
            while row > 0 and not h_walls[row, col] and (row - 1, col) not in occupied:
                row -= 1
        elif direction == DOWN:
            while row < self.height - 1 and not h_walls[row + 1, col] and (row + 1, col) not in occupied:
                row += 1
        elif direction == LEFT:
            while col > 0 and not v_walls[row, col] and (row, col - 1) not in occupied:
                col -= 1
        elif direction == RIGHT:
            while col < self.width - 1 and not v_walls[row, col + 1] and (row, col + 1) not in occupied:
                col += 1
        updated = list(canonical)
        updated[rid] = (row, col)
        return tuple(updated)

    def legal_action_mask(self, positions: Sequence[Tuple[int, int]]) -> np.ndarray:
        canonical = _normalise_positions(positions, height=self.height, width=self.width)
        return np.asarray(
            [self._transition_canonical(canonical, action) != canonical for action in range(self.n_actions)],
            dtype=bool,
        )

    def _goal(self, state: Tuple[Tuple[int, int], ...]) -> bool:
        return state[0] == tuple(self.puzzle.goal)

    def solve(self, positions: Sequence[Tuple[int, int]]) -> ExactSolveResult:
        """Return exact distance and every optimal first action, or explicit censoring."""

        start = _normalise_positions(positions, height=self.height, width=self.width)
        key = self.cache_key(start)
        cached = self._cache.get(key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            return cached
        self._stats["cache_misses"] += 1
        result = self._solve_uncached(start)
        self._cache[key] = result
        self._stats[f"status_{result.status}"] += 1
        if result.censor_reason:
            self._stats[f"censor_{result.censor_reason}"] += 1
        self._stats["nodes_discovered"] += result.nodes_discovered
        self._stats["nodes_expanded"] += result.nodes_expanded
        return result

    def _solve_uncached(self, start: Tuple[Tuple[int, int], ...]) -> ExactSolveResult:
        if self._goal(start):
            return ExactSolveResult("known", True, 0, 0, 1, 0)

        # A frontier value is the union of first-action bits for all shortest
        # paths reaching that state.  It is fully assembled before the next
        # layer is expanded, which is the critical multiple-solution contract.
        frontier: Dict[Tuple[Tuple[int, int], ...], int] = {start: 0}
        seen_depth: Dict[Tuple[Tuple[int, int], ...], int] = {start: 0}
        nodes_discovered = 1
        nodes_expanded = 0

        for depth in range(self.contract.max_depth):
            next_frontier: Dict[Tuple[Tuple[int, int], ...], int] = {}
            goal_action_bits = 0
            for state in sorted(frontier):
                incoming_bits = frontier[state]
                nodes_expanded += 1
                for action in range(self.n_actions):
                    successor = self._transition_canonical(state, action)
                    if successor == state:  # exact parity with action_masks()
                        continue
                    first_bits = (1 << action) if depth == 0 else incoming_bits
                    successor_depth = depth + 1
                    prior_depth = seen_depth.get(successor)
                    if prior_depth is None:
                        if nodes_discovered >= self.contract.max_nodes:
                            return ExactSolveResult(
                                "unknown",
                                None,
                                None,
                                0,
                                nodes_discovered,
                                nodes_expanded,
                                "max_nodes",
                            )
                        seen_depth[successor] = successor_depth
                        nodes_discovered += 1
                    elif prior_depth < successor_depth:
                        continue
                    elif prior_depth > successor_depth:  # impossible under layer BFS
                        raise AssertionError("BFS discovered a state at a shallower later layer")

                    if self._goal(successor):
                        goal_action_bits |= first_bits
                    else:
                        next_frontier[successor] = next_frontier.get(successor, 0) | first_bits

            # Stop only here: every parent in the first goal layer has been
            # expanded, and all equal-depth first-action masks are unioned.
            if goal_action_bits:
                return ExactSolveResult(
                    "known",
                    True,
                    depth + 1,
                    goal_action_bits,
                    nodes_discovered,
                    nodes_expanded,
                )
            if not next_frontier:
                return ExactSolveResult("known", False, None, 0, nodes_discovered, nodes_expanded)
            frontier = next_frontier

        return ExactSolveResult(
            "unknown",
            None,
            None,
            0,
            nodes_discovered,
            nodes_expanded,
            "max_depth",
        )

    def _goal_within(self, start: Tuple[Tuple[int, int], ...], depth_limit: int) -> _BoundedReachability:
        """Independent existence BFS used to verify distance-reducing actions."""

        key = _sha256_json(
            {
                "kind": "goal_within",
                "geometry_goal": self.geometry_goal_fingerprint,
                "canonical_positions": start,
                "depth_limit": int(depth_limit),
                "max_nodes": self.contract.max_nodes,
                "solver_protocol": self.contract.protocol_version,
            }
        )
        cached = self._bounded_cache.get(key)
        if cached is not None:
            self._stats["bounded_cache_hits"] += 1
            return cached
        self._stats["bounded_cache_misses"] += 1

        if self._goal(start):
            result = _BoundedReachability(True, 1, 0, None)
            self._bounded_cache[key] = result
            return result
        frontier = {start}
        seen = {start}
        discovered, expanded = 1, 0
        for _depth in range(depth_limit):
            following = set()
            for state in sorted(frontier):
                expanded += 1
                for action in range(self.n_actions):
                    successor = self._transition_canonical(state, action)
                    if successor == state or successor in seen:
                        continue
                    if discovered >= self.contract.max_nodes:
                        result = _BoundedReachability(None, discovered, expanded, "max_nodes")
                        self._bounded_cache[key] = result
                        return result
                    seen.add(successor)
                    discovered += 1
                    if self._goal(successor):
                        result = _BoundedReachability(True, discovered, expanded, None)
                        self._bounded_cache[key] = result
                        return result
                    following.add(successor)
            if not following:
                result = _BoundedReachability(False, discovered, expanded, None)
                self._bounded_cache[key] = result
                return result
            frontier = following
        # All paths of length <= depth_limit were checked.  This is a known
        # negative predicate, even though it would be a depth-censored query
        # for the successor's unrestricted numeric distance.
        result = _BoundedReachability(False, discovered, expanded, None)
        self._bounded_cache[key] = result
        return result

    def label(self, positions: Sequence[Tuple[int, int]]) -> ExactStateLabels:
        """Solve a state and independently derive its reducing-action mask."""

        canonical = _normalise_positions(positions, height=self.height, width=self.width)
        solve = self.solve(canonical)
        optimal = np.asarray(
            [bool(solve.optimal_action_bits & (1 << action)) for action in range(self.n_actions)],
            dtype=bool,
        )
        empty = np.zeros(self.n_actions, dtype=bool)
        if not solve.known or not solve.solvable:
            return ExactStateLabels(solve, optimal, empty, False, solve.censor_reason, 0)
        if solve.distance == 0:
            if optimal.any():
                raise AssertionError("Goal states must have an empty optimal action mask")
            return ExactStateLabels(solve, optimal, empty, True, None, 0)

        reducing = np.zeros(self.n_actions, dtype=bool)
        verification_nodes = 0
        for action in np.flatnonzero(self.legal_action_mask(canonical)):
            successor = self._transition_canonical(canonical, int(action))
            check = self._goal_within(successor, int(solve.distance) - 1)
            verification_nodes += check.nodes_discovered
            if check.reachable is None:
                return ExactStateLabels(
                    solve,
                    optimal,
                    empty,
                    False,
                    check.censor_reason,
                    verification_nodes,
                )
            reducing[int(action)] = bool(check.reachable)
        if not np.array_equal(reducing, optimal):
            raise AssertionError("Independent successor search disagrees with all-optimal-first-action BFS")
        return ExactStateLabels(solve, optimal, reducing, True, None, verification_nodes)

    def stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in sorted(self._stats.items())}


def _network_signature(net: DRCNet) -> Dict[str, object]:
    return {
        "class": f"{type(net).__module__}.{type(net).__qualname__}",
        "observation_shape": list(net.observation_shape),
        "n_actions": int(net.n_actions),
        "depth": int(net.depth),
        "ticks": int(net.ticks),
        "hidden_channels": int(net.hidden_channels),
        "encoder_relu": bool(net.encoder_relu),
        "encoder_skip": bool(net.encoder_skip),
        "encoder": [
            {
                "in_channels": int(layer.in_channels),
                "out_channels": int(layer.out_channels),
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride),
            }
            for layer in net.encoder
        ],
        "cells": [
            {
                "input_channels": int(cell.input_channels),
                "hidden_channels": int(cell.hidden_channels),
                "kernel_size": int(cell.kernel_size),
                "pool_and_inject": bool(cell.pool_and_inject),
                "pool_projection": str(cell.pool_projection),
                "output_activation": str(cell.output_activation),
                "forget_bias": float(cell.forget_bias),
                "fence": str(cell.fence),
            }
            for cell in net.cells
        ],
        "mlp_in_features": int(net.mlp.in_features),
        "mlp_out_features": int(net.mlp.out_features),
        "state_dict_shapes": {name: list(value.shape) for name, value in net.state_dict().items()},
    }


def _assert_matched_networks(trained: DRCNet, random_control: DRCNet) -> None:
    trained_signature = _network_signature(trained)
    random_signature = _network_signature(random_control)
    if trained_signature != random_signature:
        raise ValueError("Random control architecture does not exactly match trained DRC")
    if next(trained.parameters()).device != next(random_control.parameters()).device:
        raise ValueError("Trained and random-control networks must be on the same device")


def _state_dict_fingerprint(net: DRCNet) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _append_state(rows: Dict[str, list], prefix: str, state: DRCState, dtype: np.dtype) -> None:
    rows[f"{prefix}_h"].append(state.h[:, 0].detach().cpu().numpy().astype(dtype, copy=True))
    rows[f"{prefix}_c"].append(state.c[:, 0].detach().cpu().numpy().astype(dtype, copy=True))


def capture_carry_trajectories(
    net: DRCNet,
    random_net: DRCNet,
    puzzles: Sequence[Puzzle],
    env_config: EnvConfig,
    *,
    solver_contract: SolverContract = SolverContract(),
    rollout_seed: int = 0,
    recent_action_lags: int = 3,
    carry_dtype: np.dtype | type = np.float16,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Capture one masked deterministic full-start episode per puzzle.

    The random network receives each trained trajectory observation in the
    same episode order and is reset only at episode boundaries.  Its logits are
    discarded and it is never consulted for an environment action.
    """

    if not puzzles:
        raise ValueError("Cannot capture an empty puzzle suite")
    if recent_action_lags < 1:
        raise ValueError("recent_action_lags must be at least 1")
    if not env_config.canonical_identity:
        raise ValueError("Carry-state oracle requires EnvConfig.canonical_identity=True")
    if env_config.rgb_obs:
        raise ValueError("DRC carry capture currently requires a tensor observation")
    _assert_matched_networks(net, random_net)
    carry_dtype = np.dtype(carry_dtype)
    if carry_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("carry_dtype must be float16 or float32")

    first = puzzles[0]
    if any((p.height, p.width, p.num_robots) != (first.height, first.width, first.num_robots) for p in puzzles):
        raise ValueError("All captured puzzles must share board and robot dimensions")
    if net.n_actions != 4 * first.num_robots:
        raise ValueError(f"Checkpoint has {net.n_actions} actions but suite requires {4 * first.num_robots}")

    env = RicochetEnv(puzzles, env_config, sampler=None, seed=rollout_seed)
    if tuple(env.observation_space.shape or ()) != tuple(net.observation_shape):
        raise ValueError(
            f"Environment observation shape {env.observation_space.shape} != checkpoint {net.observation_shape}"
        )

    row_names = (
        "obs",
        "positions",
        "action_mask",
        "policy_logits",
        "chosen_action",
        "intended_robot",
        "intended_direction",
        "landing_cell",
        "slide_length",
        "moved",
        "immediate_success",
        "future_success",
        "recent_actions",
        "move_index",
        "horizon_progress",
        "optimal_distance_progress",
        "episode_id",
        "suite_puzzle_index",
        "layout_hash",
        "puzzle_seed",
        "puzzle_optimal_length",
        "puzzle_robots_moved",
        "solver_known",
        "solver_solvable",
        "exact_label_known",
        "exact_distance",
        "optimal_action_mask",
        "reducing_action_mask",
        "reducing_label_known",
        "solver_censor_code",
        "reducing_censor_code",
        "solver_nodes_discovered",
        "solver_nodes_expanded",
        "reducing_verification_nodes",
        "trained_carry_in_h",
        "trained_carry_in_c",
        "trained_carry_out_h",
        "trained_carry_out_c",
        "random_carry_in_h",
        "random_carry_in_c",
        "random_carry_out_h",
        "random_carry_out_c",
    )
    rows: Dict[str, list] = {name: [] for name in row_names}
    censor_codes = {None: 0, "max_depth": 1, "max_nodes": 2}
    solver_totals = Counter()
    successes = 0
    device = next(net.parameters()).device
    was_training = net.training
    random_was_training = random_net.training
    net.eval()
    random_net.eval()

    try:
        with torch.no_grad():
            progress_interval = max(1, (len(puzzles) + 9) // 10)
            for episode_id, puzzle in enumerate(puzzles):
                if progress is not None and episode_id % progress_interval == 0:
                    progress(f"Capturing puzzle {episode_id + 1}/{len(puzzles)}")
                obs, _info = env.reset(
                    options={
                        "puzzle_index": episode_id,
                        "start_distance": puzzle.optimal_length,
                    }
                )
                trained_state = net.initial_state(1, device=device)
                random_state = random_net.initial_state(1, device=device)
                oracle = ExactStateOracle(puzzle, solver_contract)
                action_history: list[int] = []
                episode_rows: list[int] = []
                done = False
                episode_success = False

                while not done:
                    row_index = len(rows["chosen_action"])
                    episode_rows.append(row_index)
                    positions = tuple(env._robots)
                    legal_mask = env.action_masks()
                    oracle_mask = oracle.legal_action_mask(positions)
                    if not np.array_equal(legal_mask, oracle_mask):
                        raise AssertionError("Exact solver action mask diverged from RicochetEnv")
                    labels = oracle.label(positions)
                    if not action_history and labels.solve.known:
                        if not labels.solve.solvable:
                            raise AssertionError("Bank full-start state is exactly unsolvable")
                        if labels.solve.distance != puzzle.optimal_length:
                            raise AssertionError(
                                "Exact full-start distance disagrees with bank optimal_length: "
                                f"{labels.solve.distance} != {puzzle.optimal_length}"
                            )

                    obs_tensor = torch.as_tensor(obs[None], device=device)
                    mask_tensor = torch.as_tensor(legal_mask[None], device=device)
                    trained_in = trained_state
                    random_in = random_state
                    logits, _value, trained_out = net(obs_tensor, trained_in, action_mask=mask_tensor)
                    # The recurrent computation sees the identical observation
                    # sequence; random logits are intentionally thrown away.
                    _random_logits, _random_value, random_out = random_net(
                        obs_tensor, random_in, action_mask=mask_tensor
                    )
                    action = int(logits.argmax(dim=-1).item())
                    if not legal_mask[action]:
                        raise AssertionError("Masked deterministic policy selected an illegal action")
                    rid, direction = divmod(action, 4)
                    landing = env._slide_from(positions, rid, direction, puzzle)
                    start_position = positions[rid]
                    slide_length = abs(landing[0] - start_position[0]) + abs(landing[1] - start_position[1])
                    moved = landing != start_position
                    immediate_success = rid == 0 and landing == puzzle.goal

                    rows["obs"].append(np.asarray(obs).copy())
                    rows["positions"].append(np.asarray(positions, dtype=np.int16))
                    rows["action_mask"].append(legal_mask.copy())
                    rows["policy_logits"].append(logits[0].detach().cpu().numpy().copy())
                    rows["chosen_action"].append(action)
                    rows["intended_robot"].append(rid)
                    rows["intended_direction"].append(direction)
                    rows["landing_cell"].append(landing[0] * puzzle.width + landing[1])
                    rows["slide_length"].append(slide_length)
                    rows["moved"].append(moved)
                    rows["immediate_success"].append(immediate_success)
                    rows["future_success"].append(False)  # backfilled at episode end
                    rows["recent_actions"].append(
                        np.asarray(
                            [
                                action_history[-lag]
                                if len(action_history) >= lag
                                else (
                                    BOS_ACTION(net.n_actions)
                                    if lag == len(action_history) + 1
                                    else MISSING_ACTION(net.n_actions)
                                )
                                for lag in range(1, recent_action_lags + 1)
                            ],
                            dtype=np.int16,
                        )
                    )
                    move_index = len(action_history)
                    rows["move_index"].append(move_index)
                    rows["horizon_progress"].append(move_index / env_config.horizon)
                    if labels.solve.known and labels.solve.solvable:
                        distance_progress = 1.0 - float(labels.solve.distance) / puzzle.optimal_length
                    else:
                        distance_progress = np.nan
                    rows["optimal_distance_progress"].append(distance_progress)
                    rows["episode_id"].append(episode_id)
                    rows["suite_puzzle_index"].append(episode_id)
                    rows["layout_hash"].append(puzzle.layout_hash)
                    rows["puzzle_seed"].append(puzzle.seed)
                    rows["puzzle_optimal_length"].append(puzzle.optimal_length)
                    rows["puzzle_robots_moved"].append(puzzle.robots_moved)
                    rows["solver_known"].append(labels.solve.known)
                    rows["solver_solvable"].append(bool(labels.solve.solvable))
                    exact_known = labels.solve.known and bool(labels.solve.solvable)
                    rows["exact_label_known"].append(exact_known)
                    rows["exact_distance"].append(float(labels.solve.distance) if exact_known else np.nan)
                    rows["optimal_action_mask"].append(labels.optimal_action_mask.copy())
                    rows["reducing_action_mask"].append(labels.reducing_action_mask.copy())
                    rows["reducing_label_known"].append(labels.reducing_known)
                    rows["solver_censor_code"].append(censor_codes[labels.solve.censor_reason])
                    rows["reducing_censor_code"].append(censor_codes[labels.reducing_censor_reason])
                    rows["solver_nodes_discovered"].append(labels.solve.nodes_discovered)
                    rows["solver_nodes_expanded"].append(labels.solve.nodes_expanded)
                    rows["reducing_verification_nodes"].append(labels.reducing_verification_nodes)
                    _append_state(rows, "trained_carry_in", trained_in, carry_dtype)
                    _append_state(rows, "trained_carry_out", trained_out, carry_dtype)
                    _append_state(rows, "random_carry_in", random_in, carry_dtype)
                    _append_state(rows, "random_carry_out", random_out, carry_dtype)

                    obs, _reward, terminated, truncated, info = env.step(action)
                    if tuple(env._robots[rid]) != landing:
                        raise AssertionError("Recorded landing cell diverged from environment step")
                    if bool(info["is_success"]) != immediate_success:
                        raise AssertionError("Recorded immediate-success label diverged from environment")
                    episode_success = bool(info["is_success"])
                    done = bool(terminated or truncated)
                    action_history.append(action)
                    trained_state = trained_out
                    random_state = random_out

                for index in episode_rows:
                    rows["future_success"][index] = episode_success
                successes += int(episode_success)
                solver_totals.update(oracle.stats())
    finally:
        net.train(was_training)
        random_net.train(random_was_training)

    bool_names = {
        "action_mask",
        "moved",
        "immediate_success",
        "future_success",
        "solver_known",
        "solver_solvable",
        "exact_label_known",
        "optimal_action_mask",
        "reducing_action_mask",
        "reducing_label_known",
    }
    int16_names = {
        "positions",
        "chosen_action",
        "intended_robot",
        "intended_direction",
        "landing_cell",
        "slide_length",
        "recent_actions",
    }
    int32_names = {
        "move_index",
        "episode_id",
        "suite_puzzle_index",
        "puzzle_optimal_length",
        "puzzle_robots_moved",
        "solver_nodes_discovered",
        "solver_nodes_expanded",
        "reducing_verification_nodes",
    }
    int64_names = {"puzzle_seed"}
    uint8_names = {"solver_censor_code", "reducing_censor_code"}
    float32_names = {
        "obs",
        "policy_logits",
        "horizon_progress",
        "optimal_distance_progress",
        "exact_distance",
    }
    arrays: Dict[str, np.ndarray] = {}
    for name, values in rows.items():
        if name in bool_names:
            dtype = bool
        elif name in int16_names:
            dtype = np.int16
        elif name in int32_names:
            dtype = np.int32
        elif name in int64_names:
            dtype = np.int64
        elif name in uint8_names:
            dtype = np.uint8
        elif name in float32_names:
            dtype = np.float32
        elif "carry" in name:
            dtype = carry_dtype
        elif name == "layout_hash":
            dtype = str
        else:  # pragma: no cover - row schema above is deliberately exhaustive
            raise AssertionError(f"No storage dtype declared for {name}")
        arrays[name] = np.asarray(values, dtype=dtype)

    known = arrays["exact_label_known"]
    reducing_known = arrays["reducing_label_known"]
    random_architecture_signature = _network_signature(random_net)
    metadata: Dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "capture_protocol": CAPTURE_PROTOCOL_VERSION,
        "solver_contract": solver_contract.as_dict(),
        "environment": dataclasses.asdict(env_config),
        "rollout_seed": int(rollout_seed),
        "n_rows": int(len(arrays["chosen_action"])),
        "n_episodes": int(len(puzzles)),
        "episodes": {
            "successes": int(successes),
            "failures": int(len(puzzles) - successes),
        },
        "labels": {
            "recent_action_lags": int(recent_action_lags),
            "action_classes": {
                "actions": [0, int(net.n_actions) - 1],
                "bos": BOS_ACTION(net.n_actions),
                "missing": MISSING_ACTION(net.n_actions),
            },
            "history_padding": (
                "For lag k: the kth prior action when present; BOS at the exact "
                "episode boundary; MISSING for lags older than that boundary"
            ),
            "move_index": "zero-based decision index within the episode",
            "horizon_progress": "move_index / configured horizon before the chosen move",
            "optimal_distance_progress": (
                "1 - exact_distance / full-start bank optimal_length; may be negative "
                "after detours and is NaN when the exact label is censored"
            ),
            "landing_cell": "row * board_width + column after the chosen slide",
            "solver_censor_codes": {"none": 0, "max_depth": 1, "max_nodes": 2},
        },
        "solver_coverage": {
            "exact_known_rows": int(known.sum()),
            "exact_total_rows": int(len(known)),
            "exact_fraction": float(known.mean()),
            "reducing_known_rows": int(reducing_known.sum()),
            "reducing_fraction": float(reducing_known.mean()),
            "known_unsolvable_rows": int(np.sum(arrays["solver_known"] & ~arrays["solver_solvable"])),
            "root_nodes_discovered": int(arrays["solver_nodes_discovered"].sum()),
            "root_nodes_expanded": int(arrays["solver_nodes_expanded"].sum()),
            "reducing_verification_nodes": int(arrays["reducing_verification_nodes"].sum()),
            "censor_reasons": {
                "max_depth": int(np.sum(arrays["solver_censor_code"] == 1)),
                "max_nodes": int(np.sum(arrays["solver_censor_code"] == 2)),
            },
            "reducing_censor_reasons": {
                "max_depth": int(np.sum(arrays["reducing_censor_code"] == 1)),
                "max_nodes": int(np.sum(arrays["reducing_censor_code"] == 2)),
            },
            "oracle_query_totals": {key: int(value) for key, value in sorted(solver_totals.items())},
        },
        "random_control": {
            "seed": getattr(random_net, "_carry_control_seed", None),
            "architecture_fingerprint": _sha256_json(random_architecture_signature),
            "architecture_signature": random_architecture_signature,
            "state_dict_sha256": _state_dict_fingerprint(random_net),
            "matched_to_trained": True,
            "trajectory_source": "trained_policy",
            "chooses_actions": False,
            "reset_scope": "episode_boundaries_only",
            "network_action_input": False,
        },
        "storage": {
            "carry_dtype": str(carry_dtype),
            "observation_dtype": str(arrays["obs"].dtype),
            "logit_dtype": str(arrays["policy_logits"].dtype),
        },
        "runtime": {
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
        },
    }
    validate_carry_dataset(arrays, metadata)
    return arrays, metadata


def _fingerprint_metadata(metadata: Mapping[str, object]) -> Dict[str, object]:
    clean = dict(metadata)
    clean.pop("content_sha256", None)
    return clean


def carry_content_fingerprint(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    """SHA-256 over contract metadata and every named array byte-for-byte."""

    digest = hashlib.sha256()
    digest.update(_canonical_json(_fingerprint_metadata(metadata)).encode("utf-8"))
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"Object array {name!r} is not permitted in carry datasets")
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _contract_differences(
    actual: Mapping[str, object], expected: Mapping[str, object], prefix: str = "contract"
) -> list[str]:
    differences: list[str] = []
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}"
        if key not in actual:
            differences.append(f"{path} missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                differences.append(f"{path} is not an object")
            else:
                differences.extend(_contract_differences(actual_value, expected_value, path))
        elif actual_value != expected_value:
            differences.append(f"{path}: {actual_value!r} != {expected_value!r}")
    return differences


def validate_carry_dataset(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    """Validate row alignment and the strongest cheap capture invariants."""

    try:
        schema_version = int(metadata.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Carry dataset schema_version is not an integer") from exc
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported carry dataset schema {schema_version}; expected {DATASET_SCHEMA_VERSION}")
    capture_protocol = metadata.get("capture_protocol")
    if capture_protocol != CAPTURE_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported carry capture protocol {capture_protocol!r}; expected {CAPTURE_PROTOCOL_VERSION!r}"
        )
    random_control = metadata.get("random_control")
    if not isinstance(random_control, Mapping):
        raise ValueError("Carry dataset random_control metadata is missing")
    random_seed = random_control.get("seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("Carry dataset random-control seed is missing")
    architecture_signature = random_control.get("architecture_signature")
    if not isinstance(architecture_signature, Mapping):
        raise ValueError("Carry dataset random-control architecture signature is missing")
    if random_control.get("architecture_fingerprint") != _sha256_json(architecture_signature):
        raise ValueError("Carry dataset random-control architecture fingerprint is inconsistent")
    state_dict_sha256 = random_control.get("state_dict_sha256")
    if not isinstance(state_dict_sha256, str) or len(state_dict_sha256) != 64:
        raise ValueError("Carry dataset random-control state_dict SHA-256 is missing")
    try:
        int(state_dict_sha256, 16)
    except ValueError as exc:
        raise ValueError("Carry dataset random-control state_dict SHA-256 is invalid") from exc
    contract = metadata.get("contract")
    if isinstance(contract, Mapping) and "random_control" in contract:
        expected_random = contract["random_control"]
        if not isinstance(expected_random, Mapping):
            raise ValueError("Carry dataset contract.random_control is not an object")
        differences = _contract_differences(
            random_control,
            expected_random,
            prefix="contract.random_control",
        )
        if differences:
            raise ValueError(
                "Carry dataset random-control metadata disagrees with its contract: " + "; ".join(differences)
            )

    required = {
        "obs",
        "positions",
        "action_mask",
        "policy_logits",
        "chosen_action",
        "intended_robot",
        "intended_direction",
        "landing_cell",
        "slide_length",
        "moved",
        "future_success",
        "immediate_success",
        "recent_actions",
        "move_index",
        "horizon_progress",
        "optimal_distance_progress",
        "episode_id",
        "suite_puzzle_index",
        "layout_hash",
        "puzzle_seed",
        "puzzle_optimal_length",
        "puzzle_robots_moved",
        "solver_known",
        "solver_solvable",
        "exact_label_known",
        "exact_distance",
        "optimal_action_mask",
        "reducing_action_mask",
        "reducing_label_known",
        "solver_censor_code",
        "reducing_censor_code",
        "solver_nodes_discovered",
        "solver_nodes_expanded",
        "reducing_verification_nodes",
        "trained_carry_in_h",
        "trained_carry_in_c",
        "trained_carry_out_h",
        "trained_carry_out_c",
        "random_carry_in_h",
        "random_carry_in_c",
        "random_carry_out_h",
        "random_carry_out_c",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Carry dataset missing arrays: {missing}")
    n_rows = len(np.asarray(arrays["chosen_action"]))
    if n_rows == 0:
        raise ValueError("Carry dataset contains no decision rows")
    misaligned = sorted(name for name, value in arrays.items() if len(np.asarray(value)) != n_rows)
    if misaligned:
        raise ValueError(f"Carry arrays have inconsistent row counts: {misaligned}")

    actions = np.asarray(arrays["chosen_action"], dtype=np.int64)
    masks = np.asarray(arrays["action_mask"], dtype=bool)
    logits = np.asarray(arrays["policy_logits"])
    if masks.ndim != 2 or logits.shape != masks.shape:
        raise ValueError("action_mask and policy_logits must have matching (rows, actions) shape")
    if np.any(actions < 0) or np.any(actions >= masks.shape[1]):
        raise ValueError("chosen_action outside the action vocabulary")
    row_index = np.arange(n_rows)
    if not np.all(masks[row_index, actions]):
        raise ValueError("Carry dataset contains an illegal chosen action")
    if not np.array_equal(logits.argmax(axis=1), actions):
        raise ValueError("Recorded logits do not select recorded chosen_action")
    if not np.array_equal(np.asarray(arrays["intended_robot"]), actions // 4):
        raise ValueError("intended_robot is inconsistent with chosen_action")
    if not np.array_equal(np.asarray(arrays["intended_direction"]), actions % 4):
        raise ValueError("intended_direction is inconsistent with chosen_action")
    positions = np.asarray(arrays["positions"])
    recent = np.asarray(arrays["recent_actions"])
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (rows, robots, 2)")
    if masks.shape[1] != 4 * positions.shape[1]:
        raise ValueError("Action vocabulary is not four directions per canonical robot")
    if recent.ndim != 2 or recent.shape[1] < 1:
        raise ValueError("recent_actions must have at least one lag column")
    carry_shapes = {
        np.asarray(arrays[f"{prefix}_carry_{side}_{kind}"]).shape
        for prefix in ("trained", "random")
        for side in ("in", "out")
        for kind in ("h", "c")
    }
    if len(carry_shapes) != 1 or next(iter(carry_shapes))[0] != n_rows:
        raise ValueError("All recurrent carry arrays must share shape (rows,D,C,H,W)")
    if len(next(iter(carry_shapes))) != 5:
        raise ValueError("Recurrent carry arrays must have shape (rows,D,C,H,W)")
    slide_length = np.asarray(arrays["slide_length"])
    moved = np.asarray(arrays["moved"], dtype=bool)
    if not np.array_equal(moved, slide_length > 0):
        raise ValueError("moved and slide_length labels disagree")
    if not np.all(moved):
        raise ValueError("Legally masked rollout contains a no-op chosen move")
    board_cells = int(np.asarray(arrays["obs"]).shape[-2] * np.asarray(arrays["obs"]).shape[-1])
    landing = np.asarray(arrays["landing_cell"])
    if np.any(landing < 0) or np.any(landing >= board_cells):
        raise ValueError("landing_cell outside the board")

    exact_known = np.asarray(arrays["exact_label_known"], dtype=bool)
    reducing_known = np.asarray(arrays["reducing_label_known"], dtype=bool)
    exact_distance = np.asarray(arrays["exact_distance"])
    if np.any(~np.isfinite(exact_distance[exact_known])) or np.any(exact_distance[exact_known] < 0):
        raise ValueError("Known exact-distance rows must contain finite non-negative labels")
    if np.any(~np.isnan(exact_distance[~exact_known])):
        raise ValueError("Unknown exact-distance rows must contain NaN, never a numeric label")
    optimal = np.asarray(arrays["optimal_action_mask"], dtype=bool)
    reducing = np.asarray(arrays["reducing_action_mask"], dtype=bool)
    if optimal.shape != masks.shape or reducing.shape != masks.shape:
        raise ValueError("Exact action-set arrays must match the action-mask shape")
    if np.any(optimal[~exact_known]):
        raise ValueError("Unavailable optimal-action rows must not contain partial labels")
    if np.any(reducing[~reducing_known]):
        raise ValueError("Unavailable reducing-action rows must not contain partial labels")
    if np.any(optimal & ~masks) or np.any(reducing & ~masks):
        raise ValueError("Exact action-set label includes an environment no-op")
    verified = exact_known & reducing_known
    if not np.array_equal(optimal[verified], reducing[verified]):
        raise ValueError("Verified reducing and optimal action masks disagree")
    known_non_goal = exact_known & (exact_distance > 0)
    if np.any(~optimal[known_non_goal].any(axis=1)):
        raise ValueError("Known non-goal state has an empty optimal action set")
    known_goal = exact_known & (exact_distance == 0)
    if np.any(optimal[known_goal]) or np.any(reducing[known_goal]):
        raise ValueError("Known goal state must have empty exact action masks")
    solver_known = np.asarray(arrays["solver_known"], dtype=bool)
    solver_solvable = np.asarray(arrays["solver_solvable"], dtype=bool)
    solver_censor = np.asarray(arrays["solver_censor_code"])
    reducing_censor = np.asarray(arrays["reducing_censor_code"])
    if not np.array_equal(exact_known, solver_known & solver_solvable):
        raise ValueError("exact_label_known disagrees with known-solvable solver status")
    if np.any(~np.isin(solver_censor, [0, 1, 2])) or np.any(~np.isin(reducing_censor, [0, 1, 2])):
        raise ValueError("Unknown solver censor code")
    if np.any((solver_censor == 0) != solver_known):
        raise ValueError("Solver censor codes disagree with solver_known")
    if np.any(reducing_known & (reducing_censor != 0)) or np.any(
        ~reducing_known & exact_known & (reducing_censor == 0)
    ):
        raise ValueError("Reducing censor codes disagree with reducing_label_known")

    episode_ids = np.asarray(arrays["episode_id"])
    bos = BOS_ACTION(masks.shape[1])
    missing_action = MISSING_ACTION(masks.shape[1])
    if np.any(recent < 0) or np.any(recent > missing_action):
        raise ValueError("recent_actions contains a class outside actions/BOS/MISSING")
    move_index = np.asarray(arrays["move_index"])
    suite_puzzle_index = np.asarray(arrays["suite_puzzle_index"])
    if not np.array_equal(episode_ids, suite_puzzle_index):
        raise ValueError("One-episode-per-suite-puzzle grouping contract is broken")
    unique_episodes = np.unique(episode_ids)
    if not np.array_equal(unique_episodes, np.arange(len(unique_episodes))):
        raise ValueError("episode_id values must be contiguous and zero-based")
    if np.any(np.diff(episode_ids) < 0):
        raise ValueError("Rows from one episode must form one contiguous block")
    horizon = None
    environment = metadata.get("environment")
    if isinstance(environment, Mapping) and "horizon" in environment:
        horizon = int(environment["horizon"])
    for row in range(n_rows):
        first = row == 0 or episode_ids[row] != episode_ids[row - 1]
        for prefix in ("trained", "random"):
            in_h = np.asarray(arrays[f"{prefix}_carry_in_h"])[row]
            in_c = np.asarray(arrays[f"{prefix}_carry_in_c"])[row]
            if first:
                if np.count_nonzero(in_h) or np.count_nonzero(in_c):
                    raise ValueError(f"{prefix} carry does not reset at episode boundary")
            else:
                if not np.array_equal(in_h, np.asarray(arrays[f"{prefix}_carry_out_h"])[row - 1]) or not np.array_equal(
                    in_c, np.asarray(arrays[f"{prefix}_carry_out_c"])[row - 1]
                ):
                    raise ValueError(f"{prefix} carry-in/out threading is broken")
        episode_start = row - int(move_index[row])
        if episode_start < 0 or episode_ids[episode_start] != episode_ids[row]:
            raise ValueError("move_index is inconsistent with episode grouping")
        if move_index[row] != row - episode_start:
            raise ValueError("move_index is not the zero-based episode decision index")
        for lag in range(1, recent.shape[1] + 1):
            if lag <= move_index[row]:
                expected = actions[row - lag]
            elif lag == move_index[row] + 1:
                expected = bos
            else:
                expected = missing_action
            if recent[row, lag - 1] != expected:
                raise ValueError(f"Recent-action lag {lag} disagrees with trajectory history")
        if horizon is not None and not np.isclose(
            np.asarray(arrays["horizon_progress"])[row], move_index[row] / horizon
        ):
            raise ValueError("horizon_progress disagrees with move_index/horizon")

    future = np.asarray(arrays["future_success"], dtype=bool)
    immediate = np.asarray(arrays["immediate_success"], dtype=bool)
    if np.any(immediate & ~future):
        raise ValueError("Immediate success cannot occur in a failed episode")
    for episode_id in unique_episodes:
        episode_rows = np.flatnonzero(episode_ids == episode_id)
        if len(np.unique(future[episode_rows])) != 1:
            raise ValueError("future_success was not backfilled consistently per episode")
        if bool(immediate[episode_rows[-1]]) != bool(future[episode_rows[-1]]):
            raise ValueError("Final immediate-success label disagrees with episode outcome")
        if np.any(immediate[episode_rows[:-1]]):
            raise ValueError("An episode continued after an immediate-success move")
        for grouping_name in (
            "layout_hash",
            "puzzle_seed",
            "puzzle_optimal_length",
            "puzzle_robots_moved",
        ):
            if len(np.unique(np.asarray(arrays[grouping_name])[episode_rows])) != 1:
                raise ValueError(f"{grouping_name} changes within an episode")
    if "n_episodes" in metadata and int(metadata["n_episodes"]) != len(unique_episodes):
        raise ValueError("metadata n_episodes does not match the grouping arrays")
    episode_summary = metadata.get("episodes")
    if isinstance(episode_summary, Mapping):
        successes = int(sum(bool(future[episode_ids == episode_id][0]) for episode_id in unique_episodes))
        if (
            int(episode_summary.get("successes", -1)) != successes
            or int(episode_summary.get("failures", -1)) != len(unique_episodes) - successes
        ):
            raise ValueError("Episode outcome metadata disagrees with future_success")
    coverage = metadata.get("solver_coverage")
    if isinstance(coverage, Mapping):
        if (
            int(coverage.get("exact_known_rows", -1)) != int(exact_known.sum())
            or int(coverage.get("exact_total_rows", -1)) != n_rows
        ):
            raise ValueError("Solver coverage metadata disagrees with row availability")
        if int(coverage.get("reducing_known_rows", -1)) != int(reducing_known.sum()):
            raise ValueError("Reducing coverage metadata disagrees with row availability")
    label_metadata = metadata.get("labels")
    if isinstance(label_metadata, Mapping) and int(label_metadata.get("recent_action_lags", -1)) != recent.shape[1]:
        raise ValueError("Recent-action metadata disagrees with its array width")
    if "n_rows" in metadata and int(metadata["n_rows"]) != n_rows:
        raise ValueError("metadata n_rows does not match the arrays")


def save_carry_dataset(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    """Validate and atomically save a fingerprinted compressed NPZ dataset."""

    validate_carry_dataset(arrays, metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved_metadata = dict(metadata)
    saved_metadata["array_schema"] = {
        name: {"dtype": np.asarray(value).dtype.str, "shape": list(np.asarray(value).shape)}
        for name, value in sorted(arrays.items())
    }
    saved_metadata["content_sha256"] = carry_content_fingerprint(arrays, saved_metadata)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                **{name: np.asarray(value) for name, value in arrays.items()},
                metadata=np.asarray(_canonical_json(saved_metadata)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return saved_metadata


def load_carry_dataset(
    path: str | Path,
    *,
    expected_contract: Optional[Mapping[str, object]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load, fingerprint-check, structurally validate, and contract-check an NPZ."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if "metadata" not in payload.files:
            raise ValueError(f"Carry dataset lacks metadata: {source}")
        metadata = json.loads(str(payload["metadata"].item()))
        arrays = {name: payload[name].copy() for name in payload.files if name != "metadata"}
    expected_sha = metadata.get("content_sha256")
    actual_sha = carry_content_fingerprint(arrays, metadata)
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ValueError(f"Carry dataset fingerprint mismatch: {source}")
    if expected_contract is not None:
        actual_contract = metadata.get("contract")
        if not isinstance(actual_contract, Mapping):
            raise ValueError(f"Carry dataset contract missing: {source}")
        differences = _contract_differences(actual_contract, expected_contract)
        if differences:
            raise ValueError("Carry dataset contract mismatch: " + "; ".join(differences))
    validate_carry_dataset(arrays, metadata)
    return arrays, metadata


__all__ = [
    "BOS_ACTION",
    "MISSING_ACTION",
    "CAPTURE_PROTOCOL_VERSION",
    "DATASET_SCHEMA_VERSION",
    "SOLVER_PROTOCOL_VERSION",
    "ExactSolveResult",
    "ExactStateLabels",
    "ExactStateOracle",
    "SolverContract",
    "capture_carry_trajectories",
    "carry_content_fingerprint",
    "load_carry_dataset",
    "save_carry_dataset",
    "validate_carry_dataset",
]
