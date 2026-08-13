"""Causal recurrent-carry interventions for frozen schema-2 DRC policies.

The core experiment compares one untreated deterministic full-start rollout per
puzzle with deterministic rollouts whose *incoming* recurrent carry is altered.
The patch experiment first finds exact within-puzzle trajectory reconvergences
under locally seeded stochastic policy rollouts, then compares deterministic
suffixes from recipient and donor snapshots verified to be exactly identical.

This module deliberately contains no probe fitting.  Its experimental unit is
one selected puzzle (and at most one patch context per puzzle), never an
individual recurrent tick or environment step.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .drc import DRCNet, DRCState
from .drc_train import _file_sha256, _sha256_json
from .env import EnvConfig, RicochetEnv
from .probe_drc_carry import env_config_from_checkpoint, load_schema2_checkpoint, suite_provenance
from .puzzles import Puzzle, joint_stratified_subset, load_puzzles


RESULT_SCHEMA_VERSION = 1
INTERVENTION_PROTOCOL_VERSION = "drc-carry-causal-interventions-v1"
PATCH_PROTOCOL_VERSION = "drc-carry-exact-reconvergence-patching-v2"
BOOTSTRAP_PROTOCOL_VERSION = "paired-puzzle-percentile-bootstrap-v1"
DEFAULT_TIMING_MOVES = (1, 2, 4, 8)
PATCH_COMPONENTS = ("hc", "h", "c")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _normalize_timing_moves(timing_moves: Sequence[int]) -> tuple[int, ...]:
    values = []
    for move in timing_moves:
        if isinstance(move, (bool, np.bool_)) or not isinstance(move, (int, np.integer)):
            raise ValueError(f"timing move must be an integer, got {move!r}")
        values.append(int(move))
    normalized = tuple(sorted(set(values)))
    if not normalized or any(move < 1 for move in normalized):
        raise ValueError("timing_moves must contain positive completed-move counts")
    return normalized


def _jsonable(value: object) -> object:
    """Convert an environment snapshot to strict deterministic JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot make {type(value).__name__} JSON-safe")


def _environment_state_fingerprint(state: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(_jsonable(state)).encode("utf-8")).hexdigest()


def _json_value_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def _json_values_exactly_equal(left: object, right: object) -> bool:
    """Compare nested snapshot values directly, independently of their hashes."""

    return _canonical_json(_jsonable(left)) == _canonical_json(_jsonable(right))


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _arrays_byte_identical(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and left_array.tobytes() == right_array.tobytes()
    )


def _tensor_fingerprint(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return _array_fingerprint(array)


def clone_carry_state(state: DRCState) -> DRCState:
    """Return a tensor-independent copy of a DRC carry."""

    return DRCState(h=state.h.clone(), c=state.c.clone())


def _validate_component(component: str) -> None:
    if component not in PATCH_COMPONENTS:
        raise ValueError(f"component must be one of {PATCH_COMPONENTS}, got {component!r}")


def _validate_state(state: DRCState) -> None:
    if state.h.shape != state.c.shape:
        raise ValueError(f"h/c carry shapes differ: {tuple(state.h.shape)} != {tuple(state.c.shape)}")
    if state.h.ndim != 5:
        raise ValueError(f"DRC carry must have shape (D, B, C, H, W), got {tuple(state.h.shape)}")
    if state.h.device != state.c.device or state.h.dtype != state.c.dtype:
        raise ValueError("h/c carry device and dtype must match")


def zero_carry_state(state: DRCState, *, component: str = "hc", depth: Optional[int] = None) -> DRCState:
    """Zero exactly the requested incoming component/depth without mutation."""

    _validate_component(component)
    _validate_state(state)
    if depth is not None:
        if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)):
            raise ValueError(f"depth must be an integer, got {depth!r}")
        if not 0 <= int(depth) < state.h.shape[0]:
            raise ValueError(f"depth {depth} outside [0, {state.h.shape[0]})")
    h = state.h.clone()
    c = state.c.clone()
    target = slice(None) if depth is None else int(depth)
    if component in {"h", "hc"}:
        h[target].zero_()
    if component in {"c", "hc"}:
        c[target].zero_()
    return DRCState(h=h, c=c)


def patch_carry_state(recipient: DRCState, donor: DRCState, *, component: str = "hc") -> DRCState:
    """Copy requested donor components into a cloned recipient carry."""

    _validate_component(component)
    _validate_state(recipient)
    _validate_state(donor)
    if recipient.h.shape != donor.h.shape:
        raise ValueError("Recipient and donor carry shapes differ")
    if recipient.h.device != donor.h.device or recipient.h.dtype != donor.h.dtype:
        raise ValueError("Recipient and donor carry device/dtype differ")
    h = donor.h.clone() if component in {"h", "hc"} else recipient.h.clone()
    c = donor.c.clone() if component in {"c", "hc"} else recipient.c.clone()
    return DRCState(h=h, c=c)


@dataclass(frozen=True)
class InterventionSpec:
    """One prespecified transformation of incoming recurrent state."""

    intervention_id: str
    family: str
    component: str = "hc"
    depth: Optional[int] = None
    after_completed_moves: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_component(self.component)
        if self.family not in {"reset_every_move", "reset_once"}:
            raise ValueError(f"Unsupported intervention family {self.family!r}")
        if self.depth is not None and (
            isinstance(self.depth, (bool, np.bool_)) or not isinstance(self.depth, (int, np.integer))
        ):
            raise ValueError(f"depth must be an integer, got {self.depth!r}")
        if self.depth is not None and (self.depth < 0 or self.component != "hc"):
            raise ValueError("Per-depth interventions require a non-negative depth and component='hc'")
        if self.family == "reset_every_move" and self.after_completed_moves is not None:
            raise ValueError("reset_every_move cannot specify after_completed_moves")
        if self.family == "reset_once":
            if self.depth is not None or self.component != "hc":
                raise ValueError("One-shot timing reset must zero all h+c depths")
            if self.after_completed_moves is None or self.after_completed_moves < 1:
                raise ValueError("One-shot timing reset requires after_completed_moves >= 1")

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def intervention_specs(depth: int, timing_moves: Sequence[int] = DEFAULT_TIMING_MOVES) -> tuple[InterventionSpec, ...]:
    """Build the complete, deterministically ordered core intervention set."""

    if depth < 1:
        raise ValueError("DRC depth must be positive")
    timing = _normalize_timing_moves(timing_moves)
    specs = [
        InterventionSpec("reset_all_every_move", "reset_every_move"),
        *(
            InterventionSpec(f"reset_depth_{index}_every_move", "reset_every_move", depth=index)
            for index in range(depth)
        ),
        InterventionSpec("reset_h_every_move", "reset_every_move", component="h"),
        InterventionSpec("reset_c_every_move", "reset_every_move", component="c"),
        *(
            InterventionSpec(
                f"reset_once_after_{move}_moves",
                "reset_once",
                after_completed_moves=move,
            )
            for move in timing
        ),
    ]
    return tuple(specs)


@dataclass(frozen=True)
class EpisodeTrace:
    success: bool
    actions: tuple[int, ...]
    positions_before: tuple[tuple[tuple[int, int], ...], ...]
    final_positions: tuple[tuple[int, int], ...]
    applied_at_completed_moves: tuple[int, ...]

    @property
    def moves(self) -> int:
        return len(self.actions)

    def public_outcome(self) -> Dict[str, object]:
        return {
            "success": bool(self.success),
            "moves": int(self.moves),
            "actions": list(self.actions),
            "final_positions": [list(position) for position in self.final_positions],
        }


def _check_model_environment(net: DRCNet, puzzle: Puzzle, env: RicochetEnv) -> None:
    if not env.config.canonical_identity:
        raise ValueError("Carry interventions require canonical_identity=True")
    if env.config.rgb_obs:
        raise ValueError("Carry interventions require a tensor observation")
    if tuple(env.observation_space.shape or ()) != tuple(net.observation_shape):
        raise ValueError(
            f"Environment observation shape {env.observation_space.shape} != checkpoint {net.observation_shape}"
        )
    expected_actions = 4 * puzzle.num_robots
    if net.n_actions != expected_actions:
        raise ValueError(f"Checkpoint has {net.n_actions} actions but puzzle requires {expected_actions}")


@torch.no_grad()
def run_deterministic_episode(
    net: DRCNet,
    puzzle: Puzzle,
    env_config: EnvConfig,
    *,
    intervention: Optional[InterventionSpec] = None,
    env_seed: int = 0,
) -> EpisodeTrace:
    """Run one masked full-start episode, optionally altering incoming carry."""

    if intervention is not None and intervention.depth is not None and intervention.depth >= net.depth:
        raise ValueError(f"Intervention depth {intervention.depth} outside checkpoint depth {net.depth}")
    env = RicochetEnv([puzzle], env_config, sampler=None, seed=int(env_seed))
    _check_model_environment(net, puzzle, env)
    obs, _info = env.reset(options={"puzzle_index": 0, "start_distance": puzzle.optimal_length})
    device = next(net.parameters()).device
    state = net.initial_state(1, device=device)
    actions: list[int] = []
    positions_before: list[tuple[tuple[int, int], ...]] = []
    applications: list[int] = []
    success = False

    while True:
        completed = len(actions)
        should_apply = intervention is not None and (
            (intervention.family == "reset_every_move" and completed > 0)
            or (
                intervention.family == "reset_once"
                and completed == intervention.after_completed_moves
                and not applications
            )
        )
        if should_apply:
            state = zero_carry_state(
                state,
                component=intervention.component,
                depth=intervention.depth,
            )
            applications.append(completed)

        positions_before.append(tuple(tuple(map(int, position)) for position in env._robots))
        legal_mask = env.action_masks()
        if not legal_mask.any():
            raise RuntimeError("Encountered a state with no legal actions")
        obs_tensor = torch.as_tensor(np.asarray(obs)[None], device=device)
        mask_tensor = torch.as_tensor(legal_mask[None], device=device)
        logits, _value, state_out = net(obs_tensor, state, action_mask=mask_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Policy produced non-finite deterministic logits")
        action = int(logits.argmax(dim=-1).item())
        if not legal_mask[action]:
            raise AssertionError("Masked deterministic policy selected an illegal action")
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        state = state_out
        if terminated or truncated:
            success = bool(info["is_success"])
            break

    return EpisodeTrace(
        success=success,
        actions=tuple(actions),
        positions_before=tuple(positions_before),
        final_positions=tuple(tuple(map(int, position)) for position in env._robots),
        applied_at_completed_moves=tuple(applications),
    )


def _puzzle_identity(puzzle: Puzzle, suite_puzzle_index: int) -> Dict[str, object]:
    return {
        "suite_puzzle_index": int(suite_puzzle_index),
        "puzzle_seed": int(puzzle.seed),
        "layout_hash": str(puzzle.layout_hash),
        "optimal_length": int(puzzle.optimal_length),
        "robots_moved": int(puzzle.robots_moved),
    }


def _effect_record(
    identity: Mapping[str, object],
    *,
    baseline: Mapping[str, object],
    intervention: Mapping[str, object],
) -> Dict[str, object]:
    baseline_success = bool(baseline["success"])
    intervention_success = bool(intervention["success"])
    return {
        **dict(identity),
        "baseline_success": baseline_success,
        "intervention_success": intervention_success,
        "paired_success_delta": int(intervention_success) - int(baseline_success),
        "baseline": dict(baseline),
        "intervention": dict(intervention),
    }


def _bootstrap_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    reps: int,
) -> Dict[str, object]:
    if reps < 1:
        raise ValueError("bootstrap reps must be positive")
    deltas = np.asarray(
        [int(bool(row["intervention_success"])) - int(bool(row["baseline_success"])) for row in rows],
        dtype=np.float64,
    )
    n = int(len(deltas))
    baseline_successes = sum(bool(row["baseline_success"]) for row in rows)
    intervention_successes = sum(bool(row["intervention_success"]) for row in rows)
    common_bootstrap = {
        "protocol_version": BOOTSTRAP_PROTOCOL_VERSION,
        "unit": "puzzle",
        "confidence_level": 0.95,
        "seed": int(seed),
        "reps": int(reps),
    }
    if n == 0:
        return {
            "n": 0,
            "size_status": "empty",
            "baseline_successes": 0,
            "baseline_success_rate": None,
            "intervention_successes": 0,
            "intervention_success_rate": None,
            "mean_paired_success_delta": None,
            "improved": 0,
            "harmed": 0,
            "tied": 0,
            "bootstrap_ci": {**common_bootstrap, "status": "empty", "lower": None, "upper": None},
        }

    mean_delta = float(deltas.mean())
    if n == 1:
        lower = upper = mean_delta
        status = "single_item_degenerate"
        size_status = "single_item"
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, n, size=(reps, n))
        sampled_means = deltas[indices].mean(axis=1)
        lower, upper = (float(value) for value in np.quantile(sampled_means, [0.025, 0.975], method="linear"))
        status = "ok"
        size_status = "adequate"
    return {
        "n": n,
        "size_status": size_status,
        "baseline_successes": int(baseline_successes),
        "baseline_success_rate": float(baseline_successes / n),
        "intervention_successes": int(intervention_successes),
        "intervention_success_rate": float(intervention_successes / n),
        "mean_paired_success_delta": mean_delta,
        "improved": int(np.count_nonzero(deltas > 0)),
        "harmed": int(np.count_nonzero(deltas < 0)),
        "tied": int(np.count_nonzero(deltas == 0)),
        "bootstrap_ci": {**common_bootstrap, "status": status, "lower": lower, "upper": upper},
    }


def aggregate_paired_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int,
    bootstrap_reps: int,
    seed_namespace: str = "effect",
    reference_strata: Optional[Sequence[tuple[int, int]]] = None,
) -> Dict[str, object]:
    """Aggregate paired binary outcomes overall and across preregistered strata."""

    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    for row in rows:
        required = {"optimal_length", "robots_moved", "baseline_success", "intervention_success"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Paired effect row missing fields: {missing}")

    if reference_strata is None:
        strata = sorted({(int(row["optimal_length"]), int(row["robots_moved"])) for row in rows})
    else:
        strata = sorted({(int(length), int(robots)) for length, robots in reference_strata})
        row_strata = {(int(row["optimal_length"]), int(row["robots_moved"])) for row in rows}
        unexpected = sorted(row_strata - set(strata))
        if unexpected:
            raise ValueError(f"Paired effect rows contain strata outside reference_strata: {unexpected}")
    lengths = sorted({length for length, _robots in strata})
    robot_counts = sorted({robots for _length, robots in strata})

    def summarize(group: Sequence[Mapping[str, object]], *parts: object) -> Dict[str, object]:
        seed = _stable_seed(bootstrap_seed, seed_namespace, *parts)
        return _bootstrap_summary(group, seed=seed, reps=bootstrap_reps)

    by_length = {
        str(length): summarize(
            [row for row in rows if int(row["optimal_length"]) == length],
            "optimal_length",
            length,
        )
        for length in lengths
    }
    by_robots = {
        str(robots): summarize(
            [row for row in rows if int(row["robots_moved"]) == robots],
            "robots_moved",
            robots,
        )
        for robots in robot_counts
    }
    # Emit the full Cartesian product of observed marginals. Sparse/missing
    # joint cells are explicit empty groups rather than silently disappearing.
    by_joint = {}
    for length in lengths:
        for robots in robot_counts:
            summary = summarize(
                [row for row in rows if int(row["optimal_length"]) == length and int(row["robots_moved"]) == robots],
                "joint",
                length,
                robots,
            )
            by_joint[f"{length}|{robots}"] = {
                "optimal_length": int(length),
                "robots_moved": int(robots),
                **summary,
            }
    return {
        "estimand": "mean(intervention_success - baseline_success)",
        "bootstrap_unit": "puzzle",
        "bootstrap_base_seed": int(bootstrap_seed),
        "bootstrap_reps": int(bootstrap_reps),
        "overall": summarize(rows, "overall"),
        "by_optimal_length": by_length,
        "by_robots_moved": by_robots,
        "by_joint_cell": by_joint,
    }


def _suite_strata(puzzles: Sequence[Puzzle]) -> list[tuple[int, int]]:
    return sorted({(int(puzzle.optimal_length), int(puzzle.robots_moved)) for puzzle in puzzles})


def evaluate_core_interventions(
    net: DRCNet,
    puzzles: Sequence[Puzzle],
    env_config: EnvConfig,
    *,
    timing_moves: Sequence[int] = DEFAULT_TIMING_MOVES,
    rollout_seed: int = 0,
    bootstrap_seed: int = 42_424,
    bootstrap_reps: int = 10_000,
) -> Dict[str, object]:
    """Evaluate baseline and every prespecified reset intervention."""

    if not puzzles:
        raise ValueError("Cannot evaluate an empty puzzle suite")
    specs = intervention_specs(net.depth, timing_moves)
    strata = _suite_strata(puzzles)
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            baselines = [
                run_deterministic_episode(
                    net,
                    puzzle,
                    env_config,
                    env_seed=_stable_seed(rollout_seed, "core", index),
                )
                for index, puzzle in enumerate(puzzles)
            ]
            baseline_records = [
                {
                    **_puzzle_identity(puzzle, index),
                    "environment_seed": _stable_seed(rollout_seed, "core", index),
                    **trace.public_outcome(),
                }
                for index, (puzzle, trace) in enumerate(zip(puzzles, baselines))
            ]
            interventions: Dict[str, object] = {}
            for spec in specs:
                records = []
                for index, (puzzle, baseline) in enumerate(zip(puzzles, baselines)):
                    trace = run_deterministic_episode(
                        net,
                        puzzle,
                        env_config,
                        intervention=spec,
                        env_seed=_stable_seed(rollout_seed, "core", index),
                    )
                    identity = _puzzle_identity(puzzle, index)
                    record = _effect_record(
                        identity,
                        baseline=baseline.public_outcome(),
                        intervention=trace.public_outcome(),
                    )
                    record["environment_seed"] = _stable_seed(rollout_seed, "core", index)
                    record["intervention_applied"] = bool(trace.applied_at_completed_moves)
                    record["applications"] = int(len(trace.applied_at_completed_moves))
                    record["applied_at_completed_moves"] = list(trace.applied_at_completed_moves)
                    if spec.family == "reset_once":
                        timing = int(spec.after_completed_moves)
                        eligible = baseline.moves > timing
                        if trace.actions[:timing] != baseline.actions[:timing]:
                            raise AssertionError("One-shot intervention diverged before its intervention point")
                        if trace.positions_before[: timing + 1] != baseline.positions_before[: timing + 1]:
                            raise AssertionError("One-shot environment path diverged before intervention")
                        if bool(trace.applied_at_completed_moves) != eligible:
                            raise AssertionError("One-shot application disagrees with untreated-path eligibility")
                        if not eligible and (trace.actions != baseline.actions or trace.success != baseline.success):
                            raise AssertionError("Ineligible one-shot episode differs from untreated baseline")
                        record["eligible_pre_intervention"] = bool(eligible)
                    else:
                        record["eligible_pre_intervention"] = None
                    records.append(record)

                effects = aggregate_paired_effects(
                    records,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_reps=bootstrap_reps,
                    seed_namespace=spec.intervention_id,
                    reference_strata=strata,
                )
                interventions[spec.intervention_id] = {
                    "spec": spec.as_dict(),
                    "application_counts": {
                        "puzzles_applied": int(sum(bool(row["intervention_applied"]) for row in records)),
                        "total_applications": int(sum(int(row["applications"]) for row in records)),
                        "eligible_pre_intervention": (
                            int(sum(bool(row["eligible_pre_intervention"]) for row in records))
                            if spec.family == "reset_once"
                            else None
                        ),
                        "not_eligible_pre_intervention": (
                            int(sum(not bool(row["eligible_pre_intervention"]) for row in records))
                            if spec.family == "reset_once"
                            else None
                        ),
                    },
                    "effects": effects,
                    "paired_outcomes": records,
                }
    finally:
        net.train(was_training)

    baseline_rate = sum(record["success"] for record in baseline_records) / len(baseline_records)
    return {
        "baseline": {
            "definition": "deterministic_masked_full_start_carry_zero_only_at_episode_start",
            "n": int(len(baseline_records)),
            "success_rate": float(baseline_rate),
            "outcomes": baseline_records,
        },
        "interventions": interventions,
    }


@dataclass(frozen=True)
class PatchContext:
    """Prefix-only state at one stochastic-rollout decision point."""

    suite_puzzle_index: int
    puzzle_seed: int
    layout_hash: str
    rollout_id: int
    rollout_seed: int
    move_index: int
    positions: tuple[tuple[int, int], ...]
    action_history: tuple[int, ...]
    observation: np.ndarray
    environment_state: Mapping[str, object]
    carry: DRCState

    @property
    def environment_fingerprint(self) -> str:
        return _environment_state_fingerprint(self.environment_state)


@dataclass(frozen=True)
class PatchPair:
    recipient: PatchContext
    donor: PatchContext


def _validate_patch_context_snapshot(context: PatchContext, *, role: str) -> None:
    required_fields = {
        "puzzle_index",
        "robots",
        "steps",
        "start_distance",
        "episode_done",
        "rng_state",
    }
    missing = sorted(required_fields - set(context.environment_state))
    if missing:
        raise AssertionError(f"{role} environment snapshot is missing fields: {missing}")
    if int(context.move_index) != len(context.action_history):
        raise AssertionError(f"{role} move index/history length mismatch")
    if int(context.environment_state["steps"]) != int(context.move_index):
        raise AssertionError(f"{role} environment steps do not match the context move index")
    if context.environment_state["puzzle_index"] != 0:
        raise AssertionError(f"{role} environment snapshot does not reference the single local puzzle")
    snapshot_robots = context.environment_state["robots"]
    if not isinstance(snapshot_robots, (tuple, list)):
        raise AssertionError(f"{role} environment robots are not a sequence")
    snapshot_positions = tuple(tuple(map(int, position)) for position in snapshot_robots)
    if snapshot_positions != context.positions:
        raise AssertionError(f"{role} environment robots do not match the context positions")
    if bool(context.environment_state["episode_done"]):
        raise AssertionError(f"{role} patch context is already terminal")
    if not isinstance(context.environment_state["rng_state"], Mapping):
        raise AssertionError(f"{role} environment RNG state is not a mapping")


def _matched_context_verification(recipient: PatchContext, donor: PatchContext) -> Dict[str, object]:
    """Verify every saved-state equality required by exact reconvergence."""

    _validate_patch_context_snapshot(recipient, role="Recipient")
    _validate_patch_context_snapshot(donor, role="Donor")
    observations_equal = _arrays_byte_identical(recipient.observation, donor.observation)
    steps_equal = int(recipient.environment_state["steps"]) == int(donor.environment_state["steps"])
    rng_equal = _json_values_exactly_equal(
        recipient.environment_state["rng_state"],
        donor.environment_state["rng_state"],
    )
    snapshots_equal = _json_values_exactly_equal(recipient.environment_state, donor.environment_state)
    if not observations_equal:
        raise AssertionError("Matched positions/time produced observations that are not byte-identical")
    if not steps_equal:
        raise AssertionError("Exact matched contexts have different environment time/steps")
    if not rng_equal:
        raise AssertionError("Exact matched contexts have different environment RNG states")
    if not snapshots_equal:
        raise AssertionError("Exact matched contexts have different environment snapshots")
    return {
        "recipient_snapshot_matches_context_move_and_positions": True,
        "donor_snapshot_matches_context_move_and_positions": True,
        "environment_snapshots_exactly_equal": True,
        "environment_rng_states_exactly_equal": True,
        "environment_time_steps_exactly_equal": True,
        "saved_observations_byte_identical": True,
    }


def select_matched_patch_pair(contexts: Sequence[PatchContext]) -> Optional[PatchPair]:
    """Select the earliest exact reconvergence using prefix information only."""

    groups: dict[tuple[object, ...], list[PatchContext]] = defaultdict(list)
    for context in contexts:
        if context.move_index <= 0:
            continue
        key = (
            int(context.suite_puzzle_index),
            int(context.puzzle_seed),
            str(context.layout_hash),
            int(context.move_index),
            context.positions,
        )
        groups[key].append(context)

    for key in sorted(groups, key=lambda value: (value[3], value[4], value[0], value[1], value[2])):
        candidates = sorted(
            groups[key],
            key=lambda context: (context.rollout_id, context.action_history, context.rollout_seed),
        )
        for recipient_index, recipient in enumerate(candidates):
            for donor in candidates[recipient_index + 1 :]:
                if recipient.rollout_id == donor.rollout_id:
                    continue
                if recipient.action_history == donor.action_history:
                    continue
                _matched_context_verification(recipient, donor)
                return PatchPair(recipient=recipient, donor=donor)
    return None


@torch.no_grad()
def _collect_stochastic_contexts(
    net: DRCNet,
    puzzle: Puzzle,
    env_config: EnvConfig,
    *,
    suite_puzzle_index: int,
    rollout_count: int,
    patch_seed: int,
) -> tuple[list[PatchContext], list[Dict[str, object]]]:
    contexts: list[PatchContext] = []
    manifests: list[Dict[str, object]] = []
    device = next(net.parameters()).device
    environment_seed = _stable_seed(patch_seed, "patch_environment", suite_puzzle_index)

    for rollout_id in range(rollout_count):
        rollout_seed = _stable_seed(patch_seed, "patch_source", suite_puzzle_index, rollout_id)
        generator = torch.Generator(device="cpu").manual_seed(rollout_seed)
        env = RicochetEnv([puzzle], env_config, sampler=None, seed=environment_seed)
        _check_model_environment(net, puzzle, env)
        obs, _info = env.reset(options={"puzzle_index": 0, "start_distance": puzzle.optimal_length})
        state = net.initial_state(1, device=device)
        actions: list[int] = []
        success = False
        while True:
            if actions:
                contexts.append(
                    PatchContext(
                        suite_puzzle_index=int(suite_puzzle_index),
                        puzzle_seed=int(puzzle.seed),
                        layout_hash=str(puzzle.layout_hash),
                        rollout_id=int(rollout_id),
                        rollout_seed=int(rollout_seed),
                        move_index=int(len(actions)),
                        positions=tuple(tuple(map(int, position)) for position in env._robots),
                        action_history=tuple(actions),
                        observation=np.asarray(obs).copy(),
                        environment_state=copy.deepcopy(env.state_dict()),
                        carry=clone_carry_state(state),
                    )
                )

            legal_mask = env.action_masks()
            if not legal_mask.any():
                raise RuntimeError("Encountered a stochastic source state with no legal actions")
            obs_tensor = torch.as_tensor(np.asarray(obs)[None], device=device)
            mask_tensor = torch.as_tensor(legal_mask[None], device=device)
            logits, _value, state_out = net(obs_tensor, state, action_mask=mask_tensor)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Policy produced non-finite stochastic-source logits")
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())
            if not legal_mask[action]:
                raise AssertionError("Masked stochastic policy selected an illegal action")
            obs, _reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            state = state_out
            if terminated or truncated:
                success = bool(info["is_success"])
                break
        manifests.append(
            {
                "rollout_id": int(rollout_id),
                "rollout_seed": int(rollout_seed),
                "environment_seed": int(environment_seed),
                "moves": int(len(actions)),
                "success": bool(success),
                "action_history_sha256": _sha256_json(actions),
            }
        )
    return contexts, manifests


@torch.no_grad()
def _run_deterministic_suffix(
    net: DRCNet,
    puzzle: Puzzle,
    env_config: EnvConfig,
    *,
    environment_state: Mapping[str, object],
    carry: DRCState,
    expected_observation: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    snapshot = copy.deepcopy(dict(environment_state))
    env = RicochetEnv([puzzle], env_config, sampler=None, seed=0)
    _check_model_environment(net, puzzle, env)
    env.load_state_dict(copy.deepcopy(snapshot))
    restored_snapshot = copy.deepcopy(env.state_dict())
    if not _json_values_exactly_equal(snapshot, restored_snapshot):
        raise AssertionError("Fresh suffix environment did not restore the saved snapshot exactly")
    if env._episode_done:
        raise ValueError("Patch suffix cannot start from a completed episode")
    obs = env._obs()
    observation_matches_saved = expected_observation is not None and _arrays_byte_identical(
        np.asarray(obs), np.asarray(expected_observation)
    )
    if expected_observation is not None and not observation_matches_saved:
        raise AssertionError("Saved patch environment did not reproduce its saved observation exactly")
    initial_observation_sha256 = _array_fingerprint(np.asarray(obs))
    initial_move_index = int(env._steps)
    state = clone_carry_state(carry)
    device = next(net.parameters()).device
    actions: list[int] = []
    success = False
    while True:
        legal_mask = env.action_masks()
        if not legal_mask.any():
            raise RuntimeError("Encountered a patch suffix state with no legal actions")
        obs_tensor = torch.as_tensor(np.asarray(obs)[None], device=device)
        mask_tensor = torch.as_tensor(legal_mask[None], device=device)
        logits, _value, state = net(obs_tensor, state, action_mask=mask_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Policy produced non-finite deterministic-suffix logits")
        action = int(logits.argmax(dim=-1).item())
        if not legal_mask[action]:
            raise AssertionError("Masked deterministic suffix selected an illegal action")
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        if terminated or truncated:
            success = bool(info["is_success"])
            break
    return {
        "success": bool(success),
        "suffix_moves": int(len(actions)),
        "total_episode_moves": int(initial_move_index + len(actions)),
        "actions": actions,
        "final_positions": [list(map(int, position)) for position in env._robots],
        "initial_environment_sha256": _environment_state_fingerprint(snapshot),
        "restored_environment_sha256": _environment_state_fingerprint(restored_snapshot),
        "initial_environment_rng_sha256": _json_value_fingerprint(snapshot["rng_state"]),
        "initial_completed_move_index": initial_move_index,
        "initial_observation_sha256": initial_observation_sha256,
        "snapshot_restored_exactly": True,
        "saved_observation_reproduced_exactly": observation_matches_saved if expected_observation is not None else None,
    }


def _action_divergence(natural: Sequence[int], patched: Sequence[int]) -> Dict[str, object]:
    differences = [
        index for index, (left, right) in enumerate(zip_longest(natural, patched, fillvalue=None)) if left != right
    ]
    return {
        "diverged": bool(differences),
        "first_divergence_suffix_index": int(differences[0]) if differences else None,
        "positionwise_differences_including_length": int(len(differences)),
    }


def _verify_donor_natural_hc_control(
    donor_natural: Mapping[str, object],
    hc_transplant: Mapping[str, object],
) -> Dict[str, object]:
    """Require the full donor transplant to reproduce the donor's own suffix."""

    outcome_fields = ("success", "suffix_moves", "total_episode_moves", "final_positions")
    actions_identical = donor_natural["actions"] == hc_transplant["actions"]
    outcomes_identical = all(donor_natural[field] == hc_transplant[field] for field in outcome_fields)
    initial_environment_identical = (
        donor_natural["initial_environment_sha256"] == hc_transplant["initial_environment_sha256"]
    )
    initial_observation_identical = (
        donor_natural["initial_observation_sha256"] == hc_transplant["initial_observation_sha256"]
    )
    if (
        not actions_identical
        or not outcomes_identical
        or not initial_environment_identical
        or not initial_observation_identical
    ):
        mismatched = [field for field in outcome_fields if donor_natural[field] != hc_transplant[field]]
        if not actions_identical:
            mismatched.append("actions")
        if not initial_environment_identical:
            mismatched.append("initial_environment")
        if not initial_observation_identical:
            mismatched.append("initial_observation")
        raise AssertionError(
            "Donor-natural suffix and hc donor-to-recipient transplant differ in " + ", ".join(mismatched)
        )
    return {
        "role": "required_positive_control_not_primary_estimand",
        "donor_natural_source": "donor.environment_state + donor.carry",
        "hc_transplant_source": "recipient.environment_state + donor.h + donor.c",
        "compared_outcome_fields": list(outcome_fields),
        "outcomes_identical": True,
        "actions_identical": True,
        "initial_environment_identical": True,
        "initial_observation_identical": True,
        "passed": True,
    }


def _carry_fingerprints(state: DRCState) -> Dict[str, str]:
    return {"h": _tensor_fingerprint(state.h), "c": _tensor_fingerprint(state.c)}


def _patch_verification(
    recipient: DRCState,
    donor: DRCState,
    patched: DRCState,
    *,
    component: str,
    recipient_before: Mapping[str, str],
    donor_before: Mapping[str, str],
) -> Dict[str, object]:
    requested = set(component) if component != "hc" else {"h", "c"}
    component_values = {
        "h": (recipient.h, donor.h, patched.h),
        "c": (recipient.c, donor.c, patched.c),
    }
    requested_match = all(torch.equal(component_values[name][1], component_values[name][2]) for name in requested)
    unrequested = {"h", "c"} - requested
    unrequested_match = all(torch.equal(component_values[name][0], component_values[name][2]) for name in unrequested)
    recipient_after = _carry_fingerprints(recipient)
    donor_after = _carry_fingerprints(donor)
    if recipient_after != dict(recipient_before) or donor_after != dict(donor_before):
        raise AssertionError("Carry patch mutated a source state")
    if not requested_match or not unrequested_match:
        raise AssertionError("Carry patch changed a component outside its request")
    return {
        "requested_components": sorted(requested),
        "requested_components_equal_donor": bool(requested_match),
        "unrequested_components_equal_recipient": bool(unrequested_match),
        "recipient_input_unmutated": True,
        "donor_input_unmutated": True,
        "requested_component_values_differed": bool(
            any(not torch.equal(component_values[name][0], component_values[name][1]) for name in requested)
        ),
        "only_requested_components_written": True,
    }


def evaluate_matched_patches(
    net: DRCNet,
    puzzles: Sequence[Puzzle],
    env_config: EnvConfig,
    *,
    patch_rollouts: int,
    patch_seed: int = 73_737,
    bootstrap_seed: int = 42_424,
    bootstrap_reps: int = 10_000,
) -> Dict[str, object]:
    """Run exact within-puzzle carry patches with at most one context each."""

    if patch_rollouts < 0 or patch_rollouts == 1:
        raise ValueError("patch_rollouts must be 0 (disabled) or at least 2")
    if not puzzles:
        raise ValueError("Cannot evaluate patches on an empty puzzle suite")
    if patch_rollouts == 0:
        return {
            "status": "disabled",
            "reason": "patch_rollouts=0",
            "protocol_version": PATCH_PROTOCOL_VERSION,
            "primary_estimand": ("mean(recipient_patched_suffix_success - recipient_natural_suffix_success)"),
            "donor_natural_control": {
                "role": "required_positive_control_not_primary_estimand",
                "status": "not_evaluated",
            },
            "source_rollouts_per_puzzle": 0,
            "eligibility": {
                "status": "not_evaluated",
                "suite_puzzles": int(len(puzzles)),
                "eligible_puzzles": None,
                "no_match_puzzles": None,
            },
            "patches": {
                component: {
                    "status": "disabled",
                    "paired_outcomes": None,
                    "effects": None,
                    "action_divergence_counts": None,
                }
                for component in PATCH_COMPONENTS
            },
        }

    was_training = net.training
    net.eval()
    strata = _suite_strata(puzzles)
    records: dict[str, list[Dict[str, object]]] = {component: [] for component in PATCH_COMPONENTS}
    source_manifests = []
    no_matches = []
    eligible = 0
    try:
        with torch.no_grad():
            for puzzle_index, puzzle in enumerate(puzzles):
                contexts, rollout_manifest = _collect_stochastic_contexts(
                    net,
                    puzzle,
                    env_config,
                    suite_puzzle_index=puzzle_index,
                    rollout_count=patch_rollouts,
                    patch_seed=patch_seed,
                )
                identity = _puzzle_identity(puzzle, puzzle_index)
                source_manifests.append({**identity, "rollouts": rollout_manifest})
                pair = select_matched_patch_pair(contexts)
                if pair is None:
                    no_matches.append(
                        {
                            **identity,
                            "reason": "no_exact_same_move_positions_match_with_different_histories",
                            "source_rollout_ids": list(range(patch_rollouts)),
                        }
                    )
                    continue
                eligible += 1
                recipient, donor = pair.recipient, pair.donor
                if recipient.positions != donor.positions or recipient.move_index != donor.move_index:
                    raise AssertionError("Selected patch pair is not an exact reconvergence")
                expected_context_identity = (int(puzzle_index), int(puzzle.seed), str(puzzle.layout_hash))
                for role, context in (("Recipient", recipient), ("Donor", donor)):
                    actual_context_identity = (
                        int(context.suite_puzzle_index),
                        int(context.puzzle_seed),
                        str(context.layout_hash),
                    )
                    if actual_context_identity != expected_context_identity:
                        raise AssertionError(f"{role} patch context does not belong to the current puzzle")
                context_verification = _matched_context_verification(recipient, donor)

                recipient_before = _carry_fingerprints(recipient.carry)
                donor_before = _carry_fingerprints(donor.carry)
                recipient_environment_before = copy.deepcopy(recipient.environment_state)
                donor_environment_before = copy.deepcopy(donor.environment_state)
                recipient_observation_before = np.asarray(recipient.observation).copy()
                donor_observation_before = np.asarray(donor.observation).copy()
                recipient_natural = _run_deterministic_suffix(
                    net,
                    puzzle,
                    env_config,
                    environment_state=recipient.environment_state,
                    carry=recipient.carry,
                    expected_observation=recipient.observation,
                )
                donor_natural = _run_deterministic_suffix(
                    net,
                    puzzle,
                    env_config,
                    environment_state=donor.environment_state,
                    carry=donor.carry,
                    expected_observation=donor.observation,
                )
                context_record = {
                    "recipient_rollout_id": int(recipient.rollout_id),
                    "recipient_rollout_seed": int(recipient.rollout_seed),
                    "recipient_action_prefix": list(recipient.action_history),
                    "donor_rollout_id": int(donor.rollout_id),
                    "donor_rollout_seed": int(donor.rollout_seed),
                    "donor_action_prefix": list(donor.action_history),
                    "completed_move_index": int(recipient.move_index),
                    "canonical_positions": [list(position) for position in recipient.positions],
                    "environment_state_sha256": recipient.environment_fingerprint,
                    "environment_state_sha256_by_role": {
                        "recipient": recipient.environment_fingerprint,
                        "donor": donor.environment_fingerprint,
                    },
                    "environment_rng_state_sha256_by_role": {
                        "recipient": _json_value_fingerprint(recipient.environment_state["rng_state"]),
                        "donor": _json_value_fingerprint(donor.environment_state["rng_state"]),
                    },
                    "observation_sha256": _array_fingerprint(recipient.observation),
                    "observation_sha256_by_role": {
                        "recipient": _array_fingerprint(recipient.observation),
                        "donor": _array_fingerprint(donor.observation),
                    },
                    "environment_snapshot_fields_compared": sorted(recipient.environment_state),
                    "different_action_histories": recipient.action_history != donor.action_history,
                    "selection_rule": "earliest_move_then_positions_then_lowest_rollout_ids",
                }
                if (
                    recipient_natural["initial_environment_sha256"]
                    != context_record["environment_state_sha256_by_role"]["recipient"]
                    or donor_natural["initial_environment_sha256"]
                    != context_record["environment_state_sha256_by_role"]["donor"]
                ):
                    raise AssertionError("A natural suffix did not start from its recorded environment snapshot")
                if (
                    recipient_natural["initial_observation_sha256"]
                    != context_record["observation_sha256_by_role"]["recipient"]
                    or donor_natural["initial_observation_sha256"]
                    != context_record["observation_sha256_by_role"]["donor"]
                ):
                    raise AssertionError("A natural suffix did not reproduce its recorded observation")

                patched_carries: dict[str, DRCState] = {}
                patched_suffixes: dict[str, Dict[str, object]] = {}
                patch_verifications: dict[str, Dict[str, object]] = {}
                for component in PATCH_COMPONENTS:
                    patched_carry = patch_carry_state(recipient.carry, donor.carry, component=component)
                    patch_verification = _patch_verification(
                        recipient.carry,
                        donor.carry,
                        patched_carry,
                        component=component,
                        recipient_before=recipient_before,
                        donor_before=donor_before,
                    )
                    patched = _run_deterministic_suffix(
                        net,
                        puzzle,
                        env_config,
                        environment_state=recipient.environment_state,
                        carry=patched_carry,
                        expected_observation=recipient.observation,
                    )
                    if recipient_natural["initial_environment_sha256"] != patched["initial_environment_sha256"]:
                        raise AssertionError(
                            "Recipient-natural and patched suffixes did not start from one environment state"
                        )
                    if recipient_natural["initial_observation_sha256"] != patched["initial_observation_sha256"]:
                        raise AssertionError(
                            "Recipient-natural and patched suffixes did not start from one observation"
                        )
                    patched_carries[component] = patched_carry
                    patched_suffixes[component] = patched
                    patch_verifications[component] = patch_verification

                donor_control = _verify_donor_natural_hc_control(donor_natural, patched_suffixes["hc"])
                if _carry_fingerprints(recipient.carry) != recipient_before:
                    raise AssertionError("Suffix execution mutated the recipient source carry")
                if _carry_fingerprints(donor.carry) != donor_before:
                    raise AssertionError("Suffix execution mutated the donor source carry")
                if not _json_values_exactly_equal(recipient.environment_state, recipient_environment_before):
                    raise AssertionError("Suffix execution mutated the recipient environment snapshot")
                if not _json_values_exactly_equal(donor.environment_state, donor_environment_before):
                    raise AssertionError("Suffix execution mutated the donor environment snapshot")
                if not _arrays_byte_identical(recipient.observation, recipient_observation_before):
                    raise AssertionError("Suffix execution mutated the recipient saved observation")
                if not _arrays_byte_identical(donor.observation, donor_observation_before):
                    raise AssertionError("Suffix execution mutated the donor saved observation")

                for component in PATCH_COMPONENTS:
                    patched = patched_suffixes[component]
                    audit = _effect_record(
                        identity,
                        baseline=recipient_natural,
                        intervention=patched,
                    )
                    audit.update(
                        {
                            "patch_component": component,
                            "primary_estimand_comparison": (
                                "recipient_patched_suffix_success - recipient_natural_suffix_success"
                            ),
                            "recipient_natural_suffix_success": bool(recipient_natural["success"]),
                            "donor_natural_suffix_success": bool(donor_natural["success"]),
                            "recipient_patched_suffix_success": bool(patched["success"]),
                            "context": context_record,
                            "suffix_roles": {
                                "natural_suffix": "recipient.environment_state + recipient.carry",
                                "donor_natural_suffix": "donor.environment_state + donor.carry",
                                "patched_suffix": (f"recipient.environment_state + {component} donor carry patch"),
                            },
                            "natural_suffix": audit.pop("baseline"),
                            "donor_natural_suffix": dict(donor_natural),
                            "patched_suffix": audit.pop("intervention"),
                            "action_divergence": {
                                "comparison": "recipient_natural_vs_recipient_patched",
                                **_action_divergence(
                                    recipient_natural["actions"],
                                    patched["actions"],
                                ),
                            },
                            "donor_natural_hc_transplant_control": dict(donor_control),
                            "initial_carry_sha256": {
                                "recipient": recipient_before,
                                "donor": donor_before,
                                "patched": _carry_fingerprints(patched_carries[component]),
                            },
                            "verification": {
                                **patch_verifications[component],
                                **context_verification,
                                "same_puzzle": True,
                                "same_completed_move_index": True,
                                "same_canonical_positions": True,
                                "recipient_snapshot_reproduced_exactly": True,
                                "donor_snapshot_reproduced_exactly": True,
                                "recipient_saved_observation_reproduced_exactly": True,
                                "donor_saved_observation_reproduced_exactly": True,
                                "recipient_natural_and_patch_initial_state_identical": True,
                                "recipient_natural_and_patch_initial_observation_identical": True,
                                "donor_natural_and_hc_transplant_outcomes_identical": True,
                                "donor_natural_and_hc_transplant_actions_identical": True,
                                "different_action_histories": True,
                                "no_state_reuse_between_suffix_runs": True,
                                "source_carries_unmutated_after_all_suffixes": True,
                                "source_environment_snapshots_unmutated_after_all_suffixes": True,
                                "source_observations_unmutated_after_all_suffixes": True,
                            },
                        }
                    )
                    records[component].append(audit)
    finally:
        net.train(was_training)

    patch_results = {}
    for component in PATCH_COMPONENTS:
        component_records = records[component]
        effects = aggregate_paired_effects(
            component_records,
            bootstrap_seed=bootstrap_seed,
            bootstrap_reps=bootstrap_reps,
            seed_namespace=f"patch_{component}",
            reference_strata=strata,
        )
        effects["estimand"] = "mean(recipient_patched_suffix_success - recipient_natural_suffix_success)"
        patch_results[component] = {
            "status": "complete" if component_records else "no_eligible_matches",
            "paired_outcomes": component_records,
            "effects": effects,
            "action_divergence_counts": {
                "diverged": int(sum(bool(row["action_divergence"]["diverged"]) for row in component_records)),
                "not_diverged": int(sum(not bool(row["action_divergence"]["diverged"]) for row in component_records)),
            },
        }
    return {
        "status": "complete" if eligible else "enabled_no_matches",
        "protocol_version": PATCH_PROTOCOL_VERSION,
        "primary_estimand": "mean(recipient_patched_suffix_success - recipient_natural_suffix_success)",
        "donor_natural_control": {
            "role": "required_positive_control_not_primary_estimand",
            "status": "passed_for_all_eligible_contexts" if eligible else "no_eligible_contexts",
            "requirement": "donor-natural outcome and actions equal the hc donor-to-recipient transplant",
        },
        "source_rollouts_per_puzzle": int(patch_rollouts),
        "source_rollout_seed": int(patch_seed),
        "source_policy_sampling": {
            "distribution": "categorical_softmax_of_legally_masked_logits",
            "temperature": 1.0,
            "draw_device": "cpu",
        },
        "selection": {
            "eligibility": "same puzzle, completed move > 0, exact canonical positions, different prefixes",
            "rule": "earliest move, then lexicographic positions, then lowest rollout IDs",
            "outcome_fields_used": False,
            "contexts_per_puzzle": "at_most_one",
        },
        "eligibility": {
            "status": "evaluated",
            "suite_puzzles": int(len(puzzles)),
            "eligible_puzzles": int(eligible),
            "no_match_puzzles": int(len(puzzles) - eligible),
            "no_match_records": no_matches,
        },
        "source_rollouts": source_manifests,
        "patches": patch_results,
    }


def _localization_entry(intervention: Mapping[str, object]) -> Dict[str, object]:
    overall = intervention["effects"]["overall"]
    spec = intervention["spec"]
    return {
        "intervention_id": spec["intervention_id"],
        "family": spec["family"],
        "component": spec["component"],
        "depth": spec["depth"],
        "after_completed_moves": spec["after_completed_moves"],
        "n": overall["n"],
        "baseline_success_rate": overall["baseline_success_rate"],
        "intervention_success_rate": overall["intervention_success_rate"],
        "mean_paired_success_delta": overall["mean_paired_success_delta"],
        "improved": overall["improved"],
        "harmed": overall["harmed"],
        "tied": overall["tied"],
        "bootstrap_ci": overall["bootstrap_ci"],
        "application_counts": intervention["application_counts"],
    }


def build_localization_comparison(core: Mapping[str, object]) -> Dict[str, object]:
    """Expose the core localization contrasts on stable direct paths."""

    interventions = core["interventions"]
    return {
        "baseline_success_rate": core["baseline"]["success_rate"],
        "full_reset_every_move": _localization_entry(interventions["reset_all_every_move"]),
        "per_depth": {
            str(intervention["spec"]["depth"]): _localization_entry(intervention)
            for intervention in interventions.values()
            if intervention["spec"]["depth"] is not None
        },
        "components": {
            intervention["spec"]["component"]: _localization_entry(intervention)
            for intervention in interventions.values()
            if intervention["spec"]["family"] == "reset_every_move"
            and intervention["spec"]["depth"] is None
            and intervention["spec"]["component"] in {"h", "c"}
        },
        "timing": {
            str(intervention["spec"]["after_completed_moves"]): _localization_entry(intervention)
            for intervention in interventions.values()
            if intervention["spec"]["family"] == "reset_once"
        },
    }


def _git_provenance() -> Dict[str, object]:
    repository = Path(__file__).resolve().parents[2]

    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repository,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("drc.py"),
        Path(__file__).with_name("env.py"),
        Path(__file__).with_name("puzzles.py"),
        Path(__file__).with_name("drc_train.py"),
        Path(__file__).with_name("drc_carry.py"),
        Path(__file__).with_name("probe_drc_carry.py"),
        repository / "docs" / "DrcCarryInterventionStudy.md",
    )
    return {
        "head": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "implementation_file_sha256": {
            str(path.relative_to(repository)): _file_sha256(path) for path in implementation_files
        },
    }


def _runtime_provenance(
    *,
    requested_device: str,
    resolved_device: str,
    started_at: dt.datetime,
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "requested_device": str(requested_device),
        "resolved_model_device": str(resolved_device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def intervene_checkpoint(
    model_path: str | Path,
    bank_path: str | Path,
    *,
    split_name: str = "val",
    suite_seed: int = 9,
    per_stratum: int = 8,
    timing_moves: Sequence[int] = DEFAULT_TIMING_MOVES,
    rollout_seed: int = 0,
    bootstrap_seed: int = 42_424,
    bootstrap_reps: int = 10_000,
    patch_rollouts: int = 16,
    patch_seed: int = 73_737,
    device: str = "cpu",
) -> Dict[str, object]:
    """Load one checkpoint/bank and execute the preregistered causal study."""

    if split_name not in {"val", "train"}:
        raise ValueError("Carry interventions are restricted to validation or training data")
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    if patch_rollouts < 0 or patch_rollouts == 1:
        raise ValueError("patch_rollouts must be 0 (disabled) or at least 2")
    normalized_timing = _normalize_timing_moves(timing_moves)

    started_at = dt.datetime.now(dt.timezone.utc)
    started_clock = time.perf_counter()
    starting_git = _git_provenance()
    model_path = Path(model_path).resolve()
    bank_path = Path(bank_path).resolve()
    net, config, payload, checkpoint = load_schema2_checkpoint(model_path, device=device)
    resolved_device = str(next(net.parameters()).device)
    if not config.canonical_identity:
        raise ValueError("Checkpoint is not from the canonical-identity DRC protocol")
    checkpoint = {
        **checkpoint,
        "stored_config": dict(payload["config"]),
    }
    bank_sha256 = _file_sha256(bank_path)
    bank = {
        "absolute_path": str(bank_path),
        "sha256": bank_sha256,
        "source_split": split_name,
        "split_contract": {
            "eval_fraction": float(config.eval_fraction),
            "split_seed": int(config.split_seed),
            "val_test_seed": int(config.val_test_seed),
        },
    }
    pool = load_puzzles(
        bank_path,
        split_name,
        eval_fraction=config.eval_fraction,
        split_seed=config.split_seed,
        val_test_seed=config.val_test_seed,
    )
    if _file_sha256(bank_path) != bank_sha256:
        raise RuntimeError(f"Puzzle bank changed while it was being loaded: {bank_path}")
    puzzles = joint_stratified_subset(pool, per_stratum=per_stratum, seed=suite_seed)
    if not puzzles:
        raise ValueError("Joint-stratified suite selection returned no puzzles")
    suite = suite_provenance(puzzles, split=split_name, seed=suite_seed, per_stratum=per_stratum)
    env_config = env_config_from_checkpoint(config)

    core = evaluate_core_interventions(
        net,
        puzzles,
        env_config,
        timing_moves=normalized_timing,
        rollout_seed=rollout_seed,
        bootstrap_seed=bootstrap_seed,
        bootstrap_reps=bootstrap_reps,
    )
    patches = evaluate_matched_patches(
        net,
        puzzles,
        env_config,
        patch_rollouts=patch_rollouts,
        patch_seed=patch_seed,
        bootstrap_seed=bootstrap_seed,
        bootstrap_reps=bootstrap_reps,
    )
    if _file_sha256(model_path) != checkpoint["sha256"]:
        raise RuntimeError(f"Checkpoint changed during intervention study: {model_path}")
    if _file_sha256(bank_path) != bank_sha256:
        raise RuntimeError(f"Puzzle bank changed during intervention study: {bank_path}")
    ending_git = _git_provenance()
    if ending_git["implementation_file_sha256"] != starting_git["implementation_file_sha256"]:
        raise RuntimeError("Intervention implementation files changed during the study")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_version": INTERVENTION_PROTOCOL_VERSION,
        "study_type": "causal_drc_carry_state_interventions",
        "causal_interventions": True,
        "status": "protocol_execution_result",
        "checkpoint": checkpoint,
        "bank": bank,
        "suite": suite,
        "settings": {
            "split": split_name,
            "suite_seed": int(suite_seed),
            "per_stratum": int(per_stratum),
            "core_rollout_seed": int(rollout_seed),
            "derived_seed_protocol": "sha256-colon-delimited-first-8-bytes-mod-2^31-v1",
            "timing_completed_moves": list(normalized_timing),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_unit": "puzzle",
            "bootstrap_confidence_level": 0.95,
            "patch_rollouts_per_puzzle": int(patch_rollouts),
            "patch_seed": int(patch_seed),
            "patch_components": list(PATCH_COMPONENTS),
            "requested_device": str(device),
            "resolved_model_device": resolved_device,
            "environment": dataclasses.asdict(env_config),
        },
        "core": core,
        "localization": build_localization_comparison(core),
        "patching": patches,
        "git": ending_git,
        "runtime": _runtime_provenance(
            requested_device=device,
            resolved_device=resolved_device,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started_clock,
        ),
    }
    # Fail here rather than writing a partial or non-standard JSON document.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def write_json_atomic(path: str | Path, value: Mapping[str, object], *, force: bool = False) -> None:
    """Durably publish strict JSON, refusing an existing destination by default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary_name, destination)
        else:
            try:
                os.link(temporary_name, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing output {destination}; pass --force to replace it"
                ) from exc
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal recurrent-carry interventions for a frozen DRC policy")
    parser.add_argument("--model", dest="model_path", required=True)
    parser.add_argument("--bank", dest="bank_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", dest="split_name", choices=("val", "train"), default="val")
    parser.add_argument("--suite-seed", type=int, default=9)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--timing-moves", type=int, nargs="+", default=list(DEFAULT_TIMING_MOVES))
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42_424)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--patch-rollouts", type=int, default=16)
    parser.add_argument("--patch-seed", type=int, default=73_737)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    output = Path(args.output).resolve()
    protected_inputs = {Path(args.model_path).resolve(), Path(args.bank_path).resolve()}
    if output in protected_inputs:
        parser.error("--output must differ from --model and --bank")
    if output.exists() and not args.force:
        parser.error(f"refusing to overwrite existing --output {output}; pass --force to replace it")
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    if any(move < 1 for move in args.timing_moves):
        parser.error("--timing-moves values must be positive")
    if args.patch_rollouts < 0 or args.patch_rollouts == 1:
        parser.error("--patch-rollouts must be 0 (disabled) or at least 2")

    arguments = vars(args).copy()
    arguments.pop("output")
    force = bool(arguments.pop("force"))
    result = intervene_checkpoint(**arguments)
    write_json_atomic(output, result, force=force)
    print(
        json.dumps(
            {
                "output": str(output),
                "suite_puzzles": result["suite"]["count"],
                "baseline_success_rate": result["core"]["baseline"]["success_rate"],
                "patch_status": result["patching"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_TIMING_MOVES",
    "INTERVENTION_PROTOCOL_VERSION",
    "InterventionSpec",
    "PATCH_PROTOCOL_VERSION",
    "PatchContext",
    "PatchPair",
    "aggregate_paired_effects",
    "build_localization_comparison",
    "clone_carry_state",
    "evaluate_core_interventions",
    "evaluate_matched_patches",
    "intervene_checkpoint",
    "intervention_specs",
    "patch_carry_state",
    "run_deterministic_episode",
    "select_matched_patch_pair",
    "write_json_atomic",
    "zero_carry_state",
]
