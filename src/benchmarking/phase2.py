"""Phase 2 feed-forward benchmark with immutable held-out suites."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import copy
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np

from src.env.curriculum import BankCurriculumWrapper, create_bank_curriculum_manager
from src.env.criteria_env import PuzzleCriteria
from src.env.environment_spec import EnvironmentSpec
from src.env.puzzle_bank import (
    BankSampler,
    PuzzleBank,
    PuzzleMetadata,
    SolverKey,
    SpecKey,
    compute_board_hash,
    compute_layout_hash,
    count_robots_moved,
    create_fixed_layout_from_seed,
    serialize_layout_to_blob,
)
from src.env.ricochet_env import RicochetRobotsEnv, generate_board_from_spec_with_seed
from src.env.solver import solve_bfs_with_metadata


@dataclass(frozen=True)
class Phase2Config:
    """Small, reproducible benchmark intended for a local Apple-Silicon run."""

    height: int = 8
    width: int = 8
    num_robots: int = 2
    edge_t_per_quadrant: int = 0
    central_l_per_quadrant: int = 0
    fixture_attempts: int = 4_000
    solver_max_depth: int = 6
    solver_max_nodes: int = 50_000
    min_optimal_length: int = 1
    max_optimal_length: int = 2
    min_robots_moved: int = 1
    max_robots_moved: int = 1
    heldout_fraction: float = 0.25
    split_seed: int = 20_260_710
    validation_test_seed: int = 20_260_711
    suite_per_stratum: int = 16
    max_steps: int = 10
    cell_obs_pixel_size: int = 40
    obs_mode: str = "rgb_cell_image"
    policy: str = "stock_cnn"
    features_dim: int = 256
    conv_channels: int = 64
    n_conv_layers: int = 4
    policy_activation: str | None = None
    ortho_init: bool | None = None
    bc_reinitialize_action_head: bool = True
    bc_action_head_gain: float = 1.0
    bc_apply_action_masks: bool = False
    two_robot_band_weight: float | None = None
    run_overfit_milestone: bool = True
    curriculum_stage_max_lengths: tuple[int, ...] | None = None
    curriculum_start_distances: tuple[int, ...] | None = None
    reverse_final_full_starts: bool = False
    curriculum_frontier_weight: float = 3.0
    curriculum_frontier_window: int = 200
    curriculum_frontier_thresholds: Mapping[int, float] | None = None
    curriculum_success_threshold: float = 0.7
    curriculum_min_episodes: int = 5_000
    curriculum_window: int = 500
    curriculum_check_frequency: int = 100
    curriculum_max_episodes_per_stage: int | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    bc_pretrain_epochs: int = 0
    bc_batch_size: int = 1024
    bc_learning_rate: float = 3e-4
    bc_max_puzzles: int | None = None
    bc_select_best_by_validation: bool = False
    initial_checkpoint_path: str | None = None
    use_action_masking: bool = False
    evaluate_both_mask_modes: bool = False
    training_seeds: tuple[int, ...] = (11, 29, 47)
    checkpoints: tuple[int, ...] = (0, 10_240, 25_600, 51_200)
    n_envs: int = 8
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    learning_rate_final: float | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    ent_coef_final: float | None = None
    target_kl: float | None = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    validation_episodes_per_puzzle: int = 1
    evaluation_batch_size: int = 64
    validation_bootstrap_samples: int = 0
    fixed_validation_seeds: bool = False
    final_stochastic_episodes_per_puzzle: int = 5
    baseline_episodes_per_puzzle: int = 20
    overfit_puzzles_per_stratum: int = 4
    overfit_stochastic_episodes_per_puzzle: int = 20
    save_checkpoint_weights: bool = False
    select_checkpoint_by_validation: bool = False
    run_final_test: bool = True
    health_probe_puzzles: int = 64
    health_log_every_rollouts: int = 5
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.height != self.width:
            raise ValueError("Phase 2 fixture boards must be square")
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must be in (0, 1)")
        if not self.training_seeds:
            raise ValueError("At least one training seed is required")
        if not self.checkpoints or self.checkpoints[0] != 0:
            raise ValueError("checkpoints must start at zero")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be unique and increasing")
        if self.max_optimal_length > self.max_steps:
            raise ValueError("max_steps must accommodate every benchmark solution")
        if self.overfit_puzzles_per_stratum <= 0:
            raise ValueError("overfit_puzzles_per_stratum must be positive")
        if self.overfit_stochastic_episodes_per_puzzle <= 0:
            raise ValueError("overfit_stochastic_episodes_per_puzzle must be positive")
        if self.policy not in {"stock_cnn", "grid_cnn", "grid_resnet"}:
            raise ValueError(f"Unsupported policy: {self.policy}")
        if self.policy == "stock_cnn" and self.obs_mode not in {"rgb_image", "rgb_cell_image"}:
            raise ValueError("stock_cnn requires an RGB observation mode")
        if self.policy in {"grid_cnn", "grid_resnet"} and self.obs_mode != "image":
            raise ValueError(f"{self.policy} requires the structured-plane 'image' observation mode")
        if self.bc_pretrain_epochs < 0:
            raise ValueError("bc_pretrain_epochs must be non-negative")
        if self.bc_pretrain_epochs > 0 and self.obs_mode != "image":
            raise ValueError("BC pretraining requires the structured-plane 'image' observation mode")
        if self.initial_checkpoint_path is not None and self.bc_pretrain_epochs > 0:
            raise ValueError("initial_checkpoint_path and BC pretraining are mutually exclusive")
        if self.policy_activation not in {None, "relu", "tanh"}:
            raise ValueError("policy_activation must be one of: relu, tanh")
        if self.bc_action_head_gain <= 0:
            raise ValueError("bc_action_head_gain must be positive")
        if self.bc_apply_action_masks and not self.use_action_masking:
            raise ValueError("bc_apply_action_masks requires use_action_masking")
        if self.validation_bootstrap_samples < 0:
            raise ValueError("validation_bootstrap_samples must be non-negative")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")
        if self.select_checkpoint_by_validation and not self.save_checkpoint_weights:
            raise ValueError("select_checkpoint_by_validation requires save_checkpoint_weights")
        if self.health_probe_puzzles <= 0:
            raise ValueError("health_probe_puzzles must be positive")
        if self.health_log_every_rollouts <= 0:
            raise ValueError("health_log_every_rollouts must be positive")
        if self.learning_rate_final is not None and self.learning_rate_final <= 0:
            raise ValueError("learning_rate_final must be positive")
        if self.ent_coef_final is not None and self.ent_coef_final < 0:
            raise ValueError("ent_coef_final must be non-negative")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive")
        if self.evaluate_both_mask_modes and not self.use_action_masking:
            raise ValueError("evaluate_both_mask_modes requires use_action_masking")
        if self.curriculum_stage_max_lengths is not None:
            stages = self.curriculum_stage_max_lengths
            if tuple(sorted(set(stages))) != stages:
                raise ValueError("curriculum_stage_max_lengths must be strictly increasing")
            if stages[-1] != self.max_optimal_length:
                raise ValueError("The final curriculum stage must cover max_optimal_length")
            if stages[0] < self.min_optimal_length:
                raise ValueError("Curriculum stages cannot start below min_optimal_length")
        if self.curriculum_stage_max_lengths is not None and self.curriculum_start_distances is not None:
            raise ValueError("Length and reverse-distance curricula are mutually exclusive")
        if self.curriculum_start_distances is not None:
            distances = self.curriculum_start_distances
            if tuple(sorted(set(distances))) != distances:
                raise ValueError("curriculum_start_distances must be strictly increasing")
            if not distances or distances[0] < 1 or distances[-1] > self.max_optimal_length:
                raise ValueError("curriculum_start_distances must be within the fixture length range")
            thresholds = self.curriculum_frontier_thresholds or {}
            missing = set(distances) - set(thresholds)
            if missing:
                raise ValueError(f"Missing calibrated frontier thresholds for distances: {sorted(missing)}")
            if any(not 0.0 < float(value) <= 1.0 for value in thresholds.values()):
                raise ValueError("Frontier thresholds must be in (0, 1]")
        if self.curriculum_frontier_weight <= 0:
            raise ValueError("curriculum_frontier_weight must be positive")
        if self.curriculum_frontier_window <= 0:
            raise ValueError("curriculum_frontier_window must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Phase2Config":
        normalized = dict(values)
        for key in (
            "training_seeds",
            "checkpoints",
            "curriculum_stage_max_lengths",
            "curriculum_start_distances",
        ):
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(int(value) for value in normalized[key])
        if normalized.get("curriculum_frontier_thresholds") is not None:
            normalized["curriculum_frontier_thresholds"] = {
                int(key): float(value)
                for key, value in normalized["curriculum_frontier_thresholds"].items()
            }
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown Phase 2 configuration keys: {', '.join(sorted(unknown))}")
        return cls(**normalized)

    @property
    def spec_key(self) -> SpecKey:
        return SpecKey(
            self.height,
            self.width,
            self.num_robots,
            self.edge_t_per_quadrant,
            self.central_l_per_quadrant,
        )

    def environment_spec(self, seed: int) -> EnvironmentSpec:
        return EnvironmentSpec(
            height=self.height,
            width=self.width,
            num_robots=self.num_robots,
            include_noop=False,
            step_penalty=-0.01,
            goal_reward=1.0,
            noop_penalty=-0.01,
            max_steps=self.max_steps,
            seed=seed,
            ensure_solvable=False,
            solver_max_depth=self.solver_max_depth,
            solver_max_nodes=self.solver_max_nodes,
            obs_mode=self.obs_mode,
            channels_first=True,
            cell_obs_pixel_size=self.cell_obs_pixel_size,
        )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _puzzle_id(puzzle: PuzzleMetadata) -> str:
    return puzzle.board_hash.hex()


def _layout_id(puzzle: PuzzleMetadata) -> str:
    if puzzle.layout_hash is None:
        raise ValueError("Phase 2 puzzles require layout_hash")
    return puzzle.layout_hash.hex()


def _stratum(puzzle: PuzzleMetadata) -> str:
    return f"optimal_length={int(puzzle.optimal_length)}|robots_moved={int(puzzle.robots_moved)}"


def _level(
    config: Phase2Config,
    *,
    index: int = 0,
    name: str = "phase2-fixed-easy",
    max_optimal_length: int | None = None,
) -> Dict[str, Any]:
    top_length = config.max_optimal_length if max_optimal_length is None else max_optimal_length
    level: Dict[str, Any] = {
        "level": index,
        "name": name,
        "spec_key": config.spec_key,
        "min_optimal_length": config.min_optimal_length,
        "max_optimal_length": top_length,
        "min_robots_moved": config.min_robots_moved,
        "max_robots_moved": config.max_robots_moved,
    }
    if config.two_robot_band_weight is not None:
        # One band per achievable (optimal_length, robots_moved) pair, with
        # multi-robot plans up-weighted so they are not drowned out by the
        # fixture's single-robot majority.
        level["bands"] = [
            {
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": robots,
                "max_robots_moved": robots,
                "weight": config.two_robot_band_weight if robots >= 2 else 1.0,
            }
            for length in range(config.min_optimal_length, top_length + 1)
            for robots in range(config.min_robots_moved, config.max_robots_moved + 1)
            if robots <= length
        ]
    return level


def _curriculum_levels(config: Phase2Config) -> List[Dict[str, Any]]:
    if config.curriculum_start_distances is not None:
        return _reverse_curriculum_levels(config)
    if config.curriculum_stage_max_lengths is None:
        return [_level(config)]
    return [
        _level(
            config,
            index=index,
            name=f"phase-stage-{index}-len{config.min_optimal_length}-{stage_length}",
            max_optimal_length=stage_length,
        )
        for index, stage_length in enumerate(config.curriculum_stage_max_lengths)
    ]


def _reverse_curriculum_levels(config: Phase2Config) -> List[Dict[str, Any]]:
    distances = config.curriculum_start_distances
    if distances is None:
        raise ValueError("Reverse curriculum requires curriculum_start_distances")
    thresholds = config.curriculum_frontier_thresholds or {}
    levels: List[Dict[str, Any]] = []
    for index, frontier_distance in enumerate(distances):
        active_distances = distances[: index + 1]
        levels.append({
            "level": index,
            "name": f"reverse-stage-{index}-frontier-d{frontier_distance}",
            "spec_key": config.spec_key,
            "min_optimal_length": min(active_distances),
            "max_optimal_length": config.max_optimal_length,
            "min_robots_moved": config.min_robots_moved,
            "max_robots_moved": config.max_robots_moved,
            "bands": [
                {
                    "min_optimal_length": distance,
                    "max_optimal_length": config.max_optimal_length,
                    "min_robots_moved": config.min_robots_moved,
                    "max_robots_moved": config.max_robots_moved,
                    "start_distance": distance,
                    "band_key": f"d{distance}",
                    "weight": config.curriculum_frontier_weight if distance == frontier_distance else 1.0,
                }
                for distance in active_distances
            ],
            "frontier_band": f"d{frontier_distance}",
            "frontier_threshold": float(thresholds[frontier_distance]),
            "frontier_window": config.curriculum_frontier_window,
        })
    if config.reverse_final_full_starts:
        final = _level(
            config,
            index=len(levels),
            name="reverse-final-original-starts",
        )
        final_bands = final.get("bands")
        if final_bands is None:
            final_bands = [
                {
                    "min_optimal_length": length,
                    "max_optimal_length": length,
                    "min_robots_moved": config.min_robots_moved,
                    "max_robots_moved": config.max_robots_moved,
                    "weight": 1.0,
                }
                for length in range(config.min_optimal_length, config.max_optimal_length + 1)
            ]
        for band in final_bands:
            band["band_key"] = (
                f"L{band['min_optimal_length']}"
                f"R{band['min_robots_moved']}-{band['max_robots_moved']}"
            )
        final["bands"] = final_bands
        levels.append(final)
    return levels


def _fixture_attempt(args: tuple[int, "Phase2Config"]) -> "PuzzleMetadata | None":
    """Generate and label one fixture candidate. Deterministic given the seed."""
    seed, config = args
    board = generate_board_from_spec_with_seed(
        height=config.height,
        width=config.width,
        num_robots=config.num_robots,
        edge_t_per_quadrant=config.edge_t_per_quadrant,
        central_l_per_quadrant=config.central_l_per_quadrant,
        seed=seed,
    )
    solve_started = time.perf_counter()
    result = solve_bfs_with_metadata(
        board,
        max_depth=config.solver_max_depth,
        max_nodes=config.solver_max_nodes,
    )
    solve_ms = int((time.perf_counter() - solve_started) * 1_000)
    if result is None or not result.found or int(result.depth) == 0:
        return None
    actions = [int(robot_id) * 4 + int(direction) for robot_id, direction in result.actions]
    robots_moved = count_robots_moved(actions)
    if not config.min_optimal_length <= int(result.depth) <= config.max_optimal_length:
        return None
    if not config.min_robots_moved <= robots_moved <= config.max_robots_moved:
        return None
    solver_key = SolverKey(
        algorithm="BFS",
        max_depth=config.solver_max_depth,
        max_nodes=config.solver_max_nodes,
    )
    return PuzzleMetadata(
        seed=seed,
        spec_key=config.spec_key,
        solver_key=solver_key,
        board_hash=compute_board_hash(board),
        optimal_length=int(result.depth),
        robots_moved=robots_moved,
        nodes_expanded=int(result.nodes_expanded),
        solve_time_ms=solve_ms,
        solved_within_limits=True,
        hit_depth_limit=int(result.depth) >= config.solver_max_depth,
        hit_node_limit=int(result.nodes_expanded) >= config.solver_max_nodes,
        actions_encoded=actions,
        created_at=0.0,
        run_id="phase2-deterministic-fixture",
        board_seed=seed,
        layout_hash=compute_layout_hash(board),
        layout_blob=serialize_layout_to_blob(board),
    )


def build_fixture_bank(bank_dir: Path, config: Phase2Config, *, workers: int = 1) -> Dict[str, Any]:
    """Build or validate the deterministic solver-labelled fixture bank.

    ``workers`` parallelizes generation/labelling only; candidates are
    assembled and deduplicated in seed order, so the stored fixture and its
    fingerprint are identical for any worker count.
    """
    fixture_path = bank_dir / "phase2_fixture.json"
    expected_definition = {
        "schema_version": 1,
        "spec_key": asdict(config.spec_key),
        "attempted_seeds": [0, config.fixture_attempts],
        "solver": {
            "algorithm": "BFS",
            "max_depth": config.solver_max_depth,
            "max_nodes": config.solver_max_nodes,
        },
        "criteria": {
            "min_optimal_length": config.min_optimal_length,
            "max_optimal_length": config.max_optimal_length,
            "min_robots_moved": config.min_robots_moved,
            "max_robots_moved": config.max_robots_moved,
        },
    }
    definition_sha256 = _sha256_json(expected_definition)
    if fixture_path.exists():
        existing = json.loads(fixture_path.read_text(encoding="utf-8"))
        if existing.get("definition_sha256") != definition_sha256:
            raise ValueError(f"Existing fixture at {bank_dir} was built from a different definition")
        return existing

    bank = PuzzleBank(bank_dir, run_id="phase2-deterministic-fixture")
    if bank.get_puzzle_count() != 0:
        raise ValueError(f"Refusing to populate non-empty bank without {fixture_path.name}: {bank_dir}")

    puzzles: List[PuzzleMetadata] = []
    seen_boards: set[bytes] = set()
    started = time.perf_counter()
    attempts = [(seed, config) for seed in range(config.fixture_attempts)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            candidates: Iterator[PuzzleMetadata | None] = pool.map(
                _fixture_attempt, attempts, chunksize=64
            )
            candidate_list = list(candidates)
    else:
        candidate_list = [_fixture_attempt(attempt) for attempt in attempts]
    for candidate in candidate_list:  # seed order: dedup stays deterministic
        if candidate is None:
            continue
        if candidate.board_hash in seen_boards:
            continue
        seen_boards.add(candidate.board_hash)
        puzzles.append(candidate)

    if not puzzles:
        raise RuntimeError("The deterministic Phase 2 fixture produced no matching puzzles")
    bank.add_puzzles(puzzles)
    histogram = Counter(_stratum(puzzle) for puzzle in puzzles)
    manifest = {
        **expected_definition,
        "definition_sha256": definition_sha256,
        "fixture_sha256": _sha256_json(sorted(_puzzle_id(puzzle) for puzzle in puzzles)),
        "puzzle_count": len(puzzles),
        "histogram": dict(sorted(histogram.items())),
        "build_seconds": time.perf_counter() - started,
    }
    _atomic_json(fixture_path, manifest)
    return manifest


def build_split_manifest(bank: PuzzleBank, output_path: Path, config: Phase2Config) -> Dict[str, Any]:
    """Freeze train/validation/test membership and balanced fixed suites."""
    train_sampler = BankSampler(
        bank,
        seed=0,
        split="train",
        eval_fraction=config.heldout_fraction,
        split_seed=config.split_seed,
        split_key="layout_hash",
    )
    heldout_sampler = BankSampler(
        bank,
        seed=0,
        split="eval",
        eval_fraction=config.heldout_fraction,
        split_seed=config.split_seed,
        split_key="layout_hash",
    )
    filters = {
        "min_optimal_length": config.min_optimal_length,
        "max_optimal_length": config.max_optimal_length,
        "min_robots_moved": config.min_robots_moved,
        "max_robots_moved": config.max_robots_moved,
    }
    training = train_sampler.list_puzzles(config.spec_key, **filters)
    heldout = heldout_sampler.list_puzzles(config.spec_key, **filters)

    validation: List[PuzzleMetadata] = []
    test: List[PuzzleMetadata] = []
    by_stratum: Dict[str, List[PuzzleMetadata]] = defaultdict(list)
    for puzzle in heldout:
        by_stratum[_stratum(puzzle)].append(puzzle)
    for stratum, puzzles in sorted(by_stratum.items()):
        ordered = sorted(
            puzzles,
            key=lambda puzzle: hashlib.sha256(
                f"{config.validation_test_seed}:{stratum}:{_layout_id(puzzle)}".encode("utf-8")
            ).digest(),
        )
        validation.extend(ordered[::2])
        test.extend(ordered[1::2])

    def select_suite(puzzles: Sequence[PuzzleMetadata]) -> List[PuzzleMetadata]:
        grouped: Dict[str, List[PuzzleMetadata]] = defaultdict(list)
        for puzzle in puzzles:
            grouped[_stratum(puzzle)].append(puzzle)
        selected: List[PuzzleMetadata] = []
        for stratum in sorted(grouped):
            selected.extend(sorted(grouped[stratum], key=_puzzle_id)[: config.suite_per_stratum])
        return selected

    training_suite = select_suite(training)
    validation_suite = select_suite(validation)
    test_suite = select_suite(test)
    # Require the strata the generator actually produced rather than the
    # Cartesian product of configured ranges: some combinations inside the
    # ranges are impossible (e.g. optimal_length=1 with robots_moved=2), so
    # the product would reject valid harder fixtures. Strata with plenty of
    # fixture support must appear in every split; genuinely sparse strata are
    # reported and tolerated.
    fixture_histogram = Counter(_stratum(puzzle) for puzzle in [*training, *heldout])
    membership_strata = (
        {_stratum(puzzle) for puzzle in training}
        & {_stratum(puzzle) for puzzle in validation}
        & {_stratum(puzzle) for puzzle in test}
    )
    if not membership_strata:
        raise RuntimeError("No stratum appears in all of train/validation/test")
    sparse_threshold = 20
    dropped = sorted(set(fixture_histogram) - membership_strata)
    for stratum in dropped:
        if fixture_histogram[stratum] >= sparse_threshold:
            raise RuntimeError(
                f"Stratum {stratum} has {fixture_histogram[stratum]} fixture puzzles "
                "but is missing from a split"
            )
    if dropped:
        print(f"[phase2] warning: sparse strata not present in every split: {dropped}", flush=True)
    expected_strata = membership_strata
    for name, suite in (("training", training_suite), ("validation", validation_suite), ("test", test_suite)):
        present = {_stratum(puzzle) for puzzle in suite}
        missing = expected_strata - present
        if missing:
            raise RuntimeError(f"{name} suite is missing strata: {sorted(missing)}")

    memberships = {
        "train": sorted(_puzzle_id(puzzle) for puzzle in training),
        "validation": sorted(_puzzle_id(puzzle) for puzzle in validation),
        "test": sorted(_puzzle_id(puzzle) for puzzle in test),
    }
    membership_sets = {name: set(values) for name, values in memberships.items()}
    if membership_sets["train"] & membership_sets["validation"]:
        raise AssertionError("train and validation membership overlap")
    if membership_sets["train"] & membership_sets["test"]:
        raise AssertionError("train and test membership overlap")
    if membership_sets["validation"] & membership_sets["test"]:
        raise AssertionError("validation and test membership overlap")

    manifest = {
        "schema_version": 1,
        "split_key": "layout_hash",
        "heldout_fraction": config.heldout_fraction,
        "split_seed": config.split_seed,
        "validation_test_seed": config.validation_test_seed,
        "memberships": memberships,
        "training_suite": [_puzzle_id(puzzle) for puzzle in training_suite],
        "validation_suite": [_puzzle_id(puzzle) for puzzle in validation_suite],
        "test_suite": [_puzzle_id(puzzle) for puzzle in test_suite],
        "counts": {name: len(values) for name, values in memberships.items()},
        "suite_histograms": {
            "training": dict(sorted(Counter(_stratum(puzzle) for puzzle in training_suite).items())),
            "validation": dict(sorted(Counter(_stratum(puzzle) for puzzle in validation_suite).items())),
            "test": dict(sorted(Counter(_stratum(puzzle) for puzzle in test_suite).items())),
        },
    }
    manifest["fingerprint"] = _sha256_json(manifest)
    _atomic_json(output_path, manifest)
    return manifest


def build_confirmation_manifest(
    bank: PuzzleBank,
    parent_manifest: Mapping[str, Any],
    output_path: Path,
    config: Phase2Config,
    *,
    selection_seed: int = 20_260_717,
    max_per_stratum: int = 20,
) -> Dict[str, Any]:
    """Seal unused test layouts for one final model evaluation."""
    if max_per_stratum <= 0:
        raise ValueError("max_per_stratum must be positive")
    lookup = _lookup_puzzles(bank, config)
    exposed_ids = set(str(value) for value in parent_manifest["test_suite"])
    exposed_layouts = {_layout_id(lookup[puzzle_id]) for puzzle_id in exposed_ids}
    candidate_ids = [
        str(puzzle_id)
        for puzzle_id in parent_manifest["memberships"]["test"]
        if str(puzzle_id) not in exposed_ids and _layout_id(lookup[str(puzzle_id)]) not in exposed_layouts
    ]
    groups: Dict[str, List[PuzzleMetadata]] = defaultdict(list)
    for puzzle_id in candidate_ids:
        groups[_stratum(lookup[puzzle_id])].append(lookup[puzzle_id])
    rng = np.random.default_rng(selection_seed)
    selected: List[PuzzleMetadata] = []
    selected_layouts: set[str] = set()
    for stratum in sorted(groups):
        group = sorted(groups[stratum], key=_puzzle_id)
        order = rng.permutation(len(group))
        chosen = 0
        for index in order:
            puzzle = group[int(index)]
            layout_id = _layout_id(puzzle)
            if layout_id in selected_layouts:
                continue
            selected.append(puzzle)
            selected_layouts.add(layout_id)
            chosen += 1
            if chosen >= max_per_stratum:
                break
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "parent_split_fingerprint": parent_manifest["fingerprint"],
        "selection_seed": selection_seed,
        "max_per_stratum": max_per_stratum,
        "excluded_puzzle_ids": sorted(exposed_ids),
        "excluded_layout_ids": sorted(exposed_layouts),
        "selected_puzzle_ids": [_puzzle_id(puzzle) for puzzle in selected],
        "selected_layout_ids": [_layout_id(puzzle) for puzzle in selected],
        "histogram": dict(sorted(Counter(_stratum(puzzle) for puzzle in selected).items())),
    }
    manifest["fingerprint"] = _sha256_json(manifest)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Existing confirmation manifest differs: {output_path}")
        return existing
    _atomic_json(output_path, manifest)
    return manifest


def _lookup_puzzles(bank: PuzzleBank, config: Phase2Config) -> Dict[str, PuzzleMetadata]:
    sampler = BankSampler(bank, split="all")
    return {_puzzle_id(puzzle): puzzle for puzzle in sampler.list_puzzles(config.spec_key)}


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _episode(
    controller: str,
    puzzle: PuzzleMetadata,
    environment_spec: EnvironmentSpec,
    *,
    rng: np.random.Generator,
    model: Any = None,
    deterministic: bool = True,
    action_masking: bool = False,
) -> Dict[str, Any]:
    layout = create_fixed_layout_from_seed(puzzle)
    env = RicochetRobotsEnv(**environment_spec.to_env_kwargs(fixed_layout=layout))
    obs, _ = env.reset()
    terminated = False
    truncated = False
    episode_return = 0.0
    actions: List[int] = []
    no_movement = 0
    invalid_actions = 0
    step_index = 0
    while not (terminated or truncated):
        if controller == "model":
            predict_kwargs: Dict[str, Any] = {"deterministic": deterministic}
            if action_masking:
                predict_kwargs["action_masks"] = env.action_masks()
            action_value, _ = model.predict(obs, **predict_kwargs)
            action = int(np.asarray(action_value).reshape(-1)[0])
        elif controller == "random":
            action = int(rng.integers(0, env.action_space.n))
        elif controller == "target_robot_only":
            action = int(layout.target_robot) * 4 + int(rng.integers(0, 4))
        elif controller == "oracle":
            if step_index >= len(puzzle.actions_encoded):
                break
            action = int(puzzle.actions_encoded[step_index])
        else:
            raise ValueError(f"Unknown controller: {controller}")

        before = env.get_board().robot_positions
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
        except ValueError:
            invalid_actions += 1
            break
        after = env.get_board().robot_positions
        if before == after:
            no_movement += 1
        episode_return += float(reward)
        actions.append(action)
        step_index += 1

    success = bool(terminated and not truncated)
    record = {
        "controller": controller,
        "puzzle_id": _puzzle_id(puzzle),
        "layout_id": _layout_id(puzzle),
        "optimal_length": int(puzzle.optimal_length),
        "robots_moved": int(puzzle.robots_moved),
        "success": success,
        "episode_length": len(actions),
        "return": episode_return,
        "optimality_gap": len(actions) - int(puzzle.optimal_length) if success else None,
        "no_movement_actions": no_movement,
        "invalid_actions": invalid_actions,
        "actions": actions,
    }
    env.close()
    return record


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    successes = sum(int(bool(record["success"])) for record in records)
    total_steps = sum(int(record["episode_length"]) for record in records)
    no_movement = sum(int(record["no_movement_actions"]) for record in records)
    invalid = sum(int(record["invalid_actions"]) for record in records)
    successful_gaps = [int(record["optimality_gap"]) for record in records if record["optimality_gap"] is not None]
    action_counts: Counter[int] = Counter()
    for record in records:
        action_counts.update(int(action) for action in record["actions"])
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_wilson_95": _wilson_interval(successes, total),
        "mean_episode_length": (
            sum(int(record["episode_length"]) for record in records) / total if total else 0.0
        ),
        "mean_return": sum(float(record["return"]) for record in records) / total if total else 0.0,
        "mean_successful_optimality_gap": (
            sum(successful_gaps) / len(successful_gaps) if successful_gaps else None
        ),
        "total_steps": total_steps,
        "no_movement_rate": no_movement / total_steps if total_steps else 0.0,
        "invalid_action_rate": invalid / max(1, total_steps + invalid),
        "action_distribution": {
            str(action): count / total_steps for action, count in sorted(action_counts.items())
        } if total_steps else {},
    }


def _joint_table(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Success over the 2-D (optimal_length, robots_moved) difficulty grid.

    Difficulty is genuinely two-dimensional and its marginals cannot be
    recombined into the joint, so this contingency table is persisted per
    evaluation. Keys are ``"length,robots"``; absent cells are impossible
    puzzles (e.g. robots_moved > optimal_length).
    """
    cells: Dict[str, List[bool]] = defaultdict(list)
    for record in records:
        key = f"{int(record['optimal_length'])},{int(record['robots_moved'])}"
        cells[key].append(bool(record["success"]))
    return {
        key: {"rate": float(np.mean(values)), "n": len(values), "successes": int(sum(values))}
        for key, values in sorted(cells.items())
    }


def _puzzle_clustered_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> List[float] | None:
    """Bootstrap puzzles, preserving repeated episodes within each puzzle."""
    if samples <= 0 or not records:
        return None
    by_puzzle: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        by_puzzle[str(record["puzzle_id"])].append(float(bool(record["success"])))
    puzzle_rates = np.asarray([np.mean(values) for values in by_puzzle.values()], dtype=np.float64)
    if not len(puzzle_rates):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(puzzle_rates), size=(samples, len(puzzle_rates)))
    rates = puzzle_rates[draws].mean(axis=1)
    low, high = np.quantile(rates, [0.025, 0.975])
    return [float(low), float(high)]


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    summary = _aggregate_records(records)
    summary["elapsed_seconds"] = elapsed_seconds
    summary["environment_steps_per_second"] = summary["total_steps"] / elapsed_seconds if elapsed_seconds else 0.0
    for field in ("optimal_length", "robots_moved"):
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record[field])].append(record)
        summary[f"by_{field}"] = {
            key: _aggregate_records(group) for key, group in sorted(grouped.items(), key=lambda item: int(item[0]))
        }
    # Joint difficulty grid — enables across-training (length × robots) heatmaps.
    summary["joint"] = _joint_table(records)
    clustered = _puzzle_clustered_bootstrap(
        records,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if clustered is not None:
        summary["success_rate_puzzle_bootstrap_95"] = clustered
    return summary


@dataclass
class _BatchedModelEpisode:
    """Mutable state for one independently seeded model-evaluation episode."""

    index: int
    puzzle: PuzzleMetadata
    env: RicochetRobotsEnv
    observation: np.ndarray
    rng: np.random.Generator
    terminated: bool = False
    truncated: bool = False
    stopped: bool = False
    episode_return: float = 0.0
    actions: List[int] = field(default_factory=list)
    no_movement: int = 0
    invalid_actions: int = 0

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or self.stopped


def _model_episode_rng(seed: int, puzzle: PuzzleMetadata, repeat: int) -> np.random.Generator:
    """Give each episode a stable RNG stream independent of evaluation batching."""
    payload = f"{seed}:{_puzzle_id(puzzle)}:{repeat}".encode("utf-8")
    episode_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(episode_seed)


def _batched_model_actions(
    model: Any,
    observations: np.ndarray,
    action_masks: np.ndarray | None,
    *,
    deterministic: bool,
    episode_rngs: Sequence[np.random.Generator],
) -> np.ndarray:
    """Predict a discrete action batch without batch-coupled stochastic RNG."""
    if deterministic:
        predict_kwargs: Dict[str, Any] = {"deterministic": True}
        if action_masks is not None:
            predict_kwargs["action_masks"] = action_masks
        actions, _ = model.predict(observations, **predict_kwargs)
        return np.asarray(actions, dtype=np.int64).reshape(-1)

    import torch

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _ = policy.obs_to_tensor(observations)
    with torch.no_grad():
        distribution = policy.get_distribution(obs_tensor)
        if action_masks is not None:
            if not hasattr(distribution, "apply_masking"):
                raise TypeError("Action-masked evaluation requires a maskable policy distribution")
            distribution.apply_masking(action_masks)
        categorical = getattr(distribution, "distribution", None)
        probabilities_tensor = getattr(categorical, "probs", None)
        if probabilities_tensor is None:
            raise TypeError("Batched evaluation requires a discrete categorical policy")
        probabilities = probabilities_tensor.detach().cpu().numpy()

    actions: List[int] = []
    for probability, rng in zip(probabilities, episode_rngs, strict=True):
        normalized = np.asarray(probability, dtype=np.float64)
        total = float(normalized.sum())
        if not np.isfinite(normalized).all() or total <= 0.0:
            raise RuntimeError("Policy produced invalid action probabilities during evaluation")
        normalized /= total
        actions.append(int(rng.choice(len(normalized), p=normalized)))
    return np.asarray(actions, dtype=np.int64)


def _batched_episode_record(state: _BatchedModelEpisode) -> Dict[str, Any]:
    success = bool(state.terminated and not state.truncated)
    return {
        "controller": "model",
        "puzzle_id": _puzzle_id(state.puzzle),
        "layout_id": _layout_id(state.puzzle),
        "optimal_length": int(state.puzzle.optimal_length),
        "robots_moved": int(state.puzzle.robots_moved),
        "success": success,
        "episode_length": len(state.actions),
        "return": state.episode_return,
        "optimality_gap": len(state.actions) - int(state.puzzle.optimal_length) if success else None,
        "no_movement_actions": state.no_movement,
        "invalid_actions": state.invalid_actions,
        "actions": state.actions,
    }


def _evaluate_model_batched(
    puzzles: Sequence[PuzzleMetadata],
    environment_spec: EnvironmentSpec,
    *,
    repeats: int,
    seed: int,
    model: Any,
    deterministic: bool,
    action_masking: bool,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Advance independent fixed-puzzle episodes in batched policy calls."""
    jobs = [
        (puzzle, repeat)
        for puzzle in puzzles
        for repeat in range(repeats)
    ]
    records: List[Dict[str, Any] | None] = [None] * len(jobs)
    for chunk_start in range(0, len(jobs), batch_size):
        states: List[_BatchedModelEpisode] = []
        for offset, (puzzle, repeat) in enumerate(jobs[chunk_start : chunk_start + batch_size]):
            layout = create_fixed_layout_from_seed(puzzle)
            env = RicochetRobotsEnv(**environment_spec.to_env_kwargs(fixed_layout=layout))
            observation, _ = env.reset()
            states.append(
                _BatchedModelEpisode(
                    index=chunk_start + offset,
                    puzzle=puzzle,
                    env=env,
                    observation=np.asarray(observation),
                    rng=_model_episode_rng(seed, puzzle, repeat),
                )
            )
        try:
            while True:
                active = [state for state in states if not state.done]
                if not active:
                    break
                observations = np.stack([state.observation for state in active])
                masks = (
                    np.stack([state.env.action_masks() for state in active])
                    if action_masking
                    else None
                )
                actions = _batched_model_actions(
                    model,
                    observations,
                    masks,
                    deterministic=deterministic,
                    episode_rngs=[state.rng for state in active],
                )
                for state, action_value in zip(active, actions, strict=True):
                    action = int(action_value)
                    before = dict(state.env.get_board().robot_positions)
                    try:
                        observation, reward, terminated, truncated, _ = state.env.step(action)
                    except ValueError:
                        state.invalid_actions += 1
                        state.stopped = True
                        continue
                    after = state.env.get_board().robot_positions
                    if before == after:
                        state.no_movement += 1
                    state.observation = np.asarray(observation)
                    state.episode_return += float(reward)
                    state.actions.append(action)
                    state.terminated = bool(terminated)
                    state.truncated = bool(truncated)
        finally:
            for state in states:
                state.env.close()
        for state in states:
            records[state.index] = _batched_episode_record(state)
    if any(record is None for record in records):
        raise AssertionError("Batched evaluation failed to produce every episode record")
    return [record for record in records if record is not None]


def evaluate_controller(
    controller: str,
    puzzles: Sequence[PuzzleMetadata],
    environment_spec: EnvironmentSpec,
    *,
    episodes_per_puzzle: int,
    seed: int,
    model: Any = None,
    deterministic: bool = True,
    bootstrap_samples: int = 0,
    action_masking: bool = False,
    batch_size: int = 64,
) -> Dict[str, Any]:
    if episodes_per_puzzle <= 0:
        raise ValueError("episodes_per_puzzle must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    # Repeating a deterministic policy on a fixed deterministic puzzle only
    # duplicates the same trajectory and falsely inflates episode counts.
    repeats = 1 if controller == "oracle" or (controller == "model" and deterministic) else episodes_per_puzzle
    if controller == "model":
        if model is None:
            raise ValueError("Model evaluation requires a model")
        records = _evaluate_model_batched(
            puzzles,
            environment_spec,
            repeats=repeats,
            seed=seed,
            model=model,
            deterministic=deterministic,
            action_masking=action_masking,
            batch_size=batch_size,
        )
    else:
        records = []
        for puzzle in puzzles:
            for _ in range(repeats):
                records.append(
                    _episode(
                        controller,
                        puzzle,
                        environment_spec,
                        rng=rng,
                        model=model,
                        deterministic=deterministic,
                        action_masking=action_masking,
                    )
                )
    elapsed = time.perf_counter() - started
    return {
        "summary": summarize_records(
            records,
            elapsed,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        ),
        "records": records,
    }


def _policy_construction(config: Phase2Config) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return SB3 policy arguments and a JSON-safe effective-settings record."""
    import torch.nn as nn

    activation_by_name = {"relu": nn.ReLU, "tanh": nn.Tanh}
    if config.policy == "stock_cnn":
        policy_name = "CnnPolicy"
        policy_kwargs: Dict[str, Any] = {
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            "normalize_images": True,
        }
        if config.policy_activation is not None:
            policy_kwargs["activation_fn"] = activation_by_name[config.policy_activation]
        if config.ortho_init is not None:
            policy_kwargs["ortho_init"] = config.ortho_init
        effective_activation = config.policy_activation or "tanh"
        effective_ortho_init = True if config.ortho_init is None else config.ortho_init
    elif config.policy == "grid_resnet":
        from src.models.grid_cnn import GridPlanesResNet

        policy_name = "MlpPolicy"
        effective_activation = config.policy_activation or "relu"
        effective_ortho_init = False if config.ortho_init is None else config.ortho_init
        policy_kwargs = {
            "features_extractor_class": GridPlanesResNet,
            "features_extractor_kwargs": {
                "features_dim": config.features_dim,
                "conv_channels": config.conv_channels,
                "n_blocks": config.n_conv_layers,
            },
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            "activation_fn": activation_by_name[effective_activation],
            "ortho_init": effective_ortho_init,
            "normalize_images": False,
        }
    else:
        from src.models.grid_cnn import GridPlanesCnn

        policy_name = "MlpPolicy"
        effective_activation = config.policy_activation or "relu"
        effective_ortho_init = False if config.ortho_init is None else config.ortho_init
        policy_kwargs = {
            "features_extractor_class": GridPlanesCnn,
            "features_extractor_kwargs": {
                "features_dim": config.features_dim,
                "conv_channels": config.conv_channels,
                "n_conv_layers": config.n_conv_layers,
            },
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            "activation_fn": activation_by_name[effective_activation],
            "ortho_init": effective_ortho_init,
            "normalize_images": False,
        }

    def safe_value(value: Any) -> Any:
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, Mapping):
            return {str(key): safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_value(item) for item in value]
        return value

    effective = {
        "policy_name": policy_name,
        "activation": effective_activation,
        "ortho_init": effective_ortho_init,
        "policy_kwargs": safe_value(policy_kwargs),
    }
    return policy_name, policy_kwargs, effective


def _build_model(vec_env: Any, config: Phase2Config, *, seed: int, device: str) -> Any:
    """Construct the PPO model selected by ``config.policy`` on ``vec_env``."""
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO

    policy_name, policy_kwargs, effective = _policy_construction(config)
    algorithm_class = MaskablePPO if config.use_action_masking else PPO
    model = algorithm_class(
        policy_name,
        vec_env,
        device=device,
        seed=seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    effective["algorithm_class"] = f"{algorithm_class.__module__}.{algorithm_class.__qualname__}"
    effective["use_action_masking"] = config.use_action_masking
    model._phase2_effective_policy = effective
    return model


def _bc_dataset_impl(
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
    include_masks: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Replay stored optimal plans into (observation, optimal_action) pairs.

    Every fixture puzzle carries its BFS-optimal action sequence; the pairs
    are the states along that plan. Observations are binary planes, stored as
    uint8 to keep the dataset compact.
    """
    environment_spec = config.environment_spec(seed)
    selected = list(puzzles)
    if config.bc_max_puzzles is not None and len(selected) > config.bc_max_puzzles:
        order = np.random.default_rng(seed).permutation(len(selected))[: config.bc_max_puzzles]
        selected = [selected[index] for index in sorted(order)]
    observations: List[np.ndarray] = []
    actions: List[int] = []
    masks: List[np.ndarray] = []
    for puzzle in selected:
        layout = create_fixed_layout_from_seed(puzzle)
        env = RicochetRobotsEnv(**environment_spec.to_env_kwargs(fixed_layout=layout))
        obs, _ = env.reset()
        terminated = False
        for action in puzzle.actions_encoded:
            observations.append(obs.astype(np.uint8))
            actions.append(int(action))
            if include_masks:
                mask = env.action_masks()
                if not bool(mask[int(action)]):
                    raise RuntimeError(f"Stored expert action is masked for {_puzzle_id(puzzle)}")
                masks.append(mask)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        env.close()
        if not terminated:
            raise RuntimeError(f"Stored optimal plan failed to reach the goal for {_puzzle_id(puzzle)}")
    return (
        np.stack(observations),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks) if include_masks else None,
    )


def _bc_dataset(
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    observations, actions, _ = _bc_dataset_impl(puzzles, config, seed=seed, include_masks=False)
    return observations, actions


def _observation_probe(
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
) -> np.ndarray:
    """Build a small fixed initial-state probe from training membership."""
    selected = sorted(puzzles, key=_puzzle_id)[: config.health_probe_puzzles]
    observations: List[np.ndarray] = []
    for puzzle in selected:
        layout = create_fixed_layout_from_seed(puzzle)
        env = RicochetRobotsEnv(**config.environment_spec(seed).to_env_kwargs(fixed_layout=layout))
        obs, _ = env.reset()
        observations.append(np.asarray(obs))
        env.close()
    if not observations:
        raise ValueError("Policy-health probe requires at least one training puzzle")
    return np.stack(observations)


def _observation_and_mask_probe(
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(puzzles, key=_puzzle_id)[: config.health_probe_puzzles]
    observations: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for puzzle in selected:
        layout = create_fixed_layout_from_seed(puzzle)
        env = RicochetRobotsEnv(**config.environment_spec(seed).to_env_kwargs(fixed_layout=layout))
        obs, _ = env.reset()
        observations.append(np.asarray(obs))
        masks.append(env.action_masks())
        env.close()
    if not observations:
        raise ValueError("Policy-health probe requires at least one training puzzle")
    return np.stack(observations), np.stack(masks)


def _bc_pretrain(
    model: Any,
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
    device: str,
    wandb_run: Any = None,
    validation_puzzles: Sequence[PuzzleMetadata] | None = None,
) -> Dict[str, Any]:
    """Behaviour-clone the policy on optimal plans before PPO fine-tuning."""
    import torch

    started = time.perf_counter()
    observations, actions, action_masks = _bc_dataset_impl(
        puzzles,
        config,
        seed=seed,
        include_masks=config.use_action_masking,
    )
    validation_data = None
    if validation_puzzles:
        validation_data = _bc_dataset_impl(
            validation_puzzles,
            config,
            seed=seed + 1,
            include_masks=config.use_action_masking,
        )
    build_seconds = time.perf_counter() - started
    print(
        f"[phase2 bc] dataset: {len(actions)} pairs from {len(puzzles)} puzzles "
        f"({build_seconds:.0f}s)",
        flush=True,
    )
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, device=device)
    # The unmasked BC objective still builds masks so every stored expert
    # action is validated, but it must not transfer or randomly gather masks
    # that the loss will never consume.
    mask_tensor = (
        torch.as_tensor(action_masks, device=device)
        if config.bc_apply_action_masks and action_masks is not None
        else None
    )
    validation_tensors = None
    if validation_data is not None:
        val_observations, val_actions, val_masks = validation_data
        validation_tensors = (
            torch.as_tensor(val_observations, dtype=torch.float32, device=device),
            torch.as_tensor(val_actions, device=device),
            (
                torch.as_tensor(val_masks, device=device)
                if config.bc_apply_action_masks and val_masks is not None
                else None
            ),
        )
    # PPO's action head ships with orthogonal gain 0.01, which scales all
    # upstream supervised gradients down ~100x and leaves cross-entropy stuck
    # on the uniform plateau. Structured BC runs explicitly opt into a larger
    # gain; record the choice rather than hiding it in this helper.
    if config.bc_reinitialize_action_head:
        torch.nn.init.orthogonal_(model.policy.action_net.weight, gain=config.bc_action_head_gain)
        torch.nn.init.zeros_(model.policy.action_net.bias)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=config.bc_learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: List[Dict[str, float]] = []
    best_epoch: int | None = None
    best_validation_loss = math.inf
    best_validation_accuracy = -math.inf
    best_policy_state: Dict[str, Any] | None = None

    def distribution_for(batch_obs: Any, batch_masks: Any = None) -> Any:
        distribution = model.policy.get_distribution(batch_obs)
        if config.bc_apply_action_masks and batch_masks is not None:
            distribution.apply_masking(batch_masks)
        return distribution

    def validation_metrics() -> tuple[float, float] | None:
        if validation_tensors is None:
            return None
        val_observations, val_actions, val_masks = validation_tensors
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_correct = torch.zeros((), dtype=torch.int64, device=device)
        with torch.no_grad():
            for start in range(0, len(val_actions), config.bc_batch_size):
                stop = start + config.bc_batch_size
                batch_obs = val_observations[start:stop]
                batch_actions = val_actions[start:stop]
                batch_masks = val_masks[start:stop] if val_masks is not None else None
                distribution = distribution_for(batch_obs, batch_masks)
                loss = -distribution.log_prob(batch_actions).mean()
                total_loss.add_(loss.detach() * len(batch_actions))
                predicted = distribution.distribution.probs.argmax(dim=-1)
                total_correct.add_((predicted == batch_actions).sum())
        totals = torch.stack((total_loss, total_correct.to(dtype=torch.float32))).cpu().numpy()
        return float(totals[0] / len(val_actions)), float(totals[1] / len(val_actions))

    model.policy.train()
    for epoch in range(config.bc_pretrain_epochs):
        order = torch.randperm(len(action_tensor), generator=generator).to(device)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_correct = torch.zeros((), dtype=torch.int64, device=device)
        for start in range(0, len(order), config.bc_batch_size):
            index = order[start : start + config.bc_batch_size]
            distribution = distribution_for(
                obs_tensor[index],
                mask_tensor[index] if mask_tensor is not None else None,
            )
            log_prob = distribution.log_prob(action_tensor[index])
            loss = -log_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss.add_(loss.detach() * len(index))
            predicted = distribution.distribution.probs.argmax(dim=-1)
            total_correct.add_((predicted == action_tensor[index]).sum())
        totals = torch.stack((total_loss, total_correct.to(dtype=torch.float32))).cpu().numpy()
        mean_loss = float(totals[0] / len(action_tensor))
        mean_accuracy = float(totals[1] / len(action_tensor))
        val_metrics = validation_metrics()
        epoch_stats: Dict[str, float] = {
            "loss": mean_loss,
            "accuracy": mean_accuracy,
            "train_loss": mean_loss,
            "train_accuracy": mean_accuracy,
        }
        if val_metrics is not None:
            epoch_stats["validation_loss"] = val_metrics[0]
            epoch_stats["validation_accuracy"] = val_metrics[1]
            candidate = (val_metrics[0], -val_metrics[1])
            incumbent = (best_validation_loss, -best_validation_accuracy)
            if candidate < incumbent:
                best_epoch = epoch + 1
                best_validation_loss, best_validation_accuracy = val_metrics
                best_policy_state = copy.deepcopy(model.policy.state_dict())
        history.append(epoch_stats)
        print(
            f"[phase2 bc] epoch {epoch + 1}/{config.bc_pretrain_epochs} "
            f"loss={epoch_stats['loss']:.3f} accuracy={epoch_stats['accuracy']:.1%}"
            + (
                f" val_loss={epoch_stats['validation_loss']:.3f} "
                f"val_accuracy={epoch_stats['validation_accuracy']:.1%}"
                if "validation_loss" in epoch_stats else ""
            ),
            flush=True,
        )
        if wandb_run is not None:
            payload = {
                "bc/loss": epoch_stats["loss"],
                "bc/accuracy": epoch_stats["accuracy"],
                "bc/epoch": epoch + 1,
            }
            if "validation_loss" in epoch_stats:
                payload["bc/validation_loss"] = epoch_stats["validation_loss"]
                payload["bc/validation_accuracy"] = epoch_stats["validation_accuracy"]
            wandb_run.log(payload)
    if config.bc_select_best_by_validation:
        if best_policy_state is None:
            raise ValueError("BC validation selection requires validation puzzles")
        model.policy.load_state_dict(best_policy_state)
    model.policy.eval()
    return {
        "pairs": len(actions),
        "puzzles": len(puzzles),
        "validation_pairs": len(validation_data[1]) if validation_data is not None else 0,
        "validation_puzzles": len(validation_puzzles) if validation_puzzles is not None else 0,
        "dataset_seconds": build_seconds,
        "epochs": history,
        "selected_epoch": best_epoch if config.bc_select_best_by_validation else config.bc_pretrain_epochs,
        "best_validation_loss": best_validation_loss if best_epoch is not None else None,
        "best_validation_accuracy": best_validation_accuracy if best_epoch is not None else None,
        "action_head_reinitialized": config.bc_reinitialize_action_head,
        "action_head_gain": config.bc_action_head_gain,
        "loss_action_masking": config.bc_apply_action_masks,
        "total_seconds": time.perf_counter() - started,
    }


def _maybe_init_wandb(config: Phase2Config, *, seed: int, output_dir: Path) -> Any:
    """Start a W&B run for one training seed when configured; else None."""
    if config.wandb_project is None:
        return None
    import wandb

    run_name = config.wandb_run_name or output_dir.name
    run = wandb.init(
        project=config.wandb_project,
        name=f"{run_name}-seed{seed}",
        config=asdict(config),
        dir=str(output_dir),
    )
    # Keep BC epochs and PPO timesteps as independent chart axes. In
    # particular, do not let 25 BC log calls make a later PPO step-0 point look
    # non-monotonic and get silently discarded.
    run.define_metric("bc/epoch")
    run.define_metric("bc/*", step_metric="bc/epoch")
    run.define_metric("train/timesteps")
    metric_prefixes = (
        "train/*",
        "rollout/*",
        "health/*",
        "curriculum/*",
        "eval_train_det/*",
        "eval_train_stoch/*",
        "eval_val_det/*",
        "eval_val_stoch/*",
        "test_selected_det/*",
        "test_selected_stoch/*",
        "test_final_det/*",
        "test_final_stoch/*",
        "final/*",
    )
    for pattern in metric_prefixes:
        run.define_metric(pattern, step_metric="train/timesteps")
    return run


def _policy_health_payload(
    model: Any,
    observations: np.ndarray,
    movement_masks: np.ndarray | None = None,
) -> Dict[str, float]:
    """Summarize representation and head health on a fixed training probe."""
    import torch

    def tensor_stats(prefix: str, value: Any) -> Dict[str, float]:
        tensor = value.detach().float()
        flattened = tensor.reshape(tensor.shape[0], -1)
        return {
            f"health/{prefix}_mean": float(flattened.mean().cpu()),
            f"health/{prefix}_std": float(flattened.std(unbiased=False).cpu()),
            f"health/{prefix}_abs_max": float(flattened.abs().max().cpu()),
            f"health/{prefix}_zero_fraction": float((flattened == 0).float().mean().cpu()),
            f"health/{prefix}_between_state_variance": float(flattened.var(dim=0, unbiased=False).mean().cpu()),
        }

    policy = model.policy
    policy.eval()
    with torch.no_grad():
        obs_tensor, _ = policy.obs_to_tensor(observations)
        features = policy.extract_features(obs_tensor)
        if isinstance(features, tuple):
            policy_features, value_features = features
        else:
            policy_features = value_features = features
        latent_pi = policy.mlp_extractor.forward_actor(policy_features)
        latent_vf = policy.mlp_extractor.forward_critic(value_features)
        logits = policy.action_net(latent_pi)
        values = policy.value_net(latent_vf)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
        normalized_entropy = entropy / math.log(probabilities.shape[-1])
        payload: Dict[str, float] = {}
        payload.update(tensor_stats("extractor_output", policy_features))
        payload.update(tensor_stats("policy_latent", latent_pi))
        payload.update(tensor_stats("value_latent", latent_vf))
        payload.update(tensor_stats("logit", logits))
        payload.update(tensor_stats("value", values))
        payload["health/policy_normalized_entropy"] = float(normalized_entropy.mean().cpu())
        payload["health/mean_max_action_probability"] = float(probabilities.max(dim=-1).values.mean().cpu())
        if movement_masks is not None:
            allowed = torch.as_tensor(movement_masks, dtype=torch.bool, device=probabilities.device)
            payload["health/no_movement_probability_mass"] = float(
                probabilities.masked_fill(allowed, 0.0).sum(dim=-1).mean().cpu()
            )
    policy.train()
    return payload


def _make_wandb_rollout_callback(
    wandb_run: Any,
    manager: Any,
    *,
    health_observations: np.ndarray,
    health_movement_masks: np.ndarray | None = None,
    log_every_rollouts: int = 5,
) -> Any:
    """Log rollout-level training stats and curriculum state to W&B."""
    import numpy as _np
    from stable_baselines3.common.callbacks import BaseCallback

    class _WandbRolloutCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._rollouts = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            self._rollouts += 1
            if self._rollouts % log_every_rollouts != 0:
                return
            payload: Dict[str, Any] = {"train/timesteps": self.num_timesteps}
            # PPO populates these values in train(), after the previous
            # rollout. Logging at the next rollout start labels that one-update
            # offset honestly instead of attaching stale values to the rollout
            # that just finished.
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith("train/") and isinstance(value, (int, float, np.number)):
                    payload[key] = float(value)
            buffer = list(self.model.ep_info_buffer or [])
            if buffer:
                payload["rollout/ep_rew_mean"] = float(_np.mean([e["r"] for e in buffer]))
                payload["rollout/ep_len_mean"] = float(_np.mean([e["l"] for e in buffer]))
            if manager is not None:
                payload["curriculum/level"] = manager.get_current_level()
                payload["curriculum/window_success_rate"] = manager.get_success_rate()
                payload["curriculum/episodes_at_level"] = manager.get_episodes_at_level()
                frontier_rate = manager.get_frontier_success_rate()
                if frontier_rate is not None:
                    payload["curriculum/frontier_success_rate"] = frontier_rate
                for band_key, success_rate in manager.get_band_success_rates().items():
                    payload[f"curriculum/band/{band_key}"] = success_rate
                if manager.transitions:
                    payload["curriculum/forced_transitions"] = sum(
                        transition["reason"] == "forced" for transition in manager.transitions
                    )
            payload.update(_policy_health_payload(self.model, health_observations, health_movement_masks))
            wandb_run.log(payload)

    return _WandbRolloutCallback()


def _make_cumulative_schedule_callback(config: Phase2Config) -> Any:
    """Anneal LR/entropy against cumulative experiment timesteps.

    The harness calls ``learn`` once per checkpoint. SB3's native progress
    schedule therefore resets its denominator at every call; this callback
    deliberately keys schedules to ``max(config.checkpoints)`` instead.
    """
    if config.learning_rate_final is None and config.ent_coef_final is None:
        return None
    from stable_baselines3.common.callbacks import BaseCallback

    total_timesteps = max(config.checkpoints)

    class _CumulativeScheduleCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            fraction = min(1.0, self.num_timesteps / max(1, total_timesteps))
            if config.learning_rate_final is not None:
                value = config.learning_rate + (config.learning_rate_final - config.learning_rate) * fraction
                self.model.lr_schedule = lambda _: value
                for group in self.model.policy.optimizer.param_groups:
                    group["lr"] = value
            if config.ent_coef_final is not None:
                self.model.ent_coef = config.ent_coef + (config.ent_coef_final - config.ent_coef) * fraction

    return _CumulativeScheduleCallback()


def _wandb_eval_payload(prefix: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        f"{prefix}/success_rate": summary["success_rate"],
        f"{prefix}/no_movement_rate": summary["no_movement_rate"],
        f"{prefix}/mean_episode_length": summary["mean_episode_length"],
    }
    for length, by_length in summary.get("by_optimal_length", {}).items():
        payload[f"{prefix}/success_len_{length}"] = by_length["success_rate"]
    for robots, by_robots in summary.get("by_robots_moved", {}).items():
        payload[f"{prefix}/success_robots_{robots}"] = by_robots["success_rate"]
    return payload


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _training_factory(bank: PuzzleBank, config: Phase2Config, seed: int):
    staged = config.curriculum_stage_max_lengths is not None or config.curriculum_start_distances is not None
    if staged:
        manager = create_bank_curriculum_manager(
            bank=bank,
            curriculum_levels=_curriculum_levels(config),
            success_rate_threshold=config.curriculum_success_threshold,
            min_episodes_per_level=config.curriculum_min_episodes,
            success_rate_window_size=config.curriculum_window,
            advancement_check_frequency=config.curriculum_check_frequency,
            max_episodes_per_level=config.curriculum_max_episodes_per_stage,
            verbose=True,
        )
    else:
        manager = create_bank_curriculum_manager(
            bank=bank,
            curriculum_levels=_curriculum_levels(config),
            success_rate_threshold=1.0,
            min_episodes_per_level=1_000_000,
            success_rate_window_size=1_000,
            advancement_check_frequency=1_000,
            verbose=False,
        )
    environment_spec = config.environment_spec(seed)

    def factory():
        return BankCurriculumWrapper(
            bank=bank,
            curriculum_manager=manager,
            environment_spec=environment_spec,
            split="train",
            eval_fraction=config.heldout_fraction,
            split_seed=config.split_seed,
            split_key="layout_hash",
            track_curriculum=staged,
            verbose=False,
        )

    factory.curriculum_manager = manager
    return factory


def _fixed_subset_factory(bank: PuzzleBank, config: Phase2Config, seed: int):
    """Create environments backed by the tiny overfit-only bank."""
    criteria = PuzzleCriteria(
        config.spec_key,
        min_optimal_length=config.min_optimal_length,
        max_optimal_length=config.max_optimal_length,
        min_robots_moved=config.min_robots_moved,
        max_robots_moved=config.max_robots_moved,
        bands=[
            {
                "min_optimal_length": length,
                "max_optimal_length": length,
                "min_robots_moved": config.min_robots_moved,
                "max_robots_moved": config.max_robots_moved,
                "weight": 1.0,
            }
            for length in range(config.min_optimal_length, config.max_optimal_length + 1)
        ],
    )

    def factory():
        from src.env.criteria_env import CriteriaFilteredEnv

        return CriteriaFilteredEnv(
            bank,
            criteria,
            environment_spec=config.environment_spec(seed),
            split="all",
            verbose=False,
        )

    return factory


def _build_overfit_bank(path: Path, puzzles: Sequence[PuzzleMetadata]) -> PuzzleBank:
    definition = {"schema_version": 1, "puzzle_ids": sorted(_puzzle_id(puzzle) for puzzle in puzzles)}
    marker = path / "phase2_overfit_fixture.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != definition:
            raise ValueError(f"Existing overfit bank at {path} has different membership")
        return PuzzleBank(path)
    bank = PuzzleBank(path, run_id="phase2-overfit")
    if bank.get_puzzle_count() != 0:
        raise ValueError(f"Refusing to overwrite non-empty overfit bank at {path}")
    bank.add_puzzles(list(puzzles))
    _atomic_json(marker, definition)
    return bank


def train_overfit_milestone(
    puzzles: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Prove the selected policy can memorize a deliberately tiny training suite."""
    import torch
    from stable_baselines3.common.env_util import make_vec_env

    overfit_bank = _build_overfit_bank(output_dir / "overfit" / "bank", puzzles)
    vec_env = make_vec_env(
        _fixed_subset_factory(overfit_bank, config, seed),
        n_envs=config.n_envs,
        seed=seed,
    )
    model = _build_model(vec_env, config, seed=seed, device=device)
    curve: List[Dict[str, Any]] = []
    training_seconds = 0.0
    for checkpoint_index, checkpoint in enumerate(config.checkpoints):
        if checkpoint_index > 0:
            requested = checkpoint - config.checkpoints[checkpoint_index - 1]
            started = time.perf_counter()
            model.learn(total_timesteps=requested, reset_num_timesteps=False)
            if device == "mps":
                torch.mps.synchronize()
            training_seconds += time.perf_counter() - started
        deterministic_eval = evaluate_controller(
            "model",
            puzzles,
            config.environment_spec(seed + 500_000),
            episodes_per_puzzle=1,
            seed=seed + checkpoint + 501,
            model=model,
            deterministic=True,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        stochastic_eval = evaluate_controller(
            "model",
            puzzles,
            config.environment_spec(seed + 600_000),
            episodes_per_puzzle=config.overfit_stochastic_episodes_per_puzzle,
            seed=seed + checkpoint + 502,
            model=model,
            deterministic=False,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        curve.append(
            {
                "requested_checkpoint": checkpoint,
                "actual_timesteps": int(model.num_timesteps),
                "deterministic": deterministic_eval["summary"],
                "stochastic": stochastic_eval["summary"],
            }
        )
        print(
            f"[phase2 overfit] timesteps={int(model.num_timesteps)} "
            f"deterministic={deterministic_eval['summary']['success_rate']:.1%} "
            f"stochastic={stochastic_eval['summary']['success_rate']:.1%}",
            flush=True,
        )

    checkpoint_path = output_dir / "overfit" / "model"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    final_stochastic = curve[-1]["stochastic"]["success_rate"]
    result = {
        "seed": seed,
        "puzzle_ids": [_puzzle_id(puzzle) for puzzle in puzzles],
        "suite_histogram": dict(sorted(Counter(_stratum(puzzle) for puzzle in puzzles).items())),
        "actual_timesteps": int(model.num_timesteps),
        "training_seconds": training_seconds,
        "training_steps_per_second": int(model.num_timesteps) / training_seconds if training_seconds else 0.0,
        "curve": curve,
        "acceptance_threshold": 0.8,
        "passed": final_stochastic >= 0.8,
        "checkpoint": str(checkpoint_path.with_suffix(".zip")),
    }
    _atomic_json(output_dir / "overfit" / "result.json", result)
    vec_env.close()
    return result


def _curriculum_checkpoint_state(manager: Any) -> Dict[str, Any]:
    return {
        "current_level": int(manager.get_current_level()),
        "episode_count": int(manager.episode_count),
        "level_start_episode": int(manager.level_start_episode),
        "episodes_at_level": int(manager.get_episodes_at_level()),
        "success_rate_window": [bool(value) for value in manager.success_rate_window],
        "window_success_rate": float(manager.get_success_rate()),
        "frontier_success_rate": manager.get_frontier_success_rate(),
        "band_success_rates": manager.get_band_success_rates(),
        "transitions": list(manager.transitions),
    }


def _select_curve_point(curve: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select by stochastic validation, deterministic validation, then age."""
    return max(
        curve,
        key=lambda point: (
            float(point["stochastic"]["success_rate"]),
            float(point["deterministic"]["success_rate"]),
            -int(point["actual_timesteps"]),
        ),
    )


def train_seed(
    bank: PuzzleBank,
    training_suite: Sequence[PuzzleMetadata],
    validation_suite: Sequence[PuzzleMetadata],
    test_suite: Sequence[PuzzleMetadata],
    config: Phase2Config,
    *,
    seed: int,
    device: str,
    output_dir: Path,
    bc_puzzles: Sequence[PuzzleMetadata] | None = None,
    bc_validation_puzzles: Sequence[PuzzleMetadata] | None = None,
) -> Dict[str, Any]:
    import torch
    from stable_baselines3.common.env_util import make_vec_env

    factory = _training_factory(bank, config, seed)
    vec_env = make_vec_env(factory, n_envs=config.n_envs, seed=seed)
    model = _build_model(vec_env, config, seed=seed, device=device)
    effective_policy = dict(model._phase2_effective_policy)
    if config.initial_checkpoint_path is not None:
        initial_path = Path(config.initial_checkpoint_path)
        if not initial_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {initial_path}")
        model = type(model).load(
            initial_path,
            env=vec_env,
            device=device,
            seed=seed,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
        )
        model._phase2_effective_policy = effective_policy
    health_observations, health_movement_masks = _observation_and_mask_probe(
        training_suite,
        config,
        seed=seed + 70_000,
    )
    wandb_run = _maybe_init_wandb(config, seed=seed, output_dir=output_dir)
    callbacks: List[Any] = []
    if wandb_run is not None:
        callbacks.append(_make_wandb_rollout_callback(
            wandb_run,
            factory.curriculum_manager,
            health_observations=health_observations,
            health_movement_masks=health_movement_masks,
            log_every_rollouts=config.health_log_every_rollouts,
        ))
    schedule_callback = _make_cumulative_schedule_callback(config)
    if schedule_callback is not None:
        callbacks.append(schedule_callback)
    if len(callbacks) > 1:
        from stable_baselines3.common.callbacks import CallbackList

        learn_callback: Any = CallbackList(callbacks)
    else:
        learn_callback = callbacks[0] if callbacks else None
    bc_stats: Dict[str, Any] | None = None
    if config.bc_pretrain_epochs > 0:
        if not bc_puzzles:
            raise ValueError("BC pretraining requested but no training puzzles were provided")
        # Runs before the checkpoint loop, so the step-0 evaluation measures
        # the cloned policy on its own, before any PPO fine-tuning.
        bc_stats = _bc_pretrain(
            model,
            bc_puzzles,
            config,
            seed=seed,
            device=device,
            wandb_run=wandb_run,
            validation_puzzles=bc_validation_puzzles,
        )
    curve: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    training_seconds = 0.0
    for checkpoint_index, checkpoint in enumerate(config.checkpoints):
        if checkpoint_index > 0:
            requested = checkpoint - config.checkpoints[checkpoint_index - 1]
            started = time.perf_counter()
            model.learn(total_timesteps=requested, reset_num_timesteps=False, callback=learn_callback)
            if device == "mps":
                torch.mps.synchronize()
            training_seconds += time.perf_counter() - started
        evaluation_seed_step = 0 if config.fixed_validation_seeds else checkpoint
        training_deterministic_eval = evaluate_controller(
            "model",
            training_suite,
            config.environment_spec(seed + 50_000),
            episodes_per_puzzle=config.validation_episodes_per_puzzle,
            seed=seed + evaluation_seed_step + 11,
            model=model,
            deterministic=True,
            bootstrap_samples=config.validation_bootstrap_samples,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        training_stochastic_eval = evaluate_controller(
            "model",
            training_suite,
            config.environment_spec(seed + 60_000),
            episodes_per_puzzle=config.validation_episodes_per_puzzle,
            seed=seed + evaluation_seed_step + 12,
            model=model,
            deterministic=False,
            bootstrap_samples=config.validation_bootstrap_samples,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        deterministic_eval = evaluate_controller(
            "model",
            validation_suite,
            config.environment_spec(seed + 100_000),
            episodes_per_puzzle=config.validation_episodes_per_puzzle,
            seed=seed + evaluation_seed_step + 1,
            model=model,
            deterministic=True,
            bootstrap_samples=config.validation_bootstrap_samples,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        stochastic_eval = evaluate_controller(
            "model",
            validation_suite,
            config.environment_spec(seed + 200_000),
            episodes_per_puzzle=config.validation_episodes_per_puzzle,
            seed=seed + evaluation_seed_step + 2,
            model=model,
            deterministic=False,
            bootstrap_samples=config.validation_bootstrap_samples,
            action_masking=config.use_action_masking,
            batch_size=config.evaluation_batch_size,
        )
        counterfactual_unmasked = None
        if config.evaluate_both_mask_modes:
            counterfactual_unmasked = {
                "deterministic": evaluate_controller(
                    "model",
                    validation_suite,
                    config.environment_spec(seed + 100_000),
                    episodes_per_puzzle=config.validation_episodes_per_puzzle,
                    seed=seed + evaluation_seed_step + 1,
                    model=model,
                    deterministic=True,
                    bootstrap_samples=config.validation_bootstrap_samples,
                    action_masking=False,
                    batch_size=config.evaluation_batch_size,
                )["summary"],
                "stochastic": evaluate_controller(
                    "model",
                    validation_suite,
                    config.environment_spec(seed + 200_000),
                    episodes_per_puzzle=config.validation_episodes_per_puzzle,
                    seed=seed + evaluation_seed_step + 2,
                    model=model,
                    deterministic=False,
                    bootstrap_samples=config.validation_bootstrap_samples,
                    action_masking=False,
                    batch_size=config.evaluation_batch_size,
                )["summary"],
            }
        point = {
            "requested_checkpoint": checkpoint,
            "actual_timesteps": int(model.num_timesteps),
            "training": {
                "deterministic": training_deterministic_eval["summary"],
                "stochastic": training_stochastic_eval["summary"],
            },
            "deterministic": deterministic_eval["summary"],
            "stochastic": stochastic_eval["summary"],
            "health": _policy_health_payload(model, health_observations, health_movement_masks),
            "counterfactual_unmasked": counterfactual_unmasked,
        }
        curve.append(point)
        checkpoint_record: Dict[str, Any] = {
            "requested_checkpoint": int(checkpoint),
            "actual_timesteps": int(model.num_timesteps),
            "model": None,
        }
        if config.save_checkpoint_weights:
            checkpoint_dir = output_dir / f"seed_{seed}" / "checkpoints" / f"step_{int(model.num_timesteps)}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_model = checkpoint_dir / "model"
            model.save(checkpoint_model)
            checkpoint_record["model"] = str(checkpoint_model.with_suffix(".zip"))
            _atomic_json(checkpoint_dir / "evaluation.json", point)
            _atomic_json(checkpoint_dir / "curriculum.json", _curriculum_checkpoint_state(factory.curriculum_manager))
            _atomic_json(
                checkpoint_dir / "provenance.json",
                {
                    "seed": seed,
                    "effective_policy": effective_policy,
                    "config": asdict(config),
                    "runtime": _runtime_provenance(device),
                },
            )
        checkpoint_records.append(checkpoint_record)
        by_length = {
            length: f"{summary['success_rate']:.0%}"
            for length, summary in stochastic_eval["summary"]["by_optimal_length"].items()
        }
        print(
            f"[phase2 seed={seed}] timesteps={int(model.num_timesteps)} "
            f"training={training_stochastic_eval['summary']['success_rate']:.1%} "
            f"validation deterministic={deterministic_eval['summary']['success_rate']:.1%} "
            f"stochastic={stochastic_eval['summary']['success_rate']:.1%} "
            f"val-stoch-by-length={by_length}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/timesteps": int(model.num_timesteps),
                    **_wandb_eval_payload("eval_train_det", training_deterministic_eval["summary"]),
                    **_wandb_eval_payload("eval_train_stoch", training_stochastic_eval["summary"]),
                    **_wandb_eval_payload("eval_val_det", deterministic_eval["summary"]),
                    **_wandb_eval_payload("eval_val_stoch", stochastic_eval["summary"]),
                    **point["health"],
                }
            )
    checkpoint_path = output_dir / f"seed_{seed}" / "model"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    selected_point = _select_curve_point(curve) if config.select_checkpoint_by_validation else curve[-1]
    selected_timesteps = int(selected_point["actual_timesteps"])
    selected_record = next(
        record for record in checkpoint_records if int(record["actual_timesteps"]) == selected_timesteps
    )
    selected_model = model
    if config.select_checkpoint_by_validation and selected_timesteps != int(model.num_timesteps):
        if not selected_record["model"]:
            raise RuntimeError("Selected checkpoint has no saved model")
        selected_model = type(model).load(selected_record["model"], device=device)

    def test_model(candidate: Any, *, seed_offset: int) -> Dict[str, Any]:
        evaluation = {
            "deterministic": evaluate_controller(
                "model",
                test_suite,
                config.environment_spec(seed + 300_000 + seed_offset),
                episodes_per_puzzle=1,
                seed=seed + 3 + seed_offset,
                model=candidate,
                deterministic=True,
                bootstrap_samples=config.validation_bootstrap_samples,
                action_masking=config.use_action_masking,
                batch_size=config.evaluation_batch_size,
            ),
            "stochastic": evaluate_controller(
                "model",
                test_suite,
                config.environment_spec(seed + 400_000 + seed_offset),
                episodes_per_puzzle=config.final_stochastic_episodes_per_puzzle,
                seed=seed + 4 + seed_offset,
                model=candidate,
                deterministic=False,
                bootstrap_samples=config.validation_bootstrap_samples,
                action_masking=config.use_action_masking,
                batch_size=config.evaluation_batch_size,
            ),
        }
        if config.evaluate_both_mask_modes:
            evaluation["counterfactual_unmasked"] = {
                "deterministic": evaluate_controller(
                    "model",
                    test_suite,
                    config.environment_spec(seed + 300_000 + seed_offset),
                    episodes_per_puzzle=1,
                    seed=seed + 3 + seed_offset,
                    model=candidate,
                    deterministic=True,
                    bootstrap_samples=config.validation_bootstrap_samples,
                    action_masking=False,
                    batch_size=config.evaluation_batch_size,
                ),
                "stochastic": evaluate_controller(
                    "model",
                    test_suite,
                    config.environment_spec(seed + 400_000 + seed_offset),
                    episodes_per_puzzle=config.final_stochastic_episodes_per_puzzle,
                    seed=seed + 4 + seed_offset,
                    model=candidate,
                    deterministic=False,
                    bootstrap_samples=config.validation_bootstrap_samples,
                    action_masking=False,
                    batch_size=config.evaluation_batch_size,
                ),
            }
        return evaluation

    test_final: Dict[str, Any] | None = None
    test_selected: Dict[str, Any] | None = None
    if config.run_final_test:
        test_final = test_model(model, seed_offset=0)
        test_selected = test_final if selected_model is model else test_model(selected_model, seed_offset=10_000)
    result = {
        "seed": seed,
        "device": device,
        "actual_timesteps": int(model.num_timesteps),
        "training_seconds": training_seconds,
        "training_steps_per_second": int(model.num_timesteps) / training_seconds if training_seconds else 0.0,
        "policy": {
            "algorithm_class": type(model).__name__,
            "class": type(model.policy).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
            "observation_space": str(model.observation_space),
            "action_space": str(model.action_space),
            "effective_construction": effective_policy,
        },
        "curve": curve,
        "checkpoints": checkpoint_records,
        "selected_checkpoint_timesteps": selected_timesteps,
        "selected_checkpoint": selected_record["model"],
        "bc_pretrain": bc_stats,
        "initial_checkpoint_path": config.initial_checkpoint_path,
        "curriculum": {
            "staged": (
                config.curriculum_stage_max_lengths is not None
                or config.curriculum_start_distances is not None
            ),
            "final_level": factory.curriculum_manager.get_current_level(),
            "total_levels": len(factory.curriculum_manager.curriculum_levels),
            "episodes_at_final_level": factory.curriculum_manager.get_episodes_at_level(),
            "window_success_rate": factory.curriculum_manager.get_success_rate(),
            "frontier_success_rate": factory.curriculum_manager.get_frontier_success_rate(),
            "band_success_rates": factory.curriculum_manager.get_band_success_rates(),
            "transitions": list(factory.curriculum_manager.transitions),
        },
        "test_status": "run" if config.run_final_test else "sealed_not_run",
        "test": test_selected,
        "test_selected": test_selected,
        "test_final": test_final,
        "checkpoint": str(checkpoint_path.with_suffix(".zip")),
    }
    if wandb_run is not None:
        final_payload = {
                "train/timesteps": int(model.num_timesteps),
                "final/training_steps_per_second": result["training_steps_per_second"],
                "final/curriculum_level": result["curriculum"]["final_level"],
                "final/selected_checkpoint_timesteps": selected_timesteps,
            }
        if test_selected is not None:
            final_payload.update(_wandb_eval_payload("test_selected_det", test_selected["deterministic"]["summary"]))
            final_payload.update(_wandb_eval_payload("test_selected_stoch", test_selected["stochastic"]["summary"]))
        if test_final is not None:
            final_payload.update(_wandb_eval_payload("test_final_det", test_final["deterministic"]["summary"]))
            final_payload.update(_wandb_eval_payload("test_final_stoch", test_final["stochastic"]["summary"]))
        wandb_run.log(final_payload)
        wandb_run.finish()
    _atomic_json(output_dir / f"seed_{seed}" / "result.json", result)
    vec_env.close()
    return result


def _seed_aggregate(results: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    tested = [result for result in results if result.get("test_selected") is not None]
    rates = [float(result["test_selected"][mode]["summary"]["success_rate"]) for result in tested]
    successes = sum(int(result["test_selected"][mode]["summary"]["successes"]) for result in tested)
    episodes = sum(int(result["test_selected"][mode]["summary"]["episodes"]) for result in tested)
    mean = float(np.mean(rates)) if rates else 0.0
    std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    return {
        "seed_success_rates": rates,
        "mean": mean,
        "sample_std": std,
        "standard_error": std / math.sqrt(len(rates)) if rates else 0.0,
        "min": min(rates) if rates else 0.0,
        "max": max(rates) if rates else 0.0,
        "pooled_successes": successes,
        "pooled_episodes": episodes,
        "pooled_success_rate": successes / episodes if episodes else 0.0,
        "pooled_success_rate_wilson_95": _wilson_interval(successes, episodes),
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_provenance(device: str) -> Dict[str, Any]:
    import stable_baselines3
    import torch

    packages = {}
    for package in (
        "numpy",
        "gymnasium",
        "torch",
        "stable-baselines3",
        "sb3-contrib",
        "pandas",
        "pyarrow",
    ):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    status = _git_value("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "packages": packages,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status": status.splitlines() if status else [],
    }


def render_markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Benchmark Report",
        "",
        f"Generated: {result['completed_at']}",
        f"Device: `{result['runtime']['device']}` on Python `{result['runtime']['python']}`",
        f"Split fingerprint: `{result['split_manifest']['fingerprint']}`",
        "",
        "## Held-out baselines",
        "",
        "| Controller | Success | 95% Wilson interval | No-movement rate | Mean episode length |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("random", "target_robot_only", "oracle"):
        summary = result["baselines"][name]["summary"]
        interval = summary["success_rate_wilson_95"]
        lines.append(
            f"| {name} | {summary['success_rate']:.1%} | "
            f"{interval[0]:.1%}–{interval[1]:.1%} | {summary['no_movement_rate']:.1%} | "
            f"{summary['mean_episode_length']:.2f} |"
        )
    overfit = result["overfit"]
    if overfit.get("skipped"):
        overfit_lines = ["Skipped by configuration."]
    else:
        overfit_final = overfit["curve"][-1]
        overfit_lines = [
            f"The model trained on {len(overfit['puzzle_ids'])} training puzzles reached "
            f"**{overfit_final['deterministic']['success_rate']:.1%} deterministic** and "
            f"**{overfit_final['stochastic']['success_rate']:.1%} stochastic** success at "
            f"{overfit['actual_timesteps']} timesteps. Acceptance (≥80% stochastic): "
            f"**{'passed' if overfit['passed'] else 'failed'}**.",
        ]
    tested = all(seed_result.get("test_selected") is not None for seed_result in result["seeds"])
    lines.extend([
        "",
        "## Tiny-set overfit milestone",
        "",
        *overfit_lines,
        "",
        "## Multi-seed PPO result",
        "",
        (
            "| Seed | Timesteps | Selected checkpoint | Train steps/s | Deterministic test | Stochastic test |"
            if tested
            else "| Seed | Timesteps | Selected checkpoint | Train steps/s | Selected val det | Selected val stoch |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed_result in result["seeds"]:
        selected = next(
            point for point in seed_result["curve"]
            if int(point["actual_timesteps"]) == int(seed_result["selected_checkpoint_timesteps"])
        )
        deterministic_rate = (
            seed_result["test_selected"]["deterministic"]["summary"]["success_rate"]
            if tested else selected["deterministic"]["success_rate"]
        )
        stochastic_rate = (
            seed_result["test_selected"]["stochastic"]["summary"]["success_rate"]
            if tested else selected["stochastic"]["success_rate"]
        )
        lines.append(
            f"| {seed_result['seed']} | {seed_result['actual_timesteps']} | "
            f"{seed_result['selected_checkpoint_timesteps']} | "
            f"{seed_result['training_steps_per_second']:.0f} | "
            f"{deterministic_rate:.1%} | {stochastic_rate:.1%} |"
        )
    if tested:
        deterministic = result["aggregate"]["deterministic"]
        stochastic = result["aggregate"]["stochastic"]
        deterministic_interval = deterministic["pooled_success_rate_wilson_95"]
        stochastic_interval = stochastic["pooled_success_rate_wilson_95"]
        heldout = result["heldout_milestone"]
        lines.extend([
            "",
            f"Deterministic mean across seeds: **{deterministic['mean']:.1%}** "
            f"(sample SD {deterministic['sample_std']:.1%}; episode-pooled 95% Wilson interval "
            f"{deterministic_interval[0]:.1%}–{deterministic_interval[1]:.1%}).",
            f"Stochastic mean across seeds: **{stochastic['mean']:.1%}** "
            f"(sample SD {stochastic['sample_std']:.1%}; episode-pooled 95% Wilson interval "
            f"{stochastic_interval[0]:.1%}–{stochastic_interval[1]:.1%}).",
            f"Held-out acceptance ({heldout['acceptance_rule']}): "
            f"**{'passed' if heldout['passed'] else 'failed'}** "
            f"({heldout['observed']:.1%} observed versus {heldout['acceptance_threshold']:.1%} required).",
        ])
    else:
        lines.extend(["", "Final test remained sealed for this validation-only run."])
    lines.extend([
        "",
        "## Validation curves",
        "",
        "| Seed | Timesteps | Train deterministic | Train stochastic | Validation deterministic | Validation stochastic |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed_result in result["seeds"]:
        for point in seed_result["curve"]:
            lines.append(
                f"| {seed_result['seed']} | {point['actual_timesteps']} | "
                f"{point['training']['deterministic']['success_rate']:.1%} | "
                f"{point['training']['stochastic']['success_rate']:.1%} | "
                f"{point['deterministic']['success_rate']:.1%} | "
                f"{point['stochastic']['success_rate']:.1%} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a compact fixed-distribution benchmark, not a claim of full Ricochet Robots mastery. "
        "The test suite is never used for checkpoint selection; validation and test membership are disjoint from training by layout hash.",
        "",
    ])
    return "\n".join(lines)


def run_phase2_benchmark(
    config: Phase2Config,
    *,
    bank_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Build the fixture, run baselines and multi-seed PPO, and persist results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture_bank(bank_dir, config)
    bank = PuzzleBank(bank_dir)
    split_manifest = build_split_manifest(bank, output_dir / "split_manifest.json", config)
    lookup = _lookup_puzzles(bank, config)
    training_suite = [lookup[puzzle_id] for puzzle_id in split_manifest["training_suite"]]
    validation_suite = [lookup[puzzle_id] for puzzle_id in split_manifest["validation_suite"]]
    test_suite = [lookup[puzzle_id] for puzzle_id in split_manifest["test_suite"]]
    device = _resolve_device(config.device)
    print(
        f"[phase2] fixture={fixture['puzzle_count']} train={split_manifest['counts']['train']} "
        f"validation={split_manifest['counts']['validation']} test={split_manifest['counts']['test']} "
        f"device={device}",
        flush=True,
    )

    baselines = {
        "random": evaluate_controller(
            "random",
            test_suite,
            config.environment_spec(config.split_seed + 1),
            episodes_per_puzzle=config.baseline_episodes_per_puzzle,
            seed=config.split_seed + 101,
        ),
        "target_robot_only": evaluate_controller(
            "target_robot_only",
            test_suite,
            config.environment_spec(config.split_seed + 2),
            episodes_per_puzzle=config.baseline_episodes_per_puzzle,
            seed=config.split_seed + 102,
        ),
        "oracle": evaluate_controller(
            "oracle",
            test_suite,
            config.environment_spec(config.split_seed + 3),
            episodes_per_puzzle=1,
            seed=config.split_seed + 103,
        ),
    }
    print(
        "[phase2] baselines "
        + " ".join(
            f"{name}={evaluation['summary']['success_rate']:.1%}" for name, evaluation in baselines.items()
        ),
        flush=True,
    )
    training_membership = [lookup[puzzle_id] for puzzle_id in split_manifest["memberships"]["train"]]
    validation_membership = [lookup[puzzle_id] for puzzle_id in split_manifest["memberships"]["validation"]]
    if config.run_overfit_milestone:
        training_lookup = training_membership
        overfit_groups: Dict[str, List[PuzzleMetadata]] = defaultdict(list)
        for puzzle in training_lookup:
            overfit_groups[_stratum(puzzle)].append(puzzle)
        overfit_suite: List[PuzzleMetadata] = []
        for stratum in sorted(overfit_groups):
            overfit_suite.extend(
                sorted(overfit_groups[stratum], key=_puzzle_id)[: config.overfit_puzzles_per_stratum]
            )
        print(f"[phase2] starting overfit milestone on {len(overfit_suite)} puzzles", flush=True)
        overfit = train_overfit_milestone(
            overfit_suite,
            config,
            seed=config.training_seeds[0] + 1_000,
            device=device,
            output_dir=output_dir,
        )
    else:
        print("[phase2] overfit milestone skipped by config", flush=True)
        overfit = {"skipped": True}
    seed_results = []
    for seed in config.training_seeds:
        print(f"[phase2] starting training seed {seed}", flush=True)
        seed_results.append(train_seed(
            bank,
            training_suite,
            validation_suite,
            test_suite,
            config,
            seed=seed,
            device=device,
            output_dir=output_dir,
            bc_puzzles=training_membership,
            bc_validation_puzzles=validation_membership,
        ))
    aggregate = {
        "deterministic": _seed_aggregate(seed_results, "deterministic"),
        "stochastic": _seed_aggregate(seed_results, "stochastic"),
    }
    heldout_threshold = baselines["target_robot_only"]["summary"]["success_rate"]
    tests_ran = all(seed_result.get("test_selected") is not None for seed_result in seed_results)
    result = {
        "schema_version": 1,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "runtime": _runtime_provenance(device),
        "fixture": fixture,
        "split_manifest": split_manifest,
        "baselines": baselines,
        "overfit": overfit,
        "seeds": seed_results,
        "aggregate": aggregate,
        "heldout_milestone": {
            "metric": "mean stochastic test success across training seeds",
            "acceptance_rule": "at least the target-robot-only baseline success rate",
            "acceptance_threshold": heldout_threshold,
            "observed": aggregate["stochastic"]["mean"] if tests_ran else None,
            "passed": aggregate["stochastic"]["mean"] >= heldout_threshold if tests_ran else None,
        },
    }
    _atomic_json(output_dir / "results.json", result)
    (output_dir / "report.md").write_text(render_markdown_report(result), encoding="utf-8")
    return result
