"""E0 correctness gates for the ground-up pipeline (docs/GroundUpRLRedesign.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.env.ricochet_core import Board, apply_action
from src.groundup.curriculum import ReverseFrontierSampler
from src.groundup.distance import relaxed_distance_map
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.puzzles import Puzzle, load_puzzles, stratified_subset

BANK_16 = Path("artifacts/phase4_bank/size-16x16_numR4_ET2_CL2_gen10.parquet")
BANK_8 = Path("artifacts/phase2b_bank/size-8x8_numR2_ET1_CL1_gen10.parquet")

needs_bank16 = pytest.mark.skipif(not BANK_16.exists(), reason="phase4 bank not present")
needs_bank8 = pytest.mark.skipif(not BANK_8.exists(), reason="phase2b bank not present")


@pytest.fixture(scope="module")
def puzzles16():
    return load_puzzles(BANK_16, "val")[:300]


def _board_from_puzzle(puzzle: Puzzle, positions=None) -> Board:
    return Board(
        height=puzzle.height,
        width=puzzle.width,
        h_walls=puzzle.h_walls.copy(),
        v_walls=puzzle.v_walls.copy(),
        robot_positions={i: tuple(p) for i, p in enumerate(positions or puzzle.robots)},
        goal_position=puzzle.goal,
        target_robot=puzzle.target,
    )


@needs_bank16
def test_slide_matches_trusted_core(puzzles16):
    """Differential test: env dynamics == vetted ricochet_core rules."""
    rng = np.random.default_rng(0)
    config = EnvConfig(canonical_identity=False, shaping_coef=0.0, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    for trial in range(30):
        idx = int(rng.integers(len(puzzles16)))
        env.reset(options={"puzzle_index": idx})
        board = _board_from_puzzle(puzzles16[idx])
        for _ in range(15):
            action = int(rng.integers(env.action_space.n))
            _, _, term, trunc, _ = env.step(action)
            board = apply_action(board, action // 4, action % 4)
            expected = [board.robot_positions[i] for i in range(board.num_robots)]
            assert env._robots == expected
            if term or trunc:
                break


@needs_bank16
def test_masks_match_actual_movement(puzzles16):
    rng = np.random.default_rng(1)
    config = EnvConfig(canonical_identity=False, shaping_coef=0.0, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    for trial in range(20):
        idx = int(rng.integers(len(puzzles16)))
        env.reset(options={"puzzle_index": idx})
        mask = env.action_masks()
        before = list(env._robots)
        for action in range(env.action_space.n):
            moved_pos = env._slide_from(before, action // 4, action % 4, env._puzzle)
            moved = moved_pos != before[action // 4]
            assert bool(mask[action]) == moved, f"mask mismatch action={action}"


@needs_bank16
def test_stored_plan_solves_in_env(puzzles16):
    """With canonicalization off, replaying the bank plan terminates successfully."""
    config = EnvConfig(canonical_identity=False, shaping_coef=0.0, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    for idx in range(50):
        env.reset(options={"puzzle_index": idx})
        puzzle = puzzles16[idx]
        for step_i, (rid, direction) in enumerate(puzzle.actions):
            _, reward, term, trunc, info = env.step(rid * 4 + direction)
            if step_i < len(puzzle.actions) - 1:
                assert not term
        assert term and info["is_success"]
        assert reward > 0.5  # goal reward dominates


@needs_bank16
def test_reverse_start_distance_semantics(puzzles16):
    """From start_distance d, the plan suffix of length d reaches the goal."""
    config = EnvConfig(canonical_identity=False, shaping_coef=0.0, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    for puzzle_idx in range(30):
        puzzle = puzzles16[puzzle_idx]
        for d in range(1, puzzle.optimal_length + 1):
            env.reset(options={"puzzle_index": puzzle_idx, "start_distance": d})
            suffix = puzzle.actions[puzzle.optimal_length - d :]
            assert len(suffix) == d
            for rid, direction in suffix:
                _, _, term, _, info = env.step(rid * 4 + direction)
            assert term and info["is_success"], f"puzzle {puzzle_idx} d={d}"


@needs_bank16
def test_canonical_obs_and_actions(puzzles16):
    """Target robot is channel 5 and actions 0-3 move it."""
    config = EnvConfig(canonical_identity=True, shaping_coef=0.0, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    for idx in range(20):
        obs, _ = env.reset(options={"puzzle_index": idx})
        puzzle = puzzles16[idx]
        tr, tc = puzzle.robots[puzzle.target]
        assert obs[5, tr, tc] == 1.0 and obs[5].sum() == 1.0
        # exactly one robot per plane, robots don't overlap
        for ch in range(5, 5 + puzzle.num_robots):
            assert obs[ch].sum() == 1.0
        # actions 0-3 move the target robot only
        mask = env.action_masks()
        legal = np.flatnonzero(mask[:4])
        if len(legal):
            before_others = env._robots[1:]
            env.step(int(legal[0]))
            assert env._robots[1:] == before_others


@needs_bank16
def test_shaping_telescopes(puzzles16):
    """Discounted shaping sum == gamma^T phi(s_T) - phi(s_0)."""
    gamma, coef = 0.99, 0.1
    base = EnvConfig(canonical_identity=False, shaping_coef=0.0, time_channel=False, horizon=30, gamma=gamma)
    shaped = EnvConfig(canonical_identity=False, shaping_coef=coef, time_channel=False, horizon=30, gamma=gamma)
    env0 = RicochetEnv(puzzles16, base)
    env1 = RicochetEnv(puzzles16, shaped)
    rng = np.random.default_rng(2)
    for trial in range(20):
        idx = int(rng.integers(len(puzzles16)))
        env0.reset(options={"puzzle_index": idx})
        env1.reset(options={"puzzle_index": idx})
        phi0 = env1._potential()
        disc_shaping = 0.0
        t = 0
        while True:
            action = int(rng.integers(env0.action_space.n))
            _, r0, term, trunc, _ = env0.step(action)
            _, r1, term1, trunc1, _ = env1.step(action)
            assert (term, trunc) == (term1, trunc1)
            disc_shaping += (gamma**t) * (r1 - r0)
            t += 1
            if term or trunc:
                break
        phi_T = env1._potential()
        expected = (gamma**t) * phi_T - phi0
        assert abs(disc_shaping - expected) < 1e-6
        if term:
            assert abs(phi_T) < 1e-9  # potential vanishes at the goal


@needs_bank16
def test_distance_map_admissible(puzzles16):
    for puzzle in puzzles16[:100]:
        dmap = relaxed_distance_map(puzzle.h_walls, puzzle.v_walls, puzzle.goal)
        assert dmap[puzzle.goal] == 0
        tr, tc = puzzle.robots[puzzle.target]
        assert dmap[tr, tc] <= puzzle.optimal_length


@needs_bank16
def test_split_disjoint_and_stable():
    train = load_puzzles(BANK_16, "train")
    val = load_puzzles(BANK_16, "val")
    test = load_puzzles(BANK_16, "test")
    total = load_puzzles(BANK_16, "all")
    assert len(train) + len(val) + len(test) == len(total)
    train_hashes = {p.layout_hash for p in train}
    assert train_hashes.isdisjoint({p.layout_hash for p in val})
    assert train_hashes.isdisjoint({p.layout_hash for p in test})
    held_frac = (len(val) + len(test)) / len(total)
    assert 0.2 < held_frac < 0.3


def test_reverse_frontier_sampler_progression():
    dummy = []
    for length in (1, 2, 3, 4):
        for k in range(5):
            dummy.append(
                Puzzle(
                    h_walls=np.zeros((9, 8), dtype=bool),
                    v_walls=np.zeros((8, 9), dtype=bool),
                    robots=((0, 0), (7, 7)),
                    goal=(3, 3),
                    target=0,
                    optimal_length=length,
                    actions=tuple([(0, 1)] * length),
                    robots_moved=1,
                    layout_hash=f"h{length}-{k}",
                    seed=length * 100 + k,
                )
            )
    sampler = ReverseFrontierSampler(dummy, tau=0.75, min_episodes=10, ema_beta=0.5)
    assert sampler.frontier() == 1
    rng = np.random.default_rng(3)
    for _ in range(50):
        puzzle, d = sampler.sample(rng)
        assert 1 <= d <= puzzle.optimal_length
    for _ in range(20):
        sampler.record(1, True, dummy[0])
    assert sampler.frontier() == 2
    for _ in range(20):
        sampler.record(2, False, dummy[0])
    assert sampler.frontier() == 2  # stays until mastered


@needs_bank16
def test_time_channel_and_truncation(puzzles16):
    config = EnvConfig(canonical_identity=True, shaping_coef=0.1, time_channel=True, horizon=5)
    env = RicochetEnv(puzzles16, config)
    hard = next(i for i, p in enumerate(puzzles16) if p.optimal_length >= 6)
    obs, _ = env.reset(options={"puzzle_index": hard})
    assert obs.shape[0] == 5 + 4 + 1
    assert np.allclose(obs[-1], 1.0)
    for step in range(5):
        mask = env.action_masks()
        obs, _, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
    assert trunc and not term
    assert np.allclose(obs[-1], 0.0)


@needs_bank8
def test_8x8_bank_loads():
    puzzles = load_puzzles(BANK_8, "train")
    assert all(p.height == 8 and p.num_robots == 2 for p in puzzles)
    subset = stratified_subset(puzzles, 10, seed=0)
    assert len(subset) == 40  # 4 lengths x 10
