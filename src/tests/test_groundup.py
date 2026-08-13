"""E0 correctness gates for the ground-up pipeline (docs/GroundUpRLRedesign.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.env.ricochet_core import Board, apply_action
from src.groundup.curriculum import ReverseFrontierSampler
from src.groundup.distance import relaxed_distance_map
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.puzzles import (
    Puzzle,
    joint_stratified_subset,
    load_puzzles,
    stratified_subset,
)

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


def test_distance_map_cache_is_keyed_on_puzzle_identity_not_seed():
    """The shaping distance map must never be served for the wrong puzzle.

    ``_dmap_cache`` was keyed on ``puzzle.seed``, a *proxy* for identity. It is
    correct today only because every train-split seed happens to be unique; a
    bank with colliding seeds would silently hand one puzzle another's
    distance map, corrupting the potential-based shaping reward with no error
    anywhere. (Keying on ``layout_hash`` instead would be worse -- many puzzles
    share a layout but have different goals.)
    """
    h_walls = np.zeros((9, 8), dtype=bool)
    v_walls = np.zeros((8, 9), dtype=bool)

    def make(goal, actions):
        return Puzzle(
            h_walls=h_walls,
            v_walls=v_walls,
            robots=((3, 0), (3, 4)),
            goal=goal,
            target=0,
            optimal_length=len(actions),
            actions=actions,
            robots_moved=1,
            layout_hash="shared-layout",  # same layout ...
            seed=99,  # ... and a COLLIDING seed, with different goals
        )

    left, right = make((3, 3), ((0, 3),)), make((0, 0), ((0, 0),))
    env = RicochetEnv([left, right], EnvConfig(shaping_coef=0.1))

    for puzzle in (left, right):
        dmap = env._distance_map(puzzle)
        assert dmap[puzzle.goal] == 0, "cache served another puzzle's distance map"
        assert np.array_equal(
            dmap, relaxed_distance_map(puzzle.h_walls, puzzle.v_walls, puzzle.goal)
        )
    # Distinct goals must yield distinct maps even under a seed collision.
    assert not np.array_equal(env._distance_map(left), env._distance_map(right))


def test_env_does_not_advertise_timelimit_truncated(puzzles16):
    """The SB3/MaskablePPO path must keep treating horizon expiry as terminal.

    SB3 bootstraps a truncated episode's value only when the info dict carries
    BOTH ``terminal_observation`` and ``TimeLimit.truncated``
    (stable_baselines3/common/on_policy_algorithm.py). This env emits neither, so
    ``train.py`` correctly treats expiry as a real terminal -- which is the only
    sound choice while ``time_channel=True`` puts the remaining horizon in the
    observation, making V(s, remaining=0) == 0 by definition.

    Adding either key would silently reintroduce the defect that drove the DRC
    87M value divergence: the expiry observation is never stored in the rollout
    buffer, so V is never trained there, and bootstrapping injects an unpinned
    extrapolation straight into the reward. See
    docs/Drc87MRestartCollapseInvestigation.md 6.1.
    """
    config = EnvConfig(time_channel=True, horizon=2)
    env = RicochetEnv(puzzles16, config)
    # optimal_length >= 3 with horizon 2 makes success impossible.
    hard = next(i for i, p in enumerate(puzzles16) if p.optimal_length >= 3)
    env.reset(options={"puzzle_index": hard})
    for _ in range(2):
        mask = env.action_masks()
        _obs, _reward, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
    assert trunc and not term
    assert "TimeLimit.truncated" not in info
    assert "terminal_observation" not in info


@needs_bank8
def test_8x8_bank_loads():
    puzzles = load_puzzles(BANK_8, "train")
    assert all(p.height == 8 and p.num_robots == 2 for p in puzzles)
    subset = stratified_subset(puzzles, 10, seed=0)
    assert len(subset) == 40  # 4 lengths x 10


def _selection_puzzle(optimal_length: int, robots_moved: int, serial: int) -> Puzzle:
    return Puzzle(
        h_walls=np.zeros((5, 4), dtype=bool),
        v_walls=np.zeros((4, 5), dtype=bool),
        robots=((0, 0), (1, 1), (2, 2)),
        goal=(3, 3),
        target=0,
        optimal_length=optimal_length,
        actions=tuple((0, 0) for _ in range(optimal_length)),
        robots_moved=robots_moved,
        layout_hash=f"stratum-{optimal_length}-{robots_moved}-{serial}",
        seed=serial,
    )


def test_joint_stratified_subset_is_deterministic_and_samples_each_group_disjointly():
    from collections import Counter

    counts = {(1, 1): 7, (1, 2): 4, (2, 1): 5, (3, 3): 6}
    puzzles = [
        _selection_puzzle(length, robots, serial)
        for (length, robots), count in counts.items()
        for serial in range(count)
    ]

    first = joint_stratified_subset(puzzles, per_stratum=3, seed=19)
    second = joint_stratified_subset(puzzles, per_stratum=3, seed=19)

    assert [p.layout_hash for p in first] == [p.layout_hash for p in second]
    assert len({p.layout_hash for p in first}) == len(first)
    assert Counter((p.optimal_length, p.robots_moved) for p in first) == {
        stratum: min(3, count) for stratum, count in counts.items()
    }
    # A sparse group appearing or disappearing must not perturb which puzzles
    # are selected from any other, disjoint group.
    for stratum in counts:
        group = [
            puzzle
            for puzzle in puzzles
            if (puzzle.optimal_length, puzzle.robots_moved) == stratum
        ]
        group_only = joint_stratified_subset(group, per_stratum=3, seed=19)
        assert [p.layout_hash for p in first if (p.optimal_length, p.robots_moved) == stratum] == [
            p.layout_hash for p in group_only
        ]


def test_joint_stratified_subset_uses_only_nonempty_sparse_strata():
    from collections import Counter

    puzzles = [
        _selection_puzzle(1, 1, 0),
        *[_selection_puzzle(4, 3, serial) for serial in range(3)],
    ]

    selected = joint_stratified_subset(puzzles, per_stratum=25, seed=7)

    assert Counter((p.optimal_length, p.robots_moved) for p in selected) == {
        (1, 1): 1,
        (4, 3): 3,
    }
    assert joint_stratified_subset(puzzles, per_stratum=0, seed=7) == []


@needs_bank16
def test_policy_heads_relu_and_bounded_features(puzzles16):
    """Regression guard for the E3 v1 collapse: Tanh heads + unbounded features
    saturated the value net into a constant. Heads must be ReLU and the
    extractor must bound feature scale."""
    import torch
    import torch.nn as nn
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.groundup.model import GridTrunk

    config = EnvConfig()
    env = DummyVecEnv([lambda: RicochetEnv(puzzles16[:20], config, seed=0)])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=GridTrunk,
            features_extractor_kwargs=dict(features_dim=64, channels=16, n_blocks=1),
            net_arch=dict(pi=[32], vf=[32]),
            activation_fn=nn.ReLU,
        ),
    )
    activations = [type(m) for m in model.policy.mlp_extractor.value_net]
    assert nn.Tanh not in activations
    assert nn.ReLU in activations
    obs = env.reset()
    with torch.no_grad():
        feats = model.policy.extract_features(torch.as_tensor(obs))
        if isinstance(feats, tuple):
            feats = feats[0]
    assert float(feats.abs().max()) < 50.0  # LayerNorm keeps features bounded


@needs_bank16
def test_stop_planes_correct(puzzles16):
    """Each stop plane is one-hot at exactly the slide destination, and at
    d=1 the goal cell carries the stop plane of a solving action."""
    config = EnvConfig(stop_planes=True, time_channel=False, horizon=50)
    env = RicochetEnv(puzzles16, config)
    rng = np.random.default_rng(5)
    for _ in range(20):
        idx = int(rng.integers(len(puzzles16)))
        obs, _ = env.reset(options={"puzzle_index": idx, "start_distance": 1})
        off = env.stop_planes_offset
        goal_solving = []
        for action in range(env.action_space.n):
            plane = obs[off + action]
            assert plane.sum() == 1.0
            sr, sc = np.unravel_index(int(plane.argmax()), plane.shape)
            expect = env._slide_from(env._robots, action // 4, action % 4, env._puzzle)
            assert (sr, sc) == expect
            if action < 4 and (sr, sc) == env._puzzle.goal:
                goal_solving.append(action)
        assert goal_solving, "d=1 state must have a target-robot action stopping on the goal"


@needs_bank16
def test_spatial_policy_builds_and_masks(puzzles16):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.groundup.model import SpatialPolicy

    config = EnvConfig(stop_planes=True, distance_plane=True)
    probe = RicochetEnv(puzzles16[:10], config, seed=0)
    env = DummyVecEnv([lambda: RicochetEnv(puzzles16[:10], config, seed=0)])
    model = MaskablePPO(
        SpatialPolicy,
        env,
        n_steps=16,
        batch_size=16,
        device="cpu",
        policy_kwargs=dict(stop_offset=probe.stop_planes_offset, trunk_channels=16, trunk_blocks=2),
    )
    raw = RicochetEnv(puzzles16[:10], config, seed=1)
    obs, _ = raw.reset(options={"puzzle_index": 0})
    mask = raw.action_masks()
    obs_t = torch.as_tensor(obs[None])
    dist = model.policy.get_distribution(obs_t, action_masks=torch.as_tensor(mask[None]))
    probs = dist.distribution.probs.detach().numpy()[0]
    assert np.all(probs[~mask] < 1e-8), "masked actions must get zero probability"
    assert abs(probs.sum() - 1.0) < 1e-5
    # short learn smoke: one rollout + update must run
    model.learn(total_timesteps=32)


@needs_bank16
def test_rgb_obs_rendering(puzzles16):
    """RGB dict obs: image faithfully encodes walls/goal/robots; planes carry helpers."""
    config = EnvConfig(rgb_obs=True, rgb_cell_px=5, stop_planes=True, distance_plane=True)
    env = RicochetEnv(puzzles16, config)
    obs, _ = env.reset(options={"puzzle_index": 0})
    puzzle = puzzles16[0]
    k = 5
    img, planes = obs["image"], obs["planes"]
    assert img.shape == (3, 16 * k + 1, 16 * k + 1) and planes.shape == (16 + 1 + 1, 16, 16)
    # target robot (canonical slot 0) drawn red at its cell centre
    tr, tc = env._robots[0]
    center = img[:, tr * k + 2, tc * k + 2]
    assert center[0] > 0.8 and center[1] < 0.3 and center[2] < 0.3
    # walls are black pixels: check a known interior wall from the layout
    hw = np.argwhere(puzzle.h_walls[1:-1, :])
    r, c = hw[0][0] + 1, hw[0][1]
    assert np.all(img[:, r * k, c * k + 2] < 0.05)
    # goal cell tinted toward red (not pure white) unless a robot covers it
    gr, gc = puzzle.goal
    if (gr, gc) not in env._robots:
        goal_px = img[:, gr * k + 2, gc * k + 2]
        assert goal_px[0] > goal_px[1] and goal_px[0] > goal_px[2]
    # image changes when a robot moves; static bg (a wall pixel) unchanged
    mask = env.action_masks()
    obs2, _, _, _, _ = env.step(int(np.flatnonzero(mask)[0]))
    assert not np.array_equal(obs["image"], obs2["image"])
    assert np.all(obs2["image"][:, r * k, c * k + 2] < 0.05)


@needs_bank16
def test_rgb_rendering_geometry_regressions(puzzles16):
    """Visual-review fixes (2026-07-20): closed wall corners, uniform cell
    sizes, full-interior sprites, tint confined to the cell interior."""
    from src.groundup.render import GRID_GRAY, WALL_BLACK, render, static_background

    k = 5
    config = EnvConfig(rgb_obs=True, rgb_cell_px=k, stop_planes=True, distance_plane=True)
    env = RicochetEnv(puzzles16, config)
    obs, _ = env.reset(options={"puzzle_index": 0})
    puzzle = puzzles16[0]
    img = obs["image"]

    # (1) wall junctions are closed: every pair of touching wall segments
    # shares a black corner pixel. Central 2x2 block: all four outer corners.
    height = puzzle.height
    mid = height // 2
    for (yr, xc) in [(mid - 1, mid - 1), (mid - 1, mid + 1), (mid + 1, mid - 1), (mid + 1, mid + 1)]:
        corner = img[:, yr * k, xc * k]
        assert np.all(corner <= WALL_BLACK + 1e-6), f"open wall corner at line ({yr},{xc})"

    # (2) uniform cell interiors: every cell, including bottom/right edges,
    # has a full (k-1)x(k-1) non-boundary interior (probe: last row/col cells'
    # interior pixels are not grid-coloured).
    for (r, c) in [(height - 1, 0), (0, puzzle.width - 1), (height - 1, puzzle.width - 1)]:
        interior = img[:, r * k + 1 : (r + 1) * k, c * k + 1 : (c + 1) * k]
        assert interior.shape[1:] == (k - 1, k - 1)
        assert not np.any(np.all(np.isclose(interior, GRID_GRAY), axis=0)), (
            f"grid line bleeding into cell ({r},{c}) interior"
        )

    # (3) robot sprites fill the whole interior
    tr, tc = env._robots[0]
    sprite = img[:, tr * k + 1 : (tr + 1) * k, tc * k + 1 : (tc + 1) * k]
    assert np.allclose(sprite[0], sprite[0, 0, 0]) and sprite.shape[1:] == (k - 1, k - 1)
    assert float(sprite[0].min()) > 0.8  # red channel saturated across the sprite

    # (4) goal tint never touches boundary pixels: all four bounding lines of
    # the goal cell are grid gray or wall black, never a red-tinted blend.
    gr, gc = puzzle.goal
    bg = static_background(puzzle, k)
    for y, xs in [(gr * k, slice(gc * k, (gc + 1) * k + 1)), ((gr + 1) * k, slice(gc * k, (gc + 1) * k + 1))]:
        line = bg[:, y, xs]
        assert np.all(np.isclose(line, GRID_GRAY) | np.isclose(line, WALL_BLACK))
    for x, ys in [(gc * k, slice(gr * k, (gr + 1) * k + 1)), ((gc + 1) * k, slice(gr * k, (gr + 1) * k + 1))]:
        line = bg[:, ys, x]
        assert np.all(np.isclose(line, GRID_GRAY) | np.isclose(line, WALL_BLACK))


@needs_bank16
def test_rgb_spatial_policy_builds_and_masks(puzzles16):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.groundup.model import SpatialRgbPolicy

    config = EnvConfig(rgb_obs=True, rgb_cell_px=5, stop_planes=True, distance_plane=True)
    env = DummyVecEnv([lambda: RicochetEnv(puzzles16[:10], config, seed=0)])
    model = MaskablePPO(
        SpatialRgbPolicy,
        env,
        n_steps=16,
        batch_size=16,
        device="cpu",
        policy_kwargs=dict(cell_px=5, embed_channels=8, trunk_channels=16, trunk_blocks=1),
    )
    raw = RicochetEnv(puzzles16[:10], config, seed=1)
    obs, _ = raw.reset(options={"puzzle_index": 0})
    mask = raw.action_masks()
    obs_t = {key: torch.as_tensor(value[None]) for key, value in obs.items()}
    dist = model.policy.get_distribution(obs_t, action_masks=torch.as_tensor(mask[None]))
    probs = dist.distribution.probs.detach().numpy()[0]
    assert np.all(probs[~mask] < 1e-8) and abs(probs.sum() - 1.0) < 1e-5
    model.learn(total_timesteps=32)
