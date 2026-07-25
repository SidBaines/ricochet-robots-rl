"""CPU-only tests for the recurrent DRC PPO trainer (src/groundup/drc_train.py).

Covers the parts recurrent PPO implementations classically get wrong:
buffer/replay state-reset semantics at episode boundaries, mask storage and
application in replay, GAE against a hand-computed case, truncation
bootstrapping vs termination, end-to-end learn() smoke with the run-directory
contract, and seed determinism.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.groundup.benchmark_drc_throughput import parse_candidate
from src.groundup.drc import DRCNet, DRCState
from src.groundup.drc_train import (
    DRCTrainConfig,
    RecurrentPPO,
    _load_training_checkpoint,
    _save_training_checkpoint,
    build_and_train,
    build_net,
    clipped_value_loss,
    compute_gae,
    evaluate_drc,
    normalize_advantages,
    replay_sequences,
)
from src.groundup.curriculum import ReverseFrontierSampler
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.puzzles import Puzzle

BANK_8 = Path("artifacts/phase2b_bank/size-8x8_numR2_ET1_CL1_gen10.parquet")
PRIMARY_DRC_CONFIG = Path("configs/groundup_drc33_p1_12x12_planefree.yaml")


# ------------------------------------------------------------------- fixtures
def synthetic_puzzles():
    """Hand-built 8x8 2-robot puzzles with verified plans (lengths 1 and 2)."""

    def make(robots, goal, actions, seed):
        return Puzzle(
            h_walls=np.zeros((9, 8), dtype=bool),
            v_walls=np.zeros((8, 9), dtype=bool),
            robots=robots,
            goal=goal,
            target=0,
            optimal_length=len(actions),
            actions=actions,
            robots_moved=1,
            layout_hash=f"syn{seed}",
            seed=seed,
        )

    # Directions: UP=0, DOWN=1, LEFT=2, RIGHT=3; action = robot_id * 4 + dir.
    return [
        make(((3, 0), (3, 4)), (3, 3), ((0, 3),), 1),  # RIGHT stops before robot 1
        make(((5, 7), (2, 7)), (3, 7), ((0, 0),), 2),  # UP stops below robot 1
        make(((0, 0), (7, 4)), (7, 3), ((0, 1), (0, 3)), 3),  # DOWN then RIGHT
        make(((0, 1), (7, 5)), (7, 4), ((0, 1), (0, 3)), 4),  # DOWN then RIGHT
    ]


def tiny_net(obs_shape, n_actions, seed=0) -> DRCNet:
    torch.manual_seed(seed)
    return DRCNet(
        observation_shape=tuple(obs_shape),
        n_actions=int(n_actions),
        depth=1,
        ticks=1,
        hidden_channels=8,
        encoder_channels=(8, 8),
        encoder_kernels=(3, 3),
        encoder_strides=(1, 1),
        mlp_hidden=32,
    )


def make_algo(net, envs, **overrides) -> RecurrentPPO:
    kwargs = dict(
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        target_kl=None,
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return RecurrentPPO(net, envs, **kwargs)


class ScriptedEnv:
    """Duck-typed masked env: deterministic step-indexed obs, reward 0, ends
    (terminated or truncated, per ``mode``) after ``end_at`` steps."""

    def __init__(self, obs_shape, n_actions, end_at, mode):
        assert mode in ("terminate", "truncate")
        self.observation_space = spaces.Box(0.0, 1.0, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)
        self._end_at = end_at
        self._mode = mode
        self._steps = 0
        self._base = np.linspace(
            0.0, 0.7, num=int(np.prod(obs_shape)), dtype=np.float32
        ).reshape(obs_shape)

    def observation(self, steps):
        return np.mod(self._base + 0.1 * steps, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return self.observation(0), {}

    def step(self, action):
        self._steps += 1
        done = self._steps >= self._end_at
        terminated = done and self._mode == "terminate"
        truncated = done and self._mode == "truncate"
        return self.observation(self._steps), 0.0, terminated, truncated, {}

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[1] = True
        mask[3] = True
        return mask


# ------------------------------------------------------------------------ GAE
def test_gae_matches_hand_computed_case():
    # T=3, B=1; episode boundary before t=2; bootstrap through last_values.
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.2], [0.1]])
    episode_starts = torch.tensor([[True], [False], [True]])
    last_values = torch.tensor([0.3])
    last_episode_starts = torch.tensor([False])
    adv, ret = compute_gae(
        rewards, values, episode_starts, last_values, last_episode_starts,
        gamma=0.9, gae_lambda=0.8,
    )
    # t=2: delta = 2 + 0.9*0.3 - 0.1 = 2.17 (bootstrapped; not last-step-terminal)
    # t=1: next is an episode start -> delta = 0 - 0.2 = -0.2, no carry
    # t=0: delta = 1 + 0.9*0.2 - 0.5 = 0.68; A = 0.68 + 0.9*0.8*(-0.2) = 0.536
    expected = torch.tensor([[0.536], [-0.2], [2.17]])
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(ret, expected + values, atol=1e-6)


def test_gae_termination_stops_bootstrap_at_last_step():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.4]])
    episode_starts = torch.tensor([[True]])
    adv_alive, _ = compute_gae(
        rewards, values, episode_starts,
        torch.tensor([2.0]), torch.tensor([False]), gamma=0.9, gae_lambda=1.0,
    )
    adv_done, _ = compute_gae(
        rewards, values, episode_starts,
        torch.tensor([2.0]), torch.tensor([True]), gamma=0.9, gae_lambda=1.0,
    )
    assert torch.allclose(adv_alive, torch.tensor([[1.0 + 0.9 * 2.0 - 0.4]]))
    assert torch.allclose(adv_done, torch.tensor([[1.0 - 0.4]]))


# --------------------------------------------------------------------- replay
def test_replay_resets_state_at_done_and_applies_masks():
    obs_shape, n_actions = (5, 6, 7), 9
    net = tiny_net(obs_shape, n_actions)
    T, B = 4, 2
    gen = torch.Generator().manual_seed(3)
    obs = torch.rand((T, B, *obs_shape), generator=gen)
    masks = torch.ones((T, B, n_actions), dtype=torch.bool)
    masks[:, :, 0] = False
    masks[2, 0, 5:] = False
    episode_starts = torch.zeros((T, B), dtype=torch.bool)
    episode_starts[0, 0] = True  # env 0 starts fresh
    episode_starts[2, 0] = True  # done INSIDE the segment for env 0
    # Nonzero carried-in state: env 1 continues an old episode throughout.
    zero = net.initial_state(B)
    init = DRCState(
        h=torch.randn(zero.h.shape, generator=gen),
        c=torch.randn(zero.c.shape, generator=gen),
    )

    with torch.no_grad():
        logits, values = replay_sequences(net, obs, masks, episode_starts, init)

    # Manual per-env forward with explicit zeroing at episode starts.
    for b in range(B):
        state = DRCState(h=init.h[:, b : b + 1], c=init.c[:, b : b + 1])
        for t in range(T):
            if episode_starts[t, b]:
                state = net.initial_state(1)
            with torch.no_grad():
                logits_one, values_one, state = net(
                    obs[t, b : b + 1], state, action_mask=masks[t, b : b + 1]
                )
            assert torch.allclose(logits[t, b], logits_one[0], atol=1e-5), f"t={t} b={b}"
            assert torch.allclose(values[t, b], values_one[0], atol=1e-5), f"t={t} b={b}"

    # Masks were applied inside the replay.
    assert (logits[~masks] < -1e8).all()
    assert (logits[masks] > -1e8).all()

    # Sanity: dropping the mid-segment reset changes env 0's outputs from t=2 on.
    no_reset = episode_starts.clone()
    no_reset[2, 0] = False
    with torch.no_grad():
        logits2, _ = replay_sequences(net, obs, masks, no_reset, init)
    assert not torch.allclose(logits2[2, 0], logits[2, 0])
    assert torch.allclose(logits2[:2], logits[:2], atol=1e-6)


def test_collection_state_semantics_match_replay():
    """The definitive recurrence contract: replaying the buffer with unchanged
    weights must reproduce the behaviour logprobs/values exactly."""
    puzzles = synthetic_puzzles()
    env_config = EnvConfig(horizon=3)
    envs = [RicochetEnv(puzzles, env_config, seed=100 + r) for r in range(3)]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n)
    torch.manual_seed(0)
    algo = make_algo(net, envs, n_steps=8, batch_size=8)

    for rollout in range(2):  # second segment starts from a carried state
        buf = algo.collect_rollout()
        # horizon=3 guarantees episode boundaries inside an 8-step segment
        assert bool(buf.episode_starts[1:].any()), f"rollout {rollout}"
        if rollout == 1:
            assert float(buf.initial_state.h.abs().sum()) > 0.0
        with torch.no_grad():
            logits, values = replay_sequences(
                net, buf.obs, buf.masks, buf.episode_starts, buf.initial_state
            )
        dist = torch.distributions.Categorical(logits=logits.reshape(-1, algo.n_actions))
        log_probs = dist.log_prob(buf.actions.reshape(-1)).reshape(buf.actions.shape)
        assert torch.allclose(log_probs, buf.log_probs, atol=1e-5)
        assert torch.allclose(values, buf.values, atol=1e-5)
        # Every stored action legal under its stored mask.
        flat_masks = buf.masks.reshape(-1, algo.n_actions)
        chosen = flat_masks[torch.arange(flat_masks.shape[0]), buf.actions.reshape(-1)]
        assert bool(chosen.all())


# ---------------------------------------------------------------- truncation
def test_truncation_bootstraps_final_value_and_termination_does_not():
    obs_shape, n_actions = (4, 5, 5), 6

    def run(mode):
        net = tiny_net(obs_shape, n_actions, seed=42)  # identical weights per run
        envs = [ScriptedEnv(obs_shape, n_actions, end_at=2, mode=mode)]
        torch.manual_seed(7)  # identical action sampling per run
        algo = make_algo(net, envs, n_steps=4, batch_size=4, gamma=0.9)
        return net, algo.collect_rollout()

    net, buf_trunc = run("truncate")
    _, buf_term = run("terminate")

    # Same weights, same observations, same rng -> identical trajectories.
    assert torch.equal(buf_trunc.actions, buf_term.actions)
    assert torch.equal(buf_trunc.episode_starts, buf_term.episode_starts)
    # Episode ends at env step t=1 -> t=2 starts a fresh episode.
    assert bool(buf_trunc.episode_starts[2, 0]) and not bool(buf_trunc.episode_starts[1, 0])
    # Masked collection: only actions 1 and 3 are legal.
    assert set(buf_trunc.actions.flatten().tolist()) <= {1, 3}

    # Termination: raw env reward (0). Truncation: + gamma * V(final obs).
    assert float(buf_term.rewards[1, 0]) == 0.0
    with torch.no_grad():
        state = net.initial_state(1)
        _, _, state = net(buf_trunc.obs[0, :1], state, action_mask=buf_trunc.masks[0, :1])
        _, _, state = net(buf_trunc.obs[1, :1], state, action_mask=buf_trunc.masks[1, :1])
        final_obs = ScriptedEnv(obs_shape, n_actions, 2, "truncate").observation(2)
        _, v_boot, _ = net(torch.as_tensor(final_obs[None]), state)
    assert abs(float(v_boot)) > 1e-6  # bootstrap is non-trivial for this net
    assert abs(float(buf_trunc.rewards[1, 0]) - 0.9 * float(v_boot)) < 1e-6
    # ... and it propagates into the returns.
    assert not torch.allclose(buf_trunc.returns[:2, 0], buf_term.returns[:2, 0])


# ------------------------------- truncation bootstrap vs an observed horizon
# Regression tests for the defect that drove the drc33_p1_12x12_planefree value
# divergence (docs/Drc87MRestartCollapseInvestigation.md 6.1):
#
#   env.step increments _steps BEFORE building the observation, so at horizon
#   expiry the returned observation has time channel exactly 0. collect_rollout
#   fed that observation to the value head and added gamma*V to the reward, then
#   discarded it -- it never enters buf.obs, so V is never trained there. And
#   because time_channel=True puts the remaining horizon IN the observation, the
#   MDP is finite-horizon and fully observed, so V(s, remaining=0) == 0 by
#   definition. The bootstrap therefore injected an unpinned extrapolation
#   straight into the returns: measured V at expiry reached -12.3 against
#   goal_reward 1.0 by 87M, i.e. timing out "paid" twelve times more than
#   solving the puzzle.


class _RewardRecordingEnv(RicochetEnv):
    """RicochetEnv that records exactly what ``step`` returned, for reward audits."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rewards_returned: list[float] = []
        self.truncations = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.rewards_returned.append(float(reward))
        if truncated and not terminated:
            self.truncations += 1
        return obs, reward, terminated, truncated, info


def _forced_truncation_setup(*, time_channel: bool, value: float = 5.0, n_steps: int = 4):
    """Envs whose EVERY step truncates, and a net whose V is identically ``value``.

    The puzzles need two optimal moves and the horizon is 1, so the goal is
    unreachable and no episode can ever terminate -- every stored step is a
    truncation. Pinning the critic to a constant makes any bootstrap term
    exactly ``gamma * value``, so its presence or absence is unambiguous.
    """
    puzzles = [p for p in synthetic_puzzles() if p.optimal_length == 2]
    assert len(puzzles) >= 2, "fixture must offer 2-move puzzles"
    config = EnvConfig(horizon=1, time_channel=time_channel)
    envs = [_RewardRecordingEnv(puzzles, config, seed=700 + rank) for rank in range(2)]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
    with torch.no_grad():
        net.critic.weight.zero_()
        net.critic.bias.fill_(value)
    torch.manual_seed(31)
    algo = make_algo(net, envs, n_steps=n_steps, batch_size=n_steps, n_epochs=1)
    return envs, algo


def test_no_truncation_bootstrap_when_the_horizon_is_in_the_observation():
    """With time_channel=True, expiry is a real terminal: rewards must be untouched."""
    envs, algo = _forced_truncation_setup(time_channel=True, value=5.0)
    assert algo.bootstrap_on_truncation is False

    buf = algo.collect_rollout()

    # Non-vacuous: every single stored step really did truncate.
    assert all(env.truncations == algo.n_steps for env in envs)
    assert not bool(buf.episode_starts[1:].eq(False).any())  # every step ends an episode

    # V is pinned to 5.0, so a bootstrap would inflate every reward by gamma*5.
    for b, env in enumerate(envs):
        assert buf.rewards[:, b].tolist() == pytest.approx(env.rewards_returned, abs=1e-5)


def test_stored_observations_never_reach_horizon_expiry():
    """Why bootstrapping at expiry is unsound: that input is never a training target."""
    puzzles = synthetic_puzzles()
    envs = [
        RicochetEnv(puzzles, EnvConfig(horizon=3, time_channel=True), seed=700 + rank)
        for rank in range(2)
    ]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
    torch.manual_seed(31)
    algo = make_algo(net, envs, n_steps=12, batch_size=12, n_epochs=1)
    buf = algo.collect_rollout()
    # Time channel is the last plane; remaining == 0 only ever occurs on the
    # observation returned BY the truncating step, which is never stored.
    assert float(buf.obs[:, :, -1].min()) > 0.0


def test_truncation_bootstrap_still_applies_when_the_horizon_is_hidden():
    """Guard against over-fixing: with time_channel=False the bootstrap is correct."""
    envs, algo = _forced_truncation_setup(time_channel=False, value=5.0)
    assert algo.bootstrap_on_truncation is True

    buf = algo.collect_rollout()

    assert all(env.truncations == algo.n_steps for env in envs)
    expected = algo.gamma * 5.0
    for b, env in enumerate(envs):
        for t, raw in enumerate(env.rewards_returned):
            assert float(buf.rewards[t, b]) == pytest.approx(raw + expected, abs=1e-5)


def test_bootstrap_on_truncation_is_refused_when_the_horizon_is_observed():
    """The inconsistent combination must fail loudly, not silently corrupt rewards."""
    puzzles = synthetic_puzzles()
    envs = [RicochetEnv(puzzles, EnvConfig(horizon=4, time_channel=True), seed=1)]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n)
    with pytest.raises(ValueError, match="time_channel"):
        make_algo(net, envs, n_steps=4, batch_size=4, bootstrap_on_truncation=True)


def test_primary_drc_config_disables_the_truncation_bootstrap():
    """The registered 12x12 config sets time_channel: true, so it must not bootstrap."""
    config = DRCTrainConfig.from_yaml(PRIMARY_DRC_CONFIG)
    assert config.time_channel is True
    puzzles = synthetic_puzzles()
    envs = [RicochetEnv(puzzles, EnvConfig(horizon=4, time_channel=True), seed=1)]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n)
    assert make_algo(net, envs, n_steps=4, batch_size=4).bootstrap_on_truncation is False


# ------------------------------------------------- post-convergence hardening
def test_normalize_advantages_floor_caps_tiny_std_amplification():
    # Near-solved batch: residual advantages at 1e-3 scale.
    adv = torch.tensor([1e-3, -1e-3, 2e-3, -2e-3, 0.0])
    plain = normalize_advantages(adv, std_floor=0.0, clip=None)
    floored = normalize_advantages(adv, std_floor=0.05, clip=None)
    # Plain normalization inflates residual noise to unit scale ...
    assert float(plain.abs().max()) > 1.0
    # ... the floor keeps it at its natural scale / 0.05 (<= 20x).
    assert torch.allclose(floored, (adv - adv.mean()) / (0.05 + 1e-8))
    assert float(floored.abs().max()) < 0.1
    # Floor inactive when std is healthy: identical to the plain formula.
    healthy = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5])
    assert torch.equal(
        normalize_advantages(healthy, std_floor=0.05, clip=None),
        normalize_advantages(healthy, std_floor=0.0, clip=None),
    )
    # std_floor=0 / clip=None reproduce the pre-hardening formula exactly.
    assert torch.equal(plain, (adv - adv.mean()) / (adv.std() + 1e-8))


def test_normalize_advantages_clips_outliers():
    adv = torch.tensor([0.0] * 99 + [1.0])  # one extreme outlier
    clipped = normalize_advantages(adv, std_floor=0.0, clip=3.0)
    unclipped = normalize_advantages(adv, std_floor=0.0, clip=None)
    assert float(unclipped.max()) > 3.0
    assert float(clipped.max()) == 3.0
    assert torch.equal(clipped, unclipped.clamp(-3.0, 3.0))


def test_clipped_value_loss_matches_sb3_and_bounds_gradients():
    old_values = torch.tensor([0.0, 0.0, 0.0])
    returns = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.1, 0.9, -0.9], requires_grad=True)
    # clip_range_vf=None reproduces the plain MSE exactly.
    assert torch.equal(
        clipped_value_loss(values, old_values, returns, None),
        torch.nn.functional.mse_loss(returns, values),
    )
    # PPO2 max-of-losses semantics (NOT SB3's clip-only variant, which
    # under-corrects large errors and produced exponential value divergence in
    # drc33_8x8_gate_v2: value_loss 43 -> 9.4e6 via the truncation-bootstrap
    # feedback loop).
    loss = clipped_value_loss(values, old_values, returns, clip_range_vf=0.2)
    clipped_pred = torch.tensor([0.1, 0.2, -0.2])
    expected = torch.maximum((returns - values.detach()) ** 2, (returns - clipped_pred) ** 2).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    # element 0: inside the clip -> full gradient
    assert values.grad[0] != 0.0
    # element 1: moved toward target past the clip; pessimistic clipped branch
    # dominates and its gradient is zero -> no further incentive to move
    assert values.grad[1] == 0.0
    # element 2: far from target; unclipped branch dominates -> FULL correction
    # gradient (this is the case the clip-only variant killed)
    assert values.grad[2] != 0.0


def test_hardening_guards_inactive_on_healthy_update():
    """On a healthy first update the guards must not perturb the losses:
    at unchanged weights the value clip is a no-op, the advantage std is well
    above the floor, and normalized advantages are far inside the clamp."""

    def run_once(**hardening):
        puzzles = synthetic_puzzles()
        envs = [RicochetEnv(puzzles, EnvConfig(horizon=4), seed=800 + r) for r in range(2)]
        net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
        torch.manual_seed(31)
        # batch_size = n_steps * num_envs -> exactly one minibatch.
        algo = make_algo(net, envs, n_steps=6, batch_size=12, n_epochs=1, seed=5, **hardening)
        buf = algo.collect_rollout()
        assert float(buf.advantages.std()) > 0.05  # healthy-batch premise
        return algo.update(buf, progress_remaining=1.0)

    off = run_once(adv_std_floor=0.0, adv_clip=None, clip_range_vf=None)
    on = run_once(adv_std_floor=0.05, adv_clip=10.0, clip_range_vf=0.2)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert abs(on[key] - off[key]) < 1e-6, key
    for metrics in (off, on):
        assert np.isfinite(metrics["approx_kl_max"])
        assert metrics["approx_kl_max"] >= metrics["approx_kl"]


# ----------------------------------------------------------------- guardrails
def test_batch_size_must_be_whole_sequences():
    obs_shape, n_actions = (4, 5, 5), 6
    net = tiny_net(obs_shape, n_actions)
    envs = [ScriptedEnv(obs_shape, n_actions, 3, "truncate") for _ in range(2)]
    with pytest.raises(ValueError, match="whole number"):
        make_algo(net, envs, n_steps=8, batch_size=12)
    with pytest.raises(ValueError, match="sequences per minibatch"):
        make_algo(net, envs, n_steps=8, batch_size=24)  # 3 seqs > 2 envs


def test_env_gamma_must_match_ppo_gamma():
    puzzles = synthetic_puzzles()
    envs = [RicochetEnv(puzzles, EnvConfig(gamma=0.99), seed=0)]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n)
    with pytest.raises(ValueError, match="gamma"):
        make_algo(net, envs, gamma=0.9)


def test_config_yaml_and_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "run_name: abc\nnum_envs: 4\ndrc_depth: 2\n"
        "drc_hidden_channels: 8\ndrc_encoder_channels: [8, 8]\n"
        "adv_std_floor: 0.02\nadv_clip: 5.0\nclip_range_vf: null\n"
    )
    config = DRCTrainConfig.from_yaml(path)
    assert config.run_name == "abc"
    assert config.num_envs == 4
    assert config.drc_depth == 2
    assert config.adv_std_floor == 0.02
    assert config.adv_clip == 5.0
    assert config.clip_range_vf is None
    net = build_net(config, (8, 8, 8), 8)
    assert net.depth == 2 and net.ticks == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        DRCTrainConfig.from_yaml(bad)


def test_primary_12x12_drc_config_is_helper_free_and_extension_safe():
    config = DRCTrainConfig.from_yaml(PRIMARY_DRC_CONFIG)
    assert config.stop_planes is False
    assert config.distance_plane is False
    assert config.time_channel is True
    assert config.canonical_identity is True
    assert config.horizon == 24
    assert config.drc_depth == 3 and config.drc_ticks == 3
    assert (config.num_envs, config.n_steps, config.batch_size, config.n_epochs) == (
        256,
        32,
        2048,
        1,
    )
    assert config.total_steps == 120_000_000
    assert config.schedule_total_steps == 200_000_000
    assert config.checkpoint_every == 5_000_000
    # Updated by the 2026-07-24 battery pause: the registered continuation now
    # resumes from the immutable 87M pause checkpoint (commit 96fcbff changed
    # the config but not this assertion).
    assert config.resume_from == (
        "artifacts/groundup/drc33_p1_12x12_planefree/"
        "training_pause_87007232.pt"
    )
    # 4 walls + goal + 3 robots + time; 3 robots x 4 directions.
    net = build_net(config, (9, 12, 12), 12)
    assert net.observation_shape == (9, 12, 12)
    assert net._state_hw == (12, 12)
    assert net.n_actions == 12


def test_throughput_candidate_parser_enforces_recurrent_minibatches():
    assert parse_candidate("128x32x1024x1") == (128, 32, 1024, 1)
    with pytest.raises(argparse.ArgumentTypeError, match="divisible"):
        parse_candidate("32x32x100x1")
    with pytest.raises(argparse.ArgumentTypeError, match="exceeds"):
        parse_candidate("2x32x96x1")


# ----------------------------------------------------------------- evaluation
def test_evaluate_drc_protocol_and_determinism():
    puzzles = synthetic_puzzles()
    env_config = EnvConfig(horizon=6)
    probe = RicochetEnv(puzzles, env_config, seed=0)
    net = tiny_net(probe.observation_space.shape, probe.action_space.n)

    det = evaluate_drc(
        net, puzzles, env_config, episodes_per_puzzle=2, deterministic=True, batch_size=3
    )
    assert det["episodes"] == len(puzzles) * 2
    assert set(det["by_length"]) == {1, 2}
    assert det["deterministic"] is True
    assert set(det) >= {"success_rate", "episodes", "by_length", "by_robots_moved"}

    det2 = evaluate_drc(
        net, puzzles, env_config, episodes_per_puzzle=2, deterministic=True, batch_size=3
    )
    assert det == det2  # deterministic protocol is repeatable

    stoch = evaluate_drc(
        net, puzzles, env_config, episodes_per_puzzle=3, deterministic=False, batch_size=4, seed=1
    )
    assert stoch["episodes"] == len(puzzles) * 3
    assert stoch["deterministic"] is False


# ---------------------------------------------------------------- determinism
def test_seeded_runs_produce_identical_first_update():
    def run_once():
        puzzles = synthetic_puzzles()
        envs = [RicochetEnv(puzzles, EnvConfig(horizon=4), seed=500 + r) for r in range(2)]
        net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
        torch.manual_seed(31)
        algo = make_algo(net, envs, n_steps=6, batch_size=6, n_epochs=2, seed=5)
        buf = algo.collect_rollout()
        return algo.update(buf, progress_remaining=1.0)

    m1 = run_once()
    m2 = run_once()
    assert m1 == m2
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "approx_kl_max"):
        assert np.isfinite(m1[key])
    assert m1["approx_kl_max"] >= m1["approx_kl"]  # max-KL telemetry
    assert m1["n_grad_updates"] == 4  # 2 epochs x 2 minibatches, no early stop


def test_exact_resume_matches_uninterrupted_next_update(tmp_path):
    """Full state restore must reproduce the next rollout and optimizer update."""

    def build(seed):
        puzzles = synthetic_puzzles()
        sampler = ReverseFrontierSampler(puzzles, min_episodes=2)
        envs = [
            RicochetEnv(
                puzzles,
                EnvConfig(horizon=4),
                sampler=sampler,
                seed=700 + rank,
            )
            for rank in range(2)
        ]
        net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
        torch.manual_seed(31)
        algo = make_algo(
            net,
            envs,
            n_steps=6,
            batch_size=6,
            n_epochs=2,
            seed=seed,
        )
        return sampler, algo

    config = DRCTrainConfig(
        num_envs=2,
        n_steps=6,
        batch_size=6,
        n_epochs=2,
        schedule_total_steps=1000,
        drc_depth=1,
        drc_ticks=1,
        drc_hidden_channels=8,
        drc_encoder_channels=(8, 8),
        drc_encoder_kernels=(3, 3),
        drc_mlp_hidden=32,
        device="cpu",
    )
    sampler1, algo1 = build(seed=5)
    first = algo1.collect_rollout()
    algo1.update(first, progress_remaining=0.9)
    checkpoint = tmp_path / "training.pt"
    _save_training_checkpoint(
        algo1,
        sampler1,
        config,
        checkpoint,
        run_state={"elapsed_s": 1.0},
        metadata={"test": True},
    )

    uninterrupted_buf = algo1.collect_rollout()
    uninterrupted_metrics = algo1.update(uninterrupted_buf, progress_remaining=0.8)
    uninterrupted_weights = {
        key: value.detach().clone() for key, value in algo1.net.state_dict().items()
    }
    uninterrupted_sampler = sampler1.state_dict()

    sampler2, algo2 = build(seed=5)
    _load_training_checkpoint(checkpoint, algo=algo2, sampler=sampler2, config=config)
    resumed_buf = algo2.collect_rollout()
    resumed_metrics = algo2.update(resumed_buf, progress_remaining=0.8)

    assert torch.equal(resumed_buf.actions, uninterrupted_buf.actions)
    assert torch.equal(resumed_buf.rewards, uninterrupted_buf.rewards)
    assert torch.equal(resumed_buf.episode_starts, uninterrupted_buf.episode_starts)
    assert resumed_metrics == uninterrupted_metrics
    assert algo2.num_timesteps == algo1.num_timesteps
    for key, value in algo2.net.state_dict().items():
        assert torch.equal(value, uninterrupted_weights[key]), key
    assert np.array_equal(sampler2.ema, uninterrupted_sampler["ema"])
    assert np.array_equal(sampler2.count, uninterrupted_sampler["count"])

    changed = dataclasses.replace(config, horizon=config.horizon + 1)
    sampler3, algo3 = build(seed=5)
    with pytest.raises(ValueError, match="config contract changed"):
        _load_training_checkpoint(checkpoint, algo=algo3, sampler=sampler3, config=changed)


# ------------------------------------------------- exact-resume hardening
# Gaps in test_exact_resume_matches_uninterrupted_next_update that the 87M
# restart investigation (docs/Drc87MRestartCollapseInvestigation.md) had to
# close by hand: it resumes inside the SAME process, runs on CPU only (so the
# ``device_rng_state`` mps branch of RecurrentPPO.load_state_dict was never
# executed), compares only ONE post-resume iteration, never asserts the Adam
# moments round-trip, and builds both sides from the same seed -- so a field
# that silently failed to restore would still compare equal by accident.


def _resume_digest(algo: RecurrentPPO, sampler) -> dict:
    """Digest every piece of state that determines the future trajectory."""
    import hashlib

    state = algo.state_dict()

    def dig(*chunks: bytes) -> str:
        d = hashlib.sha256()
        for chunk in chunks:
            d.update(chunk)
        return d.hexdigest()

    def dig_tensors(mapping) -> str:
        d = hashlib.sha256()
        for key in sorted(mapping):
            value = mapping[key]
            d.update(key.encode())
            d.update(value.detach().cpu().contiguous().numpy().tobytes())
        return d.hexdigest()

    optimizer = state["optimizer_state_dict"]
    opt = hashlib.sha256()
    for pid in sorted(optimizer["state"]):
        entry = optimizer["state"][pid]
        opt.update(str(pid).encode())
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = entry[key]
            array = (
                value.detach().cpu().contiguous().numpy()
                if torch.is_tensor(value)
                else np.asarray(value)
            )
            opt.update(np.ascontiguousarray(array).tobytes())
    return {
        "num_timesteps": state["num_timesteps"],
        "net": dig_tensors(state["net_state_dict"]),
        "optimizer": opt.hexdigest(),
        "obs": dig(np.ascontiguousarray(state["obs"]).tobytes()),
        "episode_start": dig(np.ascontiguousarray(state["episode_start"]).tobytes()),
        "recurrent_h": dig(state["recurrent_state"]["h"].contiguous().numpy().tobytes()),
        "recurrent_c": dig(state["recurrent_state"]["c"].contiguous().numpy().tobytes()),
        "env_states": dig(json.dumps(state["env_states"], sort_keys=True, default=str).encode()),
        "sampler": dig(
            np.ascontiguousarray(sampler.state_dict()["ema"]).tobytes(),
            np.ascontiguousarray(sampler.state_dict()["count"]).tobytes(),
        ),
        "minibatch_generator": dig(state["minibatch_generator_state"].numpy().tobytes()),
        "torch_rng": dig(state["torch_rng_state"].numpy().tobytes()),
        "device_rng": (
            dig(state["device_rng_state"][1].cpu().numpy().tobytes())
            if state.get("device_rng_state") is not None
            else None
        ),
    }


def _resume_fixture(*, device: str, env_seed: int, net_seed: int, algo_seed: int):
    puzzles = synthetic_puzzles()
    sampler = ReverseFrontierSampler(puzzles, min_episodes=2)
    envs = [
        RicochetEnv(puzzles, EnvConfig(horizon=4), sampler=sampler, seed=env_seed + rank)
        for rank in range(2)
    ]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=net_seed)
    torch.manual_seed(31)
    algo = make_algo(
        net, envs, n_steps=6, batch_size=6, n_epochs=2, seed=algo_seed, device=device
    )
    return sampler, algo


_RESUME_CONFIG = DRCTrainConfig(
    num_envs=2,
    n_steps=6,
    batch_size=6,
    n_epochs=2,
    schedule_total_steps=1000,
    drc_depth=1,
    drc_ticks=1,
    drc_hidden_channels=8,
    drc_encoder_channels=(8, 8),
    drc_encoder_kernels=(3, 3),
    drc_mlp_hidden=32,
    device="cpu",
)

# Body of the child process used by the real-process-restart test. Kept as a
# string so the child is a genuinely fresh interpreter (a same-process resume
# cannot detect state that leaks through module globals or the device context).
_RESTART_CHILD = """
import json, sys
from src.groundup.drc_train import _load_training_checkpoint
from src.tests.test_drc_train import _RESUME_CONFIG, _resume_digest, _resume_fixture

checkpoint, iters, device, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
sampler, algo = _resume_fixture(device=device, env_seed=901, net_seed=77, algo_seed=13)
fresh = _resume_digest(algo, sampler)
_load_training_checkpoint(checkpoint, algo=algo, sampler=sampler, config=_RESUME_CONFIG)
restored = _resume_digest(algo, sampler)
digests = []
for _ in range(iters):
    buf = algo.collect_rollout()
    algo.update(buf, progress_remaining=0.8)
    digests.append(_resume_digest(algo, sampler))
json.dump({"fresh": fresh, "restored": restored, "iterations": digests}, open(out, "w"))
"""


def _run_resume_arms(device: str, tmp_path: Path, iters: int = 3) -> tuple[dict, dict, list, list]:
    """Uninterrupted arm vs same-process resumed arm, both from one checkpoint."""
    sampler_a, algo_a = _resume_fixture(device=device, env_seed=700, net_seed=9, algo_seed=5)
    algo_a.update(algo_a.collect_rollout(), progress_remaining=0.9)
    checkpoint = tmp_path / f"training_{device}.pt"
    _save_training_checkpoint(
        algo_a, sampler_a, _RESUME_CONFIG, checkpoint,
        run_state={"elapsed_s": 1.0}, metadata={"test": True},
    )
    uninterrupted = []
    for _ in range(iters):
        algo_a.update(algo_a.collect_rollout(), progress_remaining=0.8)
        uninterrupted.append(_resume_digest(algo_a, sampler_a))

    # Deliberately DIFFERENT seeds: the resumed arm's pre-restore state cannot
    # coincide with the checkpoint, so every equality below is load-bearing.
    sampler_b, algo_b = _resume_fixture(device=device, env_seed=901, net_seed=77, algo_seed=13)
    fresh = _resume_digest(algo_b, sampler_b)
    _load_training_checkpoint(checkpoint, algo=algo_b, sampler=sampler_b, config=_RESUME_CONFIG)
    restored = _resume_digest(algo_b, sampler_b)
    resumed = []
    for _ in range(iters):
        algo_b.update(algo_b.collect_rollout(), progress_remaining=0.8)
        resumed.append(_resume_digest(algo_b, sampler_b))
    return fresh, restored, uninterrupted, resumed


def test_exact_resume_overwrites_every_field_of_a_differently_seeded_process(tmp_path):
    """Restore must replace ALL state, and hold over several iterations.

    Building the resumed arm from different seeds makes every comparison
    meaningful: a field left at its fresh-construction value would show up both
    as a ``fresh``/``restored`` collision and as a trajectory divergence.
    """
    fresh, restored, uninterrupted, resumed = _run_resume_arms("cpu", tmp_path)

    # num_timesteps starts at 0 on both sides; device_rng is None on CPU.
    collisions = [
        key
        for key in fresh
        if key != "num_timesteps" and fresh[key] is not None and fresh[key] == restored[key]
    ]
    assert collisions == [], f"restore left these at the fresh-construction value: {collisions}"
    assert resumed == uninterrupted, "resumed trajectory diverged from the uninterrupted one"
    # Optimizer moments specifically (the original test only checked weights).
    assert restored["optimizer"] != fresh["optimizer"]
    assert resumed[-1]["optimizer"] == uninterrupted[-1]["optimizer"]


def test_exact_resume_survives_a_real_process_restart(tmp_path):
    """A fresh interpreter resuming the checkpoint must match the same-process arm."""
    import subprocess
    import sys

    _fresh, restored, uninterrupted, _resumed = _run_resume_arms("cpu", tmp_path)
    out = tmp_path / "child.json"
    subprocess.run(
        [sys.executable, "-c", _RESTART_CHILD,
         str(tmp_path / "training_cpu.pt"), "3", "cpu", str(out)],
        cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, text=True,
    )
    child = json.loads(out.read_text())
    assert child["restored"] == restored, "cross-process restore differs from in-process restore"
    assert child["iterations"] == uninterrupted, "cross-process resume diverged"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_exact_resume_round_trips_device_rng_state_on_mps(tmp_path):
    """Exercise the mps ``device_rng_state`` branch of load_state_dict.

    Actions are sampled with ``torch.multinomial`` on the device, so on MPS the
    action stream depends on the MPS generator, not the CPU one. Without this
    test that save/restore branch is never executed.
    """
    fresh, restored, uninterrupted, resumed = _run_resume_arms("mps", tmp_path)
    assert restored["device_rng"] is not None, "mps device_rng_state was not captured"
    assert restored["device_rng"] != fresh["device_rng"]
    assert resumed == uninterrupted, "resumed trajectory diverged from uninterrupted on MPS"


# ---------------------------------------------------------------------- smoke
def test_learn_smoke_end_to_end(tmp_path):
    config = DRCTrainConfig(
        bank_path=str(BANK_8) if BANK_8.exists() else "",
        out_dir=str(tmp_path),
        run_name="drc_smoke",
        horizon=8,
        curriculum="reverse",
        seed=3,
        num_envs=4,
        n_steps=16,
        batch_size=32,
        n_epochs=2,
        total_steps=192,
        eval_every=64,
        eval_per_length=2,
        final_eval_per_length=2,
        final_eval_episodes=2,
        eval_batch_size=8,
        checkpoint_every=64,
        drc_depth=1,
        drc_ticks=1,
        drc_hidden_channels=8,
        drc_encoder_channels=(8, 8),
        drc_encoder_kernels=(3, 3),
        drc_mlp_hidden=32,
        device="cpu",
    )
    if BANK_8.exists():
        report = build_and_train(config)
    else:
        report = build_and_train(config, puzzles_override=synthetic_puzzles())

    out = tmp_path / "drc_smoke"
    saved_config = json.loads((out / "config.json").read_text())
    assert saved_config["run_name"] == "drc_smoke"
    assert saved_config["drc_depth"] == 1

    curve = [json.loads(line) for line in (out / "curve.jsonl").read_text().splitlines()]
    assert len(curve) >= 1
    assert {
        "timesteps", "elapsed_s", "steps_per_s", "val_det", "sampler", "losses",
        "best_val_det", "best_timesteps",
    } <= set(curve[0])
    assert curve[0]["val_det"]["episodes"] > 0
    assert curve[0]["sampler"]["sampler"] == "reverse_frontier"
    assert "approx_kl_max" in curve[0]["losses"]
    # Best tracking is monotone and consistent with the entries seen so far.
    best_so_far = None
    for entry in curve:
        if best_so_far is None or entry["val_det"]["success_rate"] > best_so_far:
            best_so_far = entry["val_det"]["success_rate"]
        assert entry["best_val_det"] == best_so_far

    saved_report = json.loads((out / "report.json").read_text())
    assert saved_report == json.loads(json.dumps(report))  # file matches return value
    for key in (
        "config",
        "policy_params",
        "final_val_det",
        "final_val_stoch",
        "baseline_random",
        "baseline_target_only",
        "sampler_final",
    ):
        assert key in saved_report
    for eval_key in ("final_val_det", "final_val_stoch", "baseline_random", "baseline_target_only"):
        summary = saved_report[eval_key]
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["episodes"] > 0
    assert saved_report["final_val_stoch"]["deterministic"] is False

    # Keep-best: periodic evals ran, so a best checkpoint exists and the
    # report evals used it.
    assert (out / "model_best.pt").exists()
    assert saved_report["report_eval_weights"] == "best"
    assert saved_report["best_val_det"] == curve[-1]["best_val_det"]
    assert saved_report["best_timesteps"] == curve[-1]["best_timesteps"]

    # Model checkpoints exist and reload into a working net.
    assert (out / "model_final.pt").exists()
    payload = torch.load(out / "model_final.pt", weights_only=False)
    net = build_net(
        DRCTrainConfig(**payload["config"]), payload["observation_shape"], payload["n_actions"]
    )
    net.load_state_dict(payload["state_dict"])

    # Immutable model/training snapshots and provenance are written on the
    # declared cadence; latest/final training states support crash recovery.
    snapshots = sorted(out.glob("model_[0-9]*.pt"))
    training_snapshots = sorted(out.glob("training_[0-9]*.pt"))
    assert len(snapshots) >= 2
    assert len(training_snapshots) == len(snapshots)
    assert (out / "training_latest.pt").exists()
    assert (out / "training_final.pt").exists()
    manifest = [
        json.loads(line) for line in (out / "snapshot_manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == len(snapshots)
    assert all(
        set(entry)
        >= {
            "timesteps",
            "elapsed_s",
            "steps_per_s",
            "git",
            "config_fingerprint",
            "bank_sha256",
            "split_fingerprints",
            "policy_params",
        }
        for entry in manifest
    )


def test_report_evals_use_best_checkpoint(tmp_path):
    """Rebuilding the net from model_best.pt and re-running the deterministic
    final eval must reproduce report['final_val_det'] exactly."""
    puzzles = synthetic_puzzles()
    config = DRCTrainConfig(
        out_dir=str(tmp_path),
        run_name="drc_best",
        horizon=6,
        curriculum="reverse",
        min_episodes=10,
        seed=1,
        num_envs=4,
        n_steps=16,
        batch_size=32,
        n_epochs=2,
        total_steps=256,
        eval_every=64,
        eval_batch_size=4,
        final_eval_episodes=1,
        drc_depth=1,
        drc_ticks=1,
        drc_hidden_channels=8,
        drc_encoder_channels=(8, 8),
        drc_encoder_kernels=(3, 3),
        drc_mlp_hidden=32,
        device="cpu",
    )
    report = build_and_train(config, puzzles_override=puzzles)
    assert report["report_eval_weights"] == "best"

    payload = torch.load(tmp_path / "drc_best" / "model_best.pt", weights_only=False)
    net = build_net(
        DRCTrainConfig(**payload["config"]), payload["observation_shape"], payload["n_actions"]
    )
    net.load_state_dict(payload["state_dict"])
    env_config = EnvConfig(
        horizon=config.horizon,
        step_penalty=config.step_penalty,
        goal_reward=config.goal_reward,
        shaping_coef=config.shaping_coef,
        gamma=config.gamma,
        time_channel=config.time_channel,
        canonical_identity=config.canonical_identity,
    )
    redo = evaluate_drc(
        net,
        puzzles,  # puzzles_override: final_suite == the training list
        env_config,
        episodes_per_puzzle=1,
        deterministic=True,
        batch_size=config.eval_batch_size,
    )
    assert redo == report["final_val_det"]
