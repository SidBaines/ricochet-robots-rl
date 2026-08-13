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
import hashlib
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
    _config_contract,
    _load_training_checkpoint,
    _save_training_checkpoint,
    _sha256_json,
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
from src.groundup.evaluate import _aggregate
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


# ------------------------------------- auxiliary landing-cell loss (aux arm)
def test_landing_targets_match_the_stop_planes_the_env_can_render():
    """Aux labels must agree with the env's own slide dynamics, cell for cell."""
    puzzles = synthetic_puzzles()
    plain = RicochetEnv(puzzles, EnvConfig(horizon=8), seed=3)
    withplanes = RicochetEnv(puzzles, EnvConfig(horizon=8, stop_planes=True), seed=3)
    plain.reset(options={"puzzle_index": 2})
    obs, _ = withplanes.reset(options={"puzzle_index": 2})

    targets = plain.landing_targets()
    assert targets.shape == (4 * plain.num_robots,) and targets.dtype == np.int64
    # The observation must be untouched: labels are a TARGET, never an input.
    assert plain.observation_space.shape[0] == 5 + plain.num_robots + 1

    stops = obs[withplanes.stop_planes_offset : withplanes.stop_planes_offset + targets.size]
    for pair, plane in enumerate(stops):
        row, col = divmod(int(targets[pair]), plain.width)
        assert plane[row, col] == 1.0, f"pair {pair} disagrees with the stop plane"
        assert plane.sum() == 1.0


def test_aux_landing_loss_is_bit_identical_to_disabled_when_coef_is_zero():
    """The aux feature must be provably inert when off.

    The control arm A0 ran under a build without this code at all; it is only a
    valid control for A1 if switching the feature off reproduces that build
    exactly. Same seeds, so any leaked RNG draw or extra parameter shows up.
    """
    def run(**extra):
        puzzles = synthetic_puzzles()
        sampler = ReverseFrontierSampler(puzzles, min_episodes=2)
        envs = [
            RicochetEnv(puzzles, EnvConfig(horizon=4), sampler=sampler, seed=700 + r)
            for r in range(2)
        ]
        net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
        torch.manual_seed(31)
        algo = make_algo(net, envs, n_steps=6, batch_size=6, n_epochs=2, seed=5, **extra)
        metrics = algo.update(algo.collect_rollout(), progress_remaining=0.8)
        weights = {k: v.detach().clone() for k, v in algo.net.state_dict().items()}
        return metrics, weights

    off_metrics, off_weights = run()
    zero_metrics, zero_weights = run(aux_landing_coef=0.0)
    assert off_metrics == zero_metrics
    for key, value in zero_weights.items():
        assert torch.equal(value, off_weights[key]), key
    assert off_metrics["aux_landing_loss"] == 0.0


def test_aux_landing_loss_trains_and_reaches_the_shared_trunk():
    """With the coefficient on, the aux loss must fall AND move the trunk."""
    puzzles = synthetic_puzzles()
    sampler = ReverseFrontierSampler(puzzles, min_episodes=2)
    envs = [
        RicochetEnv(puzzles, EnvConfig(horizon=4), sampler=sampler, seed=700 + r)
        for r in range(2)
    ]
    net = tiny_net(envs[0].observation_space.shape, envs[0].action_space.n, seed=9)
    torch.manual_seed(31)
    algo = make_algo(
        net, envs, n_steps=6, batch_size=6, n_epochs=2, seed=5, aux_landing_coef=1.0
    )
    assert algo.aux_head is not None and algo.aux_optimizer is not None
    # The head is NOT part of net.state_dict(), so a checkpoint written without
    # it still round-trips (this is what lets A1 resume the 120M control state).
    assert not any(k.startswith("aux") for k in algo.net.state_dict())

    trunk_before = algo.net.encoder[0].weight.detach().clone()
    first = algo.update(algo.collect_rollout(), progress_remaining=0.8)
    losses = [first["aux_landing_loss"]]
    for _ in range(6):
        losses.append(algo.update(algo.collect_rollout(), progress_remaining=0.8)["aux_landing_loss"])

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"aux loss did not fall: {losses}"
    # chance level for a 12x12 grid is -log(1/64)... the fixture is 8x8 -> 64 cells
    assert losses[0] <= np.log(64) + 1.0
    assert not torch.equal(algo.net.encoder[0].weight, trunk_before), "aux grads never reached the trunk"


def test_config_contract_allows_additive_fields_at_their_default_only():
    """New config fields must not orphan existing checkpoints -- unless actually set.

    Adding aux_landing_coef would otherwise have made every pre-existing
    checkpoint unresumable, including the 120M one both aux arms start from.
    """
    from src.groundup.drc_train import _contract_differences

    current = _config_contract(_RESUME_CONFIG)
    older = {k: v for k, v in current.items() if k != "aux_landing_coef"}
    assert _contract_differences(older, current) == []  # field absent, at default

    enabled = _config_contract(dataclasses.replace(_RESUME_CONFIG, aux_landing_coef=1.0))
    assert _contract_differences(older, enabled) == ["aux_landing_coef"]
    # ... and a genuine mismatch on a shared field is still caught.
    assert "horizon" in _contract_differences({**older, "horizon": 999}, current)
    # The declaration itself is operational, not semantic: it must not move the
    # fingerprint, or every branch would need its own contract.
    declared = dataclasses.replace(_RESUME_CONFIG, resume_allow_contract_change=["x"])
    assert _sha256_json(_config_contract(declared)) == _sha256_json(current)
    # Hard-suite size affects closeout cost/reporting only, never the training
    # trajectory. Existing exact checkpoints must accept any requested size.
    changed_hard_suite = dataclasses.replace(_RESUME_CONFIG, hard_eval_per_stratum=3)
    assert "hard_eval_per_stratum" not in current
    assert _sha256_json(_config_contract(changed_hard_suite)) == _sha256_json(current)


def test_deliberate_contract_change_requires_an_explicit_declaration(tmp_path):
    """Branching an arm off a checkpoint with a changed objective must be declared.

    A1 (aux loss) continues A0's checkpoint with a different objective. That is a
    legitimate branch, but it is NOT an exact resume, so it has to be stated
    rather than slipped through -- otherwise the contract stops meaning anything.
    """
    sampler, algo = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    algo.update(algo.collect_rollout(), progress_remaining=0.9)
    checkpoint = tmp_path / "control.pt"
    _save_training_checkpoint(
        algo, sampler, _RESUME_CONFIG, checkpoint,
        run_state={"elapsed_s": 1.0}, metadata={"test": True},
    )
    branched = dataclasses.replace(_RESUME_CONFIG, aux_landing_coef=1.0)

    sampler2, algo2 = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    with pytest.raises(ValueError, match="resume_allow_contract_change"):
        _load_training_checkpoint(checkpoint, algo=algo2, sampler=sampler2, config=branched)

    declared = dataclasses.replace(branched, resume_allow_contract_change=["aux_landing_coef"])
    sampler3, algo3 = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    _load_training_checkpoint(checkpoint, algo=algo3, sampler=sampler3, config=declared)
    assert algo3.num_timesteps == algo.num_timesteps

    # Declaring one field must not wave through a different one.
    sneaky = dataclasses.replace(
        branched, horizon=_RESUME_CONFIG.horizon + 1,
        resume_allow_contract_change=["aux_landing_coef"],
    )
    sampler4, algo4 = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    with pytest.raises(ValueError, match="horizon"):
        _load_training_checkpoint(checkpoint, algo=algo4, sampler=sampler4, config=sneaky)


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
    # 2026-07-25: restarted from step 0 as _v2. The durable invariant is that the
    # config never resumes the abandoned first attempt, whose rewards were
    # corrupted from step 0 by the truncation bootstrap
    # (docs/Drc87MRestartCollapseInvestigation.md 6.1). resume_from itself is
    # legitimately mutable -- a battery pause sets it -- so pinning it to None
    # made this test fail the moment the run was paused.
    assert config.run_name == "drc33_p1_12x12_planefree_v2"
    assert config.resume_from is None or config.resume_from.startswith(
        "artifacts/groundup/drc33_p1_12x12_planefree_v2/"
    ), f"must not resume the abandoned first attempt: {config.resume_from}"
    # Explicit, so the contract fingerprint pins the reward semantics.
    assert config.bootstrap_on_truncation is False
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
    assert set(det) >= {
        "success_rate",
        "episodes",
        "by_length",
        "by_robots_moved",
        "by_joint_stratum",
    }

    det2 = evaluate_drc(
        net, puzzles, env_config, episodes_per_puzzle=2, deterministic=True, batch_size=3
    )
    assert det == det2  # deterministic protocol is repeatable

    stoch = evaluate_drc(
        net, puzzles, env_config, episodes_per_puzzle=3, deterministic=False, batch_size=4, seed=1
    )
    assert stoch["episodes"] == len(puzzles) * 3
    assert stoch["deterministic"] is False


def test_aggregate_reports_json_stable_joint_stratum_counts_and_rates():
    records = [
        {"success": True, "optimal_length": 1, "robots_moved": 1},
        {"success": False, "optimal_length": 1, "robots_moved": 1},
        {"success": True, "optimal_length": 1, "robots_moved": 2},
        {"success": False, "optimal_length": 2, "robots_moved": 1},
    ]

    joint = _aggregate(records)["by_joint_stratum"]

    assert joint == {
        "1": {
            "1": {"success_rate": 0.5, "n": 2},
            "2": {"success_rate": 1.0, "n": 1},
        },
        "2": {"1": {"success_rate": 0.0, "n": 1}},
    }
    assert json.loads(json.dumps(joint)) == joint


def test_evaluate_drc_leaves_the_callers_rng_untouched():
    """Evaluation must be a pure function of its ``seed``, not a global reseed.

    ``evaluate_drc`` used to call ``torch.manual_seed(seed)``, which reseeds the
    CPU *and* device generators. Because a periodic eval runs immediately before
    every checkpoint, that made the stored ``torch_rng_state`` /
    ``device_rng_state`` the same constant in every checkpoint of the
    drc33_p1_12x12_planefree run, and re-anchored the training action-sampling
    stream to seed 0 every ``eval_every`` steps -- so exploration noise repeated
    instead of continuing. See docs/Drc87MRestartCollapseInvestigation.md 3.2.
    """
    puzzles = synthetic_puzzles()
    env_config = EnvConfig(horizon=6)
    probe = RicochetEnv(puzzles, env_config, seed=0)
    net = tiny_net(probe.observation_space.shape, probe.action_space.n)

    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state().clone()
    mps_before = (
        torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    )

    kwargs = dict(episodes_per_puzzle=3, deterministic=False, batch_size=4)
    first = evaluate_drc(net, puzzles, env_config, seed=5, **kwargs)

    assert torch.equal(torch.get_rng_state(), cpu_before), "eval perturbed the CPU RNG"
    if mps_before is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_before), "eval perturbed the MPS RNG"

    # Still fully reproducible from its own seed, and the seed still drives it.
    assert evaluate_drc(net, puzzles, env_config, seed=5, **kwargs) == first
    # A stochastic eval must actually consume randomness (episode count aside,
    # the recorded step counts differ once the action stream differs).
    assert torch.equal(torch.get_rng_state(), cpu_before)


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


def test_exact_resume_refuses_a_device_change(tmp_path):
    """"Exact" resume is only exact WITHIN a device, so a device change must fail.

    ``device: auto`` resolves to mps or cpu depending on the machine, while the
    config records the unresolved string -- so the config-contract fingerprint is
    identical on an MPS box and a CPU-only box, and a cross-device resume used to
    pass every check while being measurably non-exact (from the same 87M
    checkpoint, CPU gave iteration-1 value_loss 18.1032 vs MPS 19.8086; see
    docs/Drc87MRestartCollapseInvestigation.md 10).
    """
    sampler, algo = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    algo.update(algo.collect_rollout(), progress_remaining=0.9)
    checkpoint = tmp_path / "training.pt"
    _save_training_checkpoint(
        algo, sampler, _RESUME_CONFIG, checkpoint,
        run_state={"elapsed_s": 1.0}, metadata={"test": True},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algo_state"]["device"] == "cpu", "resolved device must be recorded"
    assert payload["algo_state"]["schema_version"] == 2

    # Forge a checkpoint that was written on another device (works without a GPU).
    payload["algo_state"]["device"] = "cuda"
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    sampler2, algo2 = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    with pytest.raises(ValueError, match="device"):
        _load_training_checkpoint(forged, algo=algo2, sampler=sampler2, config=_RESUME_CONFIG)

    # Explicit opt-in still allows the migration.
    algo2.load_state_dict(payload["algo_state"], allow_device_change=True)
    assert algo2.num_timesteps == algo.num_timesteps


def test_schema_1_training_state_is_rejected(tmp_path):
    """Hard break: pre-fix checkpoints must not resume, because they cannot be exact.

    Schema 1 predates both the truncation-bootstrap fix (so its optimizer and
    weights were shaped by rewards inflated with gamma*V(expiry)) and the recorded
    resolved device. Silently accepting one would resume a run whose reward
    semantics differ from the current code's.
    """
    sampler, algo = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    algo.update(algo.collect_rollout(), progress_remaining=0.9)
    checkpoint = tmp_path / "training.pt"
    _save_training_checkpoint(
        algo, sampler, _RESUME_CONFIG, checkpoint,
        run_state={"elapsed_s": 1.0}, metadata={"test": True},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old = dict(payload["algo_state"])
    old["schema_version"] = 1
    old.pop("device")
    payload["algo_state"] = old
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    sampler2, algo2 = _resume_fixture(device="cpu", env_seed=700, net_seed=9, algo_seed=5)
    with pytest.raises(ValueError, match="schema"):
        _load_training_checkpoint(legacy, algo=algo2, sampler=sampler2, config=_RESUME_CONFIG)


def test_truncation_semantics_are_pinned_by_the_config_contract():
    """`bootstrap_on_truncation` must be a contract field, not a derived value.

    While it was derived from ``time_channel`` in code only, a run made before the
    truncation-bootstrap fix and one made after shared a config-contract
    fingerprint despite having different reward semantics. Now the fingerprint
    changes with it.
    """
    base = dataclasses.replace(_RESUME_CONFIG, bootstrap_on_truncation=False)
    flipped = dataclasses.replace(base, bootstrap_on_truncation=None)
    assert "bootstrap_on_truncation" in _config_contract(base)
    assert _sha256_json(_config_contract(base)) != _sha256_json(_config_contract(flipped))
    # ... whereas a genuinely mutable bookkeeping field must NOT move it.
    renamed = dataclasses.replace(base, run_name="something-else")
    assert _sha256_json(_config_contract(base)) == _sha256_json(_config_contract(renamed))


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
    assert curve[0]["val_det"]["provenance"]["source"]["checkpoint_kind"] == "model_latest"
    assert curve[0]["val_det"]["provenance"]["suite"]["name"] == "validation_periodic"
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
        "final_hard_val_det",
        "best_periodic_val_det",
        "best_periodic_val_stoch",
        "best_periodic_hard_val_det",
        "baseline_random",
        "baseline_target_only",
        "sampler_final",
    ):
        assert key in saved_report
    for eval_key in (
        "final_val_det",
        "final_val_stoch",
        "final_hard_val_det",
        "baseline_random",
        "baseline_target_only",
    ):
        summary = saved_report[eval_key]
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["episodes"] > 0
    assert saved_report["final_val_stoch"]["deterministic"] is False
    hard_joint = saved_report["final_hard_val_det"]["by_joint_stratum"]
    hard_cells = [cell for by_robots in hard_joint.values() for cell in by_robots.values()]
    assert sum(cell["n"] for cell in hard_cells) == saved_report["final_hard_val_det"][
        "episodes"
    ]
    assert all(0.0 <= cell["success_rate"] <= 1.0 for cell in hard_cells)

    # Keep-best is a separate comparison. The headline always describes this
    # run's just-saved final checkpoint.
    assert (out / "model_best.pt").exists()
    assert saved_report["report_eval_weights"] == "final"
    assert saved_report["best_val_det"] == curve[-1]["best_val_det"]
    assert saved_report["best_timesteps"] == curve[-1]["best_timesteps"]
    assert saved_report["best_periodic_val_det"] is not None
    assert saved_report["best_periodic_val_stoch"] is not None
    assert saved_report["best_periodic_hard_val_det"] is not None

    for eval_key in (
        "final_val_det",
        "final_val_stoch",
        "final_hard_val_det",
        "best_periodic_val_det",
        "best_periodic_val_stoch",
        "best_periodic_hard_val_det",
    ):
        source = saved_report[eval_key]["provenance"]["source"]
        assert source["source_type"] == "model_checkpoint"
        assert len(source["checkpoint_sha256"]) == 64
        assert Path(source["checkpoint_path"]).is_file()
    for baseline_key in ("baseline_random", "baseline_target_only"):
        source = saved_report[baseline_key]["provenance"]["source"]
        assert source["source_type"] == "scripted_policy"
        assert "checkpoint_sha256" not in source

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


def _reporting_test_config(tmp_path: Path, run_name: str, **changes) -> DRCTrainConfig:
    values = dict(
        out_dir=str(tmp_path),
        run_name=run_name,
        horizon=6,
        curriculum="reverse",
        min_episodes=10,
        seed=1,
        num_envs=1,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        total_steps=64,
        schedule_total_steps=256,
        eval_every=64,
        eval_batch_size=4,
        final_eval_episodes=2,
        hard_eval_per_stratum=1,
        drc_depth=1,
        drc_ticks=1,
        drc_hidden_channels=8,
        drc_encoder_channels=(8, 8),
        drc_encoder_kernels=(3, 3),
        drc_mlp_hidden=32,
        device="cpu",
    )
    values.update(changes)
    return DRCTrainConfig(**values)


def _install_checkpoint_reporting_fakes(monkeypatch):
    """Make periodic and final weights observably different without training."""
    import src.groundup.drc_train as drc_train

    def fill(net, value):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.fill_(value)

    def fake_learn(self, total_steps, on_iteration=None, *, schedule_total_steps=None):
        # Fresh parent: create a periodic best at 0.2, then a distinct final at
        # 0.8. Resumed children deliberately run no periodic evaluation.
        if self.num_timesteps == 0:
            fill(self.net, 0.2)
            self.num_timesteps = 64
            if on_iteration is not None:
                on_iteration(self, {"fake_loss": 0.0})
        fill(self.net, 0.8)
        self.num_timesteps = total_steps

    def fake_evaluate(
        net,
        puzzles,
        config,
        *,
        episodes_per_puzzle=1,
        deterministic=True,
        batch_size=32,
        seed=0,
    ):
        marker = round(float(next(net.parameters()).detach().reshape(-1)[0]), 4)
        return {
            "success_rate": marker,
            "episodes": len(puzzles) * episodes_per_puzzle,
            "by_length": {},
            "by_robots_moved": {},
            "deterministic": deterministic,
        }

    def fake_baseline(puzzles, config, *, policy, episodes_per_puzzle=5, seed=0):
        return {
            "success_rate": 0.0,
            "episodes": len(puzzles) * episodes_per_puzzle,
            "by_length": {},
            "by_robots_moved": {},
            "policy": policy,
        }

    monkeypatch.setattr(drc_train.RecurrentPPO, "learn", fake_learn)
    monkeypatch.setattr(drc_train, "evaluate_drc", fake_evaluate)
    monkeypatch.setattr(drc_train, "scripted_baseline", fake_baseline)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_immutable_eval_checkpoint_is_idempotent_and_collision_safe(tmp_path):
    from src.groundup.drc_train import _materialize_immutable_evaluation_checkpoint

    mutable_alias = tmp_path / "model_latest.pt"
    mutable_alias.write_bytes(b"first evaluated checkpoint")

    first = _materialize_immutable_evaluation_checkpoint(
        mutable_alias,
        source_timesteps=123,
    )
    repeated = _materialize_immutable_evaluation_checkpoint(
        mutable_alias,
        source_timesteps=123,
    )
    assert repeated == first
    assert first.read_bytes() == b"first evaluated checkpoint"

    mutable_alias.write_bytes(b"different checkpoint at the same timestep")
    collision = _materialize_immutable_evaluation_checkpoint(
        mutable_alias,
        source_timesteps=123,
    )
    assert collision != first
    assert collision.read_bytes() == b"different checkpoint at the same timestep"
    assert first.read_bytes() == b"first evaluated checkpoint"


def test_final_headline_uses_model_final_and_best_periodic_is_separate(
    tmp_path, monkeypatch
):
    _install_checkpoint_reporting_fakes(monkeypatch)
    config = _reporting_test_config(tmp_path, "report_sources")

    report = build_and_train(config, puzzles_override=synthetic_puzzles())
    out = tmp_path / config.run_name

    assert report["report_eval_weights"] == "final"
    assert report["final_val_det"]["success_rate"] == 0.8
    assert report["final_val_stoch"]["success_rate"] == 0.8
    assert report["final_hard_val_det"]["success_rate"] == 0.8
    assert report["best_periodic_val_det"]["success_rate"] == 0.2
    assert report["best_periodic_val_stoch"]["success_rate"] == 0.2
    assert report["best_periodic_hard_val_det"]["success_rate"] == 0.2

    final_source = report["final_val_det"]["provenance"]["source"]
    assert final_source == report["final_val_stoch"]["provenance"]["source"]
    assert final_source["originating_run_name"] == config.run_name
    assert final_source["source_timesteps"] == 64
    assert final_source["checkpoint_kind"] == "model_final"
    final_path = Path(final_source["checkpoint_path"])
    assert final_path.parent == out.resolve()
    assert final_path.name.startswith("model_eval_64_")
    assert final_path != (out / "model_final.pt").resolve()
    assert final_source["checkpoint_sha256"] == _sha256_file(final_path)

    best_source = report["best_periodic_val_det"]["provenance"]["source"]
    assert best_source["originating_run_name"] == config.run_name
    assert best_source["source_timesteps"] == 64
    assert best_source["checkpoint_kind"] == "model_latest"
    best_path = Path(best_source["checkpoint_path"])
    assert best_path.parent == out.resolve()
    assert best_path.name.startswith("model_eval_64_")
    assert best_path != (out / "model_best.pt").resolve()
    assert best_source["checkpoint_sha256"] == _sha256_file(best_path)
    assert best_source != final_source

    suites = report["provenance"]["evaluation_suites"]
    selection = report["best_selection_provenance"]
    assert selection["source"] == best_source
    assert selection["source"]["checkpoint_sha256"] == _sha256_file(best_path)
    assert selection["suite"] == suites["validation_periodic"]
    assert report["final_val_det"]["provenance"]["suite"] == suites["validation_final"]
    assert report["final_hard_val_det"]["provenance"]["suite"] == suites["validation_hard"]
    assert suites["validation_final"]["fingerprint"]
    assert suites["validation_hard"]["fingerprint"]
    assert suites["validation_hard"]["seed"] == 9

    for key in ("baseline_random", "baseline_target_only"):
        source = report[key]["provenance"]["source"]
        assert source["source_type"] == "scripted_policy"
        assert "checkpoint_path" not in source
        assert "checkpoint_sha256" not in source


def test_score_provenance_survives_later_same_directory_saves(tmp_path, monkeypatch):
    import src.groundup.drc_train as drc_train

    _install_checkpoint_reporting_fakes(monkeypatch)
    puzzles = synthetic_puzzles()
    config = _reporting_test_config(tmp_path, "continued_in_place")
    first_report = build_and_train(config, puzzles_override=puzzles)
    out = tmp_path / config.run_name
    first_curve = [json.loads(line) for line in (out / "curve.jsonl").read_text().splitlines()]

    sources = [first_curve[0]["val_det"]["provenance"]["source"]]
    sources.extend(
        first_report[key]["provenance"]["source"]
        for key in (
            "final_val_det",
            "final_val_stoch",
            "final_hard_val_det",
            "best_periodic_val_det",
            "best_periodic_val_stoch",
            "best_periodic_hard_val_det",
        )
    )
    sources.append(first_report["best_selection_provenance"]["source"])
    original_sources = {
        (source["checkpoint_path"], source["checkpoint_sha256"])
        for source in sources
    }
    assert len(original_sources) == 2  # periodic/selected-best and final bytes
    for path_text, expected_sha in original_sources:
        path = Path(path_text)
        assert path.name.startswith("model_eval_64_")
        assert _sha256_file(path) == expected_sha

    def fill(net, value):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.fill_(value)

    def continue_with_new_periodic_and_final(
        self, total_steps, on_iteration=None, *, schedule_total_steps=None
    ):
        fill(self.net, 0.6)
        self.num_timesteps = 128
        if on_iteration is not None:
            on_iteration(self, {"fake_loss": 0.0})
        fill(self.net, 0.9)
        self.num_timesteps = total_steps

    monkeypatch.setattr(
        drc_train.RecurrentPPO,
        "learn",
        continue_with_new_periodic_and_final,
    )
    continued = dataclasses.replace(
        config,
        total_steps=128,
        resume_from=str(out / "training_final.pt"),
    )
    build_and_train(continued, puzzles_override=puzzles)

    historical_report = json.loads((out / "report_64.json").read_text())
    historical_curve = json.loads((out / "curve.jsonl").read_text().splitlines()[0])
    assert historical_report["final_val_det"]["provenance"]["source"] == first_report[
        "final_val_det"
    ]["provenance"]["source"]
    assert historical_curve["val_det"]["provenance"]["source"] == first_curve[0][
        "val_det"
    ]["provenance"]["source"]
    for path_text, expected_sha in original_sources:
        path = Path(path_text)
        assert path.is_file()
        assert _sha256_file(path) == expected_sha


@pytest.mark.parametrize("hard_eval_per_stratum", [0, -1])
def test_build_rejects_non_positive_hard_eval_per_stratum(
    tmp_path, monkeypatch, hard_eval_per_stratum
):
    _install_checkpoint_reporting_fakes(monkeypatch)
    config = _reporting_test_config(
        tmp_path,
        f"invalid_hard_{hard_eval_per_stratum}",
        hard_eval_per_stratum=hard_eval_per_stratum,
    )

    with pytest.raises(ValueError, match="hard_eval_per_stratum must be positive"):
        build_and_train(config, puzzles_override=synthetic_puzzles())


def test_declared_branch_resets_parent_best_tracking(tmp_path, monkeypatch):
    _install_checkpoint_reporting_fakes(monkeypatch)
    puzzles = synthetic_puzzles()
    parent_config = _reporting_test_config(tmp_path, "parent")
    build_and_train(parent_config, puzzles_override=puzzles)
    parent_checkpoint = tmp_path / "parent" / "training_final.pt"

    branch_config = _reporting_test_config(
        tmp_path,
        "branch",
        total_steps=128,
        resume_from=str(parent_checkpoint),
        aux_landing_coef=1.0,
        resume_allow_contract_change=["aux_landing_coef"],
    )
    report = build_and_train(branch_config, puzzles_override=puzzles)

    assert report["report_eval_weights"] == "final"
    assert report["final_val_det"]["success_rate"] == 0.8
    assert report["best_val_det"] is None
    assert report["best_timesteps"] is None
    assert report["best_source"] is None
    assert report["best_selection_provenance"] is None
    assert report["best_periodic_val_det"] is None
    assert report["best_periodic_val_stoch"] is None
    assert report["best_periodic_hard_val_det"] is None
    assert not (tmp_path / "branch" / "model_best.pt").exists()


def test_declared_branch_rejects_parent_output_directory_without_deleting_best(
    tmp_path, monkeypatch
):
    _install_checkpoint_reporting_fakes(monkeypatch)
    puzzles = synthetic_puzzles()
    parent_config = _reporting_test_config(tmp_path, "same_directory_parent")
    build_and_train(parent_config, puzzles_override=puzzles)
    parent_dir = tmp_path / parent_config.run_name
    parent_checkpoint = parent_dir / "training_final.pt"
    parent_best = parent_dir / "model_best.pt"
    best_sha = _sha256_file(parent_best)
    config_sha = _sha256_file(parent_dir / "config.json")

    branch_config = dataclasses.replace(
        parent_config,
        total_steps=128,
        resume_from=str(parent_checkpoint),
        aux_landing_coef=1.0,
        resume_allow_contract_change=["aux_landing_coef"],
    )
    with pytest.raises(
        ValueError,
        match="semantic branch.*same resolved output directory",
    ):
        build_and_train(branch_config, puzzles_override=puzzles)

    assert parent_best.is_file()
    assert _sha256_file(parent_best) == best_sha
    assert _sha256_file(parent_dir / "config.json") == config_sha


def test_exact_continuation_preserves_and_reconstructs_legacy_best_source(
    tmp_path, monkeypatch
):
    _install_checkpoint_reporting_fakes(monkeypatch)
    puzzles = synthetic_puzzles()
    parent_config = _reporting_test_config(
        tmp_path,
        "legacy_parent",
        checkpoint_every=64,
    )
    build_and_train(parent_config, puzzles_override=puzzles)
    parent_dir = tmp_path / "legacy_parent"

    # Simulate the currently deployed outer schema-1 checkpoint: best weights
    # are embedded in run_state, but best-source metadata predates this repair.
    payload = torch.load(parent_dir / "training_final.pt", map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    legacy_model = torch.load(parent_dir / "model_best.pt", weights_only=False)
    assert legacy_model["schema_version"] == 2
    legacy_model["config"].pop("hard_eval_per_stratum")
    torch.save(legacy_model, parent_dir / "model_best.pt")
    payload["config"].pop("hard_eval_per_stratum")
    payload["metadata"].pop("evaluation_suites", None)
    payload["run_state"].pop("best_source", None)
    payload["run_state"].pop("best_selection_provenance", None)
    legacy_checkpoint = parent_dir / "legacy_training.pt"
    torch.save(payload, legacy_checkpoint)

    continued_config = _reporting_test_config(
        tmp_path,
        "exact_child",
        total_steps=128,
        resume_from=str(legacy_checkpoint),
        hard_eval_per_stratum=2,  # evaluation-only and therefore resume-mutable
    )
    report = build_and_train(continued_config, puzzles_override=puzzles)

    assert report["final_val_det"]["success_rate"] == 0.8
    assert report["best_periodic_val_det"]["success_rate"] == 0.2
    source = report["best_periodic_val_det"]["provenance"]["source"]
    assert source["originating_run_name"] == parent_config.run_name
    assert source["source_timesteps"] == 64
    assert source["checkpoint_kind"] == "periodic"
    assert Path(source["checkpoint_path"]) == (parent_dir / "model_64.pt").resolve()
    assert source["checkpoint_sha256"] == _sha256_file(parent_dir / "model_64.pt")
    assert source["provenance_reconstructed"] is True
    selection = report["best_selection_provenance"]
    assert selection["suite"]["fingerprint"] == report["provenance"][
        "evaluation_suites"
    ]["validation_periodic"]["fingerprint"]
    assert selection["provenance_reconstructed"] is True


def test_legacy_best_fallback_materializes_standalone_immutable_source(
    tmp_path, monkeypatch
):
    _install_checkpoint_reporting_fakes(monkeypatch)
    puzzles = synthetic_puzzles()
    parent_config = _reporting_test_config(tmp_path, "legacy_fallback_parent")
    build_and_train(parent_config, puzzles_override=puzzles)
    parent_dir = tmp_path / parent_config.run_name

    # Remove provenance as in the legacy schema, use the mutable latest file as
    # the resume source, and make model_best tensor-incompatible. The embedded
    # best_state_dict is then the only trustworthy copy of the selected state.
    legacy_path = parent_dir / "training_latest.pt"
    legacy_payload = torch.load(legacy_path, map_location="cpu", weights_only=False)
    legacy_payload["run_state"].pop("best_source", None)
    legacy_payload["run_state"].pop("best_selection_provenance", None)
    torch.save(legacy_payload, legacy_path)
    torch.save(
        torch.load(parent_dir / "model_final.pt", map_location="cpu", weights_only=False),
        parent_dir / "model_best.pt",
    )

    child_config = _reporting_test_config(
        tmp_path,
        "legacy_fallback_child",
        total_steps=128,
        resume_from=str(legacy_path),
    )
    report = build_and_train(child_config, puzzles_override=puzzles)
    source = report["best_periodic_val_det"]["provenance"]["source"]
    source_path = Path(source["checkpoint_path"])

    assert source_path.parent == (tmp_path / "legacy_fallback_child").resolve()
    assert source_path.name.startswith("model_eval_64_")
    assert source["checkpoint_kind"] == "legacy_best_eval_source"
    assert source["checkpoint_sha256"] == _sha256_file(source_path)
    assert source["provenance_reconstructed"] is True
    assert torch.load(source_path, map_location="cpu", weights_only=False)["kind"] == "drc_model"

    legacy_path.write_bytes(b"later mutable training save")
    (parent_dir / "model_best.pt").write_bytes(b"later mutable best save")
    assert _sha256_file(source_path) == source["checkpoint_sha256"]
