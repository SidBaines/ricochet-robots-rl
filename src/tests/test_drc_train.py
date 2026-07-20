"""CPU-only tests for the recurrent DRC PPO trainer (src/groundup/drc_train.py).

Covers the parts recurrent PPO implementations classically get wrong:
buffer/replay state-reset semantics at episode boundaries, mask storage and
application in replay, GAE against a hand-computed case, truncation
bootstrapping vs termination, end-to-end learn() smoke with the run-directory
contract, and seed determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.groundup.drc import DRCNet, DRCState
from src.groundup.drc_train import (
    DRCTrainConfig,
    RecurrentPPO,
    build_and_train,
    build_net,
    clipped_value_loss,
    compute_gae,
    evaluate_drc,
    normalize_advantages,
    replay_sequences,
)
from src.groundup.env import EnvConfig, RicochetEnv
from src.groundup.puzzles import Puzzle

BANK_8 = Path("artifacts/phase2b_bank/size-8x8_numR2_ET1_CL1_gen10.parquet")


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
                l, v, state = net(obs[t, b : b + 1], state, action_mask=masks[t, b : b + 1])
            assert torch.allclose(logits[t, b], l[0], atol=1e-5), f"t={t} b={b}"
            assert torch.allclose(values[t, b], v[0], atol=1e-5), f"t={t} b={b}"

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
    # SB3 semantics: prediction clamped to old_values +/- clip before the MSE.
    loss = clipped_value_loss(values, old_values, returns, clip_range_vf=0.2)
    expected_pred = torch.tensor([0.1, 0.2, -0.2])
    assert torch.allclose(loss, torch.nn.functional.mse_loss(returns, expected_pred))
    # Gradient flows only where the prediction change is inside the clip.
    loss.backward()
    assert values.grad[0] != 0.0  # |0.1| < 0.2: live
    assert values.grad[1] == 0.0  # clamped: no gradient
    assert values.grad[2] == 0.0  # clamped: no gradient


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
