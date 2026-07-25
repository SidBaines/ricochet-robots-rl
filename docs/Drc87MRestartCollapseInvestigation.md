# Did the 87M restart cause the DRC value/policy collapse?

Independent adversarial investigation commissioned by
`docs/Drc87MRestartCollapseInvestigationBrief.md`.

- Run: `artifacts/groundup/drc33_p1_12x12_planefree` (W&B `pupnnpq8`)
- Investigated at repo commit `29a0426`, branch `rl-ground-up`, working tree clean
- Device `mps`, torch 2.13.0, numpy 2.4.6, Python 3.11.15, macOS 26.5.2 arm64
- Forensic outputs: `artifacts/forensics/drc87m_restart/` (gitignored)
- The sealed test split was **not** opened. The frozen trainer (PID 86950, state
  `T+`) was **not** signalled. No primary artifact was written, renamed or deleted.

---

## 1. Conclusion

**The restart did not cause, accelerate, or expose the collapse. It was
bit-exactly inert.** The collapse was a deterministic instability that was
already running away before the pause, driven by a value-target scale runaway
that had been underway since ~82M, and it would have produced the same
89M collapse had the process never been interrupted.

Confidence: **very high** on "the restart is not causal" — this is not a
statistical argument. I established it by direct state-level experiment:

1. A process restart across a training-state checkpoint reproduces the
   uninterrupted trajectory **bit-for-bit** on this machine/device/code/config
   (1,520 field-value comparisons, 0 differences; §4).
2. The training loop is **bit-deterministic on MPS** here — five independent
   processes resumed from the same checkpoint produced identical parameter and
   optimizer digests at every iteration (§4).
3. The harness that produced (1) and (2) **reproduces the production run
   exactly**: replaying the full 87M→88M window in a fresh process lands on a
   bit-identical `state_dict` to `probe_model_88006656.pt` with every metric
   equal to the last digit (§4.1). So (1) and (2) are statements about the real
   trajectory, not about a lookalike.

Given (1)–(3), the resumed 87M→91M trajectory *is* the uninterrupted
counterfactual, exactly. There is no counterfactual left to run: the proposed
Tier-3 "Arm U vs Arm R" replay from 80M would answer with a similarity
judgement a question that (1) already answers with bit-equality.

Against the brief's five candidate explanations:

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | Checkpoint/save/load or restart bug | **Excluded.** Restore is field-complete and bit-exact; restart is provably inert (§3, §4) |
| 2 | Deterministic instability already underway before the pause | **Confirmed. This is the cause** (§5, §6) |
| 3 | Stochastic instability whose trajectory the restart changed | **Excluded.** No stochasticity to change: the loop is bit-deterministic, and the checkpointed RNG is provably the state the uninterrupted process was in (§3.2, §4) |
| 4 | Monitoring/evaluation artefact | **Excluded.** 15/15 independent re-evaluations reproduce the recorded `val_det` exactly, and the training-time curriculum EMAs collapsed too (§5) |
| 5 | Another operational interaction | **Excluded for the collapse window.** Machine slept 16:37→21:08 (before the resume) and again from 21:49 (after the collapse); the 88M→89M collapse window ran awake at full throughput (§7) |

**What actually broke it — a bug, found while writing this up (§6.1).** At
horizon expiry `env.step` returns an observation whose time channel is exactly
0. `collect_rollout` feeds that observation to the value head and adds
`0.99 × V` straight into the reward — but then discards it, so it is never a
regression target. `remaining == 0` therefore appears in **no** stored
observation (the 87M checkpoint's per-env `steps` histogram spans 0…23, never
24), while `time_channel: true` means the MDP is finite-horizon and fully
observed, so `V(s, remaining=0) ≡ 0` by definition. Measured against a
`goal_reward` of 1.0, it was **+0.71 at 80M, −12.3 at 87M, −139 at 91M** — and
never once near 0 anywhere in the run. Timing out was worth more than solving
the puzzle. That is the error-injection channel, it has no contraction
guarantee, and §6's `max_grad_norm=0.5` coupling is what converted it into
policy death: the policy's share of the update fell 98.8% → 3.7% (80M → 87M) →
0.14% (90M).

The bootstrap was never in the reference path: `src/groundup/env.py` emits no
`TimeLimit.truncated`, so the feed-forward P1 baseline's `MaskablePPO` correctly
treats expiry as terminal. `drc_train.py` hand-rolled the bootstrap to match
SB3 semantics it was never actually getting. **This is a DRC-trainer regression,
and the fix is to delete the bootstrap** (§11.2, Arm C).

---

## 2. Evidence table

### Verified facts (measured in this investigation)

| # | Fact | How verified |
|---|---|---|
| F1 | All four checkpoint SHA-256s quoted in the brief were recomputed and match exactly; no mismatch anywhere | `shasum -a 256` (§8) |
| F2 | `training_pause_87007232.pt` metadata: `t=87,007,232`, `kind=training_latest`, commit `fede440`, clean | payload audit |
| F3 | The 87M checkpoint is at a **true iteration boundary** and internally coherent: `episode_start == (env.steps == 0)` for all 256 envs, 0 envs with `episode_done=True`, all tensors finite | payload audit |
| F4 | Adam `step`=42,480 for all 25 params = 10,620 iterations × 4 minibatches × 1 epoch — i.e. **no gradient update was ever skipped** by `target_kl` in the entire run | payload audit + arithmetic |
| F5 | Optimizer steps/LRs at 87/90/91M are 42,480/43,941/44,429 and 1.9124e-4/1.8749e-4/1.8624e-4, matching the registered 200M schedule | payload audit |
| F6 | `torch_rng_state` and `device_rng_state` are **the same constant in all five checkpoints** (80/85/87/90/91M), equal to a fresh `torch.manual_seed(0)`, because `evaluate_drc` reseeds CPU+MPS RNG at every eval and checkpoints are written right after | digest comparison + direct reproduction (§3.2) |
| F7 | Restore is field-complete: 16/16 fidelity checks pass, and **no** digested field is left at its fresh-construction value | `forensic_replay.py` restore-fidelity block, all 9 harness runs |
| F8 | The training loop is **bit-deterministic on MPS**: two independent processes from the same checkpoint agree on 760 field-values over 20 iterations, 0 differences | Arm U1 vs U2 (§4) |
| F9 | **A process restart is bit-exactly inert**: splitting a 20-iteration run into 10+10 across two processes reproduces the continuous run exactly (380 field-values, 0 differences) | Arm R vs Arm U (§4) |
| F10 | The mid-arm checkpoint used for F9 was written at a **non-eval** boundary, so its MPS RNG was not the seed-0 constant — the harder round-trip was exercised and still exact | digest comparison (§4) |
| F10b | Replaying the **full** 87M→88M window (122 iterations) in a fresh process reproduces the production `probe_model_88006656.pt` **bit-identically** (digest `b37f47e638b35fb0`), with `value_loss`, `policy_loss`, `entropy`, `approx_kl`, `approx_kl_max`, `clip_fraction`, `explained_variance`, `lr` and `n_grad_updates` all equal to the last digit | §4.1 |
| F10c | `model_best.pt` (t=79,003,648) is bit-identical to `probe_model_79003648.pt` **and** to the `best_state_dict` inside all five checkpoints' `run_state`, and independently re-evaluates to `val_det` 0.830 with matching per-length breakdown | three-way digest + re-eval |
| F11 | 15/15 retained snapshots (65M…91M) re-evaluate to **exactly** the recorded `val_det` on the fixed 200-puzzle suite, whose fingerprint matches `741960337e…` | `reeval_and_vscale.py` |
| F12 | Value-head output scale on a fixed 16k-state set grows smoothly and monotonically across the pause: `V_std` 0.153 (80M) → 3.773 (87M) → **3.880 (88M)** → 21.76 (89M); `V_mean` crosses zero **before** the pause (+0.21 at 86M → −2.66 at 87M) | `reeval_and_vscale.py` |
| F13 | The 88M window is the **flattest** window of the whole escalation (`value_loss` 19.18→16.01, `V_std` +2.8%) — the opposite of a restart-induced kink | F11/F12 tables |
| F14 | Policy-gradient share of the update: 98.8% (80M) → 64.1% (85M) → **3.7% (87M)** → 0.14% (90M); effective post-clip policy-gradient magnitude 0.494 → 0.0184 → 0.00068 | `--grad-probe` (§6) |
| F15 | All 7 relevant source blobs (`drc_train.py`, `drc.py`, `env.py`, `curriculum.py`, `puzzles.py`, `distance.py`, `evaluate.py`) are **byte-identical** at `1964f7d`, `fede440`, `96fcbff`, `2817c10` and `HEAD` | `git rev-parse <commit>:<path>` (§8) |
| F16 | `fede440..96fcbff` touched only the config's `resume_from` line, the probe daemon, `drc_probe_wandb.py`, a probe test and docs — no trainer/env/net/curriculum change | `git diff --stat` |
| F17 | All 18 snapshot-manifest rows share one contract fingerprint, one split-fingerprint set, one `bank_sha256`, and none is `dirty` | manifest audit |
| F18 | The probe subsystem writes only `probe_control.json`, `.drc_probe_wandb_state.json`, `probe_model_*.pt` copies and `drc_probe_*` outputs. It never writes any trainer input or output, and copies `model_latest.pt` via `open("xb")` + `os.replace` | source grep of all write sites |
| F19 | `best_state_dict` (t=79,003,648, `val_det` 0.830) is bit-identical in all five checkpoints, i.e. the best weights survived the resume untouched | digest comparison |
| F20 | Machine slept 16:37:45→21:08:23 (battery 5%→1%, `Clamshell`/`Low Power`/`Hibernate`). The 88M→89M collapse window (21:20:14→21:29:55) ran awake at 9.6 min/1M — identical to the pre-pause 86M→87M window (9.8 min/1M) | `pmset -g log` + artifact mtimes + `curve.jsonl` elapsed (§7) |
| F21 | `_dmap_cache` is not serialized, but all 39,713 train puzzles have unique `seed`s, so it is a pure memoization and inert across restarts | uniqueness check |
| **F22** | **`V` at horizon expiry (time channel = 0), which is ≡0 by definition, was never near 0 in the entire run: −1.87 (5M), +0.35 (70M), **+0.71 (80M)**, −1.25 (85M), **−12.28 (87M)**, −91.9 (89M), −139.0 (91M) — against `goal_reward` 1.0. Its magnitude tracks `value_loss` across the whole run** | §6.1, fixed 16k probe set with time channel zeroed |
| **F23** | **That observation is never a regression target and `remaining == 0` appears in no stored observation** — the 87M checkpoint's per-env `steps` histogram spans 0…23, never 24 — yet `0.99 × V` there is added to the reward on every truncation | code read + checkpoint data |
| **F24** | **The bootstrap has no basis in the reference path**: `src/groundup/env.py` contains zero occurrences of `TimeLimit`, and SB3 bootstraps only when the info dict carries both `terminal_observation` and `TimeLimit.truncated` (`on_policy_algorithm.py:236–245`). The P1 feed-forward baseline therefore never bootstrapped; `drc_train.py:586` hand-rolled it citing SB3 semantics it never received | source read |

### Inferences (well-supported, but reasoning not measurement)

| # | Inference | Basis |
|---|---|---|
| I1 | The resumed 87M→91M trajectory **is** the uninterrupted counterfactual, bit-for-bit | F8 + F9 + F10b by induction: if state(restart + k iters) = state(continuous + k iters) exactly, the loop is deterministic, and the harness provably matches production, then all later iterations coincide |
| I2 | The interrupted pre-pause process's discarded partial rollout is irrelevant | The SIGINT landed in the *next* iteration's `update()` (traceback), after `on_iteration` had written the checkpoint; in-memory mutations were never persisted (F3 confirms the saved state is pre-rollout) |
| I3 | The collapse magnitude at 89–91M is what the **pre-pause** trend predicted | log-linear fit of `V_std` on 82M–87M only (slope 0.250 decades/1M, R²=0.952) extrapolates to 4.75/8.45/15.0/26.7; observed 3.88/21.8/28.4/29.3 — within 2.6× throughout and within 10% by 91M. The same fit on the noisier `value_loss` brackets the observations within 3.5× (§5) |
| I4 | The mechanism is value-scale runaway → gradient-budget starvation of the policy, not an exploration or curriculum failure | F14 + `approx_kl` staying ≤0.014 and `explained_variance` staying 0.81–0.84 throughout: the critic keeps *relative* fit while its absolute scale diverges |
| I5 | System sleep did not corrupt the checkpoint | It was written 16:29:20, ~8 min *before* the 16:37:45 sleep, and F3/F7 show it restores coherently |

### Unresolved / limits of this investigation

| # | Open question |
|---|---|
| U1 | MPS bit-determinism is **verified empirically for this workload** (6 processes, 122 iterations), not guaranteed by PyTorch. A different kernel mix, OS/driver update or batch shape could reintroduce nondeterminism. I found no documented determinism contract for MPS. |
| U2 | CPU and MPS produce **different** trajectories from the same checkpoint (CPU iter-1 `value_loss` 18.1032 / net `e5df3b6c…` vs MPS 19.8086 / `a4f19a65…`). Exact resume is exact *within* a device, never across devices. A production-scale CPU A/B was started and abandoned as too slow (>10 min/iteration); CPU restart exactness is instead covered by a new unit test (§9). |
| U3 | ~~Why the value runaway started around 82M is not established here.~~ **Resolved after the first draft — see §6.1.** Nothing changed at 82M because nothing needed to: the error-injection channel was present from step 0 and merely crossed into runaway. What remains untested is whether removing it *fully* stabilises the run (§11.2, Arm C). |
| U4 | I did not determine whether the runaway is arrestable by `clip_range_vf` — that is a separate arm (§11). |
| U5 | I did not verify W&B server-side row integrity; `curve.jsonl` is the authority used here and it is corroborated by independent re-evaluation (F11). |

---

## 3. Checkpoint and restore audit

### 3.1 The 87M checkpoint is coherent and at a true boundary

`_save_training_checkpoint` is called from `on_iteration`, which runs *after*
`update()` — so the persisted `(obs, episode_start, recurrent_state, env_states,
sampler, num_timesteps)` tuple is the post-update, pre-next-rollout state. The
audit confirms this is internally consistent (F3): `episode_start[b]` is true
exactly where `env[b].steps == 0`, for all 256 envs, with no mismatches and no
env left mid-`episode_done`. `env_states` carries `puzzle_index`, canonicalised
`robots`, `steps`, `start_distance`, `episode_done` and a deep-copied PCG64
`rng_state` per env — the complete mutable episode state.

The SIGINT traceback shows interruption inside
`update() → replay_sequences → ConvLSTMCell.forward` for the *following*
iteration. That iteration's `collect_rollout()` had already mutated the live
objects, but nothing was written after 16:29:20, so those mutations are absent
from the file (I2). `training_pause_87007232.pt` now has link count 1: the hard
link did its job, `training_latest.pt` was later `os.replace`d onto a new inode,
and the 87M inode survived untouched.

### 3.2 The checkpointed RNG is a constant — and that is not a bug here

`evaluate_drc` (`src/groundup/drc_train.py:756`) calls `torch.manual_seed(seed)`
with `seed=0` by default, and `torch.manual_seed` reseeds **both** the CPU and
the MPS generator. Periodic evaluation is deterministic (`argmax`, no sampling)
and consumes no further RNG. Every checkpoint is written immediately after an
evaluation. Consequently:

```
stored torch_rng_state  digest = 1ccf17250133dec5   in ALL of 80M/85M/87M/90M/91M
stored device_rng_state digest = c29a1d9c25f640eb   in ALL of 80M/85M/87M/90M/91M
fresh torch.manual_seed(0) -> cpu 1ccf17250133dec5, mps c29a1d9c25f640eb   (reproduced)
```

This kills the entire RNG-restoration class of restart hypotheses for this run:
the value being restored is provably the value the uninterrupted process held at
that instant. Note that on MPS the *action* stream comes from
`Categorical.sample() → torch.multinomial` on the device, i.e. from the MPS
generator — so the MPS branch is the one that matters, and it is captured.

Only `minibatch_generator_state` (the dedicated `torch.Generator` driving
`torch.randperm`) is genuinely stateful across evals; it differs in every
checkpoint and is serialized and restored explicitly.

**Incidental finding (not a cause here).** Because the global generators are
re-anchored to seed 0 every `eval_every` steps, the action-sampling noise stream
restarts from the same point every ~1M steps rather than continuing. That
correlates exploration noise across evaluation periods. It is scientifically
untidy and worth fixing (pass an eval-local generator instead of reseeding the
global one), but it is identical before and after the pause and cannot explain
the collapse.

### 3.3 Restore fidelity

`forensic_replay.py` reconstructs exactly the object graph `build_and_train`
builds (same `load_puzzles` → `ReverseFrontierSampler` →
`torch.manual_seed(seed)` → 256 `RicochetEnv` → `build_net` → `RecurrentPPO`
order), digests the fresh state, restores, and digests again. In all six runs:

```
restore fidelity: num_timesteps ✓ net ✓ obs ✓ episode_start ✓ rec_h ✓ rec_c ✓
                  envs ✓ sampler ✓ minibatch_gen ✓ torch_rng ✓ mps_rng ✓
                  adam_step ✓ opt ✓ live_global_torch_rng ✓ live_numpy_rng ✓
                  live_python_rng ✓                                      (16/16)
fields identical to a FRESH (unrestored) process: []
```

The second line matters: **every** digested field changed on restore, so nothing
silently kept its construction-time value. `RecurrentPPO.__init__` resets all 256
envs (consuming env RNG and mutating the sampler) before the checkpoint is
loaded, and `_load_training_checkpoint` overwrites all of it —
`sampler.load_state_dict` then `algo.load_state_dict`, with the global RNG
restored *after* `env.load_state_dict`, which is the correct order.

---

## 4. The controlled restart experiment (the decisive result)

Rather than the brief's 11M-step Arm U / Arm R replay from 80M, I ran the same
counterfactual **at the state actually in question**, at a scale where the answer
is bit-equality rather than a similarity judgement. Three arms, all from
`training_pause_87007232.pt`, all on `mps`, unmodified config, W&B never
initialised, outputs confined to `artifacts/forensics/drc87m_restart/`:

- **Arm U1** — 20 iterations, one process.
- **Arm U2** — identical repeat in a fresh process (tests MPS determinism).
- **Arm R** — 10 iterations, write a training-state checkpoint, **exit**; then a
  **new process** loads that checkpoint and runs 10 more.

After every iteration the harness digests: net `state_dict`, full optimizer state
(`step`, `exp_avg`, `exp_avg_sq`, param groups), `obs`, `episode_start`,
recurrent `h`/`c`, all 256 env states, sampler EMA/counts, minibatch generator,
CPU RNG, MPS RNG, plus all PPO metrics and rollout statistics.

```
1. MPS determinism      ArmU1 vs ArmU2                      760 field-values, 20 iters -> IDENTICAL (0 differences)
2. Restart inertness    ArmU1[1..10]  vs ArmR1[1..10]       380 field-values, 10 iters -> IDENTICAL (0 differences)
                        ArmU1[11..20] vs ArmR2[1..10]       380 field-values, 10 iters -> IDENTICAL (0 differences)
                                                            (ArmR2 = post-restart, fresh interpreter)
```

**First divergent state or random draw: none.** There is no divergence to
locate. This is not "MPS nondeterminism was small" — it is zero across every
digested tensor, at every iteration, including Adam moments and all 256
per-environment PCG64 states.

Arm R's mid-arm checkpoint was deliberately written at a **non-eval** boundary,
so its MPS RNG (`f96639d92dc6b9fd`) was *not* the seed-0 constant
(`c29a1d9c25f640eb`) — the genuinely stateful round-trip was exercised (F10).
(The CPU RNG was still the constant, because nothing in the training loop
consumes the CPU global generator on MPS.)

**Five** independent processes started from `training_pause_87007232.pt` (U1, U2,
R1, the `--grad-probe` 87M arm, and the 122-iteration validation of §4.1) all
produced iteration-1 `net=a4f19a65d297fd28`, `opt=828a79f403caf705`,
`value_loss=19.8086` — and R2, the sixth, continued that trajectory across a
process boundary without deviation.

### 4.1 Production validation of the replay harness

A restart A/B is only evidence about the real run if the harness reproduces the
real run. So I replayed the **entire** 87M→88M window — 122 iterations
(exactly 999,424 steps) in a single fresh process, no evaluation, no W&B — and
compared its endpoint against the artifacts the resumed production trainer wrote
at t=88,006,656 on 2026-07-24 at 21:20:14:

```
PRODUCTION 88M ARTIFACT  vs  INDEPENDENT REPLAY FROM THE 87M PAUSE CHECKPOINT
  timesteps              production=88006656               replay=88006656               MATCH
  net state_dict digest  production=b37f47e638b35fb0       replay=b37f47e638b35fb0       MATCH
  value_loss             production=16.011714935302734     replay=16.011714935302734     MATCH
  policy_loss            production=0.003758310107514262   replay=0.003758310107514262   MATCH
  entropy                production=0.2349110171198845     replay=0.2349110171198845     MATCH
  approx_kl              production=0.004234685453411657   replay=0.004234685453411657   MATCH
  approx_kl_max          production=0.0129606444388628     replay=0.0129606444388628     MATCH
  clip_fraction          production=0.025146484375         replay=0.025146484375         MATCH
  explained_variance     production=0.8088640809865265     replay=0.8088640809865265     MATCH
  lr                     production=0.00018999167999999997 replay=0.00018999167999999997 MATCH
  n_grad_updates         production=4                      replay=4                      MATCH
```

The `net state_dict` digest is over all 25 tensors / 1,650,701 parameters, and
the production side is `probe_model_88006656.pt` — a byte copy of the
`model_latest.pt` the production trainer wrote at 21:20:14. Every metric agrees
to the last decimal digit of its float64 repr.

The comparison is apples-to-apples despite my harness skipping the evaluation:
in production the 88M eval ran *between* iteration 122's update and the
`model_latest.pt` write, and `evaluate_drc` does not modify the network
(`DRCNet` has no batch-norm or dropout, and `net.train(was_training)` restores
the mode), so the saved weights are exactly the post-iteration-122 weights.

This is the keystone of the report. It establishes three things at once:

1. the harness is a faithful reimplementation of `build_and_train`'s loop;
2. **the production resume at 21:10 was itself a faithful, exact restore** — a
   fresh process handed the same checkpoint lands on the same 1.65M parameters
   1M steps later;
3. therefore §4's restart A/B is a statement about the actual production
   trajectory, not about a lookalike.

It also disposes of a false alarm worth recording: at replay iteration 34 the
`value_loss` had reached 71.3, four times the recorded 88M value, which looked
like a divergence. It was not — `value_loss` oscillates between 13.7 and 71.3
across these 122 iterations and arrives back at exactly 16.0117. Point samples
1M steps apart are a poor instrument for this quantity (see §5).

---

## 5. The collapse was already running away before the pause

Per-1M-step window table. `val` is the recorded `curve.jsonl` value; `reeval` is
this investigation's independent re-evaluation of the retained weights on the
fixed 200-puzzle suite (fingerprint verified). `V_std`/`V_mean` are the value
head's output statistics on the fixed 16,000-state probe set (zero initial
recurrent state). `x/win` is the multiplicative change over the previous window.

```
          t   val reeval     vloss  x/win   V_std  x/win   V_mean      kl   klmax     EV  clipf   ent front   adam  min/1M
 75,005,952 0.795  0.795     0.039      -   0.174      -     0.96 0.00181 0.00309  0.751  0.020 0.271     7            -
 80,003,072 0.800  0.800     0.037   0.97   0.153   0.88     0.97 0.00307 0.00643  0.723  0.034 0.234     8  39060    48.7
 81,002,496 0.805  0.805     0.036   0.97   0.168   1.10     0.95 0.00287 0.00621  0.758  0.024 0.253     7             9.8
 82,001,920 0.785  0.785     0.046   1.25   0.191   1.14     0.94 0.00229 0.00611  0.739  0.020 0.231     7             9.9
 83,001,344 0.800  0.800     0.076   1.66   0.233   1.22     0.88 0.00174 0.00376  0.772  0.021 0.241     6             9.9
 84,000,768 0.800  0.800     0.306   4.05   0.460   1.97     0.72 0.00154 0.00338  0.788  0.016 0.244     8             9.8
 85,000,192 0.795  0.795     0.711   2.33   0.723   1.57     0.51 0.00187 0.00376  0.825  0.015 0.239     7  41500     9.6
 86,007,808 0.785  0.785     1.718   2.42   1.154   1.60     0.21 0.00247 0.00581  0.829  0.027 0.237     7             9.8
 87,007,232 0.770  0.770    19.180  11.16   3.773   3.27    -2.66 0.00241 0.00525  0.826  0.019 0.244     6  42480     9.8
  ---------- PAUSE 16:29:20  (machine slept 16:37:45 -> 21:08:23, battery 5% -> 1%)  RESUME 21:10:29 ----------
 88,006,656 0.755  0.755    16.012   0.83   3.880   1.03    -2.39 0.00423 0.01296  0.809  0.025 0.235     6             9.6
 89,006,080 0.360  0.360   335.636  20.96  21.761   5.61   -62.25 0.01408 0.03614  0.812  0.047 0.240     3             9.6
 90,005,504 0.315  0.315   272.708   0.81  28.404   1.31   -75.81 0.01123 0.03884  0.830  0.020 0.284     2  43941    10.8
 91,004,928 0.275  0.275   313.005   1.15  29.323   1.03   -95.77 0.00307 0.00975  0.836  0.010 0.254     2  44429    69.3
```

Readings that matter:

- **The runaway predates the pause by 5M steps.** `value_loss` rises
  monotonically from 82M: ×1.25, ×1.66, ×4.05, ×2.33, ×2.42, ×11.16. By the
  pause it is **512× its 80M value** and `V_std` is **24.7×**.
- **The critic was already grossly wrong at the pause.** `V_mean` on the fixed
  probe set went from +0.97 (80M, correct scale for a task whose returns are
  ~O(1)) to **−2.66 at 87M** — a sign error — and to −95.8 by 91M. The
  crossing happened *before* the pause.
- **The largest single pre-pause window (×11.16, 86M→87M) is larger than any
  post-resume window except 89M's ×20.96.** Multi-fold jumps are characteristic
  of this phase on both sides of the pause.
- **The 88M window — the one straddling the restart — shows no discontinuity at
  all.** `V_std` moves ×1.03, the smallest change of any window up to that point
  (F13). This is precisely where a restart-induced state perturbation would
  announce itself; instead it is the quietest window before the collapse. (The
  later ×1.03 at 91M is the post-collapse saturation, a different regime.)
- **`explained_variance` stays 0.81–0.84 while the loss grows 10⁴×.** The critic
  keeps its *relative* fit; only its absolute scale diverges. That is the exact
  signature of a return-scale runaway, and it is why the periodic `val_det`
  masked the problem until the policy finally broke.
- **`approx_kl` stays ≤0.0141 (max 0.0388), never near `1.5 × target_kl =
  0.075`.** No update was ever early-stopped (F4). The collapse has no KL
  signature — consistent with the previously documented `drc33_8x8_gate`
  collapse mode, not with a policy blow-up.
- **The training-side curriculum EMAs collapsed too**, so this is not an
  evaluation artefact: at 87M the sampler EMA was
  `[.99, .98, .94, .88, .82, .68, .80, .58]` with frontier 6; by 90M
  `[.995, .68, .56, .22, .088, .060, .009, .004]` with frontier 2.

**Quantitative test of "the pre-pause trend already predicted this".**

A caveat first, established by the replay: `value_loss` is a *very* noisy
statistic. Over 122 consecutive replay iterations from the 87M state it swings
between **13.7 and 71.3** with no trend visible at that resolution (e.g. 71.3 at
iteration 34, back to 31.2 at iteration 57, 22.0 at iteration 64). The
`curve.jsonl` rows are single-iteration point samples of this, taken 1M steps
apart. Any fit to them is indicative only — and in particular the recorded
`value_loss` 16.01 at 88M is a sample from an oscillation, not evidence that the
runaway paused.

`V_std` on the fixed 16,000-state probe set is the right statistic: it is a
deterministic function of the weights, with no rollout sampling in it. Fitting
`log10(V_std)` on **82M–87M only** (six points, entirely pre-pause) gives slope
**0.250 decades per 1M steps, R² = 0.952**. Extrapolating past the pause:

| t | `V_std` predicted from pre-pause data only | observed | obs/pred |
|---|---|---|---|
| 88,006,656 | 4.753 | 3.880 | 0.82 |
| 89,006,080 | 8.445 | 21.761 | 2.58 |
| 90,005,504 | 15.007 | 28.404 | 1.89 |
| 91,004,928 | 26.667 | 29.323 | **1.10** |

The pre-pause trend predicts the post-pause value scale to within a factor of
2.6 throughout, and to within **10% by 91M**. The same fit on `value_loss` gives
slope 0.501 decades/1M (R² = 0.955) and brackets the observations within 3.5×.
The post-89M flattening is expected: once the policy has collapsed, episode
returns stop growing and the runaway saturates. Nothing in the post-resume data
requires a restart to explain it.

My own replay from the 87M state continues on this trajectory, with rollout
returns already reaching ±43…±81 at t≈87.3M — the *return scale itself* is
inflated, not merely the loss.

---

## 6. Mechanism: `max_grad_norm` converts value divergence into policy death

`update()` builds one scalar
`loss = policy_loss − ent_coef·entropy + vf_coef·value_loss` and then applies
`clip_grad_norm_(params, 0.5)` to the **combined** gradient. When the value term
explodes, clipping does not protect the policy — it rescales the whole update,
so the policy gradient's contribution shrinks in proportion.

Measured on a real minibatch at four retained checkpoints
(`--grad-probe`; value term already ×`vf_coef=0.5`, entropy ×`ent_coef=0.01`):

| ckpt | value_loss | ‖g_policy‖ | ‖g_value‖ | ‖g_total‖ | clip factor | policy share | **effective ‖g_policy‖** |
|---|---|---|---|---|---|---|---|
| 80M | 0.035 | 1.720 | 0.236 | 1.741 | 0.287 | **98.8%** | **0.4939** |
| 85M | 0.795 | 2.049 | 2.571 | 3.198 | 0.156 | 64.1% | 0.3203 |
| **87M (pause)** | 19.02 | 2.129 | 57.76 | 57.96 | 0.0086 | **3.7%** | **0.0184** |
| 90M | 389.5 | 3.062 | 2266.2 | 2265.9 | 0.00022 | 0.14% | 0.00068 |

The policy gradient never weakens in itself (‖g_policy‖ actually *grows*,
1.72 → 3.06). What collapses is its share of the budget. **At the moment of the
pause the policy was already receiving 1/27 of its healthy update magnitude**;
by 90M, 1/730. The shared ConvLSTM trunk was being optimised almost purely to
fit a diverging value target.

This also explains the timing puzzle: `val_det` held at 0.755 through 88M (3.7%
share is small but non-zero) and then fell off a cliff once the share reached
~0.1%. And it explains the ConvLSTM saturation visible in the checkpoints —
recurrent `c` absmax 27.9 (80M) → 33.6 (85M) → 52.6 (87M) → 44.3 (90M), with `h`
absmax pinned at 0.99998, i.e. `tanh(c)` fully saturated, growing smoothly across
the pause.

`clip_range_vf: null` means nothing bounds the per-update value change, so there
is no brake anywhere in this loop. The module docstring already records this
failure mode from `drc33_8x8_gate_v2` (`value_loss 43 → 9.4e6`); the guard that
was added for it was left disabled in this config.

### 6.1 Root cause: the truncation bootstrap reads an observation slice that is never trained

§6 explains how a large value error kills the policy. This explains where the
error comes from. **It is a bug, not a hyperparameter choice.**

Three facts about `collect_rollout`, all checkable by reading the code:

1. `env.step` increments `_steps` *before* building the observation, and
   `_helper_planes` sets `remaining = (horizon - _steps) / horizon`. At horizon
   expiry `_steps == horizon == 24`, so the returned observation has
   **time channel exactly 0**.
2. That observation is appended to `trunc_obs`, passed through the value head,
   and `rewards[trunc_slots] += gamma * boot_values`. It is then **discarded** —
   `env.reset()` overwrites `self._obs[b]`, so it is never written into
   `buf.obs`, and therefore **never a regression target**.
3. Consequently `remaining == 0` never appears in any stored observation. The
   87M checkpoint confirms this directly from data: across all 256 envs the
   `steps` histogram spans **0…23 only** — never 24.

So the trainer evaluates `V` on a slice of input space it has never trained on,
and adds `0.99 ×` that output straight into the reward, on every truncated
episode. With `horizon=24` and optimal lengths ≤ 8, truncation is common.

And `V` at horizon expiry is not a free parameter: because the remaining time is
*in the observation*, the MDP is genuinely finite-horizon and fully observed, so

> **V(s, remaining = 0) ≡ 0.** No reward is reachable after the horizon.

Measured on the fixed 16,000-state probe set with the time channel set to 0
(zero recurrent state), against a `goal_reward` of **1.0**:

| ckpt | `V(expiry)` mean | `V(expiry)` abs-mean | `V(expiry)` abs-max | `V(start)` mean | `value_loss` | `val_det` |
|---|---|---|---|---|---|---|
| 5.0M | −1.865 | 1.880 | | | 0.487 | 0.545 |
| 10.0M | −4.180 | 4.230 | | | 3.147 | 0.595 |
| 20.0M | −0.854 | 1.009 | | | 0.470 | 0.685 |
| 40.0M | −0.954 | 1.178 | | | 0.409 | 0.725 |
| 60.0M | −0.336 | 0.701 | | | 0.258 | 0.725 |
| 70.0M | +0.348 | 0.449 | | | 0.095 | 0.820 |
| 80.0M | +0.706 | 0.706 | 1.244 | +0.995 | 0.037 | 0.800 |
| 85.0M | −1.249 | 1.437 | 7.919 | +0.610 | 0.711 | 0.795 |
| 86.0M | −2.917 | 3.000 | 14.378 | +0.392 | 1.718 | 0.785 |
| **87.0M (pause)** | **−12.278** | **12.286** | 38.813 | −1.877 | 19.180 | 0.770 |
| 88.0M | −11.694 | 11.707 | 43.800 | −1.729 | 16.012 | 0.755 |
| 89.0M | −91.875 | 91.875 | 128.525 | −56.589 | 335.636 | 0.360 |
| 91.0M | −138.961 | 138.961 | 167.176 | −85.195 | 313.005 | 0.275 |

Readings:

- **It was never zero — not once in the entire run.** The channel was leaking
  from step 0. There was no trigger at 82M because none was needed.
- **It tracks `value_loss` across the whole run**, including the healthy phase
  (0.449 / 0.095 at 70M; 12.29 / 19.18 at 87M). This is the dominant contributor
  to return variance.
- **The magnitude is absurd relative to the task.** At 80M, timing out was worth
  **+0.70** — i.e. running the clock down paid almost as well as solving the
  puzzle (+1.0). By 87M it was **−12.3**: a penalty twelve times the goal reward.
  By 91M, −139. The agent's returns were dominated by an untrained extrapolation,
  which is exactly why replay rollouts show returns of ±43…±150 on a task whose
  true returns are O(1).
- **The untrained slice is worse-behaved than the trained one**, as you would
  expect: at 87M `|V(expiry)| = 12.3` versus `|V(start)| = 1.88`, a 6.5× gap.

That closes the loop into a positive feedback cycle with no contraction
guarantee: `V(expiry)` drifts → the drift becomes real reward → returns inflate →
`V` is trained to larger values everywhere → `V(expiry)` extrapolates further →
and (via §6) the value gradient crowds the policy out of the update budget. A
correct bootstrap is safe because the bootstrapped state is *also* a regression
target and gets pinned; this one has no such anchor.

**The underlying inconsistency.** SB3's `TimeLimit.truncated` bootstrap is
correct when the time limit is an *unobserved* training artefact — you bootstrap
because the episode "would have continued". This config sets
`time_channel: true`, which puts remaining time *into the observation*, making
the horizon part of the MDP. The two choices are mutually exclusive, and the run
used both. Either is individually sound:

- `time_channel: true` **+ treat horizon expiry as terminal** (no bootstrap) —
  correct, and it does not change the optimal policy, because the MDP genuinely
  is finite-horizon and fully observed. **Recommended.**
- `time_channel: false` **+ keep the bootstrap** — also self-consistent, and it
  incidentally removes the unpinned slice too (with no time channel, the
  truncation observation is an ordinary in-distribution board state). But it
  throws away information the agent demonstrably uses.

**The bootstrap was never in the reference path.** `drc_train.py`'s docstring
says the truncation bootstrap is "the same semantics SB3 implements via
`TimeLimit.truncated` infos" (line 30). It is not. SB3 only bootstraps when the
info dict carries **both** `terminal_observation` and `TimeLimit.truncated`
(`on_policy_algorithm.py:236–245`), and `src/groundup/env.py` emits **neither** —
zero occurrences of `TimeLimit` in the file. The legacy `src/env/ricochet_env.py`
does set it (line 428), but the ground-up env does not.

So the feed-forward P1 baseline (`train.py` → `MaskablePPO` → the same
`RicochetEnv`) already does the correct thing: horizon expiry is treated as
terminal, with no bootstrap and no unpinned-slice leak. `drc_train.py`
hand-rolled a bootstrap to match a behaviour the reference path never actually
had, and did so under `time_channel: true`, where it is doubly wrong. **This is a
DRC-trainer regression against the P1 baseline, not a shared defect** — which
also means the P1 vs DRC comparison was not like-for-like on reward semantics.

---

## 7. Operational forensics

**Code identity (F15/F16).** Independently verified, not taken on trust: the git
blob hashes of `drc_train.py`, `drc.py`, `env.py`, `curriculum.py`, `puzzles.py`,
`distance.py` and `evaluate.py` are byte-identical at `1964f7d`, `fede440`,
`96fcbff`, `2817c10` and `HEAD`. The earlier stale-snapshot incident therefore
left no residue: the 21:10 resume ran the same trainer bytes as the healthy
60M→87M continuation. `fede440..96fcbff` changed only the config's `resume_from`
line plus probe-subsystem and doc files. All 18 snapshot-manifest rows record
`dirty: false`.

**Timeline (from `pmset -g log`, artifact mtimes and `curve.jsonl` elapsed).**

| time (BST) | event |
|---|---|
| 15:20:48 | t=80,003,072 checkpoint — `val_det` 0.800, `value_loss` 0.037 |
| 16:09:48 | t=85,000,192 — 0.795, 0.711 |
| 16:19:39 | t=86,007,808 — 0.785, 1.718 |
| **16:29:20** | **t=87,007,232 — 0.770, 19.180; the pause checkpoint's inode is written** |
| 16:37:45 | **system enters Clamshell Sleep, battery 5%** — `caffeinate -is` does not block lid-close sleep |
| 16:37–17:07 | asleep with DarkWake blips |
| 17:07:54 | Wake (lid, user activity), battery 5% |
| ~17:12 | user requests the battery pause; SIGINT |
| 17:49:27 | Low Power Sleep at battery 1% |
| 21:08:23 | Wake from Hibernate, on AC, 100% |
| 21:10:29 | resume from `training_pause_87007232.pt` |
| 21:20:14 | t=88,006,656 — 0.755, 16.012 (9.6 min for 1M steps) |
| **21:29:55** | **t=89,006,080 — 0.360, 335.636 (9.6 min for 1M steps)** |
| 21:40:40 | t=90,005,504 — 0.315, 272.708 |
| 21:49:33 | Clamshell Sleep (now unplugged at 100%) — repeated cycles to 22:50 |
| 22:50:01 | t=91,004,928 — 0.275, 313.005 (69.3 min for 1M steps, sleep-throttled) |

Two things follow. First, the pre-pause process got only ~8.4 min of compute
after the 87M checkpoint (≈74 iterations, t≈87.6M) before the machine slept,
which is why no 88M row exists pre-pause — there is no missing curve row.
Second, and decisively: **the 88M→89M collapse window ran fully awake at
9.6 min/1M, identical to the pre-pause 86M→87M window at 9.8 min/1M.** Sleep,
power-source and throttling differences cannot explain the collapse; the sleep
events sit strictly before the resume and strictly after 90M.

**Probe subsystem (F18).** Every write site in `probe_drc_daemon.py`,
`retain_drc_probe_snapshots.py` and `drc_probe_wandb.py` was inspected. They
write only `probe_control.json`, `.drc_probe_wandb_state.json`, `probe_model_*.pt`
copies and `drc_probe_*` outputs — never `training_*.pt`, `model_*.pt`,
`curve.jsonl`, `run_history.jsonl` or `snapshot_manifest.jsonl`. The watcher
copies `model_latest.pt` with an exclusive-create temp (`open("xb")`) plus
`os.replace`, and re-checks the step after copying. A duplicate low-priority
daemon could therefore only have cost CPU, and F20 shows throughput in the
collapse window was normal. The two W&B clients are separate processes writing
separate run directories.

**Latent fragility worth noting (not a cause).** `RicochetEnv._dmap_cache` is
keyed on `puzzle.seed` and is *not* serialized. All 39,713 train puzzles have
unique seeds today, so it is a pure memoization and inert across restarts (F21).
If a future bank ever had colliding seeds this would be both incorrect *and*
restart-sensitive, since a resumed process starts with an empty cache.

---

## 8. Reproducing this audit

```bash
cd /Users/sidbaines/Documents/ricochet-robots-rl
D=artifacts/groundup/drc33_p1_12x12_planefree
S=artifacts/forensics/drc87m_restart      # harness scripts are kept here with their outputs

# --- hashes (recomputed; all match the brief) ---
shasum -a 256 $D/training_80003072.pt $D/training_85000192.pt \
              $D/training_pause_87007232.pt $D/training_90005504.pt $D/training_latest.pt
# cc3bb17da49d2521a83c06285afe6d691fb1d643279afa6619a069249cd0db39  training_80003072.pt
# f2ffa8cee0c17f617dca07595a4cbc7909a7cfdca7f9e69392ea7c97c54b4a92  training_85000192.pt
# 91c8e8ae197f027c7da457c98265d66cc8074ab8324c83051a723169b23c691e  training_pause_87007232.pt
# c6cb165879a6d160d79d21590e18c099256d7a29ea6686dec340a3c84241f490  training_90005504.pt
# 613a90722e1d9ae6d92e3031b3f91f01b9bdbaf632d0d295f4f43c48bc123f94  training_latest.pt

# --- source-blob identity across the two launch commits ---
for c in 1964f7d fede440 96fcbff 2817c10 HEAD; do
  for f in src/groundup/{drc_train,drc,env,curriculum,puzzles,distance,evaluate}.py; do
    echo "$c $f $(git rev-parse $c:$f)"; done; done | sort -k2
# drc_train.py 2012f65bc9f6c65c60ce1af42d3348c17447cf48 at all five commits
# drc.py       4444322a42ed34fdeac14c0fbc9dce2e849ccc8f
# env.py       29de189bd7457734f8fa3d0b7a0d0cbd2d9fe1e9
# curriculum.py 00651f5efa02669d624d93cb8511d3467ce99fa7

# --- checkpoint payload audit ---
.venv-phase2/bin/python $S/audit_ckpt.py $D/training_pause_87007232.pt
.venv-phase2/bin/python $S/audit_ckpt.py $D/training_80003072.pt $D/training_85000192.pt \
                                         $D/training_90005504.pt $D/training_latest.pt

# --- the RNG constant ---
.venv-phase2/bin/python -c "import hashlib,torch; torch.manual_seed(0); \
d=lambda t: hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]; \
print(d(torch.get_rng_state()), d(torch.mps.get_rng_state()))"
# 1ccf17250133dec5 c29a1d9c25f640eb   == the value stored in every checkpoint

# --- independent re-evaluation + value-scale trajectory (~95 s) ---
.venv-phase2/bin/python $S/reeval_and_vscale.py --device mps

# --- the restart A/B (~5.5 min total) ---
CK=$D/training_pause_87007232.pt; O=artifacts/forensics/drc87m_restart
.venv-phase2/bin/python $S/forensic_replay.py --resume-from $CK --iters 20 --out $O/armU1.json --tag U1
.venv-phase2/bin/python $S/forensic_replay.py --resume-from $CK --iters 20 --out $O/armU2.json --tag U2
.venv-phase2/bin/python $S/forensic_replay.py --resume-from $CK --iters 10 \
        --save-at 10 --save-to $S/mid_87089152.pt --out $O/armR1.json --tag R1
.venv-phase2/bin/python $S/forensic_replay.py --resume-from $S/mid_87089152.pt \
        --iters 10 --out $O/armR2.json --tag R2      # <- NEW process

# --- gradient budget ---
for t in 80003072 85000192 pause_87007232 90005504; do
  .venv-phase2/bin/python $S/forensic_replay.py --resume-from $D/training_$t.pt \
        --iters 2 --grad-probe --out $O/grad_$t.json --tag grad$t; done

# --- production validation: full 87M->88M window, 122 iterations (~14 min) ---
.venv-phase2/bin/python $S/forensic_replay.py --resume-from $CK --iters 122 \
        --out $O/full_87to88M.json --tag FULL
.venv-phase2/bin/python $S/validate_full.py         # compares against probe_model_88006656.pt

# --- regression tests ---
.venv-phase2/bin/python -m pytest src/tests/test_drc_train.py -q
```

Everything is retained in `artifacts/forensics/drc87m_restart/` (gitignored, so
copy it out if it must survive): the four harness scripts (`audit_ckpt.py`,
`reeval_and_vscale.py`, `forensic_replay.py`, `validate_full.py`) plus
`ckpt_audit_all.txt`, `reeval_and_vscale.json`, `armU1.json`, `armU2.json`,
`armR1.json`, `armR2.json`, `grad_*.json`, `full_87to88M.json`,
`full_87to88M.log` and `full_validation.txt`.

`forensic_replay.py` is the piece worth promoting into the repo if this ever
needs re-running after a torch or macOS upgrade (it invalidates U1); the three
new unit tests in §9 cover the same ground far more cheaply.

---

## 9. New regression tests

`src/tests/test_drc_train.py` already had
`test_exact_resume_matches_uninterrupted_next_update`, but it could not have
caught a restart bug of the kind hypothesised, because it:

1. resumes **inside the same process** (no fresh interpreter, no fresh device
   context);
2. runs on **CPU only**, so the `elif kind == "mps"` branch of
   `RecurrentPPO.load_state_dict` — the branch that matters, since MPS actions
   are drawn from the device generator — was **never executed by the suite**;
3. compares only **one** post-resume iteration;
4. never asserts the **Adam moments** round-trip (only net weights and metrics);
5. builds both arms from the **same seeds**, so a field that silently failed to
   restore would still compare equal by accident.

Added three tests closing all five gaps:

- `test_exact_resume_overwrites_every_field_of_a_differently_seeded_process` —
  builds the resumed arm from **different** seeds, asserts no digested field
  survives at its fresh-construction value, asserts optimizer-state equality,
  and checks three iterations rather than one.
- `test_exact_resume_survives_a_real_process_restart` — resumes the checkpoint in
  a **genuinely fresh interpreter** via `subprocess` and requires bit-equality
  with the in-process arm. This is the minimal reproducer for the whole
  hypothesis class, and it establishes CPU restart exactness (U2).
- `test_exact_resume_round_trips_device_rng_state_on_mps` — MPS-gated; exercises
  the previously untested `device_rng_state` save/restore branch.

Also fixed a **pre-existing** red test: `test_primary_12x12_drc_config_is_helper_free_and_extension_safe`
still asserted `resume_from == training_60006400.pt`, which commit `96fcbff`
invalidated when it repointed the config at the pause checkpoint. Verified red at
`HEAD` before my edits (via `git stash`).

```
.venv-phase2/bin/python -m pytest src/tests/test_drc_train.py -q
22 passed
```

---

## 10. Is "exact resume" technically justified?

- **On MPS, for this run: yes, empirically — and it is the strongest claim in
  this report.** Five independent processes resumed from the same checkpoint
  produced bit-identical parameters, Adam moments, env RNG states and metrics,
  and a sixth continued that trajectory across a genuine process boundary
  without deviation — including a 122-iteration (1M-step) production-scale
  replay. Every field that determines the trajectory is either serialized and
  verified to round-trip, or (for the global generators) a provable constant.
- **On CPU: yes, at unit-test scale**, now asserted across a real process
  boundary by `test_exact_resume_survives_a_real_process_restart`.
- **Across devices: no, and the term should never be used that way.** CPU and
  MPS diverge immediately from the same checkpoint (iteration-1 `value_loss`
  18.1032 vs 19.8086). `device` is correctly excluded from the resume-mutable
  field set, so the contract check does not currently prevent a CPU↔MPS resume
  from being described as exact. Worth tightening.
- **Caveat (U1).** MPS determinism is an empirical property of this
  kernel/shape/driver combination, not a PyTorch guarantee. The claim should be
  re-verified after any torch or macOS upgrade; the new tests do exactly that
  cheaply.

---

## 11. Recommendations

### 11.1 Disposition of the frozen primary run

The run is not recoverable by resuming it. At 91M the critic is ~100× out of
scale, the policy gradient receives ~0.1% of the update budget, and `val_det`
0.275 is below the frontier-2 curriculum's own floor. Continuing to 120M would
only burn compute.

Recommended, in order:

1. **Terminate the frozen trainer, snapshot watcher and probe daemon** — but
   only after taking a final exact-state snapshot for the record. `SIGCONT` is
   not needed and not advised: the 91M state is already fully persisted in
   `training_latest.pt` (verified: `num_timesteps` 91,004,928, Adam step 44,429,
   coherent). Simply `SIGKILL`/`SIGTERM` the three screens.
2. **Do not overwrite `model_best.pt`.** It holds t=79,003,648, which
   independently re-evaluates to `val_det` **0.830** (per-length
   `{1:.96, 2:.96, 3:1.0, 4:.84, 5:.88, 6:.80, 7:.60, 8:.60}`) — the best point
   of the entire run. F10c/F19 confirm it is bit-identical to
   `probe_model_79003648.pt` and to the `best_state_dict` carried inside every
   checkpoint, so those weights survived the resume untouched. Copy it out to a
   name the trainer will never write before doing anything else.
3. **Report the run as a value-divergence failure at 82–89M**, not as a restart
   casualty, and record the 60–80M plateau (`val_det` 0.79–0.83) as its actual
   result. Note that this already beats the feed-forward P1 baseline's 66–73%
   validation.
4. **Leave the sealed test unopened.** Opening it on a run whose registered stop
   target was never reached, and whose best weights predate a known divergence,
   would spend the one-shot test on an unclear question. If a sealed-test number
   is wanted for the DRC arm, spend it on a *rescued* run that reaches a
   pre-registered stop cleanly.
5. **Fix the defects found here**, in priority order, regardless of what happens
   next:
   1. **the truncation bootstrap (§6.1)** — a real bug, and the root cause. Fix
      it before launching anything else, or every future run on this env
      inherits it;
   2. the `evaluate_drc` global reseed (§3.2);
   3. the CPU↔MPS "exact resume" loophole (§10).
6. **Operationally:** `caffeinate -is` did not prevent clamshell sleep. For long
   runs use `caffeinate -dis` with the lid open, or keep the machine on AC, and
   treat "battery below ~20%" as a stop condition rather than a pause condition.

### 11.2 Rescue arm — separate from this investigation

This is a *scientific* proposal about the instability, and it is **not** evidence
about the restart. Keep it in a new run directory with a new W&B run; do not
append to `pupnnpq8`.

The correct restart point is `training_80003072.pt` (`val_det` 0.800,
`value_loss` 0.0375, `V_mean` +0.97, policy-gradient share 98.8%) — the last
checkpoint clearly *before* the runaway. `training_85000192.pt` is already 19×
into it and 85M's `V_mean` has fallen to +0.51.

Because §4 established that resume is bit-exact and the loop deterministic, a
single arm per configuration is sufficient — there is no seed noise to average
over, which makes this cheap.

Proposed arms from 80M, each to ~92M (≈12M steps, ~2 h at 1,700 st/s):

- **Arm C — fix the bug (§6.1). Run this first; it is the only arm that
  addresses a defect rather than a symptom.** ✅ **The code fix is now
  implemented and verified — see §12.** The predictions below were confirmed:
  returns dropped to O(1), `value_loss` to ~0.12, and the policy-gradient share
  recovered from 3.67% to 65.6%, all from the corrupted 87M state within 20
  iterations. Arm C is now just "run the fixed trainer from 80M". With
  `time_channel: true`, horizon expiry is a genuine terminal, so the bootstrap
  is dropped:

  ```python
  # collect_rollout: with the remaining time in the observation, the MDP is
  # finite-horizon and fully observed, so V(s, remaining=0) == 0 by definition.
  # Bootstrapping there injects an untrained extrapolation into the reward.
  if terminated or truncated:
      obs_b, _info = env.reset()
      new_starts[b] = True
  ```

  i.e. delete the `trunc_slots` / `trunc_obs` collection and the
  `rewards[trunc_slots] += self.gamma * boot_values` block. This makes the DRC
  trainer match the feed-forward P1 baseline, which never bootstrapped. Falsifiable
  predictions: `V` at time-channel 0 becomes irrelevant (never evaluated);
  rollout returns stay O(1) instead of reaching ±150; `value_loss` stays O(0.01–0.1);
  the policy-gradient share stays ~99% instead of decaying to 3.7%.

- **Arm B — `clip_range_vf: 0.2`** (the brief's registered candidate). Worth
  running, but note it treats the symptom: it bounds how fast `V` can move
  without removing the channel that feeds it bad targets. Also heed the
  `clipped_value_loss` docstring — the clip-only form previously *caused*
  exponential divergence in `drc33_8x8_gate_v2`; the element-wise-max form here
  is the safer variant. If Arm C works, Arm B is optional insurance rather than
  a rescue.

- **Arm A (control):** unchanged config. Its value is now purely as a documented
  reference curve — §4 guarantees it reproduces the 87M→91M collapse bit-for-bit,
  so run it once and cite it.

- **Cheap insurance orthogonal to all three:** switch the value loss to Huber /
  smooth-L1. The per-sample gradient is then bounded, so no value error however
  large can consume the whole `max_grad_norm=0.5` budget and starve the policy
  (§6). It leaves the fixed point unchanged for small errors. This does not fix
  the root cause and should not substitute for Arm C.

- **Instrumentation every arm should add**, because all three of these moved
  5–20M steps before `val_det` did, whereas `value_loss` did not (it swings
  13.7↔71.3 iteration-to-iteration and `explained_variance` stayed 0.81–0.84 the
  whole way down):
  1. `V` at time-channel 0 on the fixed probe set — should be ~0; was 0.71 at
     80M and −12.3 at 87M. **This is the single best alarm and it is nearly free.**
  2. `V_mean`/`V_std` on the fixed probe set — `V_mean` had already flipped sign
     by 87M.
  3. Policy-gradient share of the update — 98.8% → 3.7% between 80M and 87M.

Each arm needs `out_dir`/`run_name`, `total_steps`, `resume_from` and W&B
overrides only — all are in `_RESUME_MUTABLE_CONFIG_FIELDS`, so the contract
fingerprint `af7aae49…` is preserved. Arm B/C change contract fields and will
therefore (correctly) require a new fingerprint; record it in
`docs/DrcPlaneFreeResearchPlan.md` before launching.

Launch detached, per CLAUDE.md:
`nohup caffeinate -dis <cmd> > <log> 2>&1 &`.

---

## 12. Fix applied and verified

Implemented test-first at the user's request, after the report above.

### 12.1 The failing test

`test_no_truncation_bootstrap_when_the_horizon_is_in_the_observation` forces the
condition deterministically: two envs on 2-move puzzles with `horizon=1`, so the
goal is unreachable and **every** stored step truncates, and the critic is pinned
to a constant (`critic.weight.zero_()`, `critic.bias.fill_(5.0)`) so any
bootstrap term is exactly `gamma * 5 = 4.95`. It then asserts the buffered reward
equals what `env.step` actually returned. Pre-fix output:

```
env0 env-returned reward  : [-0.008, 0.091, -0.008, -0.008]
env0 stored in buffer     : [ 4.942, 5.041,  4.942,  4.942]
env0 injected by bootstrap: [ 4.950, 4.950,  4.950,  4.950]
```

Real rewards are O(0.01); the bootstrap injected 4.95 — a ~600× corruption of
every truncated step's reward.

### 12.2 The change

`src/groundup/drc_train.py`, ~40 lines:

- `RecurrentPPO.__init__` gains `bootstrap_on_truncation: Optional[bool] = None`.
  `None` auto-detects from the envs' `config.time_channel`: bootstrap iff the
  horizon is **not** observable. Envs exposing no `config.time_channel`
  (duck-typed/foreign) are assumed to hide their limit — the SB3-compatible
  default, which keeps the legitimate bootstrap path intact.
- Passing `bootstrap_on_truncation=True` while any env has `time_channel=True`
  now **raises**, so the inconsistent combination cannot be reintroduced silently.
- Mixed `time_channel` across envs raises.
- `collect_rollout` skips the bootstrap accordingly.
- The resolved value is printed at startup and recorded in `provenance`, hence in
  `run_history.jsonl` and every snapshot's metadata.

Deliberately **not** added as a `DRCTrainConfig` field: that would change the
config-contract fingerprint and make every existing checkpoint unresumable,
which would block the rescue arm. Deriving it from the existing `time_channel`
field keeps `af7aae49…` valid. The cost is that the fingerprint alone no longer
distinguishes pre- from post-fix runs — hence the explicit provenance entry; the
git commit in `run_history.jsonl` is the authority.

### 12.3 Verification

Compatibility: `training_pause_87007232.pt` still restores with **16/16** field
fidelity post-fix. All existing checkpoints remain resumable.

Behaviour, replaying from the **corrupted** 87M checkpoint (same state that
previously ran away to `value_loss` 336 by 89M):

| iteration | 1 | 2 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| `value_loss` | 53.93 | 34.64 | 4.81 | **0.163** | 0.137 | 0.123 | **0.115** |
| `ret_absmax` | 38.9 | 38.0 | 24.1 | **1.25** | 1.52 | 1.48 | **1.59** |
| `approx_kl` | 0.025 | 0.070 | 0.061 | 0.0074 | 0.042 | 0.0086 | **0.0042** |

Within 10 iterations (82k steps) the return scale collapses from ±39 to the
task's true O(1) scale and the value loss falls two and a half orders of
magnitude, then holds. Pre-fix, the same checkpoint went 19.8 → 30.3 over 20
iterations and on to 336 by 89M. **The fix does not merely prevent the
divergence; it reverses it** — which suggests the run may be salvageable from 87M,
not only from the clean 80M state.

Gradient budget at weights re-converged under the fixed objective (§6's
falsifiable prediction):

| state | `value_loss` | policy share | effective ‖g_policy‖ |
|---|---|---|---|
| 80M, buggy objective (healthy) | 0.035 | 98.79% | 0.494 |
| 87M, buggy objective (dying) | 19.02 | 3.67% | 0.0184 |
| 90M, buggy objective (collapsed) | 389.5 | 0.14% | 0.00068 |
| **87M + 20 iters, fixed objective** | **0.132** | **65.60%** | **0.328** |

The policy's share of the update recovers from 3.67% to 65.6% and its effective
magnitude 18×, from the damaged 87M weights. It does not reach 80M's 98.8%
because those weights are still mis-scaled and the ConvLSTM is still saturated
(`c` abs-max stays 35–48 throughout, a slower-moving quantity).

**Caveat — do not read iteration 1 as the steady state.** Removing the bootstrap
drops the targets to their true scale while `V` still predicts what it was
trained to, so the *instantaneous* value error rises before it falls: the first
iteration shows `value_loss` 0.035 → 0.079 at 80M, 19.0 → 61.1 at 87M and
389 → 4003 at 90M, with correspondingly worse policy shares. That is the
re-fitting transient, visible in the table above as iterations 1–5. It also means
a rescue resumed from a *late* corrupted checkpoint will see elevated `approx_kl`
(up to 0.081 here, above `1.5 × target_kl = 0.075`) for a few iterations and may
trip the early stop — another reason to prefer the clean 80M start.

### 12.4 Tests added

`src/tests/test_drc_train.py` (6 new):

- `test_no_truncation_bootstrap_when_the_horizon_is_in_the_observation` — the
  primary regression test (§12.1).
- `test_stored_observations_never_reach_horizon_expiry` — the invariant that
  makes the bootstrap unsound: no stored observation ever has time channel 0.
- `test_truncation_bootstrap_still_applies_when_the_horizon_is_hidden` — guards
  against over-fixing; with `time_channel=False` the bootstrap must still fire.
- `test_bootstrap_on_truncation_is_refused_when_the_horizon_is_observed`.
- `test_primary_drc_config_disables_the_truncation_bootstrap` — pins the
  registered 12x12 config's resolved semantics.
- (plus the pre-existing `test_truncation_bootstraps_final_value_and_termination_does_not`
  still passes on `ScriptedEnv`, confirming the legitimate path is intact.)

`src/tests/test_groundup.py`, `src/tests/test_env_obs.py` (2 new) — cross-path
guards, described in §12.5.

`pytest src/tests/ -q` → **204 passed, 9 skipped** (was 197 before this
investigation).

### 12.5 Similar-error audit

Searched for the same class — *a learned quantity evaluated on inputs outside its
training distribution and fed back into rewards or targets* — and for
observation/algorithm inconsistencies generally.

**Only one instance existed, and it is the one fixed.** `drc_train.py:600` was
the sole place any value estimate entered the rewards anywhere in `src/groundup`.

Checked and found **correct**:

| Site | Finding |
|---|---|
| GAE `last_values` bootstrap (`collect_rollout`) | Evaluates `V` on `self._obs`, which is always mid-episode or post-reset (`steps ≤ 23`), i.e. in-distribution — and `compute_gae` masks it with `last_episode_starts` where an episode just ended. Sound. |
| `train.py` / SB3 / `MaskablePPO` + groundup env | `src/groundup/env.py` emits neither `terminal_observation` nor `TimeLimit.truncated`, and SB3 requires **both**, so expiry is treated as a real terminal — correct under `time_channel=True`. Guarded by a new test, because adding either key would silently reintroduce the bug in the *other* trainer. |
| Legacy `src/env/ricochet_env.py` + SB3 | Does set `TimeLimit.truncated`, so SB3 **does** bootstrap — but its observation is clock-free (4 wall + goal + target + robots, no time plane), so the horizon is unobservable and the bootstrap is the correct treatment. The pairing is now pinned by `test_timelimit_truncated_is_paired_with_a_hidden_horizon`. |
| Action masks vs stored observations | Masks are recomputed at the top of each step from the same env state the observation came from, and the same masks are replayed in the update. Consistent. |
| `EnvConfig.gamma` vs PPO `gamma` | Already validated at construction. |

Known defects **not** fixed here, because they are a different failure mode and
each needs its own change and tests — flagged, not bundled:

1. **`evaluate_drc`'s `torch.manual_seed(seed)`** (§3.2) — a global side effect
   that re-anchors the exploration noise stream every `eval_every` steps. Fixing
   it properly means threading a local generator through `Categorical.sample`
   (which takes no `generator=`), so it is a real change, not a one-liner.
2. **`RicochetEnv._dmap_cache` keyed on `puzzle.seed`** (F21) — correct today
   only because all 39,713 train-split seeds are unique. A colliding bank would
   make it silently wrong *and* restart-sensitive. Note the obvious "fix" of
   keying on `layout_hash` would be **worse**: multiple puzzles share a layout
   with different goals.
3. **CPU↔MPS "exact resume"** (§10) — `device` is resume-mutable, so a CPU↔MPS
   resume passes the contract check while being non-exact.
