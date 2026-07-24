# Investigation brief: did the 87M restart cause the DRC collapse?

## Your role

Act as an independent, adversarial investigator. The user does **not** accept
the current working conclusion that the restart and subsequent collapse were
coincidental. Do not assume that conclusion is correct merely because it is
written elsewhere in the repository.

Your task is to determine whether pausing and restoring the exact-state
checkpoint at 87M caused, accelerated, or exposed the performance collapse
seen at 89M. A convincing answer must distinguish:

1. a checkpoint/save/load or restart bug;
2. a deterministic instability that was already underway before the pause;
3. a stochastic instability for which the restart changed the trajectory;
4. a monitoring/evaluation artefact;
5. another operational interaction not yet considered.

Begin with read-only forensics. Do not make the primary run resume, train,
overwrite artifacts, or inspect the sealed test. If a controlled replay
requires significant compute or terminating the frozen process, explain the
proposed experiment and obtain user approval first.

## Repository and current state

- Repository: `/Users/sidbaines/Documents/ricochet-robots-rl`
- Branch: `rl-ground-up`
- Primary run directory:
  `artifacts/groundup/drc33_p1_12x12_planefree`
- Training W&B run: project `ricochet-robots-rl`, ID `pupnnpq8`
- Dedicated live-probe W&B run: ID `bkvhg71c`
- Research plan and running log:
  `docs/DrcPlaneFreeResearchPlan.md`
- Formal 60M review:
  `docs/DrcPlaneFree60MReview.md`
- Current config:
  `configs/groundup_drc33_p1_12x12_planefree.yaml`

At the time this brief was written, the trainer Python process was deliberately
frozen with `SIGSTOP` at PID 86950 after writing the exact 91M checkpoint. PID
values can become stale, so inspect the process tree before relying on this.
Do **not** send `SIGCONT`, signals, or Screen commands until the user approves
a next action. The snapshot watcher and probe daemon may still be idle.

The primary artifacts are evidence. Do not delete, rename, truncate, mutate,
or replace them. In particular, do not overwrite:

- `training_pause_87007232.pt`
- `training_80003072.pt`
- `training_90005504.pt`
- `training_latest.pt`
- `curve.jsonl`
- `run_history.jsonl`
- `snapshot_manifest.jsonl`
- any `model_*.pt`, `probe_model_*.pt`, or `probe_t*.json`

Use separate temporary/output directories for any replay. Disable W&B or use a
new, explicitly labelled offline/test run; never append replay data to
`pupnnpq8` or `bkvhg71c`.

The sealed test has not been opened. It must remain unopened during this
investigation.

## Scientific background

The preceding feed-forward P1 experiment trained a convolutional policy on
12x12 Ricochet Robots boards without stop-cell or distance observation planes.
It plateaued around 66–73% validation and achieved 69.2% on its one-time sealed
test. Linear probes did not find a clear medium/long landing-cell circuit.

This experiment asks whether a recurrent DRC can do better while keeping the
task and structured, non-pixel observation comparable:

- DRC(3,3): depth 3, 3 recurrent ticks;
- 1,650,701 policy parameters;
- observation shape `(9, 12, 12)`: walls, goal, three robots, and time;
- no stop-cell plane and no distance plane;
- 12 masked actions;
- reverse curriculum over solution lengths 1–8;
- recurrent masked PPO, 256 environments, rollout length 32, batch size 2048,
  one epoch;
- 120M current stop target with a pre-registered 200M learning-rate horizon;
- fixed 200-puzzle periodic validation suite every approximately 1M steps;
- fixed 480-puzzle final validation suite, still not used for sealed testing;
- zero-state DRC landing probes on a fixed 16,000-state dataset.

The run uses seed 11 and config-contract fingerprint:

```text
af7aae4967d56957deedb11647b46045164efa437e51e885f789873e92611719
```

The registered primary config includes:

```yaml
learning_rate: 3.0e-4
learning_rate_final: 5.0e-5
vf_coef: 0.5
max_grad_norm: 0.5
target_kl: 0.05
adv_std_floor: 0.05
adv_clip: 10.0
clip_range_vf: null
schedule_total_steps: 200000000
```

Do not change these settings in a causal replay. A possible
`clip_range_vf: 0.2` rescue is a separate scientific arm, not evidence about
whether the restart caused the primary collapse.

## Relevant implementation

Read these files closely:

- `src/groundup/drc_train.py`
  - `RecurrentPPO.state_dict()` / `load_state_dict()`
  - `_save_training_checkpoint()` / `_load_training_checkpoint()`
  - rollout collection, GAE, PPO update, evaluation, checkpoint timing,
    learning-rate scheduling, and resume contract checking
- `src/groundup/drc.py`
  - recurrent state construction and device movement
- `src/groundup/env.py`
  - exact environment serialization and restoration
- `src/groundup/curriculum.py`
  - sampler state serialization and restoration
- `src/tests/test_drc_train.py`
  - current exact-resume tests and what they do **not** cover
- `src/groundup/retain_drc_probe_snapshots.py`
- `src/groundup/probe_drc_daemon.py`

Important Git points:

- `1964f7d`: original helper-free DRC run preparation;
- `fede440`: formal 60M review and exact continuation to 120M;
- `96fcbff`: battery pause recorded and config pointed at the immutable 87M
  pause checkpoint;
- `34523f3`: resume from 87M recorded;
- `2817c10`: current value-divergence diagnosis recorded.

`git diff fede440..96fcbff` shows no changes to `drc_train.py`, `drc.py`,
`env.py`, or `curriculum.py`. The resumed process therefore used the same
trainer/environment implementation as the healthy 60–87M continuation.
Verify this independently rather than taking it on trust.

There was also an unrelated repository incident earlier that day: another
agent replaced five tracked files with a stale July 20 snapshot. Those
uncommitted changes were identified and restored exactly to `HEAD` before the
87M resume; the live pre-pause trainer had already loaded its code in memory.
Confirm via Git history/status that no stale trainer code remained at the
21:10 resume.

## Exact operational timeline

### Initial run and 60M extension

- The original run began at step 0 under commit `1964f7d`.
- It stopped normally at t=60,006,400.
- It resumed from `training_60006400.pt` under commit `fede440` and trained
  successfully through 87M.
- This matters: the same exact-resume machinery had already been exercised at
  60M without an immediate collapse.

### Performance before the battery pause

Selected deterministic validation points:

| Step | Validation | Value loss | Frontier |
|---:|---:|---:|---:|
| 79,003,648 | 0.830 | not central here | 8 |
| 80,003,072 | 0.800 | 0.03745 | 8 |
| 81,002,496 | 0.805 | 0.03649 | 7 |
| 82,001,920 | 0.785 | 0.04557 | 7 |
| 83,001,344 | 0.800 | 0.07551 | 6 |
| 84,000,768 | 0.800 | 0.30560 | 8 |
| 85,000,192 | 0.795 | 0.71098 | 7 |
| 86,007,808 | 0.785 | 1.71811 | 7 |
| 87,007,232 | 0.770 | 19.18042 | 6 |

The rising pre-pause value loss is evidence for a pre-existing instability,
but it is not by itself proof that the restart was irrelevant. Investigate
whether the serialized 87M state was internally coherent and whether the
restart altered the subsequent trajectory.

### Pause at 87M

- At approximately 17:12 BST on 2026-07-24, the user asked to preserve
  battery.
- The last completed evaluation/checkpoint was t=87,007,232.
- The trainer was interrupted with `SIGINT` after that checkpoint existed.
  The traceback shows interruption inside a DRC forward pass, i.e. during
  post-checkpoint work toward the next boundary.
- `training_latest.pt` was validated, then hard-linked to:
  `training_pause_87007232.pt`.
- The hard link protected the 87M inode when later `training_latest.pt`
  rewrites used atomic replacement.
- SHA-256 of the pause checkpoint:

```text
91c8e8ae197f027c7da457c98265d66cc8074ab8324c83051a723169b23c691e
```

- Embedded metadata reports t=87,007,232 and clean launch commit `fede440`.
- `curve.jsonl` also ended at t=87,007,232.

### Resume at 87M

- At approximately 21:10 BST, the run resumed under commit `96fcbff`.
- Config pointed to `training_pause_87007232.pt`.
- Console output reported:

```text
[drc] resumed exact training state from
artifacts/groundup/drc33_p1_12x12_planefree/training_pause_87007232.pt
at t=87,007,232
```

- `run_history.jsonl` records:
  - `start_timesteps: 87007232`
  - `target_total_steps: 120000000`
  - `schedule_total_steps: 200000000`
  - unchanged contract fingerprint
  - clean Git commit `96fcbff`
- Exactly one trainer, one snapshot watcher, and one probe daemon were launched.
- W&B training ID remained `pupnnpq8` with `WANDB_RESUME=must`.

### Collapse after resume

| Step | Validation | Value loss | Mean/max KL | Frontier |
|---:|---:|---:|---:|---:|
| 88,006,656 | 0.755 | 16.0117 | 0.00423 / 0.01296 | 6 |
| 89,006,080 | 0.360 | 335.6356 | 0.01408 / 0.03614 | 3 |
| 90,005,504 | 0.315 | 272.7082 | 0.01123 / 0.03884 | 2 |
| 91,004,928 | 0.275 | 313.0047 | 0.00307 / 0.00975 | 2 |

The trainer was frozen with `SIGSTOP` after the exact 91M checkpoint to avoid
compounding the failure.

## Existing forensic observations to verify

These are observations, not conclusions:

1. The network state inside `training_pause_87007232.pt` is bit-identical to
   `probe_model_87007232.pt`.
2. All examined parameters and Adam moments are finite.
3. Adam did not obviously reset:
   - 87M: optimizer step 42,480, LR 0.00019124096;
   - 90M: optimizer step 43,941, LR 0.00018749312;
   - 91M: optimizer step 44,429, LR 0.00018624384.
4. Those learning rates match the registered 200M schedule.
5. The 91M standalone model and the network state inside
   `training_latest.pt` are bit-identical.
6. No trainer or W&B runtime error was found around the resume.
7. Parameters did not become NaN or Inf; failure is behavioral/numerical, not
   a simple non-finite explosion.
8. The probe curves did not show formation of the hypothesized medium/long
   landing-cell circuit during the collapse.

Checkpoint hashes at the time of the incident:

```text
training_80003072.pt
cc3bb17da49d2521a83c06285afe6d691fb1d643279afa6619a069249cd0db39

training_pause_87007232.pt
91c8e8ae197f027c7da457c98265d66cc8074ab8324c83051a723169b23c691e

training_90005504.pt
c6cb165879a6d160d79d21590e18c099256d7a29ea6686dec340a3c84241f490

training_latest.pt (91M)
613a90722e1d9ae6d92e3031b3f91f01b9bdbaf632d0d295f4f43c48bc123f94
```

Recompute hashes before use and report any mismatch.

## Specific restart-failure hypotheses to test

Do not limit the investigation to this list.

### Checkpoint consistency and save timing

- Is the checkpoint captured at a true PPO iteration boundary?
- Does it serialize a mutually consistent tuple of network, optimizer,
  observation, episode-start flags, recurrent state, environments, sampler,
  minibatch generator, and all RNG states?
- Could evaluation/checkpoint ordering leave `num_timesteps`, current
  observations, recurrent state, or environment state one rollout out of
  phase?
- Are any saved tensors or dictionaries aliased and later mutated despite the
  atomic file write or hard link?
- Did `SIGINT` during later MPS work have any plausible way to mutate or
  incompletely flush the already-written checkpoint?

### Restore ordering and hidden state

- Does startup construct environments, sample puzzles, initialize the network,
  initialize Adam, or consume RNG before restore?
- Are all such side effects subsequently overwritten, including sampler and
  per-environment RNG?
- Are carried DRC hidden/cell states restored with the correct environment
  ordering, dtype, shape, and device?
- Are observation arrays, episode-start flags, current puzzle indices, robot
  positions, elapsed steps, and RNG states restored exactly?
- Does environment restoration accidentally call reset or resample?

### RNG and MPS

- Are Python, NumPy, torch CPU, minibatch-generator, and MPS/device RNG states
  all saved and restored?
- Is `device_rng_state` meaningful and correctly restored for MPS in the
  installed torch version?
- Are there nondeterministic MPS kernels that make “exact resume” a stronger
  claim than the implementation can support?
- Could a process restart change kernel selection, convolution algorithms,
  thread scheduling, or floating-point accumulation enough to push an
  already-unstable critic into a different basin?

### Optimizer and schedule

- Does Adam state map to the same named parameters after reconstruction?
- Are optimizer tensors transferred to MPS correctly and without dtype
  changes?
- Is Adam `step` preserved for every parameter?
- Is the learning-rate update applied at the same point relative to rollout
  collection and PPO update before and after resume?
- Does changing only `resume_from`/`total_steps` alter any supposedly fixed
  contract or schedule behavior?

### Evaluation and training mode

- Does periodic evaluation leave the network in the same train/eval mode?
- Does evaluation consume any training RNG, mutate recurrent state, sampler
  state, environments, or model buffers?
- Is the fixed validation suite truly identical across the boundary?
- Re-evaluate retained 87–91M models independently on the fixed suite to rule
  out corrupted curve rows.

### Operational interactions

- A stale-source overwrite incident occurred earlier but was restored. Verify
  actual source blobs used at both launches.
- A duplicate low-priority probe daemon was later discovered. It read/collated
  model snapshots but should not have written trainer state. Verify that its
  file access, CPU pressure, or race with `model_latest.pt` could not affect
  training.
- The dedicated probe W&B client and training W&B client are separate. Verify
  no shared-process or file collision can affect the trainer.
- Check system sleep/wake, power mode, thermal throttling, and device changes.
  These should affect throughput, not gradients, but do not dismiss them
  without evidence.

## Recommended investigation sequence

### Tier 1: read-only audit

1. Read the implementation and existing exact-resume tests in full.
2. Verify Git blobs, config fingerprints, contract fingerprints, hashes,
   checkpoint metadata, run history, snapshot manifest, and process history.
3. Build a compact per-checkpoint table from 75–91M containing:
   validation, by-length/by-robots results, curriculum state, LR, policy loss,
   value loss, entropy, mean/max KL, clip fraction, explained variance,
   optimizer step, model norms, and relevant probe metrics.
4. Independently evaluate retained models at 80, 85, 87, 88, 89, 90, and 91M
   on the fixed periodic validation suite. Do not use the sealed test.
5. Audit every field of the 87M checkpoint and show that reconstructed objects
   match the serialized state before any new rollout.

### Tier 2: minimal deterministic resume tests

Add tests or a standalone forensic harness in a separate output directory:

1. Load the same 87M checkpoint in two fresh processes.
2. Digest and compare:
   - restored observation and episode-start arrays;
   - recurrent hidden/cell states;
   - all environment and sampler states;
   - global and minibatch RNG states;
   - first rollout actions, rewards, dones, values, log-probabilities, and
     recurrent states;
   - GAE/returns;
   - minibatch order;
   - first PPO update metrics and resulting model/optimizer hashes.
3. Run on CPU first if practical for determinism, then test MPS separately.
4. If two identical restores differ, identify the first divergent tensor or
   random draw. Do not summarize this merely as “MPS nondeterminism.”
5. Extend the repository’s resume tests to cover whatever gap is discovered.

Two independent restarts from 87M test restore reproducibility, but they do
**not** alone establish what an uninterrupted 87→89M trajectory would have
done.

### Tier 3: controlled causal replay

The strongest counterfactual starts from the same healthy exact checkpoint:
`training_80003072.pt` (80.0% validation, value loss 0.03745).

Subject to user approval and available compute, create two isolated arms with
the original config unchanged and W&B disabled/offline:

- **Arm U — uninterrupted:** restore the 80M checkpoint once and train
  continuously through at least 91M.
- **Arm R — restart:** restore the same 80M checkpoint, train to the same 87M
  boundary, stop cleanly after its exact checkpoint, start a new process from
  that checkpoint, and train through at least 91M.

Requirements:

- distinct output directories;
- identical bank, splits, seed, device, code, config, 200M schedule, and
  evaluation suite;
- no `clip_range_vf` or other rescue intervention;
- per-iteration/checkpoint hashes and telemetry detailed enough to locate the
  first divergence;
- ideally repeat on MPS if nondeterminism prevents a single-pair conclusion.

Interpretation:

- If Arm U reproduces the pre-pause value-loss escalation and 89M collapse,
  that is strong evidence the restart was not necessary for the failure.
- If Arm U remains healthy while Arm R reliably collapses, the restart is
  causal.
- If both outcomes vary across identical replays, characterize the stochastic
  instability and test whether restart-induced numerical variation changes
  its probability/timing.
- If neither reproduces the original trajectory, first determine why exact
  replay failed before drawing a causal conclusion.

## Deliverables

Write a new report, preferably:

```text
docs/Drc87MRestartCollapseInvestigation.md
```

The report must include:

1. a concise causal conclusion and confidence level;
2. an evidence table separating verified facts, inferences, and unresolved
   questions;
3. the first state/update at which a replay diverges, if any;
4. hashes and exact commands sufficient to reproduce the audit;
5. any new regression test or minimal reproducer;
6. whether the phrase “exact resume” is technically justified on CPU and MPS;
7. recommended disposition of the frozen primary run;
8. a separate recommendation for any rescue arm, clearly distinguished from
   the causal investigation.

Do not simply restate “value loss was rising before the pause.” The user’s
question is whether the restart causally changed what happened next. Answer
that with state-level evidence or a controlled counterfactual.

