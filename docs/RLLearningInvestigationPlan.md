# RL Learning Investigation and Remediation Plan

**Status:** execution handoff  
**Written:** 2026-07-17  
**Primary target:** 16×16, four-robot, optimal-length 1–8 Ricochet Robots  
**Reference fixture:** `artifacts/phase4_bank`  
**Reference harness:** `src/benchmarking/phase2.py`

This document is the execution plan for determining why reinforcement-learning
fine-tuning is weak, correcting the known experimental confounds, and producing
a trustworthy improved feed-forward baseline. It is intentionally explicit. A
future agent should follow the phases in order, satisfy every gate, and record
the requested evidence instead of improvising a large training run.

The immediate objective is **not** to build the DRC/recurrent agent. First make
the feed-forward PPO experiment causally interpretable and stable. A recurrent
comparison made before this plan is complete would still be confounded by
initialization, exploration, action legality, imitation learning, and evaluation
noise.

---

## 1. Read this before changing code

Read these files in order:

1. `CLAUDE.md` — operational rules and hard-won performance lessons.
2. This document in full.
3. `docs/GridCnnGeneralizationReport.md` — historical experiment arc.
4. `WorkInProgress.md`, especially the Phase 3, Phase 4, and Phase 4b entries.
5. `configs/phase4_16x16_big.yaml` and `configs/phase4b_bc_16x16.yaml`.
6. The relevant implementation sections in:
   - `src/benchmarking/phase2.py`
   - `src/models/grid_cnn.py`
   - `src/env/ricochet_env.py`
   - `src/env/criteria_env.py`
   - `src/env/puzzle_bank.py`

Treat this plan as a correction and refinement of the historical diagnosis.
Do not delete or rewrite old chronological notes. When an old conclusion is
superseded, append a clearly dated correction to `WorkInProgress.md` and update
the current interpretation in `docs/GridCnnGeneralizationReport.md`.

---

## 2. Executive diagnosis

There are two distinct problems that must not be conflated.

### 2.1 Historical from-scratch Phase 4 failure

The Phase 4 result does **not** isolate sparse-reward exploration failure.
The saved Phase 4 checkpoint used:

- SB3's default `Tanh` policy/value heads;
- `ortho_init=True`;
- SB3's small-gain action-head initialization;
- no behaviour cloning.

The successful Phase 4b run used, in the same commit that added BC:

- `ReLU` policy/value heads;
- `ortho_init=False`;
- unit-gain reinitialization of the action head before BC;
- behaviour cloning before PPO.

Therefore Phase 4 versus Phase 4b changed at least four material factors. The
historical language saying that Phase 4b was “identical but with BC” is too
strong and must not be used as a causal conclusion.

The saved Phase 4 model has a concrete saturation pathology:

- the residual convolutional trunk still produces state-varying features;
- the two Tanh MLP heads produce exactly state-independent latent vectors;
- action logits are constant across held-out states;
- the value prediction is constant across held-out states;
- stored-optimal action probability across expert states is approximately
  6.3%, effectively the 1/16 chance level;
- deterministic evaluation selects action 0 everywhere.

This is not a generic failure of structured observations. The older 8×8
`GridPlanesCnn` models used Tanh/orthogonal initialization successfully. The
pathology is specifically associated with scaling the unnormalized residual
extractor and its heads.

### 2.2 Current BC-warmstarted Phase 4b plateau

Phase 4b is competent but PPO adds little net aggregate performance:

- BC training accuracy: 89.3% on approximately 210,947 expert state/action
  pairs from 41,395 training puzzles;
- post-BC, pre-PPO validation: 30.6% stochastic / 24.8% deterministic;
- final validation: 30.6% stochastic / 24.0% deterministic;
- final held-out test: 28.8% stochastic / 24.2% deterministic;
- final test by robots moved: 53% / 19% / 4% for one-/two-/three-robot plans.

PPO did redistribute some competence toward longer puzzles, but it did not
produce a clear aggregate improvement over BC alone. The current evidence
points to several interacting causes:

1. sparse success reward on length ≥3 and especially multi-robot puzzles;
2. legal action space containing many actions that leave the board unchanged;
3. compounding BC errors after leaving the expert trajectory;
4. weak generalization to blocker-robot choices;
5. a mostly location-specific flatten-and-linear projection;
6. a single optimizer updating actor, untrained critic, and shared extractor;
7. curriculum stages advancing almost immediately after BC;
8. insufficient diagnostics and noisy checkpoint evaluation.

The task is to isolate these factors with controlled experiments, not to choose
one explanation in advance.

---

## 3. Evidence that is already established

Do not spend another run rediscovering the following facts.

### 3.1 Environment and data path

- The structured `image` observation is lossless for the board state: four wall
  planes, goal, target-robot mask, and one plane per robot.
- Solver action encoding and environment action decoding both use
  `robot_id * 4 + direction`.
- Stored optimal plans replay successfully through the environment.
- Goal detection, rewards, and success reporting agree on ordinary episodes.
- Train/validation/test membership is disjoint by `layout_hash`.
- The Phase 4 bank has 55,143 puzzles and fixture fingerprint prefix
  `3579113b230a`.
- The Phase 4 split fingerprint is
  `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`.
- The current complete test suite passes: 88 passed, with nine legacy recurrent
  tests intentionally skipped.

### 3.2 Sparse exploration

On the Phase 4 balanced held-out suite, random success by optimal length is
approximately:

| Length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random success | 35.0% | 19.6% | 5.0% | 1.1% | 0.3% | 0% | 0% | 0% |

Sparse reward is therefore a real difficulty. It is not yet proven to be the
only or dominant cause in a correctly initialized from-scratch residual model.

### 3.3 Action waste and recovery

A read-only inference probe on the Phase 4b final checkpoint found:

| Evaluation | Unmasked | Mask actions that do not move |
|---|---:|---:|
| Deterministic success | 24.2% | 30.0% |
| Stochastic success | 28.2% | 33.3% |
| Deterministic no-movement rate | 50.6% | 0% |
| Stochastic no-movement rate | 32.5% | 0% |
| Three-robot stochastic success | 1.3% | 11.3% |

This was a diagnostic, not a formal multi-seed result, but it makes movement
masking the highest-value low-cost intervention.

Extending the evaluation horizon from 20 to 60 raised stochastic success from
28.5% to 35.3%, but deterministic success stayed at 24.2%. Greedy failures are
looping rather than merely needing a few more moves. Masking is preferable to
tripling the horizon because it improves efficiency and attacks a known source
of self-loops.

### 3.4 Generalization and imitation gaps

The BC action classes are already balanced: each of the 16 actions contributes
roughly 6% of expert pairs. Simple action-class rebalancing is not the answer.

Difficulty is not balanced in the expert pairs:

- one-robot expert pairs: 63,442;
- two-robot expert pairs: 123,975;
- three-robot expert pairs: 23,530.

On the final Phase 4b model, agreement with the single stored optimal action on
held-out initial states is approximately:

- one-robot plans: 56%;
- two-robot plans: 29%;
- three-robot plans: 17%.

On all held-out states along stored expert trajectories, agreement is about 59%,
versus about 79% on the fixed training probe. Some disagreement may correspond
to alternate optimal actions, so this metric is a lower bound on decision
quality, but the train/held-out and robots-moved gaps are still meaningful.

### 3.5 Architecture generalization risk

`GridPlanesResNet` preserves spatial resolution in its convolutional trunk, but
then flattens `32 × 16 × 16` activations into a dense projection. In the Phase
4b policy:

- projection parameters: 4,194,816;
- total policy parameters: 6,148,145;
- projection share: 68.2%.

Most parameters are therefore location-specific. This can memorize layouts and
does not encode the permutation symmetry of non-target robots.

### 3.6 Curriculum and evaluation weakness

W&B history shows Phase 4b reached:

- stage 1 by about 20k PPO steps;
- stage 2 by about 41k PPO steps;
- final stage by about 82k PPO steps.

More than 99% of the 10.24M-step run was therefore on the full length 1–8
distribution. The nominal staged curriculum barely constrained fine-tuning.

Checkpoint validation used one stochastic episode per puzzle. The reported
7.2M “peak” was 40/121 successes versus 37/121 at the endpoint—only three
episodes—and the 7.2M weights were not saved. Treat the apparent late fade as a
hypothesis, not a fact.

---

## 4. Scope, non-goals, and definition of done

### 4.1 In scope

- Make policy initialization and architecture semantics explicit and recorded.
- Add checkpointing and diagnostic logging sufficient to explain failures.
- Implement and evaluate movement-action masking.
- Run the missing current-architecture, no-BC control.
- Run a controlled BC × masking experiment matrix.
- Improve BC/PPO transfer if the controlled results justify it.
- Improve multi-robot generalization using data and then architecture changes.
- Produce a formal three-seed feed-forward result on a sealed confirmation
  suite.

### 4.2 Not yet in scope

- DRC, ConvLSTM, RecurrentPPO, or mechanistic interpretability experiments.
- Replacing PPO before proving that PPO under the corrected setup fails.
- Regenerating or mutating `artifacts/phase4_bank`.
- Changing the fixture or split fingerprints.
- Broad hyperparameter searches with no causal question.
- Selecting models using the sealed confirmation suite.

### 4.3 Definition of done

This plan is complete only when all of the following are true:

1. The harness records effective activation, initialization, action masking,
   BC, PPO, reward, horizon, fixture, and split settings.
2. Intermediate checkpoint weights and per-checkpoint metrics are saved.
3. PPO loss/optimization and activation-health telemetry are available.
4. A current ReLU/`ortho_init=False`, no-BC control has been run.
5. Movement masking has a tested training and evaluation implementation.
6. The BC × masking causal matrix has at least one screening seed per cell.
7. The chosen recipe has three training seeds.
8. The final recipe is selected using validation only, then evaluated once on
   a separately sealed confirmation suite.
9. `WorkInProgress.md`, `docs/GridCnnGeneralizationReport.md`, configs, results,
   and commands agree.

The aspirational performance target is:

- mean stochastic confirmation success ≥32%;
- mean deterministic confirmation success ≥28%;
- two-robot stochastic success ≥22%;
- three-robot stochastic success ≥10%;
- no-movement action rate exactly 0 when masking is enabled;
- no seed below the 8.5% target-robot-only baseline.

These targets are planning gates, not guarantees. A well-instrumented negative
result is acceptable if it identifies the failing mechanism.

---

## 5. Non-negotiable operational rules

1. Use `.venv-phase2/bin/python` and Python 3.11.
2. Run `git status --short` before editing. Preserve unrelated user changes.
3. Use `apply_patch` for edits.
4. Never mutate or delete `artifacts/phase4_bank`.
5. Never change fixture or split membership as a side effect of refactoring.
6. After bank/split/sampler changes, verify both recorded fingerprints.
7. Do not use `pandas.iterrows()` in reset or per-episode paths.
8. Benchmark throughput through `_training_factory`, not a synthetic base env.
9. Run the full tests before launching any experiment.
10. Launch jobs longer than approximately 15 minutes detached with `nohup` and
    `caffeinate`.
11. Use a unique config, output directory, log file, and W&B run name for every
    experiment. Never overwrite an earlier result.
12. Append experiments chronologically to `WorkInProgress.md`.
13. Never tune on the sealed confirmation suite.
14. Do not claim a multi-seed milestone from seed 11 alone.

Standard commands:

```bash
.venv-phase2/bin/python -m pytest src/tests/ -q

nohup caffeinate -is .venv-phase2/bin/python \
  src/scripts/run_phase2_benchmark.py \
  --config configs/<experiment>.yaml \
  --bank-dir artifacts/phase4_bank \
  --output-dir artifacts/phase2_experiments/<experiment> \
  > artifacts/phase2_experiments/<experiment>.log 2>&1 &
```

Record the PID, start time, Git commit, config path, output directory, and log
path in `WorkInProgress.md` when starting a long run.

---

## 6. Phase 0 — preserve evidence and correct provenance

**Purpose:** ensure a future run can be interpreted and reproduced before
spending more compute.

### 6.1 Add explicit policy-construction fields

Extend `Phase2Config` in `src/benchmarking/phase2.py` with validated fields such
as:

```text
policy_activation: relu | tanh
ortho_init: bool
bc_reinitialize_action_head: bool
bc_action_head_gain: float
use_action_masking: bool
curriculum_max_level: int | null
```

Recommended defaults for new structured-plane experiments:

```yaml
policy_activation: relu
ortho_init: false
bc_reinitialize_action_head: true
bc_action_head_gain: 1.0
use_action_masking: false
curriculum_max_level: null
```

Requirements:

- `_build_model` must derive `activation_fn` from config instead of hard-coding
  it.
- `_bc_pretrain` must only reinitialize the action head when explicitly asked.
- Record requested and effective values in `results.json` and per-seed result.
- Record the complete effective `policy_kwargs`, not only class, parameter
  count, and spaces.
- Do not infer historical settings from today's defaults. The saved Phase 4
  model itself is the source of truth for the historical checkpoint.
- Do not silently change stock-CNN behavior.

Add config-validation tests for unknown activation names, non-positive gains,
and incompatible masking/algorithm selections.

`curriculum_max_level` is an experiment control, not part of the bank
definition. When set, the shared curriculum manager must not advance beyond
that zero-based level. Validate it against the configured stage list and record
it in curriculum checkpoint state. This permits a true stage-0 experiment while
keeping the Phase 4 fixture definition unchanged.

### 6.2 Persist intermediate evaluable checkpoints

Inside the checkpoint loop in `train_seed`, after evaluation, save:

```text
seed_<seed>/checkpoints/step_<actual_timesteps>/model.zip
seed_<seed>/checkpoints/step_<actual_timesteps>/evaluation.json
```

The step-0 post-BC checkpoint must be saved. This is essential for distinguishing
BC competence from PPO changes.

Also save curriculum state alongside each checkpoint:

```text
current_level
episode_count
level_start_episode
success_rate_window
episodes_at_level
window_success_rate
```

Exact mid-rollout resume is not required in the first implementation. Be honest
in documentation: these checkpoints must be evaluable and selectable, but not
claimed as bit-exact resumable training states unless RNG, sampler, vector-env,
optimizer, and rollout state are all restored and tested.

Add tests that a tiny benchmark produces step-0 and final checkpoint files and
that their metadata matches `actual_timesteps`.

### 6.3 Add optimization telemetry

The existing W&B callback logs episode and curriculum summaries but not PPO
training health. Add the following at least once every five rollouts:

```text
train/approx_kl
train/clip_fraction
train/entropy_loss
train/explained_variance
train/learning_rate
train/policy_gradient_loss
train/value_loss
train/loss
train/n_updates
```

SB3 performs `_on_rollout_end` before the following optimization step. Do not
mistakenly label stale metrics as belonging to the just-finished rollout. A
practical implementation is to read the previous optimization metrics in
`_on_rollout_start`, after the prior `train()` call has populated the logger.
Document this one-update offset.

Also log:

```text
health/extractor_output_mean
health/extractor_output_std
health/extractor_output_abs_max
health/extractor_zero_fraction
health/policy_latent_between_state_variance
health/value_latent_between_state_variance
health/logit_between_state_variance
health/value_mean
health/value_std
health/policy_normalized_entropy
health/mean_max_action_probability
health/no_movement_probability_mass
```

Compute health metrics on a small fixed **training** probe, not on the sealed
confirmation suite. Keep the probe observations in CPU memory and evaluate them
in batches. Do not run a full validation rollout every five rollouts.

Add warning/abort thresholds, but initially warn rather than abort until normal
ranges are established. A policy-latent or logit between-state variance below
`1e-10` for multiple reports is a clear collapse signal.

### 6.4 Fix W&B step axes

BC logs currently advance W&B's global step to 25, after which the post-BC
evaluation tries to log at PPO step 0 and is ignored. Define independent step
metrics, for example:

```text
bc/epoch as the step metric for bc/*
train/timesteps as the step metric for rollout/*, train/*, health/*, eval/*
```

Confirm that the post-BC step-0 evaluation appears in W&B.

### 6.5 Add a policy-health audit script

Create `src/scripts/audit_policy_health.py` with CLI arguments for checkpoint,
config, bank, split manifest, and suite. It must report:

- policy kwargs and initialization settings from the checkpoint;
- parameter counts by component;
- feature, policy-latent, value-latent, logit, and value statistics;
- stored-optimal top-1 agreement and probability;
- normalized entropy and maximum action probability;
- valid-action count and no-movement probability mass;
- breakdowns by optimal length and robots moved.

Use this script to machine-record the already observed Phase 4 saturation and
Phase 4b health. Store JSON plus a human-readable summary under a new
`artifacts/diagnostics/` subdirectory; do not modify the original run results.

### Phase 0 gate

Do not continue until:

- full tests pass;
- a tiny benchmark saves intermediate checkpoints;
- effective policy settings appear in results;
- PPO and activation metrics appear in a short W&B or local logging smoke run;
- the audit script distinguishes the constant Phase 4 heads from Phase 4b.

---

## 7. Phase 1 — seal evaluation and stop test-set tuning

The current 120-puzzle `test_suite` has already been inspected repeatedly in
reports and in the masking/horizon diagnostics. Treat it as a development test,
not as untouched confirmation data.

### 7.1 Create a separate confirmation manifest

Do **not** change the existing split manifest or its fingerprint. Create a new
script or harness helper that:

1. loads the recorded Phase 4 split manifest;
2. takes candidates only from `memberships["test"]`;
3. excludes every puzzle **and every layout hash** represented in the existing
   `test_suite`, so another target variant of a seen layout cannot enter the
   confirmation suite;
4. selects a balanced suite by `(optimal_length, robots_moved)` using a new,
   recorded selection seed;
5. selects up to 20 puzzles per non-sparse achievable stratum;
6. records the parent split fingerprint, excluded IDs, selected IDs, histogram,
   seed, and its own SHA-256 fingerprint;
7. writes `artifacts/phase2_experiments/phase4_confirmation_manifest.json`.

The confirmation suite must remain disjoint from training, validation, and the
development test at both puzzle-ID and layout-hash level. Add automated tests of
all of these disjointness properties; do not check only board/puzzle hashes.

Once created, do not evaluate any learning model on it until a final recipe and
checkpoint-selection rule are locked. Oracle replay and fixed random/heuristic
baselines may be computed once and stored, but do not inspect model performance.

### 7.2 Add a `run_final_test` control

Add a config field such as `run_final_test: bool = true`. Screening experiments
must set it to `false` and stop after validation. Results should clearly say
`test_status: sealed_not_run` rather than emitting empty or zero-valued test
metrics.

The final locked three-seed run may evaluate the development test for continuity
and then, exactly once, the confirmation suite.

### 7.3 Improve validation precision

For screening runs use at least:

```yaml
validation_episodes_per_puzzle: 5
```

For final candidate selection prefer 10 stochastic episodes per puzzle if
runtime is acceptable. Always report deterministic evaluation separately.

Uncertainty must be clustered by puzzle and summarized across training seeds.
Do not treat repeated stochastic episodes of the same puzzle as fully
independent. Implement a puzzle-level bootstrap or at least state clearly when
an interval ignores within-puzzle correlation.

### Phase 1 gate

- The confirmation manifest exists and has a fingerprint.
- No candidate model has been evaluated on it.
- Screening configs can skip final-test evaluation.
- Validation metrics include puzzle-level uncertainty.

---

## 8. Phase 2 — implement movement-action masking

**Motivation:** this is the cheapest intervention with direct positive evidence.

### 8.1 Define the mask contract

For `include_noop=False`, an action is allowed exactly when applying
`robot_id * 4 + direction` changes that robot's position.

Add an environment method with one stable name, preferably:

```python
def action_masks(self) -> np.ndarray:
    """Return a boolean vector of shape (action_space.n,)."""
```

Requirements:

- `True` means allowed.
- The method must not mutate the board.
- At least one movement action must be allowed on every valid nonterminal board.
- If an explicit thinking/no-op action is enabled in a later recurrent
  experiment, define its mask semantics separately; do not silently allow it in
  this feed-forward experiment.
- Forward the method through `CriteriaFilteredEnv` and
  `BankCurriculumWrapper`, because those are the real training envs.

The mask may use `slide_until_blocked` or equivalent logic. Avoid constructing
16 cloned boards if a direct endpoint calculation is measurably faster. First
implement correctly, then benchmark the real factory.

### 8.2 Integrate MaskablePPO

`sb3-contrib==2.9.0` is already installed and declared. When
`use_action_masking=true`:

- construct `sb3_contrib.MaskablePPO` instead of ordinary SB3 `PPO`;
- use the mask-aware policy and prediction path;
- pass masks during training collection automatically through the environment;
- pass masks explicitly in benchmark evaluation;
- ensure model loading knows which algorithm class produced the checkpoint;
- record algorithm class and masking in the manifest.

Do not implement masking only at evaluation time for the formal experiment.
The earlier evaluation-only probe is evidence for the hypothesis, not the final
training method.

### 8.3 BC compatibility

Every stored expert action should be allowed by the corresponding mask. Add a
test over a representative bank sample and fail loudly on any violation.

BC can initially compute unmasked cross-entropy because every target is valid,
but the preferred implementation supplies the mask to the maskable policy's
distribution so pretraining and rollout semantics match. If that adds excessive
complexity, document the difference and verify that masked and unmasked BC
losses are numerically close on expert states.

### 8.4 Required tests

Add tests that:

1. mask shape and dtype match the action space;
2. each allowed action moves and each masked action does not move;
3. masks are unchanged by repeated calls;
4. wrapper forwarding returns the current inner board's mask;
5. all stored expert actions are allowed;
6. a tiny MaskablePPO smoke run completes;
7. masked deterministic and stochastic evaluation produce zero no-movement
   actions;
8. ordinary PPO behavior is unchanged when masking is false.

### 8.5 Throughput benchmark

Measure reset-inclusive steps/second on `_training_factory` for masked and
unmasked environments. A mask implementation that destroys throughput is not
acceptable. Record:

- environment steps/second;
- reset time;
- mask-computation time;
- percentage overhead.

Target less than 10% throughput loss. If overhead is larger, optimize the mask
calculation before long runs.

### Phase 2 gate

- All tests pass.
- Expert actions are always valid.
- Masked evaluation has exactly zero no-movement actions.
- Real-factory throughput loss is understood and preferably <10%.

---

## 9. Phase 3 — run the missing from-scratch controls

This phase answers whether correctly initialized PPO can learn without BC.

### 9.1 Short stage-0 control

Create `configs/phase5a_current_nobc_stage0.yaml` based on Phase 4 with:

```yaml
# Keep these at 1 and 8: they are part of the existing bank definition.
min_optimal_length: 1
max_optimal_length: 8
curriculum_stage_max_lengths: [2, 4, 6, 8]
curriculum_max_level: 0
policy: grid_resnet
policy_activation: relu
ortho_init: false
bc_pretrain_epochs: 0
use_action_masking: false
training_seeds: [11]
checkpoints: [0, 102400, 256000, 512000, 1024000]
run_final_test: false
validation_episodes_per_puzzle: 5
```

Use the existing Phase 4 bank; do not rebuild it. The harness validates an
existing bank using `min_optimal_length`, `max_optimal_length`, robot bounds,
solver limits, attempted seeds, and board spec. Changing those fields merely to
restrict training will fail fixture validation. `curriculum_max_level: 0` must
freeze training at the first length 1–2 stage while leaving the bank and the
full validation distribution intact. Run through the real factory.

At every checkpoint inspect:

- online curriculum success;
- validation success by length and robots moved;
- no-movement rate;
- entropy;
- KL and clipping;
- explained variance and value loss;
- activation-health metrics;
- stored-optimal action probability on the fixed training probe.

Interpretation:

- If held-out performance improves and activations remain healthy, the corrected
  architecture can learn and the old Phase 4 result was partly an initialization
  failure.
- If training improves but held-out does not, prioritize generalization and
  architecture/data augmentation.
- If neither improves while activations are healthy, sparse reward/optimization
  remains the likely cause.
- If latents become state-independent, stop immediately and fix normalization or
  optimization before any full-distribution run.

### 9.2 Matched masked stage-0 control

Create `configs/phase5b_current_nobc_masked_stage0.yaml`, changing only:

```yaml
use_action_masking: true
```

Use the same seed, checkpoints, fixture, split, and validation protocol.

The masking hypothesis is promising if it yields any of:

- ≥3 percentage-point validation improvement at matched steps;
- materially faster curriculum mastery;
- lower variance across checkpoints;
- clear improvement on two-robot length-2 puzzles;
- zero no-movement rate with no compensating collapse in entropy or KL.

### 9.3 Optional update-ratio ablation

Only if both stage-0 controls remain flat with healthy activations, compare the
Phase 4 update settings against the earlier stronger update ratio:

```yaml
# Current Phase 4
batch_size: 1024
n_epochs: 2

# Ablation
batch_size: 256 or 512
n_epochs: 4
```

Change no other factor. Monitor throughput, KL, clip fraction, and gradient
health. Do not assume more epochs are better; reject settings that produce high
KL, excessive clipping, or renewed feature-scale growth.

### Phase 3 gate

Write a short decision record in `WorkInProgress.md` answering:

1. Can current-initialization PPO learn length 1–2 from scratch?
2. Does masking improve sample efficiency?
3. Are policy/value representations healthy?
4. Is the update ratio adequate?

Do not proceed to a 10M-step factorial until these questions have evidence.

---

## 10. Phase 4 — controlled BC × masking matrix

Run the following under the **same explicit current architecture**:

| Cell | BC | Masking | Purpose |
|---|---:|---:|---|
| A | No | No | corrected from-scratch baseline |
| B | No | Yes | isolate masking for exploration |
| C | Yes | No | instrumented reproduction of current recipe |
| D | Yes | Yes | candidate improved recipe |

### 10.1 Screening protocol

Use seed 11 and a 2.048M-step budget first:

```yaml
training_seeds: [11]
checkpoints: [0, 102400, 256000, 512000, 1024000, 2048000]
run_final_test: false
validation_episodes_per_puzzle: 5
```

Keep fixed:

- Phase 4 bank and split;
- 16×16 / four robots;
- structured observations;
- residual architecture size;
- ReLU and `ortho_init=false`;
- curriculum definition;
- reward and horizon;
- PPO hyperparameters unless the stage-0 update-ratio ablation already selected
  a replacement;
- checkpoint and evaluation schedule.

Use names such as:

```text
phase5c_nobc_unmasked_screen
phase5d_nobc_masked_screen
phase5e_bc_unmasked_screen
phase5f_bc_masked_screen
```

### 10.2 Metrics and selection

The primary screening metrics are:

1. validation stochastic success, clustered by puzzle;
2. deterministic validation success;
3. joint `(optimal_length, robots_moved)` success;
4. two- and three-robot success;
5. no-movement rate;
6. optimality gap on successes;
7. activation and optimization health;
8. wall-clock steps/second.

Do not select only by aggregate success. Prefer a candidate that improves
multi-robot cells without destroying easy strata.

### 10.3 Promotion rule

Promote at most two cells to the full 10.24M budget. Cell D should be one unless
it is clearly unstable. Promote a no-BC cell only if it shows a meaningful
upward learning curve; a flat no-BC run does not need another eight million
steps merely for completeness.

For full single-seed runs use the original checkpoint schedule plus additional
early checkpoints if inexpensive. Save every model.

### Phase 4 gate

- The BC contribution is estimated under matched initialization.
- The masking contribution is estimated with and without BC.
- At least one stable candidate exists, or the evidence clearly identifies the
  next failing mechanism.

---

## 11. Phase 5 — stabilize BC-to-PPO transfer

Perform this phase only if BC is still required and PPO fails to improve it
reliably.

### 11.1 Add held-out expert-state validation

Extend BC evaluation to use expert trajectories from the existing validation
membership, never the confirmation suite. Log at every epoch:

- train and validation cross-entropy;
- top-1 stored-action agreement;
- target action probability;
- entropy and calibration;
- metrics by optimal length and robots moved;
- first-state versus all-expert-state metrics.

Save the best BC checkpoint by held-out BC loss or a predeclared combination of
held-out action agreement and rollout success. Do not choose solely by training
accuracy.

### 11.2 Separate actor and critic adaptation

The current shared optimizer begins PPO with a competent actor/extractor and an
untrained critic. Compare, one factor at a time:

1. lower global PPO learning rate: `1e-4`;
2. actor/extractor `3e-5` or `1e-4`, critic `3e-4` using optimizer parameter
   groups;
3. entropy coefficient `0`, `0.001`, and current `0.01`;
4. a short critic warm-up with actor logits frozen;
5. an annealed BC or KL-to-BC-prior loss during PPO.

Recommended first comparison:

```text
Current: actor/extractor/value all 3e-4, ent_coef 0.01
Candidate: actor/extractor 1e-4, value 3e-4, ent_coef 0.001
```

If implementing a BC/KL auxiliary loss:

- snapshot the post-BC policy;
- apply the penalty only on a fixed or replayed expert minibatch;
- log PPO policy loss, BC/KL loss, and their gradient norms separately;
- anneal the coefficient to zero over a declared number of PPO steps;
- do not allow the auxiliary batch to include validation or confirmation data.

### 11.3 Slow and strengthen curriculum gates

The existing average success gate lets strong easy strata hide weak new strata.
Change progression so a stage requires:

- a minimum number of PPO steps or rollouts at that stage, not only episodes;
- sufficient samples for each newly introduced stratum;
- a threshold on the new maximum-length strata or a minimum across cells;
- no activation-health failure.

Keep earlier lengths in the sampling mix to avoid forgetting. Record the exact
episode distribution per joint stratum, rather than assuming configured weights
equal realized weights.

Do not use the stall safety valve as evidence of mastery. Record separately
whether a transition was caused by mastery or forced advancement.

### 11.4 Decide whether “fade” is real

With saved checkpoints and ≥5 validation repetitions, compare post-BC, 2M, 4M,
7.2M, and final weights using identical evaluation seeds. Use paired
puzzle-level bootstrap differences.

Only call the change catastrophic forgetting if the confidence interval and
per-stratum results support it. Three successes out of 121 are not sufficient.

### Phase 5 gate

Promote a fine-tuning recipe only if it beats its own post-BC checkpoint on
validation or clearly improves the hard joint cells without a material aggregate
loss.

---

## 12. Phase 6 — improve multi-robot generalization

Use this order: data weighting, recovery data, symmetry, then architecture.

### 12.1 Generalize robot-band weighting

Replace the misleading single `two_robot_band_weight` control with explicit
weights, for example:

```yaml
robots_moved_band_weights:
  1: 1.0
  2: 1.5
  3: 3.0
```

Apply weights separately and explicitly to:

- PPO curriculum sampling;
- BC example sampling or loss weighting.

Record realized puzzle and expert-pair fractions by joint stratum. Do not assume
that puzzle weights imply pair weights, because longer plans contribute more
expert states.

Start with loss weighting or a balanced minibatch sampler rather than duplicating
the entire in-memory dataset.

### 12.2 Handle ambiguous optimal labels

The bank stores one BFS-optimal plan, but multiple first actions may be equally
optimal. Single-label cross-entropy penalizes valid alternatives and can create
overconfidence.

Add an offline analysis that, for a manageable sample, enumerates all optimal
first actions or computes whether alternate first actions retain the same
optimal distance. Measure ambiguity by stratum.

If ambiguity is material, train against a multi-hot set of optimal actions using
log-probability mass over the valid optimal set rather than one arbitrary label.
Do not change the bank fingerprint merely to add derived cached analysis; store
derived labels in a separate artifact keyed by puzzle ID and solver version.

### 12.3 Add recovery states with DAgger-style correction

BC currently sees expert states only. A single rollout mistake enters an unseen
state and errors compound with solution length.

Implement a bounded offline loop:

1. roll out the current policy on training puzzles only;
2. collect visited states where the policy diverges or loops;
3. solve those states within declared depth/node limits;
4. add the solver's corrective action and solver metadata to a separate recovery
   dataset;
5. mix recovery and original expert states during BC or PPO auxiliary updates;
6. evaluate on validation, never confirmation.

Track how many queried states are solved, hit limits, or are duplicates. Never
silently label solver cutoffs as unsolvable.

### 12.4 Add exact symmetry augmentation

Ricochet Robots admits board rotations/reflections and some robot permutations.
Implement transformations only with exhaustive transition-equivalence tests:

- remap wall planes correctly;
- remap goal and target mask;
- remap robot channels;
- remap action robot IDs and directions;
- verify that “transform then act” equals “act then transform”;
- verify that transformed expert trajectories still solve optimally.

Start with the eight dihedral board transformations. Robot permutation is more
delicate because target identity and action IDs must change together; add it only
after board transforms are correct.

### 12.5 Replace the location-specific action head if needed

If data interventions leave a large train/validation blocker gap, build a more
relational policy. Preferred design direction:

1. preserve a spatial feature map through the trunk;
2. gather shared features at each robot location;
3. include whether that robot is the target;
4. use a shared robot-direction scorer;
5. emit four direction logits per robot;
6. concatenate to the existing 16-action order;
7. use shared or permutation-aware blocker processing;
8. normalize feature scale with GroupNorm/LayerNorm or controlled residual
   scaling.

Avoid another giant flatten-and-linear layer unless an ablation proves it is
needed. Preserve exact board geometry and enough receptive field to cover the
16×16 board.

Before RL, require the new architecture to:

- overfit a tiny expert set;
- learn BC with healthy gradients;
- report held-out expert action metrics;
- pass transformed-transition tests;
- maintain state-dependent logits and values.

### Phase 6 gate

The selected solution should improve two-/three-robot validation cells, not just
aggregate success on one-robot puzzles.

---

## 13. Phase 7 — resolve horizon and truncation semantics

This is a real contract question, not a tuning detail.

Current behavior marks failure at `max_steps` as Gymnasium truncation. SB3 then
adds `gamma * V(terminal_observation)` to the final reward and bootstraps as if
the same puzzle continued. The observation does not include remaining time.

Choose one intended task definition:

### Option A — 20 moves is part of the task

- Reaching 20 moves without success is a terminal failure, not an external time
  limit.
- Consider an explicit terminal failure reward.
- Include remaining steps in the observation if optimal behavior should depend
  on the budget.
- Regenerate BC observations only if the observation shape changes; the puzzle
  bank itself need not change.

### Option B — 20 moves is only a training cutoff

- Keep truncation and bootstrapping.
- Train with a horizon large enough not to truncate plausible recovery behavior,
  or randomize the cutoff.
- Continue evaluating success-at-20 separately if that is the reported metric.
- Clearly state that PPO optimizes a continuing-puzzle approximation rather than
  the strict solve-within-20 objective.

Run a controlled ablation only after masking, because masking already removes a
large class of wasted steps. Compare horizon/termination semantics with all
other settings fixed. Log timeout fraction, terminal values, and success versus
allowed moves.

Do not combine a horizon change with architecture or BC changes.

---

## 14. Final three-seed protocol

Once the recipe and checkpoint-selection rule are locked:

1. Freeze the config in a new YAML.
2. Use seeds `[11, 29, 47]`.
3. Run from clean output directories on one Git commit.
4. Validate fixture and split fingerprints at startup.
5. Save step-0 and every PPO checkpoint.
6. Select a checkpoint independently per seed using the predeclared validation
   rule, or use one fixed timestep for every seed. State which rule was chosen
   before seeing confirmation results.
7. Record throughput and wall time.
8. Evaluate the development validation/test suites.
9. Evaluate the sealed confirmation suite exactly once.
10. Aggregate across seeds and bootstrap across puzzles.

Report:

- deterministic and stochastic success;
- sample mean, sample standard deviation, and standard error across seeds;
- puzzle-clustered confidence intervals;
- joint length × robots-moved table;
- no-movement rate;
- episode length and successful optimality gap;
- action distribution and target-robot action mass;
- BC-only versus selected PPO checkpoint;
- curriculum transition times and reasons;
- optimization and activation-health summaries;
- parameter count and effective policy kwargs;
- fixture, split, confirmation, Git, Python, PyTorch, SB3, and sb3-contrib
  provenance.

The confirmation result is the headline. If it disagrees materially with
development validation, report the discrepancy rather than selecting another
model.

---

## 15. Required experiment record template

Append one block like this to `WorkInProgress.md` for every run:

```markdown
### <experiment name> — <date/time>

- Question:
- Single changed factor(s):
- Config:
- Git commit / dirty state:
- Bank and fixture fingerprint:
- Split fingerprint:
- Confirmation suite touched: no
- Seed(s):
- Requested / actual timesteps:
- Device:
- Output directory:
- Log path:
- W&B run:
- PID and launch command:
- Expected duration:
- Result:
- Validation by length:
- Validation by robots moved:
- No-movement rate:
- Curriculum transitions:
- PPO/activation health:
- Throughput:
- Interpretation:
- Decision / next experiment:
```

Never use “failed to learn” without specifying whether the failure was:

- no training improvement;
- training-only memorization;
- activation/logit collapse;
- sparse reward/no successful rollouts;
- critic failure;
- destructive fine-tuning;
- action looping;
- validation noise;
- or failure on particular difficulty strata.

---

## 16. Suggested implementation commits

Keep changes reviewable. A good sequence is:

1. `test: pin phase4 policy provenance and health diagnostics`
2. `feat: make policy activation and initialization explicit`
3. `feat: save per-checkpoint models and evaluation metadata`
4. `feat: log PPO optimization and activation health`
5. `fix: separate BC and PPO W&B step axes`
6. `feat: seal a confirmation suite and allow validation-only runs`
7. `feat: expose movement action masks through bank wrappers`
8. `feat: add MaskablePPO to deterministic benchmark harness`
9. `test: verify expert actions and masked evaluation contracts`
10. `exp: add current-init no-BC stage-0 controls`
11. `exp: add BC × masking screening matrix`

Do not mix architecture redesign, DAgger, timeout semantics, and masking in one
commit or experiment.

---

## 17. Stop conditions and escalation rules

Stop a run early and investigate if any of these persist across several reports:

- policy/logit between-state variance approaches zero;
- feature or latent magnitudes grow by orders of magnitude;
- NaN/Inf in observations, logits, values, losses, or gradients;
- KL or clipping remains extreme;
- explained variance is strongly negative and deteriorating;
- masked rollouts contain a no-movement action;
- realized curriculum sampling omits configured strata;
- fixture or split fingerprint differs;
- throughput drops by more than 20% without explanation;
- W&B/checkpoint outputs are missing;
- the run uses the confirmation suite before recipe lock.

Do not respond to a stopped run by increasing the total timestep budget. First
identify whether the problem is data, optimization, representation, environment
semantics, or instrumentation.

Ask the user before:

- regenerating the Phase 4 bank;
- changing the formal confirmation membership after it is sealed;
- adding a materially different RL algorithm;
- beginning the DRC/recurrent work;
- launching a large multi-seed run expected to consume substantially more than
  the existing Phase 4b budget.

---

## 18. Priority summary for the executing agent

If there is any ambiguity, follow this order:

1. **Make runs reproducible and observable.** Explicit init, checkpoints,
   losses, activations, correct W&B axes.
2. **Seal confirmation data.** Stop tuning on the current exposed test suite.
3. **Implement movement masking.** It has the strongest cheap causal evidence.
4. **Run the missing current-init no-BC control.** This repairs the Phase 4/4b
   causal comparison.
5. **Run BC × masking screening.** Promote only evidence-backed candidates.
6. **Stabilize PPO after BC.** Separate actor/critic adaptation, reduce entropy,
   or retain the BC prior if needed.
7. **Target multi-robot generalization.** Balanced BC, recovery states,
   symmetry, then a relational head.
8. **Resolve horizon semantics.** Do it as a separate controlled experiment.
9. **Run three seeds and confirmation once.** Only then update the headline
   milestone.
10. **Consider DRC.** At that point recurrence will answer a clean planning
    question rather than compensating for unresolved baseline defects.
