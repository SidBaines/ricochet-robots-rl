# RL Learning Remediation Execution Log

This is the live, detailed handoff document for
`docs/RLLearningSynthesizedPlan.md`. Update the snapshot whenever execution
state changes. Append chronological entries; never rewrite historical entries.

## Current handoff snapshot

- Updated: 2026-07-17 19:55 EDT
- Branch: `sid-learnable-agent`
- Phase: acceleration steps 2–8 complete; canonical 512k A/B rung next
- Latest completed commit: `476af90 exp: add BC training lever screens`
- Working tree at start: the two independent source plans were untracked; no
  other changes were present
- Active experiment: recipe A completed; launching `phase5_screen_b_rung1` to
  the same predeclared 512k boundary
- Last test gate: 129 passed, 9 intentionally skipped, 4 benign checkpoint or
  small-rollout warnings in 15.72s
- Immutable fixture: `artifacts/phase4_bank`, fingerprint
  `3579113b230a04406ad8f17e5695c3a1fa0ea2145c0262f9da42b16b2162e130`
- Immutable split: `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`
- Sealed confirmation: 375 unique puzzles/layouts, fingerprint
  `816d8583c7cdb75deaf978f0e5b275d30f6a7d506e644df1c04d15994d42776b`;
  no learning model has been evaluated on it
- Current action: launch the matched canonical 512k A/B rung; retain 128×4 as
  a separate promising architecture candidate rather than silently changing
  the already-trained canonical initialization
- Canonical initialization: `artifacts/phase2_experiments/phase5_canonical_bc_seed11_retry1/seed_11/checkpoints/step_0/model.zip`,
  SHA-256 `d64eed2dcde7f3c84df0d56ece3d0d9392c92ba1939837439a402bee9f056181`

## Locked decisions

- Performance-first target: BC→PPO feed-forward policy.
- Later work should produce RL-only, BC-only, and BC→RL models, but the formal
  comparison is outside this session.
- Experiment budget: two short matched screens, one promoted full seed, and
  seeds 29/47 only after a material seed-11 improvement.
- Primary selection: stochastic validation success, with deterministic and
  multi-robot non-regression checks.
- Git discipline: one small green commit per work package.
- Reverse curriculum: implement, calibrate, test, and smoke now; defer the full
  16×16 RL-only run because BC already supervises every expert-trajectory state
  used as a reverse reset.

## Deferred-work register

| Item | Why deferred | Prerequisite / resumption trigger |
|---|---|---|
| Full 16×16 RL-only reverse run | Current session prioritizes the best performing BC→PPO model | Reverse smoke and real-factory throughput pass; baseline-calibrated frontier thresholds are sealed |
| Formal RL-only / BC-only / BC→RL comparison | Requires three separately validated recipes and materially more compute | Performance-first recipe frozen; RL-only pilot clears original-start validation gate |
| Robot-band weighting and alternate-optimal labels | Secondary to action waste and BC→PPO stability | Selected masked recipe still has a large two-/three-robot gap |
| Exact dihedral augmentation | Separate generalization factor | Baseline masked/stabilized result recorded; transition-equivalence tests designed |
| DAgger recovery states | Requires solver queries on off-plan states and new derived artifacts | BC errors remain dominated by off-trajectory recovery after masking |
| Relational action head | High-cost architecture change | Data interventions leave a material train/validation blocker gap |
| Horizon/termination ablation | Must not be confounded with masking | Masked recipe frozen and timeout/terminal-value telemetry available |
| DRC/recurrent comparison | Feed-forward baseline must be trustworthy first | Final feed-forward three-seed result complete |
| Grid-aligned RGB-sprite observation comparison | Performance-first work currently uses the lossless structured planes; the old 40×40 RGB-cell result was confounded by a 17×17 single-pixel board, black padding, and NatureCNN's aggressive strides | Freeze and reproduce the structured-plane recipe; add renderer/encoder contract tests; regenerate BC data; compare observations before adding recurrence/DRC |

## Chronological log

### 2026-07-17 13:29 EDT — execution started

- User approved the synthesized performance-first plan and asked for full
  implementation.
- Confirmed branch `sid-learnable-agent`; only the two source plan documents
  were untracked.
- Ran the full baseline suite: 88 passed, 9 legacy recurrent tests skipped, one
  benign small-rollout batch-size warning.
- Confirmed `sb3-contrib==2.9.0` is installed and exposes MaskablePPO prediction,
  load, distribution masking, and environment mask helpers.
- Read-only reverse-start calibration probe on 200 Phase 4 training puzzles,
  ten episodes each:
  - d=1: random 45.9%, target-only 68.1%
  - d=2: random 14.8%, target-only 44.0%
  - d=3: random 4.1%, target-only 15.8%
- Decision: the proposed fixed 70% d=1 frontier threshold is not a meaningful
  mastery gate. Reverse thresholds will be derived from persisted baselines.
- Decision: full reverse training is deferred because the existing BC dataset
  already includes every on-plan state used by reverse resets.

### 2026-07-17 13:55 EDT — observability and evaluation foundation complete

- Made requested/effective policy activation, orthogonal initialization, and
  BC action-head initialization explicit and persisted.
- Added step-0/intermediate checkpoint model, evaluation, curriculum, config,
  and runtime provenance artifacts plus validation-based checkpoint selection.
- Added validation-only runs (`test_status=sealed_not_run`), fixed validation
  RNG support, and puzzle-clustered bootstrap intervals.
- Separated BC epoch and PPO timestep axes in W&B. PPO optimizer metrics are
  forwarded with their one-update offset documented; fixed training probes log
  extractor, latent, logit, value, entropy, and confidence health.
- Created the sealed confirmation manifest from unused Phase 4 test membership:
  375 unique puzzle/layout pairs, fingerprint
  `816d8583c7cdb75deaf978f0e5b275d30f6a7d506e644df1c04d15994d42776b`.
  Verified zero puzzle-ID and layout-hash overlap with train, validation, and
  the exposed development-test suite. No model was evaluated.
- Test gate: 92 passed, 9 legacy recurrent tests skipped.

### 2026-07-17 14:12 EDT — movement masking and BC→PPO stability complete

- Added exact, non-mutating movement masks to the base environment and both
  bank/curriculum wrapper paths. MaskablePPO now uses them during collection
  and prediction; masked evaluation is the headline and optional unmasked
  evaluation is explicitly labelled counterfactual.
- BC validates every expert action against its mask and applies masks to the
  supervised distribution. It now records held-out loss/agreement per epoch
  and can restore the best held-out epoch.
- Added exact initial-checkpoint loading so the two screens can share one
  canonical seed-11 BC policy. Added fixed optimizer stability controls,
  `target_kl`, and cumulative-timestep LR/entropy schedules that do not restart
  at checkpoint boundaries.
- Added the policy-health audit script and ran it on the historical endpoints:
  - Phase 4: expert top-1 6.9%, target probability 6.1%, deterministic 0.0%,
    stochastic 4.7%, logit between-state variance exactly 0.
  - Phase 4b: expert top-1 52.9%, target probability 52.6%, deterministic
    24.0%, stochastic 30.9%, logit variance 57.03.
- Optimized mask computation to a direct adjacent-cell feasibility check.
  Reset-inclusive real-factory probe improved from an initial 15.9% overhead
  to 5.7% (11,605 unmasked versus 10,937 masked environment steps/s in the
  isolated env probe), satisfying the <10% gate.
- Test gate: 99 passed, 9 legacy recurrent tests skipped.

### 2026-07-17 14:25 EDT — reverse infrastructure implemented; smoke queued

- Added exact stored-plan prefix replay without bank mutation; every tested
  suffix reaches the goal in exactly its declared remaining distance.
- Added deterministic weighted band identity, terminal `curriculum_band`
  plumbing, per-band windows, frontier-only mastery, and explicit
  mastery/forced transition records. Legacy unkeyed band sampling retains its
  previous path.
- Added a calibration command and persisted artifacts:
  - 8×8 fingerprint `991c7e1b4404…`; thresholds d1=73.53%, d2/d3=70%.
  - 16×16 fingerprint `d6b32bd0ee70…`; threshold d1=79.5%, d2–d8=70%.
- Added smoke, dry-run, and deferred full-run configs. The deferred full config
  is documented but must not be launched in this session.
- Re-derived the immutable Phase 4 contracts after the changes: fixture
  `3579113b230a…`, split `c0e97329b109…`, unchanged.
- Test gate: 103 passed, 9 legacy recurrent tests skipped.

### 2026-07-17 14:27 EDT — smoke attempt 1 stopped before training

- Command used output `artifacts/phase2_experiments/phase5_reverse_smoke_8x8`.
- W&B run `o5tjfizz` failed during initialization before step 0 because W&B
  0.28.0 rejects `eval_*/*`; it supports suffix globs only.
- Replaced the generic glob with concrete evaluation-prefix suffix globs and
  added a fake-W&B regression test enforcing the installed API contract.
- No training steps ran, no checkpoint was produced, and no bank data changed.
- Full gate after the fix: 104 passed, 9 legacy recurrent tests skipped.

### 2026-07-17 14:44 EDT — 8×8 reverse smoke passed

- Retry output: `artifacts/phase2_experiments/phase5_reverse_smoke_8x8_retry1`;
  W&B run `vrzh40da`; split fingerprint `d22582f1d4cc…`.
- The curriculum made all three required transitions by mastery and made zero
  forced transitions: d1 at 3,100 episodes (frontier 74.5%), d2 at 6,700
  cumulative episodes (71.0%), and d3 at 14,700 (70.0%). It reached the final
  original-start stage with 286k+ training timesteps remaining.
- Validation improved monotonically from 10.9% deterministic / 21.2%
  stochastic at step 0, through 43.6% / 44.8% at 100,352 steps, to **65.5% /
  67.9%** at 301,056. Final stochastic success by optimal length was
  100%/68.9%/62.5%/56.3% for lengths 1/2/3/4. Masked invalid/no-movement rates
  were zero. Throughput was 2,130 training steps/s.
- Acceptance decision: pass. The reset construction, calibrated frontier
  gating, transition bookkeeping, checkpointing, and W&B telemetry all worked
  together in a real training run.
- Found a presentation-only defect after success: validation-only runs printed
  an aggregate “held-out 0%” because their sealed test aggregate is empty. The
  report itself was correct. The CLI now explicitly prints selected validation
  performance when the final test is sealed, with two regression tests; 11
  targeted tests pass. Full gate: 106 passed, 9 intentionally skipped.
- Next bounded check: run the 16×16 no-BC dry run for reset correctness,
  throughput, and early curriculum behavior. It is not a model-quality result
  and does not authorize the deferred full RL-only run.

### 2026-07-17 14:58 EDT — 16×16 reverse dry run completed

- Output: `artifacts/phase2_experiments/phase5_reverse_dryrun_16x16`; immutable
  Phase 4 split fingerprint exactly `c0e97329b109…`. The 55,143-puzzle bank and
  4.4% random / 8.5% target-only baselines reproduced unchanged.
- The 200,704-step run completed in 303.4 training seconds at 662 steps/s,
  close to historical 16×16 throughput. Checkpoint model/evaluation artifacts
  exist at step 0 and step 200,704; the final test remained sealed.
- The calibrated d1 frontier ended at 72.0% versus its 79.5% mastery threshold,
  so the manager correctly remained at level 0 with **no mastery or forced
  transition**. This bounded run was not expected to train a useful RL-only
  model, and the threshold was not relaxed after seeing the result.
- Masked validation was 1.7% deterministic / 6.1% stochastic at step 0 and
  5.8% / 5.8% at step 200,704. Selection correctly retained step 0 by the
  primary stochastic metric. At the endpoint, the explicitly labelled
  unmasked counterfactual fell to 0.8%/0.8% with 95.1% deterministic and 80.7%
  stochastic no-movement actions, while the masked policy had zero
  no-movement actions. This is direct real-board evidence that movement masking
  is necessary, not merely a synthetic correctness result.
- Acceptance decision: reverse infrastructure is ready for the deferred
  RL-only program. Do not launch `phase5_reverse_16x16_deferred.yaml` in this
  session. Proceed to the performance-first canonical BC initialization and
  matched masked BC→PPO screens.

### 2026-07-17 15:05 EDT — canonical BC and matched screens defined

- Added `phase5_canonical_bc_seed11.yaml`: 25 masked BC epochs over every Phase
  4 training trajectory, held-out expert-state validation each epoch,
  best-validation-loss epoch restoration, and a saved/evaluated step-0 policy
  with no PPO updates or final-test access.
- Added two 2.048M-step masked BC→PPO screens that load that exact step-0 model:
  - A uses historical Phase 4b optimization (`lr=3e-4`, `ent_coef=0.01`, no KL
    guard).
  - B uses the stability bundle (`lr=1e-4`, `ent_coef=0.001`,
    `target_kl=0.03`).
- Both screens otherwise have identical explicit model/init/masking, Phase 4
  fixture/split, length curriculum, seed 11, checkpoint schedule
  0/102.4k/256k/512k/1.024M/2.048M, five fixed-seed validation repetitions,
  puzzle-clustered bootstrap, masked/unmasked counterfactual evaluation,
  validation selection, and sealed final test.
- Added a config contract test that fails on any undeclared screen difference
  and verifies that canonical BC is validation-only and saves step 0. Full gate:
  108 passed, 9 intentionally skipped.

### 2026-07-17 15:07 EDT — canonical seed-11 BC run launched

- Launch commit `b82f899`; config `configs/phase5_canonical_bc_seed11.yaml`;
  output `artifacts/phase2_experiments/phase5_canonical_bc_seed11`; log
  `artifacts/phase2_experiments/phase5_canonical_bc_seed11.log`.
- Two normal `nohup ... &` launches (reported shell PIDs 28233 and 28322)
  exited before Python startup because this command-execution wrapper cleans up
  background children when its shell ends. Both left an empty log, created no
  output directory, and performed no experiment work.
- Relaunched under detached macOS `screen` session `phase5_bc`, which reparented
  to PID 1 and was verified alive: screen PID 28369, launch-time Python PID
  28372, `caffeinate` helper PID 28374. The log immediately recorded fixture
  validation startup.
- This run performs 25 BC epochs, restores the minimum held-out BC-loss epoch,
  evaluates/saves step 0, and keeps the final test sealed. Do not start either
  PPO screen until the checkpoint and BC result metadata have been audited.

### 2026-07-17 15:33 EDT — canonical attempt stopped; supervised mask stall fixed

- Attempt output/log/W&B were preserved at
  `phase5_canonical_bc_seed11` / W&B `bfh8kzng`. The run was interrupted during
  epoch 4 after three completed epochs stayed at masked chance:
  - epoch 1: train 2.534/9.0%, validation 2.519/8.4%;
  - epoch 2: train 2.520/8.4%, validation 2.519/8.8%;
  - epoch 3: train 2.520/8.7%, validation 2.519/8.5%.
  It produced no model checkpoint and never reached rollout or final-test
  evaluation.
- A same-batch real-residual audit showed nonzero action-head/extractor
  gradients in both modes, ruling out a disconnected MaskablePPO graph.
  Standard PPO and MaskablePPO also behaved identically on matched subsets,
  ruling out the algorithm-class swap.
- The decisive full 210,947-pair, same-seed, same-model, same-optimizer audit
  changed only the supervised loss mask. One **unmasked** epoch reproduced the
  historical Phase 4b number exactly: loss 2.522, accuracy 15.0%, in 296.7s.
  The first masked epoch was 2.534/9.0% and then plateaued. Conclusion: masking
  the BC softmax removes useful invalid-action contrast and traps this large
  model near its masked action prior; rollout masking itself is not implicated.
- Added independent `bc_apply_action_masks` provenance/control, default false.
  When rollout masking is enabled, BC dataset construction still calculates
  masks and rejects any invalid stored expert action. The canonical loss uses
  all 16 logits, while MaskablePPO collection and every headline rollout remain
  masked. The optional masked BC-loss path remains tested.
- Added unique retry config/output/W&B identity and repointed both screens to
  its exact future step-0 checkpoint. Full gate: 110 passed, 9 skipped.

### 2026-07-17 15:35 EDT — canonical BC retry 1 launched

- Launch commit `baecf88`; config
  `configs/phase5_canonical_bc_seed11_retry1.yaml`; output/log stem
  `artifacts/phase2_experiments/phase5_canonical_bc_seed11_retry1`.
- Detached screen session `phase5_bc_retry1`: launch-time screen PID 31858,
  Python PID 31861, `caffeinate` helper PID 31863.
- Immediate checks passed: immutable fixture/split validation, historical
  4.4%/8.5% baselines, MPS selection, and seed-11 startup. Epoch-1 acceptance
  reference is the exact audit/historical 15.0% training accuracy; a return to
  ~9% would stop the run for further diagnosis.

### 2026-07-17 16:35 EDT — Sokoban-style RGB observations added to future work

- Checked the primary source for the Sokoban work this repository builds on:
  Guez et al., _An Investigation of Model-Free Planning_ ([paper](https://proceedings.mlr.press/v97/guez19a.html),
  [official supplement](https://proceedings.mlr.press/v97/guez19a/guez19a-supp.pdf)).
  Its visual input was not an uncontrolled natural screenshot. Each cell of a
  fixed 10×10 Sokoban board was rendered as an 8×8 RGB sprite, giving an
  80×80×3 observation. A two-layer encoder (32/32 channels, 8/4 kernels, 4/2
  strides) deliberately recovered a 10×10×32 spatial tensor for the planning
  core.
- Deferred a controlled Ricochet observation ablation: current native-grid
  structured planes versus a deterministic, cell-aligned RGB-sprite render.
  Action masks remain a separate policy input/constraint. The comparison must
  hold fixture, split, expert actions, rewards, masking, BC/PPO protocol,
  evaluation seeds, budget, and selection rules fixed, and should match model
  capacity closely enough for the observation encoding to be the main change.
- Do not combine this first RGB comparison with DRC/recurrent changes. Start
  with a spatial feed-forward encoder, then test DRC separately if RGB is
  viable; otherwise observation encoding and internal planning depth would be
  confounded.
- The earlier `rgb_cell_image` failure does not settle this question. That run
  centered a 17×17 single-pixel board in a 40×40 black canvas and fed it to
  NatureCNN, whose stride-4 first layer aliased fine board geometry. A future
  RGB encoder must have an explicit renderer-to-feature-map contract (ideally
  one stable feature location per board cell), deterministic pixel/cache tests,
  lossiness/injectivity checks, and throughput/memory measurements.
- Changing observation tensors requires regenerating BC artifacts and makes
  existing checkpoints incompatible. Resume only after the current
  structured-plane recipe is frozen and reproduced; the active canonical BC
  retry was not changed or interrupted.

### 2026-07-17 17:10 EDT — training-acceleration implementation authorized

- User authorized implementation, while the canonical BC retry continues, of:
  1. low-risk BC synchronization and tensor-caching fixes;
  2. deterministic-evaluation deduplication; and
  3. batched evaluation.
- The active `phase5_canonical_bc_seed11_retry1` process and its artifacts must
  not be modified or interrupted. The completed run remains the unchanged
  reference for parity and timing comparisons.
- After those changes are implemented and verified, continue in this explicit
  order:
  1. benchmark PPO rollout geometry at `16×128`, `32×64`, and `64×32` for a
     bounded 100–200k steps per candidate;
  2. implement true resumable promotion, restoring optimizer, timestep,
     schedule, RNG, and curriculum state;
  3. add predeclared successive-halving and plateau-stopping rules; and
  4. only then test larger BC batches, mixed precision, and smaller models.
- Acceptance gates for the current work package: BC model-update and metric
  parity within declared numerical tolerances; one deterministic episode per
  fixed puzzle with unchanged aggregate meaning; batched evaluator record and
  aggregation parity, deterministic reproducibility, batch-width-invariant
  stochastic RNG behavior, action-mask parity, and materially improved
  measured evaluation throughput.

### 2026-07-17 17:16 EDT — BC and batched-evaluation implementation complete

- Commit `d4ec33e` keeps BC validation tensors resident on the selected device,
  retains expert-action mask validation without transferring/gathering unused
  loss masks, accumulates train/validation metrics on-device, and reduces
  hundreds of minibatch scalar synchronizations to one metric transfer per
  phase and epoch.
- Fixed-puzzle deterministic model evaluation now runs exactly once per puzzle
  even when stochastic evaluation requests multiple repetitions.
- Model evaluation now advances configurable batches of independent episodes
  per policy call (`evaluation_batch_size`, default 64). Stochastic sampling
  uses a stable RNG stream derived separately for each puzzle/repetition, so
  records do not depend on evaluator batch width and evaluation does not
  perturb PyTorch's training RNG.
- Added exact CPU model-update parity, serial-versus-batched deterministic
  record parity, batch-width-invariant stochastic record/aggregate parity,
  RNG isolation, and action-mask tests. Targeted gates passed: 7 existing BC
  tests, then 15 BC/batched/masking tests, plus the existing Phase 2/Phase 5
  contract tests.
- The active canonical process had already imported the launch-time code. It
  completed all 25 original BC epochs unchanged and is now running the old
  serial step-0 evaluator; its timing/result remains the intended reference.

### 2026-07-17 17:21 EDT — canonical BC complete; acceleration gate passed

- The unchanged canonical retry completed normally. BC built 210,947 training
  and 35,091 validation pairs, took 6,026.2 seconds total (26.4 seconds for
  dataset construction), and restored epoch 12: validation loss 1.3477 and
  action accuracy 61.05%. Epoch 25 had overfit to 86.0% train accuracy but
  validation loss 2.9709, confirming best-epoch restoration was necessary.
- The saved step-0 checkpoint is
  `phase5_canonical_bc_seed11_retry1/seed_11/checkpoints/step_0/model.zip`,
  SHA-256 `d64eed2dcde7f3c84df0d56ece3d0d9392c92ba1939837439a402bee9f056181`.
  Masked validation was 30.6% deterministic / 35.5% stochastic with zero
  no-movement actions. The old serial unmasked counterfactual was 26.4% / 30.4%
  with 41.1% / 22.9% no-movement rates. The final test remained sealed.
- Full post-change gate: 114 passed, 9 intentionally skipped, two benign
  tiny-rollout warnings. Commit `637f5d2` makes batch size 64 explicit in the
  canonical and both matched screen configs.
- Production MPS benchmark on the exact 121-puzzle validation suite and the
  canonical checkpoint:
  - deterministic batch 1: 1,878 steps in 4.2520s; batch 64: the exact same
    records in 0.7425s (**5.73× faster**);
  - stochastic batch 1: 9,610 steps in 21.9648s; batch 64: 9,610 steps in
    2.1611s (**10.16× faster**), with identical record SHA-256
    `dfbd5514bf6689526f61fa5651e2bb550609009c41d03b8f387c35666480f97b`.
- Deterministic deduplication additionally changes five identical fixed-puzzle
  repetitions to one, making that part of the Phase 5 evaluation approximately
  29× cheaper than the previous serial five-repeat protocol. The current work
  package's parity, reproducibility, masking, full-test, and measured-throughput
  gates are satisfied.

### 2026-07-17 17:34 EDT — PPO rollout geometry benchmark complete

- Commit `02cd3c6` defined three configs that differ only in `n_envs` and
  `n_steps`, all with the same canonical checkpoint, stable optimizer bundle,
  2,048-transition rollout buffer, 102,400-step budget, evaluation protocol,
  fixture/split, seed, and sealed final test. The config contract test passed.
- All candidates reproduced the identical batched step-0 validation result
  (30.6% deterministic / 33.9% stochastic) and advanced through the first two
  curriculum boundaries by mastery, so the timing comparison includes matched
  reset/curriculum work.
- Results:

  | Geometry | Training seconds | Steps/s | 102.4k val det/stoch |
  |---|---:|---:|---:|
  | `16×128` | 164.47 | **622.60** | **34.7% / 39.5%** |
  | `32×64` | 169.70 | 603.41 | 34.7% / 36.0% |
  | `64×32` | 186.14 | 550.13 | 31.4% / 36.4% |

- Decision: retain `16×128`. `32×64` was 3.1% slower and `64×32` was 11.6%
  slower; widening batches does not amortize enough MPS synchronization to
  offset serial `DummyVecEnv` work on this machine. The short quality values
  are supporting non-regression evidence, not standalone recipe selection.
- The three bounded results are stored under
  `artifacts/phase2_experiments/phase5_geometry_{16x128,32x64,64x32}`. No final
  test was run.

### 2026-07-17 18:02 EDT — exact resumable promotion implemented

- Commit `0ff548d` adds a versioned `training_state.pkl` beside every saved PPO
  checkpoint. It binds the sidecar to the model SHA-256 and records the
  optimizer/model state already present in the SB3 archive plus the timestep,
  update counters, training-critical config contract, schedule horizon,
  curriculum level and rolling windows, per-worker puzzle-sampler streams,
  vector-environment pending seeds/options, and Python/NumPy/Torch RNG state.
- `resume_checkpoint_path` and `resume_state_path` are now distinct from a
  weights-only `initial_checkpoint_path`. Resume rejects mismatched seeds,
  worker geometry, devices, schedules, model hashes, timestep boundaries, or
  training-critical configuration instead of silently starting a different
  trajectory.
- Checkpoints are canonical clean reset boundaries: all workers are reset and
  the policy's last observations/episode-start flags are updated before the
  archive is saved. Uninterrupted training follows that same boundary, while a
  restored job reconstructs and hashes the identical reset observations. This
  deliberately discards any uncounted partial episodes at checkpoint time but
  makes promotion reproducible without serializing arbitrary nested Gym state.
- The exact-resume regression interrupts a tiny PPO run, reconstructs fresh
  model/env objects, and verifies that its next update matches uninterrupted
  training in policy tensors, Adam state, last observations, timesteps,
  curriculum, sampler streams, and global RNG. The full gate is 118 passed and
  9 intentionally skipped.
- Existing geometry artifacts predate this feature and are timing evidence,
  not resumable promotion sources. Future Phase 5 checkpoint directories will
  contain both `model.zip` and `training_state.pkl`; the canonical BC model
  remains a valid weights-only initializer.

### 2026-07-17 18:28 EDT — prospective allocation and plateau rules implemented

- Commit `ba36356` caps both matched A/B screen configs at the first 512k-step
  successive-halving rung. They can only be continued through
  `src/scripts/adjudicate_phase5_rung.py`, which consumes completed results and
  the versioned rules in `configs/phase5_allocation_rules.yaml` and emits a
  machine-readable decision plus validated exact-resume configs.
- A candidate must have zero masked no-movement and invalid actions and may not
  regress deterministic or either two-/three-robot stochastic validation by
  more than two points from its rung run's first checkpoint. Stochastic
  validation is primary, deterministic validation breaks ties, and a clear
  halving decision additionally requires at least a one-point margin,
  disjoint puzzle-clustered 95% intervals, and a lead at both recent common
  checkpoints. Ambiguity retains all healthy candidates; it never forces an
  under-supported elimination merely to meet the nominal halving ratio.
- Rung 1 promotes from 512k to checkpoints at 1.024M and 2.048M. Rung 2 promotes
  from 2.048M toward 10.24M. Generated configs require both `model.zip` and
  `training_state.pkl`, clearing the weights-only initializer so optimizer,
  schedules, RNG, curriculum, and sampler streams are preserved.
- Full promotion prospectively enables stochastic-validation plateau stopping:
  minimum budget 4.096M, patience three checkpoint evaluations, and minimum
  material improvement one percentage point. The rule uses fixed validation
  seeds, runs only after the checkpoint is safely persisted, is recorded in
  each result, and retains ordinary best-checkpoint selection.
- Unit coverage exercises clear promotion, overlapping uncertainty,
  inconsistent curves, health rejection, exact-resume config generation, and
  plateau recovery/reset semantics. Full gate: 126 passed, 9 intentionally
  skipped. No A/B learning results have yet been inspected and the final test
  remains sealed.

### 2026-07-17 18:30 EDT — bounded training-lever screen in progress

- Commit `476af90` adds guarded MPS/CUDA FP16 autocast plus gradient scaling to
  BC, a real-policy forward/backward throughput benchmark, and a restartable
  five-epoch held-out screen. Default BC remains FP32 and the canonical config
  is unchanged. Full gate before launch: 129 passed, 9 intentionally skipped.
- MPS kernel results for the real 6.15M-parameter policy:
  - batch 1,024 / 2,048 / 4,096 FP32: 1,856 / 1,867 / 1,868 examples/s, so
    larger batches provide effectively no throughput benefit while using up to
    9.19GB driver-allocated memory;
  - batch-1,024 FP16 autocast: 4,215 examples/s (**2.27×**), with finite
    losses, gradients, and parameters in the bounded update test;
  - 128×4 and 96×4 FP32: 2,721 and 3,041 examples/s (**1.47×** and
    **1.64×**), with 5.56M and 5.04M parameters respectively.
- The microbenchmark is throughput/numerical-health evidence only. The active
  quality screen uses the exact Phase 4 fixture/split, seed, data, learning
  rate, and five epochs, changing one declared factor at a time. It reuses the
  original canonical run's first five epochs as the 1,024/128×6 FP32 reference.
- Batch 2,048 completed in 993.7s and selected epoch 3: validation loss 2.189,
  accuracy 25.13%, deterministic rollout 3.31%, stochastic rollout 8.76%.
  Canonical equal-epoch reference reached loss 1.535 and accuracy 56.97% at
  epoch 5, while the final canonical selected model reached 30.6% / 35.5%
  deterministic/stochastic validation. Batch 2,048 is rejected as an
  unretuned drop-in change; this does not rule out a separately retuned
  learning-rate/update-budget recipe.
- Batch 4,096 is currently training under PID 51569. Remaining variants, in
  fixed order, are FP16 128×6, FP32 128×4, and FP32 96×4. Results persist per
  variant under `artifacts/phase2_experiments/phase5_bc_training_lever_screen`,
  so restarting the runner skips completed variants. The sealed final test
  remains untouched.

### 2026-07-17 19:12 EDT — bounded training-lever screen complete

- All five variants completed normally on the exact Phase 4 fixture/split and
  the process exited. Results are persisted under
  `artifacts/phase2_experiments/phase5_bc_training_lever_screen`.

  | Variant | BC seconds | Best val loss / accuracy | Val det / stoch |
  |---|---:|---:|---:|
  | batch 2,048 | 993.7 | 2.189 / 25.13% | 3.31% / 8.76% |
  | batch 4,096 | 998.5 | 1.854 / 42.97% | 9.09% / 12.89% |
  | FP16 128×6 | 953.6 | 1.463 / 56.77% | 22.31% / **26.61%** |
  | FP32 128×4 | 687.6 | **1.460 / 58.68%** | **23.14%** / 24.63% |
  | FP32 96×4 | **491.6** | 1.504 / 57.92% | 18.18% / 24.79% |

- The equal-five-epoch canonical reference was validation loss 1.535 and
  accuracy 56.97%. Its launch-time run did not save an epoch-5 rollout model,
  so only supervised metrics are directly equal-epoch comparable; the final
  restored epoch-12 canonical model achieved 30.6% / 35.5% validation rollout
  success and must not be compared as though it had only five epochs.
- Decision on larger batches: reject 2,048 and 4,096 as unretuned drop-in
  changes. They did not improve kernel throughput and substantially reduced
  supervised and rollout quality by cutting optimizer updates per epoch.
- Decision on FP16: numerical and learning quality passed the bounded screen,
  but end-to-end BC time improved only 4.5% relative to the similarly sized
  FP32 lever runs despite a 2.27× isolated-kernel result. Keep it experimental
  until profiler-guided scaler/indexing work and a full validation run; do not
  alter PPO precision or the canonical recipe yet.
- Decision on smaller models: 128×4 is the best bounded speed/quality tradeoff
  (30.8% less BC time than the full network, better equal-epoch validation
  loss/accuracy, and the best deterministic rollout of the smaller models).
  96×4 is fastest but loses deterministic rollout quality. Neither replaces
  the canonical checkpoint without a longer BC and PPO quality comparison.
- The performance-first critical path now returns to the already-trained
  canonical 128×6 initialization and the predeclared 512k A/B optimizer rung.
  The 128×4 full comparison is retained as follow-up work if the canonical PPO
  recipe plateaus or after the first trustworthy promoted model exists. The
  sealed final test remains untouched.

### 2026-07-17 19:14 EDT — canonical A/B rung launched

- Start recipe A first, using `configs/phase5_screen_a_historical.yaml` and
  output `artifacts/phase2_experiments/phase5_screen_a_rung1`. It begins from
  the immutable canonical BC checkpoint and stops at 512k by configuration.
- Recipe B will run sequentially after A so the single saturated MPS device is
  not contended. Adjudication will consume both completed results and the
  already-committed prospective rules; the sealed final test remains disabled.

### 2026-07-17 19:55 EDT — recipe A rung complete; recipe B starting

- Recipe A completed 512k in 756.3 training seconds at 676.9 steps/s. Validation
  deterministic/stochastic success progressed `30.58%/33.88%` at step 0,
  `28.10%/35.21%` at 102.4k, `33.88%/37.02%` at 256k, and
  `33.88%/39.01%` at 512k. The selected rung checkpoint is 512k.
- Health gates improved: two-robot stochastic success rose 26.19%→31.90% and
  three-robot success 12.26%→21.29%; masked no-movement and invalid-action
  rates remained exactly zero. The curriculum advanced by mastery through
  levels 0 and 1 and is currently at level 2 of 4.
- Every A checkpoint contains its bound `model.zip` and `training_state.pkl`,
  so it is exactly promotable. Recipe B now starts from the byte-identical
  canonical BC checkpoint under the matched first-rung protocol. Final test
  remains sealed.
