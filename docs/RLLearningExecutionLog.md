# RL Learning Remediation Execution Log

This is the live, detailed handoff document for
`docs/RLLearningSynthesizedPlan.md`. Update the snapshot whenever execution
state changes. Append chronological entries; never rewrite historical entries.

## Current handoff snapshot

- Updated: 2026-07-17 15:35 EDT
- Branch: `sid-learnable-agent`
- Phase: canonical BC retry 1 active
- Latest completed commit: `baecf88 fix: preserve full contrast in BC loss`
- Working tree at start: the two independent source plans were untracked; no
  other changes were present
- Active experiment: `phase5_canonical_bc_seed11_retry1`, detached screen
  session `phase5_bc_retry1` (screen PID 31858; launch-time Python PID 31861)
- Last test gate: 110 passed, 9 intentionally skipped, 1 benign small-rollout
  batch-size warning in 13.14s
- Immutable fixture: `artifacts/phase4_bank`, fingerprint
  `3579113b230a04406ad8f17e5695c3a1fa0ea2145c0262f9da42b16b2162e130`
- Immutable split: `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`
- Sealed confirmation: 375 unique puzzles/layouts, fingerprint
  `816d8583c7cdb75deaf978f0e5b275d30f6a7d506e644df1c04d15994d42776b`;
  no learning model has been evaluated on it
- Current action: verify retry epoch 1 reproduces the 15% unmasked reference,
  then monitor held-out BC loss until best-epoch selection and step-0 rollout
- Exact monitor command: `tail -n 50 artifacts/phase2_experiments/phase5_canonical_bc_seed11_retry1.log`

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
