# RL Learning Remediation Report

**Status:** seed-11 BC→PPO promotion successful; independent seeds pending

**Run dates:** 2026-07-17 to 2026-07-18

**Branch:** `sid-learnable-agent`
**Detailed audit trail:** `docs/RLLearningExecutionLog.md`

## Executive summary

This work produced a learning 16×16 feed-forward Ricochet Robots policy and a
reproducible path for promoting it without tuning on the sealed confirmation
set. The selected seed-11 model uses behaviour cloning followed by masked PPO.
It achieved **39.67% deterministic** and **41.82% stochastic** validation
success, compared with 30.58% / 33.88% for the matched BC-only starting
checkpoint. It executes no invalid or no-movement actions under deployment
masking.

The promoted run stopped prospectively at 6.144M PPO transitions after three
eligible validation checkpoints failed to improve materially. The best model
remained the 2.048M checkpoint. This saved 4.096M transitions, approximately
1h43m at measured throughput, without retrospective selection of a stopping
rule.

The selected seed passes every predeclared expansion gate. It is not yet a
three-seed result: independent seeds 29 and 47 and the single sealed
confirmation evaluation remain outstanding.

## Selected model

| Property | Selected value |
|---|---|
| Training recipe | Behaviour cloning → movement-masked PPO |
| Observation | Lossless structured planes, `(10, 16, 16)`, float32 |
| Action space | 16 actions: four robots × four directions |
| Architecture | `GridPlanesResNet`, 128 channels, six residual blocks |
| Parameters | 6,148,145 |
| PPO geometry | 16 environments × 128 steps = 2,048 transitions/rollout |
| PPO optimization | Batch 1,024, two epochs, four updates/rollout |
| Learning rate / entropy | `3e-4` / `0.01` |
| Episode horizon | 20 actions |
| Selected training age | 2.048M PPO transitions |
| Validation deterministic | **39.67%** (48/121 puzzles) |
| Validation stochastic | **41.82%** (253/605 episodes) |
| Puzzle-clustered stochastic 95% interval | 34.05%–49.75% |
| Masked no-movement / invalid action rate | **0% / 0%** |

Selected checkpoint:

`artifacts/phase2_experiments/phase5_screen_a_promoted_seed11/seed_11/checkpoints/step_2048000/model.zip`

Its exact continuation sidecar is in the same directory as
`training_state.pkl`. The final 6.144M stopping boundary is also exactly
resumable, but it was not selected for deployment.

## Data and evaluation contract

- Immutable Phase 4 fixture fingerprint:
  `3579113b230a04406ad8f17e5695c3a1fa0ea2145c0262f9da42b16b2162e130`.
- Immutable split fingerprint:
  `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`.
- Training, validation, development-test, and confirmation layouts are disjoint
  by stable layout hash.
- Recipe and checkpoint selection use validation only. Stochastic validation
  uses five fixed-seed episodes for each of 121 puzzles and uncertainty is
  clustered by puzzle.
- The sealed confirmation manifest contains 375 unused unique puzzles/layouts,
  fingerprint
  `816d8583c7cdb75deaf978f0e5b275d30f6a7d506e644df1c04d15994d42776b`.
- No model in this remediation program has been evaluated on that confirmation
  set. The selected run records `test_status: sealed_not_run`.

## Behaviour-cloning initialization

The canonical seed-11 BC run replayed optimal plans from the immutable bank:

| Metric | Result |
|---|---:|
| Training state/action pairs | 210,947 |
| Validation pairs | 35,091 |
| Configured epochs | 25 |
| Selected epoch | 12 |
| Best validation loss | 1.3477 |
| Best validation accuracy | 61.05% |
| Total BC time | 6,026.2s |
| Dataset construction | 26.4s |

The BC loss is deliberately unmasked because applying movement masks inside
the supervised objective stalled near masked chance. Expert actions are still
validated against movement masks, and PPO collection/evaluation remains
masked. The canonical checkpoint SHA-256 is
`d64eed2dcde7f3c84df0d56ece3d0d9392c92ba1939837439a402bee9f056181`.

## Successive-halving result

Two recipes began from the byte-identical canonical checkpoint:

- A: historical optimizer, learning rate `3e-4`, entropy `0.01`, no KL guard.
- B: stability bundle, learning rate `1e-4`, entropy `0.001`, `target_kl=0.03`.

### First rung: 512k transitions

| Recipe | Deterministic | Stochastic | Two-robot | Three-robot | Health |
|---|---:|---:|---:|---:|---|
| A | 33.88% | 39.01% | 31.90% | 21.29% | passed |
| B | 31.40% | 40.50% | 36.67% | 17.42% | passed |

B's stochastic point estimate led, but the puzzle-clustered confidence
intervals overlapped. The prospective ambiguity rule therefore retained both.

### Second rung: 2.048M transitions

| Recipe | Deterministic | Stochastic | Two-robot | Three-robot | Decision |
|---|---:|---:|---:|---:|---|
| A | **39.67%** | **41.82%** | **42.86%** | **19.35%** | promote |
| B | 35.54% | 38.35% | 32.86% | 9.03% | eliminate |

B was eliminated by rules fixed before the run: relative to its own 512k
boundary, its two-robot rate regressed 3.81 points and its three-robot rate
regressed 8.39 points, exceeding the two-point tolerances. A remained healthy,
reached the final curriculum stage, and was promoted alone.

## Promoted seed-11 curve and stopping decision

| PPO transitions | Deterministic | Stochastic | Two-robot | Three-robot |
|---:|---:|---:|---:|---:|
| 2.048M | **39.67%** | **41.82%** | **42.86%** | 19.35% |
| 3.072M | 38.02% | 40.17% | 40.00% | 20.65% |
| 4.096M | 34.71% | 35.70% | 32.86% | 14.84% |
| 5.120M | 33.88% | 38.84% | 41.43% | 18.06% |
| 6.144M | 37.19% | 38.68% | 41.43% | 23.87% |

Plateau stopping was declared before promotion:

- stochastic validation success is the monitored metric;
- no stop before 4.096M transitions;
- a material improvement is at least one percentage point;
- stop after three non-improving eligible checkpoints;
- continue saving and selecting the best validation checkpoint normally.

The 4.096M, 5.120M, and 6.144M checkpoints did not improve by one point over
the 2.048M best, so the run stopped at 6.144M. Across its three exact segments,
A trained for approximately 9,286 seconds (2h35m) at roughly 650–677 steps/s.

## Seed-expansion gates

| Predeclared gate | Requirement | Selected result | Status |
|---|---:|---:|---|
| Stochastic validation | at least 36% | **41.82%** | pass |
| Deterministic validation | at least 25% | **39.67%** | pass |
| Two-robot regression from BC-only | no worse than -2 points | +16.67 points | pass |
| Three-robot regression from BC-only | no worse than -2 points | +7.10 points | pass |
| Masked no-movement actions | exactly zero | **zero** | pass |
| Invalid actions | exactly zero | **zero** | pass |

These gates authorize freezing recipe A and training independent seeds 29 and
47. They do not authorize tuning on confirmation results.

## Engineering changes

### Trustworthy training and selection

- Added explicit policy construction/provenance, validation-only runs,
  puzzle-clustered intervals, health telemetry, best-checkpoint selection, and
  sealed-test controls.
- Added movement masks to MaskablePPO collection and evaluation, with expert
  action validation and zero-waste deployment behavior.
- Added held-out BC selection, explicit action-head initialization, cumulative
  schedules, and a KL-guard option.
- Implemented calibrated reverse-start curriculum infrastructure and smoke
  tests. A full RL-only reverse run remains deferred because the current target
  is the strongest BC→PPO model.

### Exact resumable promotion

Every promotable checkpoint binds `model.zip` to `training_state.pkl` by model
SHA-256 and saves/restores:

- policy and optimizer state, timesteps, and update counters;
- schedule horizon and training-critical configuration contract;
- curriculum level, windows, and transitions;
- each worker's puzzle-sampler stream and vector-environment seeds/options;
- Python, NumPy, Torch CPU, CUDA, and MPS RNG state;
- a hashed clean-reset observation boundary.

The regression test reconstructs fresh model/environment objects and matches an
uninterrupted next PPO update exactly in policy tensors, Adam state, last
observations, timesteps, curriculum, sampler streams, and global RNG.

### Evaluation and BC acceleration

- BC validation tensors stay on-device, unused loss masks are not gathered,
  metrics transfer once per phase/epoch, and gradients use
  `zero_grad(set_to_none=True)`.
- Fixed-puzzle deterministic evaluation runs once rather than repeating five
  identical episodes.
- Evaluation batches independent episodes while preserving exact records,
  action masks, per-puzzle stochastic RNG, and batch-width invariance.
- On the production validation suite, batching improved deterministic
  evaluation **5.73×** and stochastic evaluation **10.16×**. Including
  deterministic deduplication makes that protocol component about 29× cheaper.

The final full repository gate after implementation was **129 passed, 9 legacy
recurrent tests intentionally skipped**. The remaining warnings are benign
small-rollout/checkpoint-path warnings in tests.

## Throughput and architecture investigations

### PPO rollout geometry

All candidates retained a 2,048-transition buffer:

| Geometry | Steps/s | 102.4k validation det/stoch | Decision |
|---|---:|---:|---|
| 16×128 | **622.6** | **34.7% / 39.5%** | retain |
| 32×64 | 603.4 | 34.7% / 36.0% | reject |
| 64×32 | 550.1 | 31.4% / 36.4% | reject |

Wider inference batches did not offset serial `DummyVecEnv` work on this MPS
machine.

### BC batch, precision, and smaller-model screen

| Variant | BC seconds | Best five-epoch val loss / accuracy | Val det/stoch | Decision |
|---|---:|---:|---:|---|
| batch 2,048 | 993.7 | 2.189 / 25.13% | 3.31% / 8.76% | reject drop-in |
| batch 4,096 | 998.5 | 1.854 / 42.97% | 9.09% / 12.89% | reject drop-in |
| FP16 128×6 | 953.6 | 1.463 / 56.77% | 22.31% / 26.61% | investigate |
| FP32 128×4 | 687.6 | **1.460 / 58.68%** | **23.14%** / 24.63% | promising |
| FP32 96×4 | **491.6** | 1.504 / 57.92% | 18.18% / 24.79% | secondary |

Larger batches delivered essentially unchanged kernel throughput and reduced
optimizer updates per epoch, so they are not safe drop-in changes. FP16 passed
bounded numerical/quality checks but improved complete BC time by only 4.5%,
despite a 2.27× isolated-kernel result. The 128×4 model is the best bounded
speed/quality candidate, but changing architecture would invalidate the
already-trained canonical checkpoint. It remains a separate longer comparison,
not a silent recipe substitution.

## Remaining and deferred work

### Required to complete the performance-first result

1. Freeze recipe A and produce independent BC→PPO seeds 29 and 47 without
   confirmation-set feedback.
2. Select each seed's checkpoint using the same validation/stopping contract.
3. Evaluate the sealed confirmation manifest once after all model choices are
   fixed.
4. Report across-seed mean, variation, difficulty strata, and confirmation
   uncertainty whether or not aspirational targets are met.

### Explicitly deferred

- Full RL-only reverse-curriculum training.
- Formal RL-only versus BC-only versus BC→RL comparison.
- Grid-aligned RGB-sprite observations modelled after the Sokoban planning work;
  this requires a new encoder contract, regenerated BC data, and incompatible
  checkpoints, and must be separated from recurrent/DRC changes.
- Full 128×4 architecture promotion, PPO mixed precision, channels-last, and
  profiler-guided scaler/synchronization work.
- Robot-band reweighting, alternate-optimal labels, symmetry augmentation,
  DAgger recovery states, relational heads, horizon ablations, and recurrent
  or DRC comparisons.

## Reproducibility and artifact map

| Artifact | Path |
|---|---|
| Canonical BC | `artifacts/phase2_experiments/phase5_canonical_bc_seed11_retry1` |
| Geometry benchmarks | `artifacts/phase2_experiments/phase5_geometry_{16x128,32x64,64x32}` |
| BC lever screen | `artifacts/phase2_experiments/phase5_bc_training_lever_screen` |
| First-rung A/B | `artifacts/phase2_experiments/phase5_screen_{a,b}_rung1` |
| First-rung decision | `artifacts/phase2_experiments/phase5_rung1_decision` |
| Second-rung A/B | `artifacts/phase2_experiments/phase5_screen_{a,b}_rung2` |
| Promotion decision | `artifacts/phase2_experiments/phase5_rung2_decision` |
| Promoted seed 11 | `artifacts/phase2_experiments/phase5_screen_a_promoted_seed11` |
| Detailed handoff | `docs/RLLearningExecutionLog.md` |

Generated experiment artifacts are intentionally Git-ignored; configs, code,
tests, decisions, and this report are tracked. The implementation/report commit
series for this remediation begins at `2f1cb44` and the promoted seed-11 result
was recorded at `18c23d9`.
