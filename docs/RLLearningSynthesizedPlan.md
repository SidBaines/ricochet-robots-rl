# Performance-First RL Learning Remediation Plan

**Status:** seed-11 promotion complete; seeds 29/47 and confirmation pending
**Created:** 2026-07-17  
**Inputs:** `RLLearningInvestigationPlan.md` and `Phase5ReverseCurriculumPlan.md`

The consolidated implementation and seed-11 result are reported in
`RLLearningRemediationReport.md`; the chronological handoff remains
`RLLearningExecutionLog.md`.

## Objective

Produce the strongest trustworthy 16×16 feed-forward BC→PPO model first while
building, testing, and documenting the reverse-curriculum machinery needed by a
later RL-only program. The current session does not run the formal RL-only,
BC-only, or three-way comparison.

The Phase 4 bank, fixture fingerprint `3579113b230a…`, split fingerprint
`c0e97329b109…`, structured observations, residual model size, reward, and
20-step horizon remain fixed.

## Success and selection rules

- Select recipes and checkpoints by stochastic validation success.
- Break ties by deterministic success and then earlier timestep.
- Require exactly zero no-movement actions for the deployed masked policy,
  healthy optimizer/representation telemetry, and no material deterministic or
  multi-robot regression relative to the recipe's own BC-only checkpoint.
- Promote a seed-11 full run to seeds 29 and 47 only if its selected checkpoint
  reaches at least 36% validation stochastic success, at least 25%
  deterministic success, and loses no more than two points on either
  multi-robot marginal relative to BC-only.
- Aspirational three-seed confirmation targets are 32% stochastic, 28%
  deterministic, 22% two-robot stochastic, and 10% three-robot stochastic.
  Missing a target is a reported result, not permission to tune on confirmation.

## Execution sequence

1. Make policy construction, run provenance, checkpoints, PPO/activation
   telemetry, W&B axes, validation uncertainty, and final-test controls
   explicit. Create a fresh confirmation manifest excluding every exposed test
   puzzle and layout.
2. Add training-time movement masking with MaskablePPO, mask-aware evaluation
   and BC, wrapper forwarding, expert-action validation, and real-factory
   throughput gates.
3. Add held-out BC evaluation, best-epoch selection, a reusable initial
   checkpoint, cumulative optimizer schedules, and `target_kl`.
4. Implement reverse plan-prefix start states, per-band/frontier tracking,
   baseline-calibrated gates, forced-transition labelling, and smoke/performance
   tests. Do not run the full RL-only experiment in this session.
5. Train one canonical seed-11 BC initialization. Run two matched 2.048M-step
   masked BC→PPO screens using the existing length curriculum:
   - A: historical optimizer (`lr=3e-4`, `ent_coef=0.01`, no KL guard).
   - B: stability bundle (`lr=1e-4`, `ent_coef=0.001`, `target_kl=0.03`).
6. Promote the best stable screen to 10.24M steps. If its seed-11 promotion
   gates pass, freeze the recipe, run seeds 29 and 47, and evaluate the sealed
   confirmation suite once.

## Evaluation contract

- The existing test suite is development data. Screening is validation-only.
- Screening uses five stochastic episodes per validation puzzle; the full run
  uses ten with fixed seeds across checkpoints.
- Uncertainty is clustered by puzzle.
- Every checkpoint, including post-BC step 0, saves model, evaluation,
  curriculum state, effective config, and provenance.
- Confirmation is selected deterministically from unused test layouts with
  seed `20260717`, up to 20 puzzles per achievable joint difficulty cell.
- Long jobs use unique output directories and detached `nohup caffeinate`.

## Documentation contract

`docs/RLLearningExecutionLog.md` is the detailed handoff source of truth. Its
current snapshot is refreshed whenever state changes and chronological entries
are appended before and after every work package, run, interruption, result,
and promotion decision. `WorkInProgress.md` receives concise milestone
summaries. Each coherent work package is committed with the full test suite
green.

## Deferred work

- Full 16×16 RL-only reverse-curriculum experiment.
- Formal RL-only versus BC-only versus BC→RL comparison.
- Robot-band reweighting, alternate-optimal labels, exact symmetry
  augmentation, DAgger recovery states, and a relational action head.
- Horizon/termination semantics ablation.
- DRC/recurrent comparison.
- Controlled structured-plane versus grid-aligned RGB-sprite observation
  comparison, following the Sokoban setup in Guez et al. The observation
  change and the DRC/recurrent architecture change must be tested separately.

The execution log records prerequisites and resumption criteria for each item.
