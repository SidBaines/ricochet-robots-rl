# RL Learning Remediation Execution Log

This is the live, detailed handoff document for
`docs/RLLearningSynthesizedPlan.md`. Update the snapshot whenever execution
state changes. Append chronological entries; never rewrite historical entries.

## Current handoff snapshot

- Updated: 2026-07-17 14:45 EDT
- Branch: `sid-learnable-agent`
- Phase: masking/BC stability complete; reverse-curriculum infrastructure next
- Latest completed commit: `f9fae11 feat: make RL benchmark observable and selectable`
- Working tree at start: the two independent source plans were untracked; no
  other changes were present
- Active experiment: none
- Last test gate: 99 passed, 9 intentionally skipped, 1 warning in 8.80s
- Immutable fixture: `artifacts/phase4_bank`, fingerprint
  `3579113b230a04406ad8f17e5695c3a1fa0ea2145c0262f9da42b16b2162e130`
- Immutable split: `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`
- Sealed confirmation: 375 unique puzzles/layouts, fingerprint
  `816d8583c7cdb75deaf978f0e5b275d30f6a7d506e644df1c04d15994d42776b`;
  no learning model has been evaluated on it
- Current action: commit masking/BC stability, then implement reverse start
  states and calibrated frontier gating
- Exact next command after handoff: `.venv-phase2/bin/python -m pytest src/tests/ -q`

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

### 2026-07-17 14:05 EDT — observability and evaluation foundation complete

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

### 2026-07-17 14:45 EDT — movement masking and BC→PPO stability complete

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
