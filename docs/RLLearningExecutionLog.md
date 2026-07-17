# RL Learning Remediation Execution Log

This is the live, detailed handoff document for
`docs/RLLearningSynthesizedPlan.md`. Update the snapshot whenever execution
state changes. Append chronological entries; never rewrite historical entries.

## Current handoff snapshot

- Updated: 2026-07-17 13:29 EDT
- Branch: `sid-learnable-agent`
- Phase: documentation foundation
- Latest completed commit: `a863b53 feat: 2-D difficulty heatmaps + persist per-checkpoint joint table`
- Working tree at start: the two independent source plans were untracked; no
  other changes were present
- Active experiment: none
- Last test gate: 88 passed, 9 intentionally skipped, 1 warning in 6.51s
- Immutable fixture: `artifacts/phase4_bank`, fingerprint
  `3579113b230a04406ad8f17e5695c3a1fa0ea2145c0262f9da42b16b2162e130`
- Immutable split: `c0e97329b1096f0531de49f3c0694162a811402fea54c7e2866553711d9dab79`
- Current action: commit the plan/log foundation, then implement observability
  and evaluation controls
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
