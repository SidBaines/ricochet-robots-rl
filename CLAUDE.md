# CLAUDE.md

RL agents for Ricochet Robots + mechanistic interpretability (planned). Python 3.11,
SB3 PPO on MPS. Reference environment: `.venv-phase2` (see README for setup).

## Commands

- Tests: `.venv-phase2/bin/python -m pytest src/tests/ -q`
- Benchmark harness: `.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py --config configs/<name>.yaml --bank-dir artifacts/<bank> --output-dir artifacts/phase2_experiments/<name>`

## Hard-won lessons (do not relearn these)

- **Benchmark throughput on the real training factory, not a synthetic env.**
  PPO throughput is often dominated by episode *resets*, and the bank-backed
  training env (`_training_factory` → `BankCurriculumWrapper` →
  `CriteriaFilteredEnv`) samples a puzzle from the parquet bank and rebuilds the
  inner env on every reset. A synthetic `RicochetRobotsEnv` factory hides all of
  that. A 2026-07 run measured 712 steps/s synthetic but 250 steps/s real —
  the gap was an `iterrows()` scan of the whole 55k-row partition in
  `BankSampler._split_mask` (now vectorized; regression test in
  `src/tests/test_bank_split_perf.py`).
- **No `pandas.iterrows()` in per-episode/per-reset paths.** It builds a Series
  per row; on large partitions that is seconds per call.
- **Fingerprints are contracts.** Fixture and split fingerprints
  (`fixture_sha256`, split manifest `fingerprint`) must never change from a
  refactor. When touching bank/split/generation code, re-derive a recorded
  manifest (e.g. phase2b) and assert the fingerprint is identical before
  trusting the change.
- **Launch long jobs detached.** Harness background tasks have been killed at
  ~15–20 minutes on this machine. Anything longer:
  `nohup caffeinate -is <cmd> > <log> 2>&1 &`, then watch the log file.
- **Observation encoding beats architecture tweaks.** The historical
  "models don't learn" failure was NatureCNN aliasing single-pixel walls on
  `rgb_cell_image`. Use `obs_mode: image` structured planes with the
  stride-1 same-padded extractors (`grid_cnn`/`grid_resnet` in
  `src/models/grid_cnn.py`); never feed planes through pooling/striding
  layers that destroy grid geometry. See
  `docs/GridCnnGeneralizationReport.md`.
- **Mixed hard distributions stall PPO from scratch.** Use the staged length
  curriculum (`curriculum_stage_max_lengths`) with the stall safety-valve
  (`curriculum_max_episodes_per_stage`) rather than training on lengths 1–8
  uniformly.
- **Never bootstrap on truncation when the horizon is in the observation.**
  (Fixed 2026-07-25 in `RecurrentPPO`, which now auto-detects this from
  `EnvConfig.time_channel` and raises on the inconsistent combination; the
  lesson stays because the same trap exists in any new trainer.)
  `EnvConfig.time_channel=True` puts remaining time in the obs, so the MDP is
  finite-horizon and fully observed and `V(s, remaining=0) ≡ 0`. The env
  increments `_steps` before building the obs, so the truncation observation has
  time channel exactly 0 — and it is never stored in the buffer, so it is never
  a regression target. `drc_train.py`'s `collect_rollout` nonetheless fed it to
  the value head and added `gamma*V` to the reward: an untrained extrapolation
  injected straight into the returns, with no contraction guarantee. Measured on
  `drc33_p1_12x12_planefree`, `V(expiry)` was **+0.71 at 80M and −12.3 at 87M**
  against `goal_reward=1.0` (timing out paid better than solving), and it drove
  the 89M value/policy collapse. `src/groundup/env.py` emits no
  `TimeLimit.truncated`, so SB3/`train.py` never did this — it was a DRC-trainer
  regression. See `docs/Drc87MRestartCollapseInvestigation.md` §6.1.
  Diagnostic: evaluate `V` with the time channel zeroed; it must be ~0.
- **`value_loss` is a bad alarm; `explained_variance` is worse.** In the same
  run `value_loss` swung 13.7↔71.3 between consecutive iterations while
  `explained_variance` held 0.81–0.84 as the loss grew 10⁴×. Use `V_mean`/`V_std`
  on a fixed probe set, plus the policy-gradient share of the update. Both led
  `val_det` by 5–20M steps.
- **`max_grad_norm` couples the value and policy gradients.** One
  `clip_grad_norm_` over the combined loss means an exploding value term
  starves the policy: share of the update fell 98.8% (80M) → 3.7% (87M) → 0.14%
  (90M) while `‖g_policy‖` itself *grew*. Huber value loss bounds this cheaply.
- **`caffeinate -is` does not prevent clamshell (lid-close) sleep.** Use
  `caffeinate -dis`, and treat low battery as a stop condition, not a pause.
- **SB3's default Tanh MLP heads saturate on CNN features.** Unbounded ReLU
  features (observed O(1000) at 16×16) drive every Tanh pre-activation past
  ±2 → the value head collapses to a literally constant output and learning
  stalls. Hit twice independently (legacy Phase 4 audit; groundup E3 v1).
  Always pass `activation_fn=nn.ReLU` and bound feature scale (LayerNorm on
  the extractor output). Regression test:
  `src/tests/test_groundup.py::test_policy_heads_relu_and_bounded_features`.
  Diagnostic signature: `V(s)` identical across states ⇒ check head
  saturation before blaming exploration or curriculum.

## Conventions

- Experiments run through the deterministic harness in
  `src/benchmarking/phase2.py`; one YAML per experiment in `configs/`;
  outputs under `artifacts/phase2_experiments/<name>/` (gitignored).
- Record every experiment chronologically in `WorkInProgress.md` (append,
  never rewrite history) and keep `docs/GridCnnGeneralizationReport.md`
  current for headline results.
- W&B: set `wandb_project`/`wandb_run_name` in the experiment YAML (opt-in;
  machine is already authenticated via ~/.netrc).
