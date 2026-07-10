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

## Conventions

- Experiments run through the deterministic harness in
  `src/benchmarking/phase2.py`; one YAML per experiment in `configs/`;
  outputs under `artifacts/phase2_experiments/<name>/` (gitignored).
- Record every experiment chronologically in `WorkInProgress.md` (append,
  never rewrite history) and keep `docs/GridCnnGeneralizationReport.md`
  current for headline results.
- W&B: set `wandb_project`/`wandb_run_name` in the experiment YAML (opt-in;
  machine is already authenticated via ~/.netrc).
