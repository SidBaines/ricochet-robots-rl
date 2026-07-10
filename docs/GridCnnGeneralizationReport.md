# Grid-CNN Generalization Report

Run date: 2026-07-10
Branch: `sid-learnable-agent`
Baseline being fixed: `docs/Phase2BenchmarkReport.md` (stock-CNN PPO failed the held-out gate at 31.7% stochastic vs the 46.7% target-robot-only baseline, with 3.1% deterministic success and 91% no-movement actions)

## Executive summary

The Phase 2 stock-CNN failure was an **observation-encoding × architecture mismatch**, not a task, reward, curriculum, or PPO problem. Fixing the encoding and network — with no change to the fixture, splits, baselines, reward, or PPO hyperparameters other than budget — takes held-out performance from *below the trivial heuristic* to *near-solved*:

| Benchmark | Baselines (random / target-only) | Stock CNN (old) | Grid CNN (this work) |
|---|---|---|---|
| Phase 2 easy (8×8, no interior walls, lengths 1–2, 1 robot moved) | 29.5% / 46.7% | 31.7% stoch, 3.1% det | **75.0% ± 7.8% stoch, 69.8% ± 6.5% det** (3 seeds) |
| Phase 2b hard (8×8, interior walls, lengths 1–4, ≤2 robots moved) | 14.1% / 16.6% | — | 70.7% stoch, 68.5% det (1 seed, 409k steps) |
| Phase 2c (same fixture, two-robot bands up-weighted, 819k steps) | 14.1% / 16.6% | — | **84.8% stoch, 87.0% det**; single-robot strata 97–100%, two-robot 67–68% |

All numbers are held-out test suites disjoint from training by layout hash, on the unchanged deterministic fixtures and evaluation protocol.

## Diagnosis

The failed baseline fed SB3's NatureCNN a 40×40 `rgb_cell_image` canvas whose real content is a 17×17 patch of single-pixel cells and walls centered in black padding. NatureCNN's first convolution (8×8 kernel, stride 4) aliases away single-pixel geometry, and the whole board is compressed to a 1×1×64 tensor before the linear head. The policy literally could not see the walls it needed to plan around — consistent with its 91% no-movement collapse.

The environment already provided a lossless encoding: `obs_mode="image"` structured planes (4 wall channels + goal + target mask + one channel per robot) at native board resolution.

## Fix

- **`src/models/grid_cnn.py` — `GridPlanesCnn`**: stride-1, same-padded 3×3 CNN (no pooling, no striding), flatten + linear projection. Exact robot/goal/wall positions stay recoverable through the whole network. ~1.26M policy parameters at 8×8.
- **`src/benchmarking/phase2.py`**: `Phase2Config` gains `obs_mode`, `policy` (`stock_cnn` | `grid_cnn`), `features_dim`, `conv_channels`, `n_conv_layers`, `two_robot_band_weight`, `run_overfit_milestone`; a shared `_build_model`; and fixture-derived (rather than Cartesian-product) split strata checks so harder fixtures validate.
- **`src/env/criteria_env.py`**: curriculum levels may carry explicit stratified `bands` (with weights); the auto-built equal-length bands remain the default.
- Contract tests in `src/tests/test_grid_cnn.py`.

## Experiment log

### Experiment 1 — hypothesis test (`configs/phase2_exp1_grid_cnn.yaml`)

Same fixture (fingerprint `29c1aeb0…`), split (`b72eb184…`), and baselines as the Phase 2 benchmark; planes obs + GridPlanesCnn; seed 11; 204,800 steps (~2.4k steps/s on MPS).

- Tiny-set overfit gate: **100% deterministic at 25,600 steps** (stock CNN: 37.5% at 51,200).
- Held-out test: **62.5% det / 66.9% stoch**. Validation rose monotonically (9.4% → 71.9% det) — distribution learning, not memorization.

### Experiment 2 — three-seed confirmation (`configs/phase2_exp2_grid_cnn_3seed.yaml`)

Per-seed stochastic test success 66.9% / 82.5% / 75.6% (seeds 11/29/47); mean **75.0% ± 7.8%** (pooled Wilson 70.9%–78.7%); deterministic mean **69.8% ± 6.5%**. Every seed clears the gate individually. **Formal held-out milestone: passed.**

### Phase 2b — harder fixture (`configs/phase2b_grid_cnn.yaml`, bank `artifacts/phase2b_bank`)

8,922-puzzle deterministic fixture: 8×8, 2 robots, 1 edge-T + 1 central-L wall per quadrant, optimal lengths 1–4, robots-moved 1–2; split 6,736/1,096/1,090 by layout hash. Baselines collapse here (random 14.1%, target-only 16.6%, 0% on two-robot puzzles).

Seed 11, 409,600 steps: held-out **68.5% det / 70.7% stoch**. Single-robot strata 99–100% at every length; two-robot blocker puzzles 23–30%.

### Phase 2c — two-robot reweighting (`configs/phase2c_grid_cnn_2robot.yaml`)

Same fixture/split. Explicit band weights raise two-robot training exposure from ~13% to ~41%; budget 819,200 steps.

Held-out test: **87.0% det / 84.8% stoch**. Single-robot 100%/97%; two-robot **68%/67%** (up from 23%/30%). Mean successful optimality gap 0.66 moves. At the matched 409,600-step checkpoint the reweighted run led the unweighted one 76.4% vs 63.6% deterministic validation, so the distribution shift itself — not just the doubled budget — drives most of the gain.

## Interpretation

- A feed-forward CNN generalizes well across unseen 8×8 layouts once the observation preserves board geometry. Sparse-ish reward, the bank pipeline, PPO defaults, and modest budgets were never the blocker.
- Two-robot blocker puzzles are the genuinely hard skill; they respond strongly to exposure reweighting and remain the frontier (68% vs 100% single-robot).
- The measurement protocol (deterministic fixtures, layout-hash splits, baselines, gates) held up: every claim above is a like-for-like comparison on frozen suites.

## Next gate — Phase 3 (in progress)

Scale the identical recipe to the real game footprint: 16×16 board, 4 robots, 16 actions, interior walls, optimal lengths 1–6 (`configs/phase3_16x16_grid_cnn.yaml`, bank `artifacts/phase3_bank`), 6-layer GridPlanesCnn (~4.5M parameters, ~1.4k steps/s on MPS), 1.64M-step budget. Acceptance: mean stochastic held-out success ≥ the target-robot-only baseline, with stratified reporting as above.

## Reproduction

```bash
uv venv --python 3.11 .venv-phase2
uv pip install --python .venv-phase2/bin/python -r requirements-ci.txt
uv pip install --python .venv-phase2/bin/python -e .
# Easy benchmark, three seeds:
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase2_exp2_grid_cnn_3seed.yaml \
  --output-dir artifacts/phase2_experiments/exp2_grid_cnn_3seed
# Harder fixture (fresh bank dir is created and labelled deterministically):
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase2c_grid_cnn_2robot.yaml \
  --bank-dir artifacts/phase2b_bank \
  --output-dir artifacts/phase2_experiments/phase2c_grid_cnn_2robot
```

Each run writes `results.json` (full per-episode records, provenance, fingerprints) and `report.md` under its output directory.
