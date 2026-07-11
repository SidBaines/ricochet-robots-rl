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

## Phase 3 — 16×16, 4 robots: honest negative result and open problem

Scaling the identical recipe to the real game footprint (16×16, 4 robots, 16 actions, interior walls, lengths 1–6; `configs/phase3_16x16_grid_cnn.yaml`, bank `artifacts/phase3_bank`, 6,588 puzzles; baselines random 6.3% / target-only 12.3%) did **not** clear the gate within this session's scale:

| Run | Budget | Held-out test | Gate (12.3%) |
|---|---|---|---|
| No curriculum (`phase3_16x16_grid_cnn`) | 1.64M steps | 1.6% det / 5.1% stoch | failed — flat throughout |
| Staged curriculum (`phase3b_16x16_staged`, lengths 1–2 → 1–4 → 1–6) | 1.64M steps | 3.2% det / 7.0% stoch | failed |

The staged mechanism itself works: stage 0 reached the 70% training-success threshold and advanced at ~400k steps, and length-1 held-out success reached 53% stochastic. But training-suite gains did not transfer beyond that (lengths 3+ at 0%): with only ~600 stage-0 training layouts, the model memorizes rather than generalizes, and the run ended stuck mid-stage-1.

**Diagnosis: data/scale starvation, not a broken mechanism.** Recommended next steps, in order:

1. **10–50× fixture size** — the parallel deterministic builder makes this cheap (10k attempts ≈ 3 min at 8 workers; 100k+ attempts is practical).
2. **Wider/deeper residual grid CNN** — 6×64-channel same-padded convs is likely undersized for 16×16 pattern diversity.
3. **10M+ step budget** with per-stage held-out evaluation so memorize-vs-generalize is visible during training.
4. **Behaviour-cloning warmstart** from the bank's stored optimal plans (every fixture puzzle carries its BFS-optimal action sequence).
5. Multi-seed formal gating only after a single seed clears.

## Phase 4 — the scaling run: items 1–3 alone are not enough

Phase 4 executed the first three recommendations together (`configs/phase4_16x16_big.yaml`, bank `artifacts/phase4_bank`): a 55,143-puzzle fixture (lengths 1–8, robots-moved 1–3, BFS depth 8), a 6.1M-parameter residual `GridPlanesResNet`, a 4-stage length curriculum (1–2 → 1–4 → 1–6 → 1–8) with a 250k-episode stall safety-valve, and 10.24M steps at a measured 716 steps/s. Baselines on this distribution: random 4.4%, target-robot-only 8.5%.

Result: **held-out gate failed** — 3.3% stochastic vs 8.5% required; deterministic 0.8% with 95% no-movement. Mid-run validation showed genuine learning on short puzzles (length-1 held-out peaked at 67% during stages 0–1), but every stage advanced via the safety-valve rather than the 65% mastery threshold (windows 25% / 18% / 10%), and the final policy had regressed to ~random on the test mix.

Updated diagnosis: past the easiest stage, random success on length ≥3 puzzles is ~0–5%, so from-scratch PPO receives almost no reward signal — the binding constraint at 16×16 is **exploration/credit assignment**, which more data, capacity, and wall-clock do not fix. The productive next lever is item 4: **behaviour-cloning warmstart from the bank's stored optimal plans**, then PPO fine-tuning under the same gates. A perf note for anyone rerunning: `BankSampler._split_mask` is now vectorized (was `iterrows()`, 83ms per reset on this bank — see `src/tests/test_bank_split_perf.py`).

## Phase 4b — BC warmstart: implemented, paused mid-first-run

Behaviour-cloning warmstart is implemented and committed (`bc_pretrain_epochs` etc. on `Phase2Config`; `configs/phase4b_bc_16x16.yaml`): before PPO, the policy is trained supervised on every `(state, optimal-first-action)` pair replayed from the training split's stored BFS plans — **210,947 pairs from 41,395 puzzles**, extracted in 19s. The first run was intentionally stopped at BC epoch 7/25 to free the machine; by then optimal-action accuracy had already climbed 15% → **65%** (random 6.25%), so the supervised signal is strong. The harness has no mid-run resume, so continuing means rerunning `phase4b_bc_16x16` from scratch (~5.5–6h). The full resume command, expected timeline, and decision criteria for the continuing agent are in `WorkInProgress.md` under "Phase 4b".

## Reproduction (16×16 fixture)

```bash
# Parallel deterministic fixture build (identical fingerprints to sequential):
.venv-phase2/bin/python - <<'PY'
from pathlib import Path
import yaml
from src.benchmarking.phase2 import Phase2Config, build_fixture_bank
config = Phase2Config.from_mapping(yaml.safe_load(Path("configs/phase3_16x16_grid_cnn.yaml").read_text()))
build_fixture_bank(Path("artifacts/phase3_bank"), config, workers=8)
PY
```

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
