# Grid-CNN Generalization Report

Run dates: 2026-07-10 – 2026-07-12
Branch: `sid-learnable-agent` (pushed to origin)
Baseline being fixed: `docs/Phase2BenchmarkReport.md` (stock-CNN PPO failed the held-out gate at 31.7% stochastic vs the 46.7% target-robot-only baseline, with 3.1% deterministic success and 91% no-movement actions)

**Reading order for a new contributor:** this report (the full experiment arc) → `CLAUDE.md` (hard-won operational lessons) → `WorkInProgress.md` "Phase 4b" (the live handoff and resume command). Every experiment runs through the deterministic harness in `src/benchmarking/phase2.py`; one YAML per experiment under `configs/`; outputs (gitignored) under `artifacts/phase2_experiments/<name>/` with a full `results.json` and machine-written `report.md`.

## Executive summary

The Phase 2 stock-CNN failure was an **observation-encoding × architecture mismatch**, not a task, reward, curriculum, or PPO problem. Fixing the encoding and network — with no change to the fixture, splits, baselines, reward, or PPO hyperparameters other than budget — takes held-out performance from *below the trivial heuristic* to *near-solved*:

| Benchmark | Baselines (random / target-only) | Stock CNN (old) | Grid CNN (this work) |
|---|---|---|---|
| Phase 2 easy (8×8, no interior walls, lengths 1–2, 1 robot moved) | 29.5% / 46.7% | 31.7% stoch, 3.1% det | **75.0% ± 7.8% stoch, 69.8% ± 6.5% det** (3 seeds) |
| Phase 2b hard (8×8, interior walls, lengths 1–4, ≤2 robots moved) | 14.1% / 16.6% | — | 70.7% stoch, 68.5% det (1 seed, 409k steps) |
| Phase 2c (same fixture, two-robot bands up-weighted, 819k steps) | 14.1% / 16.6% | — | **84.8% stoch, 87.0% det**; single-robot strata 97–100%, two-robot 67–68% |
| Phase 4 (16×16, 4 robots, lengths 1–8; scale-only) | 4.4% / 8.5% | — | 3.3% stoch — **failed** (from-scratch PPO, exploration wall) |
| Phase 4b (same 16×16; BC warmstart) | 4.4% / 8.5% | — | **30.6% stoch / 24.8% det from BC alone** (step 0, pre-PPO); PPO fine-tune in progress |

All numbers are held-out test suites disjoint from training by layout hash, on the unchanged deterministic fixtures and evaluation protocol. The 8×8 milestones are final; 16×16 is the live frontier (Phase 4/4b below).

## Diagnosis

The failed baseline fed SB3's NatureCNN a 40×40 `rgb_cell_image` canvas whose real content is a 17×17 patch of single-pixel cells and walls centered in black padding. NatureCNN's first convolution (8×8 kernel, stride 4) aliases away single-pixel geometry, and the whole board is compressed to a 1×1×64 tensor before the linear head. The policy literally could not see the walls it needed to plan around — consistent with its 91% no-movement collapse.

The environment already provided a lossless encoding: `obs_mode="image"` structured planes (4 wall channels + goal + target mask + one channel per robot) at native board resolution.

## Fix

- **`src/models/grid_cnn.py` — `GridPlanesCnn`**: stride-1, same-padded 3×3 CNN (no pooling, no striding), flatten + linear projection. Exact robot/goal/wall positions stay recoverable through the whole network. ~1.26M policy parameters at 8×8.
- **`src/benchmarking/phase2.py`**: `Phase2Config` gains `obs_mode`, `policy` (`stock_cnn` | `grid_cnn`), `features_dim`, `conv_channels`, `n_conv_layers`, `two_robot_band_weight`, `run_overfit_milestone`; a shared `_build_model`; and fixture-derived (rather than Cartesian-product) split strata checks so harder fixtures validate.
- **`src/env/criteria_env.py`**: curriculum levels may carry explicit stratified `bands` (with weights); the auto-built equal-length bands remain the default.
- Contract tests in `src/tests/test_grid_cnn.py`.

## Architectures (`src/models/grid_cnn.py`)

Both extractors take channels-first structured planes and never pool or stride, so a robot one cell over changes the output — the property NatureCNN destroyed. Observation is `(C, H, W)` where `C = 4 wall planes + goal + target mask + num_robots` (8 channels at 8×8/2-robot, 10 channels at 16×16/4-robot).

- **`GridPlanesCnn`** (8×8 work): `n_conv_layers` stride-1 same-padded 3×3 convs (default 4 × 64 ch) → flatten → linear to `features_dim`. ~1.26M policy params at the 8×8 default.
- **`GridPlanesResNet`** (16×16 work): 3×3 stem → `n_blocks` residual blocks (two same-padded 3×3 convs + identity skip, default 6 × 128 ch) → 1×1 conv reducing to `reduce_channels` (32) → flatten → linear to `features_dim` (512). ~6.1M policy params at the 16×16 default. The 1×1 reduction keeps the flattened vector small (`32·16·16`) despite the wide conv trunk.

Both are wired as SB3 `MlpPolicy` + custom `features_extractor` with `normalize_images=False` (planes are already 0/1 floats), and are selected by `policy:`/`model.type:` `grid_cnn`|`grid_resnet` in the harness and `train_agent.py`, with policy↔obs-mode validation and construction/predict/update contract tests.

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

## Phase 4b — BC warmstart (`configs/phase4b_bc_16x16.yaml`)

Behaviour-cloning warmstart addresses the exploration bottleneck directly: teach the policy the solver's move function *supervised* before PPO, so PPO starts from a policy that already reaches goals often enough to get reward signal on hard puzzles.

**Mechanism** (`_bc_dataset` / `_bc_pretrain` in `src/benchmarking/phase2.py`):
- Each training-split puzzle carries its BFS-optimal action sequence. We replay that plan in the env and record a `(observation, action)` pair at **every state along the trajectory** — not just the first move — yielding **210,947 pairs from the 41,395 training puzzles** (extracted in ~18s, planes stored as uint8 to stay compact). A sanity check asserts every stored plan actually reaches the goal.
- Train the policy's action distribution by NLL / cross-entropy (Adam, `bc_learning_rate`, `bc_batch_size`) for `bc_pretrain_epochs`, then hand the warmstarted network to the unchanged staged-curriculum PPO loop. Value head is left to PPO. BC metrics log to W&B as `bc/loss`, `bc/accuracy`.
- **Hard-won lesson (in code comment):** SB3's PPO initializes the action head with orthogonal gain **0.01**, which scales all upstream supervised gradients down ~100× and pins cross-entropy on the uniform-policy plateau (accuracy stuck near 1/n_actions). BC only learns after re-initializing `action_net` at unit gain. Any future supervised-pretraining-into-SB3 work must do the same.

**Config:** 25 BC epochs (batch 1024, lr 1e-3) → the exact Phase 4 PPO setup (6.1M-param `GridPlanesResNet`, 4-stage length curriculum 1–2→1–4→1–6→1–8 with the 250k-episode safety-valve, 10.24M steps, ~630–715 steps/s on MPS). Baselines on this distribution: random 4.4%, target-robot-only 8.5%.

**Result — BC alone clears the gate before any PPO (2026-07-12).** The 25-epoch BC phase reached **89.3% optimal-action accuracy** on the training pairs. The **step-0 checkpoint (BC weights, zero PPO steps)** already scores **30.6% stochastic / 24.8% deterministic on the held-out test**, on the full length-1–8 mix — vs the 8.5% target-robot-only gate and the 3.3% that the identical-but-from-scratch Phase 4 run *ended* at. It is non-trivial at every stratum (held-out stochastic by optimal length: 1→100%, 2→58%, 3→69%, 4→33%, 5→17%, 6→17%, 7→11%, 8→6%).

This is the decisive confirmation of the Phase 4 diagnosis: **16×16 was an exploration/credit-assignment problem, not capacity, data, or wall-clock.** Given the solver's move function as a prior, the "wall" at lengths ≥3 disappears immediately. From-scratch PPO could not discover reward on deep puzzles; supervised cloning hands it that competence directly.

**PPO fine-tuning in progress.** The warmstarted network is now under the staged-curriculum PPO loop. The open question is whether PPO *builds on* the prior or *forgets* it (entropy bonus + policy-gradient noise can wash out a cloned policy early). The critical check is the first PPO checkpoint (1.02M steps): it must hold or beat the 30.6% step-0 number, not regress. If it collapses, the fix is a gentler fine-tune (lower PPO lr, lower `ent_coef`, or a KL-to-BC-prior penalty). Full run ~5.5–6h; W&B run `phase4b-bc-16x16-seed11`. Continuing-agent notes and resume command in `WorkInProgress.md` under "Phase 4b".

## Connection to the original DRC research goal

The project's `docs/ResearchPlan.md` and `docs/rr_base_prompt.md` are built around the Deep Repeating ConvLSTM (DRC) from Guez et al. 2019, as analyzed for Sokoban in [arXiv:2407.15421](https://arxiv.org/html/2407.15421v3), with mechanistic interpretability of the learned *plan* as the eventual payoff. The work in this report is deliberately **feed-forward, not recurrent** — the January review's recovery plan said to establish a trustworthy non-recurrent baseline before repairing the (still-broken) DRC/RecurrentPPO paths in `src/models/`. Three connections:

1. **Same core design principle, minus recurrence.** The reason DRC hidden states are interpretable in that paper is that computation stays spatial and grid-aligned throughout; the old baseline violated exactly this by aliasing the grid through a stride-4 conv. `GridPlanesCnn`/`GridPlanesResNet` restore the spatial-computation property — "the DRC encoder philosophy without the recurrent core."
2. **8×8 calibrates where planning is *not* needed.** A fixed-depth net solves held-out 8×8 puzzles up to length 4 with two-robot blockers at ~87% — so for shallow instances amortized pattern-matching suffices, no iterative computation required. That is the control the DRC comparison needs and didn't previously exist.
3. **16×16 marks where recurrence should earn its keep.** Fixed-depth compute stalls on lengths ≥5 (the exploration wall). That is precisely the regime where DRC's "more ticks = more planning compute / pondering helps hardest levels" result predicts a recurrent core would help — and the env already has the optional no-op "thinking step" action the plan earmarked for it. The benchmark harness now gives a clean A/B (same fixture, splits, gates: feed-forward vs DRC). Caveat: the paper's DRC used ~2B IMPALA steps; since even the feed-forward 16×16 run looks compute/exploration-starved, fix the data/exploration axis (BC warmstart) before attributing any residual gap to architecture, or the comparison is confounded. Ricochet Robots is arguably a *better* interpretability probe than Sokoban because the bank stores BFS-optimal plans for every puzzle — ground-truth "which robot, which direction" labels for probes, the direct analogue of the paper's box-direction probes.

## Artifacts, fixtures and fingerprints

Fixtures are deterministic; `build_fixture_bank(..., workers=N)` produces byte-identical fingerprints for any worker count (verified against the recorded phase2b manifest). Banks live under `artifacts/<bank>/` (gitignored); the fingerprint is in each bank's `phase2_fixture.json`.

| Bank | Board / robots | Walls | Lengths / robots-moved | Puzzles | `fixture_sha256` (prefix) |
|---|---|---|---|---:|---|
| `phase2_bank` | 8×8 / 2 | none | 1–2 / 1 | 460 | `29c1aeb066ab` |
| `phase2b_bank` | 8×8 / 2 | 1 ET + 1 CL /quad | 1–4 / 1–2 | 8,922 | `001a87948d54` |
| `phase3_bank` | 16×16 / 4 | 2 ET + 2 CL /quad | 1–6 / 1–2 | 6,588 | `f4aea81c9e8c` |
| `phase4_bank` | 16×16 / 4 | 2 ET + 2 CL /quad | 1–8 / 1–3 | 55,143 | `3579113b230a` |

Split manifests are keyed by `layout_hash` and carry their own `fingerprint`; both fixture and split fingerprints are contracts — a refactor that changes them is a bug. `test_bank_split_perf.py` pins the split hash contract and the `_split_mask` scaling bound.

## Reproduction

```bash
# Environment
uv venv --python 3.11 .venv-phase2
uv pip install --python .venv-phase2/bin/python -r requirements-ci.txt
uv pip install --python .venv-phase2/bin/python -e .

# 8×8 headline: three-seed held-out benchmark (~30 min MPS)
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase2_exp2_grid_cnn_3seed.yaml \
  --output-dir artifacts/phase2_experiments/exp2_grid_cnn_3seed

# 8×8 hard, two-robot reweighted (best 8×8 result, 87% det held-out)
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase2c_grid_cnn_2robot.yaml \
  --bank-dir artifacts/phase2b_bank \
  --output-dir artifacts/phase2_experiments/phase2c_grid_cnn_2robot

# 16×16 current frontier: BC warmstart + staged PPO (~5.5–6h MPS)
nohup caffeinate -is .venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase4b_bc_16x16.yaml \
  --bank-dir artifacts/phase4_bank \
  --output-dir artifacts/phase2_experiments/phase4b_bc_16x16 \
  > artifacts/phase2_experiments/phase4b_bc_16x16.log 2>&1 &
```

Fixture banks are created and labelled deterministically on first use if absent (the 55k-row phase4 bank takes ~30 min; use the parallel builder in `WorkInProgress.md`). Each run writes `results.json` (full per-episode records, provenance, both fingerprints, `bc_pretrain` stats) and `report.md` under its output directory. Long runs must be launched detached (`nohup caffeinate -is … &`) — harness background tasks have been killed at ~15–20 min on this machine.
