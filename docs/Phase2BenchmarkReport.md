# Phase 2 Benchmark Report

Run date: 2026-07-10

Implementation commit: `15230080c60fababd39be253d3c168bb0b2e824f`

Run state: clean worktree

Outcome: benchmark infrastructure complete; tiny overfit milestone passed; held-out milestone failed

## Executive summary

Phase 2 now has a reproducible feed-forward benchmark with immutable train, validation, and test layouts; random, target-robot-only, and oracle baselines; stratified metrics; fixed training and validation curves; three PPO seeds; an automated tiny-set overfit gate; uncertainty estimates; and complete run provenance.

The MPS training path is healthy. The stock CNN learned the eight-board overfit suite to 82.5% stochastic success and trained at roughly 2.3k–2.5k environment steps/second. It did not learn the broader distribution successfully: final stochastic test success averaged 31.7% across three seeds, compared with 29.5% for random actions and 46.7% for the target-robot-only heuristic. Deterministic success averaged only 3.1%, with 91.0% of its actions producing no movement.

This is a useful negative result. It separates a working training/device pipeline from a feed-forward PPO baseline that does not yet generalize, and it means the held-out success gate must remain failed.

## MPS and reference-environment audit

The original active environment used Python 3.14, while the project declares Python 3.11. A clean Python 3.11.15 environment was therefore created for Phase 2. The committed `requirements-ci.txt` also contained `numpy==2.5.1`, which is not installable on Python 3.11; it is now pinned to the compatible `2.4.6` release.

The clean benchmark used:

- macOS 26.5.1 on Apple Silicon
- Python 3.11.15
- PyTorch 2.13.0 with MPS available
- Stable-Baselines3 2.9.0
- Gymnasium 1.3.0
- NumPy 2.4.6, pandas 3.0.3, and PyArrow 25.0.0

Readiness fixes made during the audit:

- Fresh PPO runs now receive the configured seed explicitly.
- Accelerator profiling synchronizes both before and after a measured region, so asynchronous MPS work is included in timings.
- Stochastic benchmark evaluation uses its own explicitly seeded PyTorch RNG and restores CPU, CUDA, and MPS RNG state afterward. Evaluation therefore does not alter subsequent training.
- An MPS-only integration test performs a stock-CNN update, checks finite parameters, saves, reloads, and predicts on MPS.
- `.venv*` directories are ignored.

Direct stock-CNN PPO smoke tests passed for prediction, learning, save, reload, and post-reload prediction. Puzzle generation and BFS labelling also passed, but they are CPU/NumPy/Python workloads and do not benefit from selecting MPS.

## Benchmark definition

The benchmark deliberately fixes every tensor contract and varies only solution length:

| Property | Value |
|---|---|
| Board | 8×8, two robots, no extra edge-T or central-L walls |
| Observation | RGB-cell image, channels first, `(3, 40, 40)`, `uint8` |
| Action space | Eight actions: two robots × four directions |
| Episode horizon | 10 actions |
| Difficulty strata | Optimal lengths 1 and 2; exactly one robot used by the optimal plan |
| Solver label | BFS, depth limit 6, node limit 50,000 |
| Policy | SB3 `ActorCriticCnnPolicy`, 274,729 parameters |
| Training seeds | 11, 29, 47 |
| Checkpoints | 0, 10,240, 25,600, and 51,200 steps |

PPO used eight vector environments, 128 rollout steps per environment, batch size 256, four epochs, learning rate `3e-4`, `gamma=0.99`, `gae_lambda=0.95`, and entropy coefficient `0.01`.

Four thousand explicit generation seeds produced 460 unique, solver-labelled fixtures:

| Stratum | Puzzles |
|---|---:|
| Optimal length 1, robots moved 1 | 219 |
| Optimal length 2, robots moved 1 | 241 |

Stable layout-hash splitting produced 359 training, 51 validation, and 50 test puzzles. Fixed evaluation suites contain 16 puzzles from each solution-length stratum, or 32 puzzles per suite. Training and validation use one episode per puzzle at every checkpoint. The untouched test suite is evaluated only after training, with one deterministic and five stochastic episodes per puzzle. Baselines use 20 episodes per test puzzle, except the single optimal oracle replay.

- Fixture fingerprint: `29c1aeb066ab02d9e3d4279f9e3ec88a92d8fac111b1fd12db4cde481d366f28`
- Split fingerprint: `b72eb184db770c5d9e85875375cdc04b7316456006e7e3c3bbbc7509b1364e5d`

## Results

### Held-out baselines

| Controller | Overall success | 95% Wilson interval | Length 1 | Length 2 | No movement | Mean episode length |
|---|---:|---:|---:|---:|---:|---:|
| Random | 29.5% | 26.1%–33.2% | 36.9% | 22.2% | 34.3% | 8.22 |
| Target robot only | 46.7% | 42.9%–50.6% | 40.6% | 52.8% | 39.5% | 7.16 |
| Oracle | 100.0% | 89.3%–100.0% | 100.0% | 100.0% | 0.0% | 1.50 |

The unexpectedly strong target-only baseline is important: on this easy two-robot distribution, repeatedly moving the target robot is a substantially stronger control than uniform random actions.

### Tiny-set overfit milestone

One stock CNN was trained on eight training layouts, balanced across the two strata. The acceptance rule was at least 80% stochastic success at 51,200 steps.

| Steps | Deterministic | Stochastic |
|---:|---:|---:|
| 0 | 0.0% | 36.2% |
| 10,240 | 25.0% | 46.2% |
| 25,600 | 25.0% | 49.4% |
| 51,200 | 37.5% | 82.5% |

Result: **passed**. This establishes that the current observation, reward, action, PPO, and MPS paths can learn a deliberately small training set. It is not a generalization result.

### Multi-seed held-out PPO

| Seed | Train steps/s | Deterministic test | Stochastic test | Stochastic length 1 | Stochastic length 2 | Stochastic no movement |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 2,498 | 3.1% | 26.9% | 35.0% | 18.8% | 41.5% |
| 29 | 2,283 | 0.0% | 32.5% | 33.8% | 31.2% | 35.9% |
| 47 | 2,467 | 6.2% | 35.6% | 41.2% | 30.0% | 39.8% |

Across seeds, deterministic test success was **3.1% ± 3.1% sample SD**, with an episode-pooled 95% Wilson interval of 1.1%–8.8%. Stochastic test success was **31.7% ± 4.4% sample SD**, with an episode-pooled 95% Wilson interval of 27.7%–36.0%. The pooled interval describes episode-level uncertainty; the across-seed SD is the more relevant indication of training-run variation.

The held-out acceptance rule requires mean stochastic PPO success to at least match the target-robot-only baseline. Result: **failed**, 31.7% observed versus 46.7% required.

### Fixed training and validation curves

All values use the same 32-puzzle, one-episode-per-puzzle evaluation budget.

| Seed | Steps | Train deterministic | Train stochastic | Validation deterministic | Validation stochastic |
|---:|---:|---:|---:|---:|---:|
| 11 | 0 | 0.0% | 43.8% | 9.4% | 43.8% |
| 11 | 10,240 | 21.9% | 28.1% | 9.4% | 28.1% |
| 11 | 25,600 | 21.9% | 40.6% | 9.4% | 28.1% |
| 11 | 51,200 | 21.9% | 37.5% | 9.4% | 9.4% |
| 29 | 0 | 18.8% | 34.4% | 9.4% | 34.4% |
| 29 | 10,240 | 0.0% | 43.8% | 9.4% | 37.5% |
| 29 | 25,600 | 9.4% | 40.6% | 0.0% | 40.6% |
| 29 | 51,200 | 3.1% | 46.9% | 9.4% | 28.1% |
| 47 | 0 | 0.0% | 34.4% | 12.5% | 28.1% |
| 47 | 10,240 | 9.4% | 40.6% | 0.0% | 40.6% |
| 47 | 25,600 | 9.4% | 34.4% | 0.0% | 21.9% |
| 47 | 51,200 | 9.4% | 28.1% | 0.0% | 37.5% |

There is no consistent improvement on either the fixed training probe or validation suite. This agrees with the final test result and argues against selecting one seed or checkpoint as a success.

### Action distribution

Actions 0–3 move robot 0 and actions 4–7 move robot 1. Values below are percentages of all executed actions.

| Controller | a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 12.4% | 12.7% | 12.0% | 12.7% | 12.5% | 12.5% | 12.0% | 13.3% |
| Target robot only | 9.5% | 10.5% | 10.0% | 9.9% | 14.8% | 15.6% | 14.6% | 15.2% |
| PPO stochastic, pooled | 20.5% | 10.5% | 18.9% | 16.7% | 6.2% | 13.1% | 7.8% | 6.2% |
| PPO deterministic, pooled | 32.4% | 0.0% | 33.3% | 34.3% | 0.0% | 0.0% | 0.0% | 0.0% |

The deterministic policies selected only three robot-0 actions and produced no movement on 91.0% of steps. Stochastic policies retained all actions but were strongly skewed toward robot 0 and produced no movement on 39.1% of steps. Mean stochastic episode length was 8.19 actions; successful stochastic episodes exceeded the oracle length by 2.87 actions on average.

## Interpretation and next gate

Phase 2 has re-established a trustworthy measurement protocol, but it has not produced a successful held-out agent. The overfit pass rules out an obvious device or end-to-end wiring failure. The broad curves and test metrics show that 51,200-step stock-CNN PPO does not reliably extract a useful policy from the 359-layout training distribution, and deterministic action selection collapses badly.

The next experiment should preserve this exact benchmark and change one factor at a time. The most useful immediate controls are a longer-budget stock-CNN run, a length-1-only training distribution before adding length 2, and a comparison against invalid/no-movement action masking or solver-supervised first-action pretraining. Recurrent/DRC work should remain out of scope until a feed-forward control clears the held-out gate.

## Reproduction

From the repository root:

```bash
uv venv --python 3.11 .venv-phase2
uv pip install --python .venv-phase2/bin/python -r requirements-ci.txt
uv pip install --python .venv-phase2/bin/python -e .
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py
```

Generated banks, models, exact episode records, manifests, and the machine-generated report are written beneath `artifacts/phase2_bank/` and `artifacts/phase2_benchmark/`. These outputs are intentionally ignored by Git. Reusing the same bank/config validates the fixture definition and reproduces the split fingerprint; changing the fixture definition requires a new bank directory.
