# Structured Helper-Free DRC: 60M Review

Date: 2026-07-24  
Run: `groundup-drc33-p1-12x12-planefree`  
Seed: 11  
Decision: **extend the same run to 120M steps**

## Executive conclusion

The 60M structured-input DRC experiment is a clear task-learning success and
a clear negative result for the registered linear landing-cell hypothesis so
far.

The DRC reached 76.5% deterministic success on the fixed 200-puzzle periodic
validation suite near 60M, compared with 66.5% for P1 at the matched budget.
The validation-selected 56M DRC checkpoint achieved 79.6% deterministic and
78.2% stochastic success when re-evaluated on the larger fixed 480-puzzle
validation suite. Gains are not confined to easy target-only puzzles:
deterministic success was 59.6% on two-robot puzzles and 55.0% on the small
three-robot stratum.

However, the fixed 16,000-state zero-state probes do not show a learned
linearly decodable mid- or long-slide landing-cell representation. Nearly
every trained-minus-random comparison is negative. Tick 1 briefly matches
random initialization around 21-23M, but the difference is only 0.04
percentage points and is not sustained. By 60M, mid/long accuracy is below
random control at the encoder, every recurrent tick, and the readout map.

The most defensible interpretation is:

- repeated recurrent computation substantially improves policy learning on
  this helper-free task;
- that improvement is not explained by the registered local linear
  landing-cell representation; and
- the policy may instead use a nonlinear, distributed, action-relevant, or
  temporally carried representation that this zero-state probe does not
  capture.

No sealed-test result was inspected. Under the preregistered allocation rules,
the run must extend to at least 120M because it exceeds P1 at matched budgets
and continued to improve late in the sweep.

## Experimental integrity

- Actual stop: 60,006,400 environment steps.
- Learning-rate schedule horizon: 200M; it was not exhausted or restarted.
- Selected checkpoint: 56,000,512 steps, using periodic validation only.
- Observation: 9 structured channels (walls, goal, three robots, time).
- Stop-cell and distance helper planes: absent.
- Validation periodic suite: fixed 200 puzzles, 25 per optimal length.
- Validation final suite: fixed 480 puzzles, 60 per optimal length.
- Probe dataset: fixed 16,000 validation states.
- Probe dataset fingerprint:
  `e97ef31a7e5c0bf133ee508b4ea8bbb27bf32a6a9a58893d9f41d1df5c48adfc`.
- Probe checkpoints: 57, covering 4,005,888 through 60,006,400 steps.
- Sealed test: not evaluated or used for any decision in this review.

The trainer, snapshot watcher, and probe worker exited normally after their
targets were complete. Immutable model and exact-resume checkpoints exist at
every registered 5M boundary, including 60,006,400.

## Task-learning result

Matched periodic validation:

| Budget | DRC checkpoint | DRC | P1 checkpoint | P1 | DRC - P1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5M | 4.006M | 45.5% | 4.002M | 35.0% | +10.5 pp |
| 10M | 9.003M | 61.0% | 9.003M | 48.0% | +13.0 pp |
| 20M | 19.005M | 65.5% | 19.001M | 57.5% | +8.0 pp |
| 30M | 29.008M | 67.0% | 29.004M | 62.5% | +4.5 pp |
| 40M | 39.002M | 71.5% | 39.002M | 61.5% | +10.0 pp |
| 50M | 49.005M | 74.5% | 49.000M | 65.0% | +9.5 pp |
| 60M | 59.007M | 76.5% | 59.003M | 66.5% | +10.0 pp |

The periodic DRC best of 76.5% occurred at 56.001M, 57.008M, and
59.007M. The final 60.006M reading was 72.5%; the keep-best checkpoint
therefore protected against the noisy final dip.

### Larger validation re-evaluation of the selected checkpoint

| Evaluation | Success | Episodes |
| --- | ---: | ---: |
| Deterministic | 79.58% | 480 |
| Stochastic, three episodes per puzzle | 78.19% | 1,440 |
| Random action-masked baseline | 14.08% | 2,400 |
| Target-only action-masked baseline | 24.29% | 2,400 |

Indicative Wilson intervals are 75.8-83.0% for deterministic and 76.0-80.3%
for stochastic success. The fixed suite is a controlled benchmark rather
than an IID population sample, so these intervals describe evaluation noise,
not full experimental uncertainty.

Deterministic validation by optimal length:

| L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100.0% | 93.3% | 93.3% | 90.0% | 68.3% | 76.7% | 68.3% | 46.7% |

Deterministic validation by robots moved:

| One robot | Two robots | Three robots |
| ---: | ---: | ---: |
| 93.97% (265/282) | 59.55% (106/178) | 55.0% (11/20) |

The three-robot estimate is encouraging but imprecise because it contains
only 20 puzzles. The two-robot result is the stronger evidence that the gain
extends beyond target-only behavior.

## Probe result

The random-initialization control and every checkpoint use the same probe
dataset and train/test split. The table below shows the best trained
mid/long result anywhere in the sweep and the final 60M result.

| Representation | Bucket | Random control | Best trained | Best gain | 60M trained | 60M gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Encoder | Mid | 4.68% | 2.17% | -2.51 pp | 0.93% | -3.75 pp |
| Encoder | Long | 6.23% | 2.39% | -3.84 pp | 0.52% | -5.71 pp |
| Tick 1 | Mid | 3.60% | 3.64% | +0.04 pp | 2.57% | -1.03 pp |
| Tick 1 | Long | 6.66% | 6.70% | +0.04 pp | 3.60% | -3.06 pp |
| Tick 2 | Mid | 3.50% | 2.91% | -0.59 pp | 1.88% | -1.62 pp |
| Tick 2 | Long | 7.16% | 4.56% | -2.60 pp | 1.67% | -5.49 pp |
| Tick 3 | Mid | 3.34% | 2.77% | -0.57 pp | 1.66% | -1.68 pp |
| Tick 3 | Long | 6.86% | 4.26% | -2.60 pp | 1.67% | -5.19 pp |
| Readout | Mid | 4.25% | 2.06% | -2.19 pp | 0.83% | -3.42 pp |
| Readout | Long | 6.68% | 2.67% | -4.01 pp | 0.38% | -6.30 pp |

The +0.04-point tick-1 maxima are quantitatively negligible and appear only
at isolated checkpoints. They do not satisfy the registered requirement for
a sustained gain.

Overall probe accuracy does increase substantially with training, but this is
driven by blocked and short slides. At 60M, for example, encoder overall
accuracy is 32.9% versus 10.8% at random initialization, while encoder
mid/long accuracy is markedly worse than random. Overall accuracy must
therefore not be presented as evidence for learned long-range slide
dynamics.

Within the trained model, tick 1 remains more informative about long slides
than the trained encoder. Because the same pattern is stronger at random
initialization and shrinks during training, it is an architectural property,
not evidence that RL learned the registered endpoint circuit.

## Optimizer, curriculum, and operations

- No non-finite telemetry occurred.
- No target-KL early stop occurred.
- Mean approximate KL: median 0.0040, maximum 0.0150.
- Maximum minibatch KL: maximum 0.0281, below the 0.05 guard.
- Explained variance: median 0.771, final 0.821.
- Entropy declined from 0.850 at 1M to 0.297 at 60M without collapsing to
  zero.
- The reverse curriculum reached frontier 7 at 59M and ended at frontier 6,
  with substantial sampling and success across all lengths.

There were two clamshell-sleep pauses, after approximately 6M and 12M. The
same live process resumed in memory each time; schedules and environment
steps did not advance while asleep. These pauses distort aggregate
wall-clock throughput but do not alter the scientific step-budget
comparison. No checkpoint reload or optimizer restart occurred.

## Persistence

Local canonical directory:

`artifacts/groundup/drc33_p1_12x12_planefree/`

Key local files:

- `drc_probe_series.csv`: flat 57-checkpoint trajectory;
- `drc_probe_manifest.json`: SHA-256 manifest over the dataset, control,
  validation context, and every checkpoint JSON;
- `probe_dataset.npz`: fixed 16,000-state dataset;
- `probe_control.json`: matched random-initialization control;
- `probe_t<steps>.json`: one result per checkpoint;
- `drc_probe_summary.md` and `drc_probe_curve.png`: human-readable outputs.

Remote persistence:

- [W&B training run with `probe/*` history, table, and chart](https://wandb.ai/luke-sid-baines-blank/ricochet-robots-rl/runs/pupnnpq8)
- [W&B probe artifact `v0`, aliases `60m` and `latest`](https://wandb.ai/luke-sid-baines-blank/ricochet-robots-rl/artifacts/probe-results/drc33_p1_12x12_planefree-probes/v0)

W&B API verification found 57 probe history rows and 66 artifact payload
files. Artifact identity:

`luke-sid-baines-blank/ricochet-robots-rl/drc33_p1_12x12_planefree-probes:v0`

## Extension decision

The preregistered 60M rules say to extend to at least 120M if any of the
following holds:

- DRC matches or exceeds P1 at comparable steps;
- validation or hard-stratum performance is still materially improving;
- probes show sustained growth; or
- the result remains compatible with delayed learning.

The first two conditions clearly hold. The negative probe trajectory does
not override them. The same run will therefore resume exactly from the
immutable 60,006,400-step training checkpoint, retain the 200M schedule
horizon, preserve the validation-only selection discipline, and continue to
120M. The sealed test remains unopened until the final stop decision.

