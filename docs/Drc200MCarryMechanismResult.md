# DRC 200M Closeout and Carry-Mechanism Result

Date: 2026-08-13
Status: priorities 0, 1, and 2 complete; single-seed exploratory result

## Executive conclusion

The repaired reporting contract, exact 200M continuation, one test evaluation
frozen for this DRC line, and the observational and causal carry studies are
complete.

The final 200M DRC policy is the strongest honest result on the 12x12
helper-free task. On the fixed 480-puzzle DRC-held-out test suite it reached **80.42%
deterministic / 81.25% stochastic success**, compared with the frozen
feed-forward P1 artifact's **69.17% / 68.33%**. The gains are **+11.25 and
+12.92 percentage points**, and grow along the hard tail: deterministic gains
are +15 points at length 5, +18.33 at length 7, +30 at length 8, and +29.42 on
the small 17-puzzle three-robot stratum.

“Held out” and “sealed” are scoped to this DRC line: the suite was excluded
from DRC training and selection, the checkpoint rule was written first, and
only the 200M final DRC checkpoint was evaluated. It is **not** a globally
untouched confirmation set for the overall programme. P1 had already been
evaluated on what is strongly evidenced to be the same selection, and its
aggregate difficulty results informed the subsequent DRC research plan. The
legacy P1 JSON predates suite fingerprints, so identity is reconstructed from
exactly matching scripted-baseline aggregates rather than hash-proven. The
comparison is therefore a fixed-selection recipe comparison, not a fresh
external confirmation.

The recurrent carry is causally important, but it does not look like a single
explicit stored route. Continuously zeroing the full carry reduced success by
32--38 points at four checkpoints from 9M through 200M; at 200M the effect was
**-38.22 points** (paired 95% bootstrap interval **[-46.50, -29.94]**). Both
`h` and `c` and especially the deeper recurrent depths contributed. In
contrast, a single reset at move 1, 2, 4, or 8 changed mean success by only
-2.55, -2.55, +1.27, and -0.64 points. This resilience to one-time zeroing is
consistent with rapid reconstruction, redundancy, or local bypass; the present
study does not distinguish them. Removing carry on every decision is
devastating.

Exact-state donor patching gives the complementary result. A coherent donor
`h+c` carry changed the action sequence in 39/68 eligible final-checkpoint
contexts, proving that history-dependent carry can change decisions, but it
had zero mean suffix-success effect (95% interval [-7.35, +7.35]). Among these
eligible reconvergences, no directional mean effect was detected at this
precision.

The probe study agrees with that causal picture. Carry strongly encodes action
and recent-history information, and learned carry contains nonlinear
solution-adjacent information about distance and optimal/reducing actions. It
does **not** expose a clean shortest-path-progress variable or accurate slide
simulator: final landing-cell accuracy is only about 10--13%, distance R2 is
modest, and several apparent targets are matched by the random-network
control. The best current description is a **distributed, history-dependent
dynamical policy state that is resilient to an isolated zero intervention**,
not a readable symbolic plan. Rapid reconstruction is one hypothesis for that
resilience, not an established mechanism.

These are strong single-seed results. They do not isolate recurrence from the
different trainer and 200M-vs-181M compute budgets, and the cross-checkpoint
probe samples come from different on-policy state distributions.

## 1. Reporting repair and checkpoint rule

Priority 0 was completed before closeout. The trainer now:

- evaluates the run's own final weights for the headline;
- reports validation-selected periodic-best weights separately;
- records the originating run, timestep, immutable checkpoint path, kind, and
  SHA-256 for every score;
- verifies inherited best weights and rejects an undeclared or same-directory
  semantic branch before output mutation;
- evaluates a deterministic hard suite sampled jointly by
  `(optimal_length, robots_moved)` and publishes the joint cells; and
- uses content-addressed evaluation checkpoints, so a later continuation
  cannot invalidate an earlier result's path/SHA pair.

The 190M resume had no semantic contract differences and legitimately
reconstructed the inherited 135,004,160-step periodic best from immutable
`model_135004160.pt`. The final-only rule was frozen before the 200M report and
before evaluating the test split for this DRC line. The 135M periodic best was
never evaluated on that split.

Relevant implementation commits are `9ccf245` and `84d91f0`. The complete
source and study implementations were frozen at `e3f91c6` before the formal
mechanistic sweep.

## 2. Exact continuation to 200M

The continuation resumed exactly from
`training_pause_190005248.pt` and stopped at the next legal rollout boundary,
**200,007,680 environment steps**. The primary immutable source is:

- run: `drc33_p1_aux_A0_control`
- kind: `model_final`
- checkpoint: `model_eval_200007680_8760ba70...pt`
- SHA-256: `8760ba70d07abaa6c21ba29fb5810a8b40c47d542444f8547a8de82a7e933363`
- report: `report_200007680.json`
- report SHA-256: `d3002974e56ba24eb5b1fd7148b9bda34a9ce083fff2ef13131c6030bb2f5ef8`

The final weights also beat the max-selected 135M periodic best on both fixed,
separately seeded validation suites:

| Frozen validation evaluation | 200M final | 135M periodic best | Final minus best |
| --- | ---: | ---: | ---: |
| 480-puzzle deterministic | **82.08%** | 79.79% | **+2.29 pp** |
| 480 puzzles x 3 stochastic episodes | **82.29%** | 80.83% | **+1.46 pp** |
| 470-puzzle joint hard deterministic | **75.96%** | 74.89% | **+1.07 pp** |

The 190M--200M periodic window is a healthy plateau rather than a collapse:
11 deterministic measurements averaged 79.68%, ranged from 78.5% to 82.5%,
and ended at 80.0%. The final learning rate reached its registered 0.00005
floor; final value loss was 0.0261, explained variance 0.783, entropy 0.065,
and no early-stop alarm fired.

## 3. One test evaluation frozen for the DRC line

The decision file was written before the final report existed. It named the
200M final checkpoint as the sole primary, fixed test split, suite seed 17,
60 puzzles per length, one deterministic and three stochastic episodes, and
forbade evaluating the 135M periodic best. Its SHA-256 is
`19c459630dd90a84a87faefeaf10f67a3b8260d2c362b70cfcda82a3aa517e1d`.

The DRC test evaluator then ran once. It verified the report, immutable source
SHA, timestep/run/kind, clean tracked worktree, MPS device, suite counts,
provenance, atomic no-overwrite publication, and a self-hashed scientific
payload.

- suite fingerprint:
  `9cf385d281b46e4d4e20959a0b0ed2d3de13247dbfaf2ad005ccbf86c747701b`
- result file: `sealed_test_final200.json`
- result file SHA-256:
  `3858a1168b0ca6b08ec9fd9b90ab0b9d18891a3b79607df6678425ec9194d9a5`
- scientific payload SHA-256:
  `85d351c20071a9e0669d43ab4e99e6bbc93a1ee2765353dbb11d0513bd3fa83e`

### Overall comparison

| Policy | Deterministic | Stochastic |
| --- | ---: | ---: |
| Scripted random | -- | 14.25% |
| Scripted target-only | -- | 25.96% |
| Frozen feed-forward P1, 112M | 69.17% | 68.33% |
| **DRC final, 200M** | **80.42%** | **81.25%** |
| **DRC - P1** | **+11.25 pp** | **+12.92 pp** |

This was the DRC line's first and only evaluation of the split, not the
programme's first use of it: the frozen P1 result and its aggregate strata were
already known. The P1 result predates the new suite-fingerprint field. Its JSON
and model are hash-pinned, and reconstructing the current suite reproduced its
random and target-only aggregates exactly. This is strong evidence that the
selection is the same, while preserving the historical-provenance caveat.

### Deterministic difficulty profile

| Optimal length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DRC 200M | 98.33% | 96.67% | 93.33% | 93.33% | 80.00% | 65.00% | 65.00% | 51.67% |
| P1 112M | 100.00% | 91.67% | 90.00% | 81.67% | 65.00% | 56.67% | 46.67% | 21.67% |
| Difference | -1.67 | +5.00 | +3.33 | +11.67 | +15.00 | +8.33 | +18.33 | +30.00 pp |

| Robots moved | 1 (`n=295`) | 2 (`n=168`) | 3 (`n=17`) |
| --- | ---: | ---: | ---: |
| DRC 200M | 90.85% | 66.07% | 41.18% |
| P1 112M | 84.07% | 48.81% | 11.76% |
| Difference | +6.78 | +17.26 | +29.42 pp |

The largest gains occur where a recurrent mechanism should be most useful:
long and multi-robot solutions. The three-robot comparison is directionally
strong but has only 17 puzzles. Because this is one seed and the DRC received
more environment steps under a different trainer, the result establishes a
better complete DRC recipe, not an architecture-only causal effect.

## 4. Observational carry probes

The version-2 probe protocol captured actual `h` and `c` carry during one
deterministic rollout per puzzle. It used a 157-puzzle validation suite spanning
all 20 observed joint difficulty cells, fingerprint
`5410a2bee4096e73594c2368a1e43c9a697c95a4da73c06c33b09bc4b42da9a7`.
Whole layouts were split deterministically into 118 train and 39 test layouts
with zero leakage (split fingerprint `235624ed...`). Exact reverse-BFS labels
and reducing-action masks had 100% coverage. Every target used an independent
linear or one-hidden-layer MLP probe, with an architecture-matched random DRC
on the exact trained observation stream and a shuffled-training-label control.

The four frozen checkpoints were 9,003,008 (`454eac...`), 59,006,976
(`a76aed...`), 135,004,160 (`d8cb50...`), and 200,007,680 (`8760ba...`). Their
trajectory datasets contained 2,671, 2,120, 2,048, and 2,012 rows respectively;
the row-count change is itself a consequence of different policy trajectories.

The table below shows representative `h_final` MLP test metrics as
`trained / random-network / shuffled-label`. It is a compact view, not a
multiple-comparison significance test.

| Target / metric | 9M | 59M | 135M | 200M |
| --- | --- | --- | --- | --- |
| Intended action accuracy | .691 / .327 / .179 | .570 / .217 / .177 | .565 / .185 / .175 | .633 / .210 / .177 |
| Previous action accuracy | .857 / .352 / .184 | .658 / .247 / .138 | .622 / .237 / .185 | .667 / .260 / .128 |
| Exact distance R2 | -.324 / -.691 / -.624 | .133 / -.544 / -.750 | .325 / -.904 / -.406 | .229 / -1.534 / -.354 |
| Optimal-action top-1 in set | .294 / .251 / .220 | .358 / .229 / .176 | .388 / .203 / .155 | .391 / .235 / .159 |
| Reducing-action top-1 in set | .295 / .240 / .182 | .355 / .203 / .188 | .414 / .183 / .171 | .427 / .210 / .246 |
| Future-success ROC-AUC | .572 / .736 / .559 | .750 / .567 / .487 | .776 / .608 / .455 | .548 / .405 / .572 |
| Landing-cell accuracy | .141 / .044 / .017 | .099 / .016 / .007 | .097 / .030 / .020 | .098 / .018 / .025 |
| Immediate-success ROC-AUC | .728 / .525 / .607 | .947 / .538 / .272 | .960 / .528 / .309 | .956 / .470 / .472 |

At 200M, looking across all registered carry representations rather than only
`h_final`:

- exact-distance R2 peaks at .229 (`h_final` MLP), while both matched controls
  are below zero;
- optimal-action top-1 peaks at .418 (`h_c_final` MLP), versus roughly
  .22--.24 for the random network;
- reducing-action top-1 peaks at .427 (`h_final` MLP);
- landing-cell accuracy peaks around .13 (`h_c_final`), well above its matched
  controls and 1/144 chance but still too weak to be an explicit slide simulator;
- future-success decoding no longer beats shuffled labels at 200M, so there is
  no defensible monotonic-development story for that target; and
- horizon progress is decoded by both trained and random networks, showing the
  architectural time channel rather than learned planning knowledge.

The learned carry therefore has robust action/history code and modest
nonlinear access to solution-adjacent variables. It does not have a clean,
linearly readable route or shortest-path-progress scalar. Cross-checkpoint
changes must not be read as a controlled learning trajectory because each
checkpoint induces a different on-policy state distribution; trained-vs-control
comparisons within a checkpoint are the cleaner evidence.

Probe result fingerprints are `952dff97...` (9M), `46670b77...` (59M),
`d245a4a9...` (135M), and `37ef57e0...` (200M). The final JSON file SHA-256 is
`014e347e1c3696661ab0a14a84be57c8e3346956654cf4b7318dda6407f01ceb`.

## 5. Causal carry interventions

The intervention study used the same 157 puzzles. The primary full-reset arm
zeroed all depths of `h` and `c` before every post-initial decision. All
effects are within-puzzle differences in deterministic success.

| Checkpoint | Untreated success | Full reset effect | Paired 95% interval |
| --- | ---: | ---: | ---: |
| 9M | 49.68% | **-32.48 pp** | [-40.13, -24.84] |
| 59M | 70.06% | **-33.12 pp** | [-40.76, -25.48] |
| 135M | 74.52% | **-31.85 pp** | [-39.49, -24.20] |
| 200M | 77.71% | **-38.22 pp** | [-46.50, -29.94] |

This is the central causal result: useful recurrent state is necessary at all
four sampled checkpoints spanning 9M--200M. It is not merely an incidental
late correlate.

### Final-checkpoint localization

| Intervention | Paired success effect |
| --- | ---: |
| Reset depth 0 every move | -3.18 pp |
| Reset depth 1 every move | -10.19 pp |
| Reset depth 2 every move | -16.56 pp |
| Reset all `h` every move | -28.66 pp |
| Reset all `c` every move | -21.02 pp |
| One reset after move 1 | -2.55 pp |
| One reset after move 2 | -2.55 pp |
| One reset after move 4 | +1.27 pp |
| One reset after move 8 | -0.64 pp |

The depth/component effects are nonadditive, as expected for an off-manifold
ablation in a coupled nonlinear system. Their intervals and every subgroup
interval are pointwise, multiplicity-unadjusted, descriptive localization;
they must not be promoted to independent confirmatory claims. The robust
pattern is that deeper carry and both state components matter under continuous
removal, while one-time zeroing has only small mean success effects through
mechanisms this study does not distinguish.

At 200M, the full-reset effect was 0 points at length 1 and ranged from -12.5
to -58.3 points over lengths 2--8. By robots moved it was -29.7, -46.4, and
-40.5 points for one-, two-, and three-robot strata. These are descriptive
subgroups, but they support the performance result's hard-tail interpretation.

### Exact-state patching

Sixteen stochastic source rollouts per puzzle produced 68 eligible exact
reconvergences at 200M: same complete environment snapshot, RNG/time state,
byte-identical observation, and canonical positions, but different action
histories. Every suffix ran in a fresh environment. The independently replayed
donor-natural suffix matched the full `h+c` transplant for all eligible
contexts, which is the required positive control.

| Patch into recipient | Action sequence diverged | Suffix-success effect | 95% interval | Role |
| --- | ---: | ---: | ---: | --- |
| Full donor `h+c` | 39/68 | **0.00 pp** | [-7.35, +7.35] | preregistered primary |
| Donor `h` only | 30/68 | -1.47 pp | [-7.35, +4.41] | descriptive |
| Donor `c` only | 28/68 | -8.82 pp | [-16.18, -2.94] | descriptive |

The coherent full-carry result separates **decision dependence** from
**directional competence**: carry history changes behavior in over half the
eligible cases, but no directional mean success effect was detected at this
precision among these arbitrary donor histories. The final `c`-only effect is
a useful hypothesis about `h/c`
co-adaptation or destructive mismatch, not evidence that `c` uniquely stores a
named variable; it is a descriptive member of a large localization family.

All four causal JSONs passed the donor control. Their file SHA-256 prefixes are
`0b0bfab7...`, `de650a13...`, `eac83be9...`, and `30cb932e...`.

## 6. What is established, and what is not

Established for this seed and protocol:

1. The final 200M policy generalizes substantially better than the frozen P1
   comparator on the fixed DRC-held-out suite, especially on long/multi-robot
   puzzles; this suite was previously used for P1 and is not globally untouched.
2. Continuously zeroing carry lowers absolute success by 32--38 percentage
   points across the observed learning trajectory; this is causal, not merely
   decodable correlation.
3. Sensitivity is distributed across depths and `h/c`; one-time zeroing has
   small mean effects, consistent with reconstruction, redundancy, or local
   bypass but not distinguishing those mechanisms.
4. Carry contains learned action/history and modest solution-adjacent
   information, but the registered probes do not reveal a simple explicit
   shortest path or slide simulator.
5. Different matched histories at the exact same observable state frequently
   change actions, demonstrating genuine history-dependent computation.

Not established:

- an architecture-only DRC advantage over feed-forward P1;
- a multi-seed population effect;
- that any one decoded variable mediates the carry's causal performance gain;
- a monotonic developmental trajectory for every probe target;
- simultaneous significance for the many descriptive localization/subgroup
  intervals; or
- that the descriptive `c`-only patch result identifies a unique memory role.

## 7. Recommended next programme

### 1. Replicate the performance result before spending another 200M on novelty

Run two additional DRC seeds with the repaired final-only contract, or first
use a smaller registered stopping budget if validation/hard curves show that
200M is unnecessary. The known seed-17 test suite may serve as a fixed
replication endpoint held out from each seed's training and selection, but it
must not be described as globally unopened. Pre-register and reserve a new
layout-disjoint, globally untouched suite or bank for the eventual
programme-level confirmation. Do not continue seed 11 beyond 200M by default:
its schedule is at the floor and its last 10M are a stable plateau.

### 2. Measure carry reconstruction directly

The largest mechanistic contrast is continuous reset versus tiny one-shot
effects. Pre-register a recovery-time experiment that resets full carry once
and measures, over the next 1--6 decisions, carry similarity, logit recovery,
action divergence, optimal-action probability, and success. Include repeated
reset spacings (every 2/4/8 decisions). This directly discriminates rapid
reconstruction from redundancy/local bypass and should be more informative
than another broad probe sweep.

### 3. Make patching target a falsifiable difference

No directional arbitrary-donor full-carry success effect was detected despite
large action divergence. A next patch study should select exact reconvergences
without outcomes and use prefix-only properties or a frozen preregistered probe
score for stratification when next-action/logit change is the proximal outcome.
A separate cohort may be enriched for donor/recipient natural-action
disagreement, but there suffix success must be the estimand and action
divergence only a selection/manipulation check: full-`h+c` action change is
guaranteed by that selection and donor-control construction. This can test
whether a specific decoded feature mediates behavior without making the
proximal endpoint tautological.

### 4. Test component interactions on-manifold

The final `c`-only patch asymmetry suggests co-adaptation. Use a factorial
depth-by-component patch design with coherent per-depth donor pairs and
explicit interaction contrasts. Prefer on-manifold donor replacement to zero
ablation, and correct or hierarchically control the planned comparison family.

### 5. Separate recurrence from training-budget effects

For an architectural claim, train matched feed-forward and DRC arms under the
same reward semantics, environment-step budget, selection rule, suites, and
seeds. The present 200M-vs-181M comparison is a credible recipe comparison but
cannot assign the gain solely to recurrence.

### 6. RGB entry gate: met, but begin with a short gate

The structured DRC is healthy, task-positive, and mechanistically understood
well enough to satisfy the registered RGB entry gate. Freeze the remaining-time
signal choice before looking at RGB learning, run renderer/correctness and
real-factory throughput gates, then a short learnability study. A full 200M RGB
run should depend on that gate rather than be the immediate next job.

## 8. Artifact index

All raw artifacts are gitignored but preserved locally:

- closeout report and DRC-line test evaluation:
  `artifacts/groundup/drc33_p1_aux_A0_control/`
- probe datasets/results and causal results:
  `artifacts/groundup/drc_carry_study_v2/`
- observational protocol: `docs/DrcCarryStateStudy.md`
- causal protocol: `docs/DrcCarryInterventionStudy.md`
- original programme and append-only execution history:
  `docs/DrcPlaneFreeResearchPlan.md`

The tracked worktree was clean when each frozen evaluation/study started. The
untracked local `AGENTS.md` is user-owned and was not included or modified.
