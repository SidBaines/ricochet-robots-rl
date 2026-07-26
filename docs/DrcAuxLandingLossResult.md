# Auxiliary landing-cell loss: result of the mid-training screen

Executed overnight 2026-07-26 against the plan in
`docs/DrcAuxLandingLossPlan.md`, which targeted the negative probe finding of
`docs/DrcPlaneFree120MReview.md` §5.

**Conclusion: the bottleneck hypothesis is falsified.** The landing-cell
knowledge that 120M steps of RL failed to build can be installed by supervision —
easily, to a very large degree, and at essentially no cost to the policy — and
performance does not improve. The 20 h from-scratch arm is **not** justified.

---

## 1. Arms

All three resume the identical v2 checkpoint `training_120004608.pt` and differ
**only** in `aux_landing_coef`, so every comparison is paired.

| arm | λ | ran to | note |
|---|---|---|---|
| **A0 control** | 0 | 135,004,160 (complete) | isolates "more training" |
| **A1** | 1.0 | 124,002,304 (stopped early) | too strong; stopped once its answer was clear |
| **A2** | 0.3 | 135,004,160 (complete) | the informative arm |

λ chosen by a two-stage sweep. Stage 1 (gradient decomposition) gave an
independent confirmation of the probe result before any training: the freshly
initialised head reads the trunk at **5.053** against chance ln(144) = **4.969**,
i.e. exactly chance. Policy share of the trunk gradient by λ: 93.3% (0.1),
92.4% (0.3), 87.2% (1.0), 65.4% (3.0), 27.8% (10) — against 98.8% at the healthy
80M reference and 3.7% at the dying 87M one.

## 2. The knowledge is installable, and cheaply

Landing-cell decodability, readout layer, gain over a random-init control, on the
fixed 16,000-state probe set (hash `e97ef31a…`, identical to the one used for
v2, the abandoned run and P1):

| arm | t | blocked | short (1–3) | **mid (4–7)** | **long (8+)** |
|---|---|---|---|---|---|
| A0 control | 135,004,160 | +0.588 | +0.178 | **−0.032** | **−0.062** |
| A1 λ=1.0 | 124,002,304 | +0.785 | +0.858 | +0.365 | +0.031 |
| **A2 λ=0.3** | **135,004,160** | **+0.799** | **+0.904** | **+0.555** | **+0.066** |

- A2 moves mid-band decodability by **+0.588** relative to control, from *below*
  a randomly initialised network to far above it. Raw accuracy climbs
  monotonically 0.028 → ~0.60 over 15M steps. Aux cross-entropy falls
  **4.969 (chance) → 0.395**.
- **The A0 control is flat to ±0.001 across 15 evaluations and 15M steps.** This
  is what makes the result trustworthy: without the aux loss nothing moves, so
  the 120M probe finding is a stable property of the training process and not
  drift, and every change above is attributable to the aux term.

## 3. And it does not help

Two independent measurements, which agree.

**Endpoint, 480-puzzle final suite, each arm's own final weights at
t=135,004,160:**

| | `val_det` | vs control |
|---|---|---|
| v2 parent @120M | 0.7708 | — |
| **A0 control @135M** | **0.7979** | — |
| **A2 λ=0.3 @135M** | **0.7875** | **−0.0104** (0.56 SE) |

**Matched-step, 200-puzzle periodic suite, 15 paired evaluations:**
mean difference **−0.0220** (SD 0.0241, SE 0.0062).

So A2 is somewhere between neutral and ~2pp worse. It is certainly not better.
By length there is no systematic long-puzzle gain — the hypothesis' central
prediction: A2 beats A0 at lengths 5 (0.800 vs 0.767) and 8 (0.500 vs 0.467) but
loses at 3 (0.867 vs 0.900) and 7 (0.633 vs 0.717).

This is branch 2 of the pre-registered decision rule: **decodability up,
`val_det` flat → the knowledge is installable but is not what gates
performance.**

## 4. The control was load-bearing

**A0 gained +2.7pp from plain training alone** (0.7708 → 0.7979 on the 480 suite;
best periodic `val_det` 0.81 → **0.84**). The 120M run genuinely had not
converged, exactly as the review warned.

Had the aux arm been run without a control and compared against the 120M
baseline, A2's 0.7875 would have looked like the aux loss **helping by +1.7pp**.
It is in fact 1pp *behind* what plain training achieves. The plan called this out
as "the single easiest way to get a false positive here", and it was right.

## 5. Dose response: the cost is gradient strength, not knowledge

A1 (λ=1.0) and A2 (λ=0.3) both installed the knowledge, but paid very differently:

| λ | aux loss reached | mid gain | `val_det` vs control |
|---|---|---|---|
| 0.3 | 0.395 | +0.555 | −0.01 to −0.02 |
| 1.0 | 0.713 | +0.365 | **−0.150** |

A2 ended with *more* knowledge installed than A1 and paid ~1/12 the price. So the
15pp collapse at λ=1.0 is not the price of the knowledge — it is a transient
optimisation cost of pushing a large auxiliary gradient through a shared trunk.
Install it gently and it is nearly free. It is also nearly useless.

## 6. What this does and does not license

**Does:** drop the from-scratch aux arm. The mechanism it was meant to test —
that missing landing-cell knowledge gates performance beyond ~77% — is not
supported. Supervising it to a decodability level six-tenths above a random
network changes nothing about how well the agent plays.

**Does not:** conclude that the DRC lacks planning-relevant computation. §4 of the
120M review is still true and still unexplained: this DRC beats the feed-forward
P1 baseline by **+30pp at length 8** while having *worse* linearly-decodable
long-slide knowledge than a random network. Whatever recurrence buys at long
solution lengths, it is not a linearly-readable landing-cell map — and this
experiment now shows that supplying such a map does not buy it either.

**Retires:** linear landing-cell decodability as a progress metric. It can be
driven to near-ceiling without touching task performance, so it does not measure
what we care about.

## 7. Caveats

1. `long` (8+) only reached gain **+0.066**. Mid-band is decisively tested;
   long-band is only weakly installed, so a hypothesis specifically about
   *long*-range ray tracing survives more of this than the general one does.
2. Single seed per arm.
3. A1 had no head warmup — its randomly initialised head injected a large
   gradient into the trunk from step 0. That confound applies only to A1's 15pp
   drop, which §5 explains as an optimisation transient; A2's near-zero cost
   makes it moot for the conclusion.
4. 15M steps of aux training. Longer might change the picture, but decodability
   was already near ceiling at the endpoint and still rising, while `val_det`
   was not.

## 8. Trap found in the harness (worth fixing)

**A resumed branch that never beats its parent's `best_val_det` silently reports
the *parent's* weights as its own final result.** `best_state_dict` /
`best_val_det` are inherited through `run_state`, and `report_eval_weights=best`
then evaluates those inherited weights. A2 never exceeded v2's inherited 0.81, so
it wrote **no `model_best.pt` at all** and its `report.json` reported
`final_val_det = 0.7667` — v2's 118M number, identical to v2's own report — while
A0, which did beat it, reported its own weights.

Taken at face value this produced a spurious A2−A0 gap of **−3.1pp**, three times
the real one. §3 avoids it by evaluating each arm's own `model_final.pt`.

Suggested fixes: reset `best_val_det` when a branch declares a contract change;
or record in `report.json` which run produced the best weights; or refuse
`report_eval_weights="best"` when no `model_best.pt` was written this run.

## 9. Recommended next steps

1. **Extend the control to 200M.** A0 is still improving (+2.7pp in 15M, best
   0.84, LR schedule only 68% consumed). This is now the cheapest source of
   further gain and needs no new machinery.
2. **Do not run the from-scratch aux arm.**
3. **Fix the keep-best inheritance trap** (§8) before any further branch arms.
4. If planning-circuit questions continue, change the *probe*, not the loss: the
   +30pp-at-length-8 gap versus P1 says the computation exists and this probe
   cannot see it. Non-linear or state-dependent probes, or probes of the
   recurrent state across ticks rather than the zero-state readout, are better
   places to look than more supervision.

## 10. Provenance

- configs `groundup_drc33_aux_A0_control.yaml`,
  `groundup_drc33_aux_A1_lam1p0.yaml`, `groundup_drc33_aux_A1_lam0p3.yaml`
- all three declare `resume_allow_contract_change: [aux_landing_coef]` where
  applicable and print a BRANCH warning at startup
- W&B: A0 control + probes, A1 λ=1.0 + probes, A2 λ=0.3 + probes, all grouped
  under training run `yidjviev`
- sweep and analysis artefacts in `artifacts/forensics/aux/`
  (`sweep_stage1.json`, `sweep_stage2.json`, `final_eval_own_weights.json`,
  `aux_sweep.py`, `analyze_aux.py`)
- 215 tests passing; aux feature verified bit-identical to the pre-aux build when
  disabled, at unit scale and at production scale
