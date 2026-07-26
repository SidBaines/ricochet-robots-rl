# Plan: auxiliary landing-cell loss for the plane-free DRC

Targets the central negative result of `docs/DrcPlaneFree120MReview.md` §5: after
120M steps, medium/long landing cells are **not** linearly decodable at any depth
or tick, and are *below* a random-init control — decodability *declines*
monotonically through training. The hypothesis is that this missing ray-tracing
knowledge gates performance beyond ~77%, and that supervising it directly (as an
output head, with labels free from the env) will build it while keeping the
observation plane-free.

This preserves the helper-removal ladder (`docs/GroundUpRLRedesign.md` §4): the
landing cells become a **target**, never an input. The observation stays
`(9, 12, 12)`.

---

## 1. Recommendation: mid-training screen first, fresh run only if it pays

**Start from the trained v2 network, not from scratch — but run a control arm
alongside it.** Then run from-scratch only if the screen is positive.

I initially expected the opposite, on the reasoning that a network which has
spent 120M steps compressing this information away would be too stiff to move.
The measurement says otherwise.

### 1.1 The trained network is plastic — measured, not assumed

The worry was ConvLSTM saturation. Cell-state distribution at the 120M
checkpoint (3.54M cell units):

| statistic | v2 @ 120M | old run @ 87M |
|---|---|---|
| abs-max `\|c\|` | 65.5 | 52.6 |
| **median `\|c\|`** | **0.200** | 0.263 |
| **mean `tanh'(c)`** | **0.816** | 0.819 |
| units with `tanh'(c) < 0.01` ("dead") | **2.7%** | 1.2% |
| `\|h\| > 0.9` | 1.8% | — |

The alarming abs-max is a 0.4% tail. **97.3% of the cell state sits in the
high-gradient regime**, with mean gradient factor 0.82 through the
nonlinearity. Gradients from a new head will flow into the trunk essentially
unimpeded. (This also corrects a claim in
`docs/Drc87MRestartCollapseInvestigation.md` §6, now annotated there.)

So the stiffness objection to mid-training does not hold.

### 1.2 What each design actually tests — they are different questions

| | question answered | cost |
|---|---|---|
| **mid-training** | *Can the missing knowledge be installed in this network, and does installing it improve the policy?* | ~3 h per arm |
| **from scratch** | *Does the auxiliary signal change what representation forms in the first place?* | ~20 h |

The mid-training arm is not merely a cheap approximation of the fresh run — it
is a **cleaner causal test of the bottleneck hypothesis**, because it holds the
network fixed and manipulates only the objective. The fresh run answers a
representation-learning question that only becomes worth 20 h once the
bottleneck hypothesis survives.

### 1.3 The decision rule

Both mid-training arms resume from the *same* `training_latest.pt`, and resume is
bit-exact (`Drc87MRestartCollapseInvestigation.md` §4), so this is a genuinely
paired comparison.

| outcome | interpretation | next |
|---|---|---|
| decodability ↑ **and** `val_det` ↑ vs control | bottleneck hypothesis supported | run the fresh 120M arm for the clean headline; then sealed test |
| decodability ↑ **and** `val_det` flat | **hypothesis falsified.** The knowledge is installable but is not what gates performance | stop; ~20 h saved; write it up as a negative result and look elsewhere (capacity? credit assignment? exploration?) |
| decodability will not rise | ambiguous — either baked-in compression or genuinely unlearnable by this trunk | fresh arm needed to disambiguate |

Two of the three outcomes are decisive, and the middle row is the one that saves
the most time. Given §1.1, the third row is unlikely.

### 1.4 The control arm is mandatory, not optional

The 120M run **had not converged** (review §3: still +1.4pp per 20M window, and
the final four evals were the run's highest). Any `val_det` gain in a
continuation would therefore be partly just *more training*. Without a
no-aux-loss control run for the identical step budget from the identical
checkpoint, the experiment cannot attribute anything. This is the single easiest
way to get a false positive here.

---

## 2. Arms

| arm | from | steps | aux weight | purpose | cost |
|---|---|---|---|---|---|
| **A0 control** | v2 `training_latest.pt` (120M) | → 135M | 0 | isolates "more training" | ~2.5 h |
| **A1 aux** | same checkpoint | → 135M | λ (tuned) | the screen | ~2.5 h |
| **A2 fresh** *(conditional)* | step 0 | → 120M | λ | representation learning | ~20 h |

A0 and A1 differ **only** in the aux term. 15M steps is ~1,830 iterations, enough
for the probes (evaluated every 1M on the existing fixed 16k dataset) to show a
clear decodability trend — the *decline* in the main run was visible within the
first 10M steps, so a rise should be too.

Before A1, do a **λ sweep**: 3 short runs (~2M steps, ~20 min each) at
λ ∈ {0.1, 1.0, 10} checking (a) the aux loss actually falls and (b) the
policy-gradient share stays healthy (§4). Pick λ, then run A1 properly.

---

## 3. Design

**Labels.** For each state, for each of the 12 `(robot, direction)` pairs, the
index of the cell that robot would stop at — exactly what
`EnvConfig.stop_planes` already computes via `RicochetEnv._slide_from`, but as a
target rather than an observation channel. Store as `(T, B, 12)` int64 in the
rollout buffer (144-way classification per pair), not as 12×144 planes.

**Head.** A small conv head on the final-tick hidden map `h_D` (the same map the
policy/value readout uses, and one of the representations the probes score):
`Conv2d(hidden_channels → 12, k=1)` → per-channel spatial softmax over the 144
cells → cross-entropy against the true landing cell. Predicting *where*
spatially, rather than a flat regression, matches the probe's own formulation so
the two are directly comparable.

**Loss.** `loss = policy - ent_coef*entropy + vf_coef*value + λ*aux`, with `aux`
the mean cross-entropy over the 12 pairs. Consider masking pairs whose robot
cannot move (the `blocked` case is already solved at 0.79 and would dominate the
average) — **weight the mid/long bands up, or report per-band aux loss**, since
those are the bands we care about.

**Where else to attach.** Worth trying attaching at *every* tick, not just the
last: the DRC's premise is iterative refinement, and supervising each tick would
test whether ticks can be made to carry successive refinement. Keep this for a
follow-up; start with `h_D` only.

---

## 4. The hazard, stated explicitly

**A new loss term sharing the trunk under `max_grad_norm=0.5` is exactly the
failure mode we just spent a day diagnosing.** `clip_grad_norm_` renormalises the
*combined* gradient, so an aux term that grows large will crowd the policy
gradient out of the update budget — the value loss did precisely this, taking the
policy's share from 98.8% to 3.7% to 0.14% and destroying the run
(`Drc87MRestartCollapseInvestigation.md` §6).

Mitigations, in order:

1. **Log the policy-gradient share of the update every eval.** This is the alarm
   that would have caught the last failure 5–20M steps early. It is ~10 lines
   (the measurement harness already exists in
   `artifacts/forensics/drc87m_restart/forensic_replay.py::grad_split`).
   **Implement this before A1, not after.**
2. Keep λ small enough that the aux gradient norm stays well below the policy's;
   the λ sweep should report `‖g_aux‖ / ‖g_total‖` directly.
3. Cross-entropy is already bounded per sample (unlike the unbounded value MSE),
   so the runaway dynamic is much less likely — but bounded-per-sample is not the
   same as bounded-in-aggregate.

---

## 5. Prerequisites

Three things need building before A1 can run. All are small, and each should
land test-first.

1. **Landing targets without changing the observation.** A new
   `EnvConfig.landing_targets: bool` that computes the 12 landing-cell indices
   and returns them in `info` (or via a method), leaving the obs at 9 channels.
   The obs shape *must not* change or the v2 checkpoint cannot be resumed.
   Benchmark the cost on the real 256-env path — this is 12 `_slide_from` calls
   per env step, in Python (CLAUDE.md: benchmark the real factory, not a
   synthetic env).
2. **Resuming a checkpoint into a network that gained a head.** `net.state_dict`
   will not match. Cleanest fix: give the aux head its **own optimizer**, so the
   main Adam state loads exactly as before while aux gradients still flow into
   the shared trunk. Load the trunk with `strict=False` and initialise the head
   fresh. This keeps A0 and A1 starting from a bit-identical trunk+optimizer.
3. **Additive config-contract evolution.** Adding `landing_targets` / `aux_coef`
   to `DRCTrainConfig` changes `_config_contract`, which would make *every*
   existing checkpoint unresumable — including the one both arms start from.
   Fix the comparison so a field absent from the stored contract is treated as
   compatible **iff** the current config has it at its default; reject if it is
   set to anything else. That preserves the guarantee (semantics stay pinned)
   while letting the config grow. Needed for A0 as much as A1.

---

## 6. What we measure

Per eval (every 1M), for both arms:

- `val_det` on the fixed 200-puzzle suite, and **per length** — the hypothesis
  predicts gains concentrated at lengths 5–8, not uniformly
- probe decodability on the **existing fixed 16k dataset** (hash `e97ef31a…`),
  so results are directly comparable to v2, the abandoned run, and P1
- aux loss, per band (blocked / short / mid / long)
- **policy-gradient share** (§4)
- `value_loss`, `explained_variance`, `approx_kl` as always

Primary endpoint: **mid/long probe gain vs the random-init control**, currently
−0.03 and −0.06. Secondary: `val_det` at lengths 6–8 in A1 vs A0.

A subtlety worth pre-registering: v2 is *already* much better than P1 at long
puzzles (+30pp at length 8) while having *worse* long-slide decodability than a
random net. So the link from "linear decodability" to "performance" is not
established — it is the thing being tested. If A1 raises decodability without
raising length-6-8 success, that link is broken, and it would also retire the
probe as a progress metric.
