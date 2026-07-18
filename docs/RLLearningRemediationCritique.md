# Critique of the RL Learning Remediation Program (Phase 5)

**Reviewer:** independent audit agent, 2026-07-18
**Scope:** `docs/RLLearningInvestigationPlan.md`, `docs/RLLearningSynthesizedPlan.md`,
`docs/RLLearningExecutionLog.md`, `docs/RLLearningRemediationReport.md`, plus the
underlying artifacts, configs, git history, and tests on branch `sid-learnable-agent`.

**Review method — nothing below is taken from the docs on trust.** Every headline
number was re-derived from `results.json` / per-checkpoint `evaluation.json` files;
the predeclaration story was checked against git commit timestamps and
`git log --follow` on the rules files; the confirmation manifest's disjointness was
recomputed against the split manifest; the full test suite was re-run (129 passed,
9 skipped — matches the claim); and three **new** analyses were run that the program
did not do: (1) a paired per-difficulty-cell comparison of the promoted model vs its
BC-only start, (2) masked random / masked target-only baselines on the validation
suite, (3) extraction of the curriculum-transition timeline across recipe A's three
segments.

---

## 1. Verification summary — the docs are honest

Every specific factual claim checked out. This matters and should be said first:

| Claim | Status |
|---|---|
| All 16 A/B validation curve points, incl. 39.67%/41.82% at 2.048M | verified exactly |
| BC: 210,947/35,091 pairs; epoch 12 restored (val loss 1.3477 / acc 61.05%); epoch 25 overfit to val loss 2.97 | verified |
| Rung decisions match the committed rules (rung 1 ambiguity retention; rung 2 B eliminated on 2/3-robot regressions 3.81/8.39 pts vs 2-pt tolerance) | verified |
| Plateau stop at 6.144M (three non-improving checkpoints after 2.048M) | verified |
| Sealed confirmation set: 375 puzzles, all from test membership, zero overlap with train/validation/dev-test at puzzle **and** layout level; `test_status=sealed_not_run` everywhere | verified, recomputed |
| Every promoted checkpoint carries `model.zip` + `training_state.pkl` | verified |
| Geometry / BC-lever screen tables | verified |
| Seed-expansion gates predeclared | verified — committed 13:32 (2f1cb44), ~4h before the first suggestive result, never amended |
| A/B configs differ only in lr / ent_coef / target_kl; contract test genuinely enforces this (`from_mapping` rejects unknown keys) | verified |

Process integrity nits (real but minor): the execution log's wall-clock headings are
unreliable — several entries claim times up to 34 minutes **after** their own commit
timestamps (conservative direction for predeclaration, but the log's clock cannot be
read literally). And the allocation rules (ba36356, 17:53) were committed 19 minutes
after a config-identical deterministic preview of recipe B's first 102.4k steps had
been recorded (the geometry 16×128 arm — byte-identical results to B's rung-1 curve
at 102.4k), so the later claim "no A/B learning results have yet been inspected" is
technically-about-the-rungs but substantively overstated.

---

## 2. What is genuinely strong

1. **Evaluation governance is the best in the repo's history.** Sealed confirmation
   set with recomputed-disjoint membership; validation-only screens
   (`test_status: sealed_not_run` verified across every phase-5 run); predeclared,
   never-amended adjudication and plateau rules applied mechanically, including an
   ambiguity rule that correctly *refused* to eliminate on overlapping CIs at rung 1;
   full provenance (fingerprints, SHAs, per-checkpoint sidecar state).
2. **The BC-overfitting discovery is a real correction to phase 4b.** Held-out BC
   validation revealed that the historical 25-epoch BC recipe was badly overfit
   (val loss 1.35 at epoch 12 → 2.97 at epoch 25). Phase 4b's prior was worse than
   anyone knew, and best-epoch restoration is now the single most defensible recipe
   change of the program.
3. **The phase-4 saturation audit repairs the historical record.** Machine-verified
   evidence (logit between-state variance exactly 0, expert top-1 at chance) that the
   phase-4 from-scratch failure was at least partly a Tanh-head/initialization
   collapse — so "phase 4 vs 4b proves exploration was the only problem" was
   over-claimed, exactly as the independent investigation plan argued.
4. **The improvement over BC-only is probably real** (at the masked-policy level).
   My paired analysis over the 21 (length × robots) cells: 14 better / 4 worse /
   3 tied (sign test p ≈ 0.03), and the direction replicates across three
   independent-ish arms (A, B, geometry). This is not pure selection noise.
5. **Engineering quality:** exact-resume checkpoints with a genuine
   interrupted-vs-uninterrupted regression test; parity-tested batched evaluation
   (5.7×/10.2× speedups, identical record hashes); single-factor screens with honest
   rejections (batch 2,048/4,096 rejected; FP16 kept experimental despite a 2.27×
   kernel win because end-to-end gain was 4.5%).
6. **The 8×8 reverse-curriculum smoke deserves more prominence than it got:**
   genuinely from-scratch (no BC), all three stage transitions by frontier mastery,
   validation 21.2% → 67.9% stochastic in 300k steps. The reverse-start machinery is
   validated end-to-end and idle.
7. **Disclosed adaptivity.** The one mid-program reversal (unmasking the BC loss
   after the stall) was diagnosed with a decisive single-factor audit on the full
   dataset and logged openly rather than buried.

---

## 3. The headline claim, re-read

The report's summary — "a learning 16×16 policy … 39.67%/41.82% vs 30.58%/33.88% for
BC-only" — is numerically accurate and internally consistent. But three buried facts
change what it *means*.

### 3.1 The unmasked policy collapsed — PPO learned to lean on the mask (top finding)

Recorded in every `evaluation.json` under `counterfactual_unmasked`, never surfaced
in either doc:

| Checkpoint | Masked det/stoch | **Unmasked** det/stoch | Unmasked no-movement (det/stoch) |
|---|---:|---:|---:|
| BC-only (step 0) | 30.6% / 33.9% | 26.4% / 30.4% | 41% / 23% |
| A @ 2.048M (selected) | 39.7% / 41.8% | **14.9% / 20.2%** | 86% / 72% |
| A @ 6.144M | 37.2% / 38.7% | **9.1% / 13.4%** | 90% / 85% |

The raw network got **monotonically worse than its own BC initialization** while the
masked controller improved. Health telemetry agrees: probe no-movement probability
mass rose 2.9% → 7.8%, normalized entropy collapsed 0.50 → 0.10. Under masking there
is no gradient pressure to keep invalid-action logits down, so the policy freely
piles probability there.

Consequences:

- **Cross-phase comparisons are apples-to-oranges.** Phase 4b's 30.6% was unmasked.
  On the unmasked protocol, the new model is far *below* phase 4b's endpoint. The
  legitimate claim is about the masked *system* (policy + mask), and the report
  should say so explicitly.
- **Part of the masked gains is mask-assisted wandering, not better planning.** Mean
  successful optimality gap grew from 1.28 (phase 4b) to 2.3–2.8 moves; with a third
  of the action space force-pruned, a 20-step random-ish walk completes more puzzles.
  (Mitigating evidence: masked-random solves only 5.0% — see §3.4 — so the mask alone
  explains little; but the gap-worsening says solution *quality* declined.)
- **It matters for the project's real goal.** The interpretability agenda studies the
  network. A network that increasingly outsources action-validity to an external
  filter is a worse study object, and the planned DRC "thinking no-op action" will
  interact badly with a policy that loves invalid actions.
- The report's "executes no invalid or no-movement actions under deployment masking"
  is true **by construction** — the mask guarantees it for any policy, including a
  random one. It is presented in the summary as if it were a learned property.

### 3.2 Where the gains actually came from — and where they stopped

Paired-cell deltas (BC-only → selected, stochastic, 30 eps/cell):

- **+3.5 of the +7.9 aggregate points come from one cell**, (length 2, 2-robot):
  13% → 83%. The new BC start was anomalously weak there (old phase-4b BC scored
  ~58% on length 2); PPO repaired a hole that the *new* BC recipe itself introduced.
- Real but small gains on the deep two/three-robot tail (7,2): +27pp, (7,3): +23pp;
  a real regression on (5,1): −30pp.
- **The curriculum timeline is the tell:** recipe A reached the final stage (full
  length-1–8 mix) at episode ~82,300 — essentially at its 2.048M-step peak. It then
  spent **377k episodes at the final stage with zero validation improvement**
  (final-stage window success 45.6%; validation 41.8 → 35.7 → 38.8 → 38.7). This is
  the phase-4b pathology reproduced almost exactly: PPO improves while the staged
  curriculum restricts the mix, and stalls the moment it faces the full distribution.

So the accurate one-line summary is not "PPO now learns." It is: *masked PPO
consolidates and repairs the BC prior within the staged distribution, then hits the
same full-mix wall as phase 4b.* The program's own title question — why RL doesn't
learn at the frontier — remains open, and every lever aimed at it (frontier-weighted
sampling, DAgger/kickstarting, the 16×16 reverse run, relational head) sits in the
deferred register.

### 3.3 Decisions were made inside the noise floor

- The validation suite is 121 puzzles; puzzle-clustered 95% CIs are ±7pp wide. The
  A-vs-B margins that drove promotion were 1–3pp. B's elimination rested on 2/3-robot
  regressions measured on 42 and 31 puzzles (three-robot: 27 → 14 successes out of
  155 episodes). The rules were prospective and consistently applied — procedurally
  excellent — but statistically underpowered: B's elimination is plausibly an unlucky
  draw, and A's 41.82% is the max over six checkpoints selected on the same suite
  (winner's curse; expect roughly +2–3pp optimism). **Prediction to hold the program
  to: the confirmation-set stochastic result will land materially below 41.8%,
  probably mid-to-high 30s.** That would be consistent, not a scandal — but plan
  language accordingly.
- The same 121 puzzles (plus the same fixed eval seeds) were reused for BC epoch
  selection, the lever screen, geometry, both rungs, plateau stopping, and final
  selection. Adaptive reuse of a small suite is exactly what the sealed set protects
  against — but selection *quality* still suffers. The 10× batched-eval speedup the
  program built makes a 5–10× larger validation suite nearly free (6,881 validation
  members are available; the suite uses ~2%). They built the tool and didn't spend
  its dividend on statistical power.
- Small protocol wobbles bound the floor: the same BC weights scored 35.5% vs 33.9%
  stochastic under the serial vs batched step-0 evaluations, and the step-0 unmasked
  counterfactual differs by 1.3pp between runs despite `fixed_validation_seeds`.
  Nothing sinister — but ~1.5pp is the reproducibility floor, and several gates and
  margins are of that magnitude. (Also: length-7 stochastic 33% > length-5 27% at the
  selected checkpoint — strata of 90 episodes are noisy.)

### 3.4 Baseline mismatch (new measurement)

The recorded baselines (random 4.4%, target-only 8.5%) are **unmasked**, and the
harness structurally cannot produce masked ones (`random`/`target_robot_only`
controllers ignore `action_masking`). I ran masked baselines on the validation suite
(20 eps/puzzle, uniform over valid actions):

| Baseline | Unmasked (historical) | **Masked (this review)** |
|---|---:|---:|
| random | 4.4% | **5.0%** (1-robot 7.9 / 2-robot 4.4 / 3-robot 1.3) |
| target-robot-only | 8.5% | **10.7%** (1-robot 22.2 / 2-robot 5.0 / 3-robot 0.8) |

Honest finding in both directions: the shift is modest, so masking does **not**
trivialize the task (this *strengthens* the program's headline more than I expected).
But masked target-only at 10.7% — and 22% on single-robot puzzles — is the correct
floor for a masked model, and the confirmation report must be written against masked
baselines, which currently do not exist in the harness.

### 3.5 Attribution is still confounded — the causal matrix was dropped

Relative to phase 4b, the winning recipe changed several things at once: BC best-epoch
selection, train+eval masking (MaskablePPO), and the whole measurement pipeline
(fixed seeds, 5 episodes, batching, bootstrap CIs, validation-based checkpoint
selection). Both source plans made the BC × masking factorial the centerpiece; the
executed program ran only the (BC=yes, mask=yes) column. The from-scratch,
corrected-initialization control — the single experiment that would repair the
phase-4-vs-4b causal record — was never run at 16×16. Three missing cells are now
extremely cheap at the 512k rung protocol (~15–20 min each at observed throughput):

1. **BC-new + unmasked PPO** — isolates the masking contribution;
2. **old-style BC (no epoch selection) + masked PPO** — isolates the BC fix;
3. **no-BC + masked PPO from scratch** — the missing phase-3 control.

### 3.6 The masked-BC-loss stall was papered over, not explained

Masking the BC softmax pinned accuracy at masked chance (~9%, loss ≈ ln 12.4) for
three epochs with verified-nonzero gradients, on a model that reaches 15% in one
unmasked epoch. The offered mechanism ("masking removes useful invalid-action
contrast") would predict *slower* learning, not a hard plateau at chance. Something
is unexplained here, and the same masked-categorical machinery is used by MaskablePPO
during training. The workaround (unmasked BC loss, masked rollouts) is reasonable and
disclosed, but this anomaly is exactly the kind of loose thread that later turns out
to be a real bug. Worth a bounded half-day: sweep BC lr under the masked loss, check
gradient norms per layer, and test on the small 8×8 model where iteration is fast.

---

## 4. Prioritized issues and uncertainties

**P0 — affects the meaning of the headline result**

1. Unmasked-policy collapse is unreported in both docs; all "learning" claims are
   mask-dependent, and solution quality (optimality gap) regressed.
2. The full-mix frontier stall is reproduced, not solved; the recipe's own plateau
   data says it has saturated at ~2M steps. More seeds of the same recipe will
   confirm the number, not improve it.
3. Winner's-curse exposure on 41.82%: selection and gating all ran on 121 reused
   puzzles; confirmation likely lands lower. No masked baselines exist for that
   comparison yet.

**P1 — cheap to fix, high information value**

4. Missing causal cells (BC × masking × from-scratch) — each now ~15–20 min.
5. Masked baselines absent from the harness (controllers ignore the mask).
6. Validation suite is 2% of available validation members while eval is now 10×
   faster; selection decisions for seeds 29/47 should not rerun at current power.
7. Masked-BC-loss stall unexplained (shared machinery with MaskablePPO).

**P2 — hygiene**

8. Execution-log wall-clock headings disagree with commit timestamps (up to 34 min).
9. Thin strata drive hard gates: (L3,R3) has n=5 episodes; three-robot claims rest
   on 31 puzzles.
10. ~1.3pp irreproducibility between "fixed-seed" evaluation paths — find the stray
    RNG stream.
11. Deterministic selection metric rests on one episode per puzzle (n=121) since
    deduplication; fine for reporting, marginal for gating.

**Open questions for the project owner**

- Is the mask contractually part of the deployed/studied system? If yes, the
  headline stands (with masked baselines); if the *network* is the study object (the
  interpretability agenda suggests so), the unmasked collapse makes the promoted
  model a worse artifact than its own BC start, and mask-reliance needs an explicit
  countermeasure before further promotion.
- Is "performance-first" still the right frame now that the recipe has visibly
  plateaued? The locked decision was made when it was plausible PPO would keep
  climbing; the data now says the next points live in the deferred register
  (frontier sampling, reverse run, DAgger), not in seeds 29/47.

---

## 5. Recommendations

**A. Before launching seeds 29 and 47** (all cheap; roughly one day total)

1. **Add masked baselines to the harness** (respect `action_masking` in the
   `random`/`target_robot_only` controllers), record them on validation now and on
   the confirmation suite once, alongside the single sealed model evaluation.
2. **Enlarge the validation suite for the seed runs** (e.g. 25–30 per stratum,
   ~500–600 puzzles; ~2–3 min per checkpoint with batched eval). Keep the old
   121-puzzle suite reportable for continuity.
3. **Pre-register the confirmation analysis** in one committed paragraph: headline =
   masked stochastic vs masked baselines; unmasked counterfactual reported alongside;
   paired puzzle-level bootstrap vs the BC-only checkpoint; no re-selection whatever
   the outcome.
4. **Run the three missing 512k causal cells** (§3.5). They convert the report's
   "PPO improved things" into an attribution with numbers.
5. **Amend the two docs** with the unmasked story and the noise-floor caveats before
   any milestone claim propagates to `GridCnnGeneralizationReport.md`.

**B. The actual research frontier (after or instead of seed expansion)**

6. **Counter mask-reliance during training**: add a small auxiliary loss on masked-out
   logits (or compute the PPO entropy/log-probs over valid actions but add an
   invalid-mass penalty), and promote `counterfactual_unmasked` from telemetry to a
   health gate. Success criterion: masked performance retained while unmasked stays
   ≥ BC-only.
7. **Run the deferred 16×16 reverse-curriculum program.** Its stated resumption
   triggers (smoke + throughput + calibrated thresholds) are all met, the
   infrastructure is tested, and it is the only pending experiment that addresses
   the full-mix stall mechanism rather than the recipe.
8. **Frontier-weighted sampling / kickstarting-DAgger** at the final stage, per the
   deferral register — the paired-cell table shows exactly which cells are trainable
   frontier (2-robot lengths 4–8 at 20–40%) vs saturated vs dead.
9. Bounded investigation of the masked-BC stall (§3.6).

**C. Reporting discipline going forward**

10. Timestamp log entries from `date` output, not memory; keep the excellent habit of
    committing rules before results — and extend it by also committing the *analysis
    plan* for each result before the run finishes.

---

## 6. Bottom line

This is the most trustworthy experimental work in the repository to date: the
numbers are real, the sealed-set discipline held, the rules were predeclared and
mechanically applied, and two genuine discoveries (BC overfitting; the phase-4
initialization confound) permanently improve the project's understanding. But the
headline sentence overstates what happened. What was demonstrated is that *a masked
controller wrapped around a better-selected BC prior gains ~6–8 validation points
from a short PPO consolidation phase, concentrated in cells the new BC left weak,
while the raw policy degrades and the full-distribution stall that motivated the
whole investigation reappears on schedule at ~2M steps.* The program bought excellent
instruments and then pointed them at recipe polishing; the open scientific question —
what gives PPO a gradient at the multi-robot frontier — is still waiting, and the
tools to attack it (reverse curriculum, frontier weighting, causal cells, bigger
validation) are now cheap. Spend the next compute there, not on two more seeds of a
plateaued recipe.
