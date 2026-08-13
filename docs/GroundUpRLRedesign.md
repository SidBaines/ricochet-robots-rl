# Ground-Up RL Redesign

Date: 2026-07-18. Branch: `rl-ground-up`. Author: independent redesign pass;
the game environment (rules, board generator, solver, puzzle banks) is treated
as a given, everything RL-side is rebuilt from scratch.

## 1. Red-team verdict: the environment is learnable; the RL setup was the problem

Empirical probes (scripts run against `artifacts/phase4_bank`, 55,143 verified
16×16/4-robot puzzles):

| Check | Result |
| --- | --- |
| Stored optimal plans replay to the goal | 200/200 valid |
| BFS optimality re-derivation | 50/50 lengths match |
| `image` obs planes vs board ground truth | 0 mismatches; walls static across steps |
| Goal structure | goal cell always has exactly 2 walls (an L) — consistent, learnable cue |
| Legal-move branching | mean 14.1 of 16 actions |
| Interior wall density (16×16 ET2/CL2) | 32 of 480 interior edges (6.7%) — very sparse, long slides |

The decisive measurement — **masked-random success within the 20-step horizon,
by optimal length** (1,000 rollouts/length):

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random success | 0.40 | 0.18 | 0.105 | 0.030 | 0.010 | 0.002 | 0.002 | 0.000 |

PPO learns from its own successes. With a sparse terminal reward, no shaping,
and a length-staged curriculum that jumps 4→6→8, there is effectively **zero
learning signal beyond length ~4**. That is sufficient to explain every
from-scratch failure at 16×16 (Phase 3: 5.1%, Phase 3b: 7.0%, Phase 4: 3.3%)
without invoking any environment bug. BC warmstart "worked" because it
sidesteps exploration entirely — it treated the symptom.

Two candidate fixes were validated empirically before committing to them:

- **Potential-based shaping on a relaxed slide distance.** `d(c)` = minimum
  slides for the target robot from cell `c` to the goal if it could stop
  anywhere along a ray (walls only; classic admissible Ricochet heuristic).
  Verified: 0/300 admissibility violations, mean `d0/optimal` = 0.52,
  0.5 ms/layout to build (cacheable), 98.4% of cells reachable. A half-greedy
  walk on this potential lifts L=5 success from 4/250 to 33/250 — an ~8×
  stronger exploration signal, and as a potential-based term it cannot change
  the optimal policy (Ng et al. 1999).
- **Reverse starts along stored optimal plans.** Masked-random success from
  start-distance d: d=1 → 0.51, d=2 → 0.16, d=3 → 0.05. A frontier curriculum
  over d always has a foothold; the bank already stores the plans (validated
  above), so this is free.

## 2. Current setup vs. typical successful RL-for-puzzles setups

Similarities (kept):

- Structured one-hot planes at native grid resolution — standard for board
  games (AlphaZero, Sokoban DRC) and the single best decision in the repo.
- Flat `robot×direction` discrete action space — standard.
- Solver-labelled puzzle banks with layout-hash train/test splits and
  fingerprint discipline — better than typical; kept wholesale.
- PPO with clip, GAE — standard.

Differences (what a typical successful setup does that this repo did not):

| Axis | This repo | Typical | Consequence here |
| --- | --- | --- | --- |
| Reward | sparse +1 terminal, −0.01/step, no shaping | dense-ified via potential shaping, reverse curriculum, or off-policy relabelling (HER) when random success ≈ 0 | no gradient beyond L≈4 |
| Exploration budget | horizon 20 for L≤8 (2.5× optimal) | 3–10× optimal slack, time-awareness in obs | truncation before discovery; value aliasing near timeout |
| Illegal/no-op actions | unmasked for Phases 1–4b (91–95% argmax no-movement collapses) | masked from day one (AlphaZero always; MaskablePPO) | wasted capacity, collapse modes, later bolted on |
| Robot identity | fixed per-robot channels + target mask; policy must learn "which channel is the target" indirection | canonicalization/equivariance: target always slot 0, others order-canonicalized | 3!× redundant hypothesis space at 4 robots |
| Curriculum | staged max-length mixture gates; advancement certifies the easy-dominated mixture, plus forced safety-valve advancement | frontier-targeted sampling (reverse-from-goal, PLR): gate on the frontier band only, adaptive mixture, no deadlock | Phase 4 advanced purely via safety valve, then collapsed |
| Head architecture | flatten → 512-dim linear = ~68% of params, location-specific | small heads on pooled/reduced features; spatial or pointer heads | memorizes layouts; train/held-out gap |
| Optimizer schedule | constant lr 3e-4, 2 epochs, no KL guard | lr anneal, KL guard as safety net | late-run instability, brittle to scale |
| Evaluation | 121-puzzle suite reused for every selection decision; 1 ep/puzzle early phases; single seed; masked policy vs unmasked baselines | large fixed suites, ≥3 eps, multi-seed for claims, baselines under identical protocol | decisions made inside the noise floor |
| Throughput | ~700 steps/s (env object rebuilt every reset) | envs step in O(µs), experiments in minutes | 10M-step runs cost hours → too few iterations |

## 3. Concrete change list

1. **Reward: add potential-based shaping** `F = γ·φ(s′) − φ(s)`,
   `φ(s) = −α·d_relax(target robot cell)`, `α ≈ 0.1`, `φ(goal) = 0` naturally.
   Keep −0.01 step penalty and +1 goal. Optimal policy provably unchanged.
2. **Horizon: 20 → 32** at 16×16 (≥4× median optimal), with
   `TimeLimit.truncated` info set so PPO bootstraps truncations, plus an
   optional remaining-steps observation channel.
3. **Action masking from step 0** (MaskablePPO). All baselines and evals run
   masked too — one protocol everywhere.
4. **Canonical identity encoding**: at reset, relabel so the target robot is
   always robot 0 (channel 5, actions 0–3); non-target robots sorted by
   initial position. Removes the target-mask indirection and the robot
   permutation redundancy.
5. **Curriculum: replace staged-length mixture gates with a reverse-start
   frontier sampler.** Bands keyed by start distance d ∈ {1..L}. Frontier =
   smallest d with EMA success < τ (0.75). Sample ~50% frontier, ~25%
   retention (d < frontier), ~25% uniform mix (all d, including full starts).
   Self-pacing, no mastery deadlock, no safety-valve that advances on failure,
   and the gate measures exactly the frontier band.
6. **Architecture: kill the giant flatten-linear.** Conv trunk (stride-1,
   same-padded residual blocks — keep this part of the old design) → 1×1
   reduce to few channels → small flatten → modest heads. ~1–2M params, most
   of them convolutional. (Pointer/per-robot readout head kept as a later
   ablation, not in v1.)
7. **PPO hygiene**: rollout 16 env × 256 steps = 4,096; batch 512; 4 epochs;
   lr 3e-4 linearly annealed; `target_kl = 0.05` safety guard; ent_coef 0.01;
   γ 0.99, λ 0.95. Nothing exotic — the levers above carry the load.
8. **Evaluation protocol**: fixed held-out suites of ~400 puzzles (stratified
   by length, 25% layout-hash held out — same split discipline as before);
   deterministic + 3-episode stochastic; per-length and per-robots-moved
   breakdown; masked-random and masked-target-only baselines computed under
   the identical protocol; final claims require ≥3 seeds; no checkpoint
   selection on the reporting suite.
9. **Throughput: lean fixed-layout env.** Puzzle table decoded from the
   parquet banks once at startup; resets are array copies + curriculum
   sampling; per-layout static planes and distance maps cached. No env-object
   reconstruction, no pandas in the reset path.
10. **From-scratch first, BC optional.** BC warmstart masked the exploration
    bug and its gains under honest (unmasked-consistent) protocols were much
    smaller than reported. It stays available but off by default; the goal is
    an agent that learns the game, not one that memorizes BFS traces.

Explicitly reused from the existing repo (vetted): `ricochet_core.py` rules,
`solver.py` BFS, the parquet banks (`phase2b_bank`, `phase4_bank`) with their
stored optimal plans, layout-hash split convention, and the
detached-long-run/logging conventions from CLAUDE.md.

## 4. End goal and the helper-removal ladder

The v3 recipe deliberately runs "with all the helpers on" to first prove the
game is learnable and get a strong reference agent. The agreed end goal
(2026-07-18) is an agent trained under progressively *fewer* helpers, ending
at:

- **RGB image observations** (rendered board, not structured planes);
- **variable board grid dimensions** (one network across sizes);
- **coloured robots with no canonical target slot** — the network must
  identify the target robot by matching its colour to the goal marker, so we
  can study generalisation to unseen colours and the training conditions that
  produce it;
- **no action mask** (for at least some planned experiments);
- **no stop planes / no distance plane** (both are eval-time input
  enrichments and must be gone in the target setting).

Strategy: get it working with all helpers, then remove them **one at a time**,
confirming learning survives each removal before the next. Proposed order
(easiest→hardest, each step keeps everything else fixed):

1. Remove the distance plane (heuristic field; most "helpful" input).
2. Remove the stop planes (net must learn one-step slide dynamics; the
   pointer head must be replaced by a flat/spatial head that does not index
   landing cells — expect to need a deeper trunk for ray-tracing receptive
   field).
3. Remove canonical target-first identity (target robot identified via an
   explicit target-mask channel first, then via colour matching in RGB).
4. Remove the action mask.
5. Structured planes → RGB rendering (sprite renderer; conv stem over
   pixels).
6. Fixed 16×16 → variable board dimensions. Note the SpatialPolicy trunk and
   pooled value head are already size-agnostic; flatten-based heads are not.

Training-time-only helpers (reward shaping, reverse-start curriculum, the
solver-labelled banks) do not appear in the deployed observation and may be
kept throughout, subject to the E4 ablations. Comparisons of v3 numbers
against legacy results must carry the caveat that v3's observation space is
richer (stop/distance planes) until ladder steps 1–2 land.

## 5. Experiment ladder (sequential, each gates the next)

- **E0 — unit tests**: obs/mask/shaping/canonicalization correctness; shaping
  telescopes to a constant per episode; reverse starts reproduce plan states.
- **E1 — overfit sanity**: 8 fixed 16×16 puzzles, no curriculum. Gate: ≥95%
  deterministic within ~200k steps. Verifies optimizer + reward plumbing.
- **E2 — 8×8 from scratch** (phase2b bank, lengths 1–4, 2 robots): gate:
  ≥85% deterministic held-out within ≤1M steps (old best: 87% at 819k with
  band reweighting; we want to match/beat it with the new recipe, no BC).
- **E3 — 16×16 from scratch** (phase4 bank, lengths 1–8, 4 robots): the
  headline. Reference points under masked protocol: masked-random ≈5%,
  BC→PPO best ≈40–42% validation (single seed, selection-biased). Success =
  from-scratch (no BC) materially above the BC→PPO line, with honest
  held-out evaluation.
- **E4 — ablations + seeds**: no-shaping and no-reverse-curriculum arms at
  matched budget for causal attribution; 3 seeds on the winner.

Results are appended chronologically to `WorkInProgress.md` per repo
convention.

## 6. Results to date (updated 2026-08-13)

All numbers are masked, full-start evaluations. Each "sealed test" suite was
excluded from the reported model/checkpoint's own training and selection;
some selections were known from earlier experiments, as specifically noted
for D2 below. Single seed (11) throughout — exploratory phase by agreement. W&B project:
`ricochet-robots-rl`, runs prefixed `groundup-`.

| Experiment | Setup | Result |
| --- | --- | --- |
| E1 overfit gate | 8 fixed 16×16 puzzles | 100% det by 60k steps |
| E2 8×8 from scratch | planes, full recipe | **99.6% sealed-test det** (1M steps, ~4 min; legacy best 87%) |
| E3 16×16 from scratch | planes, full recipe, 10M | **78.1% sealed-test det / 80.9% stoch** (legacy from-scratch 3–7%; legacy BC→PPO ≈40% val) |
| E4 ablations (2M, matched) | v3 minus one lever | no-shaping 60.3%, no-reverse 58.8%, no-planes+flat-head **12.0%** (full ≈63%) — spatial perception/head is the decisive lever |
| R1 8×8 RGB gate | pixels + helper planes | 99.6% det (matches planes) |
| R1 16×16 RGB | pixels + helper planes, 10M | **77.1% sealed-test det** (matches planes 78.1%) |
| D1 DRC(3,3) 8×8 gate | faithful DRC, recurrent masked PPO | **99.2% det, stable through 1M** (v3; v1/v2 exposed and fixed two trainer bugs) |
| D1 DRC(3,3) 16×16 | all helpers, 10M | **64.8% sealed-test det** — matches feedforward on 1-robot puzzles, collapses on multi-robot (39/41% vs 68/59%); faithful flatten readout is the suspected binding constraint |
| P1 12×12 plane-free | no stop/distance planes, flat head, 181M steps | **69.2% sealed-test det** (best ckpt 112M); task plateaued 66–73% from 50M; landing-cell probes: mid/long slides never exceeded ~6%/4% decodability — the ray-tracing circuit never formed; motivates auxiliary landing-cell loss for ladder step 2 |
| D2 DRC(3,3) 12×12 plane-free | no stop/distance planes, recurrent PPO, 200M | **80.42% det / 81.25% stoch** on the fixed DRC-held-out seed-11 suite, +11.25/+12.92pp over P1; full-carry reset costs 38.22pp at 200M; see `docs/Drc200MCarryMechanismResult.md` |

For D2, “held out” is scoped to DRC training and selection. Legacy P1 predates
suite fingerprints, but reproducing every scripted-baseline aggregate strongly
evidences that it used the same selection; its aggregate strata informed the
DRC plan. Exact identity is therefore reconstructed rather than hash-proven
from the old JSON.

Failures found and fixed along the way (each within ~1M steps of training,
via behavioral probes; all now regression-tested):

1. SB3 default Tanh heads saturated on unbounded CNN features → constant
   value function (E3 v1). Fix: ReLU heads + LayerNorm. In CLAUDE.md.
2. Flat global readout could not learn "this slide ends on the goal" at
   16×16 (E3 v2). Fix: stop-cell planes + pointer-style spatial head.
3. Recurrent trainer post-convergence collapse (D1 v1): advantage
   normalization amplifies noise at near-zero advantage std. Fix: std floor
   + clamp + keep-best checkpoints.
4. Value divergence via clip-only value loss + truncation bootstrap feedback
   (D1 v2). Fix: PPO2 max-of-losses form, clipping default off.
5. RGB renderer geometry (open wall corners, shrunken edge cells, off-centre
   sprites, tint bleed). Fix: +1px canvas, closed segments, interior-only
   sprites/tint.
6. Resumed branch reports could silently headline an inherited parent best.
   Fix: final-only headline evaluation from immutable content-addressed
   checkpoints, explicit best-periodic diagnostics, verified provenance, and
   preflight rejection of unsafe semantic branches.

Infrastructure: `src/groundup/` now includes the faithful DRC core, recurrent
trainer, immutable evaluation provenance, exact carry-state probes, and paired
causal interventions. At the final mechanism-study commit the full suite was
273 passed, 9 intentionally skipped. Throughput remains 1,000–4,000 st/s
depending on configuration (legacy: ~700).
