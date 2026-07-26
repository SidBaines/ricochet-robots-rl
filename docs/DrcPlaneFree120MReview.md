# DRC(3,3) plane-free 12x12 — 120M review

Formal review of the completed clean run `drc33_p1_12x12_planefree_v2`
(W&B `yidjviev`, probes `s107q3qd`), which reached its registered 120M stop
target on 2026-07-25 at 23:26 BST.

This run replaced an earlier attempt whose rewards were corrupted from step 0 by
a truncation-bootstrap bug; that attempt and its forensics are documented in
`docs/Drc87MRestartCollapseInvestigation.md`. **The sealed test remains
unopened.**

---

## 1. Headline

| | value |
|---|---|
| steps | 120,004,608 (20.4 h wall, ~1,640 st/s) |
| final validation, deterministic (480 puzzles) | **0.7667** |
| final validation, stochastic (480, 3 eps/puzzle) | **0.7743** |
| best periodic validation (200 puzzles) | 0.810 at t=118,005,760 |
| baseline: random | 0.1408 |
| baseline: target-only | 0.2429 |
| contract fingerprint | `e7b3bf2186299ec0a75ff408f97eec03e38719c3866e9c171e92391fa3f7d4ee` |
| policy params | 1,650,701 |

**Read 0.7667, not 0.810.** The 0.810 is the maximum over ~120 evaluations on a
200-puzzle suite whose per-point standard error is ≈0.031. Taking the max of 120
noisy draws is a selection bias, and the independent 480-puzzle suite scores
those same weights at 0.767 — which is exactly where the run's *mean* sat
(last-10-row mean 0.776). Treat `best_val_det` as a checkpoint-selection device,
not as a result.

## 2. The instability is gone

This was the primary thing the run had to demonstrate, and it did so
unambiguously. Across **all 120 eval rows**:

- `value_loss` ∈ **[0.076, 0.155]** — it never drifted, at any point
- `explained_variance` rose monotonically 0.257 → 0.677
- `approx_kl` stayed ~0.001–0.003; 3 updates out of 46,876 hit the `target_kl`
  valve (0.006%)
- no NaN/Inf, no errors

For contrast, the corrupted attempt's `value_loss` was 3.589 by 1M, sat at 0.037
around 80M, then ran away 0.31 → 0.71 → 1.72 → 19.18 → **335.6**, taking the
policy from 0.770 to 0.275. This run passed that region and continued for
another 31M steps with no sign of it.

A mid-run pause at t=96,002,048 (battery) is invisible in the curve:
`val_det` went 0.765 at the pause boundary → 0.770 → 0.780. Both launches share
one contract fingerprint and one set of split fingerprints.

## 3. Learning curve

`val_det` on the fixed 200-puzzle periodic suite, windowed means:

| window | mean |
|---|---|
| 0–20M | 0.541 |
| 20–40M | 0.623 |
| 40–60M | 0.693 |
| 60–80M | 0.743 |
| 80–100M | 0.753 |
| 100–120M | 0.767 |
| last 10 rows | 0.776 |

**It had not fully converged at the stop.** The last two windows still gain
~1.4pp each, and the final four evals (0.795 / 0.810 / 0.780 / 0.785) are the
highest of the run. The pre-registered LR horizon is 200M and the schedule was
only 60% consumed (lr 3.0e-4 → 1.8e-4 of a 5.0e-5 floor), so the curve is still
being driven. An extension is scientifically defensible on the curve alone.

## 4. Comparison to the P1 feed-forward baseline

| length | v2 DRC (validation) | P1 feed-forward (test) |
|---|---|---|
| 1 | 0.983 | 1.000 |
| 2 | 0.983 | 0.917 |
| 3 | 0.867 | 0.900 |
| 4 | 0.750 | 0.817 |
| 5 | 0.700 | 0.650 |
| 6 | **0.750** | 0.567 |
| 7 | **0.583** | 0.467 |
| 8 | **0.517** | 0.217 |
| overall | **0.767** | 0.692 |

**Caveat: these are different splits.** P1's 0.692 is its one-time *sealed test*;
v2's 0.767 is *validation*. They are not a like-for-like headline until v2's
sealed test is opened, and validation typically flatters. The comparison is
nonetheless informative because the shape differs so sharply:

- v2 is **slightly worse on short puzzles** (1–4) — P1 is better at lengths 1, 3
  and 4
- v2 is **much better on long puzzles**: +18pp at length 6, +12pp at 7, and
  **+30pp at length 8** (0.517 vs 0.217)

That is the signature you would want from recurrence: the gain is concentrated
exactly where multi-step lookahead should matter. It is also the strongest
argument that the DRC is doing something the feed-forward net cannot, and it
survives the split caveat, since a validation-vs-test gap would not
preferentially inflate long puzzles by 30pp.

This is also the first **reward-semantics-clean** comparison with P1: the
corrupted attempt bootstrapped on truncation while P1's SB3 path never did.

## 5. Probes: the medium/long landing circuit still does not form

Zero-state linear landing-cell probes on the fixed 16,000-state dataset
(content hash `e97ef31a…`, **byte-identical to the one used for the previous
attempt and for P1**, so all three are directly comparable). Final row,
t=120,004,608, against a random-init control:

| representation | blocked | short (1–3) | mid (4–7) | long (8+) |
|---|---|---|---|---|
| encoder | 0.787 (+0.586) | 0.220 (+0.153) | 0.015 (**−0.032**) | 0.005 (**−0.058**) |
| tick 1 | 0.498 (+0.450) | 0.150 (+0.103) | 0.013 (**−0.023**) | 0.015 (**−0.052**) |
| tick 3 | 0.445 (+0.392) | 0.172 (+0.117) | 0.012 (**−0.021**) | 0.010 (**−0.059**) |
| readout | 0.789 (+0.596) | 0.235 (+0.168) | 0.010 (**−0.033**) | 0.004 (**−0.063**) |

(gain = trained − random-init control)

Three observations, and the third is the important one:

1. **Blocked-move detection and short slides are strongly encoded** (+0.59 and
   +0.17 over control), at every depth. The network clearly represents local
   one-step dynamics.
2. **Mid and long landing cells are not encoded at any depth or tick** — and
   they are *below* the random-init control, at every one of the four
   representations measured.
3. **They decline monotonically through training.** Long-slide decodability at
   the readout falls 0.054 (1M) → 0.025 (10M) → 0.013 (20M) → 0.007 (30M) →
   0.004 (120M). The information is *more* linearly available in a randomly
   initialised network than after 120M steps of RL. Training is actively
   compressing it away.

Adding three recurrent ticks and 120M steps did not produce the ray-tracing
circuit. This replicates the P1 finding (mid never exceeded ~6%, long 2–4%) and
strengthens it: on the DRC the trend is not merely flat but negative.

Note the tension with §4: the DRC is **much better at long puzzles** while
having **less linearly-decodable long-slide knowledge** than a random net. So
whatever the recurrence is buying at lengths 6–8, it is not a linearly-readable
landing-cell map. Either the computation is non-linear/distributed in a way this
probe cannot see, or the policy is solving long puzzles by a route that does not
require explicit ray-tracing. The probe is a *linear, zero-state* probe on a
fixed state set; it is evidence about linear decodability, not about the absence
of the computation.

## 6. Disposition

1. **Do not open the sealed test yet.** The run had not converged (§3), the
   obvious next step is an LR-horizon extension or the aux-loss arm, and the test
   is one-shot. Spend it on the configuration you intend to report.
2. **Keep `model_best.pt`** (t=118,005,760, periodic 0.810, 480-suite 0.767).
3. **Two candidate next moves**, in order of expected information per hour:
   - the **auxiliary landing-cell loss** arm (see
     `docs/DrcAuxLandingLossPlan.md`) — directly targets the §5 finding;
   - a **200M extension** of this run — cheap, exact-resumable from
     `training_latest.pt`, and justified by the un-converged curve, but it tests
     scale rather than a hypothesis.
4. Still unimplemented from the investigation's recommendations: `V_mean`/`V_std`
   on a fixed probe set and the policy-gradient share, logged per eval. The
   second becomes important for any aux-loss arm (§ plan).

## 7. Provenance

- config `configs/groundup_drc33_p1_12x12_planefree.yaml`,
  `run_name: drc33_p1_12x12_planefree_v2`
- launched at commit `fc310c8` from step 0, resumed at `1d901e0` from
  `training_pause_96002048.pt`; both clean, both same contract fingerprint
- bank `size-12x12_numR3_ET2_CL2.parquet`, sha256 `0103a7b9…`
- split fingerprints: train `86c69a75…`, val pool `a5029fe4…`, periodic suite
  `741960337e…`, final suite `1b6e5052…` — **identical to the previous attempt
  and to P1**, so all curves are directly comparable
- `bootstrap_on_truncation: false`, explicit in the config and therefore pinned
  by the contract fingerprint
- DRC(3,3), 32 channels, 1,650,701 params, obs `(9, 12, 12)`, no stop/distance
  planes, horizon 24, seed 11
- 120 retained per-1M model snapshots and 120 probe evaluations
