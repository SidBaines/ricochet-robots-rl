# DRC Carry-State Intervention Study: Preregistered Causal Protocol

Status: protocol and implementation only. No checkpoint-wide intervention
study has been run, and this document reports no empirical result. In
particular, no carry-dependent performance difference is explained by this
document or by implementation tests.

## Question and causal scope

This study asks which recurrent depths, state components, and times contribute
causally to the deterministic full-start performance of one frozen DRC policy.
It complements the observational decoding protocol in
`docs/DrcCarryStateStudy.md`. Decodability alone is correlational; the present
protocol alters incoming carry while holding the checkpoint, puzzle suite,
observation, action mask, and evaluation rule fixed.

The primary outcome is binary episode success. Every core contrast is paired
within puzzle and estimates

`mean(intervention_success - untreated_baseline_success)`.

The patch contrasts estimate the analogous paired suffix-success difference
from an exact shared environment state. Every patch arm uses the estimand
`mean(recipient_patched_suffix_success -
recipient_natural_suffix_success)`. The donor-natural suffix described below
is a required positive control, not a second baseline, additional experimental
unit, or part of an effect estimate. Positive values mean the intervention
improved success; negative values mean it harmed success. These interventions
can establish that a carry part matters under the stated ablation or patch,
but they do not by themselves assign a human-interpretable algorithm to that
part. Sharing this estimand does not share an inference role: only the overall
full-`hc` transplant is preregistered-primary, while partial `h` and `c`
transplants are descriptive localization. Accordingly, every patch component
uses the neutral machine label
`estimand_role=paired_recipient_suffix_estimand`.

## Frozen checkpoint and puzzle suite

The runner accepts only `kind=drc_model`, `schema_version=2` checkpoints and
reconstructs them strictly with the existing loader. It records the resolved
checkpoint path, full SHA-256, checkpoint kind, originating run, timestep,
complete stored config, config fingerprint, reconstructed architecture
contract, architecture fingerprint, and state-dict shapes. The checkpoint is
hashed again after the study so replacement during a run is fatal.

Validation is the default and primary split. Training is available as an
explicit secondary diagnostic. The sealed test split is not exposed by the
CLI and is rejected by the programmatic entry point. Split membership uses the
checkpoint's `eval_fraction`, `split_seed`, and `val_test_seed` exactly.

Within the requested split, `joint_stratified_subset` selects up to
`per_stratum` puzzles from every actually observed
`(optimal_length, robots_moved)` cell. The default is eight per cell and the
suite seed is 9. `per_stratum` must be positive. Sparse cells contribute all
available puzzles and missing cells are not invented during sampling. The
result records bank path/SHA-256, split seeds, suite seed, requested count,
actual per-cell counts, puzzle count, and the repository's existing suite
fingerprint. There is one core experimental unit per selected puzzle.

All episodes start at the bank's full initial state. Actions are always legally
masked. Deterministic actions are `argmax` of the same frozen-policy forward
whose returned carry is threaded to the next decision. The initial carry is
zero. The environment horizon and all observation/reward settings come from
the checkpoint config.

## Core intervention arms

The untreated baseline is one deterministic episode per puzzle with carry
zeroed only at episode start. A "completed move" means one environment action
has already been selected and stepped. An intervention at move index `k` is
applied to the carry returned by move `k-1`, immediately before the forward
that selects move `k`. Thus no every-move arm performs a redundant intervention
at move index 0.

The following arms are preregistered:

1. `reset_all_every_move`: after every completed move (`move_index > 0`), zero
   every recurrent depth in both `h` and `c`. This is the direct comparator to
   the prior full-carry/no-carry evaluation.
2. `reset_depth_d_every_move`: separately for every zero-based recurrent depth
   `d`, zero only incoming `h[d]` and `c[d]` after every completed move. All
   other depths are preserved byte-for-byte.
3. `reset_h_every_move`: zero incoming `h` at every depth after every completed
   move while preserving all of `c`.
4. `reset_c_every_move`: zero incoming `c` at every depth after every completed
   move while preserving all of `h`.
5. `reset_once_after_k_moves`: once, immediately before the next forward after
   exactly `k` completed moves, zero all depths of both `h` and `c`, then carry
   normally. The preregistered default values are `k = 1, 2, 4, 8`.

Every core arm retains the all-selected-puzzle estimand and uses the neutral
machine label `estimand_role=all_selected_puzzles_itt_estimand`. Separate
inference-role metadata marks only the overall full reset as
preregistered-primary; depth, component, and timing arms are descriptive
localization.

One-shot eligibility is defined from the untreated path: a puzzle is eligible
at `k` only if its untreated episode reaches a decision after `k` completed
moves. Before the intervention, the treated and untreated action/position
prefixes must be exactly equal or the run aborts. A puzzle that terminates or
expires before that forward remains in the intention-to-treat denominator,
with `eligible_pre_intervention=false`, `intervention_applied=false`, and an
outcome identical to baseline. Each arm records per-puzzle application flags,
application move indices, puzzle-level application totals, and explicit
eligible/not-eligible counts for timing arms.

The timing ITT estimand/contrast includes all selected puzzles. Its inference
role is descriptive, hypothesis-generating localization. Its `overall`,
`by_optimal_length`, `by_robots_moved`, and `by_joint_cell` summaries each
report `n_eligible`, `eligibility_rate`, `n_applied`, and `application_rate`.
Both named rates use all puzzles in that particular effect summary as their
denominator; the denominator definition and integer denominator are emitted
explicitly. Application among eligible puzzles is also reported with its own
eligible-puzzle denominator. For an empty cell, both counts are zero, both
all-puzzle rates are null, and the eligible-denominator application rate is
null. Thus a small ITT effect at a late `k` cannot be read without that cell's
reach/application coverage.

Every timing arm additionally emits
`effects.eligible_puzzles_descriptive_effects`, with the same overall and full
marginal/joint layout, restricted to puzzles whose untreated path was eligible.
Protocol checks require every such puzzle to receive the intervention. This is
a secondary, descriptive principal-stratum summary. It is neither the
all-selected-puzzle ITT estimand nor a randomized estimand, and it does not
replace or reweight the all-puzzle result. Its purpose is to separate dilution
by pre-intervention reach from the observed effect within the reached subset;
it does not license a population-causal claim about a post-selected subgroup.

State transformations clone both tensors before changing them. Tests pin that
only the named component/depth is zeroed and that the source carry is not
mutated.

## Exact matched-trajectory patching

Patching is a separate within-puzzle experiment. For every puzzle, the runner
generates a prespecified number of stochastic source rollouts of the same
frozen, legally masked policy. The default is 16. Each rollout uses a locally
constructed CPU `torch.Generator` with a seed derived from `patch_seed`, puzzle
index, and rollout ID. Sampling probabilities may be produced on another
requested model device, but the categorical draw occurs on CPU. Source
generation does not reseed or consume the caller's global Torch or NumPy RNG.

A prefix context is eligible only when two distinct source rollouts:

- are on the same selected puzzle;
- have completed the same positive number of moves;
- have different exact action histories; and
- have the exact same ordered canonical robot positions.

The move count fixes the time channel, but it is not accepted as a proxy for
state equality. The implementation directly requires byte-identical saved
observations and exact equality of the complete serialized environment
snapshots. The compared snapshot fields include local puzzle index, ordered
robot positions, completed-step count (environment time), start distance,
episode-done flag, and the complete environment RNG state. Each context's
snapshot step and robot positions must also agree with its recorded move index
and canonical positions. These comparisons are direct value comparisons, not
acceptance based only on matching hashes. The protocol never uses a different
board, approximate state similarity, a covariate match, or a different move
count as fallback.

At most one context is selected per puzzle. Selection uses prefix information
only: earliest completed-move index, then lexicographic canonical positions,
then the lowest rollout IDs. Source success and all suffix outcomes are absent
from the selector and cannot influence the choice. Honest zero exact matches
is a valid result. The output records each selected recipient/donor rollout ID
and local seed, both action prefixes, move index, positions, separate recipient
and donor observation, environment-state, and environment-RNG hashes, the list
of compared snapshot fields, and source-rollout action-history hashes. It also
lists every puzzle with no eligible match.

Before any patch effect is recorded, two deterministic natural suffixes are
run in separate fresh environments:

- recipient-natural starts from `recipient.environment_state` plus
  `recipient.carry`;
- donor-natural starts from `donor.environment_state` plus `donor.carry`.

Each environment must restore its supplied snapshot exactly, including RNG and
completed-step state, and must reproduce that context's saved observation
byte-for-byte. The recipient's saved environment snapshot is then deep-copied
into another fresh environment for each of three deterministic patched
suffixes:

- `hc`: donor `h` and donor `c` at every depth;
- `h`: donor `h`, preserving recipient `c` exactly;
- `c`: donor `c`, preserving recipient `h` exactly.

The donor-natural suffix and the full `hc` donor-to-recipient transplant have
identical environment state and identical incoming carry by construction.
They are therefore a mandatory implementation positive control: their complete
action sequences, success, suffix and total move counts, and final positions
must be identical or the run aborts. This control is reported for every
eligible context but is never substituted for recipient-natural in the paired
recipient-suffix effect estimate.

No tensor is stored in JSON. Component hashes and explicit verification flags
show that requested components equal the donor, unrequested components equal
the recipient, both source carries remain unchanged through all suffixes, both
natural snapshots restore exactly and reproduce their observations, the full
recipient/donor snapshots (including RNG and time) match, and the donor-natural
positive control passes. Each suffix uses a fresh environment and cloned carry;
state is never continued from another suffix. The output records recipient-
natural, donor-natural, and recipient-patched successes, suffix and total move
counts, action sequences, the recipient-natural-versus-patched paired success
delta, and first/action-count divergence for that paired recipient-suffix
comparison. Move counts are audit fields only; each patch arm also totals
diverged and non-diverged paired suffixes. The paired recipient-suffix analysis
does not condition move counts on post-treatment success.

`--patch-rollouts 0` disables patching for a checkpoint-wide core sweep. The
JSON then reports `status=disabled`, `eligibility.status=not_evaluated`, null
eligibility counts, null patch effects, and a donor-control status of
`not_evaluated`. It does not convert disabled work into zero eligible matches
or a zero effect. A positive patch setting must be at least two. Enabled runs
with no match report `enabled_no_matches`, zero eligible puzzles, a donor-
control status of `no_eligible_contexts`, and effect estimates as null with
`n=0`.

## Paired analysis and uncertainty

For every core arm, each selected puzzle contributes exactly one paired binary
delta, including ineligible timing puzzles. For every patch component, each
eligible puzzle contributes at most one paired recipient-natural-versus-
patched suffix delta. Donor-natural is only a required positive control and
contributes no delta or extra sample. Therefore the ordinary bootstrap unit is
always a puzzle; environment moves, recurrent ticks, and individual suffix
actions are never resampled as if independent.

Every report includes `n`, baseline/natural and intervention/patched success
counts and rates, mean paired success delta, and improved/harmed/tied puzzle
counts. A deterministic percentile paired bootstrap resamples the vector of
puzzle deltas with replacement using a local NumPy generator. The default is
10,000 replicates, a 95% interval, and base seed 42,424; group-specific seeds
are deterministic hashes of the base seed, arm, and stratum.

All reported 95% intervals are nominal, pointwise intervals only. No
multiplicity adjustment is performed, and no interval has family-wise or
simultaneous coverage. The overall full-reset-every-move contrast and overall
full-`hc` transplant contrast retain their preregistered-primary pointwise
labels. Overall depth, `h`/`c` component, timing, and partial-patch component
localization intervals are explicitly descriptive and hypothesis-generating.
Every `optimal_length`, `robots_moved`, and joint-cell interval is likewise
descriptive and hypothesis-generating regardless of its parent arm. Every
eligible-puzzle timing interval is labeled secondary descriptive
principal-stratum. No scientific claim may be selected or justified by which
of these localization, subgroup, or eligible-subset intervals happen to
exclude zero.

This contract is machine-readable twice: every `bootstrap_ci` records
`coverage_scope=pointwise_only`, `multiplicity_adjustment=none`, false
family-wise/simultaneous coverage flags, an `inference_role`, and a false
zero-exclusion claim flag; the result-level `inference` object defines those
roles and the global claim policy. The labels describe interpretation only and
do not silently convert any interval into adjusted inference.

The same analysis is emitted overall, by `optimal_length`, by `robots_moved`,
and by their joint cell. Joint output contains the full Cartesian product of
the suite's observed marginal values, so a missing sparse cell appears with
`n=0`, null rates/effect/interval, and `size_status=empty`. A one-puzzle group
is marked `single_item`; its bootstrap interval is explicitly degenerate at
that puzzle's delta rather than presented as stable population uncertainty.

`localization` is a direct machine-readable view containing the overall
baseline rate and separate maps for full reset-every-move, per-depth resets,
`h`/`c` component ablations, and one-shot timing resets. Full stratified
analyses and per-puzzle action audits remain under `core.interventions`.
Timing localization entries also expose their overall eligibility/application
coverage and overall eligible-puzzle descriptive effect directly; the full
eligible-puzzle stratification remains under the corresponding core arm.

## Provenance, determinism, and output

The result uses schema 2 and intervention protocol
`drc-carry-causal-interventions-v2`. This version bump records the additive
timing-coverage, eligible-subset, and inference-interpretation contract. It does
not change intervention execution, suite selection, source-rollout selection,
seed derivation, the patch protocol, or the paired-bootstrap algorithm, so no
fixture, split, or suite fingerprint is intentionally changed by the bump.

The result records schema and protocol versions; all suite, rollout, bootstrap,
and patch seeds/settings; the complete environment config; checkpoint and bank
identities; suite identity; and SHA-256 hashes of the implementation and shared
model/environment/selection files plus this preregistration. Those source hashes
are checked again after execution and a mid-run change is fatal. Variable
timestamps, elapsed time, runtime versions/platform, and git state live only in
the clearly separated `runtime` and `git` objects. For fixed files, settings,
device, and software stack, the remaining scientific payload is deterministic.

Model train/eval mode is restored even on failure. The implementation uses
local Torch and NumPy generators and tests that caller RNG streams are
unchanged. JSON serialization uses `allow_nan=False`, contains no tensor blobs,
and is published atomically. Existing output is rejected unless `--force` is
explicit; the no-force publication path uses an atomic no-clobber link, so a
race cannot silently overwrite another result.

The default CLI reports phase changes and approximately ten puzzle-level
milestones per long core/patch phase to stderr. Stdout remains one concise JSON
completion record, so existing capture behavior is preserved. Pass
`--quiet-progress` to suppress progress. Progress reporting does not enter the
scientific result and does not alter atomic publication.

Example core-only invocation (not run as part of this implementation task):

```bash
.venv-phase2/bin/python -m src.groundup.intervene_drc_carry \
  --model artifacts/example/model_190000000.pt \
  --bank artifacts/example/puzzles.parquet \
  --output artifacts/example/carry_interventions.json \
  --split val --suite-seed 9 --per-stratum 8 \
  --timing-moves 1 2 4 8 --patch-rollouts 0 --device cpu
```

## Interpretation limits

Reset effects need not add across depths or components: DRC dynamics are
nonlinear, and zero is an off-manifold intervention after episode start. A
large depth/component effect localizes sensitivity to that ablation but does
not prove the component stores only one variable. A small effect can reflect
redundancy, rapid reconstruction from the current observation, or an
insufficiently disruptive intervention rather than irrelevance.

Patching is more state-controlled than zero ablation but answers a narrower
question: whether history-dependent carry differences at exact stochastic
reconvergences alter the recipient's deterministic suffix. Eligible
reconvergences can be rare and are a selected subset of puzzles/states; no-match
counts and stratum coverage must accompany any claim. Donor histories may
carry many simultaneous differences, so even a causal patch effect does not
identify a single decoded feature without additional targeted interventions.

No claim about what explains a carry-dependent performance difference is
permitted until the preregistered runner is executed on the named checkpoint
and suite, its JSON is preserved, and the full-reset result, localization arms,
patch eligibility, paired intervals, and per-stratum heterogeneity are reviewed
together.
