# DRC Carry-State Study: Preregistered Observational Protocol

Status: protocol and implementation only. No carry dataset has been generated
and this document reports no empirical result.

## Question and scope

This study asks which trajectory variables are decodable from the recurrent
state of an existing schema-2 DRC policy while it acts. It is observational:
probe accuracy is evidence of decodability, not evidence that the policy uses a
variable causally. Carry interventions, patching, ablations, and behavioral
claims belong to Task 2B and are excluded here.

The preregistered primary representation is the final recurrent depth after the
forward pass that emitted the chosen action (`h_final`, `c_final`, and their
concatenation `h_c_final`). Linear probes test affine decodability. A small
one-hidden-layer ReLU MLP tests modest nonlinear decodability. Observations,
logits, puzzle identifiers, and labels are never probe features.

Each representation is standardized with training-row feature means and
standard deviations only (zero-variance dimensions use scale one). Regression
targets are likewise standardized from applicable training rows and converted
back to native units for metrics. Shared representations have separate typed
heads trained with Adam: MSE for regression, cross-entropy for multiclass, and
binary cross-entropy with logits for binary/multilabel targets. Multilabel
predictions use a fixed 0.5 sigmoid threshold; no test label tunes a threshold.
The minibatch objective is the unweighted mean of the applicable target-head
losses in that minibatch.

## Sampling and rollout contract

The input must be a supported `kind=drc_model`, `schema_version=2` checkpoint.
The dataset records its resolved path, full file SHA-256, originating run,
timestep and checkpoint kind, stored-config fingerprint, and a fingerprint of
the reconstructed architecture and state-dict shapes.

The puzzle pool is the validation split by default. A deterministic suite is
selected with the repository's `joint_stratified_subset`, stratified jointly by
`(optimal_length, robots_moved)`. Provenance includes the bank path and SHA-256,
the checkpoint's split seeds, suite seed, requested count per nonempty cell,
actual per-cell counts, and the existing `_puzzle_fingerprint` identity. Sparse
cells contribute all available puzzles. The sealed held-out evaluation split is
not an available CLI choice.

There is one deterministic, legally masked, full-start episode per suite
puzzle. The recurrent state is zeroed only at episode boundaries. At every
decision the implementation performs one trained-network forward, stores its
carry-in, logits, and returned carry-out, selects `argmax` from those same
masked logits, and then steps the environment with that action. Thus the saved
carry-out is the one computed by the exact forward that selected the row's
action. Episode success is backfilled after termination or horizon expiry.

## Exact oracle

Robot identities are exactly those exposed by `RicochetEnv` under
`canonical_identity=True`: robot 0 is the target and other identities remain in
their episode-start canonical order. Transitions implement the environment's
unit slide and exclude no-ops exactly as `action_masks()` does.

The distance solver is breadth-first search with a dictionary of first-action
bitsets at each depth. Equal-depth paths union their bitsets before that next
layer is expanded. The solver examines the entire first goal layer before
returning, retaining every optimal first action. Goal states have distance zero
and empty masks.

`max_depth` and `max_nodes` are resource caps. Reaching either yields an
explicit unknown status, a censor reason, distance `NaN` in storage, and false
availability; it never yields an unsolvability or numeric-distance label. A
known unsolvable status is possible only after graph exhaustion. For each known
non-goal state, a separate bounded BFS from every legal successor determines
whether the goal is reachable within `distance - 1`. The resulting reducing
mask is asserted equal to the independently computed all-optimal-first-action
mask. A node cap during this check censors only the reducing-mask label.
Whenever the full-start query is known, its distance is also asserted equal to
the bank's recorded `optimal_length`.

Cache identities include wall geometry, goal, canonical positions, robot
count, and solver protocol/caps. Row arrays and metadata report exact-label
availability, censor codes, root search nodes, reducing-verification nodes, and
aggregate coverage. Censored rows remain in the dataset and in all applicable
non-oracle targets.

## Dataset fields

All arrays are row-aligned. `layout_hash`, `episode_id`, and
`suite_puzzle_index` preserve grouping and pairing.

| Group | Fields | Definition |
|---|---|---|
| Inputs/provenance | `obs`, `positions`, `action_mask`, `policy_logits` | Actual pre-action observation, canonical robot positions, legal moves, and masked trained logits. Retained for audit/random unrolling, not used as probe features. |
| Trained carry | `trained_carry_in_{h,c}`, `trained_carry_out_{h,c}` | Shape `(row, depth, channel, height, width)`. Carry-out is from the action-producing forward. |
| Random carry | `random_carry_in_{h,c}`, `random_carry_out_{h,c}` | Architecture-matched seeded random network on the same trained-policy observation sequence. |
| Exact graph labels | `solver_known`, `solver_solvable`, `exact_label_known`, `exact_distance`, `optimal_action_mask`, `reducing_action_mask`, `reducing_label_known` | Availability is explicit. Unknown distance is `NaN`, never a numeric target; unknown masks are not treated as labels. |
| Solver audit | `solver_censor_code`, `reducing_censor_code`, `solver_nodes_discovered`, `solver_nodes_expanded`, `reducing_verification_nodes` | Censor map is recorded in metadata (`0=none`, `1=max_depth`, `2=max_nodes`). |
| Policy intent | `chosen_action`, `intended_robot`, `intended_direction` | `robot=action//4`, `direction=action%4`. |
| History/progress | `recent_actions`, `move_index`, `horizon_progress`, `optimal_distance_progress` | At lag k, the kth prior action is used when present, `BOS=n_actions` marks the exact episode boundary, and still older unavailable lags use `MISSING=n_actions+1`. Move index is zero-based before acting. Horizon progress is `move_index/horizon`. Exact-distance progress is `1-distance/full_start_optimal_length`, may be negative after detours, and is unavailable when censored. |
| Chosen-move outcome | `landing_cell`, `slide_length`, `moved`, `immediate_success` | Landing is flat `row*width+column`; length is Manhattan slide length; success means this move puts canonical robot 0 on the goal. |
| Episode/grouping | `future_success`, `episode_id`, `layout_hash`, puzzle seed/stratum fields | Future success is the completed episode outcome, repeated over its decision rows. |

Observations and logits remain float32. Carry tensors default to float16 storage;
that lossy storage choice is recorded. The NPZ is written by atomic replacement
and contains a SHA-256 over all array names, dtypes, shapes, bytes, and contract
metadata. Reuse rechecks this fingerprint, structural invariants, and every
requested checkpoint/bank/suite/solver/rollout/random-seed field. `--force-dataset`
is required to replace an existing dataset instead of validating and reusing it.

## Controls and split

The random-network control has the identical reconstructed DRC architecture and
a recorded local seed. It consumes every trained-policy observation in order,
resets only between episodes, discards its logits, never chooses an action, and
therefore cannot change the state distribution.

The shuffled-label control uses trained carries. Only applicable training labels
are permuted; features, applicability, layout assignment, and test labels stay
unchanged. Each target receives a deterministic independent permutation except
for structured tuples: intended action/robot/direction share one permutation,
and landing/length/moved/immediate-success share one. Optimal and reducing masks
are shuffled independently because independent verification can give them
different availability. The JSON records permutation hashes and changed-row
counts, including honest zero changes for constant labels. Within a fixed
representation and probe family, all three controls use the same probe
initialization seed and minibatch order.

Train/test assignment is by unique `layout_hash`, never by state. Layouts are
ordered by a seeded SHA-256 score, assigned wholly to one partition, and emitted
in a fingerprinted manifest with layout and row counts. The implementation
asserts disjointness and refuses runs with too few layouts for the requested
minimum per partition. It also refuses a split that puts only one training
class in a classification target that has multiple classes somewhere in the
dataset. Globally constant secondary labels remain reportable; their
discrimination metrics are `null` where mathematically undefined.

## Targets and metrics

Exact distance and exact action sets train and evaluate only where their
availability flag is true; every result reports row counts and coverage. Other
targets use all applicable rows. Results are nested as
`representation -> probe family -> control -> target` so trained, random, and
shuffled controls are directly paired for both linear and MLP families.

- Distance, slide length, move/progress variables: MAE and R2.
- Optimal and reducing action sets: top-1 action in the true set, micro
  multilabel F1, and exact-set accuracy.
- Future success: accuracy, balanced accuracy, and ROC-AUC.
- Intended action/robot/direction, recent actions, and landing cell: accuracy.
- Moved and immediate success: accuracy, balanced accuracy, and ROC-AUC.

When a metric is mathematically undefined—for example ROC-AUC with one test
class or R2 with zero target variance—the JSON value is `null` with an explicit
reason. No undefined value is replaced by zero. Legal-action masking makes
`moved` normally constant true; its ordinary accuracy is therefore a sanity
target, not evidence of rich representation, while balanced accuracy/AUC will
correctly be undefined.

## Interpretation limits

Probe comparisons are meaningful only as paired decodability measurements on
this fixed suite and layout split. A trained advantage over both controls is
consistent with learning-related information in the carry, but it does not show
that the policy reads or needs that information. MLP gains over linear probes
show nonlinear accessibility under the chosen capacity, not an intrinsic code.
Censoring can change the evaluated subset and must be interpreted through the
reported coverage. Per-decision rows within an episode are correlated; the
layout split prevents leakage but does not make rows independent experimental
units. No result should be claimed until the preregistered CLI has run and its
dataset/result fingerprints have been preserved.
