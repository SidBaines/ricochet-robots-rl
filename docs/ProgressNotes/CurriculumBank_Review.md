# Curriculum Bank Redesign — Review

## Context
This review consolidates two rounds of notes after implementing the bank-first curriculum generation, manifest v1.1, stratified sampling, and the controller.

## Strengths
- Offline, bank-first workflow eliminates online solving during training; throughput-friendly.
- Manifest histograms enable controller stopping criteria and quick coverage introspection.
- Robust replay via `layout_blob` reduces risk from generator drift; seed+hash remains as fallback.
- Stratified sampling with leftover reallocation smooths distributions across optimal-length bands.
- Clear CLI to drive controller and reproducible generation runs.

## Concerns and Risks
- Manifest fidelity: histogram counts can drift if partitions are externally edited or duplicated; no built-in verification yet.
- Coverage estimation: controller sums raw histogram counts per level; overlapping bands/specs can be double-counted, leading to premature stop.
- Depth override: using a global max depth for all specs within a loop can over-solve easy specs; prefer per-spec depth limits.
- Naming clarity: some CLI paths still mix curriculum constraints with solver cutoffs (`max_depth`). Continue migrating to `solver_max_depth_override` in generation paths.
- Sampler cache memory: LRU eviction is bounded by entries but not memory footprint; large filtered DataFrames can accumulate.
- Layout blob versioning: no version header; if we later change schema or decide to make target-agnostic blobs, need explicit versioning.
- Tests: controller behavior not fully validated (stop conditions, convergence under budget, overlap handling, and residual accounting).

## Decisions / Policies
- Training path must set `allow_fallback=False`; on sampling failure, error out with coverage diagnostics.
- Default solver BFS; allow A* variants via CLI; store solver fingerprints in `SolverKey`.
- Dedup strategy: prefer layout-level dedup when we eventually enforce it to allow multi-round use of layouts.

## Recommendations (Actionable)
1. Add `verify_manifest` tool
   - Recompute histograms from Parquet; detect duplicates by `board_hash`; optionally repair manifest; CI gate for demo bank.
2. Improve controller accounting
   - Compute per-spec contributions to levels; avoid double-counting; adopt per-spec `solver_max_depth_override`.
3. Strengthen CLI separation
   - When `--use_controller` is set, do not apply curriculum filters during generation; keep solver overrides only.
4. Sampler metrics & memory
   - Track approximate cached bytes/rows; expose via `monitoring/hub.py`; add a hard cap.
5. Blob versioning
   - Prepend magic + version to `layout_blob`; add round-trip tests and clear mismatch errors.
6. Tests
   - Add e2e tests for controller convergence, budget respect, and overlap cases; add round-trip hash tests for blobs; assert observation shapes and constants.

## Evidence (from code/tests)
- `PuzzleBank.add_puzzles` updates histograms per `(ol, rm)` pair; controller reads from manifest.
- `BankSampler.sample_puzzles_stratified` implements allocation + leftover reallocation; plan exported via `get_last_stratified_plan`.
- `PuzzleGenerator` records `nodes_expanded` and timing; stores `layout_blob` and hashes; RGB rendering fixes landed.

## Follow-ups
- Implement verification utility and wire it into dev workflow.
- Update `generate_curriculum_bank.py` help text to clearly warn against mixing curriculum filters with controller runs.
- Provide a tiny seeded demo bank for CI and documentation examples.
