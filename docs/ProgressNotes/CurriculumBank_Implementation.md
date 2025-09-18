# Curriculum Bank Redesign — Implementation

## Summary
- Reworked curriculum to use a bank-first, offline precomputation workflow by default.
- Added manifest v1.1 with per-partition histograms keyed by (optimal_length, robots_moved), fingerprints scaffolding, and dedup section.
- Implemented robust replay via compact binary `layout_blob` with round-trip reconstruction; retained regeneration by seed + hash.
- Introduced a band-first precompute controller that solves broadly and stops when each curriculum level has sufficient coverage as measured by histograms.
- Added stratified sampling with quota allocation and leftover reallocation; integrated into curriculum env to avoid online fallback during training by default.

## Key Components and Edits
- `env/puzzle_bank.py`
  - Manifest upgrade to v1.1; track `puzzle_count` and histograms under `histogram.by_optimal_length_and_robots_moved` with keys like `ol=4|rm=2`.
  - `PuzzleMetadata` now carries `layout_blob`, `layout_hash`, `board_seed`, solver metadata (`nodes_expanded`, `solve_time_ms`, limits flags).
  - Added binary hashing for board/layout; added `serialize_layout_to_blob`/`deserialize_layout_blob` and `create_fixed_layout_from_metadata`/`_from_seed`.
  - `BankSampler` now maintains an LRU cache for filtered iterators and supports `sample_puzzles_stratified` with monitoring via `get_last_stratified_plan`.

- `env/precompute_pipeline.py`
  - `PuzzleGenerator` solves with metadata (`solve_bfs_with_metadata`/`solve_astar_with_metadata`), stores layout blob and hashes, and records robots moved.
  - Clarified solver cutoff naming via `solver_max_depth_override` in controller path.
  - `compute_level_coverage_from_histograms` computes per-level availability using manifest histograms.
  - `run_band_first_precompute_controller` iterates in chunks per spec until each level reaches `target_per_level` or global budget is hit.

- `generate_curriculum_bank.py`
  - New flags: `--use_controller`, `--target_per_level`, `--max_puzzles_global`, `--chunk_per_spec`, plus solver options.
  - When using the controller, generation is band-first and curriculum filters are not applied during generation.

- Curriculum integration
  - `CriteriaFilteredEnv` and `BankCurriculumManager` use the bank samplers by default; `allow_fallback` is False by default to prevent silent online solving during training.

## CLI Quickstart
- Band-first controller (recommended):
```
python generate_curriculum_bank.py \
  --use_controller \
  --target_per_level 1000 \
  --max_puzzles_global 200000 \
  --chunk_per_spec 200 \
  --bank_dir ./puzzle_bank
```
- Training with bank curriculum:
```
python train_agent.py --curriculum --use_bank_curriculum --obs-mode image --n-envs 8
```

## Data and Schema Notes
- Manifest format v1.1:
  - `version`, `partitions.{partition_key}.spec_key`, `puzzle_count`, `histogram.by_optimal_length_and_robots_moved`, `fingerprints`, `dedup`.
- Parquet partition path is keyed by `SpecKey.to_partition_key()`.
- Histograms are incrementally updated during `add_puzzles`; a verification tool is recommended for periodic reconciliation.

## Behavioral Changes
- Training defaults to bank-based sampling; online filtered generation is for smoke tests and explicit opt-in only.
- `CriteriaFilteredEnv` raises with guidance if sampling underfills and `allow_fallback=False`.
- RGB renderer fixes ensure consistent visuals across renderers.

## Migration & Backward Compatibility
- Existing banks are upgraded in-memory to manifest v1.1 on load; new fields are backfilled with defaults.
- Banks written prior to layout blobs continue to work via seed regeneration; new runs will add blobs for robust replay.
- Recommend running a one-off verification to detect duplicates and reconcile histograms for legacy data.

## Open Items / Future Work
- Add `verify_manifest` utility to recompute histograms and detect duplicates by `(board_hash)`.
- Consider per-level residual accounting in controller to avoid overestimation with overlapping bands.
- Add approximate byte-size tracking for sampler cache and expose metrics via `monitoring`.
- Optional: bloom filter dedup by `layout_hash` per partition; add blob version header for future-proofing.
