# PuzzleBank_Implementation

## Overview
Offline, precomputed puzzle bank enabling curriculum learning without online solver calls during training. Stores puzzles, solution metadata, and supports efficient sampling by criteria.

## Components
- PuzzleBank (env/puzzle_bank.py): storage, querying, statistics
- SpecKey / SolverKey: generation and solver configurations
- PuzzleMetadata: puzzle data + solution traces
- BankSampler: sampling strategies for training
- Precomputation pipeline (env/precompute_pipeline.py): generators and batch creation
- Criteria filtering (env/criteria_env.py): PuzzleCriteria, CriteriaFilteredEnv, BankCurriculumManager
- Curriculum integration (env/curriculum.py): create_bank_curriculum_levels/manager, BankCurriculumWrapper

## Storage Format
- Parquet files per spec partition; manifest.json summarizing partitions and counts
- Entry schema: identifiers (seed, spec, solver, board_hash), solution metrics (optimal_length, robots_moved, nodes_expanded, solve_time_ms), flags (solved_within_limits, hit_depth_limit, hit_node_limit), plan trace (actions_encoded), optional layout_blob

## Curriculum Levels (example)
1. Level 0: 1 robot, no walls (1–3 moves)
2. Level 1: 1 robot, some walls (2–6 moves)
3. Level 2: Multiple robots, simple (3–8 moves)
4. Level 3: Multiple robots, more walls (4–10 moves)
5. Level 4: Full complexity (5–15 moves)

## Usage
- Precompute: precompute_bank.py with --bank_dir, --num_puzzles, --solver_depth
- Training: load PuzzleBank, create curriculum manager, wrap env with BankCurriculumWrapper (fixed-size RGB, channels_first)
- Direct criteria: CriteriaFilteredEnv with PuzzleCriteria

## Performance & Determinism
- Deterministic seeds and canonical solver configuration
- Fast sampling; no solver calls in the training hot path
- Partitioned storage for efficient filtering; consider compaction for large banks

## Testing
- Run test_bank_system.py and example_bank_usage.py

## Future Enhancements
- Stratified sampling across difficulty bins
- Optional layout blobs and incremental updates
- Distributed generation and advanced criteria
