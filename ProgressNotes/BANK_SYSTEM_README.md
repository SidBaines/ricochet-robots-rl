# Puzzle Bank System for Curriculum Learning

This document describes the new puzzle bank system that enables efficient curriculum learning with tight criteria constraints by precomputing and storing puzzles with their solution metadata.

## Overview

The puzzle bank system addresses the challenge of providing puzzles that meet specific criteria (board size, optimal solution length, number of robots moved) during training without requiring expensive online solving. Instead, puzzles are precomputed and stored in a lightweight format using seeds and generation specifications.

## Key Components

### 1. Puzzle Bank (`env/puzzle_bank.py`)

The core storage system that manages precomputed puzzles:

- **SpecKey**: Defines generation parameters (board size, robots, wall complexity)
- **SolverKey**: Defines solver configuration for canonicalization
- **PuzzleMetadata**: Stores puzzle data including solution traces
- **PuzzleBank**: Manages storage, querying, and statistics
- **BankSampler**: Provides sampling strategies for training

### 2. Precomputation Pipeline (`env/precompute_pipeline.py`)

Tools for generating and storing puzzles:

- **PuzzleGenerator**: Generates puzzles and their solutions
- **CurriculumSpecGenerator**: Creates curriculum level specifications
- **run_precomputation()**: Main function for batch generation

### 3. Criteria Filtering (`env/criteria_env.py`)

Environment wrapper that uses the bank:

- **PuzzleCriteria**: Defines filtering criteria
- **CriteriaFilteredEnv**: Environment that samples from bank
- **BankCurriculumManager**: Manages curriculum progression

### 4. Curriculum Integration (`env/curriculum.py`)

Updated curriculum system with bank support:

- **create_bank_curriculum_levels()**: Predefined curriculum levels
- **create_bank_curriculum_manager()**: Bank-based curriculum manager
- **BankCurriculumWrapper**: Full curriculum wrapper using bank

## Usage

### 1. Precompute Puzzle Bank

```bash
# Generate puzzles for all curriculum levels
python precompute_bank.py --bank_dir ./puzzle_bank --verbose

# Generate with custom parameters
python precompute_bank.py --bank_dir ./puzzle_bank --num_puzzles 1000 --solver_depth 30
```

### 2. Use Bank in Training

```python
from env.puzzle_bank import PuzzleBank
from env.curriculum import create_bank_curriculum_manager, BankCurriculumWrapper

# Load bank
bank = PuzzleBank("./puzzle_bank")

# Create curriculum manager
manager = create_bank_curriculum_manager(bank)

# Create training environment
env = BankCurriculumWrapper(
    bank=bank,
    curriculum_manager=manager,
    obs_mode="rgb_image",  # Fixed observation shape
    channels_first=True
)

# Use in training loop
obs, info = env.reset()
# ... training code ...
```

### 3. Direct Criteria Filtering

```python
from env.criteria_env import CriteriaFilteredEnv, PuzzleCriteria
from env.puzzle_bank import SpecKey

# Define criteria
criteria = PuzzleCriteria(
    spec_key=SpecKey(height=8, width=8, num_robots=2, edge_t_per_quadrant=1, central_l_per_quadrant=1),
    min_optimal_length=3,
    max_optimal_length=6,
    min_robots_moved=1,
    max_robots_moved=2
)

# Create environment
env = CriteriaFilteredEnv(bank, criteria)
```

## Storage Format

### Parquet-based Storage

Puzzles are stored in Parquet format with the following structure:

```
puzzle_bank/
├── manifest.json                    # Bank metadata and partition info
├── size-8x8_numR1_ET0_CL0_gen10.parquet
├── size-8x8_numR1_ET1_CL1_gen10.parquet
└── size-10x10_numR2_ET1_CL1_gen10.parquet
```

### Metadata Schema

Each puzzle entry contains:

- **Identifiers**: `seed`, `spec_key`, `solver_key`, `board_hash`
- **Solution metrics**: `optimal_length`, `robots_moved`, `nodes_expanded`, `solve_time_ms`
- **Flags**: `solved_within_limits`, `hit_depth_limit`, `hit_node_limit`
- **Plan trace**: `actions_encoded` (for deriving additional features)
- **Optional**: `layout_blob` (for generator drift recovery)

## Curriculum Levels

The system includes predefined curriculum levels:

1. **Level 0**: Single robot, no walls (1-3 moves)
2. **Level 1**: Single robot, some walls (2-6 moves)
3. **Level 2**: Multiple robots, simple (3-8 moves)
4. **Level 3**: Multiple robots, more walls (4-10 moves)
5. **Level 4**: Full complexity (5-15 moves)

## Key Features

### 1. Lightweight Storage

- Puzzles stored as seeds + generation specs
- Solution traces for deriving additional features
- Optional layout blobs for generator drift recovery

### 2. Efficient Querying

- Partitioned storage by generation spec
- Fast filtering by criteria (moves, robots moved)
- Support for both uniform and stratified sampling

### 3. Deterministic and Reproducible

- Canonical solver configuration
- Deterministic board generation from seeds
- Hash validation for integrity

### 4. Extensible

- Easy to add new criteria
- Support for custom curriculum levels
- Pluggable sampling strategies

## Testing

Run the test suite to verify the system works:

```bash
python test_bank_system.py
```

Run the example to see the system in action:

```bash
python example_bank_usage.py
```

## Dependencies

The bank system requires:

- `pandas` >= 2.0
- `pyarrow` >= 10.0 (for Parquet support)
- `tqdm` >= 4.64 (for progress bars)
- `Pillow` >= 9.0 (for RGB observations)

Install with:

```bash
pip install -r requirements.txt
```

## Performance Considerations

### Precomputation

- Generate puzzles in batches for efficiency
- Use appropriate solver limits to balance quality vs speed
- Consider parallel generation for large banks

### Training

- Use `obs_mode="rgb_image"` for consistent observation shapes across board sizes
- Bank sampling is very fast (no solver calls during training)
- Consider caching frequently used puzzles in memory

### Storage

- Parquet provides good compression and fast querying
- Partition by generation spec for efficient filtering
- Consider periodic compaction for large banks

## Future Enhancements

1. **Stratified Sampling**: Implement balanced sampling across difficulty bins
2. **Layout Blob Storage**: Add support for storing board layouts
3. **Incremental Updates**: Support adding new puzzles to existing banks
4. **Distributed Generation**: Parallel puzzle generation across multiple processes
5. **Advanced Criteria**: Support for more complex filtering criteria

## Troubleshooting

### Common Issues

1. **No puzzles found**: Check that bank has puzzles matching your criteria
2. **Import errors**: Ensure all dependencies are installed
3. **Memory issues**: Reduce batch size or use smaller banks
4. **Slow generation**: Adjust solver limits or use fewer puzzles

### Debug Mode

Enable verbose output for debugging:

```python
env = CriteriaFilteredEnv(bank, criteria, verbose=True)
```

This will show detailed information about puzzle selection and any issues encountered.
