# Decisions and Rationale

## Canonical edge walls
- We refactored to edge-wall representation (`h_walls`, `v_walls`) to match Ricochet Robots rules and literature, ensuring correct sliding and stopping behavior and comparability.

## Observation encoding
- Image mode includes four directional wall channels so CNNs can directly learn canonical geometry. This is preferable to a single wall mask because directionality matters for planning.
- Target identity exposed via a per-cell target mask. Symbolic mode includes a target one-hot.
- Channel order toggle (channels_first) added for RL-library compatibility.

## No-op action and pacing
- No-op included to allow explicit “thinking” steps; separate `noop_penalty` enables studying pacing behavior without conflating with move penalty.

## Solver choices
- BFS is the baseline optimal planner in number of slides.
- A* added with selectable heuristic modes: admissible defaults for optimality; non-admissible Manhattan option documented for speed experiments.
- Deterministic neighbor ordering simplifies reproducibility and profiling.

## Env-solvability integration
- `ensure_solvable=True` uses solver filtering; `optimal_length` and solver limits provided in `info` for curriculum.
- Fixed layouts set `level_solvable=None` (unknown), avoiding misleading flags.

## Testing philosophy
- Tests target both mechanics (movement, stopping, truncation) and solver properties (optimality, cutoff, multi-robot needs), plus observation shapes/semantics and env-solver roundtrips.

## Performance considerations (future)
- Walls are immutable; keep them shared across states. Robot positions can be a small tuple for faster hashing.
- Parent-pointer path reconstruction is used in A*; consider for BFS if we need to reduce allocations and capture metadata more cheaply.
