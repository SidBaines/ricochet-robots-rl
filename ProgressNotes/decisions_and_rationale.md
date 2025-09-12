## Decisions and Rationale

1) Constructive generation over rejection sampling:
   - Rationale: Many constraints in `WallSpecifications.md` make naive generate-and-filter expensive. We added structure-aware placement to build valid boards directly while preserving seed determinism.

2) Seed determinism contract (Step 1.3):
   - Decision: Store a base seed at init and reinitialize RNG on each `reset()` when no new seed is passed. This makes repeated resets deterministic under a given seed and updates determinism when a new seed is provided.

3) Structure separation and adjacency rules:
   - Implemented checks that avoid Central-L on edges, avoid corner-adjacent Edge-T juts, and prevent neighbors of walled cells from being structure centers. These ensure cleaner puzzles and avoid overly congested local structures.

4) Goal placement inside Central-L:
   - To focus gameplay around intentionally structured features, the target is now placed in a Central-L cell, randomized under RNG for variety, with a safe fallback.

5) Public `get_board()` accessor:
   - Avoids external code relying on `_board` internals; returns a clone to preserve env invariants.

6) RGB rendering:
   - Added basic but configurable drawer to aid debugging and qualitative checks; kept ASCII for quick console output.

Alternatives considered:
   - Rejection sampling with solver checks was considered but rejected for performance reasons.
   - Storing full placement plans per quadrant was deemed unnecessary; local checks suffice for the current spec.
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
- `level_solvable` is now a strict boolean with companion `ensure_solvable_enabled` to avoid tri-state ambiguity.

## Testing philosophy
- Tests target both mechanics (movement, stopping, truncation) and solver properties (optimality, cutoff, multi-robot needs), plus observation shapes/semantics and env-solver roundtrips.

## Performance considerations (future)
- Walls are immutable; keep them shared across states. Robot positions can be a small tuple for faster hashing.
- Parent-pointer path reconstruction is used in A*; consider for BFS if we need to reduce allocations and capture metadata more cheaply.

## Gymnasium compliance (Step 1.3)
- Adopted Gymnasium seeding (`self.np_random`), `reset(seed=...) -> (obs, info)`, and five-return `step` signature.
- Added `render_mode` with only `"ascii"` supported; `render()` returns a string frame, matching the configured mode.
- Provided `close()` and legacy `seed()`; invalid actions raise `gymnasium.error.InvalidAction` when available.
- Symbolic observation space bounds are finite and match board indices; image obs are float32 in [0,1].
- Registration helper included; supports `max_episode_steps` but warns about duplicate time limits.
