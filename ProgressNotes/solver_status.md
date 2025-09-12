## Solver Status

- Solver implementation (BFS/A*) unchanged; used for validation and optional solvability checks in env.
- `initial_testing_cells.py` examples updated to use `env.get_board()` for read-only access.
- No new constraints added to solver; it should remain consistent with the environment’s movement rules.

Potential future work:

- Add tests to validate that solver solutions can be replayed in the updated environment with spec-driven walls.
- Optionally enhance solver heuristics aware of structure patterns (not required for current scope).
# Solver Status (Step 1.2)

Two planners are provided: BFS (optimal in moves within limits) and A* (configurable heuristic).

## BFS
- Cost model: one slide = one step. Minimizes number of slides.
- Deterministic neighbor expansion (sorted robot ids).
- Pruning: skip non-moves; optional skip of repeated (robot,direction).
- Metadata: `solve_bfs(return_metadata=True)` or `solve_bfs_with_metadata` returns `SolveResult` with `actions`, `nodes_expanded`, `depth`, `found`.
- Limits: `max_depth` (path-length cutoff), `max_nodes` (expansion cap). Returning `None` (or `found=False`) may indicate cutoffs, not true unsolvability.

## A*
- Heuristic modes:
  - `admissible_zero`: h=0 (optimal, slower).
  - `admissible_one`: h ∈ {0,1} (admissible, slightly more informed).
  - `manhattan_cells`: non-admissible (may be faster but can be suboptimal).
- Metadata: `solve_astar(return_metadata=True)` or `solve_astar_with_metadata` mirrors BFS.

## Helpers
- `apply_actions(board, actions)`: apply a plan to a board.
- `serialize(board)`: visited-state key (robot positions only). Assumes fixed walls/goal/target per search instance; recompute for other boards.

## Usage guidance
- Use BFS or A* with admissible heuristic when you need guaranteed optimal lengths (for curriculum thresholds and evaluation).
- Consider `prioritize_target_first=True` for quicker finds; correctness unchanged.
- If scaling up: represent robot positions as a fixed tuple and share wall arrays to reduce allocations.

## Determinism and env integration
- Both BFS and A* are deterministic given a fixed start board and parameters.
- `ensure_solvable=True` in the env uses BFS to filter boards; the env now reports `level_solvable`, `ensure_solvable_enabled`, and (when enforced) `optimal_length`.

## Tests (solver)
- Optimality on tiny boards (length 1).
- Cutoff behavior (depth=0 -> None).
- Multi-robot prerequisite move.
- A* (admissible) matches BFS optimal plans.
- Zero-length start-on-goal.

## Future extensions
- Add generator utilities to sample boards with `optimal_length <= N` for curriculum.
- Add optional greedy neighbor ordering toward goal delta as a speed-up (correctness unaffected).
