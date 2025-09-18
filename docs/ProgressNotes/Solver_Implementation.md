# Solver_Implementation

## Overview
BFS (optimal within limits) and A* solvers used for validation, curriculum metadata, and optional solvability filtering during environment reset.

## BFS
- Cost: one slide = one step (minimizes slides)
- Deterministic neighbor expansion (sorted robot ids)
- Pruning: skip non-moves; optional repeated (robot, direction) suppression
- Metadata result (actions, nodes_expanded, depth, found)
- Limits: max_depth, max_nodes; hitting limits may return None/unsolved

## A*
- Heuristics:
  - admissible_zero (h=0)
  - admissible_one (h ∈ {0,1})
  - manhattan_cells (non-admissible; faster, not guaranteed optimal)
- Metadata mirrors BFS

## Helpers
- apply_actions(board, actions): replay a plan
- serialize(board): hashable visited-state key over robot positions

## Determinism & Integration
- Deterministic given fixed board and parameters
- Env integration: ensure_solvable=True uses BFS; env reports level_solvable, ensure_solvable_enabled, optimal_length

## Tests
- Optimality on tiny boards
- Cutoff behavior
- Multi-robot prerequisite moves
- A* (admissible) matches BFS optimal plans
- Zero-length start-on-goal

## Notes
- Prefer BFS/A* (admissible) when curriculum thresholds require optimal lengths
- Potential speedups: greedy neighbor ordering without changing correctness
