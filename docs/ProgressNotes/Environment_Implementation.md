# Environment_Implementation

## Overview
Canonical Ricochet Robots environment with edge-wall mechanics and a Gymnasium-style API. Deterministic under seeds; supports image and symbolic observations; optional no-op action.

## Core Features
- Edge walls: h_walls (H+1,W), v_walls (H,W+1) with borders
- Sliding dynamics: robots move until blocked by a wall or robot
- Actions: Discrete num_robots*4 (+1 if noop) encoded as (robot_id, direction)
- Termination: goal reached; truncation via max_steps
- Observations:
  - Image: 4 directional wall planes, goal, target-robot mask, one plane per robot; channels-first/last
  - Symbolic: [goal_r, goal_c, target_one_hot, robots_flat]
- No-op: separate noop_penalty from step penalty

## Random Generation & Spec Enforcement
- Constructive generator enforces env/WallSpecifications.md:
  - Forbidden central block (size by parity)
  - Per-quadrant Edge-T juts and Central-Ls with adjacency/placement rules
  - Balanced orientations and counts via edge_t_per_quadrant, central_l_per_quadrant
  - Target placed inside a Central-L (deterministic under seed; safe fallback)
  - Robots/goals never inside the forbidden central block

## Determinism & Seeding
- Base seed stored at init; RNG reinitialized on each reset() unless a new seed is provided
- reset(seed=...) updates base seed; info["episode_seed"] reported

## Rendering
- render_mode="ascii": returns string
- render_mode="rgb": returns RGB array; profile hooks for draw steps; configurable style

## API & Info Schema
- Gymnasium: reset(seed=...) -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info)
- info keys: is_success, TimeLimit.truncated, level_solvable, ensure_solvable_enabled, optimal_length (when enforced), solver_limits, channel_names (image)
- Public get_board() returns safe clone

## Solvability Integration
- ensure_solvable=True filters boards via BFS and reports optimal_length

## Tests
- Sliding/stopping, goal termination, truncation, multi-robot blocking
- Observation shapes/semantics for image and symbolic; channels-first/last
- No-op semantics; env–solver roundtrip

## Notes & Limitations
- Pure-Python env may bottleneck; profile/optimize if needed
- Wall-channel caching assumes walls immutable mid-episode
- Avoid double truncation if externally TimeLimit-wrapped
