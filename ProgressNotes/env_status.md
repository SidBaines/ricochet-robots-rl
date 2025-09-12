## Environment Status (Stage 1.4)

Summary of changes implemented during initial testing and spec enforcement:

- Constructive board generator implemented in `env/ricochet_env.py` to satisfy `env/WallSpecifications.md` without generate-and-reject.
  - Boards enforced to be square and include a forbidden central block (1x1 for odd sizes, 2x2 for even) with walls surrounding it.
  - Two structure types per quadrant:
    - Edge-T: single interior edge “jut” off a border, avoiding corners and adhering to adjacency constraints.
    - Central-L: two interior edges forming an L around a cell, with balanced orientations within each quadrant.
  - No cell can host more than one structure. Structures do not contact each other or the central/border walls except where permitted by spec.
  - New configurable counts per quadrant: `edge_t_per_quadrant` and `central_l_per_quadrant` (defaults: 2 if side < 10 else 4).

- Additional constraints enforced:
  - Robots and goal are never placed within the forbidden central block.
  - Central-L cannot be placed around edge cells.
  - If a cell has a non-boundary wall, none of its 4-neighbors can be the center of an Edge-T or a Central-L.
  - Edge-T juts cannot be adjacent to corners (e.g., top/bottom juts skip c in {1, w-1}; left/right juts skip r in {1, h-1}).
  - Target goal must be placed inside a Central-L cell (uniformly at random among eligible centers; deterministic under seed). Fallback to non-central cells only if no Central-Ls were placed (edge case).

- Determinism and seeding:
  - The env now stores a base seed on init and re-seeds RNG on every `reset()` when no new seed is provided. `reset(seed=...)` updates the base seed. This ensures identical boards across resets under the same seed.

- Rendering:
  - New `render_mode="rgb"` returns an RGB uint8 array. Defaults are configurable via `render_rgb_config` (grid lines faint gray, walls thick black, robots colored circles, target as darker star of target robot color).
  - Kept existing `render_mode="ascii"`.

- Public accessors:
  - Added `get_board()` returning a clone of the current `Board` for safe external use (tests/solvers).

Compatibility notes:

- Existing tests that use `FixedLayout` continue to pass (constructive generator is bypassed for fixed layouts).
- Observation spaces unchanged.
- Some prior direct uses of `_board` in the testing script were replaced with `get_board()` to avoid accessing protected members.

# Environment Status (Step 1.1)

Implemented a canonical Ricochet Robots environment with edge-wall mechanics and Gymnasium-style API.

## Core features
- Canonical edge walls: `h_walls (H+1,W)`, `v_walls (H,W+1)` with borders; movement uses edge collisions and robot blocking.
- Sliding dynamics: robots move until blocked by an edge-wall or another robot.
- Termination: episode ends when target robot reaches goal; `terminated` and `truncated` are mutually exclusive.
- Actions: Discrete `num_robots*4 (+1 if noop)` encoded as `(robot_id, direction)`.
- Observation modes:
  - Image: channels-first or channels-last
    - Channels: 4 directional wall planes (up, down, left, right), 1 goal plane, 1 target-robot mask plane, then one plane per robot.
  - Symbolic: vector `[goal_r, goal_c, target_id_one_hot(num_robots), robots_flat(num_robots*2)]`.
- No-op support: separate `noop_penalty` vs `step_penalty`.
- Deterministic expansion/testing: fixed layouts via `FixedLayout`.

## Random generation
- Generates bordered grids, adds interior edge-walls, places robots and goal at free cells.
- `ensure_solvable=True` uses the solver to filter boards; info includes `optimal_length` and `solver_limits`.
  - Note: solver returning None during generation may be due to cutoffs (depth/nodes), not true unsolvability.

## API details
- `RicochetRobotsEnv.reset(ensure_solvable=True)` returns `info` with:
  - `level_solvable` (bool): True when solvability was enforced this episode; False otherwise.
  - `ensure_solvable_enabled` (bool): whether enforcement was enabled for this episode.
  - `optimal_length`: minimal moves from solver (when enabled).
  - `solver_limits`: dict with `max_depth`, `max_nodes` (when enabled).
  - `target_robot`: index of target robot.
  - `episode_seed`: resolved seed for this episode.
  - `channel_names` (image mode): ordered channel labels for logging.

## Rendering
- ASCII renderer shows edge walls and robots; useful for QA and CLI demo.
- Gymnasium-compliant: `render_mode="ascii"` returns a string. Unsupported modes raise a clear error. `render_mode=None` returns `None`.

## Seeding and determinism
- Uses Gymnasium seeding (`self.np_random` via `gymnasium.utils.seeding.np_random`).
- Deterministic given a seed. With `ensure_solvable=True`, retries depend only on RNG draws; the solver is deterministic.

## Tests (env)
- Sliding and stopping on walls and robots.
- Goal termination and reward.
- Multi-robot blocking and `max_steps` truncation.
- Observation variants: channel order and symbolic layout.
- No-op semantics: state unchanged, distinct penalty.
- Solvability integration: `optimal_length` reported; actions applied reach goal.

## Performance
- Wall channels are cached on reset and reused each step for image observations.

## Known limitations / notes
- Image wall encoding chooses directional per-cell planes; this exposes canonical geometry directly to CNNs.
- Random generator may create tight alcoves; rely on `ensure_solvable=True` for training or later add density/connectivity controls.
- Internal time limit is enforced by `max_steps`; wrapping additionally with Gym’s TimeLimit may duplicate truncation signals unless configured carefully.
