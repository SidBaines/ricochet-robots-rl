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
