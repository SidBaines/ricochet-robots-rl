# Next Steps

## Step 2 (Training framework and models)
- Integrate with SB3 PPO/A2C:
  - Wrap env with vectorization; set obs_mode and channel order compatible with policy.
  - Baseline CNN policy first (channels_first recommended), then recurrent policy later.
- Curriculum integration:
  - Use `ensure_solvable=True` and `info["optimal_length"]` to bucket episodes.
  - Optionally build a sampler that targets `optimal_length <= N` for staged difficulty.
- Logging & evaluation:
  - Log `success`, `steps`, `optimal_length` (when available), and episode returns.
  - Periodic evaluation using solver for optimality gap on a fixed set.

## Optional solver improvements
- Add a curriculum generator utility: repeatedly sample until a board with `optimal_length <= N` is found; return `(board, length)`.
- Consider BFS parent-pointer reconstruction to count nodes and reduce allocations; return nodes_expanded cleanly.
- Expose a "greedy neighbor ordering" option for speed, without affecting correctness.

## Interpretability scaffolding (Step 5 prep)
- Plan forward hooks in policies to capture activations.
- Define a small dataset writer that records obs, actions, logits, value, hidden states per step for probe training.

## Engineering hygiene
- Add docstrings to env class and public methods specifying info keys (`optimal_length`, `solver_limits`, etc.).
- README training section: how to run PPO baseline with vectorized envs.
- Expand CLI to pick robots and boards by seed; add display of `optimal_length` when known.
