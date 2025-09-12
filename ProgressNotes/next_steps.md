# Next Steps

## Step 2 (Training framework and models)
- Integrate with SB3 PPO/A2C:
  - Wrap env with vectorization; set obs_mode and channel order compatible with policy.
  - Baseline CNN policy first (channels_first recommended), then recurrent policy later.
- Curriculum integration:
  - Use `ensure_solvable=True` and `info["optimal_length"]` to bucket episodes.
  - Optionally build a sampler that targets `optimal_length <= N` for staged difficulty.
- Logging & evaluation:
  - Log `is_success`, `steps`, `optimal_length` (when available), and episode returns.
  - Periodic evaluation using solver for optimality gap on a fixed set.

## Optional solver improvements
- Add a curriculum generator utility: repeatedly sample until a board with `optimal_length <= N` is found; return `(board, length)`.
- Consider BFS parent-pointer reconstruction to count nodes and reduce allocations; return nodes_expanded cleanly.
- Expose a "greedy neighbor ordering" option for speed, without affecting correctness.

## Interpretability scaffolding (Step 5 prep)
- Plan forward hooks in policies to capture activations.
- Define a small dataset writer that records obs, actions, logits, value, hidden states per step for probe training.

## Engineering hygiene & docs
- Add training README section: how to run PPO baseline with vectorized envs; include typical hyperparameters.
- Consider a tiny `train_agent.py` script to smoke test PPO on trivial configs.
- Package the repo (pyproject/SETUP) so users can `pip install -e .` and resolve registration entry point robustly.
- If adding RGB render later, document return type/shape and update metadata.

## Backlog/polish
- Optional: add a flag to normalize symbolic observations to [0,1] to match common pipelines.
- Optional: export typed constants for directions in `ricochet_env` mirroring `ricochet_core` for documentation clarity.
- Optional: expose board-density parameters for random generation (e.g., interior wall count bounds).
