## Next Steps

Recommended follow-ups for Stage 1.4 and beyond:

1) Add unit tests for spec enforcement:
   - Central block placement (odd/even boards).
   - Edge-T and Central-L counts per quadrant with balanced orientations.
   - No Central-L on edge cells.
   - Corner adjacency rule for Edge-T juts.
   - Neighbor-of-walled-cell cannot be a structure center.
   - Goal placed in a Central-L cell; robots/goals never in central forbidden block.
   - Reproducibility: repeated `reset()` with same seed yields identical boards.

2) Visual sanity checks:
   - Extend `initial_testing_cells.py` to save RGB renders to disk for a few seeds and sizes.
   - Optionally add overlays to indicate recognized structures during debugging.

3) Performance:
   - Profile `_generate_random_board` for large boards; consider early pruning if placements struggle due to constraints.
   - If needed, allow retries within a bounded loop to achieve requested per-quadrant counts, with clear info messages.

4) Curriculum ties (later stages):
   - Expose difficulty knobs (e.g., control counts or orientations mix) via env params for training curricula.

5) Documentation:
   - Mirror key rules from `env/WallSpecifications.md` into docstrings of generator helpers for discoverability.
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
