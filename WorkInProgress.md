# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly.
In particular, I amnoticing that the logging isn't quite working as I'd hoped:
1) Wandb doesnt ever seem to get anything logged to evals other than trajectories. I'd like the rest of the eval metrics to be logged there too
2) The trajectories that are logged don't appear to be good. They are appearing as images which show a gradient from black at the top to blue at the bottom. I don't know what is happening but I'd like you to investigate and try and make sure we're logging videos of rollouts (ideally we keep a set of centralised rendering/visualisation tools; I think this already exists)

## Progress 2025-09-17
- Reviewed the WandB integration and discovered that SB3's `EvalCallback` results were never forwarded to the monitoring hub, so WandB only saw the manual trajectory artifacts.
- Added a logger writer (`_HubEvalMetricWriter`) that forwards any `eval/` scalars emitted by SB3 into the monitoring hub so WandB and TensorBoard record the deterministic evaluation metrics alongside the trajectory media.
- Reworked `TrajectoryRecorder` to prefer the shared matplotlib-based renderer when capturing frames (with a safe fallback to env renders) so rollout videos come from the central visualisation pipeline instead of raw env captures that produced the blue gradient artifacts.
- Updated the periodic training/eval trajectory callbacks to reuse the renderer, and verified the modified modules compile cleanly (`python -m compileall monitoring/evaluators.py train_agent.py`).
- Added a lightweight callback bridge that reads `EvalCallback`'s stored statistics and forwards mean reward/length (and success rate when available) straight to the monitoring hub so WandB always receives the deterministic eval scalars even if the SB3 logger integration changes.

## Next steps
- Run a short training/eval cycle with `--monitor-wandb` to confirm eval scalars now appear in the WandB run and the logged videos show the rendered board.
- If the WandB run still misses metrics, inspect the logger output to ensure the new writer executes (add temporary debug logging if needed).
- Tweak renderer cell size or FPS if the new videos need visual adjustments once confirmed in WandB.
