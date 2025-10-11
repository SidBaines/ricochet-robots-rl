# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly.
In particular, I want to make sure that if a training run is paused (or even just ends), we can continue the training. 
I want to be able to do this in two different ways (on top of the default):
- Start training again, with randomly initialised model (default)
- Continue training, from same step/with same parameters, at same level, etc.
- Restart training (ie start from level 0/learning rate etc.), but by loading in a pre-trained model
This should be configurable in the run yaml file, with a flag indicating which mode we run in and then different blocks indicating the relevant params for the chosen case.
In the 'continue training' case, this should keep track of the learning rate & any other relevant params (I think there are methods in pytorch to do this? Or maybe I'm thinking of Hugging Face? Basically, use module if it makes sense/is convenient to do so)

## 2025-09-11
- Reviewed `train_agent.py` flow: currently only supports fresh training or loading a checkpoint via `--load_path`, always calling `model.learn(..., reset_num_timesteps=True)` implicitly. No explicit handling of resume vs warm-start modes.
- Checked `configs/train_defaults.yaml` to understand existing configuration surface. No fields yet for resume modes.
- Initial thought: introduce a `training_mode` flag with options like `fresh`, `resume`, `warmstart`, plus nested config for each. Need to ensure SB3 scheduler/state is preserved for resume (combine `model.load` with `reset_num_timesteps=False`) and reinitialised for warmstart (likely instantiate fresh model then call `set_parameters`).

- Implemented new initialisation pipeline:
  - Added CLI/config args (`init_mode`, `resume_*`, `warmstart_*`, `vecnorm_load_path`) and `initialization` YAML block parsing helper (`apply_initialization_overrides`).
  - Replaced `--load-path` handling with structured resume/warmstart flows, including validation and informative prints.
  - Enhanced VecNormalize loading to honour explicit paths and fail fast when a resume-specific stats file is missing.
  - Computed `learn_timesteps`/`reset_num_timesteps` based on mode (additional vs target total semantics) and wired into training loop.
  - Added warmstart parameter loading via `model.set_parameters` while keeping optimiser fresh.
- Updated `README.md` with resume/warmstart guidance and example `initialization` blocks.

### Testing
- `python -m compileall train_agent.py` (quick syntax validation)

### Uncertainties / Questions
- Need to confirm whether VecNormalize stats should also be restored during resume/warmstart. Likely yes for resume; warmstart might optionally load.
- Check existing checkpoint layout—does `CheckpointCallback` already drop `.zip` plus `.pkl` for VecNormalize? Need to validate to ensure we can locate resumable assets reliably.
- Should we expose additional convenience helpers for computing remaining timesteps (e.g. auto-detect from checkpoint metadata) beyond the current target-total vs additional toggle?

### Next steps
- Verify on a small dry-run that resume + warmstart behave as expected once a checkpoint is available (requires saved checkpoint to test end-to-end).
- Confirm whether we should persist any metadata about the init mode in saved config/logs (for auditability).
