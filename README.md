# Ricochet Robots RL Project

This repository trains RL agents for Ricochet Robots and analyzes how they plan using mechanistic interpretability.

## Quick start
#### Install
```
python -m venv .env
source .env
```
#### Populate curriculum bank
```
python generate_curriculum_bank.py --config configs/curriculum_config_default.json --bank_dir ./puzzle_bank
```
#### Run training
```
python train_agent.py configs/train_defaults.yaml
```


## Repository Map
### Documentation
- `WorkInProgress.md`: The current task on which we are working.
- `README.md`: This file
- `ResearchPlan.md`: Original research plan
- `docs/`: Documentation folder
  - `docs/ProgressNotes/`: Progress notes (task-based)
  - `docs/ComponentSpecifications/`: Specifications (component-based)
### Code
- `train_agent.py`: Script to run training
- `generate_curriculum_bank.py`: Script to generate the puzzle bank for use during training 
- `env/`: Code to create and set up the environments, including curriculum learning environment wrappers and solvers
  - `env/visuals`: Visualisation helpers for rendering nice, human-readable demonstrations
- `models/`: Code to define custom models and policies
- `profiling/`: Profiling tools for checking memory/computational cost of different components
- `monitoring/`: Monitoring tools for logging performance during training runs
- `tests/`: Pytest-based unit and integration tests
- `curriculum_config_example*.json`: Bank curriculum configs for generating and reading from puzzle bank(s)
- `initial_visualisation_cells.py`, `initial_testing_cells.py`: Visualisation notebooks for quick testing


## Current status

This section gives a high-level overview of the current status of the various stages of the project. The steps are laid out in rough chronological order, but they need not be executed in exactly this order.
More details on any given step, if they are present, will be in `docs/ProgressNotes`.
Recall that the task we're currently working on, if any, can be found in `WorkInProgress.md`.

1) Environment & Generator (complete)
   - Canonical edge-wall mechanics, sliding dynamics, Gymnasium API; deterministic seeding; ASCII/RGB rendering; optional no-op action.

2) Solver & Bank Precomputation (complete)
   - BFS/A* solvers for optimal lengths; offline puzzle bank with metadata (optimal_length, robots_moved, etc.); fast curriculum sampling without online solving.

3) Training Framework (complete)
   - SB3 PPO with vectorized envs, CLI, logging, checkpoints; policies: MLP (symbolic/tiny), SmallCNN (image), ConvLSTM (DRC-lite prototype).

4) Curriculum Training & Evaluation (in progress)
   - Run staged curricula on fixed-size RGB observations; evaluate success rate and optimality gap vs solver on held-out sets; produce rollouts/videos.

5) Mechanistic Interpretability Suite (planned)
   - Activation capture, linear probes, saliency/attribution, feature visualization, activation patching; summarize findings on plan representations and causal features.

6) Profiling & Monitoring (complete)
   - Lightweight profiler across env rendering/model forward; monitoring hooks and TensorBoard metrics to track throughput and resource use.

7) Documentation & Repo Hygiene (in progress)
   - Consolidate progress notes under docs, keep `WorkInProgress.md` current, and ensure README examples and references stay aligned with code.


## Installation

If using the provided virtualenv:

```bash
source rlenv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Training

Launch training by pointing the script at a YAML configuration:

```bash
# Run with the default settings from configs/train_defaults.yaml
python train_agent.py configs/train_defaults.yaml

# Copy, edit, and run a custom configuration
cp configs/train_defaults.yaml my_run.yaml
${EDITOR:-nano} my_run.yaml
python train_agent.py my_run.yaml
```

Notes:
- All previous CLI flags are now top-level YAML keys; `configs/train_defaults.yaml`
  documents the defaults for every option.
- Bank curriculum disables online fallback by default; ensure your bank has
  coverage for the requested levels or relax the criteria in your config.

### Model Architectures

Tweak these keys in your YAML config to switch policies:

- `obs_mode`: `symbolic`, `image`, or `rgb_image`
- `small_cnn`: `true` to use the compact CNN extractor for larger grids
- `convlstm`: `true` to enable the sb3-contrib CnnLstmPolicy baseline
- `drc`: `true` to enable the recurrent DRC policy (requires sb3-contrib)
- `resnet`: `true` to swap in the ResNet feature extractor baseline

### Evaluation

Evaluate a trained model:

```bash
python evaluate_agent.py --model-path checkpoints/ppo_model.zip --env-mode v0 --episodes 50
```

### Bank Precomputation (band-first controller)

Generate the curriculum bank using the band-first controller. It solves broadly and stops when each level has enough available puzzles according to bank histograms.

```bash
# Use controller to target 1000 available puzzles per level (approx.)
python generate_curriculum_bank.py \
  --use_controller \
  --target_per_level 1000 \
  --max_puzzles_global 200000 \
  --chunk_per_spec 200 \
  --bank_dir ./puzzle_bank
```

Notes:
- The controller relies on manifest histograms and can be re-run to incrementally top-up without duplicating layouts.
- Default solver is BFS; adjust with `--solver`, `--max_depth`, and `--max_nodes` if needed.

## Environment Options

- `--env-mode`: `random`, `v0`, `v1` (milestone layouts)
- `--height`, `--width`: Grid dimensions (default 8x8)
- `--num-robots`: Number of robots (default 2)
- `--include-noop`: Enable no-op actions for "thinking"
- `--obs-mode`: `image` or `symbolic` observations
- `--ensure-solvable`: Only generate solvable puzzles (slower)

## Training Options

- `--timesteps`: Total training steps
- `--n-envs`: Number of parallel environments
- `--lr`: Learning rate (default 3e-4)
- `--n-steps`: Rollout length (default 128)
- `--batch-size`: Minibatch size (default 256)

## Logging and Checkpoints

- `--log-dir`: TensorBoard log directory
- `--save-path`: Model save path
- `--save-freq`: Checkpoint frequency (timesteps)
- `--eval-freq`: Evaluation frequency (timesteps)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Visualisation

Two offline renderers are available:

- Matplotlib-style raster renderer: `env/visuals/mpl_renderer.py`
- Plotly renderer (PNG export via kaleido): `env/visuals/plotly_renderer.py`

Use `initial_visualisation_cells.py` (with `#%%` cells) to:
- Load the bank curriculum from `puzzle_bank` and `curriculum_config_example2.json`
- Load a checkpointed PPO model
- Record and preview episodes at a few curriculum levels with both renderers

## Architecture Details

### SmallCNN
- 3x3 conv layers with padding=1 (safe for small grids)
- Global average pooling
- Suitable for 4x4+ grids

### ConvLSTM (DRC-lite)
- Convolutional LSTM layers for spatial planning
- Configurable layers and repeats per timestep
- Designed for recurrent planning behavior

### Milestone Environments
- **v0**: 4x4 grid, single RIGHT move to goal
- **v1**: 5x5 grids, one move in each of four directions
