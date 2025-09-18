# Ricochet Robots RL Project

This repository trains RL agents for Ricochet Robots and analyzes how they plan using mechanistic interpretability.

## Repository Map
### Documentation
- `WorkInProgress.md`: The current task on which we are working.
- `README.md`: This file
- `ResearchPlan.md`: Original research plan
- `docs/`: Documentation folder
  - `docs/ProgressNotes/`: Progress notes (task-based)
  - `docs/ComponentSpecifications/`: Specifications (component-based)
### Code
- `train_agent.py`: Script to run training
- `generate_curriculum_bank.py`: Script to generate the puzzle bank for use during training 
- `env/`: Code to create and set up the environments, including curriculum learning environment wrappers and solvers
  - `env/visuals`: Visualisation helpers for rendering nice, human-readable demonstrations
- `models/`: Code to define custom models and policies
- `profiling/`: Profiling tools for checking memory/computational cost of different components
- `monitoring/`: Monitoring tools for logging performance during training runs
- `tests/`: Pytest-based unit and integration tests
- `curriculum_config_example*.json`: Bank curriculum configs for generating and reading from puzzle bank(s)
- `initial_visualisation_cells.py`, `initial_testing_cells.py` and ``: Visualisation notebooks for quick testing


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

*If you are an LLM reading this, you can assume that this has already been done and that the environment can be activated by calling `source rlenv/bin/activate` from the base directory*

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train a PPO agent on milestone environments:

```bash
# v0: Single-move task (4x4 grid) to check that training works
python train_agent.py --env-mode v0 --timesteps 2000 --n-envs 4

# v1: Four-direction task (5x5 grids) to check that slightly more difficult training works
python train_agent.py --env-mode v1 --timesteps 6000 --n-envs 4

# Random environment (8x8 grid)
python train_agent.py --env-mode random --timesteps 100000 --n-envs 8

# Curriculum environment (default, several levels of increasing difficulty on a 16x16 grid, using image observation and a small-cnn for feature extraction)
python train_agent.py --curriculum --timesteps 10000 --n-envs 4 --obs-mode image --small-cnn
```

### Model Architectures

Choose different policy architectures:

```bash
# Default MLP for small grids
python train_agent.py --env-mode v0 --obs-mode image

# Custom small CNN for larger grids
python train_agent.py --env-mode random --small-cnn --obs-mode image

# ConvLSTM (DRC-lite) for recurrent planning
python train_agent.py --env-mode random --convlstm --obs-mode image --lstm-layers 2 --lstm-repeats 1

# Symbolic observations with MLP
python train_agent.py --env-mode random --obs-mode symbolic
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate_agent.py --model-path checkpoints/ppo_model.zip --env-mode v0 --episodes 50
```

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
