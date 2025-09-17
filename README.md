# Ricochet Robots RL Training

This project implements reinforcement learning agents for the puzzle game Ricochet Robots, with a focus on mechanistic interpretability analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train a PPO agent on milestone environments:

```bash
# v0: Single-move task (4x4 grid)
python train_agent.py --env-mode v0 --timesteps 2000 --n-envs 4

# v1: Four-direction task (5x5 grids)
python train_agent.py --env-mode v1 --timesteps 6000 --n-envs 4

# Random environment (8x8 grid)
python train_agent.py --env-mode random --timesteps 100000 --n-envs 8
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

## Next Steps

This completes Step 2 of the research plan. The next phase involves:
- Training agents on more complex environments
- Mechanistic interpretability analysis
- Probing internal representations
- Causal intervention experiments