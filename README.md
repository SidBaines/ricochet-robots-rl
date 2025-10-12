# Ricochet Robots RL Project

This repository trains reinforcement learning agents to solve Ricochet Robots while providing tooling for curriculum generation, monitoring, profiling, and mechanistic interpretability analyses.

## Quick Start

### 1. Set up a Python environment and install dependencies
```bash
python -m venv rlenv
source rlenv/bin/activate
pip install -r requirements.txt
```

### 2. Pre-compute a curriculum puzzle bank
The training defaults expect an on-disk bank housed under `artifacts/puzzle_bank`.
```bash
python src/scripts/generate_curriculum_bank.py \
  --config configs/curriculum_config_default.json \
  --bank_dir artifacts/puzzle_bank
```

### 3. Launch training from a YAML configuration (optional)
```bash
python src/scripts/train_agent.py configs/train_defaults.yaml
```
Copy `configs/train_defaults.yaml`, tweak any keys you need (all CLI options map directly to YAML fields), and pass the new file to the same command for customised runs.

### 4. Observe a models' gameplay
Notebook-style scripts with `#%%` cells live in `src/scripts/`; `bank_agent_rollout_cells.py` is particularly useful for loading a model.
A (badly-performing) pre-trained model exists in `examples/`.

## Repository Layout
- `src/`
  - `src/env/`: Environment core, solvers, curriculum helpers, and visuals
  - `src/models/`: Custom policy implementations (CNN, ConvLSTM/DRC, ResNet extractors)
  - `src/monitoring/`: Logging backends and collectors for metrics/rollouts
  - `src/profiling/`: Lightweight profiling utilities
  - `src/scripts/`: Executable entry points (training, evaluation, bank tooling, visualisers)
  - `src/tests/`: Pytest-based unit and integration tests
- `configs/`: YAML configurations for training profiles and baselines
- `docs/`: Research plans, progress notes, and prompt documentation
- `examples/`: Reference assets such as sample checkpoints/configs
- `artifacts/`: Output directory for generated banks, checkpoints, logs, runs, and wandb exports (created/populated at runtime)
- `WorkInProgress.md`: Rolling log of the active development task

## Key Scripts & Pipelines
- `src/scripts/train_agent.py`: Main training entry point; consumes a YAML config
- `src/scripts/evaluate_agent.py`: Evaluate PPO checkpoints on fixed layouts or random boards
- `src/scripts/generate_curriculum_bank.py`: Build/refresh the curriculum puzzle bank (band-first controller)
- `src/scripts/precompute_bank.py`: Lower-level precomputation driver for bespoke experiments
- `src/scripts/bank_agent_rollout_cells.py`: Jupyter-style curriculum rollout visualiser
- `src/scripts/initial_testing_cells.py`: Smoke-test utilities for the environment and solver

## Configuration & Training Notes
- Every CLI option in `build_training_parser()` is mirrored as a YAML key. Check `configs/train_defaults.yaml` and `configs/resnet_rgb.yaml` for annotated examples.
- `bank_dir`, `log_dir`, and `save_path` defaults now point into `artifacts/` to keep generated data out of the source tree.
- Device selection defaults to `auto`; M-series Macs will pick `mps` when available.
- Recurrent policies (`--convlstm`, `--drc`) and `--resnet` are mutually exclusive. Set only one of the flags (or none for the default SmallCNN).

### Resume and Warmstart
Control run initialisation via YAML:
```yaml
initialization:
  mode: resume
  resume:
    checkpoint_path: artifacts/checkpoints/ppo_model.zip
    target_total_timesteps: 1_000_000
    vecnormalize_path: artifacts/runs/ppo/vecnormalize.pkl
```
For warmstarts swap `mode: warmstart` and provide `params_path`/`vecnormalize_path` instead. The legacy `load_path` is coerced to `init_mode: resume` for backwards compatibility.

## Puzzle Bank Generation & Maintenance
Top up or rebuild the bank with:
```bash
python src/scripts/generate_curriculum_bank.py \
  --config configs/curriculum_config_default.json \
  --bank_dir artifacts/puzzle_bank \
  --target_per_level 1000 \
  --max_puzzles_global 200000 \
  --chunk_per_spec 200
```
The controller consults manifest histograms, making it safe to rerun incrementally without duplicating layouts. Adjust `--solver`, `--max_depth`, and `--max_nodes` to trade off coverage vs computation.

## Testing
Use pytest from the project root (ensure `pytest` is installed in your environment):
```bash
python -m pytest
```

## Visualisation & Rendering
Offline renderers live under `src/env/visuals/`:
- `mpl_renderer.py`: Matplotlib-based raster renderer for quick previews
- `plotly_renderer.py`: Plotly renderer with Kaleido export for high-quality frames

For curated rollouts, open `src/scripts/initial_visualisation_cells.py`. It loads curriculum configs, banks from `artifacts/puzzle_bank`, and checkpoints from `artifacts/checkpoints/` to record or display trajectories with the renderers above.

## Current Status Overview
1. **Environment & Generator (complete)** – Canonical edge-wall mechanics, sliding dynamics, Gymnasium API; deterministic seeding; ASCII/RGB rendering; optional no-op action.
2. **Solver & Bank Precomputation (complete)** – BFS/A* solvers for optimal lengths; offline puzzle bank with metadata; curriculum sampling without online solving.
3. **Training Framework (complete)** – SB3 PPO with vectorised envs, YAML config loader, logging, checkpoints; policies include MLP (symbolic), SmallCNN (image), ConvLSTM (DRC-lite prototype).
4. **Curriculum Training & Evaluation (in progress)** – Run staged curricula on fixed-size RGB observations; evaluate success rate and optimality gap vs solver on held-out sets; produce rollouts/videos.
5. **Mechanistic Interpretability Suite (planned)** – Activation capture, linear probes, saliency/attribution, feature visualisation, activation patching; summarise findings on plan representations and causal features.
6. **Profiling & Monitoring (complete)** – Lightweight profiler across env rendering/model forward; monitoring hooks and TensorBoard/W&B metrics to track throughput and resource use.
7. **Documentation & Repo Hygiene (in progress)** – Keep `WorkInProgress.md` current, ensure README/examples stay aligned with the new `src/` + `artifacts/` structure, and consolidate progress notes under `docs/`.

## Additional Resources
- `docs/ProgressNotes/`: Day-by-day engineering updates
- `docs/ComponentSpecifications/`: Detailed component-level specs
- `docs/AiChatLogs/`: Conversations and decision logs

