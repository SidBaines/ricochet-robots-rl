# TrainingFramework_Implementation

## Overview
Complete SB3 PPO-based training stack with milestone environments, multiple policy architectures (MLP, SmallCNN, ConvLSTM), evaluation, logging, and checkpointing.

## Core
- Stable-Baselines3 PPO (primary); vectorized envs; Gymnasium-compatible
- Comprehensive CLI (20+ options), TensorBoard logging, periodic checkpoints, evaluation tools
- Auto policy selection by grid size/obs mode

## Architectures
- MLP: symbolic or tiny grids
- SmallCNN (models/policies.py): 3×3 convs, GAP, FC; safe for small images
- ConvLSTM (models/convlstm.py): multi-layer ConvLSTM; repeats per timestep (DRC-lite)

## Milestone Environments & Tests
- v0: 4×4 single-move; v1: four 5×5 one-move each direction
- tests/test_training_milestones.py: v0 ≥90% success, v1 ≥80%/direction
- tests/test_policy_outputs.py: shape checks, smoke runs

## Observation Handling
- Channels-first images, normalize_images=False
- Fixed layouts set geometry before building spaces; observation shapes consistent

## CLI Examples
- Basic, architecture selection, evaluation/checkpointing, custom hyperparams

## Notes & Limitations
- ConvLSTM not fully integrated with SB3 recurrent policy; memory-sensitive on long sequences
- Tiny grids auto-fallback to MLP; no-op action available but lightly tested

## Ready for Interpretability
- Checkpoints loadable; activation hooks via custom extractors; evaluation and logs available
