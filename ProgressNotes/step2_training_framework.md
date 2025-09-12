# Step 2: RL Training Framework & Models - Implementation Complete

## Overview
Successfully implemented a complete RL training framework for Ricochet Robots with multiple policy architectures, milestone testing environments, and comprehensive tooling for Step 3 (interpretability analysis).

## What Was Implemented

### Core Framework
- **Stable-Baselines3 PPO** as primary training algorithm
- **Vectorized environments** with proper seeding and Gymnasium compatibility
- **Comprehensive CLI** with 20+ options for environment, model, and training configuration
- **Logging & monitoring** via TensorBoard with success rate tracking
- **Checkpointing system** with periodic saves and best-model selection

### Policy Architectures
1. **SmallCNN** (`models/policies.py`): Custom 3×3 conv extractor safe for tiny grids
   - Uses padding=1 to preserve spatial dimensions
   - Global average pooling + FC projection
   - Suitable for 4×4+ grids, avoids NatureCNN kernel size issues

2. **MLP** (default): For symbolic observations and tiny grid fallback
   - Standard feedforward network with configurable architecture
   - Works with both image and symbolic observation modes

3. **ConvLSTM** (`models/convlstm.py`): DRC-lite recurrent architecture
   - Custom ConvLSTMCell implementation
   - Multi-layer ConvLSTM with configurable repeats per timestep
   - Designed for spatial planning and memory

### Training Infrastructure
- **`train_agent.py`**: Main training script with full CLI
- **`evaluate_agent.py`**: Greedy evaluation with success metrics
- **Policy selection logic**: Auto-chooses appropriate architecture based on grid size
- **Callback system**: EvalCallback and CheckpointCallback integration

### Milestone Testing Environments
- **v0** (`env/fixed_layout_v0_one_move()`): 4×4 grid, single RIGHT move to goal
- **v1** (`env/fixed_layouts_v1_four_targets()`): Four 5×5 grids, one move in each direction
- **Tests** (`tests/test_training_milestones.py`): Validates learning on both milestones

### Testing Suite
- **Policy output tests** (`tests/test_policy_outputs.py`): Shape validation and smoke runs
- **Milestone learning tests**: v0 (≥90% success), v1 (≥80% per direction)
- **Architecture compatibility**: All policy types work with both observation modes

## Key Design Decisions

### Architecture Selection Logic
```python
# Tiny grids (<8) → MLP to avoid CNN kernel issues
# Larger grids + image → CnnPolicy or custom extractors
# Symbolic obs → MLP always
```

### Observation Handling
- **Channels-first** for SB3 compatibility (`channels_first=True`)
- **Image normalization disabled** (`normalize_images=False`)
- **Dynamic policy selection** based on grid size and observation mode

### Milestone Strategy
- **Tests-first approach**: Wrote tests before implementation
- **Deterministic layouts**: Fixed boards for reproducible learning validation
- **Progressive complexity**: v0 (trivial) → v1 (slightly harder) → random (full complexity)

## CLI Usage Examples

### Basic Training
```bash
# Milestone v0 (single move)
python train_agent.py --env-mode v0 --timesteps 2000 --n-envs 4

# Milestone v1 (four directions)  
python train_agent.py --env-mode v1 --timesteps 6000 --n-envs 4

# Random environment
python train_agent.py --env-mode random --timesteps 100000 --n-envs 8
```

### Architecture Selection
```bash
# Small CNN for larger grids
python train_agent.py --env-mode random --small-cnn --obs-mode image

# ConvLSTM for recurrent planning
python train_agent.py --env-mode random --convlstm --lstm-layers 2 --lstm-repeats 1

# Symbolic observations
python train_agent.py --env-mode random --obs-mode symbolic
```

### Advanced Options
```bash
# With evaluation and checkpointing
python train_agent.py --env-mode random --eval-freq 10000 --save-freq 50000

# Custom hyperparameters
python train_agent.py --env-mode random --lr 1e-4 --n-steps 256 --batch-size 512
```

## File Structure
```
├── train_agent.py              # Main training script
├── evaluate_agent.py           # Evaluation script
├── models/
│   ├── policies.py             # SmallCNN feature extractor
│   └── convlstm.py             # ConvLSTM implementation
├── tests/
│   ├── test_training_milestones.py  # v0/v1 learning tests
│   └── test_policy_outputs.py       # Policy shape/smoke tests
├── env/
│   └── __init__.py             # Milestone layout constructors
└── README.md                   # Complete usage documentation
```

## Validation Results
- ✅ v0 milestone: PPO learns single-move policy (≥90% success)
- ✅ v1 milestone: PPO learns direction selection (≥80% per direction)
- ✅ Policy outputs: Correct shapes for all architectures
- ✅ Smoke tests: Brief training runs complete without errors
- ✅ CLI: All options work as expected

## Integration with Step 1
- **Environment compatibility**: Works with both image and symbolic observation modes
- **Fixed layouts**: Leverages `FixedLayout` for deterministic milestone testing
- **Solver integration**: Ready for curriculum learning and optimality analysis
- **Observation spaces**: Properly aligned with environment geometry

## Ready for Step 3
The framework provides everything needed for mechanistic interpretability analysis:
- **Trained models** can be saved and loaded
- **Multiple architectures** for comparison (CNN, MLP, ConvLSTM)
- **Evaluation tools** for performance measurement
- **Logging infrastructure** for monitoring training progress

## Potential Issues & Notes

### ConvLSTM Limitations
- **Hidden state management**: Current implementation doesn't integrate with SB3's recurrent policy system
- **Memory usage**: ConvLSTM may be memory-intensive for long sequences
- **Training stability**: Recurrent policies can be harder to train than feedforward

### Environment Considerations
- **Tiny grids**: 4×4 grids automatically use MLP to avoid CNN kernel size issues
- **Observation modes**: Image mode requires `channels_first=True` for SB3 compatibility
- **No-op actions**: Available but not extensively tested in training

### Future Enhancements
- **Curriculum learning**: Callback system ready for difficulty progression
- **Recurrent PPO**: Could integrate with SB3-contrib's RecurrentPPO
- **Custom policies**: Framework supports additional architectures

## Next Steps for Step 3
1. **Train agents** on complex environments using the established framework
2. **Implement interpretability tools** (probes, saliency maps, activation analysis)
3. **Conduct causal interventions** on trained models
4. **Analyze planning behavior** in ConvLSTM architectures

The training framework is complete and ready for the interpretability phase of the research.
