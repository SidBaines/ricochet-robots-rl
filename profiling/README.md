# RGB Training Pipeline Profiling System

This profiling system provides comprehensive performance analysis for the RGB training pipeline, helping identify bottlenecks in environment rendering, model forward passes, and training steps.

## Features

- **Detailed Timing**: Track execution time for individual operations
- **Memory Monitoring**: Monitor CPU and GPU memory usage
- **Bottleneck Detection**: Automatically identify performance bottlenecks
- **Category Analysis**: Group operations by function (Environment, Rendering, Model, Training)
- **Visual Reports**: Generate charts and detailed analysis reports
- **Real-time Monitoring**: Print profiling summaries during training

## Quick Start

### 1. Enable Profiling in Training

```bash
# Train with profiling enabled
python train_agent.py --obs-mode rgb_image --convlstm --enable-profiling --timesteps 10000

# Custom profiling settings
python train_agent.py --obs-mode rgb_image --convlstm --enable-profiling \
    --profiling-report my_report.json \
    --profiling-summary-freq 5000
```

### 2. Analyze Profiling Results

```bash
# Generate analysis report
python -m profiling.analyzer --input profiling_report.json --output analysis.txt

# Generate performance charts
python -m profiling.analyzer --input profiling_report.json --charts performance.png

# Export detailed analysis
python -m profiling.analyzer --input profiling_report.json --detailed detailed_analysis.json
```

### 3. Test the Profiling System

```bash
# Run the test script to verify everything works
python test_profiling.py

# Clean up test files after completion
python test_profiling.py --cleanup
```

## Profiled Operations

The system tracks the following operation categories:

### Environment Operations
- `env_rgb_observation_generation`: Complete RGB observation generation
- `env_rgb_rendering`: Board-to-RGB rendering
- `env_rgb_resizing`: PIL image resizing to fixed dimensions
- `env_rgb_draw_grid`: Drawing grid lines
- `env_rgb_draw_walls`: Drawing wall elements
- `env_rgb_draw_robots`: Drawing robot circles
- `env_rgb_draw_target`: Drawing target star

### Model Operations
- `convlstm_forward_pass`: Complete ConvLSTM forward pass
- `convlstm_boundary_padding`: Adding boundary padding channel
- `convlstm_conv_encoder`: Convolutional encoding layers
- `convlstm_skip_connection`: Skip connection processing
- `convlstm_layers`: ConvLSTM layer processing
- `convlstm_global_pooling`: Global average pooling
- `convlstm_projection`: Final feature projection

### Policy Operations
- `recurrent_policy_forward`: Complete policy forward pass
- `recurrent_policy_get_hidden_states`: Hidden state management
- `recurrent_policy_action_value`: Action and value computation
- `recurrent_policy_action_dist`: Action distribution creation
- `recurrent_policy_action_sampling`: Action sampling

### Training Operations
- `training_total`: Complete training loop
- `simulated_training_step`: Individual training steps
- `simulated_data_collection`: Data collection phase
- `simulated_model_update`: Model parameter updates

## Understanding the Output

### Profiling Summary

The profiling summary shows:
- **Operation**: Name of the profiled operation
- **Calls**: Number of times the operation was called
- **Total(s)**: Total time spent in the operation
- **Avg(s)**: Average time per call
- **Min(s)**: Minimum time for a single call
- **Max(s)**: Maximum time for a single call
- **Mem Peak(MB)**: Peak memory usage during the operation
- **GPU Peak(MB)**: Peak GPU memory usage during the operation

### Bottleneck Analysis

The analyzer identifies bottlenecks using a weighted score:
- **Time Percentage (40%)**: How much of total time the operation takes
- **Per-call Time (30%)**: Average time per individual call
- **CPU Memory (15%)**: Peak CPU memory usage
- **GPU Memory (15%)**: Peak GPU memory usage

### Recommendations

The system provides specific recommendations:
- **CRITICAL**: Operations taking >30% of total time
- **SLOW**: Operations averaging >100ms per call
- **MEMORY**: Operations using >100MB peak memory
- **GPU MEMORY**: Operations using >100MB peak GPU memory

## Example Analysis

```
TOP 10 BOTTLENECKS:
------------------------------------------------------------
Operation                      Score    Time%   Avg(ms)   Calls    
------------------------------------------------------------
env_rgb_observation_generation 45.2     25.3    12.5      2000     
convlstm_forward_pass          38.7     20.1    8.2       2000     
env_rgb_rendering              32.1     15.8    7.9       2000     
convlstm_layers                28.4     12.5    6.2       2000     
env_rgb_resizing               25.6     8.9     4.5       2000     
```

This shows that RGB observation generation is the biggest bottleneck, taking 25.3% of total time.

## Custom Profiling

### Adding Custom Profiling

```python
from profiling import profile, profile_function

# Context manager
with profile("my_operation", track_memory=True):
    # Your code here
    pass

# Function decorator
@profile_function("my_function", track_memory=True)
def my_function():
    # Your code here
    pass
```

### Accessing Profiler

```python
from profiling import get_profiler

profiler = get_profiler()
profiler.enable()  # Enable profiling
profiler.disable()  # Disable profiling
profiler.reset()  # Reset all statistics
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'profiling'**
   - Make sure you're running from the project root directory
   - Check that the profiling package is properly installed

2. **Profiling data shows all zeros**
   - Ensure profiling is enabled with `--enable-profiling`
   - Check that operations are being called (minimum call threshold)

3. **Memory tracking not working**
   - GPU memory tracking requires CUDA
   - CPU memory tracking requires psutil

### Performance Impact

The profiling system has minimal performance impact:
- Timing overhead: ~0.1ms per operation
- Memory overhead: ~1MB for statistics storage
- GPU overhead: Negligible (only memory tracking)

## File Structure

```
profiling/
├── __init__.py          # Package initialization
├── profiler.py          # Core profiling functionality
├── analyzer.py          # Analysis and reporting tools
└── README.md           # This file
```

## Dependencies

- `psutil`: For memory monitoring
- `matplotlib`: For performance charts
- `numpy`: For numerical operations
- `torch`: For GPU memory tracking (optional)

## Integration with Research Plan

This profiling system supports the research plan by:

1. **Step 2.4**: Single-GPU considerations - identify GPU memory bottlenecks
2. **Step 3.2**: Monitor learning curves - track training performance
3. **Step 5.1**: Extract activations - profile model forward passes
4. **Step 6.1**: Code organization - modular profiling components

The system is designed to be flexible and extensible for future research needs.
