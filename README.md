# Ricochet Robots RL Environment (Step 1)

This repository contains a custom Gymnasium-compatible environment for the Ricochet Robots puzzle, a BFS solver for validation/curriculum, and unit tests.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

## Usage Example

```python
from env.ricochet_env import RicochetRobotsEnv

env = RicochetRobotsEnv(height=8, width=8, num_robots=2, render_mode="ascii")
obs, info = env.reset(seed=123)
action = 0  # robot 0 up
obs, reward, terminated, truncated, info = env.step(action)
print(env.render())
```

### Gymnasium Registration (optional)

To create the env via gym.make:

```python
import gymnasium as gym
from env.ricochet_env import register_env

register_env("RicochetRobots-v0", max_episode_steps=100)
env = gym.make("RicochetRobots-v0", height=8, width=8, num_robots=2, render_mode="ascii")
```

### Notes on API compliance

- Seeding follows Gymnasium: use `reset(seed=...)`. The used seed is provided in `info["episode_seed"]`.
- Rendering uses `render_mode`. With `render_mode="ascii"`, `render()` returns a string. Other modes are not implemented and will raise a clear error.
- The `info` dict includes `is_success` each step, and `TimeLimit.truncated` when the episode hits `max_steps`.
- The environment already enforces a time limit via `max_steps`; avoid wrapping with `TimeLimit` unless you disable the internal cap.

Additional info keys on reset:
- `level_solvable` (bool): True if solvability was enforced this episode; False otherwise.
- `ensure_solvable_enabled` (bool): whether solvability enforcement was enabled.
- `channel_names` (image mode): the list of observation channel names for logging.
- Invalid actions raise `gymnasium.error.InvalidAction` when Gymnasium is installed, or `ValueError` otherwise.

### Import path

Ensure the project root is on `PYTHONPATH` (e.g., run from repo root or `pip install -e .`) so the entry point `env.ricochet_env:RicochetRobotsEnv` is importable when registering.

### Observation scaling

- Image observations are float32 in [0,1].
- Symbolic observations use absolute grid indices (float32); downstream code may normalize to [0,1].
