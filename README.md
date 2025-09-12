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

env = RicochetRobotsEnv(height=8, width=8, num_robots=2)
obs, info = env.reset()
action = 0  # robot 0 up
obs, reward, terminated, truncated, info = env.step(action)
print(env.render())
```
