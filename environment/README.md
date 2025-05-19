# Ricochet Robots RL

## Structure

- `train.py` contains the training loop
- `inspect_env.py` contains some code for playing around with/inspecting the environment
- `environment/` contains the environment classes, including for board, robot, utils, and the main environment class in `ricochet_env.py`
- `agent/` contains the agent classes
- `tests/` contains some tests


## Ideas/plans
- Current plan: get environment working, get model trained on individual boards/targets, see how it goes
- Potential ideas
    - Long term interesting: train a model to play a FULL game (not just one target on one board, but multiple targets on the same board, with the robots starting where they left when the last target was reached, and for example a single known target for each robot known from the beginning of the episode), and see if the model learns to play the game strategiually by planning multiple *targets* ahead (ie, learn to strategically position robots for future targets, if it saves steps later down the line, even if it costs more steps to get to the current target)