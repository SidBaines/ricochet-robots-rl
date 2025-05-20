# Ricochet Robots RL


## Structure

- `train.py` contains the training loop
- `inspect_env.py` contains some code for playing around with/inspecting the environment
- `environment/` contains the environment classes, including for board, robot, utils, the main base environment class in `ricochet_env.py`, and simpler environments in eg. `simpler_ricochet_env.py`
- `agent/` contains the agent classes. Currently only PPO using CNNs, but should add other agents later.
- `tests/` contains some tests

## TODO
- Fix the bug in the RicochetRobotsEnvOneStepAway environment where the target robot can get stuck if there is no valid move; currently handled by calling reset again, but this could go into an infinite loop if the environment is complicated enough.
- Make more simple environments & check that the model can learn them (eg. target one step away, target two steps away but single robot, etc). Being able to automate this process seems valuable
- Looks like boxes are currently possible (eg in the bottom right if we have sufficient edge-walls and floating walls allowed can get a 2x2 box), seems like maybe a bug in the edge wall placement?
- Make sure that the seed allows a fixed board, robot placement, target placement, etc (currently at least some of it seems to be random)

## Ideas/plans
- Current plan: get environment working, get model trained on individual boards/targets, see how it goes
- Later: do interp work (look at models internal representations, use linear probes, etc.)
- Potential ideas
    - Long term interesting: train a model to play a FULL game (not just one target on one board, but multiple targets on the same board, with the robots starting where they left when the last target was reached, and for example a single known target for each robot known from the beginning of the episode), and see if the model learns to play the game strategiually by planning multiple *targets* ahead (ie, learn to strategically position robots for future targets, if it saves steps later down the line, even if it costs more steps to get to the current target)
    - Could try encoding the inputs not as the [num_robots + 1 + 4, height,  width] size tensor that we currently do, and instead as a [3, 256, 256] picture, with walls and grid squares etc
    - Add a 'thinking' step to the agent, where it can look ahead and see what the best move is
    - Could try reformulating as a problem of minimising the 'thinking steps' where a thinking step amounts to part of a 'search-like' process in which we check possible paths, trying out a new action and then either: continuing down this path with another trial action, rolling back to a previous section, or going right back to the *actual* current board state. This opens up a whole can of complexity in terms of how we would model the agent, it's actions, it's memory etc.
    - Similar but probably much easier (and maybe a small step in the process above) we could replace the normal PPO network with an LSTM or RNN
