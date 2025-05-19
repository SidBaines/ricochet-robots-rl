# %% [markdown]
# # Ricochet Robots Environment Inspector
#
# This notebook allows you to create instances of the `RicochetRobotsEnv`,
# visualize the board, and experiment with different configurations.

# %%
# Essential imports
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt # For potential advanced visualization later, not used in basic render

# Import your custom environment
# Adjust the path if your notebook is in a different location relative to the package
# Option 1: If 'ricochet_robots_rl' is installed or in PYTHONPATH, or running from parent dir
from environment import RicochetRobotsEnv
from environment.utils import NORTH, EAST, SOUTH, WEST

# Option 2: If your notebook is inside the 'ricochet_robots_rl' directory
# from environment import RicochetRobotsEnv
# from environment.utils import NORTH, EAST, SOUTH, WEST

print("Imports successful!")

# %% [markdown]
# ## 1. Create and View a Default Board
#
# This creates an environment with the default settings used in `train.py` (16x16 board, 4 robots, standard walls).

# %%
# Default parameters similar to training script
default_board_size = 16
default_num_robots = 4
default_max_episode_steps = 50 # Not directly relevant for just viewing the board setup

# Create the environment
env_default = RicochetRobotsEnv(
    board_size=default_board_size,
    num_robots=default_num_robots,
    max_steps=default_max_episode_steps,
    use_standard_walls=True, # This adds the predefined walls
    render_mode="human" # "human" prints to console, "ansi" returns a string
)

# Reset the environment to initialize robots and target
# The seed makes the robot/target placement reproducible for this cell
obs, info = env_default.reset(seed=42)

print("\n--- Default Board (16x16 with standard walls) ---")
env_default.render()

# You can also get the ANSI string representation
# ansi_render = env_default.render_mode = "ansi"
# print(env_default.render())
# env_default.render_mode = "human" # switch back if needed

# %% [markdown]
# ## 2. Create a Custom Board
#
# Here, you can specify different parameters like board size, number of robots, and even add custom walls.

# %%
# Custom parameters
custom_board_size = 8
custom_num_robots = 2
custom_max_steps = 30

# Example custom wall configuration: List of ((row, col), direction_idx)
# Wall to the EAST of cell (1,1) and SOUTH of (3,3)
custom_walls_config = [
    ((1, 1), EAST),
    ((3, 3), SOUTH),
    ((0, custom_board_size // 2), SOUTH) # A wall in the middle of the top row
]

env_custom = RicochetRobotsEnv(
    board_size=custom_board_size,
    num_robots=custom_num_robots,
    max_steps=custom_max_steps,
    board_walls_config=custom_walls_config, # Pass custom walls
    use_standard_walls=True, # Disable standard walls if you only want custom ones
    render_mode="human"
)

# Reset to see the setup
obs_custom, info_custom = env_custom.reset(seed=123)

print(f"\n--- Custom Board ({custom_board_size}x{custom_board_size} with custom walls) ---")
env_custom.render()

# %% [markdown]
# ### Exploring the `env.board.walls` attribute
# The `env.board.walls` is a NumPy array of shape `(height, width, 4)`.
# `env.board.walls[r, c, direction_idx]` is `True` if there's a wall on that side of cell `(r,c)`.
# Directions: 0:N, 1:E, 2:S, 3:W

# %%
print(f"Custom environment board wall data for cell (1,1): {env_custom.board.walls[1,1]}")
# Expected: [F, T, F, F] (assuming no other walls around (1,1) other than the EAST one we added and perimeter)
# (It will also have a WEST wall if (1,0) is the perimeter, and NORTH if (0,1) is perimeter)

# Let's check the wall we added: East of (1,1)
print(f"Wall EAST of (1,1): {env_custom.board.has_wall(1,1,EAST)}")
# And its corresponding wall: West of (1,2)
if custom_board_size > 2:
    print(f"Wall WEST of (1,2): {env_custom.board.has_wall(1,2,WEST)}")

# %% [markdown]
# ## 3. Interactive Board Setup (Manual Wall Placement)
#
# You can create an empty board and then add walls programmatically.

# %%
env_interactive = RicochetRobotsEnv(
    board_size=6,
    num_robots=1,
    use_standard_walls=False, # Start with only perimeter walls
    render_mode="human"
)
obs_interactive, _ = env_interactive.reset(seed=1) # Reset to place robot and target

print("\n--- Interactive Board (Before Adding Walls) ---")
env_interactive.render()

# Add some walls
env_interactive.board.add_wall(r=2, c=2, direction_idx=EAST) # Wall East of (2,2)
env_interactive.board.add_wall(r=2, c=2, direction_idx=SOUTH) # Wall South of (2,2)
env_interactive.board.add_wall(r=4, c=0, direction_idx=EAST) # Wall East of (4,0)

print("\n--- Interactive Board (After Adding Walls) ---")
# The render method uses the board's current wall state.
# No need to reset unless you want to re-place robots/target.
env_interactive.render()


# %% [markdown]
# ## 4. Simulating a Few Steps
#
# You can also take a few steps in the environment to see how robots move.

# %%
env_step_test = RicochetRobotsEnv(
    board_size=8,
    num_robots=2,
    use_standard_walls=False,
    render_mode="human"
)
env_step_test.board.add_wall(2,2,EAST) # Add a wall for robot to hit
obs, info = env_step_test.reset(seed=7)

print("\n--- Step Test: Initial State ---")
env_step_test.render()

# Let's find out which robot is the target and its current position
target_robot_idx = obs["target_robot_idx"]
target_robot_obj = env_step_test.robots[target_robot_idx]
print(f"Target Robot ID: {target_robot_idx} (Color: {target_robot_obj.color_char}) at {target_robot_obj.pos}")
print(f"Target Location: {env_step_test.target_pos}")

# Example: Move robot 0 (if it exists) to the NORTH
# Action = robot_idx * 4 + direction_idx
# Directions: 0:N, 1:E, 2:S, 3:W
if 0 < env_step_test.num_robots:
    action_robot0_north = 0 * 4 + NORTH
    print(f"\nAttempting to move Robot 0 North (Action: {action_robot0_north})...")
    next_obs, reward, terminated, truncated, info = env_step_test.step(action_robot0_north)
    print("--- State After One Move ---")
    env_step_test.render()
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"Robot 0 new position: {env_step_test.robots[0].pos}")

# %% [markdown]
# This notebook should give you a good starting point for inspecting and tweaking your environment setup!
