# A little script to create an environment, load a model, and run the agent for a few timesteps
# %% 
from agent import PPOAgent
from environment.ricochet_env import RicochetRobotsEnv
import torch
import os
import numpy as np
from environment.utils import DIRECTIONS
# --- Environment Setup ---
env = RicochetRobotsEnv(
        board_size=5,
        num_robots=1,
        max_steps=100,
        use_standard_walls=True,
        board_walls_config=None,
        num_edge_walls_per_quadrant=0,
        num_floating_walls_per_quadrant=0,
        render_mode="human",
        display_step=False
    )


# %%
# --- Agent Setup ---
model_path = "ppo_ricochet_models/run_<TIMESTAMP>/ppo_ricochet_latest.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")

# Instantiate agent (hyperparameters like lr, gamma etc. don't matter for inference)
agent = PPOAgent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
try:
    agent.load_model(model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")

agent.network.eval() # Ensure model is in evaluation mode

# %%
# --- Simulation Loop ---
obs, info = env.reset(seed=42)
print("Env.current_step: ", env.current_step)
max_steps = env.max_steps
step_num = 0
render_every_n_steps = 100
terminated = False
truncated = False
while (not terminated) and (not truncated) and (step_num < max_steps):
    action, log_prob, value = agent.select_action(obs, deterministic=False)
    action_to_take = action.item() if isinstance(action, np.ndarray) and action.ndim > 0 else action
    print("Robot action: ", action_to_take, "corresponding to direction: ", DIRECTIONS[action_to_take//4])
    obs, reward, terminated, truncated, info = env.step(action_to_take)
    if ((step_num % render_every_n_steps) == 0):
        env.render()
    step_num += 1
env.render()
print(f"Terminated: {terminated}, Truncated: {truncated}, Step Num: {step_num}, Current step {env.current_step}, max steps {env.max_steps}")
# env.close()

# %%
