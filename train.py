import gymnasium as gym
import torch
import numpy as np
import time
import os
from collections import deque

from environment import RicochetRobotsEnv
from agent import PPOAgent, RolloutBuffer


def main():
    # --- Hyperparameters ---
    total_timesteps = 1_000_000
    num_steps_per_rollout = 2048  # Number of steps to run per policy update
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 10
    num_minibatches = 32 # SB3 default is 4 for num_steps=2048 -> minibatch_size=64. Let's try 32 -> 64.
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    
    board_size = 16 # Default Ricochet Robots board size
    num_robots = 4  # Default number of robots
    max_episode_steps = 100 # Max steps per episode in env

    # Logging and saving
    log_interval = 1 # Log every N rollouts
    save_interval = 20 # Save model every N rollouts
    save_dir = "ppo_ricochet_models"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    # You might want to wrap the env for monitoring, e.g., RecordEpisodeStatistics
    env = RicochetRobotsEnv(
        board_size=board_size,
        num_robots=num_robots,
        max_steps=max_episode_steps,
        use_standard_walls=True, # Use the predefined walls
        render_mode=None # "human" for visualization, None for faster training
    )
    env = gym.wrappers.RecordEpisodeStatistics(env) # Tracks episode returns and lengths

    # --- Agent and Buffer Setup ---
    agent = PPOAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        ppo_epochs=ppo_epochs,
        num_minibatches=num_minibatches,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        max_grad_norm=max_grad_norm,
        device=device
    )
    
    # Check if a saved model exists and load it
    latest_model_path = os.path.join(save_dir, "ppo_ricochet_latest.pth")
    if os.path.exists(latest_model_path):
        print(f"Loading model from {latest_model_path}")
        agent.load_model(latest_model_path)


    rollout_buffer = RolloutBuffer(
        num_steps=num_steps_per_rollout,
        obs_space=env.observation_space,
        action_space_shape=(), # For Discrete action space
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device # Buffer stores numpy, device mainly for agent
    )

    # --- Training Loop ---
    obs_dict, info = env.reset()
    current_total_steps = 0
    num_rollouts_done = 0

    # For logging
    ep_rew_mean_deque = deque(maxlen=100) # Mean reward of last 100 episodes
    ep_len_mean_deque = deque(maxlen=100) # Mean length of last 100 episodes

    start_time = time.time()

    while current_total_steps < total_timesteps:
        agent.network.train() # Ensure network is in training mode for rollouts
        
        for step in range(num_steps_per_rollout):
            current_total_steps += 1

            # Get action from agent
            action, log_prob, value = agent.get_action_and_value(obs_dict)
            
            # Step environment
            next_obs_dict, reward, terminated, truncated, info = env.step(action.item()) # action is usually [val]
            done = terminated or truncated

            # Store experience
            rollout_buffer.add(obs_dict, action, reward, done, value, log_prob)
            
            obs_dict = next_obs_dict

            if "final_info" in info: # From RecordEpisodeStatistics wrapper
                for item in info["final_info"]:
                    if item and "episode" in item:
                        ep_rew = item["episode"]["r"].item()
                        ep_len = item["episode"]["l"].item()
                        ep_rew_mean_deque.append(ep_rew)
                        ep_len_mean_deque.append(ep_len)
                        # print(f"Total Steps: {current_total_steps}, Episode Reward: {ep_rew:.2f}, Episode Length: {ep_len}")


            if done:
                obs_dict, info = env.reset()
                # No need to handle last_value specifically here if episode ends mid-rollout,
                # GAE calculation handles dones correctly.
        
        # --- After collecting num_steps_per_rollout ---
        num_rollouts_done += 1

        # Compute advantages and returns
        with torch.no_grad():
            # Get value of the last observation S_{t+N}
            last_value_np = agent.get_value(obs_dict) # obs_dict is the state after the last step of rollout
            # last_done is whether the episode ended *exactly* at the last step of the rollout.
            # If an episode ended mid-rollout, its 'done' is already in the buffer.
            # If the rollout ended mid-episode, last_done is False.
            # The 'done' status of the *very last step* of the rollout is what matters for bootstrapping.
            # This 'done' is the one associated with obs_dict (the state *after* the rollout).
            # If obs_dict is from a reset, then it's effectively the start of a new episode, so not "done" for GAE.
            # The 'dones' array in the buffer correctly marks terminations within the rollout.
            # For GAE, if the last step of the rollout was a terminal state, last_value should be 0.
            # The 'done' flag passed to compute_advantages_and_returns should reflect if S_{t+N} is terminal.
            # If the environment reset because the last step was terminal, then obs_dict is s_0 of a new episode.
            # In this case, last_done for GAE should be True for the *previous* state.
            # The `done` variable from the loop's last `env.step` is the correct one to use.
            last_done_for_gae = done # done from the last step of the rollout loop

        rollout_buffer.compute_advantages_and_returns(last_value_np, last_done_for_gae)
        
        # Get data from buffer
        (
            batch_obs_list,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            batch_values # V(s_t) from rollout, not strictly needed by PPO learn if using returns
        ) = rollout_buffer.get()

        # Update policy
        agent.learn(
            batch_obs_list,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            batch_values # Pass V(s_t) for potential use, though PPO uses returns
        )

        rollout_buffer.clear()

        # Logging
        if num_rollouts_done % log_interval == 0:
            fps = int(num_steps_per_rollout * log_interval / (time.time() - start_time))
            mean_ep_rew = np.mean(ep_rew_mean_deque) if ep_rew_mean_deque else float('nan')
            mean_ep_len = np.mean(ep_len_mean_deque) if ep_len_mean_deque else float('nan')
            
            print(f"Total Timesteps: {current_total_steps}/{total_timesteps}")
            print(f"Rollout: {num_rollouts_done}, FPS: {fps}")
            print(f"Mean Reward (last 100): {mean_ep_rew:.2f}")
            print(f"Mean Episode Length (last 100): {mean_ep_len:.2f}")
            print("-" * 40)
            start_time = time.time() # Reset timer for next log interval

        # Save model
        if num_rollouts_done % save_interval == 0:
            agent.save_model(latest_model_path)
            agent.save_model(os.path.join(save_dir, f"ppo_ricochet_ts{current_total_steps}.pth"))
            print(f"Model saved at timestep {current_total_steps}")

    env.close()
    print("Training finished.")
    agent.save_model(latest_model_path) # Final save

if __name__ == "__main__":
    main() 