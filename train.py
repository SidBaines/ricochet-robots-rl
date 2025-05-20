import gymnasium as gym
import torch
import numpy as np
import time
import os
import wandb  # Import wandb
from collections import deque
from datetime import datetime
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
    entropy_coef = 0.02
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    
    board_size = 3 # Default Ricochet Robots board size
    num_robots = 1  # Default number of robots
    max_episode_steps = 10 # Max steps per episode in env
    num_edge_walls_per_quadrant = 0
    num_floating_walls_per_quadrant = 0

    # Logging and saving
    log_interval = 1 # Log every N rollouts
    save_interval = 100 # Save model every N rollouts
    # Save in a new directory for each run
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"ppo_ricochet_models/run_{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    do_wandb = False
    if do_wandb:
        wandb_project = "ricochet_robots_rl"
        wandb_entity = None  # Set to your wandb username or team name if needed
        wandb_name = f"ppo_run_{run_id}"
    
    # Collect hyperparameters for wandb
    config = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "num_steps_per_rollout": num_steps_per_rollout,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "ppo_epochs": ppo_epochs,
        "num_minibatches": num_minibatches,
        "entropy_coef": entropy_coef,
        "value_loss_coef": value_loss_coef,
        "max_grad_norm": max_grad_norm,
        "board_size": board_size,
        "num_robots": num_robots,
        "max_episode_steps": max_episode_steps,
        "num_edge_walls_per_quadrant": num_edge_walls_per_quadrant,
        "num_floating_walls_per_quadrant": num_floating_walls_per_quadrant,
    }
    if do_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=config,
            sync_tensorboard=False,  # Set to True if also using TensorBoard
            monitor_gym=False,  # We'll log gym metrics manually
            save_code=True,  # Save a copy of the code
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    # You might want to wrap the env for monitoring, e.g., RecordEpisodeStatistics
    env = RicochetRobotsEnv(
        board_size=board_size,
        num_robots=num_robots,
        max_steps=max_episode_steps,
        use_standard_walls=False, # Use the predefined walls
        num_edge_walls_per_quadrant=num_edge_walls_per_quadrant,
        num_floating_walls_per_quadrant=num_floating_walls_per_quadrant,
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
    
    latest_model_path = os.path.join(save_dir, "ppo_ricochet_latest.pth")
    if 0:
        # Check if a saved model exists and load it
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
    
    # Additional tracking for wandb
    episodes_completed = 0
    episodes_terminated = 0  # Episodes that reached the goal
    episodes_truncated = 0   # Episodes that hit max steps
    
    # For tracking metrics within each rollout
    rollout_rewards = []
    rollout_lengths = []
    rollout_terminated = 0
    rollout_truncated = 0

    start_time = time.time()

    while current_total_steps < total_timesteps:
        agent.network.train() # Ensure network is in training mode for rollouts
        
        # Reset rollout-specific metrics
        rollout_rewards = []
        rollout_lengths = []
        rollout_terminated = 0
        rollout_truncated = 0
        
        # Track episode in progress
        current_ep_reward = 0
        current_ep_length = 0
        
        for step in range(num_steps_per_rollout):
            current_total_steps += 1
            current_ep_length += 1

            # Get action from agent
            action, log_prob, value = agent.get_action_and_value(obs_dict)
            
            # Step environment
            next_obs_dict, reward, terminated, truncated, info = env.step(action.item()) # action is usually [val]
            done = terminated or truncated
            
            # Update episode tracking
            current_ep_reward += reward

            # Store experience
            rollout_buffer.add(obs_dict, action, reward, done, value, log_prob)
            
            obs_dict = next_obs_dict

            if done:
                # Episode completed
                episodes_completed += 1
                if terminated:
                    episodes_terminated += 1
                    rollout_terminated += 1
                if truncated:
                    episodes_truncated += 1
                    rollout_truncated += 1
                
                # Store episode stats
                ep_rew_mean_deque.append(current_ep_reward)
                ep_len_mean_deque.append(current_ep_length)
                rollout_rewards.append(current_ep_reward)
                rollout_lengths.append(current_ep_length)
                
                # Reset episode tracking
                obs_dict, info = env.reset()
                current_ep_reward = 0
                current_ep_length = 0
        
        # --- After collecting num_steps_per_rollout ---
        num_rollouts_done += 1

        # Compute advantages and returns
        with torch.no_grad():
            # Get value of the last observation S_{t+N}
            last_value_np = agent.get_value(obs_dict) # obs_dict is the state after the last step of rollout
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

        # Update policy and track metrics
        policy_loss, value_loss, entropy_loss, approx_kl = agent.learn(
            batch_obs_list,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            batch_values # Pass V(s_t) for potential use, though PPO uses returns
        )

        rollout_buffer.clear()

        # Calculate metrics for this rollout
        success_rate = rollout_terminated / max(1, rollout_terminated + rollout_truncated)
        truncation_rate = rollout_truncated / max(1, rollout_terminated + rollout_truncated)
        mean_ep_reward = np.mean(rollout_rewards) if rollout_rewards else float('nan')
        mean_ep_length = np.mean(rollout_lengths) if rollout_lengths else float('nan')
        
        # Log to wandb
        if do_wandb:
            wandb.log({
                # Progress metrics
                "timesteps": current_total_steps,
                "rollouts_completed": num_rollouts_done,
                "episodes_completed": episodes_completed,
                
                # Performance metrics
                "mean_reward": mean_ep_reward,
                "mean_episode_length": mean_ep_length,
                "success_rate": success_rate,
                "truncation_rate": truncation_rate,
                "episodes_terminated": episodes_terminated,
                "episodes_truncated": episodes_truncated,
                
                # Rolling averages (last 100 episodes)
                "mean_reward_100": np.mean(ep_rew_mean_deque) if ep_rew_mean_deque else float('nan'),
                "mean_episode_length_100": np.mean(ep_len_mean_deque) if ep_len_mean_deque else float('nan'),
                
                # Training metrics
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "approx_kl_divergence": approx_kl,
                
                # Training speed
                "fps": int(num_steps_per_rollout / (time.time() - start_time)),
            }, step=current_total_steps)

        # Logging to console
        if num_rollouts_done % log_interval == 0:
            fps = int(num_steps_per_rollout * log_interval / (time.time() - start_time))
            mean_ep_rew_100 = np.mean(ep_rew_mean_deque) if ep_rew_mean_deque else float('nan')
            mean_ep_len_100 = np.mean(ep_len_mean_deque) if ep_len_mean_deque else float('nan')
            
            print(f"Total Timesteps: {current_total_steps}/{total_timesteps}")
            print(f"Rollout: {num_rollouts_done}, FPS: {fps}")
            print(f"Mean Reward (last {len(ep_rew_mean_deque)}): {mean_ep_rew_100:.2f}")
            print(f"Mean Episode Length (last {len(ep_len_mean_deque)}): {mean_ep_len_100:.2f}")
            print(f"Success Rate: {success_rate:.2f}, Truncation Rate: {truncation_rate:.2f}")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy_loss:.4f}")
            print("-" * 40)
            start_time = time.time() # Reset timer for next log interval

        # Save model
        if num_rollouts_done % save_interval == 0:
            agent.save_model(latest_model_path)
            checkpoint_path = os.path.join(save_dir, f"ppo_ricochet_ts{current_total_steps}.pth")
            agent.save_model(checkpoint_path)
            # Log model checkpoint to wandb
            if do_wandb:
                wandb.save(checkpoint_path)
            print(f"Model saved at timestep {current_total_steps}")

    env.close()
    print("Training finished.")
    agent.save_model(latest_model_path) # Final save
    if do_wandb:
        wandb.save(latest_model_path)
        wandb.finish()

if __name__ == "__main__":
    main() 