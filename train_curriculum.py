"""
Curriculum learning training script for Ricochet Robots RL.

This script extends the basic PPO training with a curriculum learning system
that progressively increases task difficulty based on agent performance.
"""

import gymnasium as gym
import torch
import numpy as np
import random
import time
import os
import wandb
from collections import deque
from datetime import datetime
from environment import RicochetRobotsEnv
from environment.simpler_ricochet_env import RicochetRobotsEnvOneStepAway
from agent import PPOAgent, RolloutBuffer
from curriculum.curriculum_env import create_curriculum_env


class CurriculumScheduler:
    """
    Curriculum scheduler that tracks performance and adjusts difficulty.
    
    Implements self-paced learning by monitoring success rate and automatically
    promoting to higher difficulty levels when performance thresholds are met.
    """
    
    def __init__(self, 
                 initial_k: int = 1,
                 success_threshold: float = 0.80,
                 promotion_cooldown: int = 10000,  # timesteps
                 max_k: int = 10):
        """
        Initialize curriculum scheduler.
        
        Args:
            initial_k: Starting curriculum difficulty level
            success_threshold: Success rate required for promotion
            promotion_cooldown: Minimum timesteps between promotions
            max_k: Maximum curriculum level
        """
        self.current_k = initial_k
        self.success_threshold = success_threshold
        self.promotion_cooldown = promotion_cooldown
        self.max_k = max_k
        
        # Performance tracking
        self.success_rates = deque(maxlen=100)  # Rolling success rate
        self.last_promotion_timestep = 0
        
        # Statistics
        self.total_promotions = 0
        self.k_history = [initial_k]
    
    def update(self, success_rate: float, current_timestep: int) -> bool:
        """
        Update curriculum based on performance.
        
        Args:
            success_rate: Current rolling success rate
            current_timestep: Current training timestep
            
        Returns:
            True if curriculum level was promoted, False otherwise
        """
        self.success_rates.append(success_rate)
        
        # Check if we should promote
        if (success_rate >= self.success_threshold and 
            current_timestep - self.last_promotion_timestep >= self.promotion_cooldown and
            self.current_k < self.max_k):
            
            # Promote to next level
            self.current_k += 1
            self.last_promotion_timestep = current_timestep
            self.total_promotions += 1
            self.k_history.append(self.current_k)
            
            print(f"🎓 Curriculum promoted to k={self.current_k} at timestep {current_timestep}")
            print(f"   Success rate: {success_rate:.3f} >= {self.success_threshold}")
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get curriculum scheduler statistics."""
        return {
            'current_k': self.current_k,
            'total_promotions': self.total_promotions,
            'last_promotion_timestep': self.last_promotion_timestep,
            'k_history': self.k_history.copy(),
            'rolling_success_rate': np.mean(self.success_rates) if self.success_rates else 0.0
        }


def main():
    # --- Curriculum Learning Hyperparameters ---
    use_curriculum = True  # Set to False for standard training
    initial_k = 1  # Starting curriculum difficulty
    success_threshold = 0.80  # Success rate for curriculum promotion
    promotion_cooldown = 20000  # Timesteps between promotions
    max_k = 5  # Maximum curriculum level
    
    # Curriculum configuration
    curriculum_config = {
        'initial_k': initial_k,
        'pool_size': 30,  # Number of boards to keep in pool
        'num_workers': min(6, max(1, os.cpu_count() - 2)),  # Board generation workers
        'cache_path': 'curriculum_cache.lmdb',
        'board_size': 5,
        'num_robots': 3,
        'use_standard_walls': False,
        'num_edge_walls_per_quadrant': 0,
        'num_floating_walls_per_quadrant': 0,
        'epsilon_random': 0.05  # Probability of random board for exploration
    }
    
    # --- Standard PPO Hyperparameters ---
    total_timesteps = 2_000_000  # Extended for curriculum learning
    num_steps_per_rollout = 2048
    learning_rate = 3e-5
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 10
    num_minibatches = 32
    entropy_coef = 0.02
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    
    # Environment parameters (used for fallback when curriculum not available)
    board_size = curriculum_config['board_size']
    num_robots = curriculum_config['num_robots']
    max_episode_steps = 15  # Slightly higher for curriculum learning
    
    # Logging and saving
    log_interval = 1
    save_interval = 100
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"ppo_curriculum_models/run_{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    do_wandb = False  # Enable for curriculum learning experiments
    if do_wandb:
        wandb_project = "ricochet_robots_curriculum"
        wandb_entity = None
        wandb_name = f"curriculum_k{initial_k}_run_{run_id}"
    
    # Collect hyperparameters for wandb
    config = {
        "algorithm": "PPO_Curriculum",
        "use_curriculum": use_curriculum,
        "initial_k": initial_k,
        "success_threshold": success_threshold,
        "promotion_cooldown": promotion_cooldown,
        "max_k": max_k,
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
    }
    config.update(curriculum_config)  # Add curriculum config
    
    if do_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=config,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if use_curriculum:
        print("🎯 Setting up curriculum learning environment...")
        env = create_curriculum_env(
            curriculum_config=curriculum_config,
            max_steps=max_episode_steps,
            render_mode=None
        )
        print(f"   Initial curriculum level: k={initial_k}")
        print(f"   Pool size: {curriculum_config['pool_size']}")
        print(f"   Workers: {curriculum_config['num_workers']}")
    else:
        print("📚 Using standard environment (no curriculum)")
        env = RicochetRobotsEnvOneStepAway(
            board_size=board_size,
            num_robots=num_robots,
            max_steps=max_episode_steps,
            use_standard_walls=False,
            seed=seed,
        )
    
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # --- Curriculum Scheduler Setup ---
    if use_curriculum:
        curriculum_scheduler = CurriculumScheduler(
            initial_k=initial_k,
            success_threshold=success_threshold,
            promotion_cooldown=promotion_cooldown,
            max_k=max_k
        )
    else:
        curriculum_scheduler = None

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
    
    latest_model_path = os.path.join(save_dir, "ppo_curriculum_latest.pth")

    rollout_buffer = RolloutBuffer(
        num_steps=num_steps_per_rollout,
        obs_space=env.observation_space,
        action_space_shape=(),
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device
    )

    # --- Training Loop ---
    obs_dict, info = env.reset()
    current_total_steps = 0
    num_rollouts_done = 0

    # For logging
    ep_rew_mean_deque = deque(maxlen=100)
    ep_len_mean_deque = deque(maxlen=100)
    
    # Additional tracking
    episodes_completed = 0
    episodes_terminated = 0
    episodes_truncated = 0
    
    # For tracking metrics within each rollout
    rollout_rewards = []
    rollout_lengths = []
    rollout_terminated = 0
    rollout_truncated = 0

    start_time = time.time()

    print(f"🚀 Starting curriculum training for {total_timesteps:,} timesteps...")

    while current_total_steps < total_timesteps:
        agent.network.train()
        
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
            next_obs_dict, reward, terminated, truncated, info = env.step(action.item())
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
                obs_dict, info = env.reset(seed=current_total_steps)
                current_ep_reward = 0
                current_ep_length = 0
        
        # --- After collecting rollout ---
        num_rollouts_done += 1

        # Compute advantages and returns
        with torch.no_grad():
            last_value_np = agent.get_value(obs_dict)
            last_done_for_gae = done

        rollout_buffer.compute_advantages_and_returns(last_value_np, last_done_for_gae)
        
        # Get data from buffer
        (
            batch_obs_list,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            batch_values
        ) = rollout_buffer.get()

        # Update policy
        policy_loss, value_loss, entropy_loss, approx_kl = agent.learn(
            batch_obs_list,
            batch_actions,
            batch_log_probs_old,
            batch_advantages,
            batch_returns,
            batch_values
        )

        rollout_buffer.clear()

        # Calculate metrics for this rollout
        success_rate = rollout_terminated / max(1, rollout_terminated + rollout_truncated)
        mean_ep_reward = np.mean(rollout_rewards) if rollout_rewards else float('nan')
        mean_ep_length = np.mean(rollout_lengths) if rollout_lengths else float('nan')
        
        # --- Curriculum Learning Updates ---
        curriculum_promoted = False
        if use_curriculum and curriculum_scheduler:
            curriculum_promoted = curriculum_scheduler.update(success_rate, current_total_steps)
            
            if curriculum_promoted:
                # Update environment curriculum level
                env.set_curriculum_level(curriculum_scheduler.current_k)
                
                # Log curriculum promotion
                if do_wandb:
                    wandb.log({
                        "curriculum/promotion": 1,
                        "curriculum/new_k": curriculum_scheduler.current_k,
                    }, step=current_total_steps)
        
        # Get curriculum statistics
        curriculum_stats = {}
        if use_curriculum:
            if hasattr(env, 'get_curriculum_stats'):
                curriculum_stats = env.get_curriculum_stats()
            if curriculum_scheduler:
                scheduler_stats = curriculum_scheduler.get_stats()
                curriculum_stats.update({f"curriculum/{k}": v for k, v in scheduler_stats.items()})
        
        # Log to wandb
        if do_wandb:
            log_data = {
                # Progress metrics
                "timesteps": current_total_steps,
                "rollouts_completed": num_rollouts_done,
                "episodes_completed": episodes_completed,
                
                # Performance metrics
                "performance/mean_reward": mean_ep_reward,
                "performance/mean_episode_length": mean_ep_length,
                "performance/success_rate": success_rate,
                "performance/episodes_terminated": episodes_terminated,
                "performance/episodes_truncated": episodes_truncated,
                
                # Rolling averages
                "performance/mean_reward_100": np.mean(ep_rew_mean_deque) if ep_rew_mean_deque else float('nan'),
                "performance/mean_episode_length_100": np.mean(ep_len_mean_deque) if ep_len_mean_deque else float('nan'),
                
                # Training metrics
                "training/policy_loss": policy_loss,
                "training/value_loss": value_loss,
                "training/entropy_loss": entropy_loss,
                "training/approx_kl_divergence": approx_kl,
                
                # Training speed
                "training/fps": int(num_steps_per_rollout / (time.time() - start_time)),
            }
            
            # Add curriculum metrics
            log_data.update(curriculum_stats)
            
            wandb.log(log_data, step=current_total_steps)

        # Console logging
        if num_rollouts_done % log_interval == 0:
            fps = int(num_steps_per_rollout * log_interval / (time.time() - start_time))
            mean_ep_rew_100 = np.mean(ep_rew_mean_deque) if ep_rew_mean_deque else float('nan')
            mean_ep_len_100 = np.mean(ep_len_mean_deque) if ep_len_mean_deque else float('nan')
            
            print(f"📊 Timesteps: {current_total_steps:,}/{total_timesteps:,}")
            print(f"   Rollout: {num_rollouts_done}, FPS: {fps}")
            print(f"   Reward (last 100): {mean_ep_rew_100:.2f}")
            print(f"   Episode Length (last 100): {mean_ep_len_100:.2f}")
            print(f"   Success Rate: {success_rate:.2f}")
            
            if use_curriculum and curriculum_scheduler:
                print(f"   🎯 Curriculum Level: k={curriculum_scheduler.current_k}")
                if curriculum_stats:
                    pool_size = curriculum_stats.get('pool_size', 'N/A')
                    usage_rate = curriculum_stats.get('curriculum_usage_rate', 0)
                    print(f"   📚 Pool: {pool_size} boards, Usage: {usage_rate:.1%}")
            
            print(f"   Training: Policy={policy_loss:.4f}, Value={value_loss:.4f}, Entropy={entropy_loss:.4f}")
            if curriculum_promoted:
                print(f"   🎓 PROMOTED to k={curriculum_scheduler.current_k}!")
            print("-" * 60)
            
            start_time = time.time()

        # Save model
        if num_rollouts_done % save_interval == 0:
            agent.save_model(latest_model_path)
            checkpoint_path = os.path.join(save_dir, f"ppo_curriculum_ts{current_total_steps}.pth")
            agent.save_model(checkpoint_path)
            
            if do_wandb:
                wandb.save(checkpoint_path)
            print(f"💾 Model saved at timestep {current_total_steps}")

    # --- Training Complete ---
    env.close()
    print("🏁 Training finished!")
    
    # Final save
    agent.save_model(latest_model_path)
    if do_wandb:
        wandb.save(latest_model_path)
        
        # Log final curriculum stats
        if use_curriculum and curriculum_scheduler:
            final_stats = curriculum_scheduler.get_stats()
            wandb.log({
                "final/total_promotions": final_stats['total_promotions'],
                "final/final_k": final_stats['current_k'],
                "final/k_history": final_stats['k_history']
            })
        
        wandb.finish()
    
    if use_curriculum and curriculum_scheduler:
        print(f"🎓 Final curriculum level: k={curriculum_scheduler.current_k}")
        print(f"   Total promotions: {curriculum_scheduler.total_promotions}")
        print(f"   K progression: {' → '.join(map(str, curriculum_scheduler.k_history))}")


if __name__ == "__main__":
    main() 