import gymnasium as gym
import torch
import numpy as np
import argparse
import json
import os

# Assuming you run this script from the directory containing 'ricochet_robots_rl'
from environment import RicochetRobotsEnv
from agent import PPOAgent # PPOAgent will import ActorCriticPPO

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a trained PPO agent for Ricochet Robots.")
    parser.add_argument("--model_path", type=str, default="ppo_ricochet_models/ppo_ricochet_latest.pth",
                        help="Path to the trained model file (.pth).")
    parser.add_argument("--board_size", type=int, default=10, help="Size of the board (height and width).")
    parser.add_argument("--num_robots", type=int, default=4, help="Number of robots.")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="Max steps per episode for the environment.")
    parser.add_argument("--use_standard_walls", action='store_true', help="Use standard Ricochet Robots walls (central block).")
    parser.add_argument("--no_standard_walls", action='store_false', dest='use_standard_walls', help="Do not use standard walls.")
    parser.set_defaults(use_standard_walls=True) # Default to using standard walls

    parser.add_argument("--num_edge_walls_per_quadrant", type=int, default=2,
                        help="Number of edge walls to generate per quadrant.")
    parser.add_argument("--num_floating_walls_per_quadrant", type=int, default=1,
                        help="Number of floating L-shaped walls to generate per quadrant.")
    parser.add_argument("--board_walls_config_str", type=str, default=None,
                        help="JSON string for custom board_walls_config. E.g., '[[[1,1],0],[[2,3],1]]'")

    parser.add_argument("--num_simulation_steps", type=int, default=20, help="Number of steps to simulate and observe.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "ansi"],
                        help="Render mode for the environment.")
    parser.add_argument("--deterministic_actions", action='store_true',
                        help="Take deterministic actions (argmax) instead of sampling.")
    parser.add_argument("--no_deterministic_actions", action='store_false', dest='deterministic_actions',
                        help="Sample actions (stochastic).")
    parser.set_defaults(deterministic_actions=True) # Default to deterministic for inspection

    return parser.parse_args()

def main():
    args = parse_args()

    print("--- Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("---------------------\n")

    # --- Environment Setup ---
    board_walls_config = None
    if args.board_walls_config_str:
        try:
            board_walls_config = json.loads(args.board_walls_config_str)
            print(f"Using custom board_walls_config: {board_walls_config}")
        except json.JSONDecodeError as e:
            print(f"Error parsing board_walls_config_str: {e}. Ignoring.")
            board_walls_config = None

    env = RicochetRobotsEnv(
        board_size=args.board_size,
        num_robots=args.num_robots,
        max_steps=args.max_episode_steps,
        use_standard_walls=args.use_standard_walls,
        board_walls_config=board_walls_config,
        num_edge_walls_per_quadrant=args.num_edge_walls_per_quadrant,
        num_floating_walls_per_quadrant=args.num_floating_walls_per_quadrant,
        render_mode=args.render_mode
    )

    # --- Agent Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        env.close()
        return

    # Instantiate agent (hyperparameters like lr, gamma etc. don't matter for inference)
    agent = PPOAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    try:
        agent.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        env.close()
        return
    
    agent.network.eval() # Ensure model is in evaluation mode

    # --- Simulation Loop ---
    obs, info = env.reset(seed=args.seed)
    
    total_reward_this_episode = 0
    current_episode_steps = 0

    for step_num in range(args.num_simulation_steps):
        print(f"\n--- Step {step_num + 1} ---")
        
        if args.render_mode == "human":
            env.render()
        elif args.render_mode == "ansi":
            print(env.render())

        # Get action from agent
        action_np, log_prob_np, value_np = agent.select_action(obs, deterministic=args.deterministic_actions)
        
        # Action details
        # Assuming action_np is a scalar or a single-element array for discrete actions
        action_to_take = action_np.item() if isinstance(action_np, np.ndarray) and action_np.ndim > 0 else action_np 


        print(f"Observation (Target Robot Idx): {obs['target_robot_idx']}")
        # Can print more obs details if needed, e.g. obs['board_features'].shape

        # Step the environment
        robot_being_moved = action_to_take // 4
        direction_idx = action_to_take % 4
        prev_pos = env.robots[robot_being_moved].pos
        next_obs, reward, terminated, truncated, info = env.step(action_to_take)
        done = terminated or truncated
        new_pos = env.robots[robot_being_moved].pos
        print(f"Agent chose Action: {action_to_take}")
        print(f"This amounts to moving the {env.robots[robot_being_moved].color_char} robot from {prev_pos} to {new_pos}")
        print(f"  Log Probability: {log_prob_np.item():.4f}")
        print(f"  Estimated Value: {value_np.item():.4f}")


        print(f"Environment responded:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        total_reward_this_episode += reward
        current_episode_steps +=1
        obs = next_obs

        if done:
            print(f"Episode finished after {current_episode_steps} steps.")
            print(f"Total reward for this episode: {total_reward_this_episode:.2f}")
            if step_num < args.num_simulation_steps - 1: # If not the last step of simulation
                print("Resetting environment...")
                obs, info = env.reset(seed=args.seed + step_num + 1 if args.seed is not None else None) # Vary seed on auto-reset
                total_reward_this_episode = 0
                current_episode_steps = 0
            else:
                break # End simulation if episode ends on the last requested step
    
    print("\n--- Inspection Finished ---")
    env.close()

if __name__ == "__main__":
    main() 