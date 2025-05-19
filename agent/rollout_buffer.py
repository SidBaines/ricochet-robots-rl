import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class RolloutBuffer:
    def __init__(self,
                 num_steps: int,
                 obs_space: Dict, # Gymnasium obs_space dict
                 action_space_shape: Tuple, # For discrete, usually () or (1,)
                 gamma: float,
                 gae_lambda: float,
                 device: torch.device):
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Determine shapes for storage based on observation space
        # Assuming obs_space is a gymnasium.spaces.Dict
        self.obs_board_shape = obs_space["board_features"].shape
        # target_robot_idx is a scalar, will be stored as such

        self.observations_board = np.zeros((self.num_steps, *self.obs_board_shape), dtype=obs_space["board_features"].dtype)
        self.observations_target_idx = np.zeros((self.num_steps, 1), dtype=obs_space["target_robot_idx"].dtype) # Store as [N,1]

        self.actions = np.zeros((self.num_steps, *action_space_shape), dtype=np.int64) # Assuming discrete actions
        self.action_log_probs = np.zeros((self.num_steps), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps), dtype=np.float32)
        self.dones = np.zeros((self.num_steps), dtype=bool) # For terminated or truncated
        self.values = np.zeros((self.num_steps), dtype=np.float32)
        
        self.advantages = np.zeros((self.num_steps), dtype=np.float32)
        self.returns = np.zeros((self.num_steps), dtype=np.float32)

        self.ptr = 0 # Current position in the buffer
        self.path_start_idx = 0
        self.max_size = self.num_steps
        self.buffer_full = False


    def add(self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: float,
            done: bool,
            value: np.ndarray, # V(s_t)
            log_prob: np.ndarray):
        """Add one step of experience to the buffer."""
        if self.ptr >= self.max_size:
            # This indicates an issue if we are trying to add more than num_steps before processing
            # For on-policy, buffer should be processed once full (num_steps collected)
            print("Warning: RolloutBuffer is full. Overwriting old data. This shouldn't happen in standard PPO rollout.")
            self.ptr = 0 # Or raise an error

        self.observations_board[self.ptr] = obs["board_features"]
        self.observations_target_idx[self.ptr] = obs["target_robot_idx"] # Ensure it's stored as [val]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.item() # value is usually a 1-element array
        self.action_log_probs[self.ptr] = log_prob.item() # log_prob is usually a 1-element array
        
        self.ptr += 1
        if self.ptr == self.max_size:
            self.buffer_full = True

    def compute_advantages_and_returns(self, last_value: np.ndarray, last_done: bool):
        """
        Compute advantages and returns using GAE.
        Call this after a rollout is complete (i.e., buffer is full or episode ended).
        last_value: V(s_{t+N}) - value of the state after the last collected step.
        last_done: Whether the episode terminated/truncated after the last collected step.
        """
        if not self.buffer_full and self.ptr < self.num_steps :
            # This might happen if an episode ends before num_steps is reached.
            # We only process the valid part of the buffer.
            # However, for PPO, we typically collect exactly num_steps.
            # This logic is more for if we decide to process partial rollouts.
            # For now, assume we always fill num_steps or an episode ends.
            pass


        # Use self.ptr to determine how much of the buffer is filled if not fully num_steps
        # For standard PPO, self.ptr should be equal to self.num_steps here.
        actual_rollout_length = self.ptr 

        last_gae_lam = 0
        for t in reversed(range(actual_rollout_length)):
            if t == actual_rollout_length - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value.item()
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        
        self.returns = self.advantages[:actual_rollout_length] + self.values[:actual_rollout_length]
        # self.returns[:actual_rollout_length] = self.advantages[:actual_rollout_length] + self.values[:actual_rollout_length]


    def get(self) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve all data from the buffer.
        Returns data for the agent's learn method.
        """
        if not self.buffer_full and self.ptr < self.num_steps:
            # This case means an episode ended before num_steps, and we are processing it.
            # We should only return the valid part of the buffer.
            # For standard PPO, we expect the buffer to be full (ptr == num_steps).
            # This assertion helps catch if get() is called prematurely.
            # assert self.ptr > 0, "Buffer is empty or get() called incorrectly."
            # For now, let's assume we always collect num_steps or process what's available.
            pass

        actual_size = self.ptr # Number of elements actually stored

        # Ensure we don't try to shuffle if num_minibatches > actual_size in agent.learn
        # The agent's learn method should handle this.

        # Reconstruct obs list of dicts
        obs_list = []
        for i in range(actual_size):
            obs_list.append({
                "board_features": self.observations_board[i],
                "target_robot_idx": self.observations_target_idx[i].item() # Convert [val] back to scalar
            })

        return (
            obs_list, # List of obs dicts
            self.actions[:actual_size],
            self.action_log_probs[:actual_size],
            self.advantages[:actual_size],
            self.returns[:actual_size],
            self.values[:actual_size] # V(s_t) from rollout
        )

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.buffer_full = False
        # Optionally re-zero arrays if memory is a concern, but not strictly necessary
        # as they will be overwritten. 