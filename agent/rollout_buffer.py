import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x):
        if np.isscalar(x):
            x = np.array([x])
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class RolloutBuffer:
    def __init__(self,
                 num_steps: int,
                 obs_space: Dict, # Gymnasium obs_space dict
                 action_space_shape: Tuple, # For discrete, usually () or (1,)
                 gamma: float,
                 gae_lambda: float,
                 device: torch.device,
                 num_envs: int = 1,
                 num_lstm_layers: int = 3,  # Should match model
                 lstm_hidden_dims: Tuple[int, ...] = (32, 32, 32),  # Should match model
                 board_height: int = None,
                 board_width: int = None,
                 ):
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.num_envs = num_envs
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_dims = lstm_hidden_dims

        # Determine shapes for storage based on observation space
        self.obs_board_shape = obs_space["board_features"].shape
        self.board_height = board_height or self.obs_board_shape[1]
        self.board_width = board_width or self.obs_board_shape[2]

        self.observations_board = np.zeros((self.num_steps, *self.obs_board_shape), dtype=obs_space["board_features"].dtype)
        self.observations_target_idx = np.zeros((self.num_steps, 1), dtype=obs_space["target_robot_idx"].dtype)

        self.actions = np.zeros((self.num_steps, *action_space_shape), dtype=np.int64)
        self.action_log_probs = np.zeros((self.num_steps), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps), dtype=np.float32)
        self.dones = np.zeros((self.num_steps), dtype=bool)
        self.values = np.zeros((self.num_steps), dtype=np.float32)
        self.advantages = np.zeros((self.num_steps), dtype=np.float32)
        self.returns = np.zeros((self.num_steps), dtype=np.float32)

        # --- Store LSTM hidden/cell states for each step ---
        self.h_states = [
            np.zeros((self.num_steps, self.lstm_hidden_dims[i], self.board_height, self.board_width), dtype=np.float32)
            for i in range(self.num_lstm_layers)
        ]
        self.c_states = [
            np.zeros((self.num_steps, self.lstm_hidden_dims[i], self.board_height, self.board_width), dtype=np.float32)
            for i in range(self.num_lstm_layers)
        ]

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = self.num_steps
        self.buffer_full = False

        # Reward normalization
        self.ret = 0
        self.ret_rms = RunningMeanStd()

    def add(self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: float,
            done: bool,
            value: np.ndarray, # V(s_t)
            log_prob: np.ndarray,
            h_states: Optional[list] = None,
            c_states: Optional[list] = None):
        """Add one step of experience to the buffer."""
        if self.ptr >= self.max_size:
            print("Warning: RolloutBuffer is full. Overwriting old data. This shouldn't happen in standard PPO rollout.")
            self.ptr = 0

        # Normalize reward
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)
        reward = reward / np.sqrt(self.ret_rms.var + 1e-8)
        if done:
            self.ret = 0

        self.observations_board[self.ptr] = obs["board_features"]
        self.observations_target_idx[self.ptr] = obs["target_robot_idx"]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.item()
        self.action_log_probs[self.ptr] = log_prob.item()

        # --- Store LSTM hidden/cell states ---
        if h_states is not None and c_states is not None:
            for i in range(self.num_lstm_layers):
                self.h_states[i][self.ptr] = h_states[i]
                self.c_states[i][self.ptr] = c_states[i]

        self.ptr += 1
        if self.ptr == self.max_size:
            self.buffer_full = True

    def compute_advantages_and_returns(self, last_value: np.ndarray, last_done: bool):
        """
        Compute advantages and returns using GAE.
        """
        actual_rollout_length = self.ptr 
        
        if actual_rollout_length == 0:
            return  # Nothing to compute
        
        # Ensure last_value is a scalar
        if isinstance(last_value, np.ndarray):
            last_value_scalar = last_value.item() if last_value.size == 1 else last_value[0]
        else:
            last_value_scalar = float(last_value)

        last_gae_lam = 0
        for t in reversed(range(actual_rollout_length)):
            if t == actual_rollout_length - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value_scalar
            else:
                next_values = self.values[t + 1]
            
            # Check if current step is terminal
            current_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + self.gamma * next_values * current_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * current_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            
            # Reset last_gae_lam to 0 if this is a terminal state
            if self.dones[t]:
                last_gae_lam = 0
        
        # Only compute returns for the valid portion
        self.returns[:actual_rollout_length] = self.advantages[:actual_rollout_length] + self.values[:actual_rollout_length]

    def get_initial_states_for_batch(self, indices: np.ndarray):
        """
        Returns the initial hidden/cell states for a batch of indices.
        indices: array of shape (batch_size,) with the step indices to start from.
        Returns: (h_states, c_states) as lists of tensors, each of shape (batch_size, C, H, W)
        """
        h_states = [torch.from_numpy(h[i]).to(self.device) for h, i in zip(self.h_states, indices)]
        c_states = [torch.from_numpy(c[i]).to(self.device) for c, i in zip(self.c_states, indices)]
        return h_states, c_states

    def get(self) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
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

        # Fix LSTM state retrieval - slice each layer separately
        h_states_batch = [h[:actual_size] for h in self.h_states]  # List of (actual_size, hidden_dim, H, W)
        c_states_batch = [c[:actual_size] for c in self.c_states]  # List of (actual_size, hidden_dim, H, W)

        return (
            obs_list, # List of obs dicts
            self.actions[:actual_size],
            self.action_log_probs[:actual_size],
            self.advantages[:actual_size],
            self.returns[:actual_size],
            self.values[:actual_size], # V(s_t) from rollout
            h_states_batch,
            c_states_batch,
        )

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.buffer_full = False
        # Optionally re-zero arrays if memory is a concern, but not strictly necessary
        # as they will be overwritten. 