import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
import torch.nn.functional as F

from .models import ActorCriticPPO, DeepRepeatedConvLSTM  # Assuming models.py is in the same directory

class PPOAgent:
    def __init__(self,
                 obs_space: spaces.Dict,
                 action_space: spaces.Discrete,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 ppo_epochs: int = 10,
                 num_minibatches: int = 4, # Number of minibatches to split a batch into
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: Optional[torch.device] = None,
                 model_type: str = "simple",
                 num_envs: int = 1,  # <--- Add this if using vectorized envs
                 num_lstm_layers: int = 3,
                 lstm_hidden_dims: Tuple[int, ...] = (32, 32, 32),
                 repeat_timesteps: int = 3,
                 ):

        self.obs_space = obs_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.total_timesteps = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model_type == "simple":
            self.network = ActorCriticPPO(obs_space, action_space).to(self.device)
            self.is_recurrent = False
        elif model_type == "convlstm":
            self.network = DeepRepeatedConvLSTM(obs_space, action_space, num_layers=num_lstm_layers, hidden_dims=lstm_hidden_dims, repeat_K=repeat_timesteps).to(self.device)
            self.is_recurrent = True
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # --- Recurrent state tracking ---
        if self.is_recurrent:
            self.h_states, self.c_states = self.network.get_initial_states(self.num_envs, self.device)

    def _get_lr(self):
        """Linear learning rate schedule"""
        progress = min(1.0, self.total_timesteps / 1_000_000)  # Adjust denominator based on total training steps
        return self.lr * (1 - progress)

    def _obs_to_torch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Converts a NumPy observation dictionary to a PyTorch tensor dictionary on the correct device."""
        # Ensure target_robot_idx is an array before converting to tensor
        target_idx = obs["target_robot_idx"]
        if not isinstance(target_idx, np.ndarray):
            target_idx = np.array(target_idx)
        if target_idx.ndim == 0: # scalar
            target_idx = target_idx[np.newaxis] # make it 1D array

        return {
            "board_features": torch.from_numpy(obs["board_features"]).to(self.device),
            "target_robot_idx": torch.from_numpy(target_idx).to(self.device)
        }
    
    def _batch_obs_to_torch(self, obs_batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Converts a list of NumPy observation dictionaries to a batched PyTorch tensor dictionary."""
        board_features_list = [torch.from_numpy(o["board_features"]).float() for o in obs_batch]
        target_robot_idx_list = [torch.tensor(o["target_robot_idx"], dtype=torch.long) for o in obs_batch]

        return {
            "board_features": torch.stack(board_features_list).to(self.device),
            "target_robot_idx": torch.stack(target_robot_idx_list).to(self.device)
        }

    def reset_states(self, env_indices=None):
        """Reset hidden/cell states for given env indices (or all if None)."""
        if not self.is_recurrent:
            return
        if env_indices is None:
            self.h_states, self.c_states = self.network.get_initial_states(self.num_envs, self.device)
        else:
            # Reset only selected envs
            for i in range(len(self.h_states)):
                self.h_states[i][env_indices] = 0
                self.c_states[i][env_indices] = 0

    def act(self, obs: Dict[str, torch.Tensor], dones: Optional[np.ndarray] = None, update_internal_states: bool = True):
        """
        Take a step in the environment.
        Args:
            obs: Dict of torch tensors for the current batch of envs.
            dones: np.ndarray of shape (num_envs,) indicating which envs are done.
            update_internal_states: Whether to update the internal states of the agent. Only matters for reccurrent agents.
                                    If we are *definitely* taking the action, then we should update the states. If we are 
                                    simply getting an action but don't expect to actually run it in the environment, then we 
                                    should not update the states.
        Returns:
            action, log_prob, entropy, value, (next_h_states, next_c_states)
        """
        if self.is_recurrent:
            # Reset hidden states for envs where done=True
            if dones is not None and np.any(dones):
                done_indices = np.where(dones)[0]
                self.reset_states(done_indices)
            action, log_prob, entropy, value, next_h, next_c = self.network.get_action_and_value(
                obs, h_states=self.h_states, c_states=self.c_states
            )
            if update_internal_states:
                self.h_states = next_h
                self.c_states = next_c
            return action, log_prob, entropy, value, next_h, next_c
        else:
            action, log_prob, entropy, value = self.network.get_action_and_value(obs)
            return action, log_prob, entropy, value, None, None

    def get_value(self, obs: Dict[str, torch.Tensor]):
        if self.is_recurrent:
            obs_tensor_dict = self._obs_to_torch(obs)
            value = self.network.get_value(obs_tensor_dict, self.h_states, self.c_states)
            return value
        else:
            obs_tensor_dict = self._obs_to_torch(obs)
            return self.network.get_value(obs_tensor_dict)

    def select_action(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects an action based on the current policy, also returning the state value.
        Args:
            obs_dict: The observation dictionary from the environment.
            deterministic: If True, take the action with the highest probability. Otherwise, sample.
        Returns:
            action: The selected action (np.ndarray).
            log_prob: The log probability of the selected action (np.ndarray).
            value: The estimated state value (np.ndarray).
        """
        
        # Ensure consistent batch dimension handling
        obs_board_features_np = obs_dict["board_features"]
        if obs_board_features_np.ndim == 3: # (C, H, W) - add batch dim
            obs_board_features_np = np.expand_dims(obs_board_features_np, axis=0)
        
        obs_target_idx_np = obs_dict["target_robot_idx"]
        if not isinstance(obs_target_idx_np, np.ndarray):
            obs_target_idx_np = np.array([obs_target_idx_np])
        elif obs_target_idx_np.ndim == 0:
            obs_target_idx_np = np.array([obs_target_idx_np])

        obs_tensor_dict = self._obs_to_torch({
            "board_features": obs_board_features_np,
            "target_robot_idx": obs_target_idx_np
        })
        
        self.network.eval() # Ensure network is in evaluation mode
        with torch.no_grad():
            if self.is_recurrent:
                action_logits, value_tensor, next_h_states, next_c_states = self.network(obs_tensor_dict, self.h_states, self.c_states) # network.forward
                self.h_states = [h.detach() for h in next_h_states]
                self.c_states = [c.detach() for c in next_c_states]
            else:
                action_logits, value_tensor = self.network(obs_tensor_dict) # network.forward
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action_tensor = torch.argmax(probs, dim=-1)
            else:
                action_tensor = dist.sample()
            
            log_prob_tensor = dist.log_prob(action_tensor)

        # Remove batch dimension for single environment
        action_np = action_tensor.squeeze(0).cpu().numpy()
        log_prob_np = log_prob_tensor.squeeze(0).cpu().numpy()
        value_np = value_tensor.squeeze().cpu().numpy()

        return action_np, log_prob_np, value_np

    def learn(self,
              obs_batch: List[Dict[str, np.ndarray]],
              actions_batch: np.ndarray,
              log_probs_old_batch: np.ndarray,
              advantages_batch: np.ndarray,
              returns_batch: np.ndarray,
              values_batch: np.ndarray,
              h_states_batch: Optional[np.ndarray] = None,
              c_states_batch: Optional[np.ndarray] = None): # values_batch are V(s_t) from rollout
        """
        Update the policy using PPO.
        Assumes all inputs are NumPy arrays or lists of dicts for obs.
        The rollout buffer would typically prepare these.
        Returns:
            policy_loss: The final policy loss value
            value_loss: The final value loss value
            entropy_loss: The final entropy loss value
            approx_kl: Approximate KL divergence between old and new policy
        """
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
        
        # Update total timesteps
        self.total_timesteps += len(obs_batch)

        # Convert numpy arrays to torch tensors
        actions_tensor = torch.from_numpy(actions_batch).to(self.device)
        log_probs_old_tensor = torch.from_numpy(log_probs_old_batch).to(self.device)
        advantages_tensor = torch.from_numpy(advantages_batch).to(self.device)
        returns_tensor = torch.from_numpy(returns_batch).to(self.device)
        # values_tensor = torch.from_numpy(values_batch).to(self.device) # V(s_t)
        if self.is_recurrent:
            assert h_states_batch is not None and c_states_batch is not None, "h_states_batch and c_states_batch must be provided if using recurrent model"
            h_states_tensor = [torch.from_numpy(h).to(self.device) for h in h_states_batch]
            c_states_tensor = [torch.from_numpy(c).to(self.device) for c in c_states_batch]
        else:
            h_states_tensor = None
            c_states_tensor = None

        # Normalize advantages (optional but often helpful)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Convert list of obs dicts to a batched obs dict of tensors
        obs_torch_batch = self._batch_obs_to_torch(obs_batch)

        batch_size = len(obs_batch)
        if batch_size == 0: 
            return 0.0, 0.0, 0.0, 0.0  # Nothing to learn from, return zeros

        minibatch_size = batch_size // self.num_minibatches
        if minibatch_size == 0 and batch_size > 0 : # handle small batch_size < num_minibatches
            minibatch_size = batch_size
            self.num_minibatches = 1

        # Track losses for reporting
        final_policy_loss = 0.0
        final_value_loss = 0.0
        final_entropy_loss = 0.0
        final_approx_kl = 0.0

        for _ in range(self.ppo_epochs):
            # Create a random permutation of indices for shuffling
            indices = np.random.permutation(batch_size)
            
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            epoch_kls = []
            
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                mb_indices = indices[start_idx:end_idx]

                # Slice minibatches from tensors
                mb_obs = {
                    "board_features": obs_torch_batch["board_features"][mb_indices],
                    "target_robot_idx": obs_torch_batch["target_robot_idx"][mb_indices]
                }
                mb_actions = actions_tensor[mb_indices]
                mb_log_probs_old = log_probs_old_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                # mb_values_old = values_tensor[mb_indices] # V(s_t) from rollout
                if self.is_recurrent:
                    mb_h_states = [h_states_tensor[i][mb_indices] for i in range(len(h_states_tensor))]
                    mb_c_states = [c_states_tensor[i][mb_indices] for i in range(len(c_states_tensor))]
                else:
                    mb_h_states = None
                    mb_c_states = None

                # Get new log_probs, values, and entropy from the current policy
                if self.is_recurrent:
                    _, new_log_probs, entropy, new_values, _, _ = self.network.get_action_and_value(mb_obs, mb_actions, mb_h_states, mb_c_states)
                else:
                    _, new_log_probs, entropy, new_values = self.network.get_action_and_value(mb_obs, mb_actions)

                # Policy ratio
                ratio = torch.exp(new_log_probs - mb_log_probs_old)
                
                # Calculate approximate KL divergence for logging
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - mb_log_probs_old)).mean().item()
                    epoch_kls.append(approx_kl)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                epoch_policy_losses.append(policy_loss.item())

                # Value loss (MSE against returns)
                # new_values are V_phi(s_t)
                # mb_returns are GAE_returns (R_t = A_t + V(s_t_old))
                value_loss = F.mse_loss(new_values, mb_returns)
                epoch_value_losses.append(value_loss.item())
                
                # Entropy loss (to encourage exploration)
                entropy_loss = -entropy.mean()
                epoch_entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Average losses for this epoch
            final_policy_loss = np.mean(epoch_policy_losses)
            final_value_loss = np.mean(epoch_value_losses)
            final_entropy_loss = np.mean(epoch_entropy_losses)
            final_approx_kl = np.mean(epoch_kls)

        return final_policy_loss, final_value_loss, final_entropy_loss, final_approx_kl

    def save_model(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.to(self.device) # Ensure model is on the correct device
        print(f"Model loaded from {path}") 