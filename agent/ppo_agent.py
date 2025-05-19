import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
import torch.nn.functional as F

from .models import ActorCriticPPO # Assuming models.py is in the same directory

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
                 device: Optional[torch.device] = None):

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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.network = ActorCriticPPO(obs_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

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


    @torch.no_grad()
    def get_action_and_value(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select an action and get its log probability and the state value.
        Operates on a single observation (NumPy).
        """
        self.network.eval() # Set to evaluation mode
        
        # Add batch dimension if obs is for a single step
        obs_board_features_np = obs["board_features"]
        if obs_board_features_np.ndim == 3: # (C, H, W)
            obs_board_features_np = np.expand_dims(obs_board_features_np, axis=0)
        
        obs_target_idx_np = obs["target_robot_idx"]
        if not isinstance(obs_target_idx_np, np.ndarray) or obs_target_idx_np.ndim == 0:
            obs_target_idx_np = np.array([obs_target_idx_np])


        obs_torch = {
            "board_features": torch.from_numpy(obs_board_features_np).float().to(self.device),
            "target_robot_idx": torch.from_numpy(obs_target_idx_np).long().to(self.device)
        }
        
        action, log_prob, _, value = self.network.get_action_and_value(obs_torch)
        self.network.train() # Set back to training mode
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def get_value(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get state value for a single observation (NumPy)."""
        self.network.eval()
        obs_board_features_np = obs["board_features"]
        if obs_board_features_np.ndim == 3: # (C, H, W)
            obs_board_features_np = np.expand_dims(obs_board_features_np, axis=0)
        
        obs_target_idx_np = obs["target_robot_idx"]
        if not isinstance(obs_target_idx_np, np.ndarray) or obs_target_idx_np.ndim == 0:
            obs_target_idx_np = np.array([obs_target_idx_np])

        obs_torch = {
            "board_features": torch.from_numpy(obs_board_features_np).float().to(self.device),
            "target_robot_idx": torch.from_numpy(obs_target_idx_np).long().to(self.device)
        }
        value = self.network.get_value(obs_torch)
        self.network.train()
        return value.cpu().numpy()

    def learn(self,
              obs_batch: List[Dict[str, np.ndarray]],
              actions_batch: np.ndarray,
              log_probs_old_batch: np.ndarray,
              advantages_batch: np.ndarray,
              returns_batch: np.ndarray,
              values_batch: np.ndarray): # values_batch are V(s_t) from rollout
        """
        Update the policy using PPO.
        Assumes all inputs are NumPy arrays or lists of dicts for obs.
        The rollout buffer would typically prepare these.
        """
        # Convert numpy arrays to torch tensors
        actions_tensor = torch.from_numpy(actions_batch).to(self.device)
        log_probs_old_tensor = torch.from_numpy(log_probs_old_batch).to(self.device)
        advantages_tensor = torch.from_numpy(advantages_batch).to(self.device)
        returns_tensor = torch.from_numpy(returns_batch).to(self.device)
        # values_tensor = torch.from_numpy(values_batch).to(self.device) # V(s_t)

        # Normalize advantages (optional but often helpful)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Convert list of obs dicts to a batched obs dict of tensors
        obs_torch_batch = self._batch_obs_to_torch(obs_batch)

        batch_size = len(obs_batch)
        if batch_size == 0: return # Nothing to learn from

        minibatch_size = batch_size // self.num_minibatches
        if minibatch_size == 0 and batch_size > 0 : # handle small batch_size < num_minibatches
            minibatch_size = batch_size
            self.num_minibatches = 1


        for _ in range(self.ppo_epochs):
            # Create a random permutation of indices for shuffling
            indices = np.random.permutation(batch_size)
            
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

                # Get new log_probs, values, and entropy from the current policy
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(mb_obs, mb_actions)

                # Policy ratio
                ratio = torch.exp(new_log_probs - mb_log_probs_old)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE against returns)
                # new_values are V_phi(s_t)
                # mb_returns are GAE_returns (R_t = A_t + V(s_t_old))
                value_loss = F.mse_loss(new_values, mb_returns)
                
                # Entropy loss (to encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

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