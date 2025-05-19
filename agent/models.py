import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class ActorCriticPPO(nn.Module):
    def __init__(self, obs_space: spaces.Dict, action_space: spaces.Discrete):
        super().__init__()

        board_shape = obs_space["board_features"].shape # (num_robots + 1, height, width)
        num_robots = obs_space["target_robot_idx"].n
        
        # CNN for board features
        # Input channels: num_robots (for their positions) + 1 (for target position)
        in_channels = board_shape[0] 
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        # To do this, we pass a dummy tensor through the CNN
        dummy_board_features = torch.zeros(1, *board_shape) # Batch size of 1
        with torch.no_grad():
            cnn_out_dim = self.cnn(dummy_board_features).shape[1]

        # We also need to incorporate the target_robot_idx.
        # An embedding layer is a good way to handle discrete inputs.
        self.target_robot_embedding_dim = 16 # Hyperparameter
        self.target_robot_embed = nn.Embedding(num_robots, self.target_robot_embedding_dim)

        # Combined feature size
        combined_features_dim = cnn_out_dim + self.target_robot_embedding_dim

        # Actor head (outputs action logits)
        self.actor_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n) # Outputs logits for each action
        )

        # Critic head (outputs state value)
        self.critic_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single value
        )

    def _process_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper to process the observation dictionary and combine features."""
        board_features = obs["board_features"]
        target_robot_idx = obs["target_robot_idx"]

        # Ensure board_features is float (CNNs expect float)
        if board_features.dtype == torch.uint8:
            board_features = board_features.float()
        
        # If target_robot_idx is a float (e.g. from a batch), cast to long for embedding
        if target_robot_idx.dtype != torch.long:
            target_robot_idx = target_robot_idx.long()
        # If target_robot_idx is 0-dim (scalar), unsqueeze it
        if target_robot_idx.ndim == 0:
            target_robot_idx = target_robot_idx.unsqueeze(0)
        # If target_robot_idx has an extra singleton dimension (e.g. [B, 1]), squeeze it
        if target_robot_idx.ndim == 2 and target_robot_idx.shape[1] == 1:
            target_robot_idx = target_robot_idx.squeeze(1)


        cnn_out = self.cnn(board_features)
        target_embed = self.target_robot_embed(target_robot_idx)
        
        combined_features = torch.cat([cnn_out, target_embed], dim=1)
        return combined_features

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the actor-critic network.
        Args:
            obs: A dictionary containing 'board_features' and 'target_robot_idx'.
        Returns:
            action_logits: Logits for the action probability distribution.
            value: The estimated state value.
        """
        combined_features = self._process_obs(obs)
        action_logits = self.actor_head(combined_features)
        value = self.critic_head(combined_features)
        return action_logits, value

    def get_action_and_value(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy, log probability of action, entropy of policy, and state value.
        Args:
            obs: The observation dictionary.
            action: (Optional) If provided, compute log_prob and entropy for this action.
                    Otherwise, sample an action from the policy.
        Returns:
            sampled_action: The action sampled from the policy.
            log_prob: The log probability of the action.
            entropy: The entropy of the action distribution.
            value: The estimated state value.
        """
        action_logits, value = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            sampled_action = dist.sample()
        else:
            sampled_action = action
        
        log_prob = dist.log_prob(sampled_action)
        entropy = dist.entropy()
        
        return sampled_action, log_prob, entropy, value.squeeze(-1) # Squeeze value to be 1D

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get only the state value."""
        _ , value = self.forward(obs)
        return value.squeeze(-1) # Squeeze value to be 1D 