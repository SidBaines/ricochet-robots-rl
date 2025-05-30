import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class ConvLSTMCell(nn.Module):
    """A basic ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h, c):
        # x: (B, input_dim, H, W)
        # h, c: (B, hidden_dim, H, W)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class DeepRepeatedConvLSTM(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Dict,
        action_space: spaces.Discrete,
        num_layers: int = 3,
        hidden_dims: Tuple[int, ...] = (32, 32, 32),
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        repeat_K: int = 3,
        pooling: str = "avg",
        target_robot_embedding_dim: int = 16,
    ):
        super().__init__()
        board_shape = obs_space["board_features"].shape  # (C, H, W)
        num_robots = obs_space["target_robot_idx"].n
        in_channels = board_shape[0]
        self.height, self.width = board_shape[1], board_shape[2]
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.repeat_K = repeat_K
        self.pooling = pooling
        self.encoder_out_dim = 32

        # Encoder 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.encoder_out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # ConvLSTM stack
        self.convlstm_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = self.encoder_out_dim if i == 0 else hidden_dims[i-1]
            self.convlstm_cells.append(
                ConvLSTMCell(input_dim, hidden_dims[i], kernel_sizes[i])
            )
        # 1x1 conv + ReLU after each ConvLSTM layer
        self.post_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i], kernel_size=1),
                nn.ReLU()
            ) for i in range(num_layers)
        ])

        # Target robot embedding
        self.target_robot_embed = nn.Embedding(num_robots, target_robot_embedding_dim)

        # Output size after pooling
        conv_out_dim = hidden_dims[-1]
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("Unknown pooling type")

        # Final feature size
        combined_features_dim = conv_out_dim + target_robot_embedding_dim

        # Policy and value heads
        self.actor_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize value function to predict 0 for initial states
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize policy to be more random initially
        for m in self.actor_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def get_initial_states(self, batch_size: int, device: torch.device = None):
        """Return a tuple of (h_states, c_states) for all layers, each a list of tensors."""
        device = device or next(self.parameters()).device
        H, W = self.height, self.width
        h_states = [torch.zeros(batch_size, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]
        return h_states, c_states

    def _process_obs(self, obs: Dict[str, torch.Tensor], h_states, c_states):
        board_features = obs["board_features"]
        target_robot_idx = obs["target_robot_idx"]

        # Ensure correct dtype and shape
        if board_features.dtype == torch.uint8:
            board_features = board_features.float()
        if board_features.ndim == 3:
            board_features = board_features.unsqueeze(0)  # Ensure batch dimension
        if target_robot_idx.dtype != torch.long:
            target_robot_idx = target_robot_idx.long()
        if target_robot_idx.ndim == 0:
            target_robot_idx = target_robot_idx.unsqueeze(0)
        if target_robot_idx.ndim == 2 and target_robot_idx.shape[1] == 1:
            target_robot_idx = target_robot_idx.squeeze(1)

        # Repeat the ConvLSTM stack K times, carrying hidden/cell state
        encoded_features = self.encoder(board_features)
        for _ in range(self.repeat_K):
            for i, cell in enumerate(self.convlstm_cells):
                h, c = h_states[i-1], c_states[i-1]
                h_next, c_next = cell(encoded_features, h, c)
                # h_next = self.post_convs[i](h_next)
                h_states[i], c_states[i] = h_next, c_next

        # After K repeats, x is the output of the last ConvLSTM+1x1+ReLU
        x = h_states[-1] + encoded_features
        pooled = self.pool(x).view(x.shape[0], -1)  # (B, hidden_dim)
        target_embed = self.target_robot_embed(target_robot_idx)
        combined = torch.cat([pooled, target_embed], dim=1)
        return combined, h_states, c_states

    def forward(self, obs: Dict[str, torch.Tensor], h_states=None, c_states=None):
        # Ensure batch dimension
        if obs["board_features"].ndim == 3:
            B = 1
        else:
            B = obs["board_features"].shape[0]
        device = obs["board_features"].device
        if h_states is None or c_states is None:
            h_states, c_states = self.get_initial_states(B, device)
        features, next_h_states, next_c_states = self._process_obs(obs, h_states, c_states)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value, next_h_states, next_c_states

    def get_action_and_value(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None, h_states=None, c_states=None):
        action_logits, value, next_h_states, next_c_states = self.forward(obs, h_states, c_states)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            sampled_action = dist.sample()
        else:
            sampled_action = action
        log_prob = dist.log_prob(sampled_action)
        entropy = dist.entropy()
        return sampled_action, log_prob, entropy, value.squeeze(-1), next_h_states, next_c_states

    def get_value(self, obs: Dict[str, torch.Tensor], h_states=None, c_states=None) -> torch.Tensor:
        _, value, _, _ = self.forward(obs, h_states, c_states)
        return value.squeeze(-1)

class ActorCriticPPO(nn.Module):
    def __init__(self, obs_space: spaces.Dict, action_space: spaces.Discrete):
        super().__init__()

        board_shape = obs_space["board_features"].shape  # (num_robots + 1 + 4, height, width)
        num_robots = obs_space["target_robot_idx"].n
        
        # CNN for board features
        # Input channels: num_robots (for their positions) + 1 (for target position) + 4 (for walls)
        in_channels = board_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        # To do this, we pass a dummy tensor through the CNN
        dummy_board_features = torch.zeros(1, *board_shape)  # Batch size of 1
        with torch.no_grad():
            cnn_out_dim = self.cnn(dummy_board_features).shape[1]

        # We also need to incorporate the target_robot_idx.
        # An embedding layer is a good way to handle discrete inputs.
        self.target_robot_embedding_dim = 16  # Hyperparameter
        self.target_robot_embed = nn.Embedding(num_robots, self.target_robot_embedding_dim)

        # Combined feature size
        combined_features_dim = cnn_out_dim + self.target_robot_embedding_dim

        # Actor head (outputs action logits)
        self.actor_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)  # Outputs logits for each action
        )

        # Critic head (outputs state value)
        self.critic_head = nn.Sequential(
            nn.Linear(combined_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs a single value
        )

    def _process_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper to process the observation dictionary and combine features."""
        board_features = obs["board_features"]
        target_robot_idx = obs["target_robot_idx"]

        # Ensure board_features is float (CNNs expect float)
        if board_features.dtype == torch.uint8:
            board_features = board_features.float()
        if board_features.ndim == 3:
            board_features = board_features.unsqueeze(0)  # Ensure batch dimension
        
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