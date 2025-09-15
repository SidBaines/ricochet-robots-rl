from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.utils import get_device
from gymnasium.spaces import Box, Discrete


class RecurrentActorCriticPolicy(BasePolicy):
    """
    Custom recurrent policy that properly handles hidden states across time steps.
    
    This policy maintains hidden states internally and provides them to the feature extractor
    on each forward pass, enabling true recurrent behavior for planning tasks.
    """
    
    def __init__(
        self,
        observation_space: Box,
        action_space: Discrete,
        lr_schedule: callable,
        features_extractor_class: type[BaseFeaturesExtractor],
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        net_arch: Optional[List[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_dim: int = 128,
        normalize_images: bool = True,
        initial_std: float = 1.0,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        # Set essential attributes
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.ortho_init = ortho_init
        self.normalize_images = normalize_images
        # Store unused-but-standard args to avoid lint warnings and for future use
        self._use_sde = use_sde
        self._log_std_init = log_std_init
        self._full_std = full_std
        self._use_expln = use_expln
        self._initial_std = initial_std
        
        # Initialize hidden states storage
        self._hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._batch_size: Optional[int] = None
        self._device = get_device("auto")
        
        # Create feature extractor first
        self.features_extractor = self.features_extractor_class(
            self.observation_space, 
            **self.features_extractor_kwargs
        )
        
        # Get the actual features_dim from the feature extractor
        self.features_dim = self.features_extractor._features_dim
        
        # Build the action and value networks
        self._build_networks()
        
        # Initialize optimizer (required by SB3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observations to float in [0,1] when normalize_images is enabled."""
        if self.normalize_images:
            if obs.dtype == torch.uint8:
                obs = obs.float().div_(255.0)
            else:
                # Ensure float32 dtype for convs and autocast stability
                obs = obs.float()
        return obs

    def _build_networks(self):
        """Build the actor and critic networks."""
        # Extract network architecture
        pi_arch = self.net_arch.get('pi', [128, 128])
        vf_arch = self.net_arch.get('vf', [128, 128])
        
        # Action network (policy head) - simple 2-layer MLP
        self.action_net = nn.Sequential(
            nn.Linear(self.features_dim, pi_arch[0]),
            self.activation_fn(),
            nn.Linear(pi_arch[0], pi_arch[1]),
            self.activation_fn(),
            nn.Linear(pi_arch[1], self.action_space.n)
        )
        
        # Value network (critic head) - simple 2-layer MLP
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, vf_arch[0]),
            self.activation_fn(),
            nn.Linear(vf_arch[0], vf_arch[1]),
            self.activation_fn(),
            nn.Linear(vf_arch[1], 1)
        )
        
        # Initialize weights
        if self.ortho_init:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _init_hidden_states(self, _batch_size: int, _device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for the recurrent feature extractor."""
        if not hasattr(self.features_extractor, 'convlstm'):
            return []
        
        # We need to know the spatial dimensions, but we don't have them yet
        # Return None and let the feature extractor handle initialization
        return None
    
    def _get_hidden_states(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get or initialize hidden states for the current batch."""
        if self._hidden_states is None or self._batch_size != batch_size:
            self._hidden_states = self._init_hidden_states(batch_size, self._device)
            self._batch_size = batch_size
        
        return self._hidden_states
    
    def reset_hidden_states(self, batch_size: Optional[int] = None):
        """Reset hidden states to zero."""
        if batch_size is not None:
            self._batch_size = batch_size
        self._hidden_states = None
    
    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the recurrent policy.
        
        Args:
            obs: Observation tensor (B, C, H, W) or (B, features_dim)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            value: State value estimate
            log_prob: Log probability of the selected action
        """
        # Import profiling here to avoid circular imports
        try:
            from profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(_name, _track_memory=True):
                from contextlib import nullcontext
                return nullcontext()
        
        with profile("recurrent_policy_forward", track_memory=True):
            obs = self._preprocess_obs(obs)
            batch_size = obs.shape[0]
            
            # Get current hidden states
            with profile("recurrent_policy_get_hidden_states", track_memory=True):
                hidden_states = self._get_hidden_states(batch_size)
            
            # Extract features using the recurrent feature extractor
            if hasattr(self.features_extractor, 'convlstm'):
                # Recurrent feature extractor - pass hidden states
                features, new_hidden_states = self.features_extractor(obs, hidden_states)
                # Update stored hidden states (detach to prevent gradient issues)
                self._hidden_states = [(h.detach(), c.detach()) for h, c in new_hidden_states]
            else:
                # Non-recurrent feature extractor
                features = self.features_extractor(obs)
            
            # Compute action logits and value
            with profile("recurrent_policy_action_value", track_memory=True):
                action_logits = self.action_net(features)
                value = self.value_net(features).squeeze(-1)
            
            # Create action distribution
            with profile("recurrent_policy_action_dist", track_memory=True):
                action_dist = self._get_action_dist_from_latent(action_logits)
            
            # Sample action
            with profile("recurrent_policy_action_sampling", track_memory=True):
                if deterministic:
                    action = action_dist.mode()
                else:
                    action = action_dist.sample()
                
                # Compute log probability
                log_prob = action_dist.log_prob(action)
            
            return action, value, log_prob
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """Get action distribution from latent policy."""
        from stable_baselines3.common.distributions import CategoricalDistribution
        dist = CategoricalDistribution(latent_pi)
        dist.proba_distribution(latent_pi)
        return dist
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            value: State value estimate
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
        """
        # Import profiling here to avoid circular imports
        try:
            from profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(_name, _track_memory=True):
                from contextlib import nullcontext
                return nullcontext()
        
        with profile("recurrent_policy_evaluate_actions", track_memory=True):
            obs = self._preprocess_obs(obs)
            batch_size = obs.shape[0]
            
            # Get current hidden states
            hidden_states = self._get_hidden_states(batch_size)
            
            # Extract features
            if hasattr(self.features_extractor, 'convlstm'):
                features, new_hidden_states = self.features_extractor(obs, hidden_states)
                # Update stored hidden states (detach to prevent gradient issues)
                self._hidden_states = [(h.detach(), c.detach()) for h, c in new_hidden_states]
            else:
                features = self.features_extractor(obs)
            
            # Compute action logits and value
            action_logits = self.action_net(features)
            value = self.value_net(features).squeeze(-1)
            
            # Create action distribution
            action_dist = self._get_action_dist_from_latent(action_logits)
            
            # Compute log probability and entropy
            log_prob = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            
            return value, log_prob, entropy
    
    def get_hidden_states(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get current hidden states for analysis."""
        return self._hidden_states
    
    def set_hidden_states(self, hidden_states: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Set hidden states (useful for analysis and interventions)."""
        self._hidden_states = hidden_states
        if hidden_states:
            self._batch_size = hidden_states[0][0].shape[0]
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action and value for a single observation.
        
        Args:
            observation: Single observation tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            value: State value estimate
        """
        # Add batch dimension if needed
        if observation.dim() == 3:  # (C, H, W)
            observation = observation.unsqueeze(0)  # (1, C, H, W)
        
        observation = self._preprocess_obs(observation)
        action, value, _ = self.forward(observation, deterministic)
        
        # Remove batch dimension for single observation
        return action.squeeze(0), value.squeeze(0)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values for observations (used by SB3).
        
        Args:
            obs: Observation tensor
            
        Returns:
            values: State value estimates
        """
        # Import profiling here to avoid circular imports
        try:
            from profiling import profile
        except ImportError:
            # Fallback if profiling not available
            def profile(_name, _track_memory=True):
                from contextlib import nullcontext
                return nullcontext()
        
        with profile("recurrent_policy_predict_values", track_memory=True):
            obs = self._preprocess_obs(obs)
            batch_size = obs.shape[0]
            
            # Get current hidden states
            hidden_states = self._get_hidden_states(batch_size)
            
            # Extract features using the recurrent feature extractor
            if hasattr(self.features_extractor, 'convlstm'):
                # Recurrent feature extractor - pass hidden states
                features, new_hidden_states = self.features_extractor(obs, hidden_states)
                # Update stored hidden states (detach to prevent gradient issues)
                self._hidden_states = [(h.detach(), c.detach()) for h, c in new_hidden_states]
            else:
                # Non-recurrent feature extractor
                features = self.features_extractor(obs)
            
            # Compute value
            value = self.value_net(features).squeeze(-1)
            
            return value
