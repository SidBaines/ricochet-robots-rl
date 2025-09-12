from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial sequences."""
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Convolution for input-to-hidden and hidden-to-hidden
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 
            4 * hidden_channels,  # i, f, g, o gates
            kernel_size=kernel_size, 
            padding=padding
        )
        
    def forward(self, input_tensor: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ConvLSTM cell.
        
        Args:
            input_tensor: (B, C, H, W) input
            hidden_state: (h, c) tuple of hidden and cell states, each (B, hidden_channels, H, W)
            
        Returns:
            (h_new, c_new) tuple of new hidden and cell states
        """
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_g, cc_o = torch.chunk(combined_conv, 4, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)
        
        # Update cell state
        c_new = f * c_cur + i * g
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTMNetwork(nn.Module):
    """Multi-layer ConvLSTM network with optional repeats per timestep."""
    
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int = 2, 
                 num_repeats: int = 1, kernel_size: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_repeats = num_repeats
        self.hidden_channels = hidden_channels
        
        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.convlstm_layers.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size, kernel_size // 2)
            )
    
    def forward(self, x: torch.Tensor, hidden_states: list[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional hidden state management.
        
        Args:
            x: (B, C, H, W) input
            hidden_states: List of (h, c) tuples for each layer, or None for zero init
            
        Returns:
            (output, new_hidden_states) where output is the last layer's hidden state
        """
        batch_size, _, height, width = x.shape
        
        if hidden_states is None:
            # Initialize with zeros
            hidden_states = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
                c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
                hidden_states.append((h, c))
        
        # Apply ConvLSTM layers with repeats
        current_input = x
        new_hidden_states = []
        
        for layer_idx, convlstm_layer in enumerate(self.convlstm_layers):
            h, c = hidden_states[layer_idx]
            
            # Repeat the layer num_repeats times
            for _ in range(self.num_repeats):
                h, c = convlstm_layer(current_input, (h, c))
                current_input = h  # Use hidden state as input to next layer
            
            new_hidden_states.append((h, c))
        
        return current_input, new_hidden_states


class ConvLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """ConvLSTM-based feature extractor for image observations.
    
    Architecture: Conv layers -> ConvLSTM layers -> Global pooling -> FC
    """
    
    def __init__(self, observation_space: Box, features_dim: int = 128, 
                 conv_channels: int = 32, lstm_channels: int = 64, 
                 num_lstm_layers: int = 2, num_repeats: int = 1):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        in_channels = observation_space.shape[0]
        
        # Initial conv layers
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM layers
        self.convlstm = ConvLSTMNetwork(
            input_channels=conv_channels,
            hidden_channels=lstm_channels,
            num_layers=num_lstm_layers,
            num_repeats=num_repeats,
        )
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(lstm_channels, features_dim),
            nn.ReLU(inplace=True),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor, hidden_states: list[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with hidden state management.
        
        Args:
            observations: (B, C, H, W) input
            hidden_states: Optional list of (h, c) tuples for each ConvLSTM layer
            
        Returns:
            (features, new_hidden_states) where features is (B, features_dim)
        """
        # Encode with conv layers
        x = self.conv_encoder(observations)
        
        # Apply ConvLSTM
        lstm_out, new_hidden_states = self.convlstm(x, hidden_states)
        
        # Global average pooling
        features = lstm_out.mean(dim=(-2, -1))  # (B, lstm_channels)
        
        # Project to final features
        features = self.proj(features)  # (B, features_dim)
        
        return features, new_hidden_states
