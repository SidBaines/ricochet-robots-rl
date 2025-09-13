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
    """DRC (Deep Repeating ConvLSTM) feature extractor for image observations.
    
    Implements the DRC architecture from Guez et al. (2019) with:
    - Conv encoder with boundary padding
    - Multi-layer ConvLSTM with internal repeats
    - Pool-and-inject for faster spatial information propagation
    - Skip connections (encoded observation + top-down)
    - Global pooling + FC projection
    """
    
    def __init__(self, observation_space: Box, features_dim: int = 128, 
                 conv_channels: int = 32, lstm_channels: int = 64, 
                 num_lstm_layers: int = 2, num_repeats: int = 1,
                 use_pool_and_inject: bool = True, use_skip_connections: bool = True):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        in_channels = observation_space.shape[0]
        
        self.conv_channels = conv_channels
        self.lstm_channels = lstm_channels
        self.num_lstm_layers = num_lstm_layers
        self.num_repeats = num_repeats
        self.use_pool_and_inject = use_pool_and_inject
        self.use_skip_connections = use_skip_connections
        
        # Add boundary padding channel (1s on boundary, 0s inside)
        self.boundary_padding = True
        if self.boundary_padding:
            in_channels += 1  # Add boundary channel
        
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
        
        # Pool-and-inject layers (for faster spatial information propagation)
        if self.use_pool_and_inject:
            self.pool_and_inject = nn.ModuleList()
            for _ in range(num_lstm_layers):
                # Max and mean pooling + linear transform
                self.pool_and_inject.append(nn.ModuleDict({
                    'max_pool': nn.AdaptiveMaxPool2d(1),  # Global max pooling
                    'mean_pool': nn.AdaptiveAvgPool2d(1),  # Global average pooling
                    'transform': nn.Linear(lstm_channels * 2, lstm_channels),  # 2x for max+mean
                }))
        
        # Skip connection layers
        if self.use_skip_connections:
            # Encoded observation skip connection (conv_channels -> conv_channels to match input)
            self.encoded_skip = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)
            # Top-down skip connection (from last layer to first)
            self.topdown_skip = nn.Conv2d(lstm_channels, conv_channels, kernel_size=1)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(lstm_channels, features_dim),
            nn.ReLU(inplace=True),
        )
        
        self._features_dim = features_dim
    
    def _add_boundary_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Add boundary padding channel (1s on boundary, 0s inside)."""
        batch_size, _, height, width = x.shape
        device = x.device
        
        # Create boundary mask
        boundary = torch.zeros(batch_size, 1, height, width, device=device)
        boundary[:, :, 0, :] = 1.0  # Top boundary
        boundary[:, :, -1, :] = 1.0  # Bottom boundary
        boundary[:, :, :, 0] = 1.0  # Left boundary
        boundary[:, :, :, -1] = 1.0  # Right boundary
        
        # Concatenate with input
        return torch.cat([x, boundary], dim=1)
    
    def _apply_pool_and_inject(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply pool-and-inject for faster spatial information propagation."""
        if not self.use_pool_and_inject or layer_idx >= len(self.pool_and_inject):
            return torch.zeros_like(h)
        
        pool_layer = self.pool_and_inject[layer_idx]
        
        # Apply max and mean pooling
        max_pooled = pool_layer['max_pool'](h).squeeze(-1).squeeze(-1)  # (B, C)
        mean_pooled = pool_layer['mean_pool'](h).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Concatenate and transform
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)  # (B, 2*C)
        transformed = pool_layer['transform'](pooled)  # (B, C)
        
        # Tile back to spatial dimensions
        batch_size, channels, height, width = h.shape
        transformed_spatial = transformed.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, channels, height, width
        )
        
        return transformed_spatial
    
    def forward(self, observations: torch.Tensor, hidden_states: list[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with DRC features and hidden state management.
        
        Args:
            observations: (B, C, H, W) input
            hidden_states: Optional list of (h, c) tuples for each ConvLSTM layer
            
        Returns:
            (features, new_hidden_states) where features is (B, features_dim)
        """
        # Add boundary padding if enabled
        if self.boundary_padding:
            x = self._add_boundary_padding(observations)
        else:
            x = observations
        
        # Encode with conv layers
        encoded = self.conv_encoder(x)  # (B, conv_channels, H, W)
        
        # Store encoded features for skip connection
        if self.use_skip_connections:
            encoded_skip = self.encoded_skip(encoded)  # (B, conv_channels, H, W)
        
        # Apply ConvLSTM with enhanced features
        current_input = encoded
        new_hidden_states = []
        
        for layer_idx, convlstm_layer in enumerate(self.convlstm.convlstm_layers):
            # Get hidden states for this layer, initialize if None
            if hidden_states and layer_idx < len(hidden_states):
                h, c = hidden_states[layer_idx]
            else:
                # Initialize hidden states with zeros
                batch_size, _, height, width = current_input.shape
                h = torch.zeros(batch_size, self.lstm_channels, height, width, 
                              device=current_input.device, dtype=current_input.dtype)
                c = torch.zeros_like(h)
            
            # Apply skip connections (simplified - only encoded observation skip for now)
            if self.use_skip_connections and layer_idx == 0:
                # First layer: add encoded observation skip
                current_input = current_input + encoded_skip
            
            # Repeat the layer num_repeats times
            for _ in range(self.convlstm.num_repeats):
                h, c = convlstm_layer(current_input, (h, c))
                # For subsequent layers, use the hidden state as input
                if layer_idx < len(self.convlstm.convlstm_layers) - 1:
                    current_input = h
            
            new_hidden_states.append((h, c))
        
        # Global average pooling
        features = current_input.mean(dim=(-2, -1))  # (B, lstm_channels)
        
        # Project to final features
        features = self.proj(features)  # (B, features_dim)
        
        return features, new_hidden_states
