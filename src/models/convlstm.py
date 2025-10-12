from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with tanh output gate variant (per DRC spec)."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Fused conv over [X, h_prev] → 4C (i, f, g, o)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Forget gate bias += 1.0
        with torch.no_grad():
            # bias layout: [i, f, g, o] chunks
            if self.conv.bias is not None:
                C = hidden_channels
                self.conv.bias[C:2 * C].add_(1.0)

    def forward(self, input_tensor: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden_state
        z = torch.cat([input_tensor, h_prev], dim=1)
        gates = self.conv(z)
        i_bar, f_bar, g_bar, o_bar = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i_bar)
        f = torch.sigmoid(f_bar)
        g = torch.tanh(g_bar)
        # Nonstandard: tanh for output gate activation
        o = torch.tanh(o_bar)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class DRCStack(nn.Module):
    """Deep Repeating ConvLSTM stack with pool-and-inject and boundary channel.

    Implements L layers repeated for R ticks per forward pass.
    """

    def __init__(
        self,
        encoded_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        repeats_per_step: int = 3,
        kernel_size: int = 3,
        use_boundary_channel: bool = True,
        use_pool_and_inject: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.repeats_per_step = int(repeats_per_step)
        self.kernel_size = int(kernel_size)
        self.use_boundary_channel = bool(use_boundary_channel)
        self.use_pool_and_inject = bool(use_pool_and_inject)

        self.boundary_cache: dict[tuple[int, int, torch.dtype, torch.device], torch.Tensor] = {}

        # Per-layer pool-and-inject 1x1 conv: 2C -> C
        if self.use_pool_and_inject:
            self.pi_layers = nn.ModuleList([
                nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1, bias=True)
                for _ in range(self.num_layers)
            ])
        else:
            self.pi_layers = nn.ModuleList()

        # Build ConvLSTM cells per layer with correct input channels
        self.cells = nn.ModuleList()
        for layer_index in range(self.num_layers):
            in_ch = encoded_channels + hidden_channels  # E_t + PI
            if self.use_boundary_channel:
                in_ch += 1  # boundary
            if layer_index > 0:
                in_ch += hidden_channels  # h^{l-1}
            self.cells.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
            )

    def _boundary_channel(self, batch: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, dtype, device)
        b = self.boundary_cache.get(key)
        if b is None:
            t = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
            t[:, :, 0, :] = 1.0
            t[:, :, H - 1, :] = 1.0
            t[:, :, :, 0] = 1.0
            t[:, :, :, W - 1] = 1.0
            self.boundary_cache[key] = t
            b = t
        return b.expand(batch, -1, -1, -1)

    def _pool_and_inject(self, h_prev: torch.Tensor, layer_index: int) -> torch.Tensor:
        if not self.use_pool_and_inject:
            return torch.zeros_like(h_prev)
        # Global mean + max pool → concat along channel → 1x1 conv → broadcast
        B, C, H, W = h_prev.shape
        mean_pool = torch.mean(h_prev, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(h_prev, dim=(2, 3), keepdim=True)
        concat = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2C, 1, 1]
        pi = self.pi_layers[layer_index](concat)  # [B, C, 1, 1]
        return pi.expand(B, C, H, W)

    def init_state(self, batch: int, H: int, W: int, device: torch.device, dtype: torch.dtype):
        h = [torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype) for _ in range(self.num_layers)]
        return h, c

    def forward(
        self,
        E_t: torch.Tensor,
        state: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
        repeats_override: int | None = None,
    ) -> tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        B, _Ce, H, W = E_t.shape
        device = E_t.device
        dtype = E_t.dtype
        R = self.repeats_per_step if repeats_override is None else int(repeats_override)

        if state is None:
            h_list, c_list = self.init_state(B, H, W, device, dtype)
        else:
            h_list, c_list = state

        boundary = self._boundary_channel(B, H, W, device, dtype) if self.use_boundary_channel else None

        for _tick in range(R):
            new_h_list: list[torch.Tensor] = []
            new_c_list: list[torch.Tensor] = []
            prev_layer_h: torch.Tensor | None = None
            for layer_index in range(self.num_layers):
                h_prev = h_list[layer_index]
                c_prev = c_list[layer_index]
                pi = self._pool_and_inject(h_prev, layer_index)
                inputs = [E_t, pi]
                if layer_index > 0 and prev_layer_h is not None:
                    inputs.append(prev_layer_h)
                if boundary is not None:
                    inputs.append(boundary)
                X = torch.cat(inputs, dim=1)
                h_cur, c_cur = self.cells[layer_index](X, (h_prev, c_prev))
                new_h_list.append(h_cur)
                new_c_list.append(c_cur)
                prev_layer_h = h_cur
            h_list, c_list = new_h_list, new_c_list

        # return top-layer hidden
        h_top = h_list[-1]
        return h_top, (h_list, c_list)


class ConvLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """DRC-aligned feature extractor (encoder → DRCStack → projection).

    Matches the repository's policy/training API while implementing DRC wiring.
    """

    def __init__(
        self,
        observation_space: Box,
        features_dim: int = 128,
        conv_channels: int = 32,
        lstm_channels: int = 32,
        num_lstm_layers: int = 3,
        num_repeats: int = 3,
        use_pool_and_inject: bool = True,
        use_skip_connections: bool = False,  # kept for API compatibility
    ):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected image observation (C,H,W)"
        in_channels = int(observation_space.shape[0])

        self.encoder_channels = int(conv_channels)
        self.hidden_channels = int(lstm_channels)
        self.num_layers = int(num_lstm_layers)
        self.num_repeats = int(num_repeats)
        self.use_pool_and_inject = bool(use_pool_and_inject)
        self.use_boundary = True
        # Keep for API compatibility
        self._use_skip_connections = bool(use_skip_connections)

        # Two convs then spatial downsample to reduce HxW before DRC
        # Halving spatial size cuts DRC memory/compute by ~4x
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.encoder_channels, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(self.encoder_channels, self.encoder_channels, kernel_size=3, padding=1, bias=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # DRC core
        self.drc = DRCStack(
            encoded_channels=self.encoder_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            repeats_per_step=self.num_repeats,
            kernel_size=3,
            use_boundary_channel=True,
            use_pool_and_inject=self.use_pool_and_inject,
        )

        # Projection: flatten → 256 → ReLU → features_dim
        self.proj1 = nn.Linear(self.hidden_channels * 1, 256)
        self.proj_act = nn.ReLU(inplace=True)
        self.proj2 = nn.Linear(256, features_dim)

        self._features_dim = features_dim
        # Expose an attribute named 'convlstm' so the recurrent policy recognizes
        # this extractor as recurrent and will pass/receive hidden states.
        # Keep parity with previous implementation that had a 'convlstm' module.
        self.convlstm = self.drc

    def forward(self, observations: torch.Tensor, hidden_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # Import profiling lazily
        try:
            from ..profiling import profile
        except ImportError:
            def profile(_name, _track_memory=True):
                from contextlib import nullcontext
                return nullcontext()

        with profile("convlstm_forward_pass", track_memory=True):
            x = observations
            device_type = x.device.type
            use_amp = device_type in ("cuda", "mps")
            try:
                autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if use_amp else torch.cpu.amp.autocast(enabled=False)
            except Exception:
                # Fallback if autocast not supported on backend
                class _NullCtx:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                autocast_ctx = _NullCtx()

            with autocast_ctx:
                # Encode observation
                with profile("convlstm_conv_encoder", track_memory=True):
                    E_t = self.encoder(x)

                # Reformat provided hidden_states to (h_list, c_list)
                state_tuple: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
                if hidden_states is not None and len(hidden_states) > 0:
                    h_list = [hc[0] for hc in hidden_states]
                    c_list = [hc[1] for hc in hidden_states]
                    state_tuple = (h_list, c_list)

                # Run DRC stack
                with profile("convlstm_drc_stack", track_memory=True):
                    h_top, (h_list_new, c_list_new) = self.drc(E_t, state_tuple, repeats_override=None)

                # Global average pooling over H,W
                with profile("convlstm_global_pooling", track_memory=True):
                    pooled = h_top.mean(dim=(-2, -1))  # [B, C]

                # Projection head
                with profile("convlstm_projection", track_memory=True):
                    z = self.proj1(pooled)
                    z = self.proj_act(z)
                    z = self.proj2(z)

            # Ensure output is float32 for SB3 compatibility
            z = z.float()
            # Pack new hidden states as list of (h, c)
            new_hidden_states = [(h_list_new[i], c_list_new[i]) for i in range(len(h_list_new))]
            return z, new_hidden_states
