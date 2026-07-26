"""Faithful PyTorch reimplementation of the DRC architecture (Deep Repeated ConvLSTM).

Primary source
    [G19] Guez et al., "An Investigation of Model-Free Planning", ICML 2019,
          arXiv:1901.03559.  Architecture: Sec. 2.1.1-2.1.2; equations and
          extras: Appendix 8 (8.1.2 pool-and-inject, 8.2.1 ConvLSTM equations);
          Sokoban hyperparameters: Appendix 9.1-9.2.

Cross-checks (used to resolve ambiguities in [G19])
    [T24] Taufeeque et al., "Planning in a recurrent neural network that plays
          Sokoban", 2024, arXiv:2407.15421 (appendix A: architecture details).
    [TLP] AlignmentResearch/train-learned-planner, `cleanba/convlstm.py` and
          `cleanba/config.py::sokoban_drc33_59` -- the faithful DRC(3,3)
          retraining whose trained checkpoint (run `drc33/bkynosqi`) is the
          subject of [T24].  Config verified from
          `learned-planner/plot/learning_curves/drc_33/bkynosqi.cfg.json`.

Architecture summary (DRC(D,N), [G19] Sec 2.1.1 + Appendix 8.2.1)
    * An encoder CNN embeds the observation once per environment step into a
      spatial map ``i_t``.
    * A stack of D ConvLSTM modules is applied N times ("ticks") per
      environment step; the SAME encoded observation ``i_t`` is fed to every
      module at every tick ([G19] App. 8: "the encoded observation i_t is fed
      as an input not only at all ticks 1..N, but also at all depths 1..D").
    * Module d at tick n receives: ``i_t``, the hidden state of the module
      below at the current tick ``h_{d-1}^n``, and its own previous state
      ``(h_d^{n-1}, c_d^{n-1})``.  For d=1 the "module below" input is the
      top-down skip connection ``h_D^{n-1}`` (last module, previous tick).
    * Pool-and-inject ([G19] App. 8.1.2): each module's own previous hidden
      state is max- and mean-pooled over space, linearly projected, tiled back
      over space, and appended to the module's input at the next tick.
    * Boundary "fence" feature ([G19] Sec 2.1.2): a feature map with ones on
      the boundary and zeros inside is appended to the convolution input.
    * Readout ([G19] Sec 2.1.1 + App. 9.1): the last module's hidden state
      after the final tick is combined with ``i_t`` through a skip connection,
      flattened, passed through a 256-unit fully connected layer with ReLU,
      then two linear heads produce policy logits and a scalar value.

All ambiguities of the paper text and the interpretation chosen (following
[TLP]/[T24]) are documented in docs/DrcFaithfulReimplementation.md.  The
constructor defaults below reproduce the verified faithful Sokoban DRC(3,3)
configuration exactly; env-specific quantities (input shape, number of
actions) have no meaningful universal default and are required arguments.

This module is self-contained (PyTorch only) and holds no hidden recurrent
state: the state is created by :meth:`DRCNet.initial_state`, passed into
:meth:`DRCNet.forward`, and returned updated.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ConvLSTMCell",
    "DRCNet",
    "DRCState",
    "sokoban_drc33",
    "sokoban_drc11",
]

# Mean(|N(0,1)| truncated to [-2, 2]) std correction used by JAX/Flax
# ``variance_scaling(..., "truncated_normal")`` initializers; [TLP] uses those
# (flax ``lecun_normal``) for every convolution and dense layer.
_TRUNC_STD_CORRECTION = 0.87962566103423978

# Additive logit value for masked-out actions (env adaptation, not in [G19]).
LOGIT_MASK_VALUE = -1e9


def _lecun_truncated_normal_(tensor: Tensor, fan_in: int) -> None:
    """Flax ``lecun_normal`` init: truncated normal (+/- 2 sigma), std sqrt(1/fan_in).

    [G19] does not specify initialization; this follows [TLP]/[T24] ("normal
    weights truncated at 2 standard deviations and scaled to have standard
    deviation 1/sqrt(fan_in)"; biases zero).
    """
    std = math.sqrt(1.0 / fan_in) / _TRUNC_STD_CORRECTION
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class _SameConv2d(nn.Conv2d):
    """Conv2d with TensorFlow/JAX "SAME" padding semantics (any stride).

    Output size is ceil(input / stride); total padding ``(out-1)*s + k - in``
    is split low = total//2, high = total - low, exactly as in
    ``jax.lax.padtype_to_pads`` (the reference implementation [TLP] uses flax
    ``nn.Conv(..., padding="SAME")`` everywhere).  For stride 1 and odd
    kernels this is ordinary symmetric zero padding.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)

    @staticmethod
    def _same_pad(size: int, kernel: int, stride: int) -> Tuple[int, int]:
        out = -(-size // stride)  # ceil division
        total = max((out - 1) * stride + kernel - size, 0)
        low = total // 2
        return low, total - low

    def forward(self, x: Tensor) -> Tensor:
        top, bottom = self._same_pad(x.shape[-2], self.kernel_size[0], self.stride[0])
        left, right = self._same_pad(x.shape[-1], self.kernel_size[1], self.stride[1])
        if top or bottom or left or right:
            x = F.pad(x, (left, right, top, bottom))
        return self._conv_forward(x, self.weight, self.bias)


class DRCState(NamedTuple):
    """Recurrent state of the DRC stack: ``s = (c_1..c_D, h_1..h_D)`` ([G19] Sec 2.1.1).

    Both tensors have shape ``(D, B, C, H, W)`` where ``(H, W)`` is the spatial
    size of the encoder output and ``C = hidden_channels``.
    """

    h: Tensor
    c: Tensor


class ConvLSTMCell(nn.Module):
    """One ConvLSTM module of the DRC stack ([G19] Appendix 8.2.1).

    Computes, for module d at tick n (all ``*`` are 2-D convolutions)::

        gates = W_ih * cat(i_t, h_below, pool_inject) + W_hh * h_d^{n-1} + W_fence * fence
        g, i, f, o = split(gates, 4)                     # each `hidden_channels` wide
        c_d^n = sigmoid(f + forget_bias) . c_d^{n-1} + sigmoid(i) . tanh(g)
        h_d^n = act_o(o) . tanh(c_d^n)

    where ``h_below`` is ``h_{d-1}^n`` (or the top-down skip ``h_D^{n-1}`` for
    d=1), ``pool_inject`` is the spatially tiled projection of the max/mean
    spatial pools of this module's own ``h_d^{n-1}`` ([G19] App. 8.1.2), and
    ``fence`` is the ones-on-boundary feature map ([G19] Sec 2.1.2).

    Faithfulness notes:
      * [G19] App. 9.1: "all ConvLSTM modules use one convolution with 128
        output channels, kernel size 3, stride 1. All cell states and hidden
        states have 32 channels" -- i.e. a single fused convolution computes
        all four gate pre-activations (4 x 32 = 128).  The gate chunk order is
        ``(g=candidate[tanh], i=input[sigmoid], f=forget[sigmoid],
        o=output)``, matching [TLP] (their variables ``i, j, f, o``) so that
        weights are portable.
      * The recurrent convolution ``W_hh`` and the fence convolution carry no
        bias; the input convolution ``W_ih`` does ([TLP]; [G19] writes a
        single shared bias per gate).
      * ``output_activation``: [G19] eq. in App. 8.2.1 writes sigmoid; the
        verified faithful retraining [TLP]/[T24] uses tanh ("a nonstandard
        tanh on the output gate", cf. Jozefowicz et al. 2015).  Default here
        is ``"tanh"`` to match the verified network; pass ``"sigmoid"`` for
        the literal paper equation.
      * ``fence``: [G19] says the boundary map is appended "to the input of
        the convolutional operators".  [TLP]/[T24] implement it as a ring of
        ones OUTSIDE the H x W grid (a ``(H+k-1, W+k-1)`` map convolved
        VALID), i.e. the ring lives where zero padding would be
        (``fence="valid"``, the default, matching the trained network).
        ``fence="same"`` instead places the ring on the boundary cells of the
        H x W map itself; ``fence="none"`` disables it.
      * ``pool_projection``: [G19] App. 8.1.2 defines
        ``p = Tile(W_p [max(h), mean(h)]^T)`` without giving ``W_p``'s shape.
        ``"per-channel"`` (default; used by the verified network) learns one
        (w_max, w_mean) pair per channel; ``"full"`` is a dense (2C -> C)
        projection.

    Args:
        input_channels: channels of the encoded observation ``i_t``.
        hidden_channels: channels of ``h`` and ``c`` (paper: 32).
        kernel_size: convolution kernel (paper: 3), stride fixed to 1.
        pool_and_inject: enable pool-and-inject (paper: on).
        pool_projection: ``"per-channel"`` or ``"full"`` (see above).
        output_activation: ``"tanh"`` (default, verified network) or ``"sigmoid"``.
        forget_bias: constant added to the forget-gate pre-activation before
            the sigmoid (verified network: 0.0).
        fence: ``"valid"`` (default), ``"same"`` or ``"none"`` (see above).
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        pool_and_inject: bool = True,
        pool_projection: str = "per-channel",
        output_activation: str = "tanh",
        forget_bias: float = 0.0,
        fence: str = "valid",
    ):
        super().__init__()
        if output_activation not in ("tanh", "sigmoid"):
            raise ValueError(f"output_activation must be 'tanh' or 'sigmoid', got {output_activation!r}")
        if fence not in ("valid", "same", "none"):
            raise ValueError(f"fence must be 'valid', 'same' or 'none', got {fence!r}")
        if pool_projection not in ("per-channel", "full"):
            raise ValueError(f"pool_projection must be 'per-channel' or 'full', got {pool_projection!r}")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pool_and_inject = pool_and_inject
        self.pool_projection = pool_projection
        self.output_activation = output_activation
        self.forget_bias = forget_bias
        self.fence = fence

        gates = 4 * hidden_channels
        in_ch = input_channels + hidden_channels  # i_t  +  h_below
        if pool_and_inject:
            in_ch += hidden_channels  # + tiled pool projection
            if pool_projection == "per-channel":
                # (2, C): row 0 scales the max-pool, row 1 the mean-pool.
                self.pool_project = nn.Parameter(torch.empty(2, hidden_channels))
            else:
                self.pool_project = nn.Linear(2 * hidden_channels, hidden_channels, bias=False)

        self.conv_ih = _SameConv2d(in_ch, gates, kernel_size, bias=True)
        self.conv_hh = _SameConv2d(hidden_channels, gates, kernel_size, bias=False)
        if fence == "valid":
            self.conv_fence = nn.Conv2d(1, gates, kernel_size, padding=0, bias=False)
        elif fence == "same":
            self.conv_fence = _SameConv2d(1, gates, kernel_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        k2 = self.kernel_size * self.kernel_size
        _lecun_truncated_normal_(self.conv_ih.weight, fan_in=self.conv_ih.in_channels * k2)
        nn.init.zeros_(self.conv_ih.bias)
        _lecun_truncated_normal_(self.conv_hh.weight, fan_in=self.hidden_channels * k2)
        if self.fence != "none":
            _lecun_truncated_normal_(self.conv_fence.weight, fan_in=k2)
        if self.pool_and_inject:
            if self.pool_projection == "per-channel":
                _lecun_truncated_normal_(self.pool_project, fan_in=2)
            else:
                _lecun_truncated_normal_(self.pool_project.weight, fan_in=2 * self.hidden_channels)

    def _fence_map(self, height: int, width: int, ref: Tensor) -> Tensor:
        """Ones-on-boundary map ([G19] Sec 2.1.2), sized per the fence mode ([TLP])."""
        if self.fence == "valid":
            height = height + self.kernel_size - 1
            width = width + self.kernel_size - 1
        fence = torch.ones(1, 1, height, width, device=ref.device, dtype=ref.dtype)
        fence[:, :, 1:-1, 1:-1] = 0.0
        return fence

    def _pool_and_project(self, own_h: Tensor) -> Tensor:
        """[G19] App. 8.1.2: m = [max_HW(h), mean_HW(h)]; p = Tile_HW(W_p m)."""
        b, c, height, width = own_h.shape
        h_max = own_h.amax(dim=(2, 3))  # (B, C)
        h_mean = own_h.mean(dim=(2, 3))  # (B, C)
        if self.pool_projection == "per-channel":
            pooled = self.pool_project[0] * h_max + self.pool_project[1] * h_mean
        else:
            pooled = self.pool_project(torch.cat([h_max, h_mean], dim=1))
        return pooled.view(b, c, 1, 1).expand(b, c, height, width)

    def forward(self, inputs: Tensor, below_h: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """One tick of this module.

        Args:
            inputs: encoded observation ``i_t``, shape (B, input_channels, H, W).
            below_h: ``h_{d-1}^n`` (or top-down skip ``h_D^{n-1}`` for the
                bottom module), shape (B, hidden_channels, H, W).
            state: this module's own ``(h_d^{n-1}, c_d^{n-1})``.

        Returns:
            ``(h_d^n, c_d^n)``.
        """
        h_prev, c_prev = state
        parts = [inputs, below_h]
        if self.pool_and_inject:
            parts.append(self._pool_and_project(h_prev))
        cell_inputs = torch.cat(parts, dim=1)

        gates = self.conv_ih(cell_inputs) + self.conv_hh(h_prev)
        if self.fence != "none":
            gates = gates + self.conv_fence(self._fence_map(inputs.shape[-2], inputs.shape[-1], ref=inputs))

        # Chunk order (g, i, f, o) matches [TLP] ``i, j, f, o`` for weight portability.
        g, i, f, o = gates.chunk(4, dim=1)
        candidate = torch.tanh(g)
        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f + self.forget_bias)
        if self.output_activation == "tanh":
            output_gate = torch.tanh(o)
        else:
            output_gate = torch.sigmoid(o)

        c_new = forget_gate * c_prev + input_gate * candidate
        h_new = output_gate * torch.tanh(c_new)
        return h_new, c_new


class DRCNet(nn.Module):
    """DRC(D, N) actor-critic network ([G19] Sec 2.1, Appendices 8-9).

    Default keyword arguments reproduce the Sokoban DRC(3,3) of [G19] as
    pinned down by the faithful retraining [TLP]/[T24] (verified against the
    released config of the trained checkpoint ``drc33/bkynosqi`` and its
    parameter count).  ``observation_shape`` and ``n_actions`` are
    environment-specific and therefore required.

    Encoder: [G19] App. 9.1 uses, for 80x80x3 *pixel* Sokoban observations,
    channels (32, 32), kernels (8, 4), strides (4, 2) -- reaching a 10x10x32
    map.  [TLP]/[T24] train on 10x10x3 tile observations and use channels
    (32, 32), kernels (4, 4), strides (1, 1), no inter-layer ReLU, reaching
    the same 10x10x32.  The defaults here are the latter (stride-1) variant,
    appropriate for small symbolic grids like this repo's boards; pass
    ``encoder_channels=(32, 32), encoder_kernels=(8, 4), encoder_strides=(4, 2),
    encoder_relu=True`` to rebuild the paper's pixel encoder.  All encoder
    convolutions use SAME padding and biases.

    Readout ([G19] Sec 2.1.1, App. 9.1; [T24] App.): ``h_D`` after the last
    tick, plus ``i_t`` via a skip connection (elementwise addition, per
    [TLP] ``skip_final``), flattened, then Linear(256) + ReLU, then separate
    linear heads for policy logits and value.  Head weights are orthogonal
    (gain 1), biases zero ([TLP]; not specified in [G19]).

    Input preprocessing is NOT included: the reference pipeline divides RGB
    observations by 255 outside the network ([TLP] ``Policy``); do the
    equivalent for your observation encoding before calling ``forward``.

    Args:
        observation_shape: (C, H, W) of the observation tensor.
        n_actions: number of discrete actions ([G19]/[T24] Sokoban: 4).
        depth: D, number of stacked ConvLSTM modules (paper: 3).
        ticks: N, module-stack applications per environment step (paper: 3).
        hidden_channels: ConvLSTM h/c channels (paper: 32).
        cell_kernel_size: ConvLSTM kernel (paper: 3, stride 1).
        encoder_channels / encoder_kernels / encoder_strides: encoder conv
            stack (see above).  Empty tuples mean "no encoder" (raw
            observation fed to the stack).
        encoder_relu: ReLU between encoder convolutions (never after the last
            one, per [TLP]); the verified DRC(3,3) uses False.
        mlp_hidden: readout hidden width (paper: 256).
        encoder_skip: add ``i_t`` to ``h_D`` before the readout (paper: on).
        pool_and_inject / pool_projection / output_activation / forget_bias /
            fence: forwarded to :class:`ConvLSTMCell` (see its docstring).
    """

    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        n_actions: int,
        depth: int = 3,
        ticks: int = 3,
        hidden_channels: int = 32,
        cell_kernel_size: int = 3,
        encoder_channels: Sequence[int] = (32, 32),
        encoder_kernels: Sequence[int] = (4, 4),
        encoder_strides: Sequence[int] = (1, 1),
        encoder_relu: bool = False,
        mlp_hidden: int = 256,
        encoder_skip: bool = True,
        pool_and_inject: bool = True,
        pool_projection: str = "per-channel",
        output_activation: str = "tanh",
        forget_bias: float = 0.0,
        fence: str = "valid",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {ticks}")
        if not (len(encoder_channels) == len(encoder_kernels) == len(encoder_strides)):
            raise ValueError("encoder_channels, encoder_kernels and encoder_strides must have equal length")

        self.observation_shape = tuple(observation_shape)
        self.n_actions = n_actions
        self.depth = depth
        self.ticks = ticks
        self.hidden_channels = hidden_channels
        self.encoder_skip = encoder_skip

        obs_c, obs_h, obs_w = self.observation_shape

        # --- Observation encoder ([G19] App. 9.1; [TLP] ``embed``) ---------
        convs = []
        in_ch, height, width = obs_c, obs_h, obs_w
        for out_ch, k, s in zip(encoder_channels, encoder_kernels, encoder_strides):
            convs.append(_SameConv2d(in_ch, out_ch, k, stride=s, bias=True))
            in_ch = out_ch
            height = -(-height // s)  # SAME padding: out = ceil(in / stride)
            width = -(-width // s)
        self.encoder = nn.ModuleList(convs)
        self.encoder_relu = encoder_relu
        self._encoded_channels = in_ch
        self._state_hw = (height, width)

        if encoder_skip and in_ch != hidden_channels:
            raise ValueError(
                f"encoder_skip requires the encoder output channels ({in_ch}) to equal "
                f"hidden_channels ({hidden_channels}) because they are added elementwise"
            )

        # --- ConvLSTM stack ([G19] Sec 2.1.1) -------------------------------
        self.cells = nn.ModuleList(
            ConvLSTMCell(
                input_channels=in_ch,
                hidden_channels=hidden_channels,
                kernel_size=cell_kernel_size,
                pool_and_inject=pool_and_inject,
                pool_projection=pool_projection,
                output_activation=output_activation,
                forget_bias=forget_bias,
                fence=fence,
            )
            for _ in range(depth)
        )

        # --- Readout ([G19] Sec 2.1.1 / App. 9.1) ---------------------------
        flat = hidden_channels * height * width
        self.mlp = nn.Linear(flat, mlp_hidden, bias=True)
        self.actor = nn.Linear(mlp_hidden, n_actions, bias=True)
        self.critic = nn.Linear(mlp_hidden, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.encoder:
            k = conv.kernel_size[0] * conv.kernel_size[1]
            _lecun_truncated_normal_(conv.weight, fan_in=conv.in_channels * k)
            nn.init.zeros_(conv.bias)
        _lecun_truncated_normal_(self.mlp.weight, fan_in=self.mlp.in_features)
        nn.init.zeros_(self.mlp.bias)
        # Heads: orthogonal, gain 1, zero bias ([TLP] ``Actor``/``Critic``).
        nn.init.orthogonal_(self.actor.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # --- Recurrent state management (explicit; no hidden globals) ----------
    def initial_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> DRCState:
        """Zero state ([TLP] ``initialize_carry``); shape (D, B, C, H, W) each."""
        ref = self.mlp.weight
        device = ref.device if device is None else device
        dtype = ref.dtype if dtype is None else dtype
        height, width = self._state_hw
        shape = (self.depth, batch_size, self.hidden_channels, height, width)
        return DRCState(
            h=torch.zeros(shape, device=device, dtype=dtype),
            c=torch.zeros(shape, device=device, dtype=dtype),
        )

    @staticmethod
    def reset_state(state: DRCState, done: Tensor) -> DRCState:
        """Zero the state rows of finished episodes.

        Args:
            state: current recurrent state.
            done: bool tensor (B,); True where an episode just ended (the
                reference zeroes the carry where ``episode_starts`` is True).
        """
        keep = (~done).to(state.h.dtype).view(1, -1, 1, 1, 1)
        return DRCState(h=state.h * keep, c=state.c * keep)

    # --- Forward ------------------------------------------------------------
    def encode(self, obs: Tensor) -> Tensor:
        x = obs
        for idx, conv in enumerate(self.encoder):
            x = conv(x)
            if self.encoder_relu and idx < len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def recurrent_trace(
        self,
        obs: Tensor,
        state: Optional[DRCState] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...], Tensor, DRCState]:
        """Expose encoder/tick spatial maps without changing policy semantics.

        Returns ``(encoded, tick_outputs, readout_map, new_state)`` where each
        tick output is the final-depth hidden map after that recurrent tick,
        and ``readout_map`` is the map flattened by the faithful policy/value
        readout (including the encoder skip when enabled). This method is
        primarily for frozen-network mechanistic probes.
        """
        if obs.dim() != 4 or obs.shape[1:] != self.observation_shape:
            raise ValueError(
                f"expected obs of shape (B, {self.observation_shape}), got {tuple(obs.shape)}"
            )
        if state is None:
            state = self.initial_state(obs.shape[0], device=obs.device, dtype=obs.dtype)

        encoded = self.encode(obs)
        h = list(state.h.unbind(0))
        c = list(state.c.unbind(0))
        tick_outputs = []
        for _ in range(self.ticks):
            # Top-down skip: read h_D of the PREVIOUS tick before updating.
            below = h[-1]
            for depth, cell in enumerate(self.cells):
                h[depth], c[depth] = cell(encoded, below, (h[depth], c[depth]))
                below = h[depth]
            tick_outputs.append(h[-1])

        readout_map = h[-1]
        if self.encoder_skip:
            readout_map = readout_map + encoded
        new_state = DRCState(h=torch.stack(h), c=torch.stack(c))
        return encoded, tuple(tick_outputs), readout_map, new_state

    def forward(
        self,
        obs: Tensor,
        state: Optional[DRCState] = None,
        action_mask: Optional[Tensor] = None,
        *,
        return_map: bool = False,
    ) -> Tuple[Tensor, Tensor, DRCState]:
        """One environment step: N ticks of the D-module stack plus readout.

        Args:
            obs: (B, C, H, W) observation (already preprocessed/normalized).
            state: recurrent state from the previous step; ``None`` means a
                fresh zero state (episode start).
            action_mask: optional (B, n_actions) bool tensor, True = legal.
                Masked logits are set to ``LOGIT_MASK_VALUE``.  This is an
                environment adaptation, not part of [G19].
            return_map: also return the spatial ``readout_map`` the heads read
                from, so an auxiliary head can attach to it without duplicating
                the readout.  Purely additive: the default return is unchanged.

        Returns:
            ``(policy_logits (B, n_actions), value (B,), new_state)``, plus
            ``readout_map (B, C, H, W)`` when ``return_map``.
        """
        _encoded, _ticks, readout_map, new_state = self.recurrent_trace(obs, state)
        hidden = F.relu(self.mlp(readout_map.flatten(1)))
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, LOGIT_MASK_VALUE)

        if return_map:
            return logits, value, new_state, readout_map
        return logits, value, new_state


def sokoban_drc33() -> DRCNet:
    """The verified faithful Sokoban DRC(3,3): 10x10x3 tile obs, 4 actions.

    Matches the trained checkpoint config ``drc33/bkynosqi`` of [TLP]/[T24];
    total parameter count 1,285,125 (reported as "1.29M" in [T24]).
    """
    return DRCNet(observation_shape=(3, 10, 10), n_actions=4)


def sokoban_drc11() -> DRCNet:
    """Sokoban DRC(1,1) ([G19] Table 1 baseline; same modules, D=N=1)."""
    return DRCNet(observation_shape=(3, 10, 10), n_actions=4, depth=1, ticks=1)
