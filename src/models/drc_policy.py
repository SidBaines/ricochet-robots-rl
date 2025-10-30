import numpy as np
import torch as th
import torch.nn as nn
# from stable_baselines3.common.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import (
    CategoricalDistribution, MultiCategoricalDistribution, DiagGaussianDistribution,
    BernoulliDistribution, StateDependentNoiseDistribution, make_proba_distribution
)
try:
    # sb3-contrib recurrent base policy
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy # type: ignore
except Exception as _e:  # pragma: no cover - optional dependency
    _SB3RecurrentPolicy = object  # type: ignore
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class DRCRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 n_layers=3, repeats=3, hidden_channels=32, ortho_init=True, forget_bias=0.0, **kwargs):
        # super().__init__(self)  # initialize nn.Module
        # nn.Module.__init__(self)  # initialize nn.Module
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        # Initialize base policy attributes
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.n_layers = n_layers
        self.repeats = repeats
        self.hidden_channels = hidden_channels
        self.use_sde = kwargs.get('use_sde', False)
        dist_kwargs = kwargs.get('dist_kwargs', None)
        # Create action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=self.use_sde, dist_kwargs=dist_kwargs)
        # Determine input shape (channels, height, width)
        if len(observation_space.shape) == 3:
            # Assume observation_space.shape = (C, H, W) or (H, W, C)
            if observation_space.shape[0] <= 4:  # likely channel-first
                in_channels, obs_h, obs_w = observation_space.shape
            else:  # channel-last
                obs_h, obs_w, in_channels = observation_space.shape
        else:
            # Non-image input (flattened) – in this context, not expected
            in_channels, obs_h, obs_w = observation_space.shape[0], 1, 1
        self.height, self.width = obs_h, obs_w
        # **Encoder:** two conv layers (4×4 filters) encoding the observation:contentReference[oaicite:25]{index=25}
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=1, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=1, padding=1)
        )
        # **ConvLSTM core:** define convolutional gates for each layer
        # Each layer input = [encoded_obs, injection, fence] so input channels = hidden_channels + hidden_channels + 1
        self.conv_x = nn.ModuleList([
            nn.Conv2d(hidden_channels*2 + 1, 4 * hidden_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(n_layers)
        ])
        self.conv_h = nn.ModuleList([
            nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(n_layers)
        ])
        # Pool-and-inject parameters (per-channel linear combination of mean & max) for each layer:contentReference[oaicite:26]{index=26}
        self.injection_w_mean = nn.ParameterList([nn.Parameter(th.zeros(hidden_channels)) for _ in range(n_layers)])
        self.injection_w_max  = nn.ParameterList([nn.Parameter(th.zeros(hidden_channels)) for _ in range(n_layers)])
        self.injection_b      = nn.ParameterList([nn.Parameter(th.zeros(hidden_channels)) for _ in range(n_layers)])
        self.forget_bias = forget_bias  # bias added to forget gate
        # **Fence channel:** binary mask with ones on the boundary and zeros inside:contentReference[oaicite:27]{index=27}
        fence = th.ones(1, 1, obs_h + 2, obs_w + 2)
        fence[:, :, 1:-1, 1:-1] = 0  # zeros in interior, ones at border
        fence = fence[:, :, 1:-1, 1:-1]  # crop to original obs size (border of ones)
        self.register_buffer("fence", fence)  # not a parameter, just a constant tensor
        # **Fully-connected head:** flatten top-layer hidden state to 256 units:contentReference[oaicite:28]{index=28}
        self.fc_embed = nn.Linear(hidden_channels * obs_h * obs_w, 256)
        # Policy and value output layers
        if isinstance(self.action_dist, DiagGaussianDistribution) or isinstance(self.action_dist, StateDependentNoiseDistribution):
            # Continuous actions (Gaussian) - use mean and log_std
            self.action_net = nn.Linear(256, self.action_space.n)
            self.log_std = nn.Parameter(th.zeros(self.action_space.n) + kwargs.get('log_std_init', 0.0))
        else:
            # Discrete or multi-binary actions
            self.action_net = nn.Linear(256, self.action_space.n)
            self.log_std = None
        self.value_net = nn.Linear(256, 1)
        # Orthogonal initialization (for stability)
        if ortho_init:
            def init_weights(m):
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, gain=th.sqrt(th.tensor(2.0)))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            self.apply(init_weights)
            # Small initial weight for policy output (improves training stability)
            nn.init.constant_(self.action_net.weight, 0.01)
        # Optimizer (Adam by default)
        optimizer_cls = kwargs.get('optimizer_class', th.optim.Adam)
        optimizer_kwargs = kwargs.get('optimizer_kwargs', {})
        self.optimizer = optimizer_cls(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)

    def _get_initial_state(self, batch_size: int) -> tuple:
        """Return a zero-initialized hidden state (hidden and cell) for a given batch size."""
        h0 = np.zeros((self.n_layers, batch_size, self.hidden_channels, self.height, self.width), dtype=np.float32)
        c0 = np.zeros((self.n_layers, batch_size, self.hidden_channels, self.height, self.width), dtype=np.float32)
        return RNNStates(h0, c0)

    def forward(self, obs, state=None, episode_start=None, deterministic=False):
        """Forward one timestep (with internal repeats) through the policy. Returns action, value, log_prob, and new_state."""
        batch_size = obs.shape[0] if obs.ndim == 4 else 1
        # Initialize or unpack the RNN state
        if state is None:
            state = self._get_initial_state(batch_size)
        hidden_np, cell_np = state  # each is an np.array of shape (n_layers, batch, ch, H, W)
        hidden = th.tensor(hidden_np, device=self.fence.device)
        cell   = th.tensor(cell_np, device=self.fence.device)
        if episode_start is None:
            episode_start = th.zeros(batch_size, dtype=th.bool, device=self.fence.device)
        else:
            episode_start = th.tensor(episode_start, dtype=th.bool, device=self.fence.device)
        # Prepare observation (ensure channel-first tensor)
        obs_tensor = obs if isinstance(obs, th.Tensor) else th.tensor(obs)
        if obs_tensor.dim() == 3:  # if a single observation without batch dim
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[1] not in [1, 3] and obs_tensor.shape[-1] in [1, 3]:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # to (B,C,H,W)
        # Encode observation through conv encoder
        embed = self.conv_encoder(obs_tensor)  # shape: (batch, hidden_channels, H, W)
        # Repeat internal computation N times (extra thinking steps):contentReference[oaicite:29]{index=29}
        for _ in range(self.repeats):
            # Reset hidden state for new episodes
            if episode_start.any():
                hidden[:, episode_start, :, :, :] = 0.0
                cell[:, episode_start, :, :, :] = 0.0
            # Process each ConvLSTM layer
            for l in range(self.n_layers):
                h_prev = hidden[l]  # previous hidden state for layer l
                # Pool-and-inject: compute per-channel stats from h_prev:contentReference[oaicite:30]{index=30}
                h_mean = th.mean(h_prev, dim=(2, 3))  # (batch, channels)
                h_max  = th.amax(h_prev, dim=(2, 3))
                # Linear combination per channel: inj_val[c] = w_max[c]*h_max[c] + w_mean[c]*h_mean[c] + b[c]
                inj = self.injection_w_max[l] * h_max + self.injection_w_mean[l] * h_mean + self.injection_b[l]
                inj_map = inj.view(batch_size, -1, 1, 1).expand(-1, -1, self.height, self.width)  # expand to map
                fence_map = self.fence
                if fence_map.shape[2:] != (self.height, self.width):
                    # If input size changed (unlikely here), adjust fence size
                    fence_map = th.ones(1, 1, self.height, self.width, device=self.fence.device)
                    fence_map[:, :, 1:-1, 1:-1] = 0
                fence_map_exp = fence_map.expand(batch_size, -1, -1, -1)
                # Concatenate inputs for layer l
                x = th.cat([embed, inj_map, fence_map_exp], dim=1)
                # ConvLSTM gating (input+hidden conv):contentReference[oaicite:31]{index=31}
                gates = self.conv_x[l](x) + self.conv_h[l](h_prev)
                i, f, o, g = th.chunk(gates, 4, dim=1)  # split into 4 gate tensors
                i = th.sigmoid(i)                           # input gate
                f = th.sigmoid(f + self.forget_bias)        # forget gate (with bias) 
                o = th.tanh(o)                              # output gate (tanh):contentReference[oaicite:32]{index=32}
                g = th.tanh(g)                              # cell candidate
                c_prev = cell[l]
                c_new = f * c_prev + i * g                  # new cell state
                h_new = o * th.tanh(c_new)                  # new hidden state
                cell[l] = c_new
                hidden[l] = h_new
            # end for each layer
        # end for repeats
        # Prepare outputs from final hidden state of top layer
        top_hidden = hidden[self.n_layers - 1]  # shape: (batch, hidden_channels, H, W)
        flat_hidden = top_hidden.view(batch_size, -1)      # flatten spatially
        latent = th.relu(self.fc_embed(flat_hidden))       # 256-dim latent vector:contentReference[oaicite:33]{index=33}
        values = self.value_net(latent)                    # Critic output
        # Actor output distribution
        if isinstance(self.action_dist, DiagGaussianDistribution):
            mean = self.action_net(latent)
            dist = self.action_dist.proba_distribution(mean_actions=mean, log_std=self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            mean = self.action_net(latent)
            dist = self.action_dist.proba_distribution(mean_actions=mean, log_std=self.log_std, latent_sde=latent)
        elif isinstance(self.action_dist, CategoricalDistribution):
            logits = self.action_net(latent)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            logits = self.action_net(latent)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        elif isinstance(self.action_dist, BernoulliDistribution):
            logits = self.action_net(latent)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:
            raise NotImplementedError("Unsupported distribution")
        # Sample or choose action
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        # Package new hidden state for output (as numpy arrays)
        new_state = RNNStates(hidden.cpu().numpy(), cell.cpu().numpy())
        return actions, values, log_prob, new_state

    def evaluate_actions(self, obs, actions, state, episode_start):
        """Evaluate given action selections for a sequence of observations (for PPO loss calculation)."""
        # Convert inputs to torch tensors
        obs = th.tensor(obs, device=self.fence.device) if not isinstance(obs, th.Tensor) else obs.to(self.fence.device)
        actions = th.tensor(actions, device=self.fence.device) if not isinstance(actions, th.Tensor) else actions.to(self.fence.device)
        hidden_np, cell_np = state
        hidden = th.tensor(hidden_np, device=self.fence.device)
        cell   = th.tensor(cell_np, device=self.fence.device)
        # Handle sequence inputs: obs shape (T, N, C, H, W) or (N, C, H, W)
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)  # add time dimension T=1
            actions = actions.unsqueeze(0)
            episode_start = np.expand_dims(episode_start, axis=0)
        T, N = obs.shape[0], obs.shape[1]
        values = []
        log_probs = []
        entropies = []
        for t in range(T):
            obs_batch = obs[t]
            if obs_batch.shape[1] not in [1, 3] and obs_batch.shape[-1] in [1, 3]:
                obs_batch = obs_batch.permute(0, 3, 1, 2)  # ensure channel-first
            done_mask = th.tensor(episode_start[t], dtype=th.bool, device=self.fence.device)
            # Reset hidden state for new episodes at this timestep
            if done_mask.any():
                hidden[:, done_mask, :, :, :] = 0.0
                cell[:, done_mask, :, :, :] = 0.0
            # Encode obs and perform internal repeats as in forward()
            embed = self.conv_encoder(obs_batch)
            for _ in range(self.repeats):
                if done_mask.any():
                    hidden[:, done_mask, :, :, :] = 0.0
                    cell[:, done_mask, :, :, :] = 0.0
                for l in range(self.n_layers):
                    h_prev = hidden[l]
                    h_mean = th.mean(h_prev, dim=(2, 3))
                    h_max  = th.amax(h_prev, dim=(2, 3))
                    inj = (self.injection_w_max[l] * h_max + self.injection_w_mean[l] * h_mean + self.injection_b[l])
                    inj_map = inj.view(N, -1, 1, 1).expand(-1, -1, self.height, self.width)
                    fence_map = self.fence.expand(N, -1, -1, -1)
                    x = th.cat([embed, inj_map, fence_map], dim=1)
                    gates = self.conv_x[l](x) + self.conv_h[l](h_prev)
                    i, f, o, g = th.chunk(gates, 4, dim=1)
                    i = th.sigmoid(i);  f = th.sigmoid(f + self.forget_bias)
                    o = th.tanh(o);     g = th.tanh(g)
                    c_new = f * cell[l] + i * g
                    h_new = o * th.tanh(c_new)
                    cell[l] = c_new;  hidden[l] = h_new
            # After repeats, get outputs for timestep t
            top_hidden = hidden[self.n_layers - 1].view(N, -1)
            latent = th.relu(self.fc_embed(top_hidden))
            value = self.value_net(latent)
            # Compute log-prob and entropy of the provided actions at this timestep
            if isinstance(self.action_dist, DiagGaussianDistribution):
                dist = self.action_dist.proba_distribution(mean_actions=self.action_net(latent), log_std=self.log_std)
            elif isinstance(self.action_dist, StateDependentNoiseDistribution):
                dist = self.action_dist.proba_distribution(mean_actions=self.action_net(latent), log_std=self.log_std, latent_sde=latent)
            elif isinstance(self.action_dist, CategoricalDistribution):
                dist = self.action_dist.proba_distribution(action_logits=self.action_net(latent))
            elif isinstance(self.action_dist, MultiCategoricalDistribution):
                dist = self.action_dist.proba_distribution(action_logits=self.action_net(latent))
            elif isinstance(self.action_dist, BernoulliDistribution):
                dist = self.action_dist.proba_distribution(action_logits=self.action_net(latent))
            log_prob = dist.log_prob(actions[t])
            entropy = dist.entropy()
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)
        # Concatenate results over time
        values = th.cat(values, dim=0)
        log_probs = th.cat(log_probs, dim=0)
        entropies = th.cat(entropies, dim=0)
        return values, log_probs, entropies
