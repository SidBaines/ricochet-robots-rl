import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# Custom CNN feature extractor for the ResNet architecture
class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_blocks=9):
        # Determine input dimensions (channel-first)
        in_channels, height, width = observation_space.shape if observation_space.shape[0] <= 4 \
                                     else (observation_space.shape[2], observation_space.shape[0], observation_space.shape[1])
        super().__init__(observation_space, features_dim=1)  # placeholder, will set actual features_dim below
        # Define the channel sizes per block (first 2 blocks 32 channels, rest 64 channels):contentReference[oaicite:6]{index=6}
        if num_blocks <= 2:
            channels_per_block = [32] * num_blocks
        else:
            channels_per_block = [32, 32] + [64] * (num_blocks - 2)
        self.num_blocks = num_blocks
        # Convolution layers for trunk and branches in each block
        self.conv_trunk = nn.ModuleList()
        self.conv_branch1 = nn.ModuleList()
        self.conv_branch2 = nn.ModuleList()
        prev_channels = in_channels
        for out_ch in channels_per_block:
            # Trunk convolution (4×4 conv)
            self.conv_trunk.append(nn.Conv2d(prev_channels, out_ch, kernel_size=4, stride=1, padding=1))
            # Two residual branch convolutions (each preceded by ReLU in forward pass)
            self.conv_branch1.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=1, padding=1))
            self.conv_branch2.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=1, padding=1))
            prev_channels = out_ch
        # Determine output feature dimension by running a dummy input through the conv layers
        with th.no_grad():
            dummy = th.zeros(1, in_channels, height, width)
            out = dummy
            for i in range(self.num_blocks):
                t = self.conv_trunk[i](out)
                b1 = self.conv_branch1[i](th.relu(t))
                b2 = self.conv_branch2[i](th.relu(t))
                out = th.relu(t + b1 + b2)  # residual addition
            out_flat = out.view(1, -1)
            self._features_dim = out_flat.shape[1]  # flattened spatial output size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Ensure input is channel-first (C×H×W)
        x = observations
        if x.dim() == 4 and x.shape[1] not in [1, 3] and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)  # convert H×W×C to C×H×W
        out = x
        # Forward through each residual block
        for i in range(self.num_blocks):
            t = self.conv_trunk[i](out)
            b1 = self.conv_branch1[i](th.relu(t))
            b2 = self.conv_branch2[i](th.relu(t))
            out = th.relu(t + b1 + b2)  # trunk + two sub-block outputs:contentReference[oaicite:7]{index=7}
        return out.view(out.size(0), -1)  # flatten feature map

# SB3 policy class using the ResNet feature extractor
class ResNetPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(ResNetPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=ResNetFeaturesExtractor,
            features_extractor_kwargs={'num_blocks': 9},
            net_arch=[256],  # one hidden layer of size 256 (shared by policy & value):contentReference[oaicite:8]{index=8}
            activation_fn=nn.ReLU,
            **kwargs
        )
