import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Expands a scalar time value into sin/cos features at exponentially spaced frequencies.

    A bare scalar fed straight into a linear layer is a known weak spot for representing
    fine-grained sensitivity to small input changes (neural nets are biased toward learning
    low-frequency functions of their raw inputs). The MeanFlow JVP term needs exactly that
    kind of sensitivity (d/dt of the network's output), so we expand time into many
    frequencies first, the same way diffusion timestep embeddings and transformer positional
    encodings do.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even (half for sin, half for cos)"
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = x * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResidualConvBlock(nn.Module):
    """Two convolutions with a skip connection, FiLM-modulated by the time embedding at every block."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(time_dim, channels * 2)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(time_features).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        hidden = self.conv1(self.act(x))
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.conv2(self.act(hidden))
        return x + hidden


class Downsample(nn.Module):
    """Stride-2 conv, halving spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Stride-2 transposed conv, doubling spatial resolution back."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyTimeConditionedCNN(nn.Module):
    """A small time-conditioned mini U-Net that predicts an RGB velocity field.

    Two downsampling stages (e.g. 32 -> 16 -> 8) with skip connections, so the network
    can mix information across the whole image cheaply at low resolution instead of only
    ever operating at full resolution with local 3x3 receptive fields. Time is injected via
    FiLM (per-block scale/shift, from a sinusoidal time embedding) at every block, at every
    resolution level.

    `num_blocks` is blocks *per resolution level* (5 levels total: down1, down2, bottleneck,
    up2, up1) -- keep it small (1-2) to stay in the same size ballpark as a deeper flat CNN,
    since width doubles at each of the two downsampling stages.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 128,
        time_dim: int = 64,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        channels_1 = hidden_channels
        channels_2 = hidden_channels * 2
        channels_3 = hidden_channels * 4

        self.input_conv = nn.Conv2d(image_channels, channels_1, kernel_size=3, padding=1)

        self.down_blocks_1 = nn.ModuleList([FiLMResidualConvBlock(channels_1, time_dim) for _ in range(num_blocks)])
        self.downsample_1 = Downsample(channels_1, channels_2)

        self.down_blocks_2 = nn.ModuleList([FiLMResidualConvBlock(channels_2, time_dim) for _ in range(num_blocks)])
        self.downsample_2 = Downsample(channels_2, channels_3)

        self.bottleneck_blocks = nn.ModuleList(
            [FiLMResidualConvBlock(channels_3, time_dim) for _ in range(num_blocks)]
        )

        self.upsample_2 = Upsample(channels_3, channels_2)
        self.merge_2 = nn.Conv2d(channels_2 * 2, channels_2, kernel_size=1)
        self.up_blocks_2 = nn.ModuleList([FiLMResidualConvBlock(channels_2, time_dim) for _ in range(num_blocks)])

        self.upsample_1 = Upsample(channels_2, channels_1)
        self.merge_1 = nn.Conv2d(channels_1 * 2, channels_1, kernel_size=1)
        self.up_blocks_1 = nn.ModuleList([FiLMResidualConvBlock(channels_1, time_dim) for _ in range(num_blocks)])

        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels_1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        time_features = self.time_mlp(torch.cat([self.time_embed(r), self.time_embed(t)], dim=1))

        hidden = self.input_conv(z_t)
        for block in self.down_blocks_1:
            hidden = block(hidden, time_features)
        skip_1 = hidden
        hidden = self.downsample_1(hidden)

        for block in self.down_blocks_2:
            hidden = block(hidden, time_features)
        skip_2 = hidden
        hidden = self.downsample_2(hidden)

        for block in self.bottleneck_blocks:
            hidden = block(hidden, time_features)

        hidden = self.upsample_2(hidden)
        hidden = self.merge_2(torch.cat([hidden, skip_2], dim=1))
        for block in self.up_blocks_2:
            hidden = block(hidden, time_features)

        hidden = self.upsample_1(hidden)
        hidden = self.merge_1(torch.cat([hidden, skip_1], dim=1))
        for block in self.up_blocks_1:
            hidden = block(hidden, time_features)

        return self.output_conv(hidden)
