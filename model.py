import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Map a (batch, 1) scalar in [0, 1] to a (batch, dim) sin/cos frequency embedding.

    A raw scalar gives the network a single, nearly-linear feature of time; the JVP target
    differentiates the network w.r.t. t, so u needs enough time features to represent a
    t-dependent derivative. Differentiable in x, which the JVP tangent (dt/dt = 1) requires.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=x.dtype) / half
    )
    args = x * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=1)


def group_norm_groups(channels: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that evenly divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Two GroupNorm+conv layers with a skip connection and per-block FiLM time conditioning.

    No normalization anywhere in the network was letting activation magnitudes drift
    unpredictably across the 5 different resolution levels, which is exactly the kind of
    instability this project kept fighting. GroupNorm before each conv is standard practice
    in diffusion U-Nets specifically to stabilize this.

    FiLM (a per-channel scale and shift computed from the time embedding) is applied after
    the first GroupNorm. A single time bias at the input conv gives deep layers no way to
    change behavior with (r, t); u(z, 0, 1) at sampling and u(z_t, r, t) mid-flow must
    produce very different outputs from similar inputs, so every block needs to see time.
    The FiLM projection is zero-initialized so each block starts exactly at its previous
    time-independent behavior.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = group_norm_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(time_features)[:, :, None, None].chunk(2, dim=1)
        hidden = self.norm1(x) * (1.0 + scale) + shift
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return x + hidden


def attention_heads(channels: int, max_heads: int = 8) -> int:
    """Largest head count <= max_heads that evenly divides channels."""
    for heads in range(min(max_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class SelfAttention2D(nn.Module):
    """Global self-attention over spatial positions, meant for a low-resolution bottleneck.

    Local 3x3 convolutions only ever compare a pixel to its immediate neighbors, so the
    network has no mechanism to compare the noisy input's overall pattern against itself
    globally. At an 8x8 bottleneck (64 tokens) this is cheap, unlike at full resolution.
    """

    def __init__(self, channels: int, max_heads: int = 8):
        super().__init__()
        self.num_heads = attention_heads(channels, max_heads)
        self.norm = nn.GroupNorm(group_norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        head_dim = channels // self.num_heads

        hidden = self.norm(x)
        query, key, value = self.qkv(hidden).chunk(3, dim=1)
        query = query.reshape(batch, self.num_heads, head_dim, height * width)
        key = key.reshape(batch, self.num_heads, head_dim, height * width)
        value = value.reshape(batch, self.num_heads, head_dim, height * width)

        attention = torch.einsum("bhdi,bhdj->bhij", query, key) * (head_dim**-0.5)
        attention = attention.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attention, value)
        out = out.reshape(batch, channels, height, width)
        return x + self.proj(out)


class MiniUNet(nn.Module):
    """A small time-conditioned U-Net that predicts an RGB (mean-)velocity field.

    Two downsampling stages (e.g. 32 -> 16 -> 8) with skip connections, so the network
    can mix information across the whole image cheaply at low resolution instead of only
    ever operating at full resolution with local 3x3 receptive fields. A self-attention layer
    at the 8x8 bottleneck gives it a second, non-local mechanism for the same purpose, cheap
    at that resolution (64 tokens).

    Time conditioning: sinusoidal embeddings of t and of the gap (t - r) -- the two
    quantities the MeanFlow target actually depends on -- concatenated, passed through an
    MLP, and injected into every residual block via FiLM.

    `num_blocks` is blocks *per resolution level* (5 levels total: down1, down2, bottleneck,
    up2, up1) -- keep it small (1-2) since width doubles at each of the two downsampling
    stages, or parameter count grows fast.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 128,
        time_dim: int = 64,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        channels_1 = hidden_channels
        channels_2 = hidden_channels * 2
        channels_3 = hidden_channels * 4

        self.input_conv = nn.Conv2d(image_channels, channels_1, kernel_size=3, padding=1)

        self.down_blocks_1 = nn.ModuleList([ResidualConvBlock(channels_1, time_dim) for _ in range(num_blocks)])
        self.downsample_1 = nn.Conv2d(channels_1, channels_2, kernel_size=3, stride=2, padding=1)

        self.down_blocks_2 = nn.ModuleList([ResidualConvBlock(channels_2, time_dim) for _ in range(num_blocks)])
        self.downsample_2 = nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=2, padding=1)

        self.bottleneck_blocks = nn.ModuleList([ResidualConvBlock(channels_3, time_dim) for _ in range(num_blocks)])
        self.bottleneck_attention = SelfAttention2D(channels_3)

        self.upsample_2 = nn.ConvTranspose2d(channels_3, channels_2, kernel_size=4, stride=2, padding=1)
        self.merge_2 = nn.Conv2d(channels_2 * 2, channels_2, kernel_size=1)
        self.up_blocks_2 = nn.ModuleList([ResidualConvBlock(channels_2, time_dim) for _ in range(num_blocks)])

        self.upsample_1 = nn.ConvTranspose2d(channels_2, channels_1, kernel_size=4, stride=2, padding=1)
        self.merge_1 = nn.Conv2d(channels_1 * 2, channels_1, kernel_size=1)
        self.up_blocks_1 = nn.ModuleList([ResidualConvBlock(channels_1, time_dim) for _ in range(num_blocks)])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(group_norm_groups(channels_1), channels_1),
            nn.SiLU(),
            nn.Conv2d(channels_1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        time_features = self.time_mlp(
            torch.cat(
                [
                    sinusoidal_embedding(t, self.time_dim),
                    sinusoidal_embedding(t - r, self.time_dim),
                ],
                dim=1,
            )
        )

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
        hidden = self.bottleneck_attention(hidden)

        hidden = self.upsample_2(hidden)
        hidden = self.merge_2(torch.cat([hidden, skip_2], dim=1))
        for block in self.up_blocks_2:
            hidden = block(hidden, time_features)

        hidden = self.upsample_1(hidden)
        hidden = self.merge_1(torch.cat([hidden, skip_1], dim=1))
        for block in self.up_blocks_1:
            hidden = block(hidden, time_features)

        return self.output_conv(hidden)
