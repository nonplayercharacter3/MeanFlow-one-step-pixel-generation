import torch
from torch import nn


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


class TinyTimeConditionedCNN(nn.Module):
    """A small time-conditioned residual CNN that predicts an RGB velocity field.

    Time is injected via FiLM (per-block scale/shift) at every residual block instead
    of a single additive bias at the input, so the network can use time information
    throughout its depth rather than only before the first block.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 128,
        time_dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.input_conv = nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResidualConvBlock(hidden_channels, time_dim) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        time_features = self.time_mlp(torch.cat([r, t], dim=1))

        hidden = self.input_conv(z_t)
        for block in self.blocks:
            hidden = block(hidden, time_features)
        return self.output_conv(hidden)
