import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """Two convolution layers with a skip connection for easier optimization."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyTimeConditionedCNN(nn.Module):
    """A small time-conditioned residual CNN that predicts an RGB velocity field."""

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
        self.time_to_channels = nn.Linear(time_dim, hidden_channels)
        self.blocks = nn.Sequential(*(ResidualConvBlock(hidden_channels) for _ in range(num_blocks)))
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
        time_bias = self.time_to_channels(time_features)[:, :, None, None]

        image_features = self.input_conv(z_t)
        hidden = self.blocks(image_features + time_bias)
        return self.output_conv(hidden)
