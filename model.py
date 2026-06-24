import torch
from torch import nn


class TinyTimeConditionedCNN(nn.Module):
    """A very small CNN that predicts a velocity-shaped RGB image."""

    def __init__(self, image_channels: int = 3, hidden_channels: int = 64, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.input_conv = nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1)
        self.time_to_channels = nn.Linear(time_dim, hidden_channels)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
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
        return self.net(image_features + time_bias)
