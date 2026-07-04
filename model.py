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
    ever operating at full resolution with local 3x3 receptive fields.

    Time conditioning is the plain scheme (raw scalar (r, t) -> MLP -> single additive bias,
    injected once right after the input conv) -- FiLM and sinusoidal time embeddings are
    deliberately not used here, so this isolates the U-Net structure as the only new
    variable versus the earlier flat-CNN experiments.

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
        self.time_mlp = nn.Sequential(
            nn.Linear(2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        channels_1 = hidden_channels
        channels_2 = hidden_channels * 2
        channels_3 = hidden_channels * 4

        self.input_conv = nn.Conv2d(image_channels, channels_1, kernel_size=3, padding=1)
        self.time_to_channels = nn.Linear(time_dim, channels_1)

        self.down_blocks_1 = nn.ModuleList([ResidualConvBlock(channels_1) for _ in range(num_blocks)])
        self.downsample_1 = Downsample(channels_1, channels_2)

        self.down_blocks_2 = nn.ModuleList([ResidualConvBlock(channels_2) for _ in range(num_blocks)])
        self.downsample_2 = Downsample(channels_2, channels_3)

        self.bottleneck_blocks = nn.ModuleList([ResidualConvBlock(channels_3) for _ in range(num_blocks)])

        self.upsample_2 = Upsample(channels_3, channels_2)
        self.merge_2 = nn.Conv2d(channels_2 * 2, channels_2, kernel_size=1)
        self.up_blocks_2 = nn.ModuleList([ResidualConvBlock(channels_2) for _ in range(num_blocks)])

        self.upsample_1 = Upsample(channels_2, channels_1)
        self.merge_1 = nn.Conv2d(channels_1 * 2, channels_1, kernel_size=1)
        self.up_blocks_1 = nn.ModuleList([ResidualConvBlock(channels_1) for _ in range(num_blocks)])

        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels_1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        time_features = self.time_mlp(torch.cat([r, t], dim=1))
        time_bias = self.time_to_channels(time_features)[:, :, None, None]

        hidden = self.input_conv(z_t) + time_bias
        for block in self.down_blocks_1:
            hidden = block(hidden)
        skip_1 = hidden
        hidden = self.downsample_1(hidden)

        for block in self.down_blocks_2:
            hidden = block(hidden)
        skip_2 = hidden
        hidden = self.downsample_2(hidden)

        for block in self.bottleneck_blocks:
            hidden = block(hidden)

        hidden = self.upsample_2(hidden)
        hidden = self.merge_2(torch.cat([hidden, skip_2], dim=1))
        for block in self.up_blocks_2:
            hidden = block(hidden)

        hidden = self.upsample_1(hidden)
        hidden = self.merge_1(torch.cat([hidden, skip_1], dim=1))
        for block in self.up_blocks_1:
            hidden = block(hidden)

        return self.output_conv(hidden)
