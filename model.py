"""
GeoSR model: a lightweight EDSR-style super-resolution network.

Real EDSR uses ~32 residual blocks and 256 channels, which needs a GPU
cluster and days of training. For a hackathon live-demo we use the same
architecture *pattern* (residual blocks, global skip connection,
pixel-shuffle upsampling) but scaled down so it trains on a handful of
satellite images on a laptop CPU/GPU in minutes. Swap in more blocks /
channels once you have a real GPU + the full Sentinel-2 dataset.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.res_scale = 0.1  # EDSR trick: scale residuals for stable training

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class GeoSR(nn.Module):
    """
    EDSR-lite: head conv -> N residual blocks -> tail conv -> global skip
    -> pixel-shuffle upsampler -> output conv.

    scale: integer upscaling factor (e.g. 3 for 10m -> ~3.3m, use 4 for
    10m -> 2.5m; the SIH target is <4m so scale=3 or 4 both qualify).
    """

    def __init__(self, in_channels=3, channels=32, n_res_blocks=6, scale=4):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_res_blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body_tail(self.body(feat))
        feat = feat + res  # global skip connection
        out = self.upsampler(feat)
        out = self.tail(out)
        return torch.clamp(out, 0.0, 1.0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = GeoSR()
    print(f"GeoSR params: {count_params(m):,}")
    x = torch.randn(1, 3, 64, 64)
    y = m(x)
    print("input", x.shape, "-> output", y.shape)
