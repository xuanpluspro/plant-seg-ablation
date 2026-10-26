"""Fixed baseline replacements used when a complete DRCF/DCAM is disabled."""
from torch import nn
from torch.nn import functional as F


def _check_pair(low, high):
    if low.ndim != 4 or high.ndim != 4 or low.shape[:2] != high.shape[:2]:
        raise ValueError("Expected paired [B,C,2h,2w] and [B,C,h,w] features")
    if low.shape[-2:] != tuple(2 * size for size in high.shape[-2:]):
        raise ValueError("Shallow spatial dimensions must be exactly twice deep dimensions")


class BasicTopDownFusion(nn.Module):
    """DRCF-off: upsample deep, add shallow, then 1x1 Conv/BN/ReLU."""
    def __init__(self, channels):
        super().__init__()
        self.fuse_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                        nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, deep, shallow):
        _check_pair(shallow, deep)
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse_conv(deep + shallow)


class BasicBottomUpFusion(nn.Module):
    """DCAM-off: average-pool shallow, add deep, then 1x1 Conv/GN/ReLU."""
    def __init__(self, channels):
        super().__init__()
        self.fuse_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                        nn.GroupNorm(8, channels), nn.ReLU(inplace=True))

    def forward(self, shallow, deep):
        _check_pair(shallow, deep)
        return self.fuse_conv(F.avg_pool2d(shallow, 2, stride=2) + deep)
