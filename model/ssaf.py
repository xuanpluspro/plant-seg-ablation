"""SACF adapted with HyperSeg-DG DRM/CBM and the existing fusion unit.

DRM refines the deep feature before upsampling; CBM refines the shallow
feature at native resolution. See SACF_SOURCE.md for source and differences.
"""
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DynamicRelationModule", "ContextBridgeModule", "SpectralSpatialAdaptiveFusion"]


class DynamicRelationModule(nn.Module):
    """Two dilated convolutions weighted by per-image scale attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d)
            for d in (1, 2)
        ])
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, channels // reduction), 2, 1),
            nn.Softmax(dim=1))
        self.fusion = nn.Conv2d(2 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = [conv(x) for conv in self.dilated_convs]
        weights = self.attention_net(x)
        branches = [feature * weights[:, i:i+1]
                    for i, feature in enumerate(features)]
        return x + self.gamma * self.fusion(torch.cat(branches, dim=1))


class ContextBridgeModule(nn.Module):
    """Local depthwise/pointwise features plus globally gated input."""
    def __init__(self, channels):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GELU())
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        local = self.local_path(x)
        global_feature = x * self.global_path(x)
        return self.out_proj(local + global_feature) + x


class SpectralSpatialAdaptiveFusion(nn.Module):
    """DRM(deep) -> upsample; CBM(shallow) -> original adaptive fusion."""
    def __init__(self, in_channels):
        super().__init__()
        self.drm = DynamicRelationModule(in_channels)
        self.cbm = ContextBridgeModule(in_channels)
        self.adp_conv = nn.Conv2d(2, 2, 1, bias=False)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))

    def forward(self, fh, fs):
        if fh.ndim != 4 or fs.ndim != 4 or fh.shape[:2] != fs.shape[:2]:
            raise ValueError("Expected deep and shallow [B,C,H,W] with matching batch/channels")
        fh = self.drm(fh)
        fh = F.interpolate(fh, size=fs.shape[2:], mode='bilinear', align_corners=True)
        fs = self.cbm(fs)
        weights = torch.sigmoid(self.adp_conv(torch.cat([
            fh.mean(1, keepdim=True), fs.mean(1, keepdim=True)], dim=1)))
        return self.fuse_conv(weights[:, :1] * fh + weights[:, 1:] * fs)
