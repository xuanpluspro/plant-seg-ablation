"""FAPM-inspired two-scale spatial affine modulation; candidate CAFR replacement.

Align shallow (B,C,2h,2w) to deep (B,C,h,w), condition on joint features
and per-location channel-normalized discrepancy, refine with separable conv
and SE, then retain the projected deep feature as a direct residual.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(channels, kernel=3, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(channels, channels, kernel, stride=stride,
                                  padding=kernel // 2, groups=groups, bias=False),
                         nn.GroupNorm(8, channels), nn.ReLU(inplace=True))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 16)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, feature):
        return feature * self.gate(feature)


class CrossScaleDynamicModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError("Fusion channels must be >=8 and divisible by 8")
        self.channels = channels
        self.align_low = block(channels, stride=2)
        self.project_high = nn.Conv2d(channels, channels, 1, bias=False)
        self.detail = block(channels)
        self.modulator = nn.Conv2d(channels * 3, channels * 2, 1)
        # Start with identity affine modulation (scale=1, shift=0).
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)
        self.refine = nn.Sequential(block(channels),
                                    block(channels, groups=channels),
                                    block(channels, kernel=1),
                                    SqueezeExcitation(channels))

    def forward(self, feat_low, feat_high):
        if (feat_low.ndim != 4 or feat_high.ndim != 4
                or feat_low.shape[:2] != feat_high.shape[:2]
                or feat_low.shape[1] != self.channels):
            raise ValueError("Expected paired [B,C,2h,2w] and [B,C,h,w] features")
        if feat_low.shape[-2:] != tuple(2 * s for s in feat_high.shape[-2:]):
            raise ValueError("Shallow spatial dimensions must be exactly twice deep dimensions")
        low = self.align_low(feat_low)
        high = self.project_high(feat_high)
        discrepancy = (F.normalize(low, dim=1, eps=1e-6)
                       - F.normalize(high, dim=1, eps=1e-6)).abs()
        a, b = self.modulator(torch.cat([low, high, discrepancy], dim=1)).chunk(2, dim=1)
        modulated = self.detail(low) * (1 + a.tanh()) + b.tanh()
        return high + self.refine(modulated)
