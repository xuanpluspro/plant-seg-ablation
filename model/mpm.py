"""MPM-5531 with CAM-conditioned sparse attention and a fusion-feature residual.

Source: https://github.com/Wangtao-Bao/MPCNet/blob/5024ea51e8abdf7bc40e36489041143c468b1fdf/MPCNet.py
Authors: Wangtao-Bao/MPCNet contributors. See model/MPM_SOURCE.md.
The MPM kernel selection is adapted; helper classes retain upstream behavior.
Sparse attention follows MSCN; see model/MPM_SOURCE.md for sources and changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SparseAttention(nn.Module):
    """Q from fused features; K/V from CAM features, with no second pooling.

    The two learned scalars are unconstrained, as in MSCN. Top-k is dense
    masked attention, not a sparse kernel or a guarantee of lower FLOPs.
    """
    def __init__(self, channels, num_heads=8, k1=2, k2=6):
        super().__init__()
        if num_heads < 1 or channels < 1 or channels % num_heads:
            raise ValueError("Attention channels must be divisible by num_heads")
        if k1 < 1 or k2 < 1:
            raise ValueError("Top-k divisors must be >= 1")
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.k1, self.k2 = k1, k2
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.attn1 = nn.Parameter(torch.tensor([0.5]))
        self.attn2 = nn.Parameter(torch.tensor([0.5]))
        self.proj = nn.Conv2d(channels, channels, 1)

    @staticmethod
    def _aggregate(scores, values, divisor):
        # Retain at least one key even for 1x1 maps; do not delete positions.
        keep = max(1, int(scores.shape[-1] / divisor))
        indices = scores.topk(keep, dim=-1, sorted=False).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return weights @ values

    def forward(self, fused, cam):
        if fused.ndim != 4 or cam.shape != fused.shape:
            raise ValueError("Expected matching fused/CAM features [B,C,H,W]")
        b, c, h, w = fused.shape
        # LayerNorm across channels independently at each spatial position.
        normalized = self.norm(cam.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        q = self.q(fused).reshape(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)
        kv = self.kv(normalized).reshape(b, 2, self.num_heads, c // self.num_heads, h * w)
        k, v = kv.unbind(dim=1)
        k, v = k.transpose(-2, -1), v.transpose(-2, -1)
        # Compute correlations and softmax in FP32, including under autocast.
        with torch.autocast(device_type=fused.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            out1 = self._aggregate(scores, v.float(), self.k1)
            out2 = self._aggregate(scores, v.float(), self.k2)
            out = self.attn1 * out1 + self.attn2 * out2
        out = out.transpose(-2, -1).reshape(b, c, h, w).to(q.dtype)
        return self.proj(out)


class MPM(nn.Module):
    def __init__(self, in_channel):
        super(MPM, self).__init__()
        if in_channel < 8 or in_channel % 8:
            raise ValueError("MPM channels must be a positive multiple of eight")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 1, 1, padding=0),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 5, 1, padding=2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )


        self.Conv = CBR(in_channel * 2, in_channel, 1)
        self.ca = CoordAttiton(in_channel,in_channel)
        self.attention = SparseAttention(in_channel)

    def forward(self, x):
        b, c, h, w = x.size()
        x_1 = x[:, :(c // 4), :, :]
        x_2 = x[:, (c // 4):(c // 4) * 2, :, :]
        x_3 = x[:, (c // 4) * 2:(c // 4) * 3, :, :]
        x_4 = x[:, (c // 4) * 3:, :, :]

        # The two 5x5 branches share convolution weights and BatchNorm.
        branches = (self.conv5(x_1), self.conv5(x_2),
                    self.conv3(x_3), self.conv1(x_4))
        fused = self.Conv(torch.cat((*branches, x), 1))
        cam = self.ca(fused)
        return fused + self.attention(fused, cam)
