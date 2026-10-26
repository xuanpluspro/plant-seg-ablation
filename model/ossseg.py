"""RGB plant segmentation adapted from MODA/OSSDet, not its detection model.

forward(x) -> logits [B,K,H,W]; forward(x, return_aux=True) -> dict.
Class 0 = background; classes 1..K-1 = foreground. Core modules are attributed in README.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from .ssaf import SpectralSpatialAdaptiveFusion
from .fem import FeatureEnhancementModule
from .mpm import MPM
from .dynamic_fusion import CrossScaleDynamicModulation
from .basic_fusion import BasicTopDownFusion, BasicBottomUpFusion


def conv_block(cin, cout, kernel=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class OSSSeg(nn.Module):
    def __init__(self, num_classes=2, width=256, freeze_backbone_bn=False,
                 use_dynamic_fusion=True, use_msca=True, use_drcf=True, use_dcam=None):
        super().__init__()
        if not 2 <= num_classes <= 256:
            raise ValueError("num_classes must include background and be in 2..256")
        if width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        if type(use_msca) is not bool or type(use_drcf) is not bool or (use_dcam is not None and type(use_dcam) is not bool):
            raise ValueError("Module switches must be booleans")
        # None preserves the original constructor's legacy FEM checkpoint path.
        # New ablation training always supplies an explicit boolean use_dcam.
        legacy_fem = use_dcam is None and not use_dynamic_fusion
        effective_dcam = use_dynamic_fusion if use_dcam is None else use_dcam
        self.model_config = dict(num_classes=num_classes, width=width,
                                 freeze_backbone_bn=freeze_backbone_bn,
                                 use_dynamic_fusion=use_dynamic_fusion,
                                 use_msca=use_msca, use_drcf=use_drcf,
                                 use_dcam=None if legacy_fem else effective_dcam)
        self.freeze_backbone_bn = freeze_backbone_bn
        # Never download weights in the constructor.
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.laterals = nn.ModuleList([
            conv_block(c, width, 1) for c in [256, 512, 1024, 2048]])
        self.mpm = MPM(width) if use_msca else nn.Identity()
        self.ssaf = nn.ModuleList([
            (SpectralSpatialAdaptiveFusion(width) if use_drcf else BasicTopDownFusion(width))
            for _ in range(3)])
        fusion = FeatureEnhancementModule if legacy_fem else (
            CrossScaleDynamicModulation if effective_dcam else BasicBottomUpFusion)
        self.fem = nn.ModuleList([fusion(width) for _ in range(3)])
        self.foreground_head = nn.Sequential(conv_block(width, 64), nn.Conv2d(64, 1, 1))
        self.foreground_smooth = conv_block(width, width, 1)
        # Segmentation-specific decoder; no rotated boxes or NMS.
        self.decoder = nn.Sequential(conv_block(width * 4, width), conv_block(width, 64))
        self.detail = nn.Sequential(conv_block(3, 32), conv_block(32, 32))
        self.segmentation_head = nn.Sequential(conv_block(96, 64), nn.Conv2d(64, num_classes, 1))

    def load_backbone(self, path):
        """Load compatible plain ResNet or extract it from a full U-Net checkpoint."""
        from .pretrained import load_resnet_backbone
        return load_resnet_backbone(self.backbone, path)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for layer in self.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image, return_aux=False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB [B,3,H,W]")
        size = image.shape[-2:]
        if min(size) < 64 or any(s % 32 for s in size):
            raise ValueError("Spatial dimensions must be >=64 and divisible by 32")
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(image))))
        features = []
        for layer in [b.layer1, b.layer2, b.layer3, b.layer4]:
            x = layer(x)
            features.append(x)
        xs = [proj(feat) for proj, feat in zip(self.laterals, features)]
        # Float32 protects global correlation and local distance computations in AMP.
        # Only feature-interaction path is FP32; backbone and dense head can use AMP.
        with torch.autocast(device_type=image.device.type, enabled=False):
            xs = [t.float() for t in xs]
            xs[-1] = self.mpm(xs[-1])
            for i in range(2, -1, -1):
                xs[i] = self.ssaf[i](xs[i + 1], xs[i])
            aux = self.foreground_head(xs[0])
            xs[0] = self.foreground_smooth(xs[0] * (1 + aux.sigmoid()))
            for i in range(3):
                xs[i + 1] = self.fem[i](xs[i], xs[i + 1])
        fused = self.decoder(torch.cat([
            F.interpolate(t, size=xs[0].shape[-2:], mode="bilinear", align_corners=False)
            for t in xs], dim=1))
        detail = self.detail(F.interpolate(image, scale_factor=0.5,
                                           mode="bilinear", align_corners=False))
        fused = F.interpolate(fused, size=detail.shape[-2:], mode="bilinear",
                              align_corners=False)
        logits = self.segmentation_head(torch.cat([fused, detail], dim=1))
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if return_aux:
            return {"logits": logits, "aux_logits": aux}
        return logits
