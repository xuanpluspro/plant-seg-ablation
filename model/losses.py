"""Pixel supervision replaces OSSDet's box-derived activation targets."""
import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=1.0, aux_weight=0.2):
        super().__init__()
        if dice_weight < 0 or aux_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        self.dice_weight, self.aux_weight = dice_weight, aux_weight

    def forward(self, output, target):
        logits = output["logits"].float() if isinstance(output, dict) else output.float()
        if target.dtype != torch.long or target.min() < 0 or target.max() >= logits.shape[1]:
            raise ValueError("Loss expects class IDs 0..K-1 with background=0")
        ce = F.cross_entropy(logits, target)
        plant = (target > 0).float()
        probabilities = logits.softmax(1)
        if logits.shape[1] == 2:
            prob = probabilities[:, 1]
            intersection = (prob * plant).sum((1, 2))
            denominator = (prob + plant).sum((1, 2))
            dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
        else:
            truth = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()[:, 1:]
            prob = probabilities[:, 1:]
            intersection = (prob * truth).sum((2, 3))
            denominator = (prob + truth).sum((2, 3))
            dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
        aux = logits.new_zeros(())
        if isinstance(output, dict) and output.get("aux_logits") is not None:
            auxiliary = F.interpolate(output["aux_logits"].float(),
                                      size=target.shape[-2:], mode="bilinear",
                                      align_corners=False)[:, 0]
            aux = F.binary_cross_entropy_with_logits(auxiliary, plant)
        return ce + self.dice_weight * dice + self.aux_weight * aux
