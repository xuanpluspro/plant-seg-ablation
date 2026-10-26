"""Torch-logit adapter to the shared dataset metric protocol."""
from segmentation_metrics import DatasetSegmentationMetrics


class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.meter = DatasetSegmentationMetrics(num_classes, ignore_index=None)

    def update(self, logits, target):
        if logits.ndim != 4 or logits.shape[1] != self.num_classes:
            raise ValueError("Expected matching K-channel logits")
        self.meter.update(logits.argmax(1), target)

    def compute(self):
        result = self.meter.compute()
        # Compatibility aliases; canonical lowercase fields are exported to CSV.
        aliases = {"Foreground IoU": "foreground_miou", "Union Foreground IoU": "union_foreground_iou",
                   "Background IoU": "background_iou", "Mean IoU": "miou", "Dice": "dice",
                   "Precision": "precision", "Recall": "recall", "Specificity": "specificity",
                   "Pixel Accuracy": "pixacc", "Mean Accuracy": "macc",
                   "Frequency Weighted IoU": "fwiou", "HD95": "hd95", "ASSD": "assd",
                   "Boundary F1": "boundary_f1"}
        result.update({old: result[new] for old, new in aliases.items()})
        for row in result["per_class"]:
            row.update(IoU=row["iou"], Accuracy=row["recall"], Dice=row["dice"],
                       Precision=row["precision"], Recall=row["recall"], pixels=row["support"])
        return result
