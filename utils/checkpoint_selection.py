"""Validation mIoU selection and honest legacy checkpoint metadata."""
import math

SELECTION_METRIC = "Mean IoU"


def validation_score(metrics):
    score = metrics[SELECTION_METRIC]
    if score is None or not math.isfinite(score):
        raise ValueError("Validation Mean IoU must be defined and finite")
    return float(score)


def checkpoint_selection_metric(checkpoint):
    metric = checkpoint.get("selection_metric")
    if metric is not None:
        return metric
    # Before explicit metadata, binary runs selected foreground IoU.
    return ("Foreground IoU" if checkpoint.get("model_config", {}).get("num_classes", 2) == 2
            else SELECTION_METRIC)


def resume_best_score(checkpoint):
    if checkpoint_selection_metric(checkpoint) != SELECTION_METRIC:
        raise ValueError(
            "Cannot resume a run selected by Foreground IoU under Mean IoU selection. "
            "Start a new run with backbone pretraining and a new --output; "
            "the old checkpoint can still be evaluated.")
    score = checkpoint.get("best_score")
    if score is None:
        # Legacy multiclass runs stored mIoU under this misleading key.
        score = checkpoint.get("best_foreground_iou")
    return validation_score({SELECTION_METRIC: score})
