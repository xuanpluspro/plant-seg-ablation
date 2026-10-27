"""Compact inference checkpoints without changing model tensor precision."""
TRAINING_STATE_KEYS = frozenset(("optimizer", "scheduler", "scaler", "rng", "history"))

def inference_checkpoint(checkpoint):
    """Retain model/config/evaluation metadata; omit resume-only training state."""
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Expected a project checkpoint with model and model_config")
    result = {key: value for key, value in checkpoint.items() if key not in TRAINING_STATE_KEYS}
    result["checkpoint_kind"] = "inference"
    return result
