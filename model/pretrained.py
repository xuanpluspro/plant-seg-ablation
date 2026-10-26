"""Strict extraction of torchvision-compatible ResNet-50 backbone tensors."""
from collections.abc import Mapping
from pathlib import Path
import torch


def load_resnet_backbone(backbone, path):
    """Accept plain torchvision weights or a scoped backbone inside a full model.

    Validate all keys/shapes before loading: never silently initialize only a
    fraction of a backbone with strict=False. Full-model decoder keys are only
    skipped when an explicit resnet./backbone. namespace identifies the encoder.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    wrappers = ("state_dict", "model_state_dict", "model", "net")
    while isinstance(state, Mapping):
        nested = [state[k] for k in wrappers if isinstance(state.get(k), Mapping)]
        if not nested:
            break
        if len(nested) > 1:
            raise ValueError("Ambiguous checkpoint: multiple model state dictionaries")
        state = nested[0]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a nonempty model state dictionary")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint parameter names must be strings")
        while key.startswith("module."):
            key = key[len("module."):]
        if key in normalized:
            raise ValueError(f"Duplicate parameter after removing module prefix: {key}")
        normalized[key] = value
    prefixes = [p for p in ("resnet.", "backbone.")
                if any(k.startswith(p) for k in normalized)]
    if len(prefixes) > 1:
        raise ValueError("Ambiguous checkpoint: both resnet. and backbone. namespaces")
    prefix = prefixes[0] if prefixes else ""
    cleaned, ignored = {}, 0
    for key, value in normalized.items():
        if prefix and not key.startswith(prefix):
            ignored += 1  # full-model decoder/head; never part of this backbone
            continue
        name = key[len(prefix):]
        if name.startswith("fc."):
            ignored += 1  # unused ImageNet classifier
            continue
        cleaned[name] = value
    expected = backbone.state_dict()
    missing = sorted(expected.keys() - cleaned.keys())
    unexpected = sorted(cleaned.keys() - expected.keys())
    mismatched = [f"{k}: expected {tuple(expected[k].shape)}, got "
                  f"{tuple(cleaned[k].shape) if isinstance(cleaned[k], torch.Tensor) else type(cleaned[k]).__name__}"
                  for k in expected.keys() & cleaned.keys()
                  if not isinstance(cleaned[k], torch.Tensor) or cleaned[k].shape != expected[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError("Incompatible ResNet-50 backbone; no weights loaded. "
                         f"Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}; "
                         f"shape/type mismatches: {mismatched[:8]}")
    backbone.load_state_dict(cleaned, strict=True)
    return dict(namespace=prefix or "torchvision", loaded_tensors=len(cleaned),
                ignored_entries=ignored, parameters=sum(p.numel() for p in backbone.parameters()))
