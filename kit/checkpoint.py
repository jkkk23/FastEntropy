"""Strict checkpoint loading for the single submitted architecture."""

import torch

from config import OPTIMAL_ARCHITECTURE


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict(checkpoint_path, map_location="cpu"):
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid entropy-model checkpoint: {checkpoint_path}")
    if "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_config: {checkpoint_path}")
    return checkpoint["model_state_dict"], checkpoint


def resolve_model_config(state_dict, checkpoint, requested=None):
    del state_dict
    config = dict(checkpoint["model_config"])
    mismatch = {
        name: (config.get(name), expected)
        for name, expected in OPTIMAL_ARCHITECTURE.items()
        if config.get(name) != expected
    }
    if mismatch:
        raise ValueError(
            "Checkpoint is not the submitted FastEntropy architecture: "
            f"{mismatch}"
        )
    if requested:
        for name in ("channels", "kernel_size"):
            value = requested.get(name)
            if value not in (None, 0) and int(value) != int(config[name]):
                raise ValueError(
                    f"Requested {name}={value} does not match checkpoint "
                    f"value {config[name]}"
                )
    return config


def load_model_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=True)
