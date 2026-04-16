"""
Utility helpers: config loading, seed setting, device resolution.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN ops (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto") -> torch.device:
    """
    Resolve the compute device.
    'auto' picks CUDA > MPS > CPU in that order.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def ensure_output_dirs(cfg: dict) -> None:
    for key in ("figures_dir", "results_dir", "models_dir"):
        path = cfg.get("outputs", {}).get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
