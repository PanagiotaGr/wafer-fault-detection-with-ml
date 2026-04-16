"""
Loss functions for wafer defect classification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for multi-class classification.

    Args:
        gamma:  Focusing parameter. Higher = harder examples weighted more.
        weight: Per-class weights (same as nn.CrossEntropyLoss `weight`).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Gather log-prob of the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.weight is not None:
            class_w = self.weight[targets]
            focal_weight = focal_weight * class_w

        return -(focal_weight * log_pt).mean()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_loss(
    loss_name: str,
    y_train: np.ndarray,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Build a loss criterion from a name string.

    Supported names:
        cross_entropy           — plain CE, no class weighting
        weighted_cross_entropy  — CE with inverse-frequency class weights
        focal                   — Focal loss with inverse-frequency weighting
    """
    class_weights = _compute_class_weights(y_train, device)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. "
                         "Choose: cross_entropy | weighted_cross_entropy | focal")


def _compute_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y)
    weights = counts.sum() / (len(counts) * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32, device=device)
