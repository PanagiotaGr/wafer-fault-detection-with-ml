"""
CNN architectures for wafer defect classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """
    Three-block convolutional network with BatchNorm, MaxPool, and a
    two-layer classifier head.

    Args:
        num_classes: Number of output classes.
        image_size:  Input spatial resolution (assumed square).
        channels:    Feature-map sizes for each conv block.
        dropout:     Dropout rate before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 64,
        channels: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]

        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Spatial size after len(channels) MaxPool2d(2) operations
        spatial = image_size // (2 ** len(channels))
        flat_dim = in_ch * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, cfg: dict) -> WaferCNN:
    """Convenience factory that reads the 'cnn' section of config."""
    return WaferCNN(
        num_classes=num_classes,
        image_size=cfg.get("image_size", 64),
        channels=cfg.get("channels", [16, 32, 64]),
        dropout=cfg.get("dropout", 0.3),
    )
