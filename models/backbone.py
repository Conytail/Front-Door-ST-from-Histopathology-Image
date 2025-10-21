"""Backbone factory for gene expression prediction.
TODO: integrate additional architectures (ViT, ConvNeXt, etc.).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TinyBackbone(nn.Module):
    """Lightweight CNN used when torchvision backbones are unavailable."""

    def __init__(self, out_channels: int = 1024) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.pool(feats)
        return pooled.flatten(1)


class DenseNetBackbone(nn.Module):
    """DenseNet121 wrapper that outputs pooled feature vectors."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import densenet121

        model = densenet121(weights=None)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.pool(feats)
        return pooled.flatten(1)


def create_backbone(name: str = "DenseNet121") -> Tuple[nn.Module, int]:
    """Instantiate a backbone that outputs global feature vectors."""

    name_lower = name.lower()
    out_dim = 1024
    if name_lower == "densenet121":
        try:
            backbone = DenseNetBackbone()
            return backbone, out_dim
        except (ImportError, ModuleNotFoundError):
            print("[BACKBONE] torchvision not available, using TinyBackbone.")
    elif name_lower != "tiny":
        print(f"[BACKBONE] Unknown backbone={name}, falling back to TinyBackbone.")
    backbone = TinyBackbone(out_channels=out_dim)
    return backbone, out_dim
