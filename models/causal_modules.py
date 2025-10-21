"""Causal module utilities for counterfactual stability."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class RandomBlockMasker(nn.Module):
    """Random block/area masking for counterfactual stability."""

    def __init__(
        self,
        num_blocks: int = 4,
        block_size: int = 64,
        area_ratio: float = 0.15,
        fill: str = "zeros",
        gaussian_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_blocks = max(1, int(num_blocks))
        self.block_size = max(0, int(block_size))
        self.area_ratio = float(area_ratio)
        self.fill = fill
        self.gaussian_std = float(gaussian_std)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("RandomBlockMasker expects input of shape (B, C, H, W).")
        batch, _, height, width = x.shape
        device = x.device
        masked = x.clone()

        if self.block_size > 0:
            block_side = self.block_size
        else:
            ratio = min(max(self.area_ratio, 1e-3), 1.0)
            total_pixels = max(1, int(ratio * height * width))
            block_side = max(1, int(math.sqrt(total_pixels / self.num_blocks)))

        for b_idx in range(batch):
            for _ in range(self.num_blocks):
                block_h = min(block_side, height)
                block_w = min(block_side, width)
                top = torch.randint(0, max(1, height - block_h + 1), (1,), device=device).item()
                left = torch.randint(0, max(1, width - block_w + 1), (1,), device=device).item()
                fill_value = self._sample_fill(masked[b_idx], block_h, block_w)
                masked[b_idx, :, top : top + block_h, left : left + block_w] = fill_value
        return torch.clamp(masked, 0.0, 1.0)

    def _sample_fill(self, img: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.fill == "mean":
            value = img.mean(dim=(1, 2), keepdim=True)
            return value.expand(-1, height, width)
        if self.fill == "gaussian":
            noise = torch.randn((img.shape[0], height, width), device=img.device, dtype=img.dtype)
            noise = noise * self.gaussian_std + 0.5
            return torch.clamp(noise, 0.0, 1.0)
        return torch.zeros((img.shape[0], height, width), device=img.device, dtype=img.dtype)


def make_masked_image(x: torch.Tensor, strategy: str, cfg: Dict) -> torch.Tensor:
    """Factory wrapper for causal masking strategies."""
    if strategy == "random_blocks":
        rb_cfg = cfg.get("random_blocks", {})
        masker = RandomBlockMasker(
            num_blocks=rb_cfg.get("num_blocks", 4),
            block_size=rb_cfg.get("block_size", 64),
            area_ratio=rb_cfg.get("area_ratio", 0.15),
            fill=rb_cfg.get("fill", "zeros"),
            gaussian_std=rb_cfg.get("gaussian_std", 0.1),
        )
        return masker(x)
    raise NotImplementedError(f"Masking strategy {strategy!r} is not implemented.")