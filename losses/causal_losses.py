"""Causal loss helpers."""
from __future__ import annotations

import torch


def stability_loss(
    pred: torch.Tensor,
    pred_masked: torch.Tensor,
    mode: str = "l1",
    delta: float = 1.0,
) -> torch.Tensor:
    """Counterfactual stability between predictions."""
    if mode == "l1":
        return torch.mean(torch.abs(pred - pred_masked))
    if mode == "huber":
        diff = pred - pred_masked
        abs_diff = diff.abs()
        delta_tensor = diff.new_full((), float(delta))
        quadratic = torch.minimum(abs_diff, delta_tensor)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 / delta_tensor + delta_tensor * linear
        return loss.mean()
    raise NotImplementedError(f"mode='{mode}' is not supported.")
