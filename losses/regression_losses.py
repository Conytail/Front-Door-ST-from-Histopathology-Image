"""Regression loss helpers with optional masking support."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for regression tasks."""
    return torch.nn.functional.mse_loss(pred, target)


def huber_loss(pred: torch.Tensor, tgt: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Element-wise Huber loss averaged over batch and gene dimensions."""
    diff = pred - tgt
    abs_diff = diff.abs()
    delta_tensor = diff.new_full((), float(delta))
    quadratic = torch.minimum(abs_diff, delta_tensor)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 / delta_tensor + delta_tensor * linear
    return loss.mean()


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Huber loss with a per-element mask.

    Args:
        pred: Predicted tensor of shape (B, G).
        target: Target tensor of shape (B, G).
        mask: Boolean or float mask with the same shape; masked entries are ignored.
        delta: Transition point for Huber loss.
    """
    mask_f = mask.to(pred.dtype)
    valid = mask_f > 0
    if not valid.any():
        return pred.new_zeros(())

    diff = (pred - target) * mask_f
    abs_diff = diff.abs()
    delta_tensor = diff.new_full((), float(delta))
    quadratic = torch.minimum(abs_diff, delta_tensor)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 / delta_tensor + delta_tensor * linear
    loss = loss * mask_f
    normaliser = mask_f.sum()
    return loss.sum() / torch.clamp(normaliser, min=1.0)


def correlation_loss(pred: torch.Tensor, tgt: torch.Tensor, mode: str = "pearson") -> torch.Tensor:
    """Compute 1 - correlation per sample averaged across the batch."""
    if mode != "pearson":
        raise NotImplementedError(f"mode={mode} is not supported")
    eps = 1e-8
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
    pred_norm = pred_centered.norm(dim=1, keepdim=True)
    tgt_norm = tgt_centered.norm(dim=1, keepdim=True)
    pred_norm = torch.where(pred_norm > eps, pred_norm, pred_norm.new_full(pred_norm.shape, eps))
    tgt_norm = torch.where(tgt_norm > eps, tgt_norm, tgt_norm.new_full(tgt_norm.shape, eps))
    pred_unit = pred_centered / pred_norm
    tgt_unit = tgt_centered / tgt_norm
    corr = (pred_unit * tgt_unit).sum(dim=1)
    valid = (pred_norm.squeeze(1) > eps) & (tgt_norm.squeeze(1) > eps)
    corr = torch.where(valid, corr, torch.zeros_like(corr))
    return 1.0 - corr.mean()


def masked_correlation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked Pearson correlation loss computed per sample then averaged.

    Implements:
        r = cov(masked_pred, masked_target) / (sigma_pred * sigma_target + eps)
        corr_loss = 1 - r
    """
    mask_f = mask.to(pred.dtype)
    valid_counts = mask_f.sum(dim=1)
    valid = valid_counts > 0
    if not valid.any():
        return pred.new_zeros(())

    pred_masked = pred * mask_f
    target_masked = target * mask_f
    denom = torch.clamp(valid_counts, min=1.0)

    mu_pred = pred_masked.sum(dim=1) / denom
    mu_target = target_masked.sum(dim=1) / denom

    pred_centered = (pred - mu_pred.unsqueeze(1)) * mask_f
    target_centered = (target - mu_target.unsqueeze(1)) * mask_f

    cov = (pred_centered * target_centered).sum(dim=1) / denom
    var_pred = (pred_centered**2).sum(dim=1) / denom
    var_target = (target_centered**2).sum(dim=1) / denom

    sigma_pred = torch.sqrt(var_pred + eps)
    sigma_target = torch.sqrt(var_target + eps)

    corr = cov / (sigma_pred * sigma_target + eps)
    corr = torch.where(valid, corr, torch.zeros_like(corr))
    loss = 1.0 - corr
    return loss[valid].mean()


def cellmix_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """Cell composition loss supporting mse, huber, and kl options."""
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "huber":
        return F.huber_loss(pred, target)
    if mode == "kl":
        p = pred.clamp_min(1e-6)
        q = target.clamp_min(1e-6)
        return F.kl_div(p.log(), q, reduction="batchmean")
    raise ValueError(f"Unknown cellmix loss mode: {mode}")
