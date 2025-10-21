"""Mask utility stubs.
TODO: Implement real spatial mask generation for causal studies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def make_dummy_mask(shape: tuple[int, int], strategy: str = "none") -> np.ndarray:
    """Return a binary mask placeholder."""
    if strategy != "none":
        print(f"[MASK] Requested strategy={strategy}, returning ones mask.")
        return np.ones(shape, dtype=np.uint8)
    return np.zeros(shape, dtype=np.uint8)


def apply_mask(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply a dummy mask that currently does nothing."""
    if mask is None:
        return image
    return np.where(mask[..., None] > 0, image, image)
