from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.transforms import AlbumentationsTransformWithCoords


def test_transform_stub() -> None:
    transform = AlbumentationsTransformWithCoords(enabled=False)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    tensor, meta = transform(image, meta={"global_id": "dummy"})
    assert tensor.shape[0] == 3
    assert meta["global_id"] == "dummy"
