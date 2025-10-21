from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from train.evaluator import compute_gene_metrics


def test_compute_gene_metrics_shapes() -> None:
    pred = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target = np.array([[0.1, 0.1, 0.3], [0.4, 0.7, 0.6]], dtype=np.float32)
    gene_names = ["g1", "g2", "g3"]
    metrics = compute_gene_metrics(pred, target, gene_names)
    assert len(metrics["gene_names"]) == 3
    assert metrics["pearson"].shape == (3,)
    assert metrics["spearman"].shape == (3,)
    assert isinstance(metrics["pearson_mean"], float)
