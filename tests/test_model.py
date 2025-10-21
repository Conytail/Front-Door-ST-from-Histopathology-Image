from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.stnet import build_model_from_cfg


def _frontdoor_cfg() -> dict:
    return {
        "enable": True,
        "fusion_mode": "concat",
        "gated": False,
        "detach_cell_head": True,
        "residual_hidden": [128, 64],
    }


def test_model_forward_smoke() -> None:
    cfg = {"backbone": "DenseNet121", "num_genes": 250, "num_cellmix": 6, "use_frontdoor": False}
    model = build_model_from_cfg(cfg)
    dummy = torch.randn(2, 3, 224, 224)
    preds, extras = model(dummy)
    assert preds.shape == (2, 250)
    assert "feats" in extras


def test_frontdoor_forward_shapes() -> None:
    cfg = {
        "backbone": "DenseNet121",
        "num_genes": 250,
        "num_cellmix": 6,
        "use_frontdoor": True,
    }
    model = build_model_from_cfg(cfg, frontdoor_cfg=_frontdoor_cfg())
    batch = 4
    num_genes = cfg["num_genes"]
    dummy = torch.randn(batch, 3, 224, 224)
    m_student = torch.softmax(torch.randn(batch, cfg["num_cellmix"]), dim=1)
    m_teacher = torch.softmax(torch.randn(batch, cfg["num_cellmix"]), dim=1)

    preds, extras = model(
        dummy,
        m_student=m_student,
        m_teacher=m_teacher,
        mode="blend",
        tau=0.5,
    )
    assert preds.shape == (batch, num_genes)
    assert extras["y_lin"].shape == (batch, num_genes)
    assert extras["y_res"].shape == (batch, num_genes)
    rowsum_median = torch.median(torch.abs(extras["m_in_rowsum"] - 1.0)).item()
    assert rowsum_median <= 1e-3


def test_frontdoor_no_leak_on_eval() -> None:
    cfg = {
        "backbone": "DenseNet121",
        "num_genes": 250,
        "num_cellmix": 6,
        "use_frontdoor": True,
    }
    model = build_model_from_cfg(cfg, frontdoor_cfg=_frontdoor_cfg())
    model.eval()
    batch = 3
    dummy = torch.randn(batch, 3, 224, 224)
    m_student = torch.softmax(torch.randn(batch, cfg["num_cellmix"]), dim=1)
    m_teacher = torch.softmax(torch.randn(batch, cfg["num_cellmix"]), dim=1)

    preds, extras = model(
        dummy,
        m_student=m_student,
        m_teacher=m_teacher,
        mode="teacher",
        tau=1.0,
    )
    assert preds.shape == (batch, cfg["num_genes"])
    assert extras["mode"] == "student"
    assert torch.allclose(extras["m_in"], m_student, atol=1e-6)


def test_detach_cell_head() -> None:
    cfg = {
        "backbone": "DenseNet121",
        "num_genes": 10,
        "num_cellmix": 4,
        "use_frontdoor": True,
    }
    frontdoor_cfg = _frontdoor_cfg()
    frontdoor_cfg["detach_cell_head"] = True
    model = build_model_from_cfg(cfg, frontdoor_cfg=frontdoor_cfg)
    model.train()

    batch = 2
    dummy = torch.randn(batch, 3, 224, 224)
    m_student = torch.softmax(torch.randn(batch, cfg["num_cellmix"]), dim=1)
    m_student.requires_grad_(True)

    preds, _ = model(
        dummy,
        m_student=m_student,
        m_teacher=None,
        mode="blend",
        tau=0.6,
        detach_cellmix=True,
    )
    loss = preds.sum()
    loss.backward()
    assert m_student.grad is None or torch.allclose(m_student.grad, torch.zeros_like(m_student.grad))
