"""ST-Net assembly with optional front-door mediation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import create_backbone
from .heads import (
    CellMixHead,
    FusionGate,
    GeneExpressionHead,
    GeneFromCellmixLinear,
    GeneResidualHead,
)


class STNet(nn.Module):
    """
    Backbone + gene head architecture with optional front-door integration.

    When front-door is enabled the model expects precomputed cell-mix proportions and
    produces gene predictions as:

        y_hat = W_lin @ M_in + b_lin + f_res([z', M_in])

    where M_in is a teacher/student blend and z' is optionally gated/FiLM adjusted.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_genes: int,
        *,
        use_frontdoor: bool,
        num_cellmix: int,
        frontdoor_cfg: Optional[Dict] = None,
        gene_head: Optional[GeneExpressionHead] = None,
        cellmix_head: Optional[CellMixHead] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_genes = num_genes
        self.use_frontdoor = use_frontdoor
        self.num_cellmix = num_cellmix
        self.cellmix_head = cellmix_head

        self.frontdoor_cfg = frontdoor_cfg.copy() if frontdoor_cfg else {}
        self.fusion_mode_default = self.frontdoor_cfg.get("fusion_mode", "concat").lower()
        self.gated_default = bool(self.frontdoor_cfg.get("gated", False))
        self.detach_default = bool(self.frontdoor_cfg.get("detach_cell_head", True))
        self.lin_from_cellmix = bool(self.frontdoor_cfg.get("lin_from_cellmix", True))
        self.residual_hidden = self.frontdoor_cfg.get("residual_hidden", [512, 256])

        if self.use_frontdoor:
            if self.num_cellmix <= 0:
                raise ValueError("num_cellmix must be positive when use_frontdoor is True")
            if not self.residual_hidden:
                raise ValueError("frontdoor residual_hidden must contain at least one layer.")

            self.gene_linear = (
                GeneFromCellmixLinear(self.num_cellmix, self.num_genes) if self.lin_from_cellmix else None
            )
            self.fusion_gate = FusionGate(
                self.feat_dim,
                self.num_cellmix,
                mode=self.fusion_mode_default,
                gated=self.gated_default,
            )
            self.gene_residual = GeneResidualHead(self.fusion_gate.output_dim, self.num_genes, self.residual_hidden)
            self.gene_head = None
        else:
            self.gene_head = gene_head or GeneExpressionHead(self.feat_dim, self.num_genes)
            self.gene_linear = None
            self.fusion_gate = None
            self.gene_residual = None

    def _blend_cellmix(
        self,
        m_student: torch.Tensor,
        m_teacher: Optional[torch.Tensor],
        mode: str,
        tau: float,
        detach: bool,
    ) -> Tuple[torch.Tensor, float, str]:
        """Compute blended front-door input M_in following teacher-student schedule."""
        mode_lower = mode.lower()
        if mode_lower not in {"teacher", "student", "blend"}:
            raise ValueError(f"Unsupported mode {mode}")

        teacher_tensor = m_teacher.detach() if m_teacher is not None else None
        student_tensor = m_student.detach() if detach else m_student

        if teacher_tensor is None:
            tau_effective = 0.0
            mode_lower = "student"
        elif mode_lower == "teacher":
            tau_effective = 1.0
        elif mode_lower == "student":
            tau_effective = 0.0
        else:
            tau_effective = float(torch.clamp(torch.tensor(tau), 0.0, 1.0).item())

        if tau_effective == 0.0 or teacher_tensor is None:
            m_in = student_tensor
        elif tau_effective == 1.0:
            m_in = teacher_tensor
        else:
            m_in = tau_effective * teacher_tensor + (1.0 - tau_effective) * student_tensor

        return m_in, tau_effective, mode_lower

    def forward(
        self,
        x: torch.Tensor,
        *,
        m_student: Optional[torch.Tensor] = None,
        m_teacher: Optional[torch.Tensor] = None,
        mode: str = "student",
        tau: float = 0.0,
        detach_cellmix: Optional[bool] = None,
        fusion_mode: Optional[str] = None,
        gated: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone(x)
        extras: Dict[str, torch.Tensor] = {"feats": feats}

        if not self.use_frontdoor:
            if self.gene_head is None:
                raise RuntimeError("Baseline gene head is not initialized.")
            pred_gene = self.gene_head(feats)
            if self.cellmix_head is not None:
                extras["cellmix"] = self.cellmix_head(feats)
            return pred_gene, extras

        if m_student is None:
            raise ValueError("m_student must be provided when front-door is enabled.")

        detach = self.detach_default if detach_cellmix is None else bool(detach_cellmix)
        if not self.training:
            # Evaluation enforces pure student mixing.
            mode = "student"
            tau = 0.0

        m_in, tau_eff, mode_used = self._blend_cellmix(
            m_student=m_student,
            m_teacher=m_teacher,
            mode=mode,
            tau=tau,
            detach=detach,
        )

        fusion_mode_used = fusion_mode or self.fusion_mode_default
        gated_used = self.gated_default if gated is None else gated

        fused = self.fusion_gate(feats, m_in, mode=fusion_mode_used, gated=gated_used)
        y_lin = self.gene_linear(m_in) if self.gene_linear is not None else torch.zeros(
            (m_in.size(0), self.num_genes), device=m_in.device, dtype=m_in.dtype
        )
        y_res = self.gene_residual(fused)
        pred_gene = y_lin + y_res

        extras.update(
            {
                "tau": torch.tensor(tau_eff, device=pred_gene.device),
                "mode": mode_used,
                "m_in": m_in,
                "m_student": m_student,
                "y_lin": y_lin,
                "y_res": y_res,
                "fusion_mode": fusion_mode_used,
                "gated": gated_used,
            }
        )
        if m_teacher is not None:
            extras["m_teacher"] = m_teacher
        row_sums = m_in.sum(dim=1)
        extras["m_in_rowsum"] = row_sums
        return pred_gene, extras


def build_model_from_cfg(cfg_dict: Dict, frontdoor_cfg: Optional[Dict] = None) -> STNet:
    """Build STNet instance from configuration dictionary."""
    backbone_name = cfg_dict.get("backbone") or cfg_dict.get("BACKBONE", "DenseNet121")
    backbone, feat_dim = create_backbone(backbone_name)
    num_genes = int(cfg_dict.get("num_genes") or cfg_dict.get("OUTPUT_DIM", 250))
    num_cellmix = int(cfg_dict.get("num_cellmix", cfg_dict.get("NUM_CELLMIX", 0)))

    frontdoor_cfg = frontdoor_cfg or cfg_dict.get("frontdoor")
    use_frontdoor = bool(
        cfg_dict.get("use_frontdoor")
        or cfg_dict.get("USE_FRONTDOOR", False)
        or (frontdoor_cfg and frontdoor_cfg.get("enable", False))
    )

    gene_head_hidden = cfg_dict.get("gene_head_hidden")
    hidden_dim = int(gene_head_hidden) if isinstance(gene_head_hidden, (int, float, str)) else None

    gene_head = None
    if not use_frontdoor:
        gene_head = GeneExpressionHead(feat_dim, num_genes, hidden=hidden_dim)

    cellmix_head = None
    if cfg_dict.get("use_cellmix_head", False):
        if num_cellmix <= 0:
            raise ValueError("num_cellmix must be positive when use_cellmix_head is True")
        cellmix_head = CellMixHead(feat_dim, num_cellmix)

    return STNet(
        backbone=backbone,
        feat_dim=feat_dim,
        num_genes=num_genes,
        use_frontdoor=use_frontdoor,
        num_cellmix=num_cellmix,
        frontdoor_cfg=frontdoor_cfg,
        gene_head=gene_head,
        cellmix_head=cellmix_head,
    )
