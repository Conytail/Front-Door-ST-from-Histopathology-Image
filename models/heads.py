"""Prediction heads for gene expression and cell mix regression."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class GeneExpressionHead(nn.Module):
    """Linear or MLP head producing gene expression scores."""

    def __init__(self, in_dim: int, num_genes: int, hidden: int | None = None) -> None:
        super().__init__()
        if hidden is None:
            self.proj = nn.Linear(in_dim, num_genes)
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_genes),
            )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


class CellMixHead(nn.Module):
    """Predict cell-composition vector from image features."""

    def __init__(self, in_dim: int, num_cellmix: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_cellmix),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class GeneFromCellmixLinear(nn.Module):
    """Linear mapping from cell-mix proportions to gene expression scores."""

    def __init__(self, num_cellmix: int, num_genes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_cellmix, num_genes)

    def forward(self, m_in: torch.Tensor) -> torch.Tensor:
        """Project cell-mix proportions of shape (B, K) to gene logits (B, G)."""
        return self.linear(m_in)


class GeneResidualHead(nn.Module):
    """Residual MLP that learns nonlinear refinements conditioned on fused features."""

    def __init__(self, in_dim: int, num_genes: int, hidden: Iterable[int]) -> None:
        super().__init__()
        hidden_layers = list(hidden)
        if not hidden_layers:
            raise ValueError("hidden must contain at least one positive integer.")

        layers: List[nn.Module] = []
        current_dim = in_dim
        for width in hidden_layers:
            layers.append(nn.Linear(current_dim, int(width)))
            layers.append(nn.GELU())
            current_dim = int(width)
        layers.append(nn.Linear(current_dim, num_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Compute residual gene predictions given fused representation."""
        return self.net(fused)


class FusionGate(nn.Module):
    """
    Fuse image features z and cell-mix M_in using concat or FiLM with optional gating.

    y_hat residuals operate on f_res([z', M_in]) where z' is optionally gated/FiLM modulated.
    """

    def __init__(
        self,
        in_dim_z: int,
        in_dim_m: int,
        mode: str = "concat",
        gated: bool = False,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        mode_lower = mode.lower()
        if mode_lower not in {"concat", "film"}:
            raise ValueError(f"Unsupported fusion mode {mode!r}")

        self.in_dim_z = in_dim_z
        self.in_dim_m = in_dim_m
        self.default_mode = mode_lower
        self.gated_default = gated
        self.hidden_dim = hidden_dim
        self.output_dim = in_dim_z + in_dim_m

        self.film_net: Optional[nn.Module] = None
        if self.default_mode == "film":
            self.film_net = self._build_film_net()

        self.gate_net: Optional[nn.Module] = self._build_gate_net() if gated else None

    def _build_film_net(self) -> nn.Module:
        net = nn.Sequential(
            nn.Linear(self.in_dim_m, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.in_dim_z * 2),
        )
        # Start near identity: gamma≈0, beta≈0
        final_linear = net[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        return net

    def _build_gate_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.in_dim_z + self.in_dim_m, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.in_dim_z),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        m_in: torch.Tensor,
        mode: Optional[str] = None,
        gated: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Merge z and M_in according to the requested fusion strategy.

        Args:
            z: Image features of shape (B, D).
            m_in: Cell-mix proportions of shape (B, K).
            mode: Optional override ("concat" | "film").
            gated: Optional override to enable/disable gating.
        """

        fusion_mode = (mode or self.default_mode).lower()
        if fusion_mode not in {"concat", "film"}:
            raise ValueError(f"Unsupported fusion mode {fusion_mode!r}")

        use_gated = self.gated_default if gated is None else gated
        gate_net = self.gate_net
        if use_gated and gate_net is None:
            gate_net = self._build_gate_net()
            self.gate_net = gate_net

        z_mod = z
        if use_gated and gate_net is not None:
            gate = gate_net(torch.cat([z, m_in], dim=1))
            z_mod = z * gate

        if fusion_mode == "film":
            film_net = self.film_net
            if film_net is None:
                film_net = self._build_film_net()
                self.film_net = film_net
            gamma_beta = film_net(m_in)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            z_mod = z_mod * (1.0 + gamma) + beta

        return torch.cat([z_mod, m_in], dim=1)
