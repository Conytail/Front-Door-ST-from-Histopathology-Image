"""Trainer implementation with multi-task baseline and counterfactual hooks."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from losses.causal_losses import stability_loss
from losses.regression_losses import (
    cellmix_loss,
    correlation_loss,
    huber_loss,
    masked_correlation_loss,
    masked_huber,
)
from models.causal_modules import make_masked_image
from train.evaluator import (
    compute_quartile_auroc,
    pearson_per_gene,
    pearson_per_gene_masked,
    spearman_per_gene,
    spearman_per_gene_masked,
)


@dataclass
class TrainerConfig:
    """Configuration subset required for training."""

    max_steps: int
    print_freq: int
    cf_cfg: Dict
    epochs: int
    save_interval: int
    auroc_quantile: float
    eval_interval: int = 0
    frontdoor_cfg: Dict = field(default_factory=dict)
    artifact_dir: Optional[Path] = None


class Trainer:
    """Trainer supporting baseline, optional cell-mix, and counterfactual losses."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        cfg: TrainerConfig,
        loss_cfg: Dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.loss_cfg = loss_cfg
        self.logger = logger or logging.getLogger(__name__)

        self.frontdoor_cfg = cfg.frontdoor_cfg or {}
        self.eval_interval = cfg.eval_interval if cfg.eval_interval > 0 else 1
        self.fd_loss_cfg = self.frontdoor_cfg.get("loss", {})
        self.fusion_mode_default = self.frontdoor_cfg.get("fusion_mode", "concat")
        self.gated_default = bool(self.frontdoor_cfg.get("gated", False))
        self.artifact_dir = Path(cfg.artifact_dir) if cfg.artifact_dir else None
        self._row_sum_history: list[Dict[str, object]] = []
        self._var_history: list[Dict[str, object]] = []

    # --------------------------------------------------------------------- #
    # Front-door helpers
    # --------------------------------------------------------------------- #
    def _get_frontdoor_state(self, epoch: int) -> Dict[str, object]:
        """
        Determine teacher/student blending schedule for the given epoch.

        Implements piecewise-linear tau schedule:
            tau_e = 1.0,                                   if e < W
            tau_e = 1.0 - (e - W) / T,                    if W <= e < W + T
            tau_e = 0.0,                                   otherwise
        where W = warmup_epochs and T = transition_epochs.
        """
        if not (self.model.use_frontdoor and self.frontdoor_cfg.get("enable", False)):
            return {
                "mode": "student",
                "tau": 0.0,
                "detach": bool(self.frontdoor_cfg.get("detach_cell_head", True)),
                "fusion_mode": self.fusion_mode_default,
                "gated": self.gated_default,
            }

        use_teacher = bool(self.frontdoor_cfg.get("use_teacher_forcing", True))
        warmup = int(self.frontdoor_cfg.get("warmup_epochs", 0))
        transition = int(self.frontdoor_cfg.get("transition_epochs", 0))
        detach = bool(self.frontdoor_cfg.get("detach_cell_head", True))

        if not use_teacher or transition < 0:
            return {
                "mode": "student",
                "tau": 0.0,
                "detach": detach,
                "fusion_mode": self.fusion_mode_default,
                "gated": self.gated_default,
            }

        tau = 0.0
        if epoch < warmup:
            tau = 1.0
        elif transition > 0 and epoch < warmup + transition:
            tau = 1.0 - (epoch - warmup) / float(max(1, transition))
        else:
            tau = 0.0
        tau = float(np.clip(tau, 0.0, 1.0))

        if tau >= 1.0 - 1e-6:
            mode = "teacher"
        elif tau <= 1e-6:
            mode = "student"
        else:
            mode = "blend"

        return {
            "mode": mode,
            "tau": tau,
            "detach": detach,
            "fusion_mode": self.fusion_mode_default,
            "gated": self.gated_default,
        }

    def _loss_hyperparameters(self) -> Dict[str, float]:
        delta = float(self.fd_loss_cfg.get("huber_delta", self.loss_cfg.get("huber_delta", 1.0)))
        lambda_corr = float(
            self.fd_loss_cfg.get(
                "lambda_corr",
                self.loss_cfg.get("gene_corr_weight", self.loss_cfg.get("corr_weight", 0.0)),
            )
        )
        huber_weight = float(self.loss_cfg.get("gene_huber_weight", self.loss_cfg.get("huber_weight", 1.0)))
        return {
            "delta": delta,
            "lambda_corr": lambda_corr,
            "huber_weight": huber_weight,
        }

    # --------------------------------------------------------------------- #
    # Training and evaluation
    # --------------------------------------------------------------------- #
    def _maybe_apply_cf(self, epoch: int) -> bool:
        cf_cfg = self.cfg.cf_cfg
        if not cf_cfg.get("enable", False):
            return False
        if epoch < int(cf_cfg.get("warmup_epochs", 0)):
            return False
        prob = float(cf_cfg.get("apply_prob", 1.0))
        return torch.rand(1).item() < prob

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        save_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        cf_cfg = self.cfg.cf_cfg
        total_epochs = max(1, self.cfg.epochs)
        save_interval = max(1, self.cfg.save_interval)
        quantile = float(self.cfg.auroc_quantile)
        loss_params = self._loss_hyperparameters()

        for epoch in range(total_epochs):
            self.model.train()
            step = 0
            cf_values: list[float] = []
            gene_values: list[float] = []
            row_sums_epoch: list[torch.Tensor] = []
            y_lin_epoch: list[torch.Tensor] = []
            y_res_epoch: list[torch.Tensor] = []
            last_tau = 0.0
            last_mode = "student"

            fd_state = self._get_frontdoor_state(epoch)

            for batch in train_loader:
                if self.cfg.max_steps and step >= self.cfg.max_steps:
                    break

                images = batch["image"].to(self.device)
                gene_targets = batch["label"].to(self.device)
                gene_mask = batch.get("gene_mask")
                if gene_mask is not None:
                    gene_mask_bool = gene_mask.to(self.device).bool()
                else:
                    gene_mask_bool = torch.ones_like(gene_targets, dtype=torch.bool)

                cell_targets = batch.get("cellmix")
                if cell_targets is not None:
                    cell_targets = cell_targets.to(self.device)

                m_student = batch.get("cellmix_student")
                m_teacher = batch.get("cellmix_teacher")
                if m_student is not None:
                    m_student = m_student.to(self.device)
                if m_teacher is not None:
                    m_teacher = m_teacher.to(self.device)

                if self.model.use_frontdoor:
                    preds, extras = self.model(
                        images,
                        m_student=m_student,
                        m_teacher=m_teacher,
                        mode=fd_state["mode"],
                        tau=fd_state["tau"],
                        detach_cellmix=fd_state["detach"],
                        fusion_mode=fd_state["fusion_mode"],
                        gated=fd_state["gated"],
                    )
                    last_tau = float(extras.get("tau", torch.tensor(fd_state["tau"], device=self.device)).item())
                    last_mode = extras.get("mode", fd_state["mode"])
                    loss_huber = masked_huber(preds, gene_targets, gene_mask_bool, delta=loss_params["delta"])
                    loss_corr = masked_correlation_loss(preds, gene_targets, gene_mask_bool)
                else:
                    preds, extras = self.model(images)
                    loss_huber = huber_loss(preds, gene_targets, delta=loss_params["delta"])
                    loss_corr = correlation_loss(preds, gene_targets)

                gene_loss = loss_params["huber_weight"] * loss_huber + loss_params["lambda_corr"] * loss_corr
                total_loss = gene_loss

                if (
                    not self.model.use_frontdoor
                    and self.model.cellmix_head is not None
                    and cell_targets is not None
                    and cell_targets.numel() > 0
                    and "cellmix" in extras
                ):
                    pred_cell = extras["cellmix"]
                    loss_cell = cellmix_loss(
                        pred_cell,
                        cell_targets,
                        mode=self.loss_cfg.get("cellmix_loss", "mse"),
                    )
                    total_loss = total_loss + float(self.loss_cfg.get("cellmix_weight", 0.0)) * loss_cell
                else:
                    loss_cell = torch.tensor(0.0, device=self.device)

                cf_loss = torch.tensor(0.0, device=self.device)
                if self._maybe_apply_cf(epoch):
                    masked_images = make_masked_image(images, cf_cfg.get("strategy", "random_blocks"), cf_cfg)
                    cf_kwargs = {}
                    if self.model.use_frontdoor:
                        cf_kwargs = {
                            "m_student": m_student,
                            "m_teacher": m_teacher,
                            "mode": fd_state["mode"],
                            "tau": fd_state["tau"],
                            "detach_cellmix": True,
                            "fusion_mode": fd_state["fusion_mode"],
                            "gated": fd_state["gated"],
                        }
                    if not cf_cfg.get("backprop_masked", False):
                        with torch.no_grad():
                            pred_mask, _ = self.model(masked_images, **cf_kwargs) if cf_kwargs else self.model(masked_images)
                    else:
                        pred_mask, _ = self.model(masked_images, **cf_kwargs) if cf_kwargs else self.model(masked_images)
                    cf_loss = stability_loss(preds, pred_mask, mode="l1")
                    total_loss = total_loss + float(cf_cfg.get("weight", 0.0)) * cf_loss
                    cf_values.append(cf_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if step % self.cfg.print_freq == 0:
                    self.logger.info(
                        "[TRAIN] epoch=%d step=%d mode=%s tau=%.4f total=%.4f gene=%.4f huber=%.4f corr=%.4f cf=%.4f",
                        epoch,
                        step,
                        last_mode,
                        last_tau,
                        total_loss.item(),
                        gene_loss.item(),
                        loss_huber.item(),
                        loss_corr.item(),
                        cf_loss.item(),
                    )
                gene_values.append(gene_loss.item())
                step += 1

            if cf_values:
                self.logger.info(
                    "[TRAIN] epoch=%d cf_weight=%.4f cf_mean=%.4f",
                    epoch,
                    cf_cfg.get("weight", 0.0),
                    float(np.mean(cf_values)),
                )

            row_stats: Dict[str, object] | None = None
            var_stats: Dict[str, object] | None = None
            if row_sums_epoch:
                row_concat = torch.cat(row_sums_epoch).cpu().numpy()
                quantiles = {
                    "q0": float(np.quantile(row_concat, 0.0)),
                    "q25": float(np.quantile(row_concat, 0.25)),
                    "q50": float(np.quantile(row_concat, 0.5)),
                    "q75": float(np.quantile(row_concat, 0.75)),
                    "q100": float(np.quantile(row_concat, 1.0)),
                }
                row_stats = {
                    "epoch": epoch,
                    "min": float(row_concat.min()),
                    "max": float(row_concat.max()),
                    "mean": float(row_concat.mean()),
                    "std": float(row_concat.std()),
                    "quantiles": quantiles,
                }
                self._row_sum_history.append(row_stats)

            if y_lin_epoch and y_res_epoch:
                y_lin_concat = torch.cat(y_lin_epoch, dim=0).cpu().numpy()
                y_res_concat = torch.cat(y_res_epoch, dim=0).cpu().numpy()
                var_lin = np.var(y_lin_concat, axis=0)
                var_res = np.var(y_res_concat, axis=0)
                denom = np.clip(var_lin + var_res, a_min=1e-8, a_max=None)
                ratio = var_lin / denom
                var_stats = {
                    "epoch": epoch,
                    "var_lin_mean": float(var_lin.mean()),
                    "var_res_mean": float(var_res.mean()),
                    "ratio_mean": float(np.mean(ratio)),
                    "ratio_median": float(np.median(ratio)),
                }
                self._var_history.append(var_stats)

            gene_mean = float(np.mean(gene_values)) if gene_values else 0.0
            cf_mean = float(np.mean(cf_values)) if cf_values else 0.0

            compute_eval = (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == total_epochs
            compute_quartile = (epoch + 1) % save_interval == 0 or (epoch + 1) == total_epochs
            epoch_logs: Dict[str, float] = {}
            if val_loader is not None and compute_eval:
                epoch_logs = self.evaluate(
                    val_loader,
                    compute_quartile=compute_quartile,
                    quantile=quantile,
                )
                logs = epoch_logs

            row_median = row_stats["quantiles"]["q50"] if row_stats else float("nan")
            var_ratio = var_stats["ratio_mean"] if var_stats else float("nan")
            val_rho = epoch_logs.get("val_spearman_median", float("nan"))

            self.logger.info(
                "[EPOCH] epoch=%d mode=%s tau=%.4f L_gene=%.4f L_cf=%.4f val_rho_med=%.4f row_sum_med=%.6f var_ratio=%.4f",
                epoch,
                last_mode,
                last_tau,
                gene_mean,
                cf_mean,
                val_rho,
                row_median,
                var_ratio,
            )

            if save_callback and ((epoch + 1) % save_interval == 0 or (epoch + 1) == total_epochs):
                save_callback(epoch + 1, epoch_logs or {})

        self._save_artifacts()
        return logs or {}
    def evaluate(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        compute_quartile: bool = False,
        quantile: float = 0.25,
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        huber_losses: list[float] = []
        cell_losses: list[float] = []

        loss_params = self._loss_hyperparameters()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                gene_mask = batch.get("gene_mask")
                if gene_mask is not None:
                    gene_mask_bool = gene_mask.to(self.device).bool()
            else:
                gene_mask_bool = torch.ones_like(labels, dtype=torch.bool)
            all_masks.append(gene_mask_bool.cpu().numpy())

            cell_targets = batch.get("cellmix")
            if cell_targets is not None:
                cell_targets = cell_targets.to(self.device)

                m_student = batch.get("cellmix_student")
                m_teacher = batch.get("cellmix_teacher")
                if m_student is not None:
                    m_student = m_student.to(self.device)
                if m_teacher is not None:
                    m_teacher = m_teacher.to(self.device)

                if self.model.use_frontdoor:
                    preds, extras = self.model(
                        images,
                        m_student=m_student,
                        m_teacher=m_teacher,
                        mode="student",
                        tau=0.0,
                        detach_cellmix=True,
                    )
                    loss_val = masked_huber(preds, labels, gene_mask_bool, delta=loss_params["delta"])
                    loss_corr = masked_correlation_loss(preds, labels, gene_mask_bool)
                else:
                    preds, extras = self.model(images)
                    loss_val = huber_loss(preds, labels, delta=loss_params["delta"])
                    loss_corr = correlation_loss(preds, labels)

                huber_losses.append((loss_params["huber_weight"] * loss_val + loss_params["lambda_corr"] * loss_corr).item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

                if (
                    not self.model.use_frontdoor
                    and self.model.cellmix_head is not None
                    and cell_targets is not None
                    and cell_targets.numel() > 0
                    and "cellmix" in extras
                ):
                    cell_loss = cellmix_loss(extras["cellmix"], cell_targets, mode=self.loss_cfg.get("cellmix_loss", "mse"))
                    cell_losses.append(cell_loss.item())

        if not all_preds:
            self.logger.warning("[VAL] no batches available; returning default metrics.")
            return {
                "val_huber": 0.0,
                "val_pearson_mean": float("nan"),
                "val_pearson_median": float("nan"),
                "val_spearman_mean": float("nan"),
                "val_spearman_median": float("nan"),
            }
        preds_np = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        mask_np = np.concatenate(all_masks, axis=0) if all_masks else None

        if mask_np is not None:
            mask_bool = mask_np.astype(bool)
            pearson_stats = pearson_per_gene_masked(preds_np, targets_np, mask_bool)
            spearman_stats = spearman_per_gene_masked(preds_np, targets_np, mask_bool)
        else:
            pearson_stats = pearson_per_gene(preds_np, targets_np)
            spearman_stats = spearman_per_gene(preds_np, targets_np)
        pearson_mean = pearson_stats["mean"]
        pearson_median = pearson_stats["median"]
        spearman_mean = spearman_stats["mean"]
        spearman_median = spearman_stats["median"]

        metrics: Dict[str, float] = {
            "val_huber": float(np.mean(huber_losses)) if huber_losses else 0.0,
            "val_pearson_mean": pearson_mean,
            "val_pearson_median": pearson_median,
            "val_spearman_mean": spearman_mean,
            "val_spearman_median": spearman_median,
        }
        if cell_losses:
            metrics["val_cellmix"] = float(np.mean(cell_losses))

        if compute_quartile:
            try:
                auroc = compute_quartile_auroc(preds_np, targets_np, quantile=quantile)
            except ImportError:
                auroc = None
                self.logger.warning("[VAL] sklearn not installed; skipping AUROC computation.")

            if auroc is not None:
                metrics["val_auroc_mean"] = auroc["mean"]
                metrics["val_auroc_median"] = auroc["median"]
                self.logger.info(
                    "[VAL] huber=%.4f auroc_mean=%.4f auroc_median=%.4f "
                    "pearson_mean=%.4f pearson_median=%.4f "
                    "spearman_mean=%.4f spearman_median=%.4f",
                    metrics["val_huber"],
                    auroc["mean"],
                    auroc["median"],
                    pearson_mean,
                    pearson_median,
                    spearman_mean,
                    spearman_median,
                )
                return metrics

        self.logger.info(
            "[VAL] huber=%.4f pearson_mean=%.4f pearson_median=%.4f "
            "spearman_mean=%.4f spearman_median=%.4f",
            metrics["val_huber"],
            pearson_mean,
            pearson_median,
            spearman_mean,
            spearman_median,
        )
        return metrics

    def _save_artifacts(self) -> None:
        if self.artifact_dir is None:
            return
        if not self._row_sum_history and not self._var_history:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self._row_sum_history:
            row_path = self.artifact_dir / "m_in_row_sum_stats.json"
            with row_path.open("w", encoding="utf-8") as f:
                json.dump({"history": self._row_sum_history}, f, indent=2)
        if self._var_history:
            var_path = self.artifact_dir / "y_lin_vs_y_res_var_explained.json"
            with var_path.open("w", encoding="utf-8") as f:
                json.dump({"history": self._var_history}, f, indent=2)
