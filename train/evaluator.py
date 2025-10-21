"""Evaluation utilities for gene prediction models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    mse = torch.mean(diff**2).item()
    return {"mse": mse}


def _safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = np.linalg.norm(x_center) * np.linalg.norm(y_center)
    if denom < eps:
        return float("nan")
    return float(np.dot(x_center, y_center) / denom)


def _rankdata(vec: np.ndarray) -> np.ndarray:
    order = np.argsort(vec, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(vec.size, dtype=np.float64)
    unique_vals, start_idx, counts = np.unique(vec, return_index=True, return_counts=True)
    for start, count in zip(start_idx, counts):
        if count > 1:
            indices = np.where(vec == vec[start])[0]
            mean_rank = ranks[indices].mean()
            ranks[indices] = mean_rank
    return ranks


def pearson_per_gene(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, np.ndarray | float]:
    num_genes = pred.shape[1]
    corrs = np.full(num_genes, np.nan, dtype=np.float64)
    for gene in range(num_genes):
        corrs[gene] = _safe_corr(pred[:, gene], tgt[:, gene])
    return {
        "per_gene": corrs,
        "mean": float(np.nanmean(corrs)),
        "median": float(np.nanmedian(corrs)),
    }


def spearman_per_gene(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, np.ndarray | float]:
    num_genes = pred.shape[1]
    corrs = np.full(num_genes, np.nan, dtype=np.float64)
    for gene in range(num_genes):
        pred_col = pred[:, gene]
        tgt_col = tgt[:, gene]
        if pred_col.size < 2 or tgt_col.size < 2:
            continue
        pred_rank = _rankdata(pred_col)
        tgt_rank = _rankdata(tgt_col)
        corrs[gene] = _safe_corr(pred_rank, tgt_rank)
    return {
        "per_gene": corrs,
        "mean": float(np.nanmean(corrs)),
        "median": float(np.nanmedian(corrs)),
    }


def compute_gene_metrics(pred: np.ndarray, tgt: np.ndarray, gene_names: Iterable[str]) -> Dict[str, object]:
    """Compute per-gene Pearson and Spearman statistics."""
    gene_names = list(gene_names)
    pearson = pearson_per_gene(pred, tgt)["per_gene"]
    spearman = spearman_per_gene(pred, tgt)["per_gene"]
    return {
        "gene_names": gene_names,
        "pearson": pearson,
        "spearman": spearman,
        "pearson_mean": float(np.nanmean(pearson)),
        "pearson_median": float(np.nanmedian(pearson)),
        "spearman_mean": float(np.nanmean(spearman)),
        "spearman_median": float(np.nanmedian(spearman)),
    }


def pearson_per_gene_masked(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray | float]:
    num_genes = pred.shape[1]
    corrs = np.full(num_genes, np.nan, dtype=np.float64)
    for gene in range(num_genes):
        valid = mask[:, gene].astype(bool)
        if valid.sum() < 2:
            continue
        corrs[gene] = _safe_corr(pred[valid, gene], tgt[valid, gene])
    return {
        "per_gene": corrs,
        "mean": float(np.nanmean(corrs)),
        "median": float(np.nanmedian(corrs)),
    }


def spearman_per_gene_masked(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray | float]:
    num_genes = pred.shape[1]
    corrs = np.full(num_genes, np.nan, dtype=np.float64)
    for gene in range(num_genes):
        valid = mask[:, gene].astype(bool)
        if valid.sum() < 2:
            continue
        pred_col = pred[valid, gene]
        tgt_col = tgt[valid, gene]
        if pred_col.size < 2 or tgt_col.size < 2:
            continue
        pred_rank = _rankdata(pred_col)
        tgt_rank = _rankdata(tgt_col)
        corrs[gene] = _safe_corr(pred_rank, tgt_rank)
    return {
        "per_gene": corrs,
        "mean": float(np.nanmean(corrs)),
        "median": float(np.nanmedian(corrs)),
    }


def compute_gene_metrics_masked(
    pred: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    gene_names: Iterable[str],
) -> Dict[str, object]:
    """Compute per-gene Pearson and Spearman with a boolean mask."""
    gene_names = list(gene_names)
    pearson = pearson_per_gene_masked(pred, tgt, mask)["per_gene"]
    spearman = spearman_per_gene_masked(pred, tgt, mask)["per_gene"]
    return {
        "gene_names": gene_names,
        "pearson": pearson,
        "spearman": spearman,
        "pearson_mean": float(np.nanmean(pearson)),
        "pearson_median": float(np.nanmedian(pearson)),
        "spearman_mean": float(np.nanmean(spearman)),
        "spearman_median": float(np.nanmedian(spearman)),
    }


def compute_quartile_auroc(pred: np.ndarray, tgt: np.ndarray, quantile: float) -> Dict[str, object]:
    """Compute AUROC by comparing top/bottom quantiles."""
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for AUROC computation.")

    num_genes = pred.shape[1]
    aurocs = np.full(num_genes, np.nan, dtype=np.float64)

    q_low = np.quantile(tgt, quantile, axis=0)
    q_high = np.quantile(tgt, 1.0 - quantile, axis=0)

    labels = np.zeros_like(tgt, dtype=np.int8)
    labels[tgt >= q_high] = 1
    mask = (tgt <= q_low) | (tgt >= q_high)

    for g in range(num_genes):
        mask_g = mask[:, g]
        y_true = labels[mask_g, g]
        y_score = pred[mask_g, g]
        if y_true.size < 2 or len(np.unique(y_true)) < 2:
            continue
        aurocs[g] = roc_auc_score(y_true, y_score)

    return {
        "per_gene": aurocs,
        "mean": float(np.nanmean(aurocs)),
        "median": float(np.nanmedian(aurocs)),
        "quantile": quantile,
    }


def compute_gene_auroc(pred: np.ndarray, tgt: np.ndarray, percentile: float = 25) -> Dict[str, object]:
    """
    Compute gene-wise AUROC using top/bottom percentile splits.

    Args:
        pred: Array of shape (N, G) with model predictions.
        tgt: Array of shape (N, G) with ground-truth scores.
        percentile: Percentile threshold for defining positive/negative classes per gene.

    Returns:
        Dictionary containing per-gene AUROC values and their nan-aware mean/median.
    """
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for AUROC computation.")
    if pred.shape != tgt.shape:
        raise ValueError("pred and tgt must share the same shape.")
    if not 0.0 < float(percentile) < 50.0:
        raise ValueError("percentile must be in the range (0, 50).")

    pct = float(percentile)
    num_genes = pred.shape[1]
    aurocs = np.full(num_genes, np.nan, dtype=np.float64)

    lower = np.nanpercentile(tgt, pct, axis=0)
    upper = np.nanpercentile(tgt, 100.0 - pct, axis=0)

    labels = np.zeros_like(tgt, dtype=np.int8)
    labels[tgt >= upper] = 1

    valid_mask = (~np.isnan(pred)) & (~np.isnan(tgt))
    selection_mask = valid_mask & ((tgt <= lower) | (tgt >= upper))

    for gene in range(num_genes):
        mask_gene = selection_mask[:, gene]
        if mask_gene.sum() < 2:
            continue
        y_true = labels[mask_gene, gene]
        if np.unique(y_true).size < 2:
            continue
        y_score = pred[mask_gene, gene]
        try:
            aurocs[gene] = roc_auc_score(y_true, y_score)
        except ValueError:
            continue

    return {
        "gene_auroc": aurocs,
        "mean": float(np.nanmean(aurocs)),
        "median": float(np.nanmedian(aurocs)),
    }


def save_metrics_csv_json(metrics: Dict[str, object], save_dir: Path, prefix: str) -> None:
    """Persist per-gene metrics to CSV and JSON files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    gene_names = metrics["gene_names"]
    pearson = metrics["pearson"]
    spearman = metrics["spearman"]

    csv_path = save_dir / f"metrics_{prefix}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("gene_name,pearson,spearman\n")
        for name, p_val, s_val in zip(gene_names, pearson, spearman):
            f.write(f"{name},{p_val},{s_val}\n")

    summary = {
        "pearson_mean": metrics["pearson_mean"],
        "pearson_median": metrics["pearson_median"],
        "spearman_mean": metrics["spearman_mean"],
        "spearman_median": metrics["spearman_median"],
    }
    json_path = save_dir / f"metrics_{prefix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
