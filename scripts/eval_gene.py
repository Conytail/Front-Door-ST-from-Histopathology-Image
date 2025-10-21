"""Gene expression evaluation script for baseline and front-door models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dataset import GeneExpressionDataset
from data.transforms import AlbumentationsTransformWithCoords
from losses.regression_losses import masked_correlation_loss, masked_huber
from models.stnet import build_model_from_cfg
from train.evaluator import compute_gene_auroc, compute_gene_metrics_masked
from train.utils import ensure_dir, load_config, set_seed


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if raw.startswith("0x"):
            return int(raw, 16)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be in key=value format, got: {override}")
        key, raw_value = override.split("=", 1)
        value = parse_override_value(raw_value)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value


def build_transform(cfg: Dict[str, Any]) -> AlbumentationsTransformWithCoords:
    return AlbumentationsTransformWithCoords(aug_cfg=cfg, enabled=False)


def collate_batch(batch: Iterable[Dict]) -> Dict[str, torch.Tensor | Dict[str, Iterable[str]]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    gene_mask = torch.stack([item["gene_mask"] for item in batch], dim=0)
    cellmix_student = torch.stack([item["cellmix_student"] for item in batch], dim=0)
    metas: Dict[str, Iterable[str]] = {"global_id": [item["meta"]["global_id"] for item in batch]}
    output: Dict[str, torch.Tensor | Dict[str, Iterable[str]]] = {
        "image": images,
        "label": labels,
        "gene_mask": gene_mask,
        "cellmix_student": cellmix_student,
        "meta": metas,
    }
    if all("cellmix_teacher" in item for item in batch):
        output["cellmix_teacher"] = torch.stack([item["cellmix_teacher"] for item in batch], dim=0)
    return output


def prepare_loader(
    cfg: Dict[str, Any],
    transform: AlbumentationsTransformWithCoords,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_teacher: bool,
) -> Tuple[DataLoader, GeneExpressionDataset]:
    paths_cfg = cfg["paths"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    dataset = GeneExpressionDataset(
        image_root=resolve_path(paths_cfg["image_root"]),
        labels_csv=resolve_path(paths_cfg["labels_csv"]),
        transform=transform,
        use_dummy=train_cfg.get("use_dummy", False),
        num_genes=model_cfg.get("num_genes", 250),
        num_cellmix=model_cfg.get("num_cellmix", 0),
        m_student_csv=resolve_path(paths_cfg.get("m_student")) if paths_cfg.get("m_student") else None,
        m_teacher_csv=resolve_path(paths_cfg.get("m_teacher")) if use_teacher and paths_cfg.get("m_teacher") else None,
        image_template=paths_cfg.get("image_template", "{spot_id}.png"),
        spot_id_lower=paths_cfg.get("spot_id_lower", True),
    )

    train_frac = train_cfg.get("train_frac", 0.8)
    generator = torch.Generator().manual_seed(seed)
    train_size = max(1, int(len(dataset) * train_frac))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size != len(dataset):
        train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    target_ds = train_ds if split == "train" else val_ds

    loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    return loader, dataset


def infer_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_frontdoor: bool,
    permute_m_in: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    masks_all: List[np.ndarray] = []
    meta_all: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            mask = batch["gene_mask"].to(device).bool()
            m_student = batch.get("cellmix_student")
            m_teacher = batch.get("cellmix_teacher")
            if m_student is not None:
                m_student = m_student.to(device)
                if permute_m_in:
                    perm = torch.randperm(m_student.size(0), device=device)
                    m_student = m_student[perm]
            if m_teacher is not None:
                m_teacher = m_teacher.to(device)

            if use_frontdoor:
                preds, _ = model(
                    images,
                    m_student=m_student,
                    m_teacher=m_teacher,
                    mode="student",
                    tau=0.0,
                    detach_cellmix=True,
                )
            else:
                preds, _ = model(images)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(labels.cpu().numpy())
            masks_all.append(mask.cpu().numpy())
            meta_all.extend(batch["meta"]["global_id"])

    preds_np = np.concatenate(preds_all, axis=0)
    targets_np = np.concatenate(targets_all, axis=0)
    masks_np = np.concatenate(masks_all, axis=0)
    return preds_np, targets_np, masks_np, meta_all


def summarize_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    gene_names: Iterable[str],
    auroc_percentile: float,
) -> Dict[str, object]:
    mask_bool = mask.astype(bool)
    per_gene = compute_gene_metrics_masked(preds, targets, mask_bool, gene_names)
    torch_preds = torch.from_numpy(preds)
    torch_targets = torch.from_numpy(targets)
    torch_mask = torch.from_numpy(mask_bool)
    huber = masked_huber(torch_preds, torch_targets, torch_mask).item()
    corr = masked_correlation_loss(torch_preds, torch_targets, torch_mask).item()
    metrics: Dict[str, object] = {
        "pearson_mean": float(per_gene["pearson_mean"]),
        "pearson_median": float(per_gene["pearson_median"]),
        "spearman_mean": float(per_gene["spearman_mean"]),
        "spearman_median": float(per_gene["spearman_median"]),
        "masked_huber": huber,
        "masked_corr": corr,
        "per_gene": {
            "gene_names": list(per_gene["gene_names"]),
            "pearson": per_gene["pearson"].tolist(),
            "spearman": per_gene["spearman"].tolist(),
        },
        "auroc_percentile": auroc_percentile,
    }
    try:
        auroc_stats = compute_gene_auroc(preds, targets, percentile=auroc_percentile)
        auroc_stats["gene_auroc"] = auroc_stats["gene_auroc"].tolist()
        metrics["auroc"] = auroc_stats
    except ImportError:
        metrics["auroc"] = None
    return metrics


def add_noise(images: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    noise = torch.empty_like(images).uniform_(-strength, strength)
    return torch.clamp(images + noise, 0.0, 1.0)


def compute_stability(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_frontdoor: bool,
    runs: int,
) -> Dict[str, float]:
    if runs <= 0:
        return {"runs": 0}

    preds_runs: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(runs):
            preds_all: List[np.ndarray] = []
            for batch in loader:
                images = add_noise(batch["image"].to(device))
                m_student = batch.get("cellmix_student")
                m_teacher = batch.get("cellmix_teacher")
                if m_student is not None:
                    m_student = m_student.to(device)
                if m_teacher is not None:
                    m_teacher = m_teacher.to(device)

                if use_frontdoor:
                    preds, _ = model(
                        images,
                        m_student=m_student,
                        m_teacher=m_teacher,
                        mode="student",
                        tau=0.0,
                        detach_cellmix=True,
                    )
                else:
                    preds, _ = model(images)
                preds_all.append(preds.cpu().numpy())
            preds_runs.append(np.concatenate(preds_all, axis=0))

    stacked = np.stack(preds_runs, axis=0)
    variance = np.var(stacked, axis=0)
    return {
        "runs": runs,
        "variance_mean": float(np.mean(variance)),
        "variance_median": float(np.median(variance)),
    }


def locate_checkpoint(paths_cfg: Dict[str, Any], run_tag: str, provided: str | None) -> Path:
    if provided:
        ckpt_path = Path(provided)
        return ckpt_path if ckpt_path.is_absolute() else resolve_path(provided)

    candidates: List[Path] = []
    base_outputs_dir = resolve_path(paths_cfg.get("outputs_dir", "outputs"))
    run_dir = base_outputs_dir / run_tag / "checkpoints"
    if run_dir.exists():
        candidates.extend(sorted(run_dir.glob("epoch_*.ckpt")))

    fallback_dir = resolve_path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
    if fallback_dir.exists():
        candidates.extend(sorted(fallback_dir.glob("epoch_*.ckpt")))

    if not candidates:
        raise FileNotFoundError("No checkpoints found; specify --ckpt explicitly.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gene prediction models")
    parser.add_argument("--run", choices=["baseline", "frontdoor"], required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "iid", "ood"])
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--stability", type=int, default=0)
    parser.add_argument("--permute_m_in", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--override", action="append", default=[], help="Override config values (key=value)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or ("configs/causal.yaml" if args.run == "frontdoor" else "configs/default.yaml")
    cfg = load_config(PROJECT_ROOT / config_path)
    apply_overrides(cfg, args.override)

    paths_cfg = cfg["paths"]
    set_seed(args.seed)

    augmentation_cfg = cfg.get("augmentation", {})
    if "config" in augmentation_cfg:
        aug_cfg = load_config(PROJECT_ROOT / augmentation_cfg["config"])
    else:
        aug_cfg = augmentation_cfg
    transform = build_transform(aug_cfg)

    loader, dataset = prepare_loader(
        cfg,
        transform,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_teacher=args.run == "frontdoor",
    )

    run_tag = "frontdoor" if args.run == "frontdoor" else "baseline"
    model_cfg = cfg.get("model", {}).copy()
    model_cfg["num_cellmix"] = dataset.num_cellmix
    model_cfg["num_genes"] = dataset.num_genes
    if args.run == "baseline":
        model_cfg["use_frontdoor"] = False
        frontdoor_cfg = None
    else:
        frontdoor_cfg = cfg.get("frontdoor", {}).copy()
        frontdoor_cfg["enable"] = True
        model_cfg["use_frontdoor"] = model_cfg.get("use_frontdoor", True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model_from_cfg(model_cfg, frontdoor_cfg=frontdoor_cfg)
    model.to(device)

    ckpt_path = locate_checkpoint(paths_cfg, run_tag, args.ckpt or None)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    preds_np, targets_np, mask_np, meta_ids = infer_loader(
        model,
        loader,
        device,
        use_frontdoor=args.run == "frontdoor",
        permute_m_in=False,
    )

    gene_names = list(pd.read_csv(resolve_path(paths_cfg["labels_csv"])).columns[1:])
    metrics = summarize_metrics(
        preds_np,
        targets_np,
        mask_np,
        gene_names,
        cfg.get("eval", {}).get("auroc_percentile", 25),
    )
    metrics["checkpoint"] = str(ckpt_path)
    metrics["split"] = args.split
    metrics["run"] = args.run
    metrics["num_samples"] = int(preds_np.shape[0])
    metrics["num_genes"] = int(preds_np.shape[1])

    if args.permute_m_in and args.run == "frontdoor":
        perm_preds, perm_targets, perm_mask, _ = infer_loader(
            model,
            loader,
            device,
            use_frontdoor=True,
            permute_m_in=True,
        )
        perm_metrics = summarize_metrics(
            perm_preds,
            perm_targets,
            perm_mask,
            gene_names,
            cfg.get("eval", {}).get("auroc_percentile", 25),
        )
        metrics["permute"] = {
            "spearman_median": perm_metrics["spearman_median"],
            "pearson_median": perm_metrics["pearson_median"],
        }

    if args.stability > 0:
        stability_stats = compute_stability(
            model,
            loader,
            device,
            use_frontdoor=args.run == "frontdoor",
            runs=args.stability,
        )
        metrics["stability"] = stability_stats

    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)
    prefix = f"gene_eval_{args.run}_{args.split}"
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"

    if args.save_json:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if args.save_csv:
        per_gene = metrics.get("per_gene", {})
        per_gene_df = pd.DataFrame(
            {
                "gene_name": per_gene.get("gene_names", []),
                "pearson": per_gene.get("pearson", []),
                "spearman": per_gene.get("spearman", []),
            }
        )
        per_gene_df.to_csv(csv_path, index=False)

    printable = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()



