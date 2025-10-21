"""Baseline training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dataset import GeneExpressionDataset
from data.transforms import AlbumentationsTransformWithCoords
from models.stnet import build_model_from_cfg
from train.trainer import Trainer, TrainerConfig
from train.utils import ensure_dir, load_config, set_seed, setup_logger


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collate_fn(batch: Iterable[Dict]) -> Dict[str, torch.Tensor | Dict[str, Iterable[str]]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    metas: Dict[str, Iterable[str]] = {"global_id": [item["meta"]["global_id"] for item in batch]}
    return {"image": images, "label": labels, "meta": metas}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline trainer")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dry_run", type=str, default="true")
    return parser.parse_args()


def compute_losses(
    preds: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    from losses.regression_losses import correlation_loss, huber_loss

    huber = huber_loss(preds, target, delta=float(loss_cfg.get("huber_delta", 1.0)))
    corr = correlation_loss(preds, target)
    huber_w = float(loss_cfg.get("gene_huber_weight", loss_cfg.get("huber_weight", 1.0)))
    corr_w = float(loss_cfg.get("gene_corr_weight", loss_cfg.get("corr_weight", 0.0)))
    total = huber_w * huber + corr_w * corr
    return {"total": total, "huber": huber, "corr": corr}


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    set_seed(train_cfg["seed"])

    outputs_dir = resolve_path(paths_cfg.get("outputs_dir", "outputs"))
    ensure_dir(outputs_dir)
    checkpoints_root = resolve_path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
    ensure_dir(checkpoints_root)
    rel_save = Path(train_cfg["trainer"].get("save_dir", ""))
    save_dir = rel_save if rel_save.is_absolute() else (checkpoints_root / rel_save if rel_save else checkpoints_root)
    ensure_dir(save_dir)
    logger = setup_logger(outputs_dir, filename="train_baseline.log")

    aug_cfg = load_config(PROJECT_ROOT / cfg["augmentation"]["config"])
    aug_enabled = bool(
        aug_cfg.get("geo", {}).get("enable", False)
        or aug_cfg.get("color", {}).get("enable", False)
    )
    transform = AlbumentationsTransformWithCoords(aug_cfg=aug_cfg, enabled=aug_enabled)

    dataset = GeneExpressionDataset(
        image_root=resolve_path(paths_cfg["data_root"]),
        labels_csv=resolve_path(paths_cfg["labels_csv"]),
        transform=transform,
        use_dummy=args.dry_run.lower() == "true",
        num_patients=2,
        spots_per_patient=16,
        num_genes=model_cfg["num_genes"],
        num_cellmix=model_cfg.get("num_cellmix", 0),
    )

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size != len(dataset):
        train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)

    model = build_model_from_cfg(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        cfg=TrainerConfig(
            max_steps=int(train_cfg.get("max_steps", 0)),
            print_freq=int(train_cfg["trainer"].get("print_freq", 1)),
            cf_cfg={"enable": False},
            epochs=int(train_cfg["trainer"].get("epochs", 1)),
            save_interval=int(train_cfg["trainer"].get("save_interval_epochs", 5)),
            auroc_quantile=float(train_cfg["trainer"].get("auroc_quantile", 0.25)),
        ),
        loss_cfg=train_cfg["loss"],
        logger=logger,
    )

    def save_checkpoint(epoch_idx: int, metrics: Dict[str, float]) -> None:
        ckpt_path = save_dir / f"epoch_{epoch_idx:03d}.ckpt"
        torch.save({"model_state": model.state_dict(), "metrics": metrics}, ckpt_path)
        logger.info("[CKPT] Saved checkpoint to %s", ckpt_path)

    trainer.fit(train_loader, val_loader, save_callback=save_checkpoint)


if __name__ == "__main__":
    main()
