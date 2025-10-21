"""Inference script to export predictions to CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.dataset import GeneExpressionDataset
from data.transforms import AlbumentationsTransformWithCoords
from models.stnet import build_model_from_cfg
from train.utils import load_config, set_seed


def collate_fn(batch: Iterable[Dict]) -> Dict[str, torch.Tensor | Dict[str, Iterable[str]]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    metas: Dict[str, Iterable[str]] = {"global_id": [item["meta"]["global_id"] for item in batch]}
    return {"image": images, "meta": metas}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_dataloader(
    dataset: GeneExpressionDataset,
    split: str,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if split == "all":
        subset = dataset
    else:
        train_size = max(1, int(len(dataset) * 0.8))
        val_size = max(1, len(dataset) - train_size)
        if train_size + val_size != len(dataset):
            train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        subset = train_ds if split == "train" else val_ds
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


def save_predictions(
    ids: list[str],
    preds: np.ndarray,
    gene_names: list[str],
    save_dir: Path,
    split: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(preds, columns=gene_names)
    df.insert(0, "global_id", ids)
    df.to_csv(save_dir / f"preds_{split}.csv", index=False)


def save_cellmix(
    ids: list[str],
    cellmix: np.ndarray,
    save_dir: Path,
    split: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"cellmix_{i}" for i in range(cellmix.shape[1])]
    df = pd.DataFrame(cellmix, columns=columns)
    df.insert(0, "global_id", ids)
    df.to_csv(save_dir / f"cellmix_{split}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference and export predictions")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--with_cellmix", type=str, default="false")
    parser.add_argument("--use_dummy", type=str, default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    paths_cfg = cfg["paths"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    set_seed(train_cfg["seed"])

    labels_csv = resolve_path(paths_cfg["labels_csv"])
    labels_df = pd.read_csv(labels_csv)
    gene_names = labels_df.columns.tolist()[1:]

    aug_cfg = load_config(PROJECT_ROOT / cfg["augmentation"]["config"])
    aug_enabled = bool(
        aug_cfg.get("geo", {}).get("enable", False)
        or aug_cfg.get("color", {}).get("enable", False)
    )
    transform = AlbumentationsTransformWithCoords(aug_cfg=aug_cfg, enabled=aug_enabled)

    dataset = GeneExpressionDataset(
        image_root=resolve_path(paths_cfg["data_root"]),
        labels_csv=labels_csv,
        transform=transform,
        use_dummy=args.use_dummy.lower() == "true",
        num_genes=model_cfg["num_genes"],
        num_cellmix=model_cfg.get("num_cellmix", 0),
    )

    loader = prepare_dataloader(
        dataset,
        split=args.split,
        seed=train_cfg["seed"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    model = build_model_from_cfg(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt_dir = resolve_path(paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
    ckpt_path = resolve_path(args.ckpt) if args.ckpt else ckpt_dir / "baseline.ckpt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    save_dir = resolve_path(args.save_dir or paths_cfg["preds_dir"])

    ids: list[str] = []
    gene_preds: list[np.ndarray] = []
    cellmix_preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            preds, extras = model(images)
            ids.extend(batch["meta"]["global_id"])
            gene_preds.append(preds.cpu().numpy())
            if args.with_cellmix.lower() == "true" and model.use_frontdoor and "cellmix" in extras:
                cellmix_preds.append(extras["cellmix"].cpu().numpy())

    gene_array = np.concatenate(gene_preds, axis=0) if gene_preds else np.empty((0, len(gene_names)))
    save_predictions(ids, gene_array, gene_names, save_dir, args.split)

    if cellmix_preds:
        cellmix_array = np.concatenate(cellmix_preds, axis=0)
        save_cellmix(ids, cellmix_array, save_dir, args.split)
    elif args.with_cellmix.lower() == "true" and model.use_frontdoor:
        print("[WARN] CellMix head enabled but no predictions collected.")

    print(f"[INFER] Saved predictions to {save_dir}")


if __name__ == "__main__":
    main()
