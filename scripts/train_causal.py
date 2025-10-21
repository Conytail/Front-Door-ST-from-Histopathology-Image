"""Front-door aware training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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


def collate_fn(batch: Iterable[Dict]) -> Dict[str, torch.Tensor | Dict[str, Iterable[str]]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    gene_mask = torch.stack([item["gene_mask"] for item in batch], dim=0)
    cellmix_student = torch.stack([item["cellmix_student"] for item in batch], dim=0)
    metas: Dict[str, Iterable[str]] = {"global_id": [item["meta"]["global_id"] for item in batch]}
    batch_dict: Dict[str, torch.Tensor | Dict[str, Iterable[str]]] = {
        "image": images,
        "label": labels,
        "gene_mask": gene_mask,
        "cellmix_student": cellmix_student,
        "meta": metas,
    }
    if all("cellmix_teacher" in item for item in batch):
        batch_dict["cellmix_teacher"] = torch.stack([item["cellmix_teacher"] for item in batch], dim=0)
    return batch_dict


def prepare_datasets(
    dataset: GeneExpressionDataset,
    split_cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    train_frac = float(split_cfg.get("train_frac", 0.8))
    seed = int(split_cfg.get("seed", 1337))
    train_size = max(1, int(len(dataset) * train_frac))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size != len(dataset):
        train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    def make_loader(ds: torch.utils.data.Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=split_cfg.get("batch_size", 32),
            shuffle=shuffle,
            num_workers=split_cfg.get("num_workers", 0),
            collate_fn=collate_fn,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Front-door causal trainer")
    parser.add_argument("--config", type=str, default="configs/causal.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--override", action="append", default=[], help="Override config entries (key=value)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    apply_overrides(cfg, args.override)

    paths_cfg = cfg.get("paths", {})
    frontdoor_cfg = cfg.get("frontdoor", {}).copy()
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {}).copy()
    if not model_cfg:
        raise KeyError("model configuration is required in causal config.")

    frontdoor_enabled = bool(frontdoor_cfg.get("enable", True))
    frontdoor_cfg["enable"] = frontdoor_enabled
    cf_cfg = frontdoor_cfg.get("cf", {}) if frontdoor_enabled else cfg.get("cf", {})

    set_seed(train_cfg.get("seed", 1337))

    run_tag = "frontdoor" if frontdoor_enabled else "baseline"
    base_outputs_dir = resolve_path(paths_cfg.get("outputs_dir", "outputs"))
    outputs_dir = base_outputs_dir / run_tag
    checkpoints_root = outputs_dir / "checkpoints"
    preds_dir = outputs_dir / "preds"
    reports_dir = outputs_dir / "reports"
    artifacts_dir = outputs_dir / "artifacts"
    for directory in (outputs_dir, checkpoints_root, preds_dir, reports_dir, artifacts_dir):
        ensure_dir(directory)

    logger = setup_logger(outputs_dir, filename=f"train_{run_tag}.log")
    logger.info("Loaded config from %s", PROJECT_ROOT / args.config)
    logger.info("Run tag=%s frontdoor_enabled=%s", run_tag, frontdoor_enabled)

    augmentation_cfg = cfg.get("augmentation", {})
    if "config" in augmentation_cfg:
        aug_cfg = load_config(PROJECT_ROOT / augmentation_cfg["config"])
    else:
        aug_cfg = augmentation_cfg
    aug_enabled = bool(
        aug_cfg.get("geo", {}).get("enable", False)
        or aug_cfg.get("color", {}).get("enable", False)
    )
    transform = AlbumentationsTransformWithCoords(aug_cfg=aug_cfg, enabled=aug_enabled)

    dataset = GeneExpressionDataset(
        image_root=resolve_path(paths_cfg["image_root"]),
        labels_csv=resolve_path(paths_cfg["labels_csv"]),
        transform=transform,
        use_dummy=train_cfg.get("use_dummy", False),
        num_genes=model_cfg.get("num_genes", 250),
        num_cellmix=model_cfg.get("num_cellmix", 0),
        m_student_csv=resolve_path(paths_cfg.get("m_student")) if paths_cfg.get("m_student") else None,
        m_teacher_csv=resolve_path(paths_cfg.get("m_teacher")) if frontdoor_enabled and paths_cfg.get("m_teacher") else None,
        image_template=paths_cfg.get("image_template", "{spot_id}.png"),
        spot_id_lower=paths_cfg.get("spot_id_lower", True),
    )

    model_cfg["num_cellmix"] = dataset.num_cellmix
    model_cfg["num_genes"] = dataset.num_genes
    if not frontdoor_enabled:
        model_cfg["use_frontdoor"] = False
    else:
        model_cfg["use_frontdoor"] = model_cfg.get("use_frontdoor", True)

    split_cfg = {
        "batch_size": train_cfg.get("batch_size", 32),
        "num_workers": train_cfg.get("num_workers", 0),
        "seed": train_cfg.get("seed", 1337),
        "train_frac": train_cfg.get("train_frac", 0.8),
    }
    train_loader, val_loader = prepare_datasets(dataset, split_cfg)

    model = build_model_from_cfg(model_cfg, frontdoor_cfg=frontdoor_cfg if frontdoor_enabled else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("optim", {}).get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("optim", {}).get("weight_decay", 1e-4)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        cfg=TrainerConfig(
            max_steps=int(train_cfg.get("max_steps", 0)),
            print_freq=int(train_cfg.get("trainer", {}).get("print_freq", 1)),
            cf_cfg=cf_cfg,
            epochs=int(train_cfg.get("trainer", {}).get("epochs", 1)),
            save_interval=int(train_cfg.get("trainer", {}).get("save_interval_epochs", 5)),
            auroc_quantile=float(train_cfg.get("trainer", {}).get("auroc_quantile", 0.25)),
            eval_interval=int(train_cfg.get("trainer", {}).get("eval_interval", 1)),
            frontdoor_cfg=frontdoor_cfg,
            artifact_dir=artifacts_dir,
        ),
        loss_cfg=train_cfg.get("loss", {}),
        logger=logger,
    )

    if args.resume:
        resume_path = resolve_path(args.resume)
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state.get("optimizer_state", optimizer.state_dict()))
        logger.info("Resumed training from %s", resume_path)

    if args.fast_dev_run:
        train_loader = [next(iter(train_loader))]
        val_loader = [next(iter(val_loader))]

    def save_checkpoint(epoch_idx: int, metrics: Dict[str, float]) -> None:
        ckpt_path = checkpoints_root / f"epoch_{epoch_idx:03d}.ckpt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "epoch": epoch_idx,
            },
            ckpt_path,
        )
        logger.info("[CKPT] Saved checkpoint to %s", ckpt_path)

    trainer.fit(train_loader, val_loader, save_callback=save_checkpoint)


if __name__ == "__main__":
    main()
