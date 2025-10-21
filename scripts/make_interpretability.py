"""Generate Grad-CAM overlays for selected genes on validation samples.

This script loads the front-door model checkpoint, prepares the validation split
from configs/causal.yaml, selects top-improved marker and non-marker genes, and
produces Grad-CAM heatmaps overlaid on the input patches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Lazy import of project modules (avoid albumentations dependency by not importing transforms)
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from data.dataset import GeneExpressionDataset  # type: ignore
from models.stnet import build_model_from_cfg  # type: ignore
from train.utils import load_config  # type: ignore


def load_eval_json(path: Path) -> Tuple[List[str], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["per_gene"]["gene_names"]
    spearman = np.array(data["per_gene"]["spearman"], dtype=float)
    return names, spearman


def load_markers_yaml(path: Path) -> List[str]:
    markers = yaml.safe_load(path.read_text(encoding="utf-8"))
    symbols: List[str] = []
    for _, genes in markers.items():
        for g in genes or []:
            if isinstance(g, str) and g.strip():
                symbols.append(g.strip().upper())
    return symbols


def load_hgnc_maps(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    sym2ens: Dict[str, str] = {}
    ens2sym: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_symbol = header.index("symbol")
        i_ensg = header.index("ensembl_gene_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_symbol, i_ensg):
                continue
            sym = parts[i_symbol].strip().upper()
            ensg = parts[i_ensg].strip()
            if sym and ensg and ensg.startswith("ENSG"):
                ensg = ensg.split(".")[0]
                sym2ens[sym] = ensg
                ens2sym.setdefault(ensg, parts[i_symbol].strip())
    return sym2ens, ens2sym


def select_top_genes(
    base_json: Path,
    fd_json: Path,
    marker_ensg: set[str],
    k_per_group: int = 3,
) -> Tuple[List[int], List[int], List[str]]:
    names, base_spear = load_eval_json(base_json)
    _, fd_spear = load_eval_json(fd_json)
    names_arr = np.array(names)
    delta = fd_spear - base_spear
    valid = ~np.isnan(delta)
    names_arr = names_arr[valid]
    delta = delta[valid]
    is_marker = np.array([n in marker_ensg for n in names_arr], dtype=bool)

    def top_idx(mask: np.ndarray, k: int) -> List[int]:
        if mask.sum() == 0:
            return []
        idx = np.argsort(-delta[mask])[:k]
        # Map back to full index positions
        full_idx = np.where(valid)[0][mask][idx]
        return full_idx.tolist()

    top_marker = top_idx(is_marker, k_per_group)
    top_non = top_idx(~is_marker, k_per_group)
    return top_marker, top_non, names


class GradCAM:
    def __init__(self, model) -> None:
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.conv_out: Optional[torch.Tensor] = None
        self.conv_grad: Optional[torch.Tensor] = None
        # Expect DenseNetBackbone with attribute `features`
        if not hasattr(model.backbone, "features"):
            raise RuntimeError("Backbone does not expose 'features' for Grad-CAM.")
        module = model.backbone.features
        self.handles.append(module.register_forward_hook(self._forward_hook))
        self.handles.append(module.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inp, out):
        self.conv_out = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.conv_grad = grad_out[0].detach()

    def compute_cam(self, score: torch.Tensor) -> np.ndarray:
        if self.conv_out is None:
            raise RuntimeError("conv_out is None; forward not run")
        if self.conv_grad is None:
            raise RuntimeError("conv_grad is None; backward not run")
        grads = self.conv_grad  # (B, C, H, W)
        feats = self.conv_out
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))  # (B,1,H,W)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def overlay_cam_on_image(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    # img: HxWx3 in [0,1], cam: HxW in [0,1]
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[:, :, :3]  # drop alpha
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    out = (1 - alpha) * img + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return out


def prepare_val_dataset(cfg: Dict) -> Tuple[GeneExpressionDataset, List[int]]:
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    dataset = GeneExpressionDataset(
        image_root=Path(paths["image_root"]),
        labels_csv=Path(paths["labels_csv"]),
        transform=None,
        use_dummy=train_cfg.get("use_dummy", False),
        num_genes=model_cfg.get("num_genes", 250),
        num_cellmix=model_cfg.get("num_cellmix", 0),
        m_student_csv=Path(paths.get("m_student")) if paths.get("m_student") else None,
        m_teacher_csv=Path(paths.get("m_teacher")) if paths.get("m_teacher") else None,
        image_template=paths.get("image_template", "{spot_id}.png"),
        spot_id_lower=paths.get("spot_id_lower", True),
    )
    # Recreate the training split
    train_frac = float(train_cfg.get("train_frac", 0.8))
    n_train = max(1, int(len(dataset) * train_frac))
    n_val = max(1, len(dataset) - n_train)
    if n_train + n_val != len(dataset):
        n_train = len(dataset) - n_val
    # Return the val indices (deterministic split as in training when no generator is specified)
    val_indices = list(range(n_train, n_train + n_val))
    return dataset, val_indices


def load_model(cfg: Dict, checkpoint: Path, device: torch.device):
    model_cfg = cfg.get("model", {}).copy()
    frontdoor_cfg = cfg.get("frontdoor", {}).copy()
    model = build_model_from_cfg(model_cfg, frontdoor_cfg=frontdoor_cfg)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM interpretability over validation samples")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/causal.yaml"))
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "outputs/frontdoor/checkpoints/epoch_015.ckpt"))
    parser.add_argument("--baseline", type=str, default=str(PROJECT_ROOT / "reports/gene_eval_baseline_val.json"))
    parser.add_argument("--frontdoor", type=str, default=str(PROJECT_ROOT / "reports/gene_eval_frontdoor_val.json"))
    parser.add_argument("--markers_yaml", type=str, default=str(PROJECT_ROOT.parents[0] / "Wu_etal_2021_BRCA_scRNASeq/configs/markers.yaml"))
    parser.add_argument("--hgnc", type=str, default=str(PROJECT_ROOT.parents[0] / "Wu_etal_2021_BRCA_scRNASeq/hgnc_complete_set.txt"))
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "reports/interpretability"))
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--k_per_group", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, Path(args.checkpoint), device)

    # Marker mapping and gene selection
    symbols = load_markers_yaml(Path(args.markers_yaml))
    sym2ens, ens2sym = load_hgnc_maps(Path(args.hgnc))
    marker_ensg = {sym2ens[s] for s in symbols if s in sym2ens}
    top_marker_idx, top_non_idx, gene_names = select_top_genes(
        Path(args.baseline), Path(args.frontdoor), marker_ensg, k_per_group=args.k_per_group
    )
    selected_gene_idx = top_marker_idx + top_non_idx
    selected_gene_ids = [gene_names[i] for i in selected_gene_idx]
    selected_gene_labels = [f"{ens2sym.get(g, g)}" for g in selected_gene_ids]

    # Dataset
    dataset, val_indices = prepare_val_dataset(cfg)
    # Pick first N val samples
    chosen = val_indices[: max(1, args.num_samples)]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grad-CAM helper
    cam_helper = GradCAM(model)

    grid_images: List[np.ndarray] = []
    for idx in chosen:
        sample = dataset[idx]
        img = sample["image"].numpy().transpose(1, 2, 0)  # to HWC
        img = np.clip(img, 0.0, 1.0)
        image_tensor = sample["image"].unsqueeze(0).to(device)
        m_student = sample["cellmix_student"].unsqueeze(0).to(device)

        row_imgs: List[np.ndarray] = []
        for gi, glabel in zip(selected_gene_idx, selected_gene_labels):
            model.zero_grad(set_to_none=True)
            pred, _extras = model(image_tensor, m_student=m_student)
            score = pred[0, gi]
            score.backward(retain_graph=True)
            cam = cam_helper.compute_cam(score)
            overlay = overlay_cam_on_image(img, cam, alpha=0.35)
            # Add title strip
            fig = plt.figure(figsize=(2.6, 2.6))
            plt.imshow(overlay)
            plt.axis('off')
            plt.title(glabel, fontsize=8)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3).astype(np.float32) / 255.0
            plt.close(fig)
            row_imgs.append(buf)

        # Concatenate row
        h_min = min(im.shape[0] for im in row_imgs)
        row_resized = [
            np.array(Image.fromarray((im * 255).astype(np.uint8)).resize(
                (int(im.shape[1] * h_min / im.shape[0]), h_min), Image.BILINEAR
            )).astype(np.float32) / 255.0
            for im in row_imgs
        ]
        row = np.concatenate(row_resized, axis=1)
        grid_images.append(row)

    # Stack rows into a grid
    max_w = max(im.shape[1] for im in grid_images)
    grid_padded = []
    for im in grid_images:
        if im.shape[1] < max_w:
            pad = np.ones((im.shape[0], max_w - im.shape[1], 3), dtype=np.float32)
            im = np.concatenate([im, pad], axis=1)
        grid_padded.append(im)
    grid = np.concatenate(grid_padded, axis=0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(out_dir / "sample_cam_grid.png")

    cam_helper.close()


if __name__ == "__main__":
    main()
