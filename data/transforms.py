"""Transformation utilities using Albumentations.
TODO: Extend with keypoint handling when needed.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandStainJitter(ImageOnlyTransform):
    """Lightweight RandStainJitter implementation with configurable modes."""

    def __init__(
        self,
        mode: str = "rgb_affine",
        rsj_cfg: dict | None = None,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.always_apply = always_apply
        cfg = rsj_cfg or {}
        self.mode = (cfg.get("mode", mode) or "rgb_affine").lower()
        self.alpha_range = cfg.get("alpha_range", [0.9, 1.1])
        self.beta_range = cfg.get("beta_range", [-0.05, 0.05])
        self.lab_delta_L = float(cfg.get("lab_delta_L", 0.08))
        self.lab_delta_ab = float(cfg.get("lab_delta_ab", 0.04))
        self.mix_strength = float(cfg.get("mix_strength", 0.05))
        if self.mode == "lab" and not hasattr(cv2, "cvtColor"):
            warnings.warn("cv2 not available; falling back to rgb_affine mode.")
            self.mode = "rgb_affine"

    def apply(self, img: np.ndarray, **params: Dict) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("RandStainJitter expects a HxWx3 image.")
        img_float = img.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0

        if self.mode == "lab":
            transformed = self._apply_lab(img_float)
        elif self.mode == "mixrgb":
            transformed = self._apply_mixrgb(img_float)
        else:  # default to rgb_affine
            transformed = self._apply_rgb_affine(img_float)

        transformed = np.clip(transformed, 0.0, 1.0).astype(np.float32)
        return (transformed * 255.0).astype(np.float32)

    def _apply_rgb_affine(self, img: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1], size=(3,),).astype(np.float32)
        beta = np.random.uniform(self.beta_range[0], self.beta_range[1], size=(3,),).astype(np.float32)
        return img * alpha + beta

    def _apply_lab(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        l_delta = np.random.uniform(-self.lab_delta_L, self.lab_delta_L) * 255.0
        ab_delta = np.random.uniform(-self.lab_delta_ab, self.lab_delta_ab, size=(2,)) * 255.0
        lab[..., 0] = np.clip(lab[..., 0] + l_delta, 0.0, 255.0)
        lab[..., 1] = np.clip(lab[..., 1] + ab_delta[0], 0.0, 255.0)
        lab[..., 2] = np.clip(lab[..., 2] + ab_delta[1], 0.0, 255.0)
        rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        return rgb / 255.0

    def _apply_mixrgb(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(-self.mix_strength, self.mix_strength, size=(3, 3)).astype(np.float32)
        matrix = np.eye(3, dtype=np.float32) + noise
        return np.tensordot(img, matrix.T, axes=1)


class AlbumentationsTransformWithCoords:
    """Albumentations-based transform wrapper with configurable pipeline."""

    def __init__(self, aug_cfg: dict | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.aug_cfg = aug_cfg or {}
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> A.BasicTransform | None:
        geo_cfg = self.aug_cfg.get("geo", {})
        color_cfg = self.aug_cfg.get("color", {})
        size_cfg = geo_cfg.get("rrc_size", [224, 224])
        height, width = int(size_cfg[0]), int(size_cfg[1])

        if not self.enabled or not self.aug_cfg:
            print("[Aug] Disabled – applying resize only.")
            return A.Compose([A.Resize(height=height, width=width)])

        transforms: list[A.BasicTransform] = []

        if geo_cfg.get("enable", False):
            if geo_cfg.get("random_resized_crop", False):
                scale = tuple(geo_cfg.get("rrc_scale", [0.8, 1.0]))
                transforms.append(
                    A.RandomResizedCrop(
                        size=(height, width),
                        scale=scale,
                        ratio=(1.0, 1.0),
                    )
                )
            else:
                transforms.append(A.Resize(height=height, width=width))

            if geo_cfg.get("hflip", False):
                transforms.append(A.HorizontalFlip(p=0.5))
            if geo_cfg.get("vflip", False):
                transforms.append(A.VerticalFlip(p=0.5))

            rotate_limit = int(geo_cfg.get("rotate_limit", 0))
            if rotate_limit:
                transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=rotate_limit,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    )
                )
        else:
            transforms.append(A.Resize(height=height, width=width))

        if color_cfg.get("enable", False):
            if color_cfg.get("use_rand_stain_jitter", False):
                rsj_cfg = color_cfg.get("rsj", {})
                transforms.append(
                    RandStainJitter(
                        mode=rsj_cfg.get("mode", "rgb_affine"),
                        rsj_cfg=rsj_cfg,
                        p=float(rsj_cfg.get("p", 0.5)),
                    )
                )
                print(
                    "[Aug] RandStainJitter enabled: mode={} p={} params={}".format(
                        rsj_cfg.get("mode", "rgb_affine"),
                        rsj_cfg.get("p", 0.5),
                        rsj_cfg,
                    )
                )
            else:
                cj_prob = float(color_cfg.get("colorjitter_p", 1.0))
                transforms.append(
                    A.ColorJitter(
                        brightness=float(color_cfg.get("brightness", 0.0)),
                        contrast=float(color_cfg.get("contrast", 0.0)),
                        saturation=float(color_cfg.get("saturation", 0.0)),
                        hue=float(color_cfg.get("hue", 0.0)),
                        p=cj_prob,
                    )
                )
                print(
                    "[Aug] RSJ disabled → using ColorJitter(p={} brightness={} contrast={} saturation={} hue={})".format(
                        cj_prob,
                        color_cfg.get("brightness", 0.0),
                        color_cfg.get("contrast", 0.0),
                        color_cfg.get("saturation", 0.0),
                        color_cfg.get("hue", 0.0),
                    )
                )
        else:
            print("[Aug] Color augmentation disabled.")

        return A.Compose(transforms)

    def __call__(
        self,
        image: np.ndarray,
        keypoints: None = None,
        meta: Dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if self.pipeline is None:
            processed = image
        else:
            processed = self.pipeline(image=image)["image"]
        if processed.dtype != np.float32:
            processed = processed.astype(np.float32)
        if processed.max() > 1.0:
            processed /= 255.0
        tensor = torch.from_numpy(processed).permute(2, 0, 1).contiguous()
        meta_out: Dict = meta.copy() if meta is not None else {}
        return tensor, meta_out
