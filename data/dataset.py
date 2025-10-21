"""
Dataset utilities for WSI -> ST gene expression project.
TODO: Replace dummy generation with real I/O pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import AlbumentationsTransformWithCoords


@dataclass
class SampleRecord:
    """Container for per-spot metadata."""

    global_id: str
    image_path: Path
    label: torch.Tensor
    gene_mask: torch.Tensor
    cellmix_student: torch.Tensor
    cellmix_teacher: Optional[torch.Tensor] = None


class GeneExpressionDataset(Dataset):
    """Torch dataset that can generate dummy samples for smoke tests."""

    def __init__(
        self,
        image_root: Path,
        labels_csv: Path,
        transform: AlbumentationsTransformWithCoords | None = None,
        use_dummy: bool = True,
        num_patients: int = 2,
        spots_per_patient: int = 16,
        num_genes: int = 250,
        num_cellmix: int = 6,
        m_student_csv: Optional[Path] = None,
        m_teacher_csv: Optional[Path] = None,
        image_template: str = "{spot_id}.png",
        spot_id_lower: bool = True,
    ) -> None:
        self.image_root = image_root
        self.labels_csv = labels_csv
        self.transform = transform
        self.use_dummy = use_dummy
        self.num_patients = num_patients
        self.spots_per_patient = spots_per_patient
        self.num_genes = num_genes
        self.num_cellmix = num_cellmix
        self.m_student_csv = m_student_csv
        self.m_teacher_csv = m_teacher_csv
        self.image_template = image_template
        self.spot_id_lower = spot_id_lower
        self._id_lookup: Dict[str, str] = {}

        if self.use_dummy:
            self.records = self._make_dummy_records()
        else:
            self.records = self._load_real_records()

    def _make_dummy_records(self) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        rng = np.random.default_rng(seed=42)
        for pid in range(self.num_patients):
            patient_id = f"DUMMY{pid:02d}"
            for spot in range(self.spots_per_patient):
                global_id = f"{patient_id}_S{spot:02d}"
                label = torch.from_numpy(
                    rng.normal(loc=0.0, scale=1.0, size=self.num_genes).astype(np.float32)
                )
                mask = torch.ones(self.num_genes, dtype=torch.bool)
                weights = rng.random(self.num_cellmix).astype(np.float32)
                weights = weights / np.clip(weights.sum(), a_min=1e-6, a_max=None)
                cellmix_student = torch.from_numpy(weights.astype(np.float32))
                dummy_image = self._make_dummy_image(rng)
                image_path = self._save_dummy_image(global_id, dummy_image)
                records.append(
                    SampleRecord(
                        global_id=global_id,
                        image_path=image_path,
                        label=label,
                        gene_mask=mask,
                        cellmix_student=cellmix_student,
                        cellmix_teacher=None,
                    )
                )
        return records

    def _make_dummy_image(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)

    def _save_dummy_image(self, global_id: str, array: np.ndarray) -> Path:
        target_dir = self.image_root / "dummy"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{global_id}.png"
        Image.fromarray(array).save(path)
        return path

    def _load_real_records(self) -> List[SampleRecord]:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv missing: {self.labels_csv}")
        if self.m_student_csv is None or not self.m_student_csv.exists():
            raise FileNotFoundError("m_student_csv is required for real dataset loading.")
        df = pd.read_csv(self.labels_csv, index_col=0)
        student_df = pd.read_csv(self.m_student_csv, index_col=0)
        self.num_cellmix = student_df.shape[1]
        self.num_genes = df.shape[1]
        df.index = df.index.astype(str)
        student_df.index = student_df.index.astype(str)
        original_index = df.index.to_list()

        if self.spot_id_lower:
            lowered = df.index.str.lower()
            df.index = lowered
            student_df.index = student_df.index.str.lower()
            self._id_lookup = dict(zip(df.index, original_index))
        else:
            self._id_lookup = {idx: idx for idx in df.index}
        teacher_df = None
        if self.m_teacher_csv is not None and self.m_teacher_csv.exists():
            teacher_df = pd.read_csv(self.m_teacher_csv, index_col=0)
            teacher_df.index = teacher_df.index.astype(str)
            if self.spot_id_lower:
                teacher_df.index = teacher_df.index.str.lower()

        missing_students = sorted(set(df.index) - set(student_df.index))
        if missing_students:
            raise KeyError(f"M_student missing spots: {missing_students[:5]}{'...' if len(missing_students) > 5 else ''}")

        records: List[SampleRecord] = []
        for global_id, row in df.iterrows():
            original_id = self._id_lookup.get(global_id, global_id)
            image_path = self.image_root / self.image_template.format(spot_id=original_id)
            if not image_path.exists():
                raise FileNotFoundError(f"Image missing for {global_id}: {image_path}")
            label_np = row.values.astype(np.float32)
            mask_np = ~np.isnan(label_np)
            label_np = np.nan_to_num(label_np, nan=0.0).astype(np.float32)

            m_student_row = student_df.loc[global_id].values.astype(np.float32)
            row_sum = float(m_student_row.sum())
            if abs(row_sum - 1.0) > 1e-3:
                raise ValueError(f"M_student row for {global_id} not simplex (sum={row_sum:.4f})")

            teacher_tensor = None
            if teacher_df is not None and global_id in teacher_df.index:
                m_teacher_row = teacher_df.loc[global_id].values.astype(np.float32)
                teacher_sum = float(m_teacher_row.sum())
                if abs(teacher_sum - 1.0) > 1e-3:
                    raise ValueError(f"M_teacher row for {global_id} not simplex (sum={teacher_sum:.4f})")
                teacher_tensor = torch.from_numpy(m_teacher_row)

            records.append(
                SampleRecord(
                    global_id=original_id,
                    image_path=image_path,
                    label=torch.from_numpy(label_np),
                    gene_mask=torch.from_numpy(mask_np.astype(np.uint8)).bool(),
                    cellmix_student=torch.from_numpy(m_student_row),
                    cellmix_teacher=teacher_tensor,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | Dict[str, str]]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_np = np.array(image)
        meta: Dict[str, str] = {"global_id": record.global_id}

        if self.transform is not None:
            tensor, meta = self.transform(image_np, meta=meta)
        else:
            tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        sample: Dict[str, torch.Tensor | Dict[str, str]] = {
            "image": tensor,
            "label": record.label,
            "gene_mask": record.gene_mask,
            "cellmix": record.cellmix_student,
            "cellmix_student": record.cellmix_student,
            "meta": meta,
        }
        if record.cellmix_teacher is not None:
            sample["cellmix_teacher"] = record.cellmix_teacher
        return sample

