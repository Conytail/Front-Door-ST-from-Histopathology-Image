# WSI → ST Gene Expression with Front-door Mediation

This project predicts spatial transcriptomics (ST) gene expression (top‑250 genes) from H&E tissue patches. We leverage a causal front‑door design: image features X drive a mediator M (cell‑type composition) which in turn informs gene expression Y. The mediator combines a teacher distribution from cell2location (trained on scRNA‑seq of 26 donors) and a student regressor from the 23 donors used for training/validation.

Key validation result (n=250 genes): per‑gene Spearman median improves from 0.493 (baseline) to 0.570 (front‑door), Δ=+0.0585; improved genes 95.2% (238/250); one‑sided sign test p≈5.54×10^-56. See `reports/compare_val.json`, `reports/gene_eval_*.json`, and figures under `reports/`.

## Contents
- Models: `models/` (DenseNet121 backbone, front‑door fusion, linear+residual heads)
- Losses: `losses/` (masked Huber, correlation, cell‑mix loss)
- Data: `data/` (dataset and transforms)
- Training: `train/` (trainer, evaluator)
- Experiments: `scripts/` (training, evaluation, reporting)
- Configs: `configs/` (`default.yaml`, `causal.yaml`)
- Reports and artifacts: `reports/`, `outputs/`

## Installation
Requires Python 3.10+ and a CUDA‑capable GPU for best performance.

```bash
pip install -r requirements.txt
```

## Data & Paths
Configure paths in YAML:
- Images (224×224 patches): `configs/default.yaml: paths.data_root` or `configs/causal.yaml: paths.image_root`
- Labels (z‑scored top‑250 genes): `paths.labels_csv`
- Mediator cell‑mix:
  - Student: `configs/causal.yaml: paths.m_student` (CSV with 9 columns summing to 1)
  - Teacher: `configs/causal.yaml: paths.m_teacher` (CSV with 9 columns summing to 1)

Example inputs (adjust to your environment):
- Images: `D:/Desktop/WSI_out/Normalized_flat`
- Labels: `D:/Desktop/WSI_out/zscore_results/train_labels_zscore.csv`
- M_student: `D:/Desktop/M-Teacher/outputs/cellmix_student/M_student.csv`
- M_teacher: `D:/Desktop/M-Teacher/outputs/teacher/teacher_intersect_simplex.csv`

## Front‑door Model (high‑level)
Let X be image patches, Z=g_φ(X) DenseNet features, M the 9‑D cell‑type composition, and Y∈R^{250} gene scores. The predictor is

Ŷ = W_M·M_t + f_θ(FiLM(Z, M_t)), with teacher‑forcing schedule α_t:
α_t=1 (epochs ≤5), then linearly 1→0 (5<epochs≤15), α_t=0 afterwards; M_t = α_t·M_teacher + (1−α_t)·M_student.

Loss = masked Huber(Ŷ,Y;Ω) + λ·masked Corr(Ŷ,Y;Ω). Metrics: per‑gene Spearman and Pearson on validation spots.

## Quickstart
Baseline (no front‑door):
```bash
# Option A: dedicated baseline script (uses configs/default.yaml)
python scripts/train_baseline.py

# Option B: causal trainer with front-door disabled via CLI overrides
python scripts/train_causal.py --config configs/causal.yaml \
  --override frontdoor.enable=false --override model.use_frontdoor=false
```

Front‑door training:
```bash
python scripts/train_causal.py --config configs/causal.yaml
```

Evaluation and comparison:
```bash
# Compare two JSON metric reports and output figures+summary
python scripts/eval_gene.py \
  --baseline reports/gene_eval_baseline_val.json \
  --frontdoor reports/gene_eval_frontdoor_val.json \
  --out reports/compare_val.json \
  --fig_dir reports
```

Marker vs non‑marker grouped analysis and interpretability:
```bash
# Grouped ΔSpearman statistics + boxplot
python scripts/make_marker_group_analysis.py

# Grad-CAM overlays for selected genes on validation samples
python scripts/make_interpretability.py
```

Outputs:
- Summary: `reports/compare_val.json`
- Per‑gene metrics: `reports/gene_eval_baseline_val.json`, `reports/gene_eval_frontdoor_val.json`
- Figures: `reports/spearman_*.png`, `reports/marker_group_boxplot.png`, `reports/interpretability/sample_cam_grid.png`

## Configuration
Edit `configs/causal.yaml` to control data locations, model size, and front‑door schedule:
- `paths`: image root, labels CSV, `m_student`, `m_teacher`, image template
- `model`: `num_genes`, `num_cellmix`, `use_frontdoor`
- `frontdoor`: `enable`, `use_teacher_forcing`, `warmup_epochs`, `transition_epochs`, `fusion_mode` (FiLM/concat), `detach_cell_head`
- `train`: `batch_size`, `num_workers`, `seed`, epochs/print/save intervals, optimizer hyper‑parameters
- `augmentation`: external YAML (e.g., `configs/augment.yaml`)

Example CLI overrides (dot notation) with `train_causal.py`:
```bash
# Change batch size and warmup/transition
python scripts/train_causal.py --config configs/causal.yaml \
  --override train.batch_size=64 \
  --override frontdoor.warmup_epochs=3 \
  --override frontdoor.transition_epochs=7
```

## Training Details (reproduced)
- Optimizer: AdamW (lr=1.0e-3, weight decay=1.0e-4), batch size 32, 15 epochs
- LR schedule: none; early stopping: none; seed: 1337
- Backbone: DenseNet121 (no ImageNet pretraining), global pooling
- Input: 224×224 RGB, offline stain normalization + [0,1] scaling
- Augmentation: none (no rotation/flip/color jitter) for mediator consistency
- Hardware: single NVIDIA GeForce RTX 5060 (8 GB, CUDA 12.8)
- Runtime: ≈20–40 min for 15 epochs (train split ≈24.7k spots)
- Inference: ≈300–500 spots/s on the same GPU (batch=64; hardware dependent)

## Results (validation)
- Spearman (median): baseline 0.493 → front‑door 0.570 (Δ=+0.0585)
- Improved genes: 95.2% (238/250); one‑sided paired sign test p≈5.54×10^-56
- Error metrics (lower is better): masked Huber 0.334 → 0.287; masked corr‑loss 0.539 → 0.438
- Grouped ΔSpearman (median; 95% bootstrap CI):
  - Marker (n=60): +0.0776 [0.0650, 0.0890], p≈5.29×10^-17
  - Non‑marker (n=190): +0.0526 [0.0473, 0.0600], p≈1.47×10^-40

## Tips & Troubleshooting
- Ensure `m_student` and `m_teacher` CSVs align with `labels_csv` indices (case‑handling via `spot_id_lower`). Rows must sum ≈1.
- Missing images or mis‑matched IDs will raise errors in the dataset loader.
- To run on CPU (slow), set CUDA invisible; to change batch size, use `--override train.batch_size=...`.

## Acknowledgments
- ST‑net inspiration for WSI→gene prediction.
- cell2location for mediator estimation from scRNA‑seq.
- M‑Teacher pipeline for producing teacher and student cell‑mix distributions.

