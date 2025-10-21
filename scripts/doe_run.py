"""DOE full-factorial batch runner for spatial transcriptomics experiments."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from train.evaluator import compute_gene_auroc  # noqa: E402

FACTORS: Dict[str, Iterable[object]] = {
    "geo": (False, True),
    "rsj": (False, True),
    "cf": (False, True),
    "corr_weight": (0.25, 0.75),
}


def parse_args() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="DOE full-factorial experiment runner")
    parser.add_argument("--default_cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--augment_cfg", type=str, default="configs/augment.yaml")
    parser.add_argument("--causal_cfg", type=str, default="configs/causal.yaml")
    parser.add_argument("--out_root", type=str, default="outputs/doe")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--auroc_percentile", type=float, default=25.0)
    parser.add_argument("--dry_run", type=str, default="false")
    return parser


def load_yaml(path: Path) -> Dict:
    """Load a YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(data: Dict, path: Path) -> None:
    """Write a dictionary to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def as_project_relative(path: Path) -> str:
    """Return path as project-relative string when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_subprocess(cmd: List[str], logger: logging.Logger) -> None:
    """Execute a subprocess command, raising on failure."""
    logger.info("Executing command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def generate_combinations() -> Iterable[Dict[str, object]]:
    """Yield every factorial combination of the configured factors."""
    keys = list(FACTORS.keys())
    for combo in itertools.product(*(FACTORS[key] for key in keys)):
        yield dict(zip(keys, combo))


def build_run_id(combo: Dict[str, object]) -> str:
    """Generate a stable run identifier from the factor levels."""
    geo = int(bool(combo["geo"]))
    rsj = int(bool(combo["rsj"]))
    cf = int(bool(combo["cf"]))
    corr_weight = int(round(float(combo["corr_weight"]) * 100))
    return f"geo{geo}_rsj{rsj}_cf{cf}_cw{corr_weight:03d}"


def prepare_configs(
    combo: Dict[str, object],
    base_default: Dict,
    base_augment: Dict,
    base_causal: Dict,
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[Path, Path, Path, Dict]:
    """Create per-run config snapshots and return their paths with the mutated default dict."""
    default_cfg = copy.deepcopy(base_default)
    augment_cfg = copy.deepcopy(base_augment)
    causal_cfg = copy.deepcopy(base_causal)

    run_outputs = run_dir
    checkpoints_dir = run_dir / "checkpoints"
    preds_dir = run_dir / "preds"
    reports_dir = run_dir / "reports"
    for directory in (run_dir, checkpoints_dir, preds_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths_cfg = default_cfg.get("paths", {})
    paths_cfg["outputs_dir"] = as_project_relative(run_outputs)
    paths_cfg["checkpoints_dir"] = as_project_relative(checkpoints_dir)
    paths_cfg["preds_dir"] = as_project_relative(preds_dir)
    paths_cfg["reports_dir"] = as_project_relative(reports_dir)
    default_cfg["paths"] = paths_cfg

    default_cfg.setdefault("model", {})
    default_cfg["model"]["use_frontdoor"] = False

    loss_cfg = default_cfg.setdefault("train", {}).setdefault("loss", {})
    loss_cfg["gene_corr_weight"] = float(combo["corr_weight"])
    default_cfg["train"]["loss"] = loss_cfg

    trainer_cfg = default_cfg["train"].setdefault("trainer", {})
    trainer_cfg["epochs"] = int(args.epochs)
    trainer_cfg["save_interval_epochs"] = int(args.save_interval)
    trainer_cfg["eval_interval"] = int(args.eval_interval)
    trainer_cfg.setdefault("print_freq", 1)
    default_cfg["train"]["trainer"] = trainer_cfg

    eval_cfg = default_cfg.setdefault("eval", {})
    eval_cfg["split"] = args.split
    eval_cfg["auroc_percentile"] = float(args.auroc_percentile)
    default_cfg["eval"] = eval_cfg

    augment_cfg.setdefault("geo", {})
    augment_cfg["geo"]["enable"] = bool(combo["geo"])

    color_cfg = augment_cfg.setdefault("color", {})
    if combo["rsj"]:
        color_cfg["enable"] = True
        color_cfg["use_rand_stain_jitter"] = True
    else:
        color_cfg["enable"] = False
        color_cfg["use_rand_stain_jitter"] = False
    augment_cfg["color"] = color_cfg

    causal_cfg.setdefault("cf", {})
    causal_cfg["cf"]["enable"] = bool(combo["cf"])

    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    default_cfg_path = config_dir / "default.yaml"
    augment_cfg_path = config_dir / "augment.yaml"
    causal_cfg_path = config_dir / "causal.yaml"

    default_cfg.setdefault("augmentation", {})
    default_cfg["augmentation"]["config"] = as_project_relative(augment_cfg_path)

    default_cfg.setdefault("causal", {})
    default_cfg["causal"]["config"] = as_project_relative(causal_cfg_path)

    dump_yaml(default_cfg, default_cfg_path)
    dump_yaml(augment_cfg, augment_cfg_path)
    dump_yaml(causal_cfg, causal_cfg_path)

    return default_cfg_path, augment_cfg_path, causal_cfg_path, default_cfg


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the most recent checkpoint path."""
    ckpt_files = sorted(checkpoints_dir.glob("epoch_*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return ckpt_files[-1]


def extract_metrics(
    preds_csv: Path,
    default_cfg: Dict,
    reports_dir: Path,
    split: str,
    auroc_percentile: float,
) -> Tuple[float, float, float, float, float, float]:
    """Compute summary correlation and AUROC metrics for a run."""
    summary_candidates = sorted(
        reports_dir.glob(f"*_{split}/metrics_{split}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not summary_candidates:
        raise FileNotFoundError(f"No metrics_{split}.json found in {reports_dir}")
    metrics_path = summary_candidates[0]
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics_json = json.load(handle)

    pearson_mean = float(metrics_json["pearson_mean"])
    pearson_median = float(metrics_json["pearson_median"])
    spearman_mean = float(metrics_json["spearman_mean"])
    spearman_median = float(metrics_json["spearman_median"])

    preds_df = pd.read_csv(preds_csv)
    preds_df = preds_df.set_index(preds_df.columns[0])

    labels_path = PROJECT_ROOT / Path(default_cfg["paths"]["labels_csv"])
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.set_index(labels_df.columns[0])

    gene_columns = [col for col in preds_df.columns if col in labels_df.columns]
    if not gene_columns:
        raise ValueError("No overlapping gene columns between predictions and labels.")

    common_ids = preds_df.index.intersection(labels_df.index)
    if common_ids.empty:
        raise ValueError("No overlapping sample identifiers between predictions and labels.")

    preds_aligned = preds_df.loc[common_ids, gene_columns]
    labels_aligned = labels_df.loc[common_ids, gene_columns]

    auroc_stats = compute_gene_auroc(
        preds_aligned.to_numpy(),
        labels_aligned.to_numpy(),
        percentile=auroc_percentile,
    )

    return (
        pearson_mean,
        pearson_median,
        spearman_mean,
        spearman_median,
        float(auroc_stats["mean"]),
        float(auroc_stats["median"]),
    )


def append_summary_row(summary_path: Path, row: Dict[str, object]) -> None:
    """Append a row to the global summary CSV, creating it if needed."""
    header = [
        "geo",
        "rsj",
        "cf",
        "corr_weight",
        "epochs",
        "ckpt_path",
        "pearson_mean",
        "pearson_median",
        "spearman_mean",
        "spearman_median",
        "auroc_mean",
        "auroc_median",
    ]
    file_exists = summary_path.exists()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_failure(out_root: Path, run_id: str, error: Exception) -> None:
    """Persist failure information for a run."""
    log_path = out_root / "failed_runs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{run_id}: {error}\n")


def main() -> None:
    args = parse_args().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("doe")

    out_root = PROJECT_ROOT / Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_default = load_yaml(PROJECT_ROOT / Path(args.default_cfg))
    base_augment = load_yaml(PROJECT_ROOT / Path(args.augment_cfg))
    base_causal = load_yaml(PROJECT_ROOT / Path(args.causal_cfg))

    summary_path = out_root / "summary.csv"

    for combo in generate_combinations():
        run_id = build_run_id(combo)
        logger.info("Starting run %s with factors %s", run_id, combo)
        run_dir = out_root / run_id

        try:
            default_cfg_path, _, causal_cfg_path, default_cfg = prepare_configs(
                combo,
                base_default,
                base_augment,
                base_causal,
                args,
                run_dir,
            )

            train_cmd = [
                sys.executable,
                "scripts/train_experiment.py",
                "--default_cfg",
                as_project_relative(default_cfg_path),
                "--causal_cfg",
                as_project_relative(causal_cfg_path),
                "--dry_run",
                args.dry_run,
            ]
            run_subprocess(train_cmd, logger)

            checkpoints_dir = run_dir / "checkpoints"
            latest_ckpt = find_latest_checkpoint(checkpoints_dir)

            preds_dir = run_dir / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            preds_csv = preds_dir / f"preds_{args.split}.csv"
            infer_cmd = [
                sys.executable,
                "scripts/inference.py",
                "--config",
                as_project_relative(default_cfg_path),
                "--ckpt",
                str(latest_ckpt),
                "--split",
                args.split,
                "--save_dir",
                as_project_relative(preds_dir),
                "--with_cellmix",
                "false",
                "--use_dummy",
                "false",
            ]
            run_subprocess(infer_cmd, logger)

            reports_dir = run_dir / "reports"
            report_cmd = [
                sys.executable,
                "scripts/report.py",
                "--config",
                as_project_relative(default_cfg_path),
                "--split",
                args.split,
                "--preds_csv",
                as_project_relative(preds_csv),
                "--out_dir",
                as_project_relative(reports_dir),
            ]
            run_subprocess(report_cmd, logger)

            metrics = extract_metrics(
                preds_csv,
                default_cfg,
                reports_dir,
                args.split,
                args.auroc_percentile,
            )

            row = {
                "geo": int(bool(combo["geo"])),
                "rsj": int(bool(combo["rsj"])),
                "cf": int(bool(combo["cf"])),
                "corr_weight": float(combo["corr_weight"]),
                "epochs": int(args.epochs),
                "ckpt_path": str(latest_ckpt),
                "pearson_mean": metrics[0],
                "pearson_median": metrics[1],
                "spearman_mean": metrics[2],
                "spearman_median": metrics[3],
                "auroc_mean": metrics[4],
                "auroc_median": metrics[5],
            }
            append_summary_row(summary_path, row)
            logger.info("Completed run %s", run_id)
        except Exception as err:  # pragma: no cover - defensive guard
            logger.error("Run %s failed: %s", run_id, err, exc_info=True)
            log_failure(out_root, run_id, err)
            continue

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by="pearson_mean", ascending=False)
            logger.info("Top-5 runs by pearson_mean:")
            logger.info("%s", summary_df.head(5).to_string(index=False))

            summary_df_auroc = summary_df.sort_values(by="auroc_mean", ascending=False)
            logger.info("Top-5 runs by auroc_mean:")
            logger.info("%s", summary_df_auroc.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
