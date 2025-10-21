"""Compare baseline and front-door evaluation metrics and produce reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from train.utils import ensure_dir
import numpy as np

from train.utils import ensure_dir


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_spearman(metrics: Dict) -> np.ndarray:
    per_gene = metrics.get("per_gene", {})
    values = per_gene.get("spearman")
    if values is None:
        raise KeyError("per_gene.spearman missing from metrics JSON")
    return np.array(values, dtype=float)


def compute_statistics(baseline: np.ndarray, frontdoor: np.ndarray) -> Tuple[float, float, float, float]:
    valid = ~np.isnan(baseline) & ~np.isnan(frontdoor)
    if not np.any(valid):
        raise ValueError("No valid overlapping genes for comparison")
    base_vals = baseline[valid]
    front_vals = frontdoor[valid]
    delta = front_vals - base_vals
    median_base = float(np.median(base_vals))
    median_front = float(np.median(front_vals))
    median_delta = float(np.median(delta))
    improved_ratio = float(np.mean(delta > 0.0))
    return median_base, median_front, median_delta, improved_ratio


def plot_comparisons(
    baseline: np.ndarray,
    frontdoor: np.ndarray,
    delta: np.ndarray,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 4))
    plt.boxplot([baseline, frontdoor], labels=["Baseline", "Frontdoor"], vert=True)
    plt.ylabel("Spearman rho per gene")
    plt.title("Per-gene Spearman Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "spearman_boxplot.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(baseline, bins=40, alpha=0.6, label="Baseline")
    plt.hist(frontdoor, bins=40, alpha=0.6, label="Frontdoor")
    plt.xlabel("Spearman rho")
    plt.ylabel("Frequency")
    plt.title("Per-gene Spearman Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "spearman_hist.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=40, alpha=0.8, color="tab:green")
    plt.xlabel("Delta Spearman rho (Frontdoor - Baseline)")
    plt.ylabel("Frequency")
    plt.title("Per-gene Improvement Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "spearman_delta_hist.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and frontdoor evaluation reports")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSON metrics")
    parser.add_argument("--frontdoor", type=str, required=True, help="Path to frontdoor JSON metrics")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for comparison result")
    parser.add_argument("--fig_dir", type=str, default=None, help="Directory to store comparison figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    frontdoor_path = Path(args.frontdoor)
    out_path = Path(args.out)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_path.parent

    baseline_metrics = load_metrics(baseline_path)
    frontdoor_metrics = load_metrics(frontdoor_path)

    baseline_spearman = extract_spearman(baseline_metrics)
    frontdoor_spearman = extract_spearman(frontdoor_metrics)

    valid_mask = ~np.isnan(baseline_spearman) & ~np.isnan(frontdoor_spearman)
    baseline_valid = baseline_spearman[valid_mask]
    frontdoor_valid = frontdoor_spearman[valid_mask]
    delta = frontdoor_valid - baseline_valid

    median_base, median_front, median_delta, improved_ratio = compute_statistics(
        baseline_spearman,
        frontdoor_spearman,
    )

    thresholds = {"delta_rho": 0.03, "improved_ratio": 0.60}
    passed = median_delta >= thresholds["delta_rho"] and improved_ratio >= thresholds["improved_ratio"]

    ensure_dir(out_path.parent)
    result = {
        "median_rho_baseline": median_base,
        "median_rho_frontdoor": median_front,
        "median_delta_rho": median_delta,
        "improved_gene_ratio": improved_ratio,
        "pass": passed,
        "thresholds": thresholds,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_comparisons(baseline_valid, frontdoor_valid, delta, fig_dir)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
