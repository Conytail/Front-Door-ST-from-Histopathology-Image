"""Marker vs non-marker grouped analysis for per-gene improvements.

Reads baseline and front-door evaluation JSONs, maps marker gene symbols to
Ensembl IDs via HGNC, splits the 250 genes into marker/non-marker, and compares
per-gene Spearman improvements. Outputs a JSON summary and a boxplot figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_eval_json(path: Path) -> Tuple[List[str], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["per_gene"]["gene_names"]
    spearman = np.array(data["per_gene"]["spearman"], dtype=float)
    return names, spearman


def load_markers_yaml(path: Path) -> Set[str]:
    markers = yaml.safe_load(path.read_text(encoding="utf-8"))
    symbols: Set[str] = set()
    for _, genes in markers.items():
        for g in genes or []:
            if isinstance(g, str) and g.strip():
                symbols.add(g.strip().upper())
    return symbols


def load_hgnc_symbol_to_ensembl(path: Path) -> Dict[str, str]:
    symbol_to_ensg: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_symbol = header.index("symbol")
            i_ensg = header.index("ensembl_gene_id")
        except ValueError as e:
            raise RuntimeError("HGNC file missing required columns 'symbol' and 'ensembl_gene_id'") from e
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_symbol, i_ensg):
                continue
            sym = parts[i_symbol].strip().upper()
            ensg = parts[i_ensg].strip()
            if sym and ensg and ensg.startswith("ENSG"):
                # Drop version if present
                ensg = ensg.split(".")[0]
                symbol_to_ensg[sym] = ensg
    return symbol_to_ensg


def compute_group_stats(
    gene_names: List[str],
    base_spear: np.ndarray,
    fd_spear: np.ndarray,
    marker_ensg: Set[str],
) -> Dict[str, object]:
    valid = ~np.isnan(base_spear) & ~np.isnan(fd_spear)
    names = np.array(gene_names)
    base_valid = base_spear[valid]
    fd_valid = fd_spear[valid]
    names_valid = names[valid]
    delta = fd_valid - base_valid

    is_marker = np.array([n in marker_ensg for n in names_valid], dtype=bool)
    groups = {"marker": is_marker, "non_marker": ~is_marker}

    out: Dict[str, object] = {
        "n_all": int(valid.sum()),
        "n_marker": int(is_marker.sum()),
        "n_non_marker": int((~is_marker).sum()),
    }
    for key, mask in groups.items():
        if mask.sum() == 0:
            out[key] = {
                "count": 0,
                "spearman_median_baseline": float("nan"),
                "spearman_median_frontdoor": float("nan"),
                "delta_median": float("nan"),
                "improved_ratio": float("nan"),
            }
            continue
        base_g = base_valid[mask]
        fd_g = fd_valid[mask]
        d_g = delta[mask]
        out[key] = {
            "count": int(mask.sum()),
            "spearman_median_baseline": float(np.median(base_g)),
            "spearman_median_frontdoor": float(np.median(fd_g)),
            "delta_median": float(np.median(d_g)),
            "improved_ratio": float(np.mean(d_g > 0.0)),
        }

    out["delta_median_gap"] = float(
        (out["marker"]["delta_median"] if out["marker"]["count"] else np.nan)
        - (out["non_marker"]["delta_median"] if out["non_marker"]["count"] else np.nan)
    )
    return out


def save_boxplot(delta_marker: np.ndarray, delta_non: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.2))
    plt.boxplot([delta_marker, delta_non], labels=["Marker", "Non-marker"], vert=True)
    plt.ylabel("$\\Delta$ Spearman (Front-door $-$ Baseline)")
    plt.title("Per-gene Improvement by Group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped analysis: marker vs non-marker")
    parser.add_argument("--baseline", type=str, default=str(PROJECT_ROOT / "reports/gene_eval_baseline_val.json"))
    parser.add_argument("--frontdoor", type=str, default=str(PROJECT_ROOT / "reports/gene_eval_frontdoor_val.json"))
    parser.add_argument("--markers_yaml", type=str, default=str(PROJECT_ROOT.parents[0] / "Wu_etal_2021_BRCA_scRNASeq/configs/markers.yaml"))
    parser.add_argument("--hgnc", type=str, default=str(PROJECT_ROOT.parents[0] / "Wu_etal_2021_BRCA_scRNASeq/hgnc_complete_set.txt"))
    parser.add_argument("--out_json", type=str, default=str(PROJECT_ROOT / "reports/marker_group_stats.json"))
    parser.add_argument("--out_fig", type=str, default=str(PROJECT_ROOT / "reports/marker_group_boxplot.png"))
    args = parser.parse_args()

    base_names, base_spear = load_eval_json(Path(args.baseline))
    fd_names, fd_spear = load_eval_json(Path(args.frontdoor))
    if base_names != fd_names:
        raise RuntimeError("Gene order mismatch between baseline and frontdoor JSONs")

    symbols = load_markers_yaml(Path(args.markers_yaml))
    sym2ens = load_hgnc_symbol_to_ensembl(Path(args.hgnc))
    marker_ensg = {sym2ens[s] for s in symbols if s in sym2ens}

    stats = compute_group_stats(base_names, base_spear, fd_spear, marker_ensg)

    # Save JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Save figure
    valid = ~np.isnan(base_spear) & ~np.isnan(fd_spear)
    names = np.array(base_names)
    delta = (fd_spear - base_spear)[valid]
    is_marker = np.array([n in marker_ensg for n in names[valid]], dtype=bool)
    save_boxplot(delta[is_marker], delta[~is_marker], Path(args.out_fig))


if __name__ == "__main__":
    main()

