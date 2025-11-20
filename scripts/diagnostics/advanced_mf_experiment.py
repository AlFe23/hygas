"""
Experiment utility to evaluate the advanced matched-filter formulation under
different configuration switches (per-cluster targets, adaptive shrinkage).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import advanced_matched_filter as amf

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _parse_flags(values: Sequence[int] | None) -> list[bool]:
    if not values:
        return [False, True]
    unique = sorted(set(values))
    for value in unique:
        if value not in (0, 1):
            raise ValueError("Flag lists only accept 0 (False) or 1 (True).")
    return [bool(v) for v in unique]


def _summarise_array(name: str, array: np.ndarray) -> dict:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"name": name, "count": int(array.size), "finite": 0}
    return {
        "name": name,
        "count": int(array.size),
        "finite": int(finite.size),
        "min": float(np.nanmin(finite)),
        "median": float(np.nanmedian(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
    }


def _prepare_background(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    array = np.load(path)
    if array.ndim == 2:
        norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array) + 1e-9)
        return np.stack([norm] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        return array[..., :3]
    raise ValueError("background array must be HxW or HxWx3/4.")


def _save_preview(
    mf_map: np.ndarray,
    labels: np.ndarray | None,
    path: Path,
    background: np.ndarray | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for previews but is not installed.")
    ncols = 3 if background is not None else (2 if labels is not None else 1)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    idx = 0
    if background is not None:
        axes[idx].imshow(np.clip(background, 0, 1))
        axes[idx].set_title("Background")
        axes[idx].axis("off")
        idx += 1
    vmax = np.nanpercentile(np.abs(mf_map), 99)
    vmax = vmax if vmax > 0 else np.nanmax(np.abs(mf_map))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    axes[idx].imshow(mf_map, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[idx].set_title("MF response")
    axes[idx].axis("off")
    idx += 1
    if labels is not None:
        axes[idx].imshow(labels, cmap="tab20")
        axes[idx].set_title("Cluster labels")
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    preview_dir = None
    if args.preview_dir:
        if plt is None:
            raise RuntimeError("matplotlib is required for --preview-dir but is not available.")
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
    background = _prepare_background(args.background) if args.background else None
    cube, profile = amf._load_radiance(args.radiance)  # type: ignore[attr-defined]
    targets = amf._load_array(args.targets)  # type: ignore[attr-defined]
    wavelengths = amf._load_array(args.wavelengths)  # type: ignore[attr-defined]
    mask = amf._load_array(args.mask) if args.mask else None  # type: ignore[attr-defined]

    per_cluster_flags = _parse_flags(args.per_cluster_targets)
    adaptive_flags = _parse_flags(args.adaptive_shrinkage)

    summary: list[dict] = []
    if args.additive_sequence:
        combinations = [(False, False), (True, False), (True, True)]
    else:
        combinations = list(itertools.product(per_cluster_flags, adaptive_flags))
    logging.info("Running %d combinations.", len(combinations))

    for per_cluster, adaptive in combinations:
        label = f"pct{int(per_cluster)}_adapt{int(adaptive)}"
        logging.info("Combination %s", label)
        mf_map = amf.run_advanced_mf(
            radiance_cube=cube,
            targets=targets,
            wavelengths=wavelengths,
            mask=mask,
            group_min=args.group_min,
            group_max=args.group_max,
            n_clusters=args.clusters,
            shrinkage=args.shrinkage,
            per_cluster_targets=per_cluster,
            adaptive_shrinkage=adaptive,
            min_clusters=args.min_clusters,
        )
        labels = amf.get_last_cluster_labels()
        np.save(os.path.join(args.output_dir, f"mf_{label}.npy"), mf_map)
        if labels is not None:
            np.save(os.path.join(args.output_dir, f"clusters_{label}.npy"), labels)
        if preview_dir is not None:
            png_path = preview_dir / f"preview_{label}.png"
            _save_preview(mf_map, labels, png_path, background=background)
        stats = _summarise_array(label, mf_map)
        stats["per_cluster_targets"] = per_cluster
        stats["adaptive_shrinkage"] = adaptive
        summary.append(stats)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logging.info("Experiment finished. Outputs written to %s", args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare advanced MF switches.")
    parser.add_argument("--radiance", required=True, help="Radiance cube (.npy/.npz/.tif).")
    parser.add_argument("--targets", required=True, help="Target spectra (.npy/.npz).")
    parser.add_argument("--wavelengths", required=True, help="Band wavelengths (.npy/.npz).")
    parser.add_argument("--mask", help="Optional mask aligned with the radiance cube.")
    parser.add_argument("--output-dir", required=True, help="Directory to store MF maps and summaries.")
    parser.add_argument("--group-min", type=int, default=10, help="Minimum column group width.")
    parser.add_argument("--group-max", type=int, default=30, help="Maximum column group width.")
    parser.add_argument("--clusters", type=int, default=3, help="Requested number of clusters per group.")
    parser.add_argument("--min-clusters", type=int, default=3, help="Minimum clusters per group.")
    parser.add_argument("--shrinkage", type=float, default=0.1, help="Covariance shrinkage coefficient.")
    parser.add_argument(
        "--per-cluster-targets",
        nargs="+",
        type=int,
        choices=(0, 1),
        help="Values to test for per-cluster targets (0=off, 1=on). Defaults to both.",
    )
    parser.add_argument(
        "--adaptive-shrinkage",
        nargs="+",
        type=int,
        choices=(0, 1),
        help="Values to test for adaptive shrinkage (0=off, 1=on). Defaults to both.",
    )
    parser.add_argument(
        "--additive-sequence",
        action="store_true",
        help="Run three additive steps: baseline, +per-cluster-targets, +adaptive-shrinkage.",
    )
    parser.add_argument(
        "--background",
        help="Optional background image (NPY/NPZ, 2D or RGB) used in preview panels.",
    )
    parser.add_argument(
        "--preview-dir",
        help="Optional directory to store PNG previews comparing MF maps and cluster labels.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
