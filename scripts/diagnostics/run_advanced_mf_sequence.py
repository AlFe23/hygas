"""
Run the advanced matched filter on a selected PRISMA or EnMAP acquisition using
an additive sequence of algorithmic tweaks (baseline, +per-cluster targets,
+adaptive shrinkage). Each configuration writes to its own subdirectory and can
optionally generate quick-look PNG previews.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.pipelines import prisma_pipeline, enmap_pipeline  # type: ignore
from scripts.satellites import enmap_utils, prisma_utils  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore


def _read_geotiff(path: Path) -> np.ndarray:
    if rasterio is None:
        raise RuntimeError("rasterio is required for preview generation but is not installed.")
    with rasterio.open(path) as src:  # type: ignore[call-arg]
        array = src.read()
    return array


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb, 0, None)
    max_val = np.percentile(rgb, 99)
    if max_val <= 0 or ~np.isfinite(max_val):
        max_val = 1.0
    rgb = rgb / max_val
    rgb = np.clip(rgb, 0, 1)
    return np.transpose(rgb[:3], (1, 2, 0))


def _plot_preview(background: np.ndarray, mf_map: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for preview generation but is not installed.")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(background)
    axes[0].set_title("RGB background")
    axes[0].axis("off")
    vmax = np.nanpercentile(np.abs(mf_map), 99)
    vmax = vmax if vmax > 0 else np.nanmax(np.abs(mf_map))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    axes[1].imshow(mf_map, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[1].set_title("MF response")
    axes[1].axis("off")
    axes[2].imshow(labels, cmap="tab20")
    axes[2].set_title("Cluster labels")
    axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_sequence(args: argparse.Namespace) -> list[tuple[str, dict]]:
    sequences: list[tuple[str, dict]] = []
    if args.sequence == "additive":
        sequences.extend(
            [
                ("baseline", {}),
                ("per_cluster", {"per_cluster_targets": True}),
                ("per_cluster_adaptive", {"per_cluster_targets": True, "adaptive_shrinkage": True}),
            ]
        )
    elif args.sequence == "blend":
        for blend in args.target_blend_values or [0.25, 0.5, 0.75]:
            label = f"blend_{int(blend * 100):03d}"
            sequences.append(
                (
                    label,
                    {
                        "per_cluster_targets": True,
                        "target_blend": float(blend),
                    },
                )
            )
    elif args.sequence == "shrinkage":
        for strength in args.adaptive_shrinkage_floor or [0.0, 0.02, 0.05]:
            label = f"adapt_floor_{int(strength * 1000):03d}"
            sequences.append(
                (
                    label,
                    {
                        "per_cluster_targets": True,
                        "adaptive_shrinkage": True,
                        "adaptive_shrinkage_min": float(strength),
                    },
                )
            )
    return sequences


def _preview_paths_prisma(l1_file: str, output_dir: Path) -> tuple[Path, Path, Path]:
    base = Path(l1_file).stem
    mf = output_dir / f"{base}_MF_concentration.tif"
    rgb = output_dir / f"{base}_rgb.tif"
    classified = output_dir / f"{base}_classified.tif"
    return mf, rgb, classified


def _preview_paths_enmap(metadata_file: str, output_dir: Path) -> tuple[Path, Path, Path]:
    base = enmap_utils.derive_basename_from_metadata(metadata_file)
    mf = output_dir / f"{base}_MF.tif"
    rgb = output_dir / f"{base}_RGB.tif"
    classified = output_dir / f"{base}_CL.tif"
    return mf, rgb, classified


def run_sequence(args: argparse.Namespace) -> None:
    combos = _build_sequence(args)
    base_output = Path(args.output_root)
    base_output.mkdir(parents=True, exist_ok=True)
    preview_root = Path(args.preview_dir) if args.preview_dir else None
    if preview_root:
        preview_root.mkdir(parents=True, exist_ok=True)

    base_options = dict(
        group_min=args.group_min,
        group_max=args.group_max,
        n_clusters=max(1, args.k),
        shrinkage=args.shrinkage,
        min_clusters=args.min_clusters,
    )

    for label, tweaks in combos:
        output_dir = base_output / label
        output_dir.mkdir(parents=True, exist_ok=True)
        options = base_options.copy()
        options.update(tweaks)
        logging.info("Running configuration %s with options %s", label, options)

        if args.satellite == "prisma":
            prisma_pipeline.ch4_detection(
                L1_file=args.l1,
                L2C_file=args.l2c,
                dem_file=args.dem,
                lut_file=args.lut,
                output_dir=str(output_dir),
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength,
                k=args.k,
                mf_mode="advanced",
                save_rads=args.save_rads,
                snr_reference_path=args.snr_reference,
                advanced_mf_options=options,
            )
            mf_path, rgb_path, classified_path = _preview_paths_prisma(args.l1, output_dir)
        else:
            enmap_pipeline.ch4_detection_enmap(
                vnir_file=args.vnir,
                swir_file=args.swir,
                metadata_file=args.metadata,
                lut_file=args.lut,
                output_dir=str(output_dir),
                k=args.k,
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength,
                mf_mode="advanced",
                snr_reference_path=args.snr_reference,
                advanced_mf_options=options,
            )
            mf_path, rgb_path, classified_path = _preview_paths_enmap(args.metadata, output_dir)

        if preview_root:
            logging.info("Generating preview for %s", label)
            mf_array = _read_geotiff(mf_path)[0]
            classified = _read_geotiff(classified_path)[0]
            rgb = _normalize_rgb(_read_geotiff(rgb_path))
            preview_path = preview_root / f"{label}.png"
            _plot_preview(rgb, mf_array, classified, preview_path, title=label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advanced MF tweaks on a PRISMA or EnMAP acquisition.")
    parser.add_argument("--satellite", choices=["prisma", "enmap"], required=True)
    parser.add_argument("--lut", required=True, help="Path to the CH4 LUT.")
    parser.add_argument("--output-root", required=True, help="Base directory to store per-configuration outputs.")
    parser.add_argument("--k", type=int, default=3, help="Cluster count parameter.")
    parser.add_argument("--min-wavelength", type=float, default=2100.0, help="Minimum wavelength (nm).")
    parser.add_argument("--max-wavelength", type=float, default=2450.0, help="Maximum wavelength (nm).")
    parser.add_argument("--group-min", type=int, default=10)
    parser.add_argument("--group-max", type=int, default=30)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument(
        "--sequence",
        choices=["additive", "blend", "shrinkage"],
        default="additive",
        help="Preset experiment to run: additive (baseline→per-cluster→adaptive), blend (target blending values), or shrinkage (different adaptive shrinkage minimums).",
    )
    parser.add_argument(
        "--target-blend-values",
        nargs="+",
        type=float,
        help="Blend coefficients (0-1) for the 'blend' sequence; defaults to 0.25, 0.5, 0.75.",
    )
    parser.add_argument(
        "--adaptive-shrinkage-floor",
        nargs="+",
        type=float,
        help="Minimum adaptive shrinkage values for the 'shrinkage' sequence; defaults to 0, 0.02, 0.05.",
    )
    parser.add_argument("--preview-dir", help="Optional directory to store quick-look PNGs.")

    # PRISMA args
    parser.add_argument("--l1", help="PRS_L1_STD_OFFL file (HE5 or ZIP).")
    parser.add_argument("--l2c", help="PRS_L2C_STD file (HE5 or ZIP).")
    parser.add_argument("--dem", help="DEM NetCDF path.")
    parser.add_argument("--snr-reference", required=True, help="Column-wise SNR reference (NPZ).")
    parser.add_argument("--save-rads", action="store_true", help="Save radiance cube GeoTIFF (PRISMA only).")

    # ENMAP args
    parser.add_argument("--vnir", help="ENMAP VNIR GeoTIFF.")
    parser.add_argument("--swir", help="ENMAP SWIR GeoTIFF.")
    parser.add_argument("--metadata", help="ENMAP METADATA.XML.")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.satellite == "prisma":
        required = ["l1", "l2c", "dem"]
    else:
        required = ["vnir", "swir", "metadata"]
    missing = [opt for opt in required if getattr(args, opt) is None]
    if missing:
        raise ValueError(f"Missing required {args.satellite} arguments: {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    validate_args(args)
    run_sequence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
