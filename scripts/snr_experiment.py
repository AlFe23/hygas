"""
Run the eight-case SNR experiment (A–H) for PRISMA or EnMAP scenes.

Usage:
    python -m hygas.scripts.snr_experiment --sensor prisma --input <L1.he5> [--input <L2C.he5>] ...
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from .core import noise, targets
from .diagnostics import plots, pca_tools, striping
from .satellites import enmap_utils, prisma_utils


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A–H SNR experiment without tiling.")
    parser.add_argument(
        "--sensor",
        choices=["prisma", "enmap"],
        required=True,
        help="Target sensor.",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input paths. PRISMA: L1 (and optionally L2C). EnMAP: directory or VNIR/SWIR/METADATA files.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help='Optional ROI "x0:x1,y0:y1" (columns,rows).',
    )
    parser.add_argument(
        "--bands",
        default=None,
        help='Optional spectral window "min_nm:max_nm".',
    )
    parser.add_argument(
        "--cases",
        default="A,B,C,D,E,F,G,H",
        help="Comma-separated list of cases to run (default: all).",
    )
    parser.add_argument(
        "--k-pca",
        type=int,
        default=6,
        help="Number of principal components for residual estimation.",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=["diff", "hp", "std"],
        default="diff",
        help="Sigma estimator for plain noise.",
    )
    parser.add_argument(
        "--diff-axis",
        choices=["rows", "columns"],
        default="columns",
        help="Axis for first-difference sigma (rows=along-track, columns=across-track).",
    )
    parser.add_argument(
        "--hp-kxy",
        type=int,
        default=31,
        help="Kernel size for the high-pass sigma estimator.",
    )
    parser.add_argument(
        "--destripe-notch",
        type=float,
        default=0.02,
        help="Half-bandwidth (cycles/pixel) for FFT notch filter.",
    )
    parser.add_argument(
        "--destripe-atten-db",
        type=float,
        default=30.0,
        help="Attenuation (dB) inside the notch filter.",
    )
    parser.add_argument(
        "--destripe-min-peak-db",
        type=float,
        default=5.0,
        help="Minimum stripe peak (dB) required to trigger the notch filter.",
    )
    parser.add_argument(
        "--disable-notch",
        action="store_true",
        help="Disable FFT notch filtering (keep column equalization only).",
    )
    parser.add_argument(
        "--mask-frac",
        type=float,
        default=0.12,
        help="Fraction of pixels retained in the homogeneous mask.",
    )
    parser.add_argument(
        "--min-column-pixels",
        type=int,
        default=16,
        help="Minimum valid pixels per column for columnwise aggregation.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output root directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_roi(spec: str, rows: int, cols: int) -> Tuple[slice, slice]:
    try:
        x_part, y_part = spec.split(",")
        x0, x1 = [int(v) for v in x_part.split(":")]
        y0, y1 = [int(v) for v in y_part.split(":")]
    except Exception as exc:
        raise ValueError("ROI must follow 'x0:x1,y0:y1'.") from exc

    if not (0 <= x0 < x1 <= cols and 0 <= y0 < y1 <= rows):
        raise ValueError(f"ROI {spec} outside image bounds (rows={rows}, cols={cols}).")

    return slice(y0, y1), slice(x0, x1)


def parse_band_range(spec: str) -> Tuple[float, float]:
    try:
        bmin, bmax = [float(v) for v in spec.split(":")]
    except Exception as exc:
        raise ValueError("Band range must follow 'min_nm:max_nm'.") from exc
    if bmin >= bmax:
        raise ValueError("Band range requires min < max.")
    return bmin, bmax


def derive_scene_id(paths: Sequence[str], fallback: str) -> str:
    for path in paths:
        name = os.path.basename(path)
        candidate = prisma_utils.get_date_from_filename(name)
        if candidate:
            return candidate
    stem = Path(fallback).stem
    return stem or "scene"


@dataclass
class SceneData:
    cube: np.ndarray
    wavelengths: np.ndarray
    scene_id: str
    metadata: Dict[str, str]


def resolve_prisma_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if abs_path.lower().endswith(".zip"):
        extracted = prisma_utils.extract_he5_from_zip(abs_path, os.path.dirname(abs_path))
        if extracted is None:
            raise FileNotFoundError(f"No .he5 found inside ZIP {path}.")
        return extracted
    return abs_path


def load_prisma_scene(inputs: Sequence[str]) -> SceneData:
    if not inputs:
        raise ValueError("PRISMA requires at least an L1 input file.")

    l1_candidate = inputs[0]
    l2_candidate = inputs[1] if len(inputs) > 1 else None

    l1_path = resolve_prisma_path(l1_candidate)
    l2_path = resolve_prisma_path(l2_candidate) if l2_candidate else None

    cube, cw_matrix, *_ = prisma_utils.prisma_read(l1_path)
    cube_brc = np.transpose(cube, (2, 0, 1)).astype(np.float32)

    cw_array = np.asarray(cw_matrix)
    if cw_array.ndim == 2:
        wl = np.nanmean(cw_array, axis=0)
    else:
        wl = cw_array
    wl = np.asarray(wl, dtype=float)

    scene_id = derive_scene_id(inputs, l1_candidate)
    metadata = {
        "l1": l1_candidate,
    }
    if l2_candidate:
        metadata["l2c"] = l2_candidate

    return SceneData(cube=cube_brc, wavelengths=wl, scene_id=scene_id, metadata=metadata)


def load_enmap_scene(inputs: Sequence[str]) -> SceneData:
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        vnir_file, swir_file, xml_file = enmap_utils.find_enmap_files(inputs[0])
        scene_id = Path(inputs[0]).name
    else:
        vnir_file = swir_file = xml_file = None
        for path in inputs:
            lower = path.lower()
            if lower.endswith(".xml"):
                xml_file = path
            elif "vnir" in lower:
                vnir_file = path
            elif "swir" in lower:
                swir_file = path
        if not (vnir_file and swir_file and xml_file):
            raise ValueError("Provide directory or VNIR/SWIR/metadata files for EnMAP.")
        scene_id = Path(xml_file).stem

    cube, wl, *_ = enmap_utils.enmap_read(vnir_file, swir_file, xml_file)
    cube_brc = np.transpose(cube, (2, 0, 1)).astype(np.float32)
    wl = np.asarray(wl, dtype=float)

    metadata = {
        "vnir": vnir_file,
        "swir": swir_file,
        "metadata": xml_file,
    }

    return SceneData(cube=cube_brc, wavelengths=wl, scene_id=scene_id, metadata=metadata)


def select_bands(cube: np.ndarray, wl: np.ndarray, band_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    idx = targets.select_band_indices(wl, band_range[0], band_range[1])
    if idx.size == 0:
        raise ValueError(f"No bands within {band_range[0]}-{band_range[1]} nm.")
    return cube[idx], wl[idx]


def write_case_csv(path: Path, result: Dict[str, object]) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["band_nm", "mu", "sigma_type", "sigma", "snr_median", "snr_p90", "case", "aggregation"])
        for bn, mu, sig, snr_m, snr_p in zip(
            np.asarray(result["band_nm"]),
            np.asarray(result["mu"]),
            np.asarray(result["sigma"]),
            np.asarray(result["snr_median"]),
            np.asarray(result["snr_p90"]),
        ):
            writer.writerow([
                float(bn),
                float(mu),
                result["sigma_kind"],
                float(sig),
                float(snr_m),
                float(snr_p),
                result["case"],
                result["aggregation"],
            ])

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_cases(
    wl: np.ndarray,
    cube_plain: np.ndarray,
    cube_ds: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    outdir: Path,
) -> Dict[str, Dict[str, object]]:
    diff_axis = 1 if args.diff_axis == "columns" else 0
    cases_requested = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
    valid_cases = list("ABCDEFGH")
    for case in cases_requested:
        if case not in valid_cases:
            raise ValueError(f"Unknown case '{case}'. Expected subset of {', '.join(valid_cases)}.")

    recon_plain, resid_plain, model_plain = pca_tools.pca_decompose(cube_plain, mask=mask, k=args.k_pca)
    recon_ds, resid_ds, model_ds = pca_tools.pca_decompose(cube_ds, mask=mask, k=args.k_pca)

    case_map = {
        "A": {"cube": cube_plain, "target": cube_plain, "columnwise": False, "sigma_kind": "total"},
        "B": {"cube": cube_plain, "target": cube_plain, "columnwise": True, "sigma_kind": "total"},
        "C": {"cube": cube_plain, "target": resid_plain, "columnwise": False, "sigma_kind": "random"},
        "D": {"cube": cube_plain, "target": resid_plain, "columnwise": True, "sigma_kind": "random"},
        "E": {"cube": cube_ds, "target": cube_ds, "columnwise": False, "sigma_kind": "total"},
        "F": {"cube": cube_ds, "target": cube_ds, "columnwise": True, "sigma_kind": "total"},
        "G": {"cube": cube_ds, "target": resid_ds, "columnwise": False, "sigma_kind": "random"},
        "H": {"cube": cube_ds, "target": resid_ds, "columnwise": True, "sigma_kind": "random"},
    }

    results: Dict[str, Dict[str, object]] = {}

    for case_id in cases_requested:
        spec = case_map[case_id]
        if spec["columnwise"]:
            res = noise.snr_columnwise(
                target_cube=spec["target"],
                radiance_cube=spec["cube"],
                band_nm=wl,
                mask=mask,
                sigma_kind=spec["sigma_kind"],
                sigma_mode=args.sigma_mode,
                hp_kxy=args.hp_kxy,
                min_valid=args.min_column_pixels,
            )
        else:
            res = noise.snr_wholeroi(
                target_cube=spec["target"],
                radiance_cube=spec["cube"],
                band_nm=wl,
                mask=mask,
                sigma_kind=spec["sigma_kind"],
                sigma_mode=args.sigma_mode,
                diff_axis=diff_axis,
                hp_kxy=args.hp_kxy,
            )

        result_dict = {
            "case": case_id,
            "band_nm": res.band_nm,
            "mu": res.mu,
            "sigma": res.sigma,
            "snr": res.snr,
            "snr_median": res.snr_median,
            "snr_p90": res.snr_p90,
            "sigma_kind": res.sigma_kind,
            "aggregation": res.aggregation,
            "sigma_mode": args.sigma_mode,
            "details": res.details,
        }
        results[case_id] = result_dict

        csv_path = outdir / f"snr_cases_{case_id}.csv"
        write_case_csv(csv_path, result_dict)

    # Always produce PCA diagnostics
    pca_tools.plot_pca_summary(
        model_plain,
        cube_plain,
        recon_plain,
        resid_plain,
        wl,
        outdir / "pca_summary_plain.png",
    )
    pca_tools.plot_pca_summary(
        model_ds,
        cube_ds,
        recon_ds,
        resid_ds,
        wl,
        outdir / "pca_summary_destriped.png",
    )

    return results


def main():
    args = parse_args()

    if args.sensor == "prisma":
        scene = load_prisma_scene(args.input)
    else:
        scene = load_enmap_scene(args.input)

    cube = scene.cube
    wl = scene.wavelengths

    rows, cols = cube.shape[1:]
    if args.roi:
        row_slice, col_slice = parse_roi(args.roi, rows, cols)
        cube = cube[:, row_slice, col_slice]
    if args.bands:
        cube, wl = select_bands(cube, wl, parse_band_range(args.bands))

    mask = noise.build_homogeneous_mask_auto(cube, frac_keep=args.mask_frac)

    cube_plain = cube.astype(np.float32, copy=True)
    cube_ds, destripe_info = striping.light_destripe_cube(
        cube_plain,
        mask=mask,
        notch_df=args.destripe_notch,
        attenuation_db=args.destripe_atten_db,
        min_peak_db=args.destripe_min_peak_db,
        use_notch=not args.disable_notch,
    )

    outdir = Path(args.outdir) / args.sensor / scene.scene_id
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_cases(wl, cube_plain, cube_ds, mask, args, outdir)

    # Striping diagnostics
    peak_scores = []
    for info in destripe_info:
        fft_plain = info.get("fft_plain") or {}
        peak_scores.append(fft_plain.get("peak_db"))
    with np.errstate(invalid="ignore"):
        if any(np.isfinite(score) for score in peak_scores):
            best_idx = int(np.nanargmax(np.asarray([score if score is not None else np.nan for score in peak_scores])))
        else:
            best_idx = cube_plain.shape[0] // 2

    info = destripe_info[best_idx]
    destripe_label = "Destripe: equalization + notch" if not args.disable_notch else "Destripe: equalization only"
    plots.plot_striping_diagnostics(
        cube_plain[best_idx],
        cube_ds[best_idx],
        mask,
        info.get("fft_plain", {}),
        info.get("fft_destriped", {}),
        info.get("f0_plain"),
        info.get("f0_destriped"),
        outdir / "striping_diagnostics.png",
        destripe_label,
    )

    # Overview plot
    curves = [results[k] for k in sorted(results)]
    plots.plot_snr_cases(wl, curves, f"{args.sensor.upper()} scene {scene.scene_id}", outdir / "snr_cases_overview.png")

    print(f"SNR experiment completed. Outputs stored in {outdir}")


if __name__ == "__main__":
    main()
