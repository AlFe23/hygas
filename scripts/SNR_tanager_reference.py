#!/usr/bin/env python3
"""
Compute a columnwise SNR reference for a Tanager radiance HDF5 product.
The output is a ColumnwiseSNRReference (.npz) compatible with the MF uncertainty
pipeline (same format as PRISMA/EnMAP references).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from scripts.core import noise, targets
from scripts.diagnostics import pca_tools
from scripts.satellites import tanager_utils


def parse_roi(spec: str, rows: int, cols: int) -> Tuple[slice, slice]:
    try:
        x_part, y_part = spec.split(",")
        x0, x1 = [int(v) for v in x_part.split(":")]
        y0, y1 = [int(v) for v in y_part.split(":")]
    except Exception as exc:  # pragma: no cover - user error
        raise ValueError("ROI must follow 'x0:x1,y0:y1' (columns,rows).") from exc
    if not (0 <= x0 < x1 <= cols and 0 <= y0 < y1 <= rows):
        raise ValueError(f"ROI {spec} outside image bounds (rows={rows}, cols={cols}).")
    return slice(y0, y1), slice(x0, x1)


def parse_band_range(spec: str) -> Tuple[float, float]:
    try:
        bmin, bmax = [float(v) for v in spec.split(":")]
    except Exception as exc:  # pragma: no cover - user error
        raise ValueError("Band range must follow 'min_nm:max_nm'.") from exc
    if bmin >= bmax:
        raise ValueError("Band range requires min < max.")
    return bmin, bmax


def resolve_input(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".zip"):
        extracted = tanager_utils.extract_hdf_from_zip(path, Path(path).parent)
        if extracted is None:
            raise FileNotFoundError(f"No HDF5 found inside ZIP: {path}")
        return extracted
    if lower.endswith((".h5", ".hdf", ".hdf5")):
        return path
    raise ValueError(f"Unsupported input format: {path}")


def _fill_missing_columns(arr: np.ndarray) -> np.ndarray:
    filled = arr.copy()
    x_idx = np.arange(arr.shape[1], dtype=float)
    for b in range(arr.shape[0]):
        row = filled[b]
        mask_valid = np.isfinite(row)
        if not mask_valid.any():
            continue
        if mask_valid.all():
            continue
        valid_idx = x_idx[mask_valid]
        filled[b] = np.interp(x_idx, valid_idx, row[mask_valid])
    return filled


def build_reference(
    input_path: str,
    output_path: str,
    band_range: tuple[float, float] | None,
    roi_spec: str | None,
    mask_frac: float,
    k_pca: int,
    sigma_mode: str,
    hp_kxy: int,
    min_column_pixels: int,
) -> Path:
    resolved = resolve_input(input_path)
    cube = tanager_utils.load_tanager_cube(resolved, dataset_path=tanager_utils.TANAGER_TOA_RADIANCE_DATASET)

    rad = cube.data.astype(np.float32, copy=True)
    nodata = cube.masks.get("nodata_pixels")
    if nodata is not None:
        rad = np.where(nodata[None, ...] != 0, np.nan, rad)

    rows, cols = rad.shape[1:]
    if roi_spec:
        row_slice, col_slice = parse_roi(roi_spec, rows, cols)
        rad = rad[:, row_slice, col_slice]

    wl = np.asarray(cube.wavelengths, dtype=float) if cube.wavelengths is not None else np.array([], dtype=float)
    if band_range is not None:
        bmin, bmax = band_range
        idx = targets.select_band_indices(wl, bmin, bmax)
        if idx.size == 0:
            raise ValueError(f"No bands within {bmin}-{bmax} nm.")
        rad = rad[idx]
        wl = wl[idx]

    mask = noise.build_homogeneous_mask_auto(rad, frac_keep=mask_frac)

    _, resid, _ = pca_tools.pca_decompose(rad, mask=mask, k=k_pca)
    res = noise.snr_columnwise(
        target_cube=resid,
        radiance_cube=rad,
        band_nm=wl,
        mask=mask,
        sigma_kind="random",
        sigma_mode=sigma_mode,
        hp_kxy=hp_kxy,
        min_valid=min_column_pixels,
    )

    bands, _, cols = rad.shape
    snr_full = np.full((bands, cols), np.nan, dtype=float)
    mu_full = np.full((bands, cols), np.nan, dtype=float)
    valid_cols = np.asarray(res.details.get("valid_columns", []), dtype=int)
    snr_cols = np.asarray(res.details.get("snr_columns")).astype(float)
    mu_cols = np.asarray(res.details.get("mu_columns")).astype(float)
    if valid_cols.size and snr_cols.size and mu_cols.size:
        snr_full[:, valid_cols] = snr_cols.T
        mu_full[:, valid_cols] = mu_cols.T

    snr_full = _fill_missing_columns(snr_full)
    mu_full = _fill_missing_columns(mu_full)

    ref = noise.ColumnwiseSNRReference(
        band_nm=res.band_nm,
        mean_radiance=mu_full,
        snr=snr_full,
        valid_columns=valid_cols.tolist(),
        band_indices=np.arange(wl.size).tolist(),
        metadata={
            "sensor": "tanager",
            "input": input_path,
            "sigma_mode": sigma_mode,
            "sigma_kind": res.sigma_kind,
            "aggregation": res.aggregation,
            "k_pca": k_pca,
            "mask_frac": mask_frac,
            "roi": roi_spec,
            "band_range": band_range,
        },
    )
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute and save a columnwise SNR reference for Tanager radiance.")
    ap.add_argument("--input", required=True, help="Tanager radiance HDF5 (or ZIP containing it).")
    ap.add_argument("--output", required=True, help="Output .npz path for ColumnwiseSNRReference.")
    ap.add_argument("--bands", default=None, help="Optional spectral window 'min_nm:max_nm'.")
    ap.add_argument("--roi", default=None, help="Optional ROI 'x0:x1,y0:y1' (columns,rows).")
    ap.add_argument("--mask-frac", type=float, default=0.12, help="Fraction of pixels kept in homogeneous mask.")
    ap.add_argument("--k-pca", type=int, default=4, help="Number of principal components for residuals.")
    ap.add_argument("--sigma-mode", choices=["diff", "hp", "std"], default="diff", help="Sigma estimator.")
    ap.add_argument("--hp-kxy", type=int, default=31, help="Kernel size for high-pass sigma estimator.")
    ap.add_argument("--min-column-pixels", type=int, default=16, help="Minimum valid pixels per column.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    band_range = parse_band_range(args.bands) if args.bands else None
    out = build_reference(
        input_path=args.input,
        output_path=args.output,
        band_range=band_range,
        roi_spec=args.roi,
        mask_frac=args.mask_frac,
        k_pca=args.k_pca,
        sigma_mode=args.sigma_mode,
        hp_kxy=args.hp_kxy,
        min_column_pixels=args.min_column_pixels,
    )
    print(f"Saved Tanager SNR reference to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
