#!/usr/bin/env python3
"""Build an experimental EMIT column-wise SNR reference from native L1B RAD.

Use all finite spectra in the requested row/spectral window. Remove a global
PCA reconstruction, then estimate random noise from native along-track first
differences of the residual, separately for each band and detector column.
No homogeneity ranking, official QA exclusions, destriping or column filling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.core import noise
from scripts.diagnostics import pca_tools
from scripts.satellites import emit_utils


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return dict(path=str(path), size=before.st_size, sha256=digest.hexdigest())


def measure_reference(scene, *, k_pca: int = 4, min_column_pixels: int = 16):
    """Return the common reference container, support arrays and PCA diagnostics.

    The PCA basis is estimated once from all finite selected spectra. Differences
    require both neighbouring *native* rows: missing rows are never stitched.
    Full native detector width is required, even if some columns lack support.
    """
    rad = scene.cube
    nb, nr, nc = rad.shape
    if not isinstance(k_pca, int) or not 1 <= k_pca < nb:
        raise ValueError("k_pca must be positive and smaller than the selected band count.")
    if not isinstance(min_column_pixels, int) or min_column_pixels < 3:
        raise ValueError("min_column_pixels must be an integer >= 3.")
    if not np.array_equal(scene.column_indices, np.arange(scene.metadata["native_shape_brc"][2])):
        raise ValueError("SNR references require the full native detector-column axis.")
    if nr < 3 or not np.all(np.diff(scene.row_indices) == 1):
        raise ValueError("At least three contiguous native rows are required.")
    mask = scene.valid_pixels
    counts = mask.sum(axis=0)
    pairs = (mask[:-1] & mask[1:]).sum(axis=0)
    eligible = (counts >= min_column_pixels) & (pairs >= 2)
    if mask.sum() < max(3, k_pca) or not eligible.any():
        raise ValueError("Insufficient finite spectra/adjacent row pairs; no fallback or interpolation.")

    reconstruction, residual, model = pca_tools.pca_decompose(rad, mask=mask, k=k_pca)
    del reconstruction
    result = noise.snr_columnwise(
        residual, rad, scene.wavelengths, mask & eligible[None, :], "random",
        sigma_mode="diff", min_valid=min_column_pixels,
    )
    del residual
    measured = np.asarray(result.details["valid_columns"], dtype=int)
    mu = np.full((nb, nc), np.nan)
    snr = np.full((nb, nc), np.nan)
    sigma = np.full((nb, nc), np.nan)
    mu[:, measured] = result.details["mu_columns"].T
    snr[:, measured] = result.details["snr_columns"].T
    sigma[:, measured] = result.details["sigma_columns"].T
    for array in (mu, snr, sigma):
        array[:, ~eligible] = np.nan
    attributes = scene.metadata["product_attributes"]
    metadata = dict(
        sensor="emit", status="experimental", scientifically_validated=False,
        scene_id=scene.scene_id, product_version=attributes.get("product_version"),
        software_build_version=attributes.get("software_build_version"),
        radiance_unit=emit_utils.RADIANCE_UNIT, unit_multiplier_to_hygas=1.0,
        native_shape=scene.metadata["native_shape_brc"][1:] + [scene.metadata["native_shape_brc"][0]],
        native_column_indices=scene.column_indices.tolist(),
        native_row_range=[int(scene.row_indices[0]), int(scene.row_indices[-1]) + 1],
        k_pca=k_pca, pca_solver="full", sigma_mode="diff", sigma_kind="random",
        aggregation="columnwise", mask="finite_full", mask_frac=None,
        random_noise_definition="PCA residual; std(native-row first differences, ddof=1)/sqrt(2)",
        snr_denominator_regularizer=noise.EPS, min_column_pixels=min_column_pixels,
        minimum_adjacent_pairs=2, official_QA_applied=False, QA_exclusions=[],
        destriping=False, equalization=False, notch=False, column_interpolation=False,
        support_file="support.npz",
        limitations=["Experimental single-scene reference; transfer across build/epoch is not validated",
                     "Finite validity is not a cloud, saturation or band-interpolation QA mask",
                     "Signed/nonpositive cells are retained but cannot support positive-radiance uncertainty scaling",
                     "PCA and along-track correlation can affect the estimated random-noise component"],
    )
    ref = noise.ColumnwiseSNRReference(
        scene.wavelengths, mu, snr, np.flatnonzero(eligible), metadata, scene.band_indices,
    )
    support = dict(
        valid_pixel_counts=counts, adjacent_pair_counts=pairs, eligible_columns=eligible,
        positive_mean_and_snr=np.isfinite(mu) & (mu > 0) & np.isfinite(snr) & (snr > 0),
        sigma_measured=sigma, native_band_indices=scene.band_indices,
        native_column_indices=scene.column_indices, native_row_indices=scene.row_indices,
    )
    pca = dict(components=model["components"], mean=model["mean_spectrum"],
               explained_variance_ratio=model["explained_variance_ratio"],
               fit_pixel_count=int(mask.sum()), k=k_pca)
    return ref, support, pca


def compare_reference(actual, expected):
    """Compare numeric arrays, independently of timestamps or provenance strings."""
    for name in ("band_indices", "valid_columns"):
        if not np.array_equal(getattr(actual, name), getattr(expected, name)):
            raise ValueError(f"Reference comparison failed: {name} identities differ.")
    result = {}
    for name in ("band_nm", "mean_radiance", "snr"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a.shape != b.shape or not np.allclose(a, b, rtol=2e-6, atol=1e-7, equal_nan=True):
            raise ValueError(f"Reference comparison failed: {name} differs beyond float32 tolerance.")
        finite = np.isfinite(a) & np.isfinite(b)
        result[name + "_max_abs_error"] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return dict(passed=True, rtol=2e-6, atol=1e-7, **result)


def build_reference(
    input_path: str | Path, output_dir: str | Path, *, bands=(1000.0, 2500.0),
    rows: slice | None = None, k_pca: int = 4, min_column_pixels: int = 16,
    compare_path: str | Path | None = None, threads: int = 2,
) -> Path:
    """Read RAD directly and write a new directory; never replace a reference."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    lo, hi = bands
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi or threads < 1:
        raise ValueError("Require 0 < min_nm < max_nm and threads >= 1.")
    path = Path(input_path).expanduser().resolve()
    print("Hashing RAD input", flush=True)
    identity = file_identity(path)
    stat = path.stat()
    with h5py.File(path, "r") as root:
        wavelengths = emit_utils._numeric(root["sensor_band_parameters/wavelengths"], slice(None))
    selected = np.flatnonzero((wavelengths >= lo) & (wavelengths <= hi))
    if not selected.size:
        raise ValueError("No EMIT bands in the requested interval.")
    scene = emit_utils.read_emit_scene(
        [str(path)], rows=rows, band_indices=selected, load_location=False,
        load_observation=False, load_glt=False,
    )
    print(f"PCA + native-column statistics: {scene.cube.shape}, k={k_pca}", flush=True)
    with threadpool_limits(limits=threads):
        ref, support, pca = measure_reference(scene, k_pca=k_pca, min_column_pixels=min_column_pixels)
    if (stat.st_size, stat.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
        raise RuntimeError("RAD input changed during reference generation.")
    comparison = None
    if compare_path is not None:
        expected = noise.ColumnwiseSNRReference.load(compare_path)
        comparison = dict(input=file_identity(compare_path), **compare_reference(ref, expected))
    ref.metadata.update(
        generated_utc=datetime.now(timezone.utc).isoformat(), source_inputs=[identity],
        spectral_interval_nm=[float(lo), float(hi)], generator="scripts/SNR_emit_reference.py",
    )
    output.mkdir(parents=True, exist_ok=False)
    ref_path = output / "snr_reference_columnwise_experimental.npz"
    ref.save(ref_path)
    np.savez_compressed(output / "support.npz", **support)
    np.savez_compressed(output / "pca.npz", **pca)
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), repo / "scripts/core/noise.py",
               repo / "scripts/diagnostics/pca_tools.py", repo / "scripts/satellites/emit_utils.py"]
    report = dict(
        metadata=ref.metadata, reader=scene.metadata, shape_bands_columns=list(ref.snr.shape),
        finite_spectra=int(scene.valid_pixels.sum()), total_selected_pixels=int(scene.valid_pixels.size),
        eligible_columns=int(support["eligible_columns"].sum()),
        positive_cells=int(support["positive_mean_and_snr"].sum()),
        comparison=comparison, threads=threads, python=sys.version,
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "h5py", "threadpoolctl")},
        sources=[file_identity(p) for p in sources],
        outputs=[file_identity(output / name) for name in
                 (ref_path.name, "support.npz", "pca.npz")],
    )
    with (output / "generation_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved experimental EMIT SNR reference: {ref_path}", flush=True)
    return ref_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical native EMIT L1B RAD NetCDF; OBS/L2A not required.")
    parser.add_argument("--output-dir", required=True, help="New directory for NPZ, support, PCA and JSON provenance.")
    parser.add_argument("--bands", nargs=2, type=float, default=(1000.0, 2500.0), metavar=("MIN_NM", "MAX_NM"))
    parser.add_argument("--rows", nargs=2, type=int, metavar=("START", "STOP"), help="Optional native rows, zero-based, stop exclusive; default all.")
    parser.add_argument("--k-pca", type=int, default=4)
    parser.add_argument("--min-column-pixels", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2, help="Numerical-library thread limit.")
    parser.add_argument("--compare-reference", help="Require numerical agreement with an existing reference (metadata may differ).")
    args = parser.parse_args(argv)
    build_reference(args.input, args.output_dir, bands=args.bands,
                    rows=slice(*args.rows) if args.rows else None, k_pca=args.k_pca,
                    min_column_pixels=args.min_column_pixels, compare_path=args.compare_reference,
                    threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
