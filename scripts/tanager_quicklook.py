#!/usr/bin/env python3

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.satellites import tanager_utils


def _parse_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. '665,565,490'")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float in triplet: {text}") from exc


def _parse_stretch(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Stretch expects two comma-separated percentiles, e.g. '2,98'")
    try:
        low, high = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid percentile values: {text}") from exc
    if not (0 <= low < high <= 100):
        raise argparse.ArgumentTypeError("Stretch percentiles must satisfy 0 <= low < high <= 100")
    return (low, high)


def build_parser():
    parser = argparse.ArgumentParser(description="Quicklook generator for Planet Tanager HDF5 radiance products.")
    parser.add_argument("file", help="Path to the Tanager .h5/.hdf/.hdf5 file (or ZIP containing it).")
    parser.add_argument(
        "--dataset-path",
        default=tanager_utils.TANAGER_TOA_RADIANCE_DATASET,
        help="Dataset path to visualize (defaults to TOA radiance).",
    )
    parser.add_argument(
        "--rgb-wavelengths",
        type=_parse_triplet,
        default=tanager_utils.DEFAULT_RGB_WAVELENGTHS,
        help="Comma-separated wavelengths (nm) for R,G,B channels. Default: 665,565,490.",
    )
    parser.add_argument(
        "--stretch",
        type=_parse_stretch,
        default=(2.0, 98.0),
        help="Comma-separated lower/upper percentiles for contrast stretch. Default: 2,98.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.6,
        help="Gamma to apply after stretching (1.0 disables). Default: 1.6",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1200,
        help="Downsample so the largest dimension is at most this size (1 disables). Default: 1200",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Skip applying nodata mask when building the RGB quicklook.",
    )
    parser.add_argument(
        "--with-geo",
        action="store_true",
        help="Load geolocation arrays (useful for validation; not needed for RGB export).",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults to <input>_rgb.png next to the input archive.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a short metadata summary to stdout.",
    )
    parser.add_argument(
        "--pixel",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Print a single-pixel spectrum at zero-based (row, col) for validation.",
    )
    return parser


def _default_output_path(input_path: Path) -> Path:
    base = input_path.stem
    return input_path.with_name(f"{base}_rgb.png")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        parser.error(f"File not found: {file_path}")
    if args.max_size <= 0:
        parser.error("--max-size must be > 0")

    resolved_path = str(file_path)
    temp_dir: tempfile.TemporaryDirectory | None = None

    if file_path.suffix.lower() == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        extracted = tanager_utils.extract_hdf_from_zip(str(file_path), temp_dir.name)
        if not extracted:
            temp_dir.cleanup()
            parser.error(f"No HDF5 file found inside ZIP archive: {file_path}")
        resolved_path = extracted
    elif file_path.suffix.lower() not in {".h5", ".hdf", ".hdf5"}:
        parser.error(f"Unsupported file format: {file_path}. Expected .h5/.hdf/.hdf5 or .zip")

    try:
        cube = tanager_utils.load_tanager_cube(
            resolved_path,
            dataset_path=args.dataset_path,
            load_masks=not args.no_mask,
            load_geolocation=args.with_geo,
        )

        if args.summary:
            print(tanager_utils.summarize_cube(cube))
            if cube.wavelengths is not None and cube.wavelengths.size:
                print(
                    f"Wavelength span: {cube.wavelengths.min():.1f}–{cube.wavelengths.max():.1f} nm "
                    f"({cube.wavelengths.size} bands)"
                )

        rgb = tanager_utils.quicklook_rgb(
            cube,
            rgb_wavelengths=args.rgb_wavelengths,
            stretch=args.stretch,
            gamma=args.gamma,
            max_size=None if args.max_size == 1 else args.max_size,
            mask_name=None if args.no_mask else "nodata_pixels",
        )

        output_path = Path(args.output).expanduser() if args.output else _default_output_path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tanager_utils.save_rgb_png(rgb, str(output_path))
        print(f"Saved RGB quicklook to {output_path}")

        if args.pixel:
            row, col = args.pixel
            bands, rows, cols = cube.data.shape
            if not (0 <= row < rows and 0 <= col < cols):
                parser.error(f"--pixel out of bounds for dataset shape (bands={bands}, rows={rows}, cols={cols})")
            spectrum = cube.data[:, row, col]
            if cube.wavelengths is not None and cube.wavelengths.shape[0] == spectrum.shape[0]:
                head = "\n".join(
                    f"  {wl:7.2f} nm : {val:.6g}"
                    for wl, val in zip(cube.wavelengths, spectrum)
                )
            else:
                head = ", ".join(f"{val:.6g}" for val in spectrum)
            print(f"Spectrum at (row={row}, col={col}):")
            print(head)
            finite = cube.data[np.isfinite(cube.data)]
            if finite.size:
                min_val, max_val = float(finite.min()), float(finite.max())
                print(f"Min/Max radiance in cube: {min_val:.6g} / {max_val:.6g}")
            else:
                print("Min/Max radiance in cube: no finite samples")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main(sys.argv[1:])
