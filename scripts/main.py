# -*- coding: utf-8 -*-
"""
Unified entry point for PRISMA and EnMAP methane matched-filter processing.
It wraps the satellite-specific pipelines built on top of the shared core modules.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable even when running as `python scripts/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipelines import prisma_pipeline, enmap_pipeline  # type: ignore
from scripts.satellites import prisma_utils  # type: ignore


def build_parser():
    parser = argparse.ArgumentParser(
        description="Methane matched-filter processing for PRISMA and EnMAP scenes."
    )
    parser.add_argument(
        "--satellite",
        choices=["prisma", "enmap"],
        required=True,
        help="Satellite to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["scene", "batch"],
        default="scene",
        help="Scene-level processing or batch directory traversal.",
    )
    parser.add_argument("--lut", required=True, help="Path to the LUT file.")
    parser.add_argument("--k", type=int, default=1, help="Number of clusters for k-means.")
    parser.add_argument(
        "--log-file",
        help="Optional path to append log output (stdout logging always enabled).",
    )
    parser.add_argument(
        "--save-rads",
        action="store_true",
        help="Save the full radiance cube GeoTIFF (PRISMA only).",
    )

    # PRISMA specific
    parser.add_argument("--l1", help="PRS_L1_STD_OFFL .he5 file path.")
    parser.add_argument("--l2c", help="PRS_L2C_STD .he5 file path.")
    parser.add_argument("--dem", help="DEM NetCDF file path.")
    parser.add_argument("--output", help="Output directory for a single scene.")
    parser.add_argument("--output-root", help="Root directory for batch outputs.")
    parser.add_argument("--root-directory", help="Input root directory for batch mode.")
    parser.add_argument("--min-wavelength", type=float, default=2100, help="Minimum wavelength (nm).")
    parser.add_argument("--max-wavelength", type=float, default=2450, help="Maximum wavelength (nm).")
    parser.add_argument(
        "--prisma-mf-mode",
        choices=["srf-column", "full-column", "advanced"],
        default="srf-column",
        help=(
            "PRISMA matched-filter variant: 'srf-column' (default) uses clustering with column-wise SRF "
            "targets, 'full-column' estimates per-column mean/covariance without clustering, and "
            "'advanced' enables the grouped PCA + shrinkage workflow."
        ),
    )

    # EnMAP specific
    parser.add_argument("--vnir", help="ENMAP VNIR GeoTIFF path.")
    parser.add_argument("--swir", help="ENMAP SWIR GeoTIFF path.")
    parser.add_argument("--metadata", help="ENMAP METADATA.XML path.")
    parser.add_argument(
        "--enmap-mf-mode",
        choices=["srf-column", "full-column", "advanced"],
        default="srf-column",
        help=(
            "EnMAP matched-filter variant: 'srf-column' (default) keeps the cluster-tuned "
            "columnwise SRF workflow, 'full-column' estimates per-column mean/covariance "
            "without clustering, and 'advanced' activates the grouped PCA + shrinkage workflow."
        ),
    )

    return parser


def configure_logging(log_file: str | None):
    formatter = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(formatter))
        handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO, format=formatter, handlers=handlers, force=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_file)
    logging.getLogger(__name__).info(
        "Starting %s %s run", args.satellite.upper(), args.mode.upper()
    )

    if args.satellite == "prisma":
        if args.mode == "scene":
            required = ["l1", "l2c", "dem"]
            missing = [opt for opt in required if getattr(args, opt) is None]
            if missing:
                parser.error(f"Missing required PRISMA scene arguments: {', '.join(missing)}")

            temp_extractions: list[str] = []

            def resolve_prisma_scene_input(path: str, label: str) -> str:
                path_obj = Path(path)
                if not path_obj.exists():
                    parser.error(f"{label} file not found: {path}")

                suffix = path_obj.suffix.lower()
                if suffix == ".he5":
                    return str(path_obj)
                if suffix == ".zip":
                    extracted = prisma_utils.extract_he5_from_zip(str(path_obj), str(path_obj.parent))
                    if not extracted:
                        parser.error(f"Zip archive for {label} does not contain a .he5 file: {path}")
                    temp_extractions.append(extracted)
                    return extracted

                parser.error(
                    f"Unsupported {label} format: {path}. Expected a .he5 or .zip file"
                )

            L1_scene_file = resolve_prisma_scene_input(args.l1, "L1")
            L2C_scene_file = resolve_prisma_scene_input(args.l2c, "L2C")
            output_dir = args.output
            if output_dir is None:
                l1_path = Path(L1_scene_file).resolve()
                scene_dir = l1_path.parent
                output_dir = str(scene_dir.parent / f"{scene_dir.name}_output")
            try:
                prisma_pipeline.ch4_detection(
                    L1_file=L1_scene_file,
                    L2C_file=L2C_scene_file,
                    dem_file=args.dem,
                    lut_file=args.lut,
                    output_dir=output_dir,
                    min_wavelength=args.min_wavelength,
                    max_wavelength=args.max_wavelength,
                    k=args.k,
                    mf_mode=args.prisma_mf_mode,
                    save_rads=args.save_rads,
                )
            finally:
                for extracted_file in temp_extractions:
                    try:
                        Path(extracted_file).unlink(missing_ok=True)
                    except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
                        logging.getLogger(__name__).warning(
                            "Could not remove temporary file %s: %s", extracted_file, cleanup_error
                        )
        else:
            required = ["root_directory", "dem"]
            missing = [opt for opt in required if getattr(args, opt) is None]
            if missing:
                parser.error(f"Missing required PRISMA batch arguments: {', '.join(missing)}")
            output_root = args.output_root
            prisma_pipeline.process_directory(
                root_dir=args.root_directory,
                dem_file=args.dem,
                lut_file=args.lut,
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength,
                k=args.k,
                mf_mode=args.prisma_mf_mode,
                output_root_dir=output_root,
                save_rads=args.save_rads,
            )
    else:
        if args.mode == "scene":
            required = ["vnir", "swir", "metadata"]
            missing = [opt for opt in required if getattr(args, opt) is None]
            if missing:
                parser.error(f"Missing required EnMAP scene arguments: {', '.join(missing)}")
            output_dir = args.output
            if output_dir is None:
                scene_dir = Path(args.vnir).resolve().parent
                output_dir = str(scene_dir.parent / f"{scene_dir.name}_output")

            enmap_pipeline.ch4_detection_enmap(
                vnir_file=args.vnir,
                swir_file=args.swir,
                metadata_file=args.metadata,
                lut_file=args.lut,
                output_dir=output_dir,
                k=args.k,
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength,
                mf_mode=args.enmap_mf_mode,
            )
        else:
            if not args.root_directory:
                parser.error("Missing required EnMAP batch argument: --root-directory")
            enmap_pipeline.process_directory_enmap(
                root_dir=args.root_directory,
                lut_file=args.lut,
                k=args.k,
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength,
                mf_mode=args.enmap_mf_mode,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
