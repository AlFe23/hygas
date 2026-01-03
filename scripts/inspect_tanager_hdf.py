#!/usr/bin/env python3

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.satellites import tanager_utils


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect and print the structure of a Tanager HDF5 file (Basic/Ortho)."
    )
    parser.add_argument("file", help="Path to the Tanager .h5/.hdf/.hdf5 file (or ZIP containing it).")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit traversal depth (root=0). Defaults to full depth.",
    )
    parser.add_argument(
        "--attrs",
        action="store_true",
        help="Include group/dataset attributes in the output.",
    )
    parser.add_argument(
        "--path",
        help=(
            "Inspect a specific dataset or group path (e.g. "
            "'HDFEOS/SWATHS/HYP/Data Fields/toa_radiance'). When omitted, prints the full tree."
        ),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="With --path pointing to a dataset, preview up to N values.",
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=30,
        help="Limit number of child entries shown when --path refers to a group.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the report as a text file.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preview < 0:
        parser.error("--preview must be >= 0")
    if args.max_members <= 0:
        parser.error("--max-members must be > 0")
    if args.preview and not args.path:
        parser.error("--preview requires --path to be specified")

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        parser.error(f"File not found: {file_path}")

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
        if args.path:
            try:
                report = tanager_utils.describe_tanager_hdf_object(
                    resolved_path,
                    args.path,
                    include_attrs=args.attrs,
                    preview=args.preview or None,
                    max_members=args.max_members,
                )
            except KeyError as exc:
                parser.error(f"Path not found in HDF file: {exc}")
        else:
            report = tanager_utils.describe_tanager_hdf_structure(
                resolved_path,
                max_depth=args.max_depth,
                include_attrs=args.attrs,
            )
        print(report)
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"\nSaved report to {output_path}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main(sys.argv[1:])
