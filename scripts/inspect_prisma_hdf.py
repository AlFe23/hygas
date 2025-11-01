#!/usr/bin/env python3

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.satellites import prisma_utils


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect and print the structure of a PRISMA HDF5 (L1/L2C) file."
    )
    parser.add_argument("file", help="Path to the PRISMA .he5 file.")
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
            "'HDFEOS/SWATHS/...'). When omitted, prints the full tree."
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

    suffix = file_path.suffix.lower()
    if suffix == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        extracted = prisma_utils.extract_he5_from_zip(str(file_path), temp_dir.name)
        if not extracted:
            temp_dir.cleanup()
            parser.error(f"No .he5 file found inside ZIP archive: {file_path}")
        resolved_path = extracted
    elif suffix != ".he5":
        parser.error(f"Unsupported file format: {file_path}. Expected .he5 or .zip")

    try:
        if args.path:
            try:
                report = prisma_utils.describe_prisma_hdf_object(
                    resolved_path,
                    args.path,
                    include_attrs=args.attrs,
                    preview=args.preview or None,
                    max_members=args.max_members,
                )
            except KeyError as exc:
                parser.error(f"Path not found in HDF file: {exc}")
        else:
            report = prisma_utils.describe_prisma_hdf_structure(
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
