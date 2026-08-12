"""Run CO2 full-column matched filtering for EnMAP ZIP products by case study.

The Borger et al. case-study folders contain L1B ZIP products rather than
already-extracted GeoTIFFs. This runner extracts one product at a time, so the
large intermediates are removed after each scene. Duplicate archive revisions
of the same acquisition are reduced to the newest filename revision.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines import enmap_pipeline


LOGGER = logging.getLogger(__name__)
ACQUISITION_PATTERN = re.compile(r"L1B-DT\d+_\d{8}T\d{6}Z_\d{3}", re.IGNORECASE)
UNASSIGNED_DIR = "unassigned_adjacent_scenes"
DEFAULT_LUT = REPO_ROOT.parent / "LUTs" / "dataset_co2_full.hdf5"
DEFAULT_SNR = (
    REPO_ROOT
    / "reference_snr"
    / "enmap"
    / "L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z"
    / "snr_reference_columnwise.npz"
)


@dataclass(frozen=True)
class ArchiveSelection:
    archive: Path
    acquisition_key: str
    skipped_revisions: tuple[Path, ...]


def acquisition_key(archive: Path) -> str:
    """Return the stable EnMAP acquisition identifier, excluding product revision."""
    match = ACQUISITION_PATTERN.search(archive.name)
    if not match:
        raise ValueError(f"Could not identify EnMAP acquisition from {archive.name}")
    return match.group(0).upper()


def select_archives(root: Path, include_unassigned: bool = False) -> list[ArchiveSelection]:
    """Choose the newest product revision for each acquisition in the case folders."""
    candidates = sorted(root.rglob("ENMAP*.ZIP"))
    if not include_unassigned:
        candidates = [path for path in candidates if UNASSIGNED_DIR not in path.relative_to(root).parts]

    grouped: dict[str, list[Path]] = {}
    for archive in candidates:
        grouped.setdefault(acquisition_key(archive), []).append(archive)

    selections = []
    for key, revisions in sorted(grouped.items()):
        revisions = sorted(revisions)
        selected = revisions[-1]
        selections.append(
            ArchiveSelection(
                archive=selected,
                acquisition_key=key,
                skipped_revisions=tuple(path for path in revisions if path != selected),
            )
        )
    return selections


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive while rejecting members that would escape destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member in {archive.name}: {member.filename}")
        zip_file.extractall(destination)


def locate_scene_files(extracted_root: Path) -> tuple[Path, Path, Path]:
    """Find the three files required by the EnMAP pipeline inside an archive."""
    vnir = sorted(extracted_root.rglob("*SPECTRAL_IMAGE_VNIR.TIF"))
    swir = sorted(extracted_root.rglob("*SPECTRAL_IMAGE_SWIR.TIF"))
    metadata = sorted(extracted_root.rglob("*METADATA.XML"))
    if len(vnir) != 1 or len(swir) != 1 or len(metadata) != 1:
        raise FileNotFoundError(
            "Expected exactly one VNIR GeoTIFF, SWIR GeoTIFF, and METADATA.XML "
            f"in {extracted_root}; found {len(vnir)}, {len(swir)}, {len(metadata)}."
        )
    return vnir[0], swir[0], metadata[0]


def validate_inputs(lut_file: Path, snr_reference: Path) -> None:
    missing = [str(path) for path in (lut_file, snr_reference) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required ancillary input(s): {', '.join(missing)}")


def write_manifest(root: Path, records: list[dict]) -> None:
    """Persist provenance and run status beside the case-study folders."""
    path = root / "co2_batch_manifest.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def load_manifest(root: Path) -> list[dict]:
    """Load prior batch status so interrupted runs can resume safely."""
    path = root / "co2_batch_manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def has_complete_outputs(output_dir: Path) -> bool:
    """Return whether the primary enhancement and uncertainty products exist."""
    return any(output_dir.glob("*_MF.tif")) and any(output_dir.glob("*_MF_uncertainty.tif"))


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Case-study root does not exist: {root}")
    validate_inputs(args.lut, args.snr_reference)
    selections = select_archives(root, include_unassigned=args.include_unassigned)
    if not selections:
        raise FileNotFoundError(f"No EnMAP ZIP products found below {root}")

    for selection in selections:
        if selection.skipped_revisions:
            LOGGER.info(
                "%s: using newest revision %s; skipping %s",
                selection.acquisition_key,
                selection.archive.name,
                ", ".join(path.name for path in selection.skipped_revisions),
            )
        else:
            LOGGER.info("%s: %s", selection.acquisition_key, selection.archive.name)

    if args.dry_run:
        return 0

    manifest = load_manifest(root)
    prior_records = {record.get("acquisition_key"): record for record in manifest}
    scratch_dir = args.scratch_dir.resolve() if args.scratch_dir else None
    if scratch_dir:
        scratch_dir.mkdir(parents=True, exist_ok=True)

    for selection in selections:
        output_dir = selection.archive.parent / "CO2_full-column" / selection.acquisition_key
        previous = prior_records.get(selection.acquisition_key)
        if (
            not args.rerun_completed
            and previous
            and previous.get("status") == "completed"
            and has_complete_outputs(output_dir)
        ):
            LOGGER.info("Skipping completed acquisition %s", selection.acquisition_key)
            continue
        record = {
            "acquisition_key": selection.acquisition_key,
            "archive": str(selection.archive),
            "skipped_revisions": [str(path) for path in selection.skipped_revisions],
            "output_dir": str(output_dir),
            "gas": "co2",
            "mf_mode": "full-column",
            "spectral_window_nm": [1900.0, 2100.0],
        }
        try:
            with tempfile.TemporaryDirectory(prefix="hygas_enmap_", dir=scratch_dir) as temporary_dir:
                extracted_root = Path(temporary_dir)
                LOGGER.info("Extracting %s", selection.archive.name)
                safe_extract(selection.archive, extracted_root)
                vnir, swir, metadata = locate_scene_files(extracted_root)
                output_dir.mkdir(parents=True, exist_ok=True)
                enmap_pipeline.detection_enmap(
                    vnir_file=str(vnir),
                    swir_file=str(swir),
                    metadata_file=str(metadata),
                    lut_file=str(args.lut),
                    output_dir=str(output_dir),
                    min_wavelength=None,
                    max_wavelength=None,
                    mf_mode="full-column",
                    snr_reference_path=str(args.snr_reference),
                    gas="co2",
                )
            record["status"] = "completed"
        except Exception as exc:
            LOGGER.exception("Failed processing %s", selection.archive.name)
            record["status"] = "failed"
            record["error"] = str(exc)
        if selection.acquisition_key in prior_records:
            manifest = [
                existing
                for existing in manifest
                if existing.get("acquisition_key") != selection.acquisition_key
            ]
        manifest.append(record)
        prior_records[selection.acquisition_key] = record
        write_manifest(root, manifest)

    failed = [record for record in manifest if record["status"] == "failed"]
    LOGGER.info("Completed %d/%d EnMAP CO2 scenes", len(manifest) - len(failed), len(manifest))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch CO2 matched filtering for EnMAP ZIP case studies.")
    parser.add_argument("--root", type=Path, required=True, help="Case-study root containing AOI folders.")
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT, help="CO2 MODTRAN LUT path.")
    parser.add_argument("--snr-reference", type=Path, default=DEFAULT_SNR, help="EnMAP SNR reference NPZ path.")
    parser.add_argument("--scratch-dir", type=Path, help="Optional temporary extraction directory.")
    parser.add_argument("--include-unassigned", action="store_true", help="Also process adjacent scenes without direct source coverage.")
    parser.add_argument("--dry-run", action="store_true", help="List selected archives without extracting or processing them.")
    parser.add_argument("--rerun-completed", action="store_true", help="Reprocess scenes already marked completed in the manifest.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
