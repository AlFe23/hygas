"""Batch runner for the EnMAP notebook case studies.

This script mirrors `notebooks/matched_filter_demo_enmap.ipynb`: for each predefined
scene, it executes every matched-filter mode (srf-column k=1/k=3, full-column,
advanced k=3, jpl) over the 1500–2500 nm window. Outputs are written into
mode-named subfolders placed alongside the input VNIR/SWIR/metadata files, and
filenames are suffixed with the MF mode for easy identification.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT_HINT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from scripts.pipelines import enmap_pipeline


def _locate_repo_root(start: Path) -> Path:
    """Walk up from *start* until a folder containing `scripts` is found."""
    cursor = start.resolve()
    while cursor != cursor.parent:
        if (cursor / "scripts").exists():
            return cursor
        cursor = cursor.parent
    raise RuntimeError("Repository root not found from starting path.")


REPO_ROOT = _locate_repo_root(Path(__file__).parent)
# WINDOW = (1500.0, 2500.0)
WINDOW = (2100.0, 2450.0)
LUT_PATH = next(
    (p for p in (REPO_ROOT.parent / "LUTs/CH4_lut.hdf5", REPO_ROOT / "LUTs/CH4_lut.hdf5") if p.exists()),
    REPO_ROOT.parent / "LUTs/CH4_lut.hdf5",
)

# Per-notebook case studies.
CASE_STUDIES = {
    # "turkmenistan_plume_enmap": {
    #     "vnir": REPO_ROOT
    #     / "test_data/enmap/Turkmenistan_20221002/20221002T074833/ENMAP01-____L1B-DT0000004147_20221002T074833Z_002_V010501_20241110T222710Z-SPECTRAL_IMAGE_VNIR.TIF",
    #     "swir": REPO_ROOT
    #     / "test_data/enmap/Turkmenistan_20221002/20221002T074833/ENMAP01-____L1B-DT0000004147_20221002T074833Z_002_V010501_20241110T222710Z-SPECTRAL_IMAGE_SWIR.TIF",
    #     "metadata": REPO_ROOT
    #     / "test_data/enmap/Turkmenistan_20221002/20221002T074833/ENMAP01-____L1B-DT0000004147_20221002T074833Z_002_V010501_20241110T222710Z-METADATA.XML",
    #     "lut": LUT_PATH,
    #     "snr_reference": REPO_ROOT
    #     / "notebooks/outputs/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz",
    # },
    # "agadez_background_enmap": {
    #     "vnir": REPO_ROOT
    #     / "test_data/enmap/Agadez_Niger_20220712/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_VNIR.TIF",
    #     "swir": REPO_ROOT
    #     / "test_data/enmap/Agadez_Niger_20220712/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_SWIR.TIF",
    #     "metadata": REPO_ROOT
    #     / "test_data/enmap/Agadez_Niger_20220712/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-METADATA.XML",
    #     "lut": LUT_PATH,
    #     "snr_reference": REPO_ROOT
    #     / "notebooks/outputs/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz",
    # },
    "BuenosAires_20240112_enmap": {
        "vnir": REPO_ROOT
        / "case_studies_data/BuenosAires_20240112/EnMAP/ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010501_20241021T060427Z-SPECTRAL_IMAGE_VNIR.TIF",
        "swir": REPO_ROOT
        / "case_studies_data/BuenosAires_20240112/EnMAP/ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010501_20241021T060427Z-SPECTRAL_IMAGE_SWIR.TIF",
        "metadata": REPO_ROOT
        / "case_studies_data/BuenosAires_20240112/EnMAP/ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010501_20241021T060427Z-METADATA.XML",
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz",
    },
    "Turkmenistan_20240911_enmap": {
        "vnir": REPO_ROOT
        / "case_studies_data/Turkmenistan_20240911/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-SPECTRAL_IMAGE_VNIR.TIF",
        "swir": REPO_ROOT
        / "case_studies_data/Turkmenistan_20240911/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-SPECTRAL_IMAGE_SWIR.TIF",
        "metadata": REPO_ROOT
        / "case_studies_data/Turkmenistan_20240911/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-METADATA.XML",
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz",
    },
}

# mf_mode, k, suffix label
MF_RUNS: Iterable[tuple[str, int, str]] = (
    ("srf-column", 1, "srf-column_k1"),
    ("srf-column", 3, "srf-column_k3"),
    ("full-column", 1, "full-column"),
    ("advanced", 3, "advanced_k3"),
    ("jpl", 1, "jpl"),
)


def _ensure_paths_exist(case_name: str, cfg: dict[str, Path]) -> None:
    missing = [key for key, path in cfg.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{case_name}: missing required files for keys {missing}")


def run_case_study(case_name: str, cfg: dict[str, Path]) -> None:
    _ensure_paths_exist(case_name, cfg)
    for mf_mode, k, suffix in MF_RUNS:
        output_dir = cfg["metadata"].parent / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running %s (%s) into %s", case_name, suffix, output_dir)
        enmap_pipeline.ch4_detection_enmap(
            vnir_file=str(cfg["vnir"]),
            swir_file=str(cfg["swir"]),
            metadata_file=str(cfg["metadata"]),
            lut_file=str(cfg["lut"]),
            output_dir=str(output_dir),
            k=k,
            min_wavelength=WINDOW[0],
            max_wavelength=WINDOW[1],
            mf_mode=mf_mode,
            snr_reference_path=str(cfg["snr_reference"]),
            output_name_suffix=suffix,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for case_name, cfg in CASE_STUDIES.items():
        try:
            run_case_study(case_name, cfg)
        except Exception:
            logging.exception("Failed processing %s", case_name)


if __name__ == "__main__":
    main()
