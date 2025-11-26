"""Batch runner for the PRISMA notebook case studies.

This script mirrors `notebooks/matched_filter_demo_prisma.ipynb`: for each
predefined scene, it executes every matched-filter mode (srf-column k=1/k=3,
full-column, advanced k=3, jpl) over the 1500–2500 nm window. Outputs are written
into mode-named subfolders placed alongside the input L1/L2C files, and filenames
are suffixed with the MF mode for easy identification.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT_HINT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from scripts.pipelines import prisma_pipeline


def _locate_repo_root(start: Path) -> Path:
    """Walk up until a directory containing `scripts` is found."""
    cursor = start.resolve()
    while cursor != cursor.parent:
        if (cursor / "scripts").exists():
            return cursor
        cursor = cursor.parent
    raise RuntimeError("Repository root not found from starting path.")


REPO_ROOT = _locate_repo_root(Path(__file__).parent)
WINDOW = (1500.0, 2500.0)
LUT_PATH = next(
    (p for p in (REPO_ROOT.parent / "LUTs/CH4_lut.hdf5", REPO_ROOT / "LUTs/CH4_lut.hdf5") if p.exists()),
    REPO_ROOT.parent / "LUTs/CH4_lut.hdf5",
)
DEM_PATH = next(
    (p for p in (REPO_ROOT.parent / "DEM_1Km/srtm30plus_v11_land.nc", REPO_ROOT / "DEM_1Km/srtm30plus_v11_land.nc") if p.exists()),
    REPO_ROOT.parent / "DEM_1Km/srtm30plus_v11_land.nc",
)

PRISMA_SCENES = {
    "ekizak_plume_prisma": {
        "l1": REPO_ROOT
        / "test_data/prisma/Ekizak_Turkmenistan_20220912/20220912072502_20220912072506/PRS_L1_STD_OFFL_20220912072502_20220912072506_0001.zip",
        "l2c": REPO_ROOT
        / "test_data/prisma/Ekizak_Turkmenistan_20220912/20220912072502_20220912072506/PRS_L2C_STD_20220912072502_20220912072506_0001.zip",
        "dem": DEM_PATH,
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/prisma/20200401085313/snr_reference_columnwise.npz",
    },
    "northern_state_background_prisma": {
        "l1": REPO_ROOT
        / "test_data/prisma/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip",
        "l2c": REPO_ROOT
        / "test_data/prisma/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001.zip",
        "dem": DEM_PATH,
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/prisma/20200401085313/snr_reference_columnwise.npz",
    },
    "Turkmenistan_20240911_prisma": {
        "l1": REPO_ROOT
        / "case_studies_data/Turkmenistan_20240911/prisma/20240911071147/PRS_L1_STD_OFFL_20240911071147_20240911071151_0001.zip",
        "l2c": REPO_ROOT
        / "case_studies_data/Turkmenistan_20240911/prisma/20240911071147/PRS_L2C_STD_20240911071147_20240911071151_0001.zip",
        "dem": DEM_PATH,
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/prisma/20200401085313/snr_reference_columnwise.npz",
    },
    "Ehrenberg_validation_prisma": {
        "l1": REPO_ROOT
        / "case_studies_data/Ehrenberg/20211021182310/PRS_L1_STD_OFFL_20211021182310_20211021182315_0001.zip",
        "l2c": REPO_ROOT
        / "case_studies_data/Ehrenberg/20211021182310/PRS_L2C_STD_20211021182310_20211021182315_0001.zip",
        "dem": DEM_PATH,
        "lut": LUT_PATH,
        "snr_reference": REPO_ROOT
        / "notebooks/outputs/prisma/20200401085313/snr_reference_columnwise.npz",
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
        output_dir = Path(cfg["l1"]).parent / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running %s (%s) into %s", case_name, suffix, output_dir)
        prisma_pipeline.ch4_detection(
            L1_file=str(cfg["l1"]),
            L2C_file=str(cfg["l2c"]),
            dem_file=str(cfg["dem"]),
            lut_file=str(cfg["lut"]),
            output_dir=str(output_dir),
            min_wavelength=WINDOW[0],
            max_wavelength=WINDOW[1],
            k=k,
            mf_mode=mf_mode,
            snr_reference_path=str(cfg["snr_reference"]),
            output_name_suffix=suffix,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for case_name, cfg in PRISMA_SCENES.items():
        try:
            run_case_study(case_name, cfg)
        except Exception:
            logging.exception("Failed processing %s", case_name)


if __name__ == "__main__":
    main()
