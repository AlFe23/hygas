"""Batch runner for the Tanager case studies (auto-discovered).

Scans the default scene root for Tanager radiance/reflectance pairs
(`basic_radiance_hdf5__*.h5` + `basic_sr_hdf5__*.h5`), then executes every
matched-filter mode (srf-column k=1/k=3, full-column, advanced k=3, jpl) over
the 2100–2450 nm window. Outputs are written into mode-named subfolders inside
each scene directory, and filenames are suffixed with the MF mode.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT_HINT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from scripts.pipelines import tanager_pipeline


def _locate_repo_root(start: Path) -> Path:
    """Walk up until a directory containing `scripts` is found."""
    cursor = start.resolve()
    while cursor != cursor.parent:
        if (cursor / "scripts").exists():
            return cursor
        cursor = cursor.parent
    raise RuntimeError("Repository root not found from starting path.")


def _discover_scenes(scene_root: Path) -> list[dict[str, Path]]:
    """Return scene configs from subfolders containing Tanager radiance+SR files."""
    scenes: list[dict[str, Path]] = []
    for scene_dir in sorted(scene_root.glob("*")):
        if not scene_dir.is_dir():
            continue
        rad_files = sorted(scene_dir.glob("basic_radiance_hdf5__*.h5"))
        sr_files = sorted(scene_dir.glob("basic_sr_hdf5__*.h5"))
        if not rad_files or not sr_files:
            continue
        scenes.append({"name": scene_dir.name, "rad": rad_files[0], "sr": sr_files[0], "root": scene_dir})
    return scenes


REPO_ROOT = _locate_repo_root(Path(__file__).parent)
WINDOW = (2100.0, 2450.0)
LUT_PATH = next(
    (p for p in (REPO_ROOT.parent / "LUTs/CH4_lut.hdf5", REPO_ROOT / "LUTs/CH4_lut.hdf5") if p.exists()),
    REPO_ROOT.parent / "LUTs/CH4_lut.hdf5",
)
DEM_PATH = next(
    (p for p in (REPO_ROOT.parent / "DEM_1Km/srtm30plus_v11_land.nc", REPO_ROOT / "DEM_1Km/srtm30plus_v11_land.nc") if p.exists()),
    REPO_ROOT.parent / "DEM_1Km/srtm30plus_v11_land.nc",
)
SNR_REFERENCE = REPO_ROOT / "reference_snr" / "tanager" / "snr_reference_columnwise.npz"
SCENE_ROOT = REPO_ROOT / "test_data" / "tanager" / "GHG-plumes"

# mf_mode, k, suffix label
MF_RUNS: Iterable[tuple[str, int, str]] = (
    ("srf-column", 1, "srf-column_k1"),
    ("srf-column", 3, "srf-column_k3"),
    ("full-column", 1, "full-column"),
    ("advanced", 3, "advanced_k3"),
    ("jpl", 1, "jpl"),
)


def _ensure_assets_exist() -> None:
    missing = [key for key, path in {"lut": LUT_PATH, "dem": DEM_PATH, "snr": SNR_REFERENCE}.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")


def _ensure_scene_paths(scene_name: str, cfg: dict[str, Path]) -> None:
    missing = [key for key, path in cfg.items() if key in {"rad", "sr"} and not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene_name}: missing required files for keys {missing}")


def run_scene(scene_cfg: dict[str, Path]) -> None:
    _ensure_scene_paths(scene_cfg["name"], scene_cfg)
    for mf_mode, k, suffix in MF_RUNS:
        output_dir = scene_cfg["root"] / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running %s [%s] into %s", scene_cfg["name"], suffix, output_dir)
        tanager_pipeline.ch4_detection_tanager(
            radiance_file=str(scene_cfg["rad"]),
            sr_file=str(scene_cfg["sr"]),
            dem_file=str(DEM_PATH),
            lut_file=str(LUT_PATH),
            output_dir=str(output_dir),
            k=k,
            min_wavelength=WINDOW[0],
            max_wavelength=WINDOW[1],
            snr_reference_path=str(SNR_REFERENCE),
            mf_mode=mf_mode,
            output_name_suffix=suffix,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_assets_exist()
    scenes = _discover_scenes(SCENE_ROOT)
    if not scenes:
        logging.error("No Tanager scenes found under %s", SCENE_ROOT)
        return
    logging.info("Discovered %d scenes: %s", len(scenes), [s["name"] for s in scenes])
    for cfg in scenes:
        try:
            run_scene(cfg)
        except Exception:
            logging.exception("Failed processing %s", cfg["name"])


if __name__ == "__main__":
    main()
