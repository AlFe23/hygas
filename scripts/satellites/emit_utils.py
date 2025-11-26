"""Utilities to read EMIT Level-1B NetCDF products.

The public EMIT catalog distributes radiance/observation files as NetCDF4
containers. This helper loads the radiance cube (in uW/cm^2/sr/nm), extracts
the wavelength metadata, and optionally summarizes the observation geometry
layers for notebook diagnostics such as the SNR experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


DEFAULT_GEOMETRY_LABELS = {
    "Path length (sensor-to-ground in meters)": "Path length",
    "To-sensor azimuth (0 to 360 degrees CW from N)": "Sensor azimuth",
    "To-sensor zenith (0 to 90 degrees from zenith)": "Sensor zenith",
    "To-sun azimuth (0 to 360 degrees CW from N)": "Sun azimuth",
    "To-sun zenith (0 to 90 degrees from zenith)": "Sun zenith",
    "Solar phase (degrees between to-sensor and to-sun vectors in principal plane)": "Solar phase",
    "Slope (local surface slope as derived from DEM in degrees)": "Slope",
    "Aspect (local surface aspect 0 to 360 degrees clockwise from N)": "Aspect",
    "Cosine(i) (apparent local illumination factor based on DEM slope and aspect and to sun vector)": "Cos(i)",
    "UTC Time (decimal hours for mid-line pixels)": "UTC time",
    "Earth-sun distance (AU)": "Earth-sun distance",
}


def _decode_strings(values: Iterable) -> List[str]:
    decoded: List[str] = []
    for val in values:
        if isinstance(val, bytes):
            decoded.append(val.decode("utf-8", "ignore"))
        else:
            decoded.append(str(val))
    return decoded


def _summarize_band(data: np.ndarray, label: str) -> Optional[str]:
    valid = np.isfinite(data)
    if not np.any(valid):
        return None
    sample = data[valid]
    mean = float(np.mean(sample))
    median = float(np.median(sample))
    vmin = float(np.min(sample))
    vmax = float(np.max(sample))
    return f"{label}: mean={mean:.3f}, median={median:.3f}, min={vmin:.3f}, max={vmax:.3f}"


def _read_dataset(root: h5py.File, candidates: Sequence[str], keywords: Sequence[str]) -> np.ndarray:
    """Try a list of dataset paths, otherwise search by keyword."""

    for cand in candidates:
        if not cand:
            continue
        try:
            return root[cand][:]
        except KeyError:
            continue

    found_path: Optional[str] = None

    def _visitor(name, obj):
        nonlocal found_path
        if found_path is not None:
            return
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].lower()
            if any(key in base for key in keywords):
                found_path = name

    root.visititems(_visitor)
    if found_path is None:
        raise KeyError(f"Unable to find dataset with keywords {keywords}")

    return root[found_path][:]


def _geometry_summary(obs_path: Optional[str]) -> List[str]:
    if not obs_path:
        return []

    geometry_lines: List[str] = []
    try:
        with h5py.File(obs_path, "r") as obs_ds:
            obs_cube = _read_dataset(obs_ds, ["obs"], ["obs", "observation"])
            try:
                names_raw = _read_dataset(
                    obs_ds,
                    ["sensor_band_parameters/observation_bands"],
                    ["observation", "band"],
                )
                names = _decode_strings(names_raw)
            except KeyError:
                names = [f"Band {i+1}" for i in range(obs_cube.shape[-1])]

        for idx, raw_name in enumerate(names):
            label = DEFAULT_GEOMETRY_LABELS.get(raw_name, raw_name)
            summary = _summarize_band(obs_cube[..., idx], label)
            if summary:
                geometry_lines.append(summary)
    except Exception as exc:  # pragma: no cover - defensive guard
        geometry_lines.append(f"Failed to parse observation geometry: {exc}")

    if geometry_lines:
        geometry_lines.insert(0, "Geometry source: EMIT L1B OBS NetCDF")
    return geometry_lines


def _extract_scene_id(path: str) -> str:
    name = Path(path).stem
    if name:
        return name
    return "emit_scene"


def _identify_inputs(paths: Sequence[str]) -> Tuple[str, Optional[str]]:
    rad_path: Optional[str] = None
    obs_path: Optional[str] = None
    for path in paths:
        lower = path.lower()
        if "rad" in lower and lower.endswith(".nc"):
            rad_path = path
        elif "obs" in lower and lower.endswith(".nc"):
            obs_path = path

    if rad_path is None:
        raise ValueError("EMIT requires a radiance NetCDF input (filename contains 'RAD').")
    return rad_path, obs_path


def load_emit_scene(paths: Sequence[str]) -> Dict[str, object]:
    """Load EMIT L1B radiance/observation NetCDF files for the SNR pipeline."""

    rad_path, obs_path = _identify_inputs(paths)

    with h5py.File(rad_path, "r") as rad_ds:
        rad_cube = _read_dataset(rad_ds, ["radiance"], ["radiance", "rad"])
        wavelengths = np.asarray(
            _read_dataset(
                rad_ds,
                ["sensor_band_parameters/wavelengths"],
                ["wavelength"],
            ),
            dtype=float,
        )

    cube_brc = np.transpose(rad_cube, (2, 0, 1)).astype(np.float32)

    scene_id = _extract_scene_id(rad_path)
    metadata = {"radiance": rad_path}
    if obs_path:
        metadata["observation"] = obs_path

    geometry_lines = [f"Scene: {scene_id}"]
    geometry_lines.extend(_geometry_summary(obs_path))

    return {
        "cube": cube_brc,
        "wavelengths": wavelengths,
        "scene_id": scene_id,
        "metadata": metadata,
        "geometry_lines": geometry_lines,
        "reference_wavelengths": {"vnir_nm": None, "swir_nm": None},
    }
