# -*- coding: utf-8 -*-
"""
Lightweight helpers to inspect Planet Tanager HDF5 products (Basic/Ortho).
Dataset paths follow the September 2025 Planet Tanager Product Specification
(`product_spec_docs/tanager/Planet-UserDocumentation-Tanager.pdf`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import zipfile
import subprocess
import tempfile

import h5py
import imageio
import numpy as np
from osgeo import gdal, osr

# Canonical dataset paths from the product specification
TANAGER_TOA_RADIANCE_DATASET = "HDFEOS/SWATHS/HYP/Data Fields/toa_radiance"
TANAGER_SURFACE_REFLECTANCE_DATASET = "HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance"
TANAGER_GEOLOCATION_PATHS = {
    "latitude": "HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude",
    "longitude": "HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude",
    "time": "HDFEOS/SWATHS/HYP/Geolocation Fields/Time",
}
TANAGER_MASK_PATHS = {
    "beta_cloud_mask": "HDFEOS/SWATHS/HYP/Data Fields/beta_cloud_mask",
    "beta_cirrus_mask": "HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask",
    "nodata_pixels": "HDFEOS/SWATHS/HYP/Data Fields/nodata_pixels",
}
DEFAULT_RGB_WAVELENGTHS = (665.0, 565.0, 490.0)  # nm; matches Planet visual ortho bands


def _format_hdf_attr(value):
    """Return a human-readable representation for an HDF attribute value."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            value = repr(value)
    elif isinstance(value, np.ndarray):
        flat = value.flatten()
        if flat.size > 6:
            head = ", ".join(map(str, flat[:6]))
            value = f"[{head}, ...] (len={flat.size})"
        else:
            value = flat.tolist()
    return value


def describe_tanager_hdf_structure(filename: str, max_depth: int | None = None, include_attrs: bool = False) -> str:
    """
    Return a multi-line string describing the hierarchy of a Tanager HDF5 file.
    """

    lines = []

    with h5py.File(filename, "r") as f:
        def _recurse(name, obj, depth):
            if max_depth is not None and depth > max_depth:
                return

            indent = "  " * depth
            label = name.split("/")[-1] if name else "/"

            if isinstance(obj, h5py.Dataset):
                lines.append(f"{indent}- {label} [dataset] shape={obj.shape} dtype={obj.dtype}")
            else:  # Group
                lines.append(f"{indent}+ {label} [group]")

            if include_attrs and obj.attrs:
                for attr_key, attr_val in obj.attrs.items():
                    formatted = _format_hdf_attr(attr_val)
                    lines.append(f"{indent}  @{attr_key} = {formatted}")

            if isinstance(obj, h5py.Group):
                for key, child in obj.items():
                    child_name = f"{name}/{key}" if name else key
                    _recurse(child_name, child, depth + 1)

        _recurse("", f, 0)

    return "\n".join(lines)


def describe_tanager_hdf_object(
    filename: str,
    path: str,
    include_attrs: bool = False,
    preview: int | None = None,
    max_members: int = 30,
) -> str:
    """
    Return a detailed string describing a specific dataset or group in a Tanager HDF5.
    """

    def _slice_for_preview(shape, count):
        if not shape:  # scalar dataset
            return ()
        slices = [slice(0, min(count, shape[0]))]
        for dim in shape[1:]:
            slices.append(slice(0, min(1, dim)))
        return tuple(slices)

    normalized_path = path if path and path != "/" else "/"
    lines: list[str] = [f"Path: {normalized_path}"]

    with h5py.File(filename, "r") as f:
        if normalized_path == "/":
            obj = f["/"]
        else:
            if normalized_path not in f:
                raise KeyError(normalized_path)
            obj = f[normalized_path]

        if isinstance(obj, h5py.Dataset):
            lines.append("Type: dataset")
            lines.append(f"Shape: {obj.shape}")
            lines.append(f"Dtype: {obj.dtype}")
            if preview:
                slice_spec = _slice_for_preview(obj.shape, preview)
                data = np.asarray(obj[slice_spec]).reshape(-1)
                head = ", ".join(map(str, data[:preview]))
                lines.append(f"Preview ({min(preview, data.size)} values): [{head}]")
        elif isinstance(obj, h5py.Group):
            lines.append("Type: group")
            members: Iterable[str] = obj.keys()
            collected = []
            for idx, key in enumerate(members):
                if idx >= max_members:
                    collected.append(f"... ({len(obj) - max_members} more)")
                    break
                child = obj[key]
                kind = "dataset" if isinstance(child, h5py.Dataset) else "group"
                collected.append(f"- {key} ({kind})")
            if collected:
                lines.append("Members:")
                lines.extend(f"  {item}" for item in collected)
        else:
            lines.append(f"Type: {type(obj)}")

        if include_attrs and obj.attrs:
            lines.append("Attributes:")
            for attr_key, attr_val in obj.attrs.items():
                formatted = _format_hdf_attr(attr_val)
                lines.append(f"  @{attr_key} = {formatted}")

    return "\n".join(lines)


@dataclass
class TanagerCube:
    """Container for radiance/reflectance data and metadata."""

    data: np.ndarray  # (bands, rows, cols)
    wavelengths: np.ndarray | None
    fwhm: np.ndarray | None
    radiometric_coefficients: np.ndarray | None
    masks: dict[str, np.ndarray]
    geolocation: dict[str, np.ndarray]
    dataset_path: str


def load_tanager_cube(
    filename: str,
    dataset_path: str = TANAGER_TOA_RADIANCE_DATASET,
    band_indices: Sequence[int] | slice | None = None,
    load_masks: bool = True,
    load_geolocation: bool = False,
    dtype=np.float32,
) -> TanagerCube:
    """
    Load a Tanager radiance/reflectance cube plus optional masks and geolocation.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    dataset_path : str
        Dataset to load (defaults to TOA radiance).
    band_indices : sequence[int] | slice | None
        Optional band subset (applied on the leading Band dimension).
    load_masks : bool
        When True, beta masks and nodata pixels are loaded when present.
    load_geolocation : bool
        When True, latitude/longitude/time arrays are loaded when present.
    dtype : numpy dtype
        Cast data to this dtype (float32 by default) to reduce memory footprint.
    """

    with h5py.File(filename, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset not found in file: {dataset_path}")

        ds = f[dataset_path]
        data = np.asarray(ds[band_indices] if band_indices is not None else ds[:], dtype=dtype)

        def _maybe_array(attr_name: str):
            return np.asarray(ds.attrs[attr_name]) if attr_name in ds.attrs else None

        wavelengths = _maybe_array("wavelengths")
        fwhm = _maybe_array("fwhm")
        radiometric_coefficients = _maybe_array("applied_radiometric_coefficients")

        masks: dict[str, np.ndarray] = {}
        if load_masks:
            for key, path in TANAGER_MASK_PATHS.items():
                if path in f:
                    masks[key] = np.asarray(f[path])

        geolocation: dict[str, np.ndarray] = {}
        if load_geolocation:
            for key, path in TANAGER_GEOLOCATION_PATHS.items():
                if path in f:
                    geolocation[key] = np.asarray(f[path])

    return TanagerCube(
        data=data,
        wavelengths=wavelengths,
        fwhm=fwhm,
        radiometric_coefficients=radiometric_coefficients,
        masks=masks,
        geolocation=geolocation,
        dataset_path=dataset_path,
    )


def open_hdf(path: str, mode: str = "r") -> h5py.File:
    """Light wrapper for h5py.File to keep imports local to this module."""
    return h5py.File(path, mode)


def bounding_box(cube: TanagerCube) -> tuple[float, float, float, float]:
    """Return (min_lon, max_lon, min_lat, max_lat) from cube geolocation."""
    lat = cube.geolocation.get("latitude")
    lon = cube.geolocation.get("longitude")
    if lat is None or lon is None:
        raise ValueError("Cube geolocation missing latitude/longitude.")
    return (
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    )


def _closest_band_indices(wavelengths: np.ndarray, targets_nm: Sequence[float]) -> list[int]:
    """
    Return band indices closest to the requested wavelengths (nm).
    """

    if wavelengths is None:
        raise ValueError("Dataset is missing 'wavelengths' attribute; cannot auto-pick bands.")
    idxs = []
    for target in targets_nm:
        idx = int(np.abs(wavelengths - target).argmin())
        idxs.append(idx)
    return idxs


def quicklook_rgb(
    cube: TanagerCube,
    rgb_wavelengths: Sequence[float] = DEFAULT_RGB_WAVELENGTHS,
    stretch: tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    max_size: int | None = None,
    mask_name: str | None = "nodata_pixels",
) -> np.ndarray:
    """
    Build a simple RGB array (values in [0, 1]) from a Tanager cube.
    """

    rgb_indices = _closest_band_indices(cube.wavelengths, rgb_wavelengths)
    rgb = np.stack([cube.data[idx] for idx in rgb_indices], axis=-1)  # (rows, cols, 3)

    mask = cube.masks.get(mask_name) if (mask_name and cube.masks) else None
    if mask is not None:
        rgb = np.where(mask[..., None] != 0, np.nan, rgb)

    def _scale_band(band: np.ndarray) -> np.ndarray:
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            return np.zeros_like(band, dtype=np.float32)
        lo, hi = np.percentile(finite, stretch)
        if hi <= lo:
            return np.zeros_like(band, dtype=np.float32)
        scaled = (band - lo) / (hi - lo)
        return np.clip(scaled, 0.0, 1.0)

    rgb_scaled = np.stack([_scale_band(rgb[..., i]) for i in range(3)], axis=-1)

    if gamma != 1.0:
        rgb_scaled = np.power(rgb_scaled, 1.0 / gamma)

    if max_size:
        rows, cols, _ = rgb_scaled.shape
        step = max(1, int(math.ceil(max(rows, cols) / max_size)))
        if step > 1:
            rgb_scaled = rgb_scaled[::step, ::step]

    return rgb_scaled.astype(np.float32)


def save_rgb_png(rgb: np.ndarray, path: str) -> None:
    """Persist an RGB array in [0, 1] to an 8-bit PNG."""
    rgb_8bit = (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)
    imageio.imwrite(path, rgb_8bit)


def summarize_cube(cube: TanagerCube) -> str:
    """Return a short textual summary of the loaded cube."""
    lines = [
        f"Dataset: {cube.dataset_path}",
        f"Shape: {cube.data.shape} (bands, rows, cols)",
        f"Wavelengths: {'present' if cube.wavelengths is not None else 'missing'}",
        f"FWHM: {'present' if cube.fwhm is not None else 'missing'}",
        f"Radiometric coefficients: {'present' if cube.radiometric_coefficients is not None else 'missing'}",
        f"Masks loaded: {', '.join(cube.masks) if cube.masks else 'none'}",
        f"Geolocation loaded: {', '.join(cube.geolocation) if cube.geolocation else 'none'}",
    ]
    return "\n".join(lines)


def extract_hdf_from_zip(zip_path: str, output_dir: str) -> str | None:
    """
    Extract the first HDF/HDF5 file from a ZIP archive and return its path.
    """

    suffixes = {".h5", ".hdf", ".hdf5"}
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [name for name in zf.namelist() if Path(name).suffix.lower() in suffixes]
        if not candidates:
            return None
        member = candidates[0]
        zf.extract(member, output_dir)
        extracted = Path(output_dir) / member
        # Flatten nested directories to keep a stable path
        flat_path = Path(output_dir) / extracted.name
        if extracted != flat_path:
            extracted.rename(flat_path)
        return str(flat_path)


def _write_temp_geolocated(dst_path: str, data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    """
    Write data (rows, cols [,bands]) to GeoTIFF using lat/lon geolocation arrays (same shape).
    """

    data = data.astype(np.float32)
    rows, cols = data.shape[:2]
    bands = 1 if data.ndim == 2 else data.shape[2]

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / "temp_data.tif"
        vrt_file = Path(tmpdir) / "temp_data.vrt"
        lat_file = Path(tmpdir) / "lat.tif"
        lon_file = Path(tmpdir) / "lon.tif"

        driver = gdal.GetDriverByName("GTiff")
        lat_ds = driver.Create(str(lat_file), cols, rows, 1, gdal.GDT_Float32)
        lon_ds = driver.Create(str(lon_file), cols, rows, 1, gdal.GDT_Float32)
        lat_ds.GetRasterBand(1).WriteArray(lat.astype(np.float32))
        lon_ds.GetRasterBand(1).WriteArray(lon.astype(np.float32))
        lat_ds = None
        lon_ds = None

        ds = driver.Create(str(temp_file), cols, rows, bands, gdal.GDT_Float32)
        if bands == 1:
            ds.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(bands):
                ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])
        ds.FlushCache()
        ds = None

        vrt_options = gdal.TranslateOptions(format="VRT")
        gdal.Translate(str(vrt_file), str(temp_file), options=vrt_options)

        vrt_ds = gdal.Open(str(vrt_file), gdal.GA_Update)
        vrt_ds.SetMetadata(
            {
                "X_DATASET": str(lon_file),
                "X_BAND": "1",
                "Y_DATASET": str(lat_file),
                "Y_BAND": "1",
                "PIXEL_OFFSET": "0",
                "LINE_OFFSET": "0",
                "PIXEL_STEP": "1",
                "LINE_STEP": "1",
            },
            "GEOLOCATION",
        )
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        vrt_ds.SetProjection(srs.ExportToWkt())
        vrt_ds = None

        subprocess.run(
            ["gdalwarp", "-geoloc", "-t_srs", "EPSG:4326", str(vrt_file), dst_path],
            check=True,
        )


def save_with_geolocation_single_band(data: np.ndarray, path: str, cube: TanagerCube) -> None:
    """Save a single-band array (rows, cols) using cube geolocation."""
    lat = cube.geolocation.get("latitude")
    lon = cube.geolocation.get("longitude")
    if lat is None or lon is None:
        raise ValueError("Latitude/Longitude not loaded in cube.geolocation.")
    _write_temp_geolocated(path, data, lat, lon)


def save_with_geolocation_multichannel(data: np.ndarray, path: str, cube: TanagerCube) -> None:
    """Save a multichannel array (rows, cols, bands) using cube geolocation."""
    lat = cube.geolocation.get("latitude")
    lon = cube.geolocation.get("longitude")
    if lat is None or lon is None:
        raise ValueError("Latitude/Longitude not loaded in cube.geolocation.")
    _write_temp_geolocated(path, data, lat, lon)
