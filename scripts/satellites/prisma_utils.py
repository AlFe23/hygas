# -*- coding: utf-8 -*-
"""
PRISMA-specific utilities: file readers, metadata access, georeferencing, and
batch helpers. Functions and comments are kept close to the original monolithic
script to remain interpretable for existing users.
"""

import os
import re
import zipfile
import subprocess
from typing import Iterable
from datetime import datetime

import h5py
import numpy as np
from osgeo import gdal, osr


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


def prismaL2C_WV_read(filename):

    # Open the HDF5 file
    with h5py.File(filename, "r") as f:
        # Read the Water vapor Mask from L2C PRISMA data
        PRS_L2C_WVM = f["HDFEOS/SWATHS/PRS_L2C_WVM/Data Fields/WVM_Map"][:]
        latitude_WVM = f["HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Latitude"][:]
        longitude_WVM = f["HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Longitude"][:]

        # Read scale factors to transform DN (uint16) into WV physical values (g/cm2)
        L2ScaleWVMMin = f.attrs["L2ScaleWVMMin"]
        L2ScaleWVMMax = f.attrs["L2ScaleWVMMax"]

    # Convert DN to WV in g/cm^2    (pg.213 product spec doc)
    WVM_unit = L2ScaleWVMMin + PRS_L2C_WVM * (L2ScaleWVMMax - L2ScaleWVMMin) / 65535
    meanWV = np.mean(WVM_unit)
    # Print the mean Water Vapor value
    print("Mean Water Vapor (g/cm^2):", meanWV)

    return meanWV, PRS_L2C_WVM, latitude_WVM, longitude_WVM


def prismaL1_SZA_read(filename):

    # Open the HDF5 file
    with h5py.File(filename, "r") as f:

        # Read SZA in degrees
        SZA = f.attrs["Sun_zenith_angle"]

    # Print the Sun Zenith Angle
    print("Sun Zenith Angle (degrees):", SZA)

    return SZA


def describe_prisma_hdf_structure(filename, max_depth=None, include_attrs=False):
    """
    Return a multi-line string describing the hierarchy of a PRISMA HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the PRISMA HDF5 file (L1 or L2C).
    max_depth : int | None
        Optional maximum depth to traverse (root depth=0). ``None`` means no limit.
    include_attrs : bool
        When True, include dataset/group attributes in the report (truncated).
    """

    lines = []

    with h5py.File(filename, "r") as f:
        def _recurse(name, obj, depth):
            if max_depth is not None and depth > max_depth:
                return

            indent = "  " * depth
            label = name.split("/")[-1] if name else "/"

            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                lines.append(f"{indent}- {label} [dataset] shape={shape} dtype={dtype}")
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


def describe_prisma_hdf_object(
    filename,
    path,
    include_attrs=False,
    preview=None,
    max_members=30,
):
    """
    Return a detailed string describing a specific dataset or group in a PRISMA HDF5.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    path : str
        Target dataset or group path (use "/" for root).
    include_attrs : bool
        When True, include object attributes in the output.
    preview : int | None
        For datasets, show up to N flattened values from the leading block.
    max_members : int
        When inspecting a group, list at most this many immediate children.
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


def _angle_stats(values, valid_range=None):
    """Return descriptive statistics for angle arrays in degrees."""

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if valid_range is not None:
        lo, hi = valid_range
        arr = arr[(arr >= lo) & (arr <= hi)]
    if arr.size == 0:
        return None

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def prisma_l2c_geometry_summary(filename):
    """
    Summarize PRISMA L2C geometric fields (observing/sun angles).

    Returns a dictionary with dataset statistics and scene-level sun angles.
    """

    dataset_map = {
        "solar_zenith": {
            "path": "HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle",
            "label": "Solar zenith angle",
            "valid_range": (0.0, 90.0),
        },
        "view_zenith": {
            "path": "HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle",
            "label": "Observing angle (view zenith)",
            "valid_range": (0.0, 90.0),
        },
        "relative_azimuth": {
            "path": "HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle",
            "label": "Relative azimuth angle",
            "valid_range": (0.0, 180.0),
        },
    }

    def _attr_float(attrs, key):
        if key not in attrs:
            return None
        try:
            return float(attrs[key])
        except Exception:
            return None

    summary = {
        "datasets": {},
        "sun_angles": {
            "zenith_deg": None,
            "azimuth_deg": None,
        },
    }
    data_arrays = {}

    with h5py.File(filename, "r") as f:
        summary["sun_angles"]["zenith_deg"] = _attr_float(f.attrs, "Sun_zenith_angle")
        summary["sun_angles"]["azimuth_deg"] = _attr_float(f.attrs, "Sun_azimuth_angle")

        for key, spec in dataset_map.items():
            path = spec["path"]
            if path not in f:
                continue
            data = f[path][:]
            data_arrays[key] = data
            stats = _angle_stats(data, spec.get("valid_range"))
            if stats is None:
                continue
            summary["datasets"][key] = {
                "label": spec["label"],
                "path": path,
                "stats": stats,
            }

    solar_arr = data_arrays.get("solar_zenith")
    view_arr = data_arrays.get("view_zenith")
    if solar_arr is not None and view_arr is not None:
        rel_zenith = np.asarray(solar_arr, dtype=np.float64) - np.asarray(view_arr, dtype=np.float64)
        rel_stats = _angle_stats(rel_zenith)
        if rel_stats is not None:
            summary["relative_zenith_stats"] = rel_stats

    rel_az_arr = data_arrays.get("relative_azimuth")
    if rel_az_arr is not None:
        rel_stats = _angle_stats(rel_az_arr, (0.0, 180.0))
        if rel_stats is not None:
            summary["relative_azimuth_stats"] = rel_stats

    # Approximate relative zenith using means if stats present but difference not computed
    if "relative_zenith_stats" not in summary and {"solar_zenith", "view_zenith"}.issubset(summary["datasets"]):
        solar_stats = summary["datasets"]["solar_zenith"]["stats"]
        view_stats = summary["datasets"]["view_zenith"]["stats"]
        summary["relative_zenith"] = {
            "mean": solar_stats["mean"] - view_stats["mean"],
            "median": solar_stats["median"] - view_stats["median"],
        }

    if "relative_azimuth_stats" not in summary and "relative_azimuth" in summary["datasets"]:
        rel_stats = summary["datasets"]["relative_azimuth"]["stats"]
        summary["relative_azimuth_summary"] = {
            "mean": rel_stats["mean"],
            "median": rel_stats["median"],
        }

    return summary


def prismaL2C_bbox_read(filename):
    # Open the HDF5 file
    with h5py.File(filename, "r") as f:
        # Read the geolocation fields
        latitude_WVM = f["HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Latitude"][:]
        longitude_WVM = f["HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Longitude"][:]

    # Calculate bounding box
    min_lat = np.min(latitude_WVM)
    max_lat = np.max(latitude_WVM)
    min_lon = np.min(longitude_WVM)
    max_lon = np.max(longitude_WVM)

    return (min_lon, max_lon, min_lat, max_lat)


def prisma_read(filename):

    # Open the HDF5 file
    with h5py.File(filename, "r") as f:
        # Read the VNIR and SWIR spectral data
        vnir_cube_DN = f["HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][:]
        swir_cube_DN = f["HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][:]

        # Read central wavelengths and FWHM for VNIR and SWIR bands
        cw_vnir = f["KDP_AUX/Cw_Vnir_Matrix"][:]
        fwhm_vnir = f["KDP_AUX/Fwhm_Vnir_Matrix"][:]
        cw_swir = f["KDP_AUX/Cw_Swir_Matrix"][:]
        fwhm_swir = f["KDP_AUX/Fwhm_Swir_Matrix"][:]

        # Read scale factor and offset to transform DN to radiance physical value
        offset_swir = f.attrs["Offset_Swir"]
        scaleFactor_swir = f.attrs["ScaleFactor_Swir"]
        offset_vnir = f.attrs["Offset_Vnir"]
        scaleFactor_vnir = f.attrs["ScaleFactor_Vnir"]

        # Read geolocation fields
        latitude_vnir = f["HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:]
        longitude_vnir = f["HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:]
        latitude_swir = f["HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR"][:]
        longitude_swir = f["HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR"][:]

    # Convert DN to radiance for VNIR and SWIR data #[W/(str*um*m^2)]
    swir_cube_rads = (swir_cube_DN / scaleFactor_swir) - offset_swir
    vnir_cube_rads = (vnir_cube_DN / scaleFactor_vnir) - offset_vnir

    # Convert PRISMA radiance from [W/(str*um*m^2)] to [μW*cm-2*nm-1*sr-1] in order to meet original AVIRIS radiance unit (i.e. the unit used in MODTRAN6 to build the LUT)
    swir_cube_rads = swir_cube_rads * 0.1
    vnir_cube_rads = vnir_cube_rads * 0.1

    # Convert from BIL to BIP format ( BIL = M-3-N ; BIP = 3-M-N ; BSQ = M-N-3 )
    vnir_cube_bip = np.transpose(vnir_cube_rads, (0, 2, 1))
    swir_cube_bip = np.transpose(swir_cube_rads, (0, 2, 1))

    # Rotate 270 degrees counterclockwise (equivalent to 90 degrees clockwise) and flip horizontally
    vnir_cube_bip = np.rot90(vnir_cube_bip, k=-1, axes=(0, 1))  # Rotate 270 degrees counterclockwise
    swir_cube_bip = np.rot90(swir_cube_bip, k=-1, axes=(0, 1))  # Rotate 270 degrees counterclockwise

    # Helper: slice CW/FWHM matrices to the spectral span that matches the cube,
    # then drop columns with CW=0 (as SNAP reader does), and reorder to ascending wavelength.
    def _slice_to_bands(arr, n_bands):
        col_max = np.nanmax(arr, axis=0)
        nz = np.where(col_max > 0)[0]
        if nz.size == 0:
            return arr[:, :n_bands]
        first, last = int(nz[0]), int(nz[-1])
        span = n_bands
        start = max(0, first - max(0, span - (last - first + 1)))
        stop = start + span
        return arr[:, start:stop]

    def _clean_cube(cube, cw_mat, fwhm_mat):
        cw_slice = _slice_to_bands(cw_mat, cube.shape[2])
        fwhm_slice = _slice_to_bands(fwhm_mat, cube.shape[2])
        mask = np.nanmax(cw_slice, axis=0) > 0
        cw_slice = cw_slice[:, mask]
        fwhm_slice = fwhm_slice[:, mask]
        cube = cube[:, :, mask]
        cw_vec = np.nanmean(cw_slice, axis=0)
        if cw_vec.size > 1 and cw_vec[0] > cw_vec[-1]:
            cw_slice = cw_slice[:, ::-1]
            fwhm_slice = fwhm_slice[:, ::-1]
            cube = cube[:, :, ::-1]
        return cube, cw_slice, fwhm_slice

    VNIR_cube_clean, cw_vnir_clean, fwhm_vnir_clean = _clean_cube(vnir_cube_bip, cw_vnir, fwhm_vnir)
    SWIR_cube_clean, cw_swir_clean, fwhm_swir_clean = _clean_cube(swir_cube_bip, cw_swir, fwhm_swir)

    # Concatenate in ascending wavelength order (VNIR then SWIR)
    concatenated_cube = np.concatenate((VNIR_cube_clean, SWIR_cube_clean), axis=2)
    concatenated_cw = np.concatenate((cw_vnir_clean, cw_swir_clean), axis=1)
    concatenated_fwhm = np.concatenate((fwhm_vnir_clean, fwhm_swir_clean), axis=1)

    # Print the RGB image
    cw_vector = np.nanmean(concatenated_cw, axis=0)

    def _closest_idx(target_nm):
        return int(np.nanargmin(np.abs(cw_vector - target_nm)))

    red_channel = concatenated_cube[:, :, _closest_idx(650.0)]
    green_channel = concatenated_cube[:, :, _closest_idx(560.0)]
    blue_channel = concatenated_cube[:, :, _closest_idx(490.0)]

    # Normalize each channel to the range [0, 1]
    red_normalized = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
    green_normalized = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
    blue_normalized = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())

    # Combine the channels into an RGB image
    rgb_image = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)

    return (
        concatenated_cube,
        concatenated_cw,
        concatenated_fwhm,
        rgb_image,
        vnir_cube_bip,
        swir_cube_bip,
        latitude_vnir,
        longitude_vnir,
        latitude_swir,
        longitude_swir,
    )


def save_as_geotiff_multichannel(data, output_file, latitude_vnir, longitude_vnir):
    """
    Save a multichannel numpy array as a GeoTIFF with EPSG:4326 projection and enhanced metadata.

    Parameters:
    data (numpy array): Multichannel data array (rows x cols x bands).
    output_file (str): Path to the output GeoTIFF file.
    latitude_vnir (numpy array): Latitude geolocation array.
    longitude_vnir (numpy array): Longitude geolocation array.
    """
    # Ensure the data is in float32 format
    data_float32 = data.astype(np.float32)

    # Revert the rotation applied in `prisma_read`
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))

    # Temporary files to hold intermediate GeoTIFF and VRT
    temp_file = "temp_output.tif"
    vrt_file = "temp_output.vrt"
    lat_file = "latitude.tif"
    lon_file = "longitude.tif"

    # Create temporary files for latitude and longitude
    driver = gdal.GetDriverByName("GTiff")
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)

    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))

    lat_ds = None
    lon_ds = None

    # Create a temporary GeoTIFF file without detailed geotransform and projection
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], data.shape[2], gdal.GDT_Float32)

    # Write multichannel data to each band
    for i in range(data.shape[2]):
        dataset.GetRasterBand(i + 1).WriteArray(data_float32[:, :, i])

    # Save and close the dataset
    dataset.FlushCache()
    dataset = None

    # Create the VRT file that uses latitude and longitude for georeferencing
    vrt_options = gdal.TranslateOptions(format="VRT")
    gdal.Translate(vrt_file, temp_file, options=vrt_options)

    # Open the VRT file and set geolocation metadata
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata(
        {
            "X_DATASET": lon_file,
            "X_BAND": "1",
            "Y_DATASET": lat_file,
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
        },
        "GEOLOCATION",
    )

    # Explicitly set the CRS as EPSG:4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())

    # Metadata statement
    description = (
        "This product has been generated by Alvise Ferrari for School of Aerospace Engineering, "
        "La Sapienza, under terms of license of CLEAR-UP, a project funded by the Italian Space Agency. "
        "The dissemination of this product is closely linked to the agreements established under the CLEAR-UP project. "
        "The authors of the code by which the product was generated cannot be held responsible for any improper use or dissemination of this product."
    )
    vrt_ds.SetMetadataItem("DESCRIPTION", description)
    vrt_ds = None

    # Use gdalwarp to finalize the projection and georeferencing
    subprocess.run(
        [
            "gdalwarp",
            "-geoloc",
            "-t_srs",
            "EPSG:4326",  # Ensure CRS is explicitly set
            vrt_file,
            output_file,
        ],
        check=True,
    )

    # Remove temporary files
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)


def save_as_geotiff_single_band(data, output_file, latitude_vnir, longitude_vnir):
    """
    Save a single-band numpy array as a GeoTIFF with EPSG:4326 projection and enhanced metadata.

    Parameters:
    data (numpy array): Single-band data array.
    output_file (str): Path to the output GeoTIFF file.
    latitude_vnir (numpy array): Latitude geolocation array.
    longitude_vnir (numpy array): Longitude geolocation array.
    """
    # Ensure the data is in float32 format
    data_float32 = data.astype(np.float32)

    # Revert the rotation applied in `prisma_read`
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))

    # Temporary files to hold intermediate GeoTIFF and VRT
    temp_file = "temp_output.tif"
    vrt_file = "temp_output.vrt"
    lat_file = "latitude.tif"
    lon_file = "longitude.tif"

    # Create temporary files for latitude and longitude
    driver = gdal.GetDriverByName("GTiff")
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)

    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))

    lat_ds = None
    lon_ds = None

    # Create a temporary GeoTIFF file without detailed geotransform and projection
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(data_float32)

    # Save and close the dataset
    dataset.FlushCache()
    dataset = None

    # Create the VRT file that uses latitude and longitude for georeferencing
    vrt_options = gdal.TranslateOptions(format="VRT")
    gdal.Translate(vrt_file, temp_file, options=vrt_options)

    # Open the VRT file and set geolocation metadata
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata(
        {
            "X_DATASET": lon_file,
            "X_BAND": "1",
            "Y_DATASET": lat_file,
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
        },
        "GEOLOCATION",
    )

    # Explicitly set the CRS as EPSG:4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())

    # Metadata statement
    description = (
        "This product has been generated by Alvise Ferrari for School of Aerospace Engineering, "
        "La Sapienza, under terms of license of CLEAR-UP, a project funded by the Italian Space Agency. "
        "The dissemination of this product is closely linked to the agreements established under the CLEAR-UP project. "
        "The authors of the code by which the product was generated cannot be held responsible for any improper use or dissemination of this product."
    )
    vrt_ds.SetMetadataItem("DESCRIPTION", description)
    vrt_ds = None

    # Use gdalwarp to finalize the projection and georeferencing
    subprocess.run(
        [
            "gdalwarp",
            "-geoloc",
            "-t_srs",
            "EPSG:4326",  # Ensure CRS is explicitly set
            vrt_file,
            output_file,
        ],
        check=True,
    )

    # Remove temporary files
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)


def extract_he5_from_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".he5"):
                zip_ref.extract(file, extract_to)
                return os.path.join(extract_to, file)
    return None


def get_date_from_filename(filename):
    match = re.search(r"(\d{8}\d{6})", filename)
    return match.group(1) if match else None
