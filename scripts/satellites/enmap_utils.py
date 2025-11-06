# -*- coding: utf-8 -*-
"""
EnMAP-specific utilities shared across the matched-filter pipelines and the
analysis scripts (smile/SNR). This module now gathers all generic helpers so
callers can avoid re-implementing file discovery, cube reading, metadata
parsing, and DN→radiance conversion.
"""

import glob
import os
import re
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


################################################################################
# Generic XML helpers
################################################################################


def strip_namespaces(root):
    """Remove namespaces so downstream find() calls can use simple tag names."""

    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _safe_float(text):
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        try:
            return float(stripped.replace(",", "."))
        except ValueError:
            return None


################################################################################
# File discovery / GDAL readers
################################################################################


def find_enmap_files(in_dir):
    """Locate VNIR, SWIR GeoTIFFs and METADATA.XML in a folder."""

    vnir = sorted(glob.glob(os.path.join(in_dir, "*SPECTRAL_IMAGE_VNIR*.TIF"))) or sorted(
        glob.glob(os.path.join(in_dir, "*VNIR*.TIF"))
    )
    swir = sorted(glob.glob(os.path.join(in_dir, "*SPECTRAL_IMAGE_SWIR*.TIF"))) or sorted(
        glob.glob(os.path.join(in_dir, "*SWIR*.TIF"))
    )
    xml = sorted(glob.glob(os.path.join(in_dir, "*METADATA.XML")))

    if not vnir or not swir or not xml:
        raise FileNotFoundError("VNIR/SWIR TIF or METADATA.XML not found in directory")

    return vnir[0], swir[0], xml[0]


def read_cube_gdal(path_tif):
    """Read multi-band GeoTIFF into (bands, rows, cols) float32 array of DN."""

    ds = gdal.Open(path_tif, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open {path_tif}")

    bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    arr = np.empty((bands, rows, cols), dtype=np.float32)
    for b in range(1, bands + 1):
        arr[b - 1] = ds.GetRasterBand(b).ReadAsArray().astype(np.float32)
    ds = None
    return arr


################################################################################
# Metadata parsing & radiance conversion
################################################################################


def parse_metadata_vnir_swir(xml_path):
    """Parse METADATA.XML and return VNIR and SWIR band lists with local indices."""

    root = ET.parse(xml_path).getroot()
    strip_namespaces(root)

    global_bands = []
    for bn in root.findall(".//bandCharacterisation/bandID"):
        gid = int(bn.attrib["number"])
        global_bands.append(
            {
                "global_id": gid,
                "cw_nm": float(bn.findtext("wavelengthCenterOfBand")),
                "fwhm_nm": float(bn.findtext("FWHMOfBand")),
                "gain": float(bn.findtext("GainOfBand")),
                "offset": float(bn.findtext("OffsetOfBand")),
            }
        )
    global_bands.sort(key=lambda d: d["global_id"])

    smile = root.find(".//smileCorrection")
    if smile is None:
        raise RuntimeError("smileCorrection not found in metadata.")

    vnir_smile = smile.find("VNIR")
    swir_smile = smile.find("SWIR")
    if vnir_smile is None or swir_smile is None:
        raise RuntimeError("Expected <smileCorrection><VNIR> and <SWIR> subsections.")

    def _read_smile_section(sec):
        coeffs_by_local = {}
        wl_by_local = {}
        for bn in sec.findall(".//bandID"):
            local_idx = int(bn.attrib["number"])
            coeffs = []
            for k in range(5):
                txt = bn.findtext(f"coeff{k}")
                if txt is None:
                    coeffs = []
                    break
                coeffs.append(float(txt))
            if len(coeffs) == 5:
                coeffs_by_local[local_idx] = np.array(coeffs, dtype=float)
            wtxt = bn.findtext("wavelength")
            if wtxt is not None:
                wl_by_local[local_idx] = float(wtxt)
        return coeffs_by_local, wl_by_local

    vnir_coeffs_by_local, vnir_wl_smile = _read_smile_section(vnir_smile)
    swir_coeffs_by_local, swir_wl_smile = _read_smile_section(swir_smile)

    Nvnir = len(vnir_coeffs_by_local)
    Nswir = len(swir_coeffs_by_local)
    if Nvnir + Nswir != len(global_bands):
        print(
            f"[w] NVNIR({Nvnir}) + NSWIR({Nswir}) != Ntotal({len(global_bands)}). Proceeding with slice by counts."
        )

    vnir_global = global_bands[:Nvnir]
    swir_global = global_bands[Nvnir : Nvnir + Nswir]

    vnir_meta = []
    for i, gb in enumerate(vnir_global, start=1):
        meta = dict(gb)
        meta["smile_coeffs"] = vnir_coeffs_by_local.get(i)
        meta["local_idx"] = i
        vnir_meta.append(meta)

    swir_meta = []
    for i, gb in enumerate(swir_global, start=1):
        meta = dict(gb)
        meta["smile_coeffs"] = swir_coeffs_by_local.get(i)
        meta["local_idx"] = i
        swir_meta.append(meta)

    def _check_alignment(meta_list, wl_dict, label):
        diffs = []
        for m in meta_list:
            li = m["local_idx"]
            if li in wl_dict:
                diffs.append(abs(m["cw_nm"] - wl_dict[li]))
        if diffs:
            dmax = max(diffs)
            if dmax > 0.5:
                print(f"[w] Max |CW_meta - wavelength_smile| in {label}: {dmax:.3f} nm")

    _check_alignment(vnir_meta, vnir_wl_smile, "VNIR")
    _check_alignment(swir_meta, swir_wl_smile, "SWIR")

    return vnir_meta, swir_meta


def dn_to_radiance(dn_cube, meta_list):
    """Radiance = gain*DN + offset per band for VNIR or SWIR cubes."""

    nb = dn_cube.shape[0]
    if nb != len(meta_list):
        print(f"[w] DN bands={nb} vs metadata={len(meta_list)}")
    nb = min(nb, len(meta_list))
    rad = np.empty_like(dn_cube[:nb], dtype=np.float32)
    for i in range(nb):
        g = meta_list[i]["gain"]
        o = meta_list[i]["offset"]
        rad[i] = g * dn_cube[i] + o
    return rad


def enmap_read(vnir_file, swir_file, metadata_file):
    """
    Reads EnMAP VNIR and SWIR data cubes and associated metadata.

    Parameters:
    - vnir_file: Path to the VNIR GeoTIFF file.
    - swir_file: Path to the SWIR GeoTIFF file.
    - metadata_file: Path to the XML metadata file.

    Returns:
    - concatenated_cube: Combined VNIR and SWIR radiance data cube in BIP format.
    - concatenated_cw: Central wavelengths for each band.
    - concatenated_fwhm: FWHM for each band.
    - rgb_image: RGB image created from the radiance data.
    - latitude: Latitude array (if available).
    - longitude: Longitude array (if available).
    """
    # Read DN cubes (bands, rows, cols)
    vnir_cube_dn = read_cube_gdal(vnir_file)
    swir_cube_dn = read_cube_gdal(swir_file)

    vnir_meta, swir_meta = parse_metadata_vnir_swir(metadata_file)

    # Convert DN -> radiance using shared helper (still [B,R,C])
    vnir_rad = dn_to_radiance(vnir_cube_dn, vnir_meta)
    swir_rad = dn_to_radiance(swir_cube_dn, swir_meta)

    # Build metadata table for downstream use (global ordering)
    band_info = [
        {
            "number": meta["global_id"],
            "wavelength_center": meta["cw_nm"],
            "fwhm": meta["fwhm_nm"],
            "gain": meta["gain"],
            "offset": meta["offset"],
        }
        for meta in (*vnir_meta, *swir_meta)
    ]

    # Ensure the number of bands matches the data cubes
    num_vnir_bands = vnir_rad.shape[0]
    num_swir_bands = swir_rad.shape[0]
    total_bands = num_vnir_bands + num_swir_bands

    if len(band_info) != total_bands:
        print(
            f"Warning: Number of bands in metadata ({len(band_info)}) does not match total bands in data cubes ({total_bands})"
        )

    # Reorder cubes to (rows, cols, bands)
    vnir_radiance = np.transpose(vnir_rad, (1, 2, 0)).astype(np.float32)
    swir_radiance = np.transpose(swir_rad, (1, 2, 0)).astype(np.float32)

    # Convert radiance units from [W/(sr*nm*m^2)] to [μW/(sr*nm*cm^2)]
    vnir_radiance *= 1e2
    swir_radiance *= 1e2

    # Concatenate VNIR and SWIR radiance cubes
    concatenated_cube = np.concatenate((vnir_radiance, swir_radiance), axis=2)

    # Collect CW and FWHM
    concatenated_cw = np.array([band["wavelength_center"] for band in band_info])
    concatenated_fwhm = np.array([band["fwhm"] for band in band_info])

    # Create an RGB image for visualization (bands near 650/550/450 nm)
    red_wavelength = 650
    green_wavelength = 550
    blue_wavelength = 450

    red_band_idx = np.argmin(np.abs(concatenated_cw - red_wavelength))
    green_band_idx = np.argmin(np.abs(concatenated_cw - green_wavelength))
    blue_band_idx = np.argmin(np.abs(concatenated_cw - blue_wavelength))

    red_band = concatenated_cube[:, :, red_band_idx]
    green_band = concatenated_cube[:, :, green_band_idx]
    blue_band = concatenated_cube[:, :, blue_band_idx]

    red_norm = (red_band - red_band.min()) / (red_band.max() - red_band.min())
    green_norm = (green_band - green_band.min()) / (green_band.max() - green_band.min())
    blue_norm = (blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())

    rgb_image = np.stack((red_norm, green_norm, blue_norm), axis=-1)

    # Latitude and Longitude arrays can be extracted if available (EnMAP provides geolocation)
    # For now, we set them to None
    latitude = None
    longitude = None

    return concatenated_cube, concatenated_cw, concatenated_fwhm, rgb_image, latitude, longitude


def enmap_scene_geometry(metadata_file):
    """
    Extract scene-level viewing/sun geometry summaries from EnMAP metadata.

    Returns a dictionary containing center-point angles (viewing, sun, relative)
    and along/across off-nadir components when available.
    """

    tree = ET.parse(metadata_file)
    root = strip_namespaces(tree.getroot())

    def _read_points(tag_name):
        node = root.find(f".//specific/{tag_name}")
        if node is None:
            return {}
        values = {}
        for child in node:
            val = _safe_float(child.text)
            if val is not None:
                values[child.tag] = val
        return values

    def _read_center(tag_name):
        values = _read_points(tag_name)
        return values.get("center"), values

    geometry = {}

    vza_center, vza_points = _read_center("viewingZenithAngle")
    if vza_points:
        geometry["viewing_zenith"] = vza_points
    geometry["viewing_zenith_center"] = vza_center

    vaa_center, vaa_points = _read_center("viewingAzimuthAngle")
    if vaa_points:
        geometry["viewing_azimuth"] = vaa_points
    geometry["viewing_azimuth_center"] = vaa_center

    saa_center, saa_points = _read_center("sunAzimuthAngle")
    if saa_points:
        geometry["sun_azimuth"] = saa_points
    geometry["sun_azimuth_center"] = saa_center

    sea_center, sea_points = _read_center("sunElevationAngle")
    if sea_points:
        geometry["sun_elevation"] = sea_points
    geometry["sun_elevation_center"] = sea_center

    # Try direct sun zenith tag, otherwise derive from elevation
    sza_center_direct, sza_points_direct = _read_center("sunZenithAngle")
    if sza_points_direct:
        geometry["sun_zenith"] = sza_points_direct
    if sza_center_direct is not None:
        geometry["sun_zenith_center"] = sza_center_direct
    elif sea_center is not None:
        geometry["sun_zenith_center"] = 90.0 - sea_center

    along_center, along_points = _read_center("alongOffNadirAngle")
    if along_points:
        geometry["along_off_nadir"] = along_points
    geometry["along_off_nadir_center"] = along_center

    across_center, across_points = _read_center("acrossOffNadirAngle")
    if across_points:
        geometry["across_off_nadir"] = across_points
    geometry["across_off_nadir_center"] = across_center

    # Relative zenith/azimuth (using center values)
    vza = geometry.get("viewing_zenith_center")
    sza = geometry.get("sun_zenith_center")
    if vza is not None and sza is not None:
        geometry["relative_zenith_center"] = sza - vza

    vaa = geometry.get("viewing_azimuth_center")
    saa = geometry.get("sun_azimuth_center")
    if vaa is not None and saa is not None:
        diff = saa - vaa
        wrapped = ((diff + 180.0) % 360.0) - 180.0
        geometry["relative_azimuth_center"] = diff
        geometry["relative_azimuth_center_abs"] = abs(wrapped)

    return geometry


def enmap_metadata_read(metadata_file):
    """
    Reads SZA and mean WV from EnMAP metadata XML file.

    Parameters:
    - metadata_file: Path to the XML metadata file.

    Returns:
    - SZA: Solar Zenith Angle in degrees.
    - meanWV: Mean Water Vapor in g/cm^2.
    """
    tree = ET.parse(metadata_file)
    root = strip_namespaces(tree.getroot())

    SZA = _safe_float(root.findtext(".//specific/qualityFlag/sceneSZA"))
    if SZA is None:
        raise ValueError("Solar Zenith Angle (sceneSZA) not found in metadata.")

    scene_wv = _safe_float(root.findtext(".//specific/qualityFlag/sceneWV"))
    if scene_wv is None:
        raise ValueError("Mean Water Vapor (sceneWV) not found in metadata.")
    meanWV = scene_wv / 1000.0

    mean_ground_elevation = _safe_float(root.findtext(".//specific/meanGroundElevation"))
    if mean_ground_elevation is None:
        print("Warning: meanGroundElevation not found in metadata.")

    print(f"Sun Zenith Angle (degrees): {SZA}")
    print(f"Mean Water Vapor (g/cm^2): {meanWV}")
    print(f"Mean Ground Elevation (m): {mean_ground_elevation}")

    return SZA, meanWV, mean_ground_elevation


def save_as_geotiff_single_band_enmap(data, output_file, reference_dataset):
    """
    Saves a single-band array as a GeoTIFF file with EnMAP georeferencing.
    Robust version: closes all handles, uses compression, BigTIFF if safer,
    and guarantees Float32 + NoData.
    """
    # Ensure 2D float32 and replace non-finite
    arr = np.asarray(data, dtype=np.float32)
    nodata_value = np.float32(-9999.0)
    np.copyto(arr, nodata_value, where=~np.isfinite(arr))

    # Dimensions from reference
    xsize = reference_dataset.RasterXSize
    ysize = reference_dataset.RasterYSize
    if arr.shape != (ysize, xsize):
        raise ValueError(f"Array shape {arr.shape} does not match reference raster size {(ysize, xsize)}")

    # Creation options: tiled, compressed, BigTIFF if needed
    driver = gdal.GetDriverByName("GTiff")
    opts = [
        "TILED=YES",
        "COMPRESS=LZW",
        "BIGTIFF=IF_SAFER",  # avoids >4GB classic-TIFF issues
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
    ]
    if os.path.exists(output_file):
        # GDAL refuses to overwrite in Create, so remove the stale file first
        driver.Delete(output_file)
    ds = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_Float32, options=opts)
    if ds is None:
        raise RuntimeError(f"Could not create {output_file}")

    # GeoTransform / Projection
    ds.SetGeoTransform(reference_dataset.GetGeoTransform())
    ds.SetProjection(reference_dataset.GetProjection())

    # Write band + NoData
    band1 = ds.GetRasterBand(1)
    band1.WriteArray(arr)
    band1.SetNoDataValue(float(nodata_value))
    band1.FlushCache()
    band1 = None  # CRITICAL: release band handle

    # Flush and close
    ds.FlushCache()
    ds = None  # CRITICAL: release dataset handle


def save_as_geotiff_rgb_enmap(rgb_data, output_file, reference_dataset):
    """
    Saves an RGB array as a GeoTIFF with EnMAP georeferencing.
    """
    # Scale and convert to uint8
    rgb = np.clip(rgb_data * 255.0, 0, 255).astype(np.uint8)

    ysize, xsize, bands = rgb.shape
    if bands != 3:
        raise ValueError("RGB array must have 3 bands")

    driver = gdal.GetDriverByName("GTiff")
    opts = [
        "TILED=YES",
        "COMPRESS=LZW",
        "BIGTIFF=IF_SAFER",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
    ]
    if os.path.exists(output_file):
        driver.Delete(output_file)
    ds = driver.Create(output_file, xsize, ysize, 3, gdal.GDT_Byte, options=opts)
    if ds is None:
        raise RuntimeError(f"Could not create {output_file}")

    ds.SetGeoTransform(reference_dataset.GetGeoTransform())
    ds.SetProjection(reference_dataset.GetProjection())

    # Write bands and set color interpretation
    for i, interp in enumerate([gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand], start=1):
        b = ds.GetRasterBand(i)
        b.WriteArray(rgb[:, :, i - 1])
        b.SetColorInterpretation(interp)
        b.FlushCache()
        b = None  # CRITICAL

    ds.FlushCache()
    ds = None  # CRITICAL


def extract_enmap_files_from_folder(folder_path):
    """
    Identify the VNIR, SWIR, and METADATA files in the given folder.
    Returns (vnir_file, swir_file, metadata_file) or (None, None, None) if not found.
    """
    vnir_file = None
    swir_file = None
    metadata_file = None

    for file in os.listdir(folder_path):
        if file.endswith("SPECTRAL_IMAGE_VNIR.TIF"):
            vnir_file = os.path.join(folder_path, file)
        elif file.endswith("SPECTRAL_IMAGE_SWIR.TIF"):
            swir_file = os.path.join(folder_path, file)
        elif file.endswith("METADATA.XML"):
            metadata_file = os.path.join(folder_path, file)

    return vnir_file, swir_file, metadata_file


def derive_output_basename(vnir_file):
    """
    Derive a simplified output basename from the VNIR file name.
    For example:
    ENMAP01-____L1B-DT0000090108_20240112T144653Z_002_V010401_...-SPECTRAL_IMAGE_VNIR.TIF
    -> L1B_20240112T144653Z
    """
    base = os.path.basename(vnir_file)
    parts = base.split("_")
    product_level = "L1B"
    date_str = parts[1]  # The date/time string
    output_basename = f"{product_level}_{date_str}"
    return output_basename


def _to_yyyymmddThhmmssZ(dt_text: str) -> str:
    """
    Normalize an EnMAP time string like '2025-06-23T11:01:29.036499Z'
    to 'YYYYMMDDTHHMMSSZ' with seconds precision.
    """
    t = dt_text.strip()
    assert t.endswith("Z"), f"Expected Zulu time, got: {dt_text}"
    t_noz = t[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(t_noz, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            dt = None
    if dt is None:
        raise ValueError(f"Unrecognized datetime format: {dt_text}")
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _first_text(elem, path):
    e = elem.find(path)
    return e.text.strip() if (e is not None and e.text) else None


def derive_basename_from_metadata(metadata_file: str) -> str:
    """
    Build a compact, informative basename using the METADATA.XML file:
      <level>_<datatake>_<tile>_<start>_<stop>

    Example: L1B_DT0000137438_010_20250623T110129Z_20250623T110133Z
    """
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    # Level (processing level)
    level = _first_text(root, ".//metadata/schema/processingLevel") or _first_text(root, ".//base/level") or "L1B"

    # temporal coverage start/stop
    start_raw = _first_text(root, ".//base/temporalCoverage/startTime")
    stop_raw = _first_text(root, ".//base/temporalCoverage/stopTime")
    if not start_raw or not stop_raw:
        raise ValueError("startTime/stopTime not found in METADATA.XML")

    start_z = _to_yyyymmddThhmmssZ(start_raw)
    stop_z = _to_yyyymmddThhmmssZ(stop_raw)

    # Datatake + tile via product name
    name_txt = _first_text(root, ".//metadata/name") or ""
    m_dt = re.search(r"(DT\d{10,})", name_txt)  # DT + at least 10 digits
    m_tile = re.search(r"_(\d{3})_", name_txt)
    datatake = m_dt.group(1) if m_dt else "DTUNKNOWN"
    tile = m_tile.group(1) if m_tile else "000"

    return f"{level}_{datatake}_{tile}_{start_z}_{stop_z}"
