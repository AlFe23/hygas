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
from datetime import datetime

import h5py
import numpy as np
from osgeo import gdal, osr


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

    # PROBLEMA:
    # Si è trovato che le bande 1,2,3 del VNIR cube e le bande 172,173 dello SWIR cube sono azzerate:
    # Q&A PRISMA_ATBD.pdf pg.118
    # VNIR: Remove bands 1, 2, 3 (0-indexed: 0, 1, 2), for these bands CWs and FWHMs are already set to 0.
    VNIR_cube_clean = np.delete(vnir_cube_bip, [0, 1, 2], axis=2)
    # SWIR: Remove bands 172, 173 (0-indexed: 171, 172), for these bands CWs and FWHMs are already set to 0.
    SWIR_cube_clean = np.delete(swir_cube_bip, [171, 172], axis=2)
    # SWIR: Remove bands 1, 2, 3, 4, since they are redundant with the last 4 bands of VNIR cube
    SWIR_cube_clean = np.delete(SWIR_cube_clean, [0, 1, 2, 3], axis=2)

    # Extract actual values of CWs and FWHMs. They are stored in standars (1000,256) arrays and have to be extracted
    cw_vnir = cw_vnir[:, 99:162]
    fwhm_vnir = fwhm_vnir[:, 99:162]
    cw_swir = cw_swir[:, 81:252]
    fwhm_swir = fwhm_swir[:, 81:252]

    # Reverse CWs and FWHMs vectors as they are in decrasing frequency order, but we want them opposite
    cw_vnir = cw_vnir[:, ::-1]
    fwhm_vnir = fwhm_vnir[:, ::-1]
    cw_swir = cw_swir[:, ::-1]
    fwhm_swir = fwhm_swir[:, ::-1]

    # SWIR: Remove bands 1, 2, 3, 4, since they are redundant with the last 4 bands of VNIR cube
    cw_swir_clean = np.delete(cw_swir, [0, 1, 2, 3], axis=1)
    fwhm_swir_clean = np.delete(fwhm_swir, [0, 1, 2, 3], axis=1)

    # Let's now concatenate the arrays for radiance cube, central wavelengths and FWHMs
    concatenated_cube = np.concatenate((SWIR_cube_clean, VNIR_cube_clean), axis=2)
    concatenated_cube = concatenated_cube[:, :, ::-1]
    concatenated_cw = np.concatenate((cw_vnir, cw_swir_clean), axis=1)
    concatenated_fwhm = np.concatenate((fwhm_vnir, fwhm_swir_clean), axis=1)

    # Print the RGB image
    red_channel = concatenated_cube[:, :, 29]  # Red channel
    green_channel = concatenated_cube[:, :, 19]  # Green channel
    blue_channel = concatenated_cube[:, :, 7]  # Blue channel

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

