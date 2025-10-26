# -*- coding: utf-8 -*-
"""
EnMAP processing pipeline leveraging shared core modules and EnMAP-specific
helpers. Derived directly from the original `enmap_MF.py`.
"""

import logging
import os
from datetime import datetime

import numpy as np
from osgeo import gdal

from scripts.core import matched_filter, targets, lut, io_utils  # type: ignore
from scripts.satellites import enmap_utils  # type: ignore

logger = logging.getLogger(__name__)


def ch4_detection_enmap(
    vnir_file,
    swir_file,
    metadata_file,
    lut_file,
    output_dir,
    k=10,
    min_wavelength=2100.0,
    max_wavelength=2450.0,
):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("EnMAP scene: %s", vnir_file)

    # Read SZA, meanWV, and mean_ground_elevation from EnMAP metadata
    SZA, meanWV, mean_ground_elevation = enmap_utils.enmap_metadata_read(metadata_file)

    # Use the extracted mean ground elevation
    if mean_ground_elevation is None:
        logger.warning("Mean ground elevation missing in metadata, defaulting to 0 m.")
        mean_elevation = 0
    else:
        mean_elevation = mean_ground_elevation

    mean_elevation_km = mean_elevation / 1000.0
    logger.info(
        "Scene parameters - SZA: %.3f deg, meanWV: %.3f g/cm^2, mean elevation: %.3f km",
        SZA,
        meanWV,
        mean_elevation_km,
    )
    if min_wavelength >= max_wavelength:
        raise ValueError(
            f"Invalid wavelength window: min_wavelength ({min_wavelength}) must be less than max_wavelength ({max_wavelength})"
        )

    output_basename = enmap_utils.derive_basename_from_metadata(metadata_file)

    target_spectra_export_name = os.path.join(output_dir, f"{output_basename}_CH4_target.npy")
    concentration_output_file = os.path.join(output_dir, f"{output_basename}_MF.tif")
    rgb_output_file = os.path.join(output_dir, f"{output_basename}_RGB.tif")
    classified_output_file = os.path.join(output_dir, f"{output_basename}_CL.tif")

    # Data extraction and processing
    rads_array, cw_array, fwhm_array, rgb_image, latitude, longitude = enmap_utils.enmap_read(
        vnir_file, swir_file, metadata_file
    )
    logger.info(
        "Radiance cube loaded: %s pixels, %s bands",
        rads_array.shape[:2],
        rads_array.shape[2],
    )

    band_indices = np.where((cw_array >= min_wavelength) & (cw_array <= max_wavelength))[0]
    if band_indices.size == 0:
        raise ValueError(
            f"No spectral bands found in the [{min_wavelength}, {max_wavelength}] nm window for the provided scene."
        )
    logger.info(
        "Spectral window %.1f-%.1f nm selected (%d bands)", min_wavelength, max_wavelength, band_indices.size
    )
    rads_array_subselection = rads_array[:, :, band_indices]
    cw_subselection = cw_array[band_indices]
    fwhm_subselection = fwhm_array[band_indices]

    classified_image = matched_filter.k_means_hyperspectral(rads_array_subselection, k)
    logger.info("k-means classification completed with k=%d", k)

    mean_radiance, covariance_matrices = matched_filter.calculate_statistics(
        rads_array_subselection, classified_image, k
    )

    concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    ground_km = lut.normalize_ground_km(mean_elevation_km)
    water_gcm2 = lut.normalize_wv_gcm2(meanWV)
    simRads_array, simWave_array = lut.generate_library(
        concentrations, lut_file, zenith=SZA, sensor=120, ground=ground_km, water=water_gcm2, order=1
    )
    logger.info("Simulated radiance spectra generated for %d concentration levels", len(concentrations))

    target_i = targets.generate_template_from_bands(
        cw_subselection, fwhm_subselection, simRads_array, simWave_array, concentrations
    )
    target_spectra = target_i[:, 1]

    np.save(target_spectra_export_name, target_spectra)

    concentration_map = matched_filter.calculate_matched_filter(
        rads_array_subselection, classified_image, mean_radiance, covariance_matrices, target_spectra, k
    )

    reference_dataset = gdal.Open(swir_file)

    enmap_utils.save_as_geotiff_single_band_enmap(concentration_map, concentration_output_file, reference_dataset)
    enmap_utils.save_as_geotiff_rgb_enmap(rgb_image, rgb_output_file, reference_dataset)
    enmap_utils.save_as_geotiff_single_band_enmap(classified_image, classified_output_file, reference_dataset)

    reference_dataset = None

    logger.info("GeoTIFF exports completed for EnMAP scene %s", output_basename)

    io_utils.generate_enmap_report(
        output_dir=output_dir,
        vnir_file=vnir_file,
        swir_file=swir_file,
        metadata_file=metadata_file,
        lut_file=lut_file,
        mean_wv=meanWV,
        SZA=SZA,
        mean_elevation_km=mean_elevation_km,
        k=k,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        concentration_output_file=concentration_output_file,
        rgb_output_file=rgb_output_file,
        classified_output_file=classified_output_file,
        target_spectra_file=target_spectra_export_name,
    )


def process_directory_enmap(root_dir, lut_file, k=1, min_wavelength=2100.0, max_wavelength=2450.0):
    """
    Batch driver for EnMAP directories, mirroring the legacy implementation.
    """
    processing_log = []
    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    for root, dirs, files in os.walk(root_dir):
        if root == root_dir:
            continue

        vnir_file, swir_file, metadata_file = enmap_utils.extract_enmap_files_from_folder(root)

        folder_name = os.path.basename(root) or os.path.basename(os.path.normpath(root))
        parent_dir = os.path.dirname(root)
        output_dir = os.path.join(parent_dir, folder_name + "_output")

        if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.info("Skipping %s because %s already contains outputs", root, output_dir)
            processing_log.append(
                (root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Skipped", f"Output directory {output_dir} not empty")
            )
            continue

        if vnir_file and swir_file and metadata_file:
            output_basename = enmap_utils.derive_output_basename(vnir_file)
            os.makedirs(output_dir, exist_ok=True)
            try:
                ch4_detection_enmap(
                    vnir_file,
                    swir_file,
                    metadata_file,
                    lut_file,
                    output_dir,
                    k=k,
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength,
                )
                status = "Success"
                details = f"Processed successfully: {output_basename}"
            except Exception as e:
                logger.exception("Error processing %s", root)
                status = "Failed"
                details = f"Error encountered: {str(e)}"
            processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
        else:
            if files:
                logger.warning(
                    "Missing required EnMAP files in %s (VNIR=%s, SWIR=%s, METADATA=%s)",
                    root,
                    vnir_file,
                    swir_file,
                    metadata_file,
                )
                status = "Failed"
                details = "Missing required VNIR, SWIR or METADATA file"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

    report_filename = f"directory_process_report_{start_time_str}.txt"
    report_filepath = os.path.join(root_dir, report_filename)
    with open(report_filepath, "w") as report_file:
        report_file.write("Directory Processing Report (EnMAP)\n")
        report_file.write("-----------------------------------\n")
        for entry in processing_log:
            folder_processed, time_processed, status, details = entry
            report_file.write(f"Folder: {folder_processed}\n")
            report_file.write(f"Time Processed: {time_processed}\n")
            report_file.write(f"Status: {status}\n")
            if details:
                report_file.write(f"Details: {details}\n")
            report_file.write("-----------------------------------\n")

    logger.info("Global process report saved at: %s", report_filepath)
