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

from scripts.core import matched_filter, targets, lut, io_utils, noise  # type: ignore
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
    snr_reference_path: str | None = None,
    mf_mode: str = "srf-column",
):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("EnMAP scene: %s", vnir_file)

    # Read SZA, meanWV, and mean_ground_elevation from EnMAP metadata
    SZA, meanWV, mean_ground_elevation = enmap_utils.enmap_metadata_read(metadata_file)

    geometry = enmap_utils.enmap_scene_geometry(metadata_file)

    vza = geometry.get("viewing_zenith_center")
    if vza is not None:
        logger.info("Viewing zenith angle (center): %.3f deg", vza)
    else:
        logger.warning("Viewing zenith angle (center) not found in metadata.")

    vaa = geometry.get("viewing_azimuth_center")
    if vaa is not None:
        logger.info("Viewing azimuth angle (center): %.3f deg", vaa)
    else:
        logger.warning("Viewing azimuth angle (center) not found in metadata.")

    saa = geometry.get("sun_azimuth_center")
    if saa is not None:
        logger.info("Sun azimuth angle (center): %.3f deg", saa)
    else:
        logger.warning("Sun azimuth angle (center) not found in metadata.")

    sza_center = geometry.get("sun_zenith_center")
    if sza_center is not None:
        logger.info("Sun zenith angle (center): %.3f deg", sza_center)

    along = geometry.get("along_off_nadir_center")
    if along is not None:
        logger.info("Along-track off-nadir angle (center): %.3f deg", along)

    across = geometry.get("across_off_nadir_center")
    if across is not None:
        logger.info("Across-track off-nadir angle (center): %.3f deg", across)

    if geometry.get("relative_zenith_center") is not None:
        logger.info(
            "Relative zenith (SZA − VZA) center: %.3f deg",
            geometry["relative_zenith_center"],
        )

    if geometry.get("relative_azimuth_center") is not None:
        diff = geometry["relative_azimuth_center"]
        abs_diff = geometry.get("relative_azimuth_center_abs")
        if abs_diff is not None:
            logger.info(
                "Relative azimuth (SAA − VAA) center: %.3f deg (|…|=%.3f deg)",
                diff,
                abs_diff,
            )
        else:
            logger.info(
                "Relative azimuth (SAA − VAA) center: %.3f deg",
                diff,
            )

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
    uncertainty_output_file = os.path.join(output_dir, f"{output_basename}_MF_uncertainty.tif")
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

    if mf_mode == "srf-column":
        classified_image = matched_filter.k_means_hyperspectral(rads_array_subselection, k)
        logger.info("k-means classification completed with k=%d (MF columnwise SRF mode).", k)
        mean_radiance, covariance_matrices = matched_filter.calculate_statistics(
            rads_array_subselection, classified_image, k
        )
    elif mf_mode == "full-column":
        n_rows, n_columns = rads_array_subselection.shape[:2]
        classified_image = np.tile(np.arange(n_columns, dtype=np.int32), (n_rows, 1))
        logger.info(
            "Full column-wise MF selected: per-column mean radiance and covariance without clustering (ignoring k=%d).",
            k,
        )
        mean_radiance, covariance_matrices = matched_filter.calculate_column_statistics(rads_array_subselection)
        if mean_radiance.shape[0] != n_columns:
            raise RuntimeError(
                f"Column statistics mismatch: expected {n_columns} columns, got {mean_radiance.shape[0]}."
            )
    else:
        raise ValueError(f"Unsupported EnMAP matched filter mode: {mf_mode}")

    k_eff = mean_radiance.shape[0]

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
    base_target = target_i[:, 1]
    num_columns = rads_array_subselection.shape[1]
    # TODO(#columnwise-srf): replace the nominal SRF tiling with true per-column CW/FWHM once detector-level metadata is available.
    target_spectra = np.repeat(base_target[:, None], num_columns, axis=1)
    if mf_mode == "srf-column":
        logger.info(
            "Running MF columnwise SRF with cluster tuning option (nominal SRF tiled across %d columns).",
            num_columns,
        )
    else:
        logger.info("Running full column-wise matched filter (SRF + per-column statistics) on %d columns.", num_columns)

    concentration_map = matched_filter.calculate_matched_filter_columnwise(
        rads_array_subselection, classified_image, mean_radiance, covariance_matrices, target_spectra, k_eff
    )

    np.save(target_spectra_export_name, target_spectra)

    snr_reference_path = snr_reference_path or os.environ.get("ENMAP_SNR_REFERENCE")
    if not snr_reference_path:
        raise RuntimeError(
            "EnMAP SNR reference not provided. "
            "Set the path via snr_reference_path parameter or ENMAP_SNR_REFERENCE environment variable."
        )
    logger.info("Loading EnMAP SNR reference from %s", snr_reference_path)
    reference = noise.ColumnwiseSNRReference.load(snr_reference_path)
    reference_subset = reference.subset_by_wavelengths(cw_subselection).ensure_column_count(
        rads_array_subselection.shape[1]
    )
    rad_cube_brc = np.transpose(rads_array_subselection, (2, 0, 1))
    sigma_cube = noise.compute_sigma_map_from_reference(reference_subset, rad_cube_brc)
    sigma_rmn = noise.propagate_rmn_uncertainty(
        sigma_cube=sigma_cube,
        classified_image=classified_image,
        mean_radiance=mean_radiance,
        target_spectra=target_spectra,
    ).astype(np.float32)

    logger.info(
        "σ_RMN (instrument noise) — min: %.4f, median: %.4f, max: %.4f",
        float(np.nanmin(sigma_rmn)),
        float(np.nanmedian(sigma_rmn)),
        float(np.nanmax(sigma_rmn)),
    )

    reference_dataset = gdal.Open(swir_file)

    enmap_utils.save_as_geotiff_single_band_enmap(concentration_map, concentration_output_file, reference_dataset)
    enmap_utils.save_as_geotiff_single_band_enmap(sigma_rmn, uncertainty_output_file, reference_dataset)
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
        uncertainty_output_file=uncertainty_output_file,
        mf_mode=mf_mode,
    )


def process_directory_enmap(
    root_dir,
    lut_file,
    k=1,
    min_wavelength=2100.0,
    max_wavelength=2450.0,
    mf_mode: str = "srf-column",
):
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
                    mf_mode=mf_mode,
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
