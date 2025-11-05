# -*- coding: utf-8 -*-
"""
PRISMA processing pipeline built on shared core modules plus PRISMA-specific
utilities. Functionality mirrors the original monolithic script while making the
flow easier to maintain.
"""

import contextlib
import logging
import os
import tempfile
from datetime import datetime

import numpy as np

from scripts.core import matched_filter, targets, lut, io_utils, noise  # type: ignore
from scripts.satellites import prisma_utils  # type: ignore

logger = logging.getLogger(__name__)


def ch4_detection(
    L1_file,
    L2C_file,
    dem_file,
    lut_file,
    output_dir,
    min_wavelength,
    max_wavelength,
    k,
    save_rads=False,
    snr_reference_path: str | None = None,
):

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with contextlib.ExitStack() as cleanup_stack:

        def _prepare_prisma_path(path: str) -> str:
            if path.lower().endswith(".zip"):
                temp_dir = cleanup_stack.enter_context(tempfile.TemporaryDirectory())
                extracted = prisma_utils.extract_he5_from_zip(path, temp_dir)
                if not extracted:
                    raise FileNotFoundError(f"No .he5 file found inside {path}")
                logger.info("Extracted %s into %s", path, extracted)
                return extracted
            return path

        L1_path = _prepare_prisma_path(L1_file)
        L2C_path = _prepare_prisma_path(L2C_file)

        logger.info("PRISMA scene: %s", L1_file)

        # Read SZA from L1 image attributes
        SZA = prisma_utils.prismaL1_SZA_read(L1_path)

        geom_summary = prisma_utils.prisma_l2c_geometry_summary(L2C_path)
        sun_angles = geom_summary.get("sun_angles", {})
        if sun_angles.get("zenith_deg") is not None:
            logger.info("Sun zenith angle (scene attribute): %.3f°", sun_angles["zenith_deg"])
        if sun_angles.get("azimuth_deg") is not None:
            logger.info("Sun azimuth angle (scene attribute): %.3f°", sun_angles["azimuth_deg"])

        dataset_summaries = geom_summary.get("datasets", {})
        if dataset_summaries:
            for key, entry in dataset_summaries.items():
                stats = entry["stats"]
                logger.info(
                    "%s — mean=%.3f°, median=%.3f°, min=%.3f°, max=%.3f°, std=%.3f° (n=%d, dataset=%s)",
                    entry["label"],
                    stats["mean"],
                    stats["median"],
                    stats["min"],
                    stats["max"],
                    stats["std"],
                    stats["count"],
                    entry["path"],
                )
        else:
            logger.warning("Geometric Fields datasets not found in PRISMA L2C file.")

        rel_z_stats = geom_summary.get("relative_zenith_stats") or geom_summary.get("relative_zenith")
        if rel_z_stats:
            logger.info(
                "Relative zenith angle (SZA − VZA): mean=%.3f°, median=%.3f°",
                rel_z_stats["mean"],
                rel_z_stats["median"],
            )
        rel_az_stats = geom_summary.get("relative_azimuth_stats") or geom_summary.get("relative_azimuth_summary")
        if rel_az_stats:
            logger.info(
                "Relative azimuth angle: mean=%.3f°, median=%.3f°",
                rel_az_stats["mean"],
                rel_az_stats["median"],
            )

        # Read meanWV from L2C Water Vapor Map product
        meanWV, PRS_L2C_WVM, latitude_WVM, longitude_WVM = prisma_utils.prismaL2C_WV_read(L2C_path)

        # Read bounding box from PRISMA L2C file
        bbox = prisma_utils.prismaL2C_bbox_read(L2C_path)

        # Analyze the DEM based on the bounding box from PRISMA
        mean_elevation = lut.mean_elev_from_dem(dem_file, bbox)
        logger.info(
            "Scene parameters - SZA: %.3f deg, meanWV: %.3f g/cm^2, mean elevation: %.3f km",
            SZA,
            meanWV,
            mean_elevation,
        )

        # Extract the file name without the extension for output files
        _, full_filename = os.path.split(L1_path)
        filename_without_extension = os.path.splitext(full_filename)[0]

        # Define output filenames in the specified output directory
        target_spectra_export_name = os.path.join(
            output_dir, f"{filename_without_extension}_CH4_target_PRISMA_conv.npy"
        )
        mf_output_file = os.path.join(output_dir, f"{filename_without_extension}_MF.tif")
        concentration_output_file = os.path.join(output_dir, f"{filename_without_extension}_MF_concentration.tif")
        uncertainty_output_file = os.path.join(output_dir, f"{filename_without_extension}_MF_uncertainty.tif")
        rgb_output_file = os.path.join(output_dir, f"{filename_without_extension}_rgb.tif")
        rads_output_file = os.path.join(output_dir, f"{filename_without_extension}_rads.tif")
        classified_output_file = os.path.join(output_dir, f"{filename_without_extension}_classified.tif")

        # Data extraction and processing
        (
            rads_array,
            cw_array,
            fwhm_array,
            rgb_image,
            vnir_cube_bip,
            swir_cube_bip,
            latitude_vnir,
            longitude_vnir,
            latitude_swir,
            longitude_swir,
        ) = prisma_utils.prisma_read(L1_path)
        logger.info(
            "Radiance cube loaded: %s pixels, %s bands",
            rads_array.shape[:2],
            rads_array.shape[2],
        )

        # Compute mean central wavelengths per band
        mean_cw = np.mean(cw_array, axis=0)
        mean_fwhm = np.mean(fwhm_array, axis=0)

        # Define the spectral window of interest
        band_indices = targets.select_band_indices(mean_cw, min_wavelength, max_wavelength)
        logger.info(
            "Selected %d bands in [%s, %s] nm window",
            len(band_indices),
            min_wavelength,
            max_wavelength,
        )

        # Subselect the bands based on wavelength
        rads_array_subselection = rads_array[:, :, band_indices]
        cw_subselection = cw_array[:, band_indices]
        fwhm_subselection = fwhm_array[:, band_indices]

        mean_cw_subselection = mean_cw[band_indices]
        mean_fwhm_subselection = mean_fwhm[band_indices]

        # k-means classification using the same bands as the matched filter or all available bands
        classified_image = matched_filter.k_means_hyperspectral(rads_array_subselection, k)
        logger.info("k-means classification completed with k=%d", k)

        mean_radiance, covariance_matrices = matched_filter.calculate_statistics(
            rads_array_subselection, classified_image, k
        )

        # Target spectrum calculation and matched filter application
        concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

        ground_km = lut.normalize_ground_km(mean_elevation)
        water_gcm2 = lut.normalize_wv_gcm2(meanWV)

        simRads_array, simWave_array = lut.generate_library(
            concentrations, lut_file, zenith=SZA, sensor=120, ground=ground_km, water=water_gcm2, order=1
        )

        logger.info("Simulated radiance spectra generated for %d concentration levels", len(concentrations))

        # Column-wise target spectrum and matched filter computation
        target_spectra = targets.generate_columnwise_targets(
            cw_subselection,
            fwhm_subselection,
            simRads_array,
            simWave_array,
            concentrations,
        )
        np.save(target_spectra_export_name, target_spectra)

        concentration_map = matched_filter.calculate_matched_filter_columnwise(
            rads_array_subselection,
            classified_image,
            mean_radiance,
            covariance_matrices,
            target_spectra,
            k,
        )

        snr_reference_path = snr_reference_path or os.environ.get("PRISMA_SNR_REFERENCE")
        if not snr_reference_path:
            raise RuntimeError(
                "PRISMA column-wise SNR reference not provided. "
                "Set the path via snr_reference_path parameter or PRISMA_SNR_REFERENCE environment variable."
            )
        logger.info("Loading PRISMA SNR reference from %s", snr_reference_path)
        reference = noise.ColumnwiseSNRReference.load(snr_reference_path)
        reference_subset = reference.subset_by_wavelengths(mean_cw_subselection).ensure_column_count(
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

        # Save results as GeoTIFF files
        prisma_utils.save_as_geotiff_single_band(
            concentration_map, concentration_output_file, latitude_vnir, longitude_vnir
        )
        prisma_utils.save_as_geotiff_single_band(
            sigma_rmn, uncertainty_output_file, latitude_vnir, longitude_vnir
        )
        prisma_utils.save_as_geotiff_multichannel(rgb_image, rgb_output_file, latitude_vnir, longitude_vnir)
        if save_rads:
            prisma_utils.save_as_geotiff_multichannel(rads_array, rads_output_file, latitude_vnir, longitude_vnir)
        else:
            logger.info("Skipping radiance cube GeoTIFF export (save_rads disabled)")
        prisma_utils.save_as_geotiff_single_band(
            classified_image, classified_output_file, latitude_vnir, longitude_vnir
        )

        logger.info("GeoTIFF exports completed for scene %s", filename_without_extension)

        # Generate the processing report
        io_utils.generate_prisma_report(
            output_dir=output_dir,
            L1_file=L1_file,
            L2C_file=L2C_file,
            dem_file=dem_file,
            lut_file=lut_file,
            meanWV=meanWV,
            SZA=SZA,
            mean_elevation=mean_elevation,
            k=k,
            mf_output_file=mf_output_file,
            concentration_output_file=concentration_output_file,
            rgb_output_file=rgb_output_file,
            classified_output_file=classified_output_file,
            uncertainty_output_file=uncertainty_output_file,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )


def process_directory(root_dir, dem_file, lut_file, min_wavelength, max_wavelength, k, output_root_dir=None, save_rads=False):
    """
    Batch processing wrapper that mirrors the behavior of the original script
    while delegating scene-level work to `ch4_detection`.
    """
    processing_log = []
    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    for root, dirs, files in os.walk(root_dir):
        L1_zip = None
        L2C_zip = None
        L1_file = None
        L2C_file = None

        if output_root_dir:
            relative_path = os.path.relpath(root, root_dir)
            if relative_path == ".":
                relative_path = os.path.basename(os.path.normpath(root))
            output_dir = os.path.join(output_root_dir, relative_path + "_output")
        else:
            folder_name = os.path.basename(os.path.normpath(root))
            parent_dir = os.path.dirname(root)
            output_dir = os.path.join(parent_dir, folder_name + "_output")

        if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.info("Skipping %s because %s already contains outputs", root, output_dir)
            processing_log.append(
                (root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Skipped", f"Output directory {output_dir} not empty")
            )
            continue

        for file in files:
            if file.startswith("PRS_L1_STD_OFFL") and file.endswith(".zip"):
                L1_zip = os.path.join(root, file)
            elif file.startswith("PRS_L2C_STD") and file.endswith(".zip"):
                L2C_zip = os.path.join(root, file)
            elif file.startswith("PRS_L1_STD_OFFL") and file.endswith(".he5"):
                L1_file = os.path.join(root, file)
            elif file.startswith("PRS_L2C_STD") and file.endswith(".he5"):
                L2C_file = os.path.join(root, file)

        if L1_file and L2C_file:
            L1_date = prisma_utils.get_date_from_filename(L1_file)
            L2C_date = prisma_utils.get_date_from_filename(L2C_file)

            if L1_date == L2C_date:
                os.makedirs(output_dir, exist_ok=True)
                try:
                    ch4_detection(
                        L1_file,
                        L2C_file,
                        dem_file,
                        lut_file,
                        output_dir,
                        min_wavelength,
                        max_wavelength,
                        k,
                        save_rads=save_rads,
                    )
                    status = "Success"
                    details = "Processed successfully (direct HE5 files)"
                except Exception as e:
                    logger.exception("Error processing %s", root)
                    status = "Failed"
                    details = f"Error encountered: {str(e)}"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
            else:
                logger.warning("Date mismatch between %s and %s", L1_file, L2C_file)
                status = "Failed"
                details = f"Date mismatch between L1 and L2C files: {L1_file}, {L2C_file}"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

        elif L1_zip and L2C_zip:
            L1_date = prisma_utils.get_date_from_filename(L1_zip)
            L2C_date = prisma_utils.get_date_from_filename(L2C_zip)

            if L1_date == L2C_date:
                try:
                    L1_file = prisma_utils.extract_he5_from_zip(L1_zip, root)
                    L2C_file = prisma_utils.extract_he5_from_zip(L2C_zip, root)
                except Exception as e:
                    logger.exception("Error extracting HE5 from %s / %s", L1_zip, L2C_zip)
                    status = "Failed"
                    details = f"Extraction failed: {str(e)}"
                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
                    continue

                if L1_file and L2C_file:
                    os.makedirs(output_dir, exist_ok=True)
                    try:
                        ch4_detection(
                            L1_file,
                            L2C_file,
                            dem_file,
                            lut_file,
                            output_dir,
                            min_wavelength,
                            max_wavelength,
                            k,
                            save_rads=save_rads,
                        )
                        status = "Success"
                        details = "Processed successfully from extracted zip files"
                    except Exception as e:
                        logger.exception("Error processing %s", root)
                        status = "Failed"
                        details = f"Error encountered: {str(e)}"

                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

                    try:
                        os.remove(L1_file)
                        os.remove(L2C_file)
                        logger.info("Removed extracted files %s and %s", L1_file, L2C_file)
                    except Exception as e:
                        logger.warning("Error removing extracted files: %s", e)
                        details += f" | Warning: Could not remove extracted files: {str(e)}"
                else:
                    logger.warning("Failed to extract .he5 files from %s or %s", L1_zip, L2C_zip)
                    status = "Failed"
                    details = "Extraction of .he5 files from zip failed or no he5 files found in the zip"
                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
            else:
                logger.warning("Date mismatch between %s and %s", L1_zip, L2C_zip)
                status = "Failed"
                details = "Date mismatch between L1 and L2C zip files"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

        else:
            if files:
                logger.warning("Missing required PRISMA files in %s", root)
                status = "Failed"
                details = "Missing required files (L1 and/or L2C in either .he5 or .zip format)"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

    report_filename = f"directory_process_report_{start_time_str}.txt"
    report_base = output_root_dir if output_root_dir else root_dir
    report_filepath = os.path.join(report_base, report_filename)
    with open(report_filepath, "w") as report_file:
        report_file.write("Directory Processing Report\n")
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
