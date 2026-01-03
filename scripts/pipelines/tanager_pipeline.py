# -*- coding: utf-8 -*-
"""
Tanager processing pipeline mirroring PRISMA/EnMAP flows. Loads TOA radiance
and companion surface reflectance (for water vapour), synthesises CH4 targets
from the LUT, runs the matched filter, and propagates σ_RMN using the Tanager
columnwise SNR reference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

import advanced_matched_filter
from scripts.core import matched_filter, targets, lut, noise, jpl_matched_filter  # type: ignore
from scripts.satellites import tanager_utils  # type: ignore

logger = logging.getLogger(__name__)


def _scene_stats(rad_path: str, sr_path: str) -> tuple[float, float]:
    """Return (SZA, WV) as single scalar values (median)."""

    with tanager_utils.open_hdf(rad_path) as f_rad:  # type: ignore[attr-defined]
        sza = np.asarray(f_rad["HDFEOS/SWATHS/HYP/Data Fields/sun_zenith"])
        sza_val = float(np.nanmedian(sza))

    with tanager_utils.open_hdf(sr_path) as f_sr:  # type: ignore[attr-defined]
        # Dataset spelled "vapour" in the spec/products
        wv = np.asarray(f_sr["HDFEOS/SWATHS/HYP/Data Fields/column_water_vapour"])
        wv_val = float(np.nanmedian(wv))

    return sza_val, wv_val


def ch4_detection_tanager(
    radiance_file: str,
    sr_file: str,
    dem_file: str,
    lut_file: str,
    output_dir: str,
    k: int = 1,
    min_wavelength: float = 2100.0,
    max_wavelength: float = 2450.0,
    snr_reference_path: str | None = None,
    mf_mode: str = "srf-column",
    advanced_mf_options: dict | None = None,
    output_name_suffix: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Tanager scene: %s", radiance_file)

    # Scene stats
    sza_deg, wv_gcm2 = _scene_stats(radiance_file, sr_file)
    logger.info("Sun zenith (median): %.3f deg", sza_deg)
    logger.info("Water vapour (median): %.3f g/cm^2", wv_gcm2)

    # Load radiance cube (bands, rows, cols) and geolocation
    cube = tanager_utils.load_tanager_cube(
        radiance_file,
        dataset_path=tanager_utils.TANAGER_TOA_RADIANCE_DATASET,
        load_masks=False,
        load_geolocation=True,
    )
    rad_brc = cube.data.astype(np.float32) * 0.1  # to µW·cm⁻²·sr⁻¹·nm⁻¹
    rad = np.transpose(rad_brc, (1, 2, 0))  # (rows, cols, bands)

    wl = np.asarray(cube.wavelengths, dtype=float)
    fwhm = np.asarray(cube.fwhm, dtype=float) if cube.fwhm is not None else np.zeros_like(wl)
    mean_cw = wl
    mean_fwhm = fwhm

    band_indices = np.where((mean_cw >= min_wavelength) & (mean_cw <= max_wavelength))[0]
    if band_indices.size == 0:
        raise ValueError(f"No bands within {min_wavelength}-{max_wavelength} nm for this scene.")
    rad_sel = rad[:, :, band_indices]
    cw_sel = mean_cw[band_indices]
    fwhm_sel = mean_fwhm[band_indices]

    concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    ground_km = lut.normalize_ground_km(lut.mean_elev_from_dem(dem_file, tanager_utils.bounding_box(cube)))
    water_gcm2 = lut.normalize_wv_gcm2(wv_gcm2)

    simRads_array, simWave_array = lut.generate_library(
        concentrations, lut_file, zenith=sza_deg, sensor=120, ground=ground_km, water=water_gcm2, order=1
    )
    logger.info("Simulated radiance spectra generated for %d concentration levels", len(concentrations))

    target_i = targets.generate_template_from_bands(cw_sel, fwhm_sel, simRads_array, simWave_array, concentrations)
    base_target = target_i[:, 1]
    num_columns = rad_sel.shape[1]
    target_spectra = np.repeat(base_target[:, None], num_columns, axis=1)

    # MF modes
    if mf_mode == "srf-column":
        classified_image = matched_filter.k_means_hyperspectral(rad_sel, k)
        mean_radiance, covariance_matrices = matched_filter.calculate_statistics(rad_sel, classified_image, k)
        concentration_map = matched_filter.calculate_matched_filter_columnwise(
            rad_sel, classified_image, mean_radiance, covariance_matrices, target_spectra, k
        )
        classified_image_for_noise = classified_image
    elif mf_mode == "full-column":
        n_rows, n_columns = rad_sel.shape[:2]
        classified_image = np.tile(np.arange(n_columns, dtype=np.int32), (n_rows, 1))
        mean_radiance, covariance_matrices = matched_filter.calculate_column_statistics(rad_sel)
        concentration_map = matched_filter.calculate_matched_filter_columnwise(
            rad_sel, classified_image, mean_radiance, covariance_matrices, target_spectra, n_columns
        )
        classified_image_for_noise = classified_image
    elif mf_mode == "advanced":
        advanced_kwargs = dict(group_min=10, group_max=30, n_clusters=max(1, k), shrinkage=0.1)
        if advanced_mf_options:
            advanced_kwargs.update(advanced_mf_options)
        concentration_map = advanced_matched_filter.run_advanced_mf(
            radiance_cube=rad_sel,
            targets=target_spectra,
            wavelengths=cw_sel,
            mask=None,
            **advanced_kwargs,
        )
        classified_image = advanced_matched_filter.get_last_cluster_labels()
        stats = advanced_matched_filter.get_last_cluster_statistics()
        if classified_image is None or stats is None:
            raise RuntimeError("Advanced matched filter did not expose cluster statistics.")
        mean_radiance, covariance_matrices = stats
        classified_image_for_noise = np.where(classified_image < 0, 0, classified_image)
    elif mf_mode == "jpl":
        logger.info("Running JPL matched filter for Tanager.")
        good_pixel_mask = np.all(np.isfinite(rad_sel), axis=2)
        classified_image = np.zeros(rad_sel.shape[:2], dtype=np.int32)
        classified_image_for_noise = classified_image

        rad_cube_brc_jpl = np.transpose(rad_sel, (2, 0, 1))
        reference_jpl = noise.ColumnwiseSNRReference.load(
            snr_reference_path or os.environ.get("TANAGER_SNR_REFERENCE")
        )
        reference_subset_jpl = reference_jpl.subset_by_wavelengths(cw_sel).ensure_column_count(rad_sel.shape[1])
        sigma_cube_jpl = noise.compute_sigma_map_from_reference(reference_subset_jpl, rad_cube_brc_jpl)
        noise_cube_for_jpl = np.transpose(sigma_cube_jpl, (1, 2, 0))
        concentration_map, sigma_rmn, sensitivity_map = jpl_matched_filter.run_jpl_mf(
            rads_array=rad_sel,
            target_spectra=base_target,
            good_pixel_mask=good_pixel_mask,
            noise_cube=noise_cube_for_jpl,
        )
    else:
        raise ValueError(f"Unsupported Tanager matched filter mode: {mf_mode}")

    snr_reference_path = snr_reference_path or os.environ.get("TANAGER_SNR_REFERENCE")
    if not snr_reference_path:
        raise RuntimeError(
            "Tanager SNR reference not provided. Set snr_reference_path or TANAGER_SNR_REFERENCE env var."
        )

    if mf_mode != "jpl":
        reference = noise.ColumnwiseSNRReference.load(snr_reference_path)
        reference_subset = reference.subset_by_wavelengths(cw_sel).ensure_column_count(rad_sel.shape[1])
        rad_cube_brc = np.transpose(rad_sel, (2, 0, 1))
        sigma_cube = noise.compute_sigma_map_from_reference(reference_subset, rad_cube_brc)
        sigma_rmn = noise.propagate_rmn_uncertainty_per_pixel(
            radiance_cube=rad_cube_brc,
            sigma_cube=sigma_cube,
            target_spectra=base_target,
        ).astype(np.float32)
        if mf_mode == "advanced":
            sigma_rmn[classified_image < 0] = np.nan

    logger.info(
        "σ_RMN (instrument noise) — min: %.4f, median: %.4f, max: %.4f",
        float(np.nanmin(sigma_rmn)),
        float(np.nanmedian(sigma_rmn)),
        float(np.nanmax(sigma_rmn)),
    )

    basename = Path(radiance_file).stem
    if output_name_suffix:
        basename = f"{basename}_{output_name_suffix}"
    conc_path = os.path.join(output_dir, f"{basename}_MF.tif")
    sigma_path = os.path.join(output_dir, f"{basename}_MF_uncertainty.tif")
    rgb_path = os.path.join(output_dir, f"{basename}_RGB.tif")
    cl_path = os.path.join(output_dir, f"{basename}_CL.tif")

    tanager_utils.save_with_geolocation_single_band(concentration_map, conc_path, cube)
    tanager_utils.save_with_geolocation_single_band(sigma_rmn, sigma_path, cube)
    tanager_utils.save_with_geolocation_single_band(classified_image_for_noise, cl_path, cube)
    rgb = tanager_utils.quicklook_rgb(cube)
    tanager_utils.save_with_geolocation_multichannel(rgb, rgb_path, cube)

    logger.info("GeoTIFF exports completed for Tanager scene %s", basename)

