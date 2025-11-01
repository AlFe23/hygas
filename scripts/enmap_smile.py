# -*- coding: utf-8 -*-
"""

EnMAP L1B VNIR/SWIR Radiance and Spectral Smile Analysis
--------------------------------------------------------

This script reads and analyses EnMAP Level-1B products containing separate
VNIR and SWIR hyperspectral cubes together with the corresponding metadata XML.
It performs radiometric conversion and spectral smile characterization for
each instrument independently, following the definitions in the EnMAP L1B
Processor documentation (EN-PCV-TN-4006 and EN-PCV-ICD-2009-2).

Main features
--------------
1. **Independent VNIR/SWIR handling**
   - Reads VNIR and SWIR GeoTIFF cubes separately.
   - Extracts global band metadata (<bandCharacterisation>) and the respective
     <smileCorrection><VNIR> / <SWIR> subsections with local band numbering.
   - Associates each local band with its specific 4th-order polynomial smile
     coefficients (if available).

2. **Radiometric calibration**
   - Converts digital numbers (DN) to top-of-atmosphere radiance using
     per-band calibration coefficients:
       Radiance = Gain * DN + Offset.

3. **Spectral smile computation**
   - For a selected band, computes wavelength shifts across the detector
     columns as:
         Δλ(x) = c0 + c1·x + c2·x² + c3·x³ + c4·x⁴  [nm]
       where coefficients c0…c4 are read from the corresponding smile section.
   - Derives column-wise center wavelengths:
         λ(x) = λ_nominal + Δλ(x)
   - (In current L1B products, FWHM(x) is constant across track.)

4. **Outputs and plots**
   - Mean top-of-atmosphere radiance spectra for VNIR and SWIR separately.
   - Spectral smile curves Δλ(x) for user-selected VNIR and SWIR bands.
   - CW(x) and FWHM(x) plots across the detector for the same bands.

Usage
-----
    python enmap_vnir_swir_smile_fixed.py

or within another script:

    run_vnir_swir_independent(
        in_dir=<path_to_L1B_directory>,
        vnir_local_band_to_plot=<VNIR band index>,
        swir_local_band_to_plot=<SWIR band index>
    )

The input directory must contain:
    - *SPECTRAL_IMAGE_VNIR*.TIF
    - *SPECTRAL_IMAGE_SWIR*.TIF
    - *METADATA.XML*

Dependencies
------------
    - Python ≥ 3.8
    - GDAL
    - NumPy
    - Matplotlib

References
----------
    - EN-PCV-ICD-2009-2 : EnMAP HSI Product Specification Level-1/2
    - EN-PCV-TN-4006   : Level-1B Processor – Systematic and Radiometric Correction
    - DLR (2020–2024)  : EnMAP Processor and Calibration Documents

Author
------
    Alvise Ferrari, Sapienza University
    Date: 2025-10-05


"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.satellites import enmap_utils


# --------------------------- Analysis & Plots ---------------------------


def _print_enmap_metadata(xml_path: str):
    """Print LUT-relevant EnMAP metadata plus viewing geometry."""

    enmap_utils.enmap_metadata_read(xml_path)
    geometry = enmap_utils.enmap_scene_geometry(xml_path)

    summary_lines: list[str] = []

    def _append(label, key, unit="°"):
        value = geometry.get(key)
        if value is not None:
            line = f"{label}: {value:.3f}{unit}"
            print(f"[EnMAP] {line}")
            summary_lines.append(line)
            return True
        return False

    _append("Viewing zenith angle (center)", "viewing_zenith_center")
    _append("Viewing azimuth angle (center)", "viewing_azimuth_center")
    _append("Sun azimuth angle (center)", "sun_azimuth_center")
    _append("Sun elevation angle (center)", "sun_elevation_center")
    _append("Sun zenith angle (center)", "sun_zenith_center")
    _append("Along off-nadir (center)", "along_off_nadir_center")
    _append("Across off-nadir (center)", "across_off_nadir_center")

    if geometry.get("relative_zenith_center") is not None:
        val = geometry["relative_zenith_center"]
        line = f"Relative zenith (SZA − VZA) center: {val:.3f}°"
        print(f"[EnMAP] {line}")
        summary_lines.append(line)

    if geometry.get("relative_azimuth_center") is not None:
        diff = geometry["relative_azimuth_center"]
        abs_diff = geometry.get("relative_azimuth_center_abs")
        if abs_diff is not None:
            line = f"Relative azimuth (SAA − VAA) center: {diff:.3f}° (|…|={abs_diff:.3f}°)"
        else:
            line = f"Relative azimuth (SAA − VAA) center: {diff:.3f}°"
        print(f"[EnMAP] {line}")
        summary_lines.append(line)

    return summary_lines


def mean_spectrum(rad_cube, cw_vec):
    return cw_vec, rad_cube.reshape(rad_cube.shape[0], -1).mean(axis=1)


def plot_mean_spectrum_ax(ax, cw, spec, title):
    ax.plot(cw, spec, lw=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def delta_lambda_across_track(band_meta, ncols):
    x = np.arange(1, ncols+1, dtype=np.float64)
    c = band_meta['smile_coeffs']
    if c is None:
        return x, np.zeros_like(x, dtype=float)
    # Δλ(x) [nm] = c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4
    delta = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4
    return x, delta

def plot_smile_delta_lambda_ax(ax, band_meta, ncols, prefix=""):
    x, delta = delta_lambda_across_track(band_meta, ncols)
    note = "" if band_meta['smile_coeffs'] is not None else " (no coeffs in metadata)"
    lambda_nominal = band_meta['cw_nm']
    ax.plot(x, delta, lw=1)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_xlabel("Across-track column (x)")
    ax.set_ylabel("Δλ(x) = CW(x) − CW_nominal  [nm]")
    ax.set_title(
        f"{prefix}Spectral smile Δλ(x) — local band {band_meta['local_idx']}"
        f" (global {band_meta['global_id']}, λ0={lambda_nominal:.2f} nm){note}"
    )
    ax.grid(alpha=0.3)


def plot_cw_and_fwhm_across_track_ax(ax, band_meta, ncols, prefix=""):
    x = np.arange(1, ncols+1, dtype=np.float64)
    lam0 = band_meta['cw_nm']
    c = band_meta['smile_coeffs']
    if c is None:
        cw_x = np.full_like(x, lam0, dtype=float)
    else:
        cw_x = lam0 + (c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4)
    fwhm_x = np.full_like(x, band_meta['fwhm_nm'], dtype=float)  # constant in L1B
    ax.plot(x, cw_x, lw=1)
    ax.set_xlabel("Across-track column (x)")
    ax.set_ylabel("Center wavelength λ(x) [nm]")
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{prefix}CW(x) & FWHM(x) — local band {band_meta['local_idx']} "
        f"(global {band_meta['global_id']}, λ0={lam0:.2f} nm)"
    )
    ax2 = ax.twinx()
    ax2.plot(x, fwhm_x, lw=1, ls="--", color="orange")
    ax2.set_ylabel("FWHM(x) [nm]")


def render_summary_plots(mean_entries, smile_entries, metadata_lines=None):
    axes_needed = len(mean_entries) + 2 * len(smile_entries)
    if axes_needed == 0:
        return

    nrows, ncols = 3, 2
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(10, 11))
    axes = axes_grid.flatten()

    idx = 0
    for entry in mean_entries:
        if idx >= axes.size:
            break
        plot_mean_spectrum_ax(axes[idx], entry["cw"], entry["spec"], entry["title"])
        idx += 1

    for entry in smile_entries:
        if idx >= axes.size:
            break
        plot_smile_delta_lambda_ax(axes[idx], entry["meta"], entry["ncols"], prefix=entry["prefix"])
        idx += 1
        if idx >= axes.size:
            break
        plot_cw_and_fwhm_across_track_ax(axes[idx], entry["meta"], entry["ncols"], prefix=entry["prefix"])
        idx += 1

    for ax in axes[idx:]:
        ax.axis('off')

    if metadata_lines:
        fig.suptitle("\n".join(metadata_lines), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    plt.show()

# --------------------------- Main pipeline ---------------------------

def run_vnir_swir_independent(in_dir, vnir_local_band_to_plot=None, swir_local_band_to_plot=None):
    """Treat VNIR and SWIR independently and use LOCAL band indices for smile."""
    vnir_path, swir_path, xml_path = enmap_utils.find_enmap_files(in_dir)
    print(f"[i] VNIR: {os.path.basename(vnir_path)}")
    print(f"[i] SWIR: {os.path.basename(swir_path)}")
    print(f"[i] META: {os.path.basename(xml_path)}")
    metadata_lines = _print_enmap_metadata(xml_path)

    # VNIR/SWIR meta with local smile indices
    vnir_meta, swir_meta = enmap_utils.parse_metadata_vnir_swir(xml_path)
    print(f"[i] VNIR meta: {len(vnir_meta)} bands (CW {vnir_meta[0]['cw_nm']:.2f}–{vnir_meta[-1]['cw_nm']:.2f} nm)")
    print(f"[i] SWIR meta: {len(swir_meta)} bands (CW {swir_meta[0]['cw_nm']:.2f}–{swir_meta[-1]['cw_nm']:.2f} nm)")

    cube, cw_full, _, _, _, _ = enmap_utils.enmap_read(vnir_path, swir_path, xml_path)
    rad_full = np.transpose(cube, (2, 0, 1))  # (bands, rows, cols), already in µW cm^-2 sr^-1 nm^-1

    n_vnir = len(vnir_meta)
    n_swir = len(swir_meta)
    vnir_rad = rad_full[:n_vnir]
    swir_rad = rad_full[n_vnir:n_vnir + n_swir]

    # Mean spectra
    cw_vnir = cw_full[:n_vnir]
    cw_swir = cw_full[n_vnir:n_vnir + n_swir]
    wl_vnir, mean_vnir = cw_vnir, vnir_rad.reshape(vnir_rad.shape[0], -1).mean(axis=1)
    wl_swir, mean_swir = cw_swir, swir_rad.reshape(swir_rad.shape[0], -1).mean(axis=1)

    mean_entries = []
    mean_entries.append({
        "cw": wl_vnir,
        "spec": mean_vnir,
        "title": "Mean TOA Radiance — VNIR",
    })
    mean_entries.append({
        "cw": wl_swir,
        "spec": mean_swir,
        "title": "Mean TOA Radiance — SWIR",
    })

    # Smile plots — VNIR
    smile_entries = []
    if vnir_local_band_to_plot is not None:
        if not (1 <= vnir_local_band_to_plot <= len(vnir_meta)):
            raise ValueError(f"VNIR local band must be in 1..{len(vnir_meta)}")
        meta_b = vnir_meta[vnir_local_band_to_plot - 1]
        ncols = vnir_rad.shape[2]
        print(
            f"[i] VNIR local band {vnir_local_band_to_plot} (global {meta_b['global_id']}): "
            f"λ0={meta_b['cw_nm']:.2f} nm, FWHM={meta_b['fwhm_nm']:.2f} nm"
        )
        smile_entries.append({
            "meta": meta_b,
            "ncols": ncols,
            "prefix": "VNIR — ",
        })

    # Smile plots — SWIR
    if swir_local_band_to_plot is not None:
        if not (1 <= swir_local_band_to_plot <= len(swir_meta)):
            raise ValueError(f"SWIR local band must be in 1..{len(swir_meta)}")
        meta_b = swir_meta[swir_local_band_to_plot - 1]
        ncols = swir_rad.shape[2]
        print(
            f"[i] SWIR local band {swir_local_band_to_plot} (global {meta_b['global_id']}): "
            f"λ0={meta_b['cw_nm']:.2f} nm, FWHM={meta_b['fwhm_nm']:.2f} nm"
        )
        smile_entries.append({
            "meta": meta_b,
            "ncols": ncols,
            "prefix": "SWIR — ",
        })

    render_summary_plots(mean_entries, smile_entries, metadata_lines=metadata_lines)

# --------------------------- Example entry ---------------------------

if __name__ == "__main__":
    # main_dir = r"/mnt/d/.../L1B-DT0000004147_20221002T074828Z_001_V010501_20241110T222720Z"
    main_dir = (
        "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/"
        "EnMAP_calibration_data/Agadez_Niger_20220712/"
        "L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z"
    )

    # Choose LOCAL band indices for VNIR and SWIR (1-based within each sensor)
    run_vnir_swir_independent(
        main_dir,
        vnir_local_band_to_plot=60,   # e.g., VNIR local band #60
        swir_local_band_to_plot=114   # e.g., SWIR local band #100
    )
