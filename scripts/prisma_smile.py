# -*- coding: utf-8 -*-
"""
PRISMA L1B radiance plus spectral-smile inspection. Replicates the plots
available for EnMAP (mean spectra plus Δλ/FWHM diagnostics) but grabs the
per-column center wavelength/FWHM matrices directly from the PRISMA HDF file.
"""

#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.satellites import prisma_utils


# --------------------------- I/O helpers ---------------------------


def _resolve_l1_path(l1_path: str) -> str:
    path = os.path.abspath(l1_path)
    if path.lower().endswith(".zip"):
        extracted = prisma_utils.extract_he5_from_zip(path, os.path.dirname(path))
        if extracted is None:
            raise FileNotFoundError(
                f"Could not locate a .he5 file inside ZIP {path}. Extract it manually or provide the HE5 path."
            )
        return extracted
    return path


# --------------------------- Analysis utilities ---------------------------


def mean_spectrum(rad_cube, cw_vec):
    return cw_vec, rad_cube.reshape(rad_cube.shape[0], -1).mean(axis=1)


def plot_mean_spectrum_ax(ax, cw, spec, title, unit_label="µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$"):
    ax.plot(cw, spec, lw=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Radiance ({unit_label})")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_prisma_smile_delta_ax(ax, cw_matrix, band_idx, wavelength_nm, label):
    idx = band_idx - 1
    if idx < 0 or idx >= cw_matrix.shape[1]:
        raise ValueError(f"Band index {band_idx} outside 1..{cw_matrix.shape[1]}")
    profile = cw_matrix[:, idx]
    baseline = np.nanmean(profile)
    delta = profile - baseline
    x = np.arange(1, profile.size + 1)
    ax.plot(x, delta, lw=1)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("Across-track sample")
    ax.set_ylabel("Δλ(x) [nm]")
    ax.set_title(f"{label}Δλ(x) — band #{band_idx} ({wavelength_nm:.1f} nm)")
    ax.grid(alpha=0.3)


def plot_prisma_cw_fwhm_ax(ax, cw_matrix, fwhm_matrix, band_idx, wavelength_nm, label):
    idx = band_idx - 1
    if idx < 0 or idx >= cw_matrix.shape[1]:
        raise ValueError(f"Band index {band_idx} outside 1..{cw_matrix.shape[1]}")
    profile = cw_matrix[:, idx]
    fwhm_profile = fwhm_matrix[:, idx]
    x = np.arange(1, profile.size + 1)
    ax.plot(x, profile, lw=1)
    ax.set_xlabel("Across-track sample")
    ax.set_ylabel("λ(x) [nm]")
    ax.grid(alpha=0.3)
    ax.set_title(f"{label}CW & FWHM — band #{band_idx} ({wavelength_nm:.1f} nm)")
    ax2 = ax.twinx()
    ax2.plot(x, fwhm_profile, lw=1, ls="--", color="orange")
    ax2.set_ylabel("FWHM(x) [nm]")


def render_summary_plots(mean_entries, smile_entries, cw_matrix, fwhm_matrix):
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
        plot_prisma_smile_delta_ax(
            axes[idx],
            cw_matrix,
            entry["band_idx"],
            entry["wavelength"],
            entry["label"],
        )
        idx += 1
        if idx >= axes.size:
            break
        plot_prisma_cw_fwhm_ax(
            axes[idx],
            cw_matrix,
            fwhm_matrix,
            entry["band_idx"],
            entry["wavelength"],
            entry["label"],
        )
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


# --------------------------- Main runner ---------------------------


def run_prisma_smile(l1_file, vnir_threshold_nm=1000.0, vnir_band_index=None, swir_band_index=None):
    """Load a PRISMA L1 scene and visualize mean spectra plus smile diagnostics."""

    he5_path = _resolve_l1_path(l1_file)
    cube, cw_matrix, fwhm_matrix, *_ = prisma_utils.prisma_read(he5_path)
    rad = np.transpose(cube, (2, 0, 1))

    mean_cw = np.nanmean(cw_matrix, axis=0)
    _, mean_spec = mean_spectrum(rad, mean_cw)

    mean_entries = [
        {
            "cw": mean_cw,
            "spec": mean_spec,
            "title": "Mean TOA Radiance — Full range",
        }
    ]

    vnir_mask = mean_cw <= vnir_threshold_nm
    if np.any(vnir_mask):
        mean_entries.append(
            {
                "cw": mean_cw[vnir_mask],
                "spec": mean_spec[vnir_mask],
                "title": "Mean TOA Radiance — VNIR",
            }
        )

    swir_mask = mean_cw > vnir_threshold_nm
    if np.any(swir_mask):
        mean_entries.append(
            {
                "cw": mean_cw[swir_mask],
                "spec": mean_spec[swir_mask],
                "title": "Mean TOA Radiance — SWIR",
            }
        )

    smile_entries = []

    def _append_smile_entry(band_index, label):
        if band_index is None:
            return
        if not (1 <= band_index <= mean_cw.size):
            raise ValueError(f"Band index must be in 1..{mean_cw.size}")
        smile_entries.append(
            {
                "band_idx": band_index,
                "label": label,
                "wavelength": float(mean_cw[band_index - 1]),
            }
        )

    _append_smile_entry(vnir_band_index, "VNIR — ")
    _append_smile_entry(swir_band_index, "SWIR — ")

    render_summary_plots(mean_entries, smile_entries, cw_matrix, fwhm_matrix)


# --------------------------- Example entry ---------------------------


if __name__ == "__main__":
    l1_example = (
        "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/"
        "Roger_et_al_PRISMA_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/"
        "PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip"
    )

    run_prisma_smile(
        l1_example,
        vnir_threshold_nm=1000.0,
        vnir_band_index=40,
        swir_band_index=140,
    )
