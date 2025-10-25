# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:26:23 2025

@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Compute SNR over a homogeneous, plume-free area and plot mean radiance & SNR.

This follows the EnMAP/PRISMA paper methodology:
  - Work in at-sensor radiance (L1B).
  - Select bright, homogeneous, plume-free pixels.
  - Estimate per-band noise std with detrending (first differences or high-pass).
  - SNR_ref = L_mean / sigmaMN   (per band, preferably per column).
  - (Optionally) scale SNR to other scenes with sqrt(L/L_ref).

Requirements:
 - enmap_smile.py (helpers for file discovery, metadata parsing, DN→radiance)
 - numpy, matplotlib
 - scipy.ndimage (for Gaussian blur & morphology)
 - scikit-image (optional; only for Sobel edge; code falls back if absent)

Author: you :)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from enmap_smile import (
    find_enmap_files, parse_metadata_vnir_swir, read_cube_gdal, dn_to_radiance
)

# Optional: scikit-image for Sobel (edge magnitude)
try:
    from skimage.filters import sobel
except Exception:
    sobel = None

# SciPy for Gaussian blurs and morphology
try:
    from scipy.ndimage import (
        gaussian_filter, gaussian_filter1d,
        binary_opening, binary_closing
    )
except Exception as _e:
    raise ImportError(
        "This script requires scipy.ndimage. Please install SciPy.\n"
        f"Original error: {_e}"
    )


# ------------------------------
# utility: unit conversions
# ------------------------------
def convert_radiance_units(L_mean, unit):
    """
    Convert radiance for plotting axes (SNR is unit-invariant).
    Input L_mean is in native [W m^-2 sr^-1 nm^-1] (from dn_to_radiance).
    """
    if unit == "W_m2_sr_nm":
        return L_mean, "W m$^{-2}$ sr$^{-1}$ nm$^{-1}$", 1.0
    elif unit == "uW_cm2_sr_nm":
        # 1 W/m^2 = 100 µW/cm^2
        return L_mean * 1e2, "µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$", 1e2
    elif unit == "W_m2_sr_um":
        # per nm -> per µm (×1000)
        return L_mean * 1e3, "W m$^{-2}$ sr$^{-1}$ µm$^{-1}$", 1e3
    else:
        raise ValueError("radiance_unit must be one of: "
                         "'W_m2_sr_nm', 'uW_cm2_sr_nm', 'W_m2_sr_um'.")


# ------------------------------
# band selection
# ------------------------------
def band_selector(cw_nm, window_nm=None):
    cw = np.asarray(cw_nm, dtype=float)
    if window_nm is None:
        return np.ones_like(cw, dtype=bool)
    lo, hi = window_nm
    return (cw >= lo) & (cw <= hi)


# ------------------------------
# homogeneous mask (auto)
# ------------------------------
def build_homogeneous_mask_auto(rad_cube, frac_keep=0.10, edge_wt=0.5, k=9):
    """
    Auto-select homogeneous pixels using low local variance + low edge magnitude.

    Parameters
    ----------
    rad_cube : (B, R, C) radiance in the target window (e.g., 2280–2380 nm)
    frac_keep : float in (0,1), fraction of flattest pixels to keep
    edge_wt   : [0..1], relative weight between edges and local variance
    k         : int, local window size for variance/mean

    Returns
    -------
    mask : (R, C) boolean
    """
    B, R, C = rad_cube.shape
    img = np.nanmean(rad_cube, axis=0)  # collapse to 2D (mean across bands)

    # local mean / variance via box filter in frequency domain (fast)
    pad = k // 2
    pad_img = np.pad(img, pad, mode='reflect')
    box = np.ones((k, k), dtype=float) / (k * k)
    local_mean = np.real(np.fft.ifft2(np.fft.fft2(pad_img) * np.fft.fft2(box, s=pad_img.shape)))
    local_mean = local_mean[pad:-pad, pad:-pad]
    local_var = (img - local_mean) ** 2
    # smooth the variance
    local_var = np.real(np.fft.ifft2(np.fft.fft2(local_var) * np.fft.fft2(box, s=local_var.shape)))

    # edge magnitude (Sobel if available, else gradient magnitude)
    if sobel is not None:
        edge_mag = sobel(img.astype(np.float32))
    else:
        gy, gx = np.gradient(img.astype(np.float32))
        edge_mag = np.hypot(gx, gy)

    # normalize to [0,1] robustly (1–99 percentiles)
    def norm01(a):
        p1, p99 = np.nanpercentile(a, (1, 99))
        return np.clip((a - p1) / (p99 - p1 + 1e-12), 0, 1)

    v_n = norm01(local_var)
    e_n = norm01(edge_mag)

    # lower score => flatter region
    score = edge_wt * e_n + (1 - edge_wt) * v_n
    thr = np.quantile(score, frac_keep)
    mask = score <= thr

    # morphological cleanup
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((5, 5)))
    return mask


# ------------------------------
# robust sigma estimators
# ------------------------------
def _sigma_from_first_diff_2d(img2d, axis=0, mask=None):
    """Noise std from first differences along an axis, divided by sqrt(2)."""
    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    dif = np.diff(arr, axis=axis)
    return np.nanstd(dif, ddof=1) / np.sqrt(2.0)

def _sigma_from_highpass_2d(img2d, mask=None, kxy=31):
    """Noise std after removing low-frequency content via Gaussian blur subtract."""
    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    # quick masked Gaussian normalization to fill NaNs
    m = np.isnan(arr)
    if np.any(m):
        tmp = arr.copy(); tmp[m] = 0.0
        w = (~m).astype(float)
        num = gaussian_filter(tmp, sigma=kxy/6.0, mode='nearest')
        den = gaussian_filter(w,   sigma=kxy/6.0, mode='nearest') + 1e-12
        arr = num / den
    low = gaussian_filter(arr, sigma=kxy/6.0, mode='nearest')
    res = arr - low
    return np.nanstd(res, ddof=1)

def estimate_sigmaMN_cube(rad_cube, mask=None, mode="diff", axis=0, kxy=31):
    """
    Estimate per-band noise std (sigmaMN) from a radiance cube (B, R, C).
    mode: 'diff' (first-diff/√2), 'hp' (high-pass), 'std' (plain std; use only for VERY flat masks)
    axis: 0=along-track diffs (rows), 1=across-track diffs (cols)
    """
    B, R, C = rad_cube.shape
    sig = np.zeros(B, dtype=np.float64)
    for b in range(B):
        img = rad_cube[b]
        if mode == "diff":
            sig[b] = _sigma_from_first_diff_2d(img, axis=axis, mask=mask)
        elif mode == "hp":
            sig[b] = _sigma_from_highpass_2d(img, mask=mask, kxy=kxy)
        elif mode == "std":
            vals = (img[mask] if mask is not None else img.ravel()).astype(float)
            vals = vals[np.isfinite(vals)]
            sig[b] = np.std(vals, ddof=1) if vals.size > 1 else np.nan
        else:
            raise ValueError("mode must be 'diff', 'hp', or 'std'")
    return sig


# ------------------------------
# SNR computation (mask-wide)
# ------------------------------
def compute_snr_over_mask(rad_cube, cw_nm, mask_bool,
                          sigma_mode="diff", diff_axis=0, hp_kxy=31):
    """
    Compute per-band mean radiance & sigmaMN over a homogeneous mask, then SNR.
    Detrending via 'diff' (first-difference) or 'hp' (high-pass) is recommended.
    """
    B, R, C = rad_cube.shape
    m = mask_bool.astype(bool)

    # per-band mean radiance over mask
    L_mean = np.nanmean(np.where(m[None, ...], rad_cube, np.nan), axis=(1, 2))

    # robust noise std per band
    if sigma_mode == "diff":
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="diff", axis=diff_axis)
    elif sigma_mode == "hp":
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="hp", kxy=hp_kxy)
    else:
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="std")

    SNR = L_mean / (sigmaMN + 1e-12)
    return np.asarray(cw_nm), L_mean, sigmaMN, SNR


# ------------------------------
# SNR computation (per column → column-median curve)
# ------------------------------
def compute_snr_per_column(rad_cube, cw_nm, mask_bool,
                           sigma_mode="diff", hp_kxy=31):
    """
    Compute SNR per band PER DETECTOR COLUMN, then take column-median curve.
    This follows the paper’s recommendation to mitigate striping.
    """
    B, R, C = rad_cube.shape
    m = mask_bool.astype(bool)

    L_mean_cols = np.full((B, C), np.nan, dtype=float)
    sig_cols = np.full((B, C), np.nan, dtype=float)

    for col in range(C):
        col_mask = m[:, col]
        if not np.any(col_mask):
            continue
        # For each band, compute mean & sigma on this single column
        for b in range(B):
            col_vec = rad_cube[b, :, col].astype(float)
            vals = col_vec.copy()
            vals[~col_mask] = np.nan

            # mean over masked rows
            L_mean_cols[b, col] = np.nanmean(vals)

            # sigma via chosen detrending
            if sigma_mode == "diff":
                # 1D first differences along rows
                dif = np.diff(vals)
                sig_cols[b, col] = np.nanstd(dif, ddof=1) / np.sqrt(2.0)
            elif sigma_mode == "hp":
                # 1D gaussian high-pass along rows
                arr = vals.copy()
                m1 = np.isnan(arr)
                if np.any(m1):
                    tmp = arr.copy(); tmp[m1] = 0.0
                    w = (~m1).astype(float)
                    num = gaussian_filter1d(tmp, sigma=hp_kxy/6.0, axis=0, mode='nearest')
                    den = gaussian_filter1d(w,   sigma=hp_kxy/6.0, axis=0, mode='nearest') + 1e-12
                    arr = num / den
                low = gaussian_filter1d(arr, sigma=hp_kxy/6.0, axis=0, mode='nearest')
                res = arr - low
                sig_cols[b, col] = np.nanstd(res, ddof=1)
            else:
                vals2 = vals[np.isfinite(vals)]
                sig_cols[b, col] = np.std(vals2, ddof=1) if vals2.size > 1 else np.nan

    # Median across columns → robust column-aggregate curves
    L_mean = np.nanmedian(L_mean_cols, axis=1)
    sigmaMN = np.nanmedian(sig_cols, axis=1)
    SNR = L_mean / (sigmaMN + 1e-12)
    return np.asarray(cw_nm), L_mean, sigmaMN, SNR


# ------------------------------
# plotting
# ------------------------------
def plot_radiance_and_snr(
    cw_nm,
    L_mean,
    SNR,
    title="Homogeneous-area SNR",
    radiance_unit="W_m2_sr_um"
):
    L_plot, label_unit, scale_factor = convert_radiance_units(L_mean, radiance_unit)

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    # Left axis → SNR
    ax1.plot(cw_nm, SNR, lw=1.4, color="C0", label="SNR")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("SNR", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)

    # Right axis → Radiance
    ax2 = ax1.twinx()
    ax2.plot(cw_nm, L_plot, lw=1.2, ls="--", color="C1", label="Mean radiance")
    ax2.set_ylabel(f"Radiance ({label_unit})", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    fig.suptitle(
        f"{title}\n(radiance scaled by {scale_factor:g} for unit '{radiance_unit}')",
        fontsize=10
    )
    fig.tight_layout()
    plt.show()


# ------------------------------
# SNR scaling law (photon-limited)
# ------------------------------
def scale_snr_to_scene(SNR_ref, L_mean_ref, L_mean_scene):
    """
    Photon-limited scaling of SNR from reference → target:
      SNR_scene = SNR_ref * sqrt(L_scene / L_ref)
    All inputs are 1D arrays per band.
    """
    return SNR_ref * np.sqrt((L_mean_scene + 1e-12) / (L_mean_ref + 1e-12))


# ------------------------------
# main runner
# ------------------------------
def run_snr_homogeneous(
    in_dir,
    sensor="SWIR",
    window_nm=(2280, 2380),
    auto_mask=True,
    provided_mask=None,
    frac_keep=0.10,
    sigma_mode="diff",        # 'diff' (recommended) or 'hp' or 'std'
    per_column=True,          # per-detector-column SNR aggregation
    hp_kxy=31,
    radiance_unit="W_m2_sr_um"
):
    """
    Compute SNR over a homogeneous area for VNIR or SWIR.
    """
    vnir_path, swir_path, xml_path = find_enmap_files(in_dir)
    vnir_meta, swir_meta = parse_metadata_vnir_swir(xml_path)

    if sensor.upper() == "VNIR":
        dn = read_cube_gdal(vnir_path)
        rad = dn_to_radiance(dn, vnir_meta)  # -> [W m^-2 sr^-1 nm^-1]
        cw = np.array([m['cw_nm'] for m in vnir_meta], dtype=float)
        sens_label = "VNIR"
    else:
        dn = read_cube_gdal(swir_path)
        rad = dn_to_radiance(dn, swir_meta)
        cw = np.array([m['cw_nm'] for m in swir_meta], dtype=float)
        sens_label = "SWIR"

    # select methane window (or all)
    sel = band_selector(cw, window_nm)
    rad_win = rad[sel]
    cw_win = cw[sel]

    # build homogeneous mask
    if provided_mask is not None:
        mask = provided_mask.astype(bool)
    elif auto_mask:
        mask = build_homogeneous_mask_auto(rad_win, frac_keep=frac_keep)
    else:
        raise ValueError("Either auto_mask=True or provide 'provided_mask'.")

    # compute SNR
    if per_column:
        cw_sel, L_mean, sigmaMN, SNR = compute_snr_per_column(
            rad_win, cw_win, mask, sigma_mode=sigma_mode, hp_kxy=hp_kxy
        )
    else:
        cw_sel, L_mean, sigmaMN, SNR = compute_snr_over_mask(
            rad_win, cw_win, mask, sigma_mode=sigma_mode, diff_axis=0, hp_kxy=hp_kxy
        )

    # plot (SNR left; radiance right, chosen units)
    lohi = f"{int(cw_sel.min())}-{int(cw_sel.max())} nm" if window_nm else "full range"
    plot_radiance_and_snr(
        cw_sel, L_mean, SNR,
        title=f"{sens_label} homogeneous-area SNR • window: {lohi}",
        radiance_unit=radiance_unit
    )

    return {
        "cw_nm": cw_sel,
        "L_mean": L_mean,        # [W m^-2 sr^-1 nm^-1]
        "sigmaMN": sigmaMN,      # same units as L_mean
        "SNR": SNR,              # dimensionless
        "mask": mask,            # (R, C) boolean
        "sensor": sens_label,
        "window_nm": window_nm
    }


# ------------------------------
# example
# ------------------------------
if __name__ == "__main__":
    main_dir = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\codes\test_data\20221002T074828\L1B-DT0000004147_20221002T074828Z_001_V010501_20241110T222720Z"

    out = run_snr_homogeneous(
        in_dir=main_dir,
        sensor="SWIR",
        window_nm=None,
        #window_nm=(2280, 2380),    # methane window; set None for full range
        auto_mask=True,
        frac_keep=0.12,             # keep ~12% flattest pixels
        sigma_mode="diff",          # 'diff' recommended; try 'hp' if strong striping
        per_column=True,            # per-column aggregation
        hp_kxy=31,
        radiance_unit="W_m2_sr_um"  # right-axis units (SNR left axis is unitless)
    )

    # Example of scaling SNR to another scene (photon-limited law):
    # out_ref = out  # suppose this is a desert reference
    # out_tar = ...  # recompute on a target scene to get L_mean_tar
    # SNR_tar = scale_snr_to_scene(out["SNR"], out["L_mean"], out_tar["L_mean"])
