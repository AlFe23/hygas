# -*- coding: utf-8 -*-
"""
Signal-to-noise estimation for PRISMA L1 scenes. Mirrors the EnMAP-specific
workflow (homogeneous-mask selection, detrended noise estimates, optional
per-column aggregation) but uses the PRISMA HDF readers to obtain radiance
cubes together with band metadata.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.core import targets
from scripts.satellites import prisma_utils

# Optional: scikit-image for Sobel (edge magnitude)
try:
    from skimage.filters import sobel
except Exception:  # pragma: no cover - optional dependency
    sobel = None

# SciPy for Gaussian blurs and morphology
try:
    from scipy.ndimage import (
        gaussian_filter,
        gaussian_filter1d,
        binary_opening,
        binary_closing,
    )
except Exception as _e:  # pragma: no cover - hard dependency
    raise ImportError(
        "This script requires scipy.ndimage. Please install SciPy.\n"
        f"Original error: {_e}"
    )


# ------------------------------
# PRISMA I/O helpers
# ------------------------------


def _resolve_l1_path(l1_path: str) -> str:
    """Return an absolute path to the PRISMA HE5 file, extracting from ZIP if needed."""

    path = os.path.abspath(l1_path)
    if path.lower().endswith(".zip"):
        extracted = prisma_utils.extract_he5_from_zip(path, os.path.dirname(path))
        if extracted is None:
            raise FileNotFoundError(
                f"Could not locate a .he5 file inside ZIP {path}. Ensure the archive is valid."
            )
        return extracted
    return path


def _resolve_l2c_path(l2c_path: str | None) -> str | None:
    """Resolve PRISMA L2C path, extracting from ZIP when provided."""

    if l2c_path is None:
        return None
    path = os.path.abspath(l2c_path)
    if path.lower().endswith(".zip"):
        extracted = prisma_utils.extract_he5_from_zip(path, os.path.dirname(path))
        if extracted is None:
            raise FileNotFoundError(
                f"Could not locate a .he5 file inside ZIP {path}. Ensure the archive is valid."
            )
        return extracted
    return path


# ------------------------------
# plotting helpers
# ------------------------------


def convert_to_aviris_units(L_mean):
    """Return radiance expressed in µW cm⁻² sr⁻¹ nm⁻¹."""

    return L_mean


# ------------------------------
# homogeneous mask (auto)
# ------------------------------


def build_homogeneous_mask_auto(rad_cube, frac_keep=0.10, edge_wt=0.5, k=9):
    """Select the flattest pixels by combining local variance and edge magnitude."""

    _, R, C = rad_cube.shape
    img = np.nanmean(rad_cube, axis=0)

    pad = k // 2
    pad_img = np.pad(img, pad, mode="reflect")
    box = np.ones((k, k), dtype=float) / (k * k)
    local_mean = np.real(np.fft.ifft2(np.fft.fft2(pad_img) * np.fft.fft2(box, s=pad_img.shape)))
    local_mean = local_mean[pad:-pad, pad:-pad]
    local_var = (img - local_mean) ** 2
    local_var = np.real(np.fft.ifft2(np.fft.fft2(local_var) * np.fft.fft2(box, s=local_var.shape)))

    if sobel is not None:
        edge_mag = sobel(img.astype(np.float32))
    else:
        gy, gx = np.gradient(img.astype(np.float32))
        edge_mag = np.hypot(gx, gy)

    def _norm01(a):
        p1, p99 = np.nanpercentile(a, (1, 99))
        return np.clip((a - p1) / (p99 - p1 + 1e-12), 0, 1)

    v_n = _norm01(local_var)
    e_n = _norm01(edge_mag)
    score = edge_wt * e_n + (1 - edge_wt) * v_n
    mask = score <= np.quantile(score, frac_keep)
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((5, 5)))
    return mask


# ------------------------------
# robust sigma estimators
# ------------------------------


def _sigma_from_first_diff_2d(img2d, axis=0, mask=None):
    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    dif = np.diff(arr, axis=axis)
    return np.nanstd(dif, ddof=1) / np.sqrt(2.0)


def _sigma_from_highpass_2d(img2d, mask=None, kxy=31):
    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    m = np.isnan(arr)
    if np.any(m):
        tmp = arr.copy()
        tmp[m] = 0.0
        w = (~m).astype(float)
        num = gaussian_filter(tmp, sigma=kxy / 6.0, mode="nearest")
        den = gaussian_filter(w, sigma=kxy / 6.0, mode="nearest") + 1e-12
        arr = num / den
    low = gaussian_filter(arr, sigma=kxy / 6.0, mode="nearest")
    res = arr - low
    return np.nanstd(res, ddof=1)


def estimate_sigmaMN_cube(rad_cube, mask=None, mode="diff", axis=0, kxy=31):
    B, _, _ = rad_cube.shape
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


def compute_snr_over_mask(rad_cube, cw_nm, mask_bool, sigma_mode="diff", diff_axis=0, hp_kxy=31):
    B, _, _ = rad_cube.shape
    m = mask_bool.astype(bool)
    L_mean = np.nanmean(np.where(m[None, ...], rad_cube, np.nan), axis=(1, 2))

    if sigma_mode == "diff":
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="diff", axis=diff_axis)
    elif sigma_mode == "hp":
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="hp", kxy=hp_kxy)
    else:
        sigmaMN = estimate_sigmaMN_cube(rad_cube, mask=m, mode="std")

    SNR = L_mean / (sigmaMN + 1e-12)
    return np.asarray(cw_nm), L_mean, sigmaMN, SNR


# ------------------------------
# SNR computation (per column)
# ------------------------------


def compute_snr_per_column(rad_cube, cw_nm, mask_bool, sigma_mode="diff", hp_kxy=31):
    B, R, C = rad_cube.shape
    m = mask_bool.astype(bool)

    L_mean_cols = np.full((B, C), np.nan, dtype=float)
    sig_cols = np.full((B, C), np.nan, dtype=float)

    for col in range(C):
        col_mask = m[:, col]
        if not np.any(col_mask):
            continue
        for b in range(B):
            col_vec = rad_cube[b, :, col].astype(float)
            vals = col_vec.copy()
            vals[~col_mask] = np.nan
            L_mean_cols[b, col] = np.nanmean(vals)

            if sigma_mode == "diff":
                dif = np.diff(vals)
                sig_cols[b, col] = np.nanstd(dif, ddof=1) / np.sqrt(2.0)
            elif sigma_mode == "hp":
                arr = vals.copy()
                m1 = np.isnan(arr)
                if np.any(m1):
                    tmp = arr.copy()
                    tmp[m1] = 0.0
                    w = (~m1).astype(float)
                    num = gaussian_filter1d(tmp, sigma=hp_kxy / 6.0, axis=0, mode='nearest')
                    den = gaussian_filter1d(w, sigma=hp_kxy / 6.0, axis=0, mode='nearest') + 1e-12
                    arr = num / den
                low = gaussian_filter1d(arr, sigma=hp_kxy / 6.0, axis=0, mode='nearest')
                res = arr - low
                sig_cols[b, col] = np.nanstd(res, ddof=1)
            else:
                vals2 = vals[np.isfinite(vals)]
                sig_cols[b, col] = np.std(vals2, ddof=1) if vals2.size > 1 else np.nan

    L_mean = np.nanmedian(L_mean_cols, axis=1)
    sigmaMN = np.nanmedian(sig_cols, axis=1)
    SNR = L_mean / (sigmaMN + 1e-12)
    return np.asarray(cw_nm), L_mean, sigmaMN, SNR


# ------------------------------
# plotting entry point
# ------------------------------


def plot_radiance_and_snr(cw_nm, L_mean, SNR, title="Homogeneous-area SNR", metadata_lines=None):
    L_plot = convert_to_aviris_units(L_mean)

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(cw_nm, SNR, lw=1.4, color="C0", label="SNR")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("SNR", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cw_nm, L_plot, lw=1.2, ls="--", color="C1", label="Mean radiance")
    ax2.set_ylabel("Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    header = f"{title}\n(radiance in µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)"
    if metadata_lines:
        header = header + "\n" + "\n".join(metadata_lines)
        fig.suptitle(header, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.suptitle(header, fontsize=10)
        fig.tight_layout()
    plt.show()


# ------------------------------
# main runner
# ------------------------------


def run_snr_homogeneous(
    l1_file,
    l2c_file=None,
    window_nm=(2100, 2450),
    auto_mask=True,
    provided_mask=None,
    frac_keep=0.10,
    sigma_mode="diff",
    per_column=True,
    hp_kxy=31,
):
    """Compute PRISMA SNR statistics over a homogeneous area."""

    he5_path = _resolve_l1_path(l1_file)
    l2c_path = _resolve_l2c_path(l2c_file)

    metadata_lines: list[str] = []
    try:
        sza_l1 = prisma_utils.prismaL1_SZA_read(he5_path)
        metadata_lines.append(f"L1 Sun zenith angle: {sza_l1:.3f}°")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[PRISMA] Could not read Sun_zenith_angle from L1 file: {exc}")

    if l2c_path:
        geom_summary = prisma_utils.prisma_l2c_geometry_summary(l2c_path)
        sun = geom_summary.get("sun_angles", {})
        if sun.get("zenith_deg") is not None:
            line = f"L2C Sun zenith angle: {sun['zenith_deg']:.3f}°"
            print(f"[PRISMA] {line}")
            metadata_lines.append(line)
        if sun.get("azimuth_deg") is not None:
            line = f"L2C Sun azimuth angle: {sun['azimuth_deg']:.3f}°"
            print(f"[PRISMA] {line}")
            metadata_lines.append(line)

        for entry in geom_summary.get("datasets", {}).values():
            stats = entry["stats"]
            line = (
                f"{entry['label']}: mean={stats['mean']:.3f}°, median={stats['median']:.3f}°, "
                f"min={stats['min']:.3f}°, max={stats['max']:.3f}°"
            )
            print(f"[PRISMA] {line}")
            metadata_lines.append(line)

        rel_z_stats = geom_summary.get("relative_zenith_stats") or geom_summary.get("relative_zenith")
        if rel_z_stats:
            msg = f"Relative zenith (SZA−VZA) ≈ mean={rel_z_stats['mean']:.3f}°, median={rel_z_stats['median']:.3f}°"
            print(f"[PRISMA] {msg}")
            metadata_lines.append(msg)
        rel_az_stats = geom_summary.get("relative_azimuth_stats") or geom_summary.get("relative_azimuth_summary")
        if rel_az_stats:
            msg = f"Relative azimuth ≈ mean={rel_az_stats['mean']:.3f}°, median={rel_az_stats['median']:.3f}°"
            print(f"[PRISMA] {msg}")
            metadata_lines.append(msg)
    else:
        print("[PRISMA] L2C file not provided: geometric statistics omitted.")

    cube, cw_matrix, _, *_ = prisma_utils.prisma_read(he5_path)
    rad = np.transpose(cube, (2, 0, 1))  # (bands, rows, cols)
    cw = np.nanmean(cw_matrix, axis=0)

    if window_nm is None:
        idx = np.arange(cw.size)
    else:
        idx = targets.select_band_indices(cw, window_nm[0], window_nm[1])
        if idx.size == 0:
            raise ValueError(f"No bands found in window {window_nm} nm for PRISMA scene {l1_file}.")

    rad_win = rad[idx]
    cw_win = cw[idx]

    if provided_mask is not None:
        mask = provided_mask.astype(bool)
    elif auto_mask:
        mask = build_homogeneous_mask_auto(rad_win, frac_keep=frac_keep)
    else:
        raise ValueError("Either auto_mask=True or provide 'provided_mask'.")

    if per_column:
        cw_sel, L_mean, sigmaMN, SNR = compute_snr_per_column(
            rad_win, cw_win, mask, sigma_mode=sigma_mode, hp_kxy=hp_kxy
        )
    else:
        cw_sel, L_mean, sigmaMN, SNR = compute_snr_over_mask(
            rad_win, cw_win, mask, sigma_mode=sigma_mode, diff_axis=0, hp_kxy=hp_kxy
        )

    lohi = f"{int(cw_sel.min())}-{int(cw_sel.max())} nm" if window_nm else "full range"
    plot_radiance_and_snr(
        cw_sel,
        L_mean,
        SNR,
        title=f"PRISMA homogeneous-area SNR • window: {lohi}",
        metadata_lines=metadata_lines,
    )

    return {
        "cw_nm": cw_sel,
        "L_mean": L_mean,
        "sigmaMN": sigmaMN,
        "SNR": SNR,
        "mask": mask,
        "sensor": "PRISMA",
        "window_nm": window_nm,
    }


# ------------------------------
# example
# ------------------------------


if __name__ == "__main__":
    l1_example = (
        "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/"
        "Northern_State_Sudan_20200401/20200401085313_20200401085318/"
        "PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip"
    )
    l2c_example = (
        "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/"
        "Northern_State_Sudan_20200401/20200401085313_20200401085318/"
        "PRS_L2C_STD_20200401085313_20200401085318_0001.zip"
    )

    out = run_snr_homogeneous(
        l1_file=l1_example,
        l2c_file=l2c_example,
        window_nm=None, #(2000, 2450),
        auto_mask=True,
        frac_keep=0.12,
        sigma_mode="diff",
        per_column=True,
        hp_kxy=31,
    )
