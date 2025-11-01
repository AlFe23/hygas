"""
Noise estimation and SNR aggregation utilities shared across sensors.

This module collects the homogeneous-mask logic and sigma estimators that were
previously embedded in the individual SNR scripts. Functions operate on data
 cubes shaped as (bands, rows, cols).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # optional dependency for edge magnitude
    from skimage.filters import sobel
except Exception:  # pragma: no cover - optional
    sobel = None

from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, gaussian_filter1d

EPS = 1e-12


def build_homogeneous_mask_auto(
    rad_cube: np.ndarray,
    frac_keep: float = 0.12,
    edge_wt: float = 0.5,
    k: int = 9,
) -> np.ndarray:
    """
    Select a homogeneous mask by combining local variance and edge magnitude.

    Parameters
    ----------
    rad_cube : np.ndarray
        Radiance cube with shape (bands, rows, cols). Only the mean across bands
        is used to score pixels.
    frac_keep : float
        Fraction (0–1) of lowest-score pixels to keep.
    edge_wt : float
        Weight applied to edge magnitude relative to local variance.
    k : int
        Window size (odd) for local box filtering.

    Returns
    -------
    np.ndarray
        Boolean mask with shape (rows, cols).
    """

    if rad_cube.ndim != 3:
        raise ValueError("Expected a 3-D cube shaped as (bands, rows, cols).")

    _, rows, cols = rad_cube.shape
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

    def _norm01(a: np.ndarray) -> np.ndarray:
        p1, p99 = np.nanpercentile(a, (1, 99))
        return np.clip((a - p1) / (p99 - p1 + EPS), 0, 1)

    v_n = _norm01(local_var)
    e_n = _norm01(edge_mag)
    score = edge_wt * e_n + (1 - edge_wt) * v_n
    threshold = np.quantile(score, frac_keep)
    mask = score <= threshold

    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((5, 5)))
    return mask


def first_diff_sigma(img2d: np.ndarray, axis: int = 0, mask: Optional[np.ndarray] = None) -> float:
    """Estimate noise sigma via first differences along the chosen axis."""

    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    diff = np.diff(arr, axis=axis)
    return np.nanstd(diff, ddof=1) / np.sqrt(2.0)


def _masked_gaussian(arr: np.ndarray, sigma: float, mode: str = "nearest") -> np.ndarray:
    """Gaussian blur that respects NaNs by normalising with the weight image."""

    mask = ~np.isfinite(arr)
    if not np.any(mask):
        return gaussian_filter(arr, sigma=sigma, mode=mode)

    tmp = arr.copy()
    tmp[mask] = 0.0
    weights = (~mask).astype(float)

    num = gaussian_filter(tmp, sigma=sigma, mode=mode)
    den = gaussian_filter(weights, sigma=sigma, mode=mode) + EPS
    return num / den


def highpass_sigma(img2d: np.ndarray, mask: Optional[np.ndarray] = None, kxy: int = 31) -> float:
    """Estimate noise sigma using a Gaussian high-pass residual."""

    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)

    sigma = kxy / 6.0
    low = _masked_gaussian(arr, sigma=sigma)
    residual = arr - low
    return np.nanstd(residual, ddof=1)


def _masked_std(img2d: np.ndarray, mask: Optional[np.ndarray]) -> float:
    arr = img2d.astype(float)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    vals = arr[np.isfinite(arr)]
    return np.std(vals, ddof=1) if vals.size > 1 else np.nan


def estimate_sigma_cube(
    cube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mode: str = "diff",
    diff_axis: int = 0,
    kxy: int = 31,
) -> np.ndarray:
    """
    Estimate per-band sigma for the provided cube (bands, rows, cols).

    Parameters
    ----------
    cube : np.ndarray
        Input cube from which sigma is estimated. This is typically the original
        radiance (for total noise) or the PCA residuals (for random noise).
    mask : np.ndarray | None
        Boolean mask selecting valid pixels.
    mode : str
        Either ``"diff"`` (first differences), ``"hp"`` (Gaussian high-pass), or
        ``"std"`` (plain standard deviation).
    diff_axis : int
        Axis for the first-difference estimator (0=rows, 1=cols).
    kxy : int
        Kernel size for the high-pass method.
    """

    if cube.ndim != 3:
        raise ValueError("Expected a cube shaped as (bands, rows, cols).")

    bands = cube.shape[0]
    sigma = np.zeros(bands, dtype=float)
    for b in range(bands):
        img = cube[b]
        if mode == "diff":
            sigma[b] = first_diff_sigma(img, axis=diff_axis, mask=mask)
        elif mode == "hp":
            sigma[b] = highpass_sigma(img, mask=mask, kxy=kxy)
        elif mode == "std":
            sigma[b] = _masked_std(img, mask)
        else:
            raise ValueError("mode must be one of {'diff', 'hp', 'std'}.")
    return sigma


def _column_sigma(
    values: np.ndarray,
    mask: np.ndarray,
    mode: str,
    hp_kxy: int,
) -> float:
    """Helper to compute sigma on a 1-D column with masking."""

    col = values.astype(float)
    col[~mask] = np.nan

    if mode == "diff":
        dif = np.diff(col)
        return np.nanstd(dif, ddof=1) / np.sqrt(2.0)

    if mode == "hp":
        sigma = hp_kxy / 6.0
        arr = col.copy()
        nan_mask = ~np.isfinite(arr)
        if np.any(nan_mask):
            tmp = arr.copy()
            tmp[nan_mask] = 0.0
            weights = (~nan_mask).astype(float)
            num = gaussian_filter1d(tmp, sigma=sigma, axis=0, mode="nearest")
            den = gaussian_filter1d(weights, sigma=sigma, axis=0, mode="nearest") + EPS
            arr = num / den
        low = gaussian_filter1d(arr, sigma=sigma, axis=0, mode="nearest")
        res = arr - low
        return np.nanstd(res, ddof=1)

    vals = col[np.isfinite(col)]
    return np.std(vals, ddof=1) if vals.size > 1 else np.nan


@dataclass
class SNRResult:
    band_nm: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    snr: np.ndarray
    snr_median: np.ndarray
    snr_p90: np.ndarray
    sigma_kind: str
    aggregation: str
    details: Dict[str, np.ndarray]


def snr_wholeroi(
    target_cube: np.ndarray,
    radiance_cube: np.ndarray,
    band_nm: np.ndarray,
    mask: Optional[np.ndarray],
    sigma_kind: str,
    sigma_mode: str = "diff",
    diff_axis: int = 0,
    hp_kxy: int = 31,
) -> SNRResult:
    """
    Compute SNR over the full mask/ROI.

    Parameters
    ----------
    target_cube : np.ndarray
        Cube used for sigma estimation (e.g. radiance or PCA residuals).
    radiance_cube : np.ndarray
        Original cube used to compute mean radiance (mu).
    band_nm : np.ndarray
        Wavelength per band (nm).
    mask : np.ndarray | None
        Boolean mask with shape (rows, cols).
    sigma_kind : str
        Either ``"total"`` or ``"random"`` (for labelling outputs).
    sigma_mode : str
        ``"diff"``, ``"hp"``, or ``"std"``.
    diff_axis : int
        Axis for first-difference estimator.
    hp_kxy : int
        Kernel size for the high-pass estimator.
    """

    mask_bool = mask.astype(bool) if mask is not None else np.ones(radiance_cube.shape[1:], dtype=bool)

    mu = np.nanmean(np.where(mask_bool[None, ...], radiance_cube, np.nan), axis=(1, 2))
    sigma = estimate_sigma_cube(target_cube, mask=mask_bool, mode=sigma_mode, diff_axis=diff_axis, kxy=hp_kxy)
    snr = mu / (sigma + EPS)

    return SNRResult(
        band_nm=np.asarray(band_nm, dtype=float),
        mu=mu,
        sigma=sigma,
        snr=snr,
        snr_median=snr,
        snr_p90=snr,
        sigma_kind=sigma_kind,
        aggregation="roi",
        details={"mask": mask_bool},
    )


def snr_columnwise(
    target_cube: np.ndarray,
    radiance_cube: np.ndarray,
    band_nm: np.ndarray,
    mask: Optional[np.ndarray],
    sigma_kind: str,
    sigma_mode: str = "diff",
    hp_kxy: int = 31,
    min_valid: int = 16,
) -> SNRResult:
    """
    Compute column-wise SNR curves then aggregate with median and P90.
    """

    mask_bool = mask.astype(bool) if mask is not None else np.ones(radiance_cube.shape[1:], dtype=bool)
    bands, rows, cols = target_cube.shape

    valid_counts = mask_bool.sum(axis=0)
    valid_cols = np.where(valid_counts >= min_valid)[0]
    if valid_cols.size == 0:
        raise RuntimeError("No detector columns have enough valid pixels for SNR estimation.")

    mu_cols = []
    sigma_cols = []
    snr_cols = []

    for col in valid_cols:
        col_mask = mask_bool[:, col]
        if not np.any(col_mask):
            continue

        mu_b = np.full(bands, np.nan, dtype=float)
        sigma_b = np.full(bands, np.nan, dtype=float)

        for b in range(bands):
            rad_vals = radiance_cube[b, :, col].astype(float)
            rad_vals[~col_mask] = np.nan
            mu_b[b] = np.nanmean(rad_vals)

            tgt_vals = target_cube[b, :, col].astype(float)
            sigma_b[b] = _column_sigma(tgt_vals, col_mask, sigma_mode, hp_kxy)

        mu_cols.append(mu_b)
        sigma_cols.append(sigma_b)
        snr_cols.append(mu_b / (sigma_b + EPS))

    mu_cols = np.stack(mu_cols, axis=0)
    sigma_cols = np.stack(sigma_cols, axis=0)
    snr_cols = np.stack(snr_cols, axis=0)

    snr_median = np.nanmedian(snr_cols, axis=0)
    snr_p90 = np.nanpercentile(snr_cols, 90, axis=0)

    mu_median = np.nanmedian(mu_cols, axis=0)
    sigma_median = np.nanmedian(sigma_cols, axis=0)

    return SNRResult(
        band_nm=np.asarray(band_nm, dtype=float),
        mu=mu_median,
        sigma=sigma_median,
        snr=np.nanmedian(snr_cols, axis=0),
        snr_median=snr_median,
        snr_p90=snr_p90,
        sigma_kind=sigma_kind,
        aggregation="columnwise",
        details={
            "valid_columns": valid_cols,
            "mu_columns": mu_cols,
            "sigma_columns": sigma_cols,
            "snr_columns": snr_cols,
        },
    )

