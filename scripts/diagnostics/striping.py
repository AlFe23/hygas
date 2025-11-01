"""
Light destriping utilities: column equalisation, narrow-band FFT notch filters,
and diagnostics to quantify striping strength.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.signal.windows import hann

from ..core.noise import EPS


def equalize_columns(img2d: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply robust per-column gain/offset normalisation.

    Parameters
    ----------
    img2d : np.ndarray
        Input image (rows, cols).
    mask : np.ndarray | None
        Boolean mask selecting valid pixels. If None, all finite pixels are used.

    Returns
    -------
    tuple
        Equalised image and a dictionary containing column statistics.
    """

    arr = img2d.astype(float)
    rows, cols = arr.shape

    if mask is None:
        mask_bool = np.isfinite(arr)
    else:
        mask_bool = mask.astype(bool)
        mask_bool &= np.isfinite(arr)

    global_vals = arr[mask_bool]
    global_med = np.nanmedian(global_vals)
    global_mad = np.nanmedian(np.abs(global_vals - global_med)) + EPS

    adjusted = arr.copy()
    col_med = np.full(cols, np.nan, dtype=float)
    col_mad = np.full(cols, np.nan, dtype=float)
    col_scale = np.ones(cols, dtype=float)

    mad_floor = max(global_mad * 0.05, 1e-6)
    scale_cap = 5.0

    for c in range(cols):
        col_mask = mask_bool[:, c]
        if not np.any(col_mask):
            continue

        values = adjusted[:, c]
        med = np.nanmedian(values[col_mask])
        mad = np.nanmedian(np.abs(values[col_mask] - med)) + EPS

        if mad < mad_floor or not np.isfinite(mad):
            scale = 1.0
            adjusted[:, c] = global_med + (values - med)
        else:
            scale = global_mad / mad
            scale = float(np.clip(scale, 1.0 / scale_cap, scale_cap))
            adjusted[:, c] = global_med + (values - med) * scale

        col_med[c] = med
        col_mad[c] = mad
        col_scale[c] = scale

    stats = {
        "col_median": col_med,
        "col_mad": col_mad,
        "global_median": np.array([global_med]),
        "global_mad": np.array([global_mad]),
        "col_scale": col_scale,
        "mad_floor": mad_floor,
    }
    return adjusted, stats


def _prepare_rows(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    rows, _ = arr.shape
    cleaned = arr.astype(float).copy()
    if mask is not None:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = np.isfinite(cleaned)

    for r in range(rows):
        row_mask = mask_bool[r]
        row_vals = cleaned[r]
        finite = row_mask & np.isfinite(row_vals)
        if not np.any(finite):
            cleaned[r] = 0.0
            continue
        mean = np.nanmean(row_vals[row_mask])
        cleaned[r] = np.where(row_mask, row_vals - mean, 0.0)
    return cleaned


def detect_stripe_frequency(
    img2d: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ignore_bins: int = 2,
) -> Dict[str, np.ndarray | float | None]:
    """
    Analyse the row-wise FFT power spectrum to detect striping frequency.
    """

    rows, cols = img2d.shape
    window = hann(cols, sym=False)
    prepared = _prepare_rows(img2d, mask)

    fft_vals = np.fft.rfft(prepared * window[None, :], axis=1)
    power = np.nanmean(np.abs(fft_vals) ** 2, axis=0)
    freqs = np.fft.rfftfreq(cols, d=1.0)

    idx_start = min(len(power), max(ignore_bins, 1))
    if idx_start >= len(power):
        return {"freqs": freqs, "power": power, "peak_freq": None, "peak_db": None}

    baseline = np.median(power[idx_start:]) + EPS
    peak_idx = idx_start + int(np.argmax(power[idx_start:]))
    peak_power = power[peak_idx]

    if peak_power <= 0:
        peak_freq = None
        peak_db = None
    else:
        peak_freq = freqs[peak_idx]
        peak_db = 10.0 * np.log10(peak_power / baseline)

    return {
        "freqs": freqs,
        "power": power,
        "peak_freq": peak_freq,
        "peak_db": peak_db,
    }


def fft_notch_rowwise(
    img2d: np.ndarray,
    f0: float,
    df: float = 0.02,
    attenuation_db: float = 30.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a cosine-tapered notch filter at frequency f0 (cycles/pixel).
    """

    if f0 is None:
        return img2d.astype(float)

    rows, cols = img2d.shape
    freqs = np.fft.rfftfreq(cols, d=1.0)
    atten = 10 ** (-attenuation_db / 20.0)

    weights = np.ones_like(freqs)
    band = np.abs(freqs - f0) <= df
    if not np.any(band):
        return img2d.astype(float)

    taper = 0.5 * (1 + np.cos(np.pi * (freqs[band] - f0) / (df + EPS)))
    weights[band] = 1 - (1 - atten) * taper

    prepared = _prepare_rows(img2d, mask)
    filtered = np.zeros_like(prepared)

    for r in range(rows):
        row = prepared[r]
        fft_row = np.fft.rfft(row)
        fft_row *= weights
        recon = np.fft.irfft(fft_row, n=cols)

        # Restore original mean on the valid support
        if mask is not None:
            row_mask = mask[r]
            mean_original = np.nanmean(np.where(row_mask, img2d[r], np.nan))
            filtered[r] = np.where(row_mask, recon + mean_original, img2d[r])
        else:
            mean_original = np.nanmean(img2d[r])
            filtered[r] = recon + mean_original

    return filtered


def light_destripe_band(
    img2d: np.ndarray,
    mask: Optional[np.ndarray] = None,
    notch_df: float = 0.02,
    attenuation_db: float = 30.0,
    min_peak_db: float = 5.0,
    use_notch: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Apply column equalisation and (optionally) an FFT notch if a narrow stripe
    peak is detected.
    """

    equalised, stats = equalize_columns(img2d, mask=mask)
    fft_plain = detect_stripe_frequency(img2d, mask=mask)

    fft_eq = detect_stripe_frequency(equalised, mask=mask)
    f0 = fft_eq.get("peak_freq") or fft_plain.get("peak_freq")
    peak_db = fft_eq.get("peak_db")

    if use_notch and f0 is not None and peak_db is not None and peak_db >= min_peak_db:
        destriped = fft_notch_rowwise(equalised, f0, df=notch_df, attenuation_db=attenuation_db, mask=mask)
        fft_ds = detect_stripe_frequency(destriped, mask=mask)
        destripe_mode = "equalization + notch"
    else:
        destriped = equalised
        fft_ds = fft_eq
        destripe_mode = "equalization only"

    info = {
        "equalised": equalised,
        "stats": stats,
        "fft_plain": fft_plain,
        "fft_equalised": fft_eq,
        "fft_destriped": fft_ds,
        "f0_plain": fft_plain.get("peak_freq"),
        "f0_destriped": fft_ds.get("peak_freq"),
        "destripe_mode": destripe_mode,
    }
    return destriped, info


def light_destripe_cube(
    cube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    notch_df: float = 0.02,
    attenuation_db: float = 30.0,
    min_peak_db: float = 5.0,
    use_notch: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Apply light destriping band-by-band to a cube shaped (bands, rows, cols).
    """

    bands, rows, cols = cube.shape
    destriped = np.zeros_like(cube, dtype=float)
    diagnostics: List[Dict[str, object]] = []

    for b in range(bands):
        clean, info = light_destripe_band(
            cube[b],
            mask=mask,
            notch_df=notch_df,
            attenuation_db=attenuation_db,
            min_peak_db=min_peak_db,
            use_notch=use_notch,
        )
        destriped[b] = clean
        diagnostics.append(info)

    return destriped, diagnostics
