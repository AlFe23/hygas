"""
Noise estimation and SNR aggregation utilities shared across sensors.

This module collects the homogeneous-mask logic and sigma estimators that were
previously embedded in the individual SNR scripts. Functions operate on data
 cubes shaped as (bands, rows, cols).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

try:  # optional dependency for edge magnitude
    from skimage.filters import sobel
except Exception:  # pragma: no cover - optional
    sobel = None

from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, gaussian_filter1d

EPS = 1e-12
MIN_SIGMA = 1e-3


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


@dataclass
class ColumnwiseSNRReference:
    """
    Container for column-wise SNR reference information derived from a homogeneous scene.

    Attributes
    ----------
    band_nm : np.ndarray
        Wavelength per band (nm) with shape (bands,).
    mean_radiance : np.ndarray
        Reference mean radiance per band/column with shape (bands, columns).
    snr : np.ndarray
        Reference signal-to-noise ratio per band/column with shape (bands, columns).
    valid_columns : np.ndarray
        Indices of detector columns that were directly observed when deriving the reference.
    metadata : Dict[str, object] | None
        Optional metadata captured when the reference dataset was generated.
    """

    band_nm: np.ndarray
    mean_radiance: np.ndarray
    snr: np.ndarray
    valid_columns: np.ndarray
    metadata: Optional[Dict[str, object]] = None
    band_indices: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        bands = np.asarray(self.band_nm, dtype=float).shape[0]
        mean_shape = np.asarray(self.mean_radiance).shape
        snr_shape = np.asarray(self.snr).shape
        if mean_shape != snr_shape:
            raise ValueError(
                f"mean_radiance shape {mean_shape} and snr shape {snr_shape} must match (bands, columns)."
            )
        if mean_shape[0] != bands:
            raise ValueError(
                f"First dimension of radiance/SNR arrays {mean_shape[0]} does not match band_nm length {bands}."
            )

        self.band_nm = np.asarray(self.band_nm, dtype=float)
        self.mean_radiance = np.asarray(self.mean_radiance, dtype=float)
        self.snr = np.asarray(self.snr, dtype=float)
        self.valid_columns = np.asarray(self.valid_columns, dtype=int)
        if self.band_indices is None:
            self.band_indices = np.arange(bands, dtype=int)
        else:
            self.band_indices = np.asarray(self.band_indices, dtype=int)
            if self.band_indices.shape[0] != bands:
                raise ValueError(
                    f"band_indices length {self.band_indices.shape[0]} does not match band count {bands}."
                )
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict if provided.")

    @property
    def sigma(self) -> np.ndarray:
        """Return reference noise sigma per band/column (µW cm⁻² sr⁻¹ nm⁻¹)."""

        return np.divide(
            self.mean_radiance,
            np.clip(self.snr, EPS, None),
            out=np.full_like(self.mean_radiance, np.nan, dtype=float),
        )

    def subset_bands(self, band_indices: Sequence[int]) -> "ColumnwiseSNRReference":
        """Return a new reference limited to the provided band indices."""

        idx = np.asarray(band_indices, dtype=int)
        return ColumnwiseSNRReference(
            band_nm=self.band_nm[idx],
            mean_radiance=self.mean_radiance[idx, :],
            snr=self.snr[idx, :],
            valid_columns=self.valid_columns,
            metadata=self.metadata,
            band_indices=self.band_indices[idx],
        )

    def subset_by_wavelengths(
        self,
        wavelengths: Sequence[float],
        *,
        atol: float = 1,
    ) -> "ColumnwiseSNRReference":
        """
        Return a reference restricted to the provided wavelength grid.

        Parameters
        ----------
        wavelengths : Sequence[float]
            Target wavelengths (nm). Each will be matched against the stored
            reference wavelengths; the nearest band within ``atol`` nanometers
            is selected.
        atol : float
            Maximum allowed absolute difference (nm) between the requested
            wavelength and the reference band centre. Increase if the sensor
            wavelengths differ slightly between scenes.
        """

        target = np.asarray(wavelengths, dtype=float)
        if target.ndim != 1:
            raise ValueError("wavelengths must be a 1-D array.")

        ref_wl = self.band_nm
        indices = np.empty(target.size, dtype=int)
        for i, wl in enumerate(target):
            band_idx = int(np.argmin(np.abs(ref_wl - wl)))
            delta = abs(ref_wl[band_idx] - wl)
            if delta > atol:
                raise ValueError(
                    f"No reference band within {atol} nm for wavelength {wl:.3f} nm (closest offset {delta:.3f} nm)."
                )
            indices[i] = band_idx

        return ColumnwiseSNRReference(
            band_nm=ref_wl[indices],
            mean_radiance=self.mean_radiance[indices, :],
            snr=self.snr[indices, :],
            valid_columns=self.valid_columns,
            metadata=self.metadata,
            band_indices=self.band_indices[indices],
        )

    def subset_by_band_numbers(self, band_numbers: Sequence[int]) -> "ColumnwiseSNRReference":
        """
        Return a reference restricted to the provided detector band indices.
        """

        band_numbers = np.asarray(band_numbers, dtype=int)
        if band_numbers.ndim != 1:
            raise ValueError("band_numbers must be a 1-D array.")

        lookup = {int(bn): idx for idx, bn in enumerate(self.band_indices)}
        positions: list[int] = []
        missing: list[int] = []
        for bn in band_numbers:
            idx = lookup.get(int(bn))
            if idx is None:
                missing.append(int(bn))
            else:
                positions.append(idx)
        if missing:
            raise ValueError(f"Reference missing detector bands: {missing}")
        pos_arr = np.asarray(positions, dtype=int)
        return ColumnwiseSNRReference(
            band_nm=self.band_nm[pos_arr],
            mean_radiance=self.mean_radiance[pos_arr, :],
            snr=self.snr[pos_arr, :],
            valid_columns=self.valid_columns,
            metadata=self.metadata,
            band_indices=self.band_indices[pos_arr],
        )

    def ensure_column_count(self, target_cols: int) -> "ColumnwiseSNRReference":
        """
        Return a reference whose column dimension matches `target_cols`.

        Data are resampled along the across-track dimension using linear interpolation
        when the requested number of columns differs from the stored reference.
        """

        current_cols = self.mean_radiance.shape[1]
        if current_cols == target_cols:
            return self

        src = np.linspace(0.0, 1.0, current_cols)
        dst = np.linspace(0.0, 1.0, target_cols)

        def _resample(arr: np.ndarray) -> np.ndarray:
            out = np.empty((arr.shape[0], target_cols), dtype=float)
            for i in range(arr.shape[0]):
                row = arr[i]
                mask = np.isfinite(row)
                if not mask.any():
                    out[i] = 0.0
                    continue
                filled = row.copy()
                # Replace NaNs with nearest valid values before interpolation
                if not mask.all():
                    valid_idx = np.flatnonzero(mask)
                    filled = np.interp(np.arange(row.size), valid_idx, row[mask])
                out[i] = np.interp(dst, src, filled)
            return out

        mean_resampled = _resample(self.mean_radiance)
        snr_resampled = _resample(self.snr)
        return ColumnwiseSNRReference(
            band_nm=self.band_nm,
            mean_radiance=mean_resampled,
            snr=snr_resampled,
            valid_columns=np.round(dst * (target_cols - 1)).astype(int),
            metadata=self.metadata,
        )

    def save(self, path: str | Path) -> None:
        """Persist the reference dataset to disk as a compressed NPZ archive."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadata or {})
        np.savez_compressed(
            path,
            band_nm=self.band_nm.astype(np.float32),
            mean_radiance=self.mean_radiance.astype(np.float32),
            snr=self.snr.astype(np.float32),
            valid_columns=self.valid_columns.astype(np.int32),
            metadata_json=metadata_json,
            band_indices=self.band_indices.astype(np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ColumnwiseSNRReference":
        """Load a reference dataset previously stored via :meth:`save`."""

        path = Path(path)
        with np.load(path, allow_pickle=False) as npz:
            metadata_json = npz.get("metadata_json")
            if metadata_json is None:
                metadata = None
            else:
                if isinstance(metadata_json, np.ndarray):
                    metadata_str = metadata_json.item()
                else:
                    metadata_str = str(metadata_json)
                metadata = json.loads(metadata_str) if metadata_str else None
            band_indices = npz.get("band_indices")
            if band_indices is None:
                band_indices = None
            return cls(
                band_nm=npz["band_nm"],
                mean_radiance=npz["mean_radiance"],
                snr=npz["snr"],
                valid_columns=npz["valid_columns"],
                metadata=metadata,
                band_indices=band_indices,
            )


def compute_sigma_map_from_reference(
    reference: ColumnwiseSNRReference,
    radiance_cube: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel noise sigma (σ_MN) via radiance scaling from a reference SNR dataset.

    Parameters
    ----------
    reference : ColumnwiseSNRReference
        Column-wise reference dataset containing band-specific SNR and mean radiance.
    radiance_cube : np.ndarray
        Scene radiance cube shaped as (bands, rows, cols) matching the wavelength sampling
        of the reference.

    Returns
    -------
    np.ndarray
        σ_MN cube with the same shape as `radiance_cube`.
    """

    if radiance_cube.ndim != 3:
        raise ValueError("radiance_cube must be shaped as (bands, rows, cols).")

    bands, rows, cols = radiance_cube.shape
    ref = reference.ensure_column_count(cols)
    if ref.band_nm.shape[0] != bands:
        raise ValueError(
            f"Band count mismatch between radiance cube ({bands}) and reference ({ref.band_nm.shape[0]})."
        )

    ref_mu = np.clip(ref.mean_radiance[:, None, :], EPS, None)
    ref_snr = np.clip(ref.snr[:, None, :], EPS, None)

    positive_radiance = np.clip(radiance_cube, a_min=0.0, a_max=None)
    scale = np.sqrt(positive_radiance / ref_mu)
    np.clip(scale, 0.1, None, out=scale)
    snr_scene = ref_snr * scale
    sigma_mn = np.divide(
        radiance_cube,
        np.clip(snr_scene, EPS, None),
        out=np.zeros_like(radiance_cube, dtype=float),
    ).astype(np.float64)
    np.clip(sigma_mn, MIN_SIGMA, None, out=sigma_mn)
    return sigma_mn


def propagate_rmn_uncertainty(
    sigma_cube: np.ndarray,
    classified_image: np.ndarray,
    mean_radiance: np.ndarray,
    target_spectra: np.ndarray,
) -> np.ndarray:
    """
    Propagate matched-filter noise to per-pixel methane uncertainty via Roger et al. (σ_RMN).
    """
    bands, rows, cols = sigma_cube.shape
    k = mean_radiance.shape[0]
    result = np.full((rows, cols), np.nan, dtype=float)

    # This is a simplified calculation that assumes a diagonal noise covariance matrix
    # It is equivalent to: denom = t.T @ np.linalg.inv(C_N) @ t
    # where C_N is diagonal with sigma_cube**2 on the diagonal.

    if target_spectra.ndim == 1:
        t_lookup = mean_radiance * target_spectra[None, :]
        for cls in range(k):
            class_mask = (classified_image == cls)
            if not np.any(class_mask):
                continue
            
            t_vec = t_lookup[cls]
            # The denominator of the uncertainty equation
            denom = np.sum(t_vec[:, np.newaxis]**2 / (sigma_cube[:, class_mask]**2), axis=0)
            result[class_mask] = 1.0 / np.sqrt(np.clip(denom, 1e-12, None))

    elif target_spectra.ndim == 2:
        if target_spectra.shape[1] != cols:
            raise ValueError(
                f"Column-wise target spectra expect {cols} columns but received {target_spectra.shape[1]}."
            )
        for c in range(cols):
            for cls in range(k):
                class_mask = (classified_image[:, c] == cls)
                if not np.any(class_mask):
                    continue

                t_vec = mean_radiance[cls] * target_spectra[:, c]
                sigma_pix = sigma_cube[:, class_mask, c]

                # The denominator of the uncertainty equation
                denom = np.sum(t_vec[:, np.newaxis] ** 2 / (sigma_pix**2), axis=0)
                result[class_mask, c] = 1.0 / np.sqrt(np.clip(denom, 1e-12, None))
    else:
        raise ValueError("target_spectra must be a 1-D vector or a (bands, columns) matrix.")

    return result
