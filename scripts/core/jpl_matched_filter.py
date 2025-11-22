"""
This module contains an adapted version of the JPL Matched Filter implementation
from the emit-ghg-main repository. It is designed to be used with PRISMA and
EnMAP data within the HyGAS project structure.

Original Authors: Philip G. Brodrick, Brian D. Bue, David R. Thompson,
                  Jay Fahlen, Red Willow Coleman
"""

import logging
import numpy as np
import scipy.linalg
import scipy.interpolate

logger = logging.getLogger(__name__)


def cov(A, **kwargs):
    """Calculate covariance of a matrix."""
    kwargs.setdefault('ddof', 1)
    return np.cov(A.T, **kwargs)


def fit_looshrinkage_alpha(data, alphas, I_reg=[]):
    """Leave-one-out cross-validation shrinkage estimation via Theiler et al."""
    stability_scaling = 100.0
    nchan = data.shape[1]
    nll = np.zeros(len(alphas))
    n = data.shape[0]

    X = data * stability_scaling
    S = cov(X)
    T = np.diag(np.diag(S)) if len(I_reg) == 0 else cov(I_reg * stability_scaling)

    nchanlog2pi = nchan * np.log(2.0 * np.pi)
    nll[:] = np.inf

    for i, alpha in enumerate(alphas):
        try:
            beta = (1.0 - alpha) / (n - 1.0)
            G_alpha = n * (beta * S) + (alpha * T)
            G_det = scipy.linalg.det(G_alpha, check_finite=False)
            if G_det == 0:
                continue
            r_k = (X.dot(scipy.linalg.inv(G_alpha, check_finite=False)) * X).sum(axis=1)
            q = 1.0 - beta * r_k
            nll[i] = 0.5 * (nchanlog2pi + np.log(G_det)) + 1.0 / (2.0 * n) * \
                     (np.log(q) + (r_k / q)).sum()
        except np.linalg.LinAlgError:
            logging.warning('looshrinkage encountered a LinAlgError')

    mindex = np.argmin(nll)
    alpha = alphas[mindex] if nll[mindex] != np.inf else 0.0
    return alpha


def apply_looshrinkage_alpha(data: np.array, alpha: float, I_reg=[]):
    """Apply shrinkage to the covariance matrix."""
    S = cov(data)
    T = np.diag(np.diag(S)) if len(I_reg) == 0 else cov(I_reg)
    C = (1.0 - alpha) * S + alpha * T
    return C


def calculate_mf_covariance(radiance: np.array, model: str, fixed_alpha: float = None):
    """Calculate covariance and mean of radiance data."""
    if model == 'looshrinkage':
        alpha = fixed_alpha if fixed_alpha is not None else fit_looshrinkage_alpha(radiance, (10.0 ** np.arange(-10, 0 + 0.05, 0.05)))
        C = apply_looshrinkage_alpha(radiance, alpha)
    elif model == 'empirical':
        C = cov(radiance)
    else:
        raise ValueError(f"Covariance model '{model}' not recognized")
    return C


def get_noise_equivalent_spectral_radiance(noise_model_parameters: np.array, radiance: np.array):
    """Calculate noise-equivalent spectral radiance from a parametric model."""
    noise_plus_meas = noise_model_parameters[:, 1] + radiance
    if np.any(noise_plus_meas <= 0):
        noise_plus_meas[noise_plus_meas <= 0] = 1e-5
        logger.warning("Parametric noise model found noise <= 0 - adjusting to be positive.")
    nedl = np.abs(noise_model_parameters[:, 0] * np.sqrt(noise_plus_meas) + noise_model_parameters[:, 2])
    return nedl


def run_jpl_mf(rads_array, target_spectra, good_pixel_mask, noise_cube=None,
               covariance_style='looshrinkage', fixed_alpha=None,
               nodata_value=-9999.0):
    """
    Apply the JPL/EMIT matched filter to a radiance cube.

    Parameters:
        rads_array (np.array): Radiance cube (rows, cols, bands).
        target_spectra (np.array): Target absorption coefficients (bands,).
        good_pixel_mask (np.array): Boolean mask of good pixels (rows, cols).
        noise_cube (np.array, optional): Noise cube (sigma) for uncertainty (rows, cols, bands).
        covariance_style (str): 'looshrinkage' or 'empirical'.
        fixed_alpha (float, optional): Fixed shrinkage parameter.
        nodata_value (float): No data value for output.

    Returns:
        tuple: (matched_filter_map, uncertainty_map, sensitivity_map)
    """
    rows, cols, bands = rads_array.shape
    rad_for_mf = np.float64(rads_array).transpose([1, 0, 2])  # to (cols, rows, bands)

    mf = np.full((cols, rows), nodata_value, dtype=np.float32)
    uncert = np.full((cols, rows), nodata_value, dtype=np.float32)
    sens = np.full((cols, rows), nodata_value, dtype=np.float32)

    no_radiance_mask_full = np.all(np.isfinite(rad_for_mf), axis=2)

    for col in range(cols):
        rad_col = rad_for_mf[col, :, :]
        col_mask = np.logical_and(good_pixel_mask[:, col], no_radiance_mask_full[col, :])
        good_pixel_idx = np.where(col_mask)[0]

        if len(good_pixel_idx) < 10:
            logger.debug(f'Too few good pixels in column {col}: skipping')
            continue

        try:
            C = calculate_mf_covariance(rad_col[good_pixel_idx, :], covariance_style, fixed_alpha)
            Cinv = scipy.linalg.inv(C, check_finite=False)
        except np.linalg.LinAlgError:
            logger.warning(f'Singular matrix in column {col}. Skipping.')
            continue

        mu = np.mean(rad_col[good_pixel_idx, :], axis=0)
        target = target_spectra * mu
        normalizer = target.dot(Cinv).dot(target.T)

        # Matched Filter
        mf_col = target.T.dot(Cinv).dot((rad_col[no_radiance_mask_full[col, :], :] - mu).T) / normalizer
        mf[col, no_radiance_mask_full[col, :]] = mf_col

        # Uncertainty
        if noise_cube is not None:
            noise_col = noise_cube[:, col, :]
            nedl_variance = noise_col[no_radiance_mask_full[col, :], :] ** 2

            sC = target.dot(Cinv)
            numer = (sC * nedl_variance) @ sC
            a_times_X = -1 * target_spectra * rad_col[no_radiance_mask_full[col, :], :]
            denom = ((target).dot(Cinv).dot(a_times_X.T)) ** 2

            # Avoid division by zero or sqrt of negative
            valid_denom = denom > 0
            uncert_col = np.full(denom.shape, np.nan)
            uncert_col[valid_denom] = np.sqrt(numer[valid_denom] / denom[valid_denom])

            sens_col = np.sqrt(denom) / normalizer

            sens[col, no_radiance_mask_full[col, :]] = sens_col
            uncert[col, no_radiance_mask_full[col, :]] = uncert_col

    mf_map = mf.T.astype(np.float32)
    uncert_map = uncert.T.astype(np.float32)
    sens_map = sens.T.astype(np.float32)

    # Set nodata values
    final_mask = ~good_pixel_mask | ~np.all(np.isfinite(rads_array), axis=2)
    mf_map[final_mask] = nodata_value
    uncert_map[final_mask] = nodata_value
    sens_map[final_mask] = nodata_value
    uncert_map[~np.isfinite(uncert_map)] = nodata_value
    sens_map[~np.isfinite(sens_map)] = nodata_value

    return mf_map, uncert_map, sens_map
