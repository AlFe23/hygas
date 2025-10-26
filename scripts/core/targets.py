"""
Target spectrum utilities shared across satellites. This includes the methane
unit absorption template creation as well as helpers for selecting the desired
spectral window and generating column-wise targets when accounting for spectral
smile.
"""

import numpy as np


def select_band_indices(mean_cw, min_wavelength, max_wavelength):
    """
    Return indices of bands whose central wavelength lies within the desired spectral window.
    """
    return np.where((mean_cw >= min_wavelength) & (mean_cw <= max_wavelength))[0]


def generate_template_from_bands(centers, fwhm, simRads, simWave, concentrations):
    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers.
    :param fwhm: full width half maximum for the gaussian kernel of each band.
    :return template: the unit absorption spectum
    """
    # import scipy.stats
    # SCALING = 1e5
    SCALING = 1
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError("Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).")
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError("Length of band center wavelengths and band fwhm arrays must be equal.")
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    var = sigma**2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-(simWave[:, None] - centers[None, :]) ** 2 / (2 * var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(axis=0), where=response.sum(axis=0) > 0, out=response)
    # implement resampling as matrix multiply
    resampled = simRads.dot(response)
    lograd = np.log(resampled, out=np.zeros_like(resampled), where=resampled > 0)
    slope, _, _, _ = np.linalg.lstsq(
        np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None
    )
    spectrum = slope[1, :] * SCALING
    target = np.stack((centers, spectrum)).T  # np.stack((np.arange(spectrum.shape[0]), centers, spectrum)).T
    return target


def generate_columnwise_targets(cw_subselection, fwhm_subselection, simRads, simWave, concentrations):
    """
    Generate target spectra tailored for each spatial column (PRISMA column-wise workflow).
    """
    target = None
    for i in range(np.size(cw_subselection, 0)):
        target_i = generate_template_from_bands(
            cw_subselection[i, :], fwhm_subselection[i, :], simRads, simWave, concentrations
        )
        if i == 0:
            target = target_i
        else:
            column_to_add = target_i[:, 1].reshape(-1, 1)
            target = np.concatenate((target, column_to_add), axis=1)
    return target[:, 1:]

