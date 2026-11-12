"""Experimental EMIT L1B retrieval using the shared HyGAS numerical methods.

MF, measurement-noise source and uncertainty method are explicit independent
settings. No default calibration, cloud mask, column interpolation or JPL-parity
claim is inferred. Atmospheric H2O comes from the exactly paired L2A MASK.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform

import h5py
import numpy as np

import advanced_matched_filter
from scripts.core import jpl_matched_filter, lut, matched_filter, noise, targets
from scripts.satellites import emit_utils

logger = logging.getLogger(__name__)
MF_MODES = ("srf-column", "full-column", "advanced", "jpl")


def _identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def _reference_for_scene(path, scene):
    ref = noise.ColumnwiseSNRReference.load(path)
    meta = ref.metadata or {}
    if meta.get("sensor") != "emit" or meta.get("radiance_unit") != emit_utils.RADIANCE_UNIT:
        raise ValueError("EMIT reference requires explicit sensor=emit and native HyGAS radiance units.")
    if meta.get("column_interpolation") is not False:
        raise ValueError("EMIT reference must explicitly declare column_interpolation=false.")
    count = scene.metadata["native_shape_brc"][2]
    if not np.array_equal(np.asarray(meta.get("native_column_indices")), np.arange(count)):
        raise ValueError("EMIT reference must document its full native detector-column axis.")
    ref = ref.subset_by_band_numbers(scene.band_indices)
    if not np.allclose(ref.band_nm, scene.wavelengths, rtol=0, atol=1.0):
        raise ValueError("EMIT reference/native band wavelengths differ by more than 1 nm.")
    ref = ref.subset_native_columns(scene.column_indices, native_column_count=count)
    # Nonpositive/signed weak signals are not a calibrated positive-noise source.
    supported = (np.isfinite(ref.mean_radiance) & (ref.mean_radiance > 0)
                 & np.isfinite(ref.snr) & (ref.snr > 0))
    ref.mean_radiance[~supported] = np.nan
    ref.snr[~supported] = np.nan
    return ref, supported


def _parametric_sigma(path, scene):
    """JPL formula and linear coefficient interpolation, with no extrapolation.

    The caller explicitly selects an EMIT calibration file in native radiance
    units. Its applicability to the scene's build/epoch remains unvalidated.
    """
    coeffs = np.loadtxt(path, comments="#", ndmin=2)
    if (coeffs.shape[0] < 2 or coeffs.shape[1] < 4 or not np.all(np.isfinite(coeffs[:, :4]))
            or np.any(np.diff(coeffs[:, 0]) <= 0)):
        raise ValueError("Noise file must contain ordered, finite wavelength/a/b/c columns.")
    wave = scene.wavelengths
    if np.min(wave) < coeffs[0, 0] or np.max(wave) > coeffs[-1, 0]:
        raise ValueError("Parametric noise wavelength extrapolation is forbidden.")
    parameters = np.column_stack([np.interp(wave, coeffs[:, 0], coeffs[:, i]) for i in (1, 2, 3)])
    rad = scene.cube.transpose(1, 2, 0)
    sigma = jpl_matched_filter.get_noise_equivalent_spectral_radiance(parameters, rad)
    return sigma.transpose(2, 0, 1), parameters


def _legacy_mf(rad, valid, target, mode, k, random_state):
    """Use shared legacy statistics/MF on observed spectra only, no imputation."""
    nr, nc, nb = rad.shape
    enhancement = np.full((nr, nc), np.nan)
    labels = np.full((nr, nc), -1, dtype=np.int32)
    if mode == "srf-column":
        pixels = rad[valid][:, None, :]
        if pixels.shape[0] < max(k, 2):
            raise ValueError("Insufficient valid spectra for clustering/statistics.")
        selected_labels = matched_filter.k_means_hyperspectral(pixels, k, random_state=random_state)
        labels[valid] = selected_labels[:, 0]
        means, covariances = matched_filter.calculate_statistics(pixels, selected_labels, k)
        counts = np.bincount(selected_labels.ravel(), minlength=k)
        if np.any(counts < 2):
            raise ValueError("A cluster has fewer than two observed spectra; reduce k.")
    for c in range(nc):
        mask = valid[:, c]
        if not np.any(mask):
            continue
        values = rad[mask, c, :][:, None, :]
        if mode == "full-column":
            if values.shape[0] < 2:
                continue
            means, covariances = matched_filter.calculate_column_statistics(values)
            local_labels = np.zeros((values.shape[0], 1), dtype=np.int32)
            labels[mask, c] = c
            n_classes = 1
        else:
            local_labels = labels[mask, c][:, None]
            n_classes = k
        enhancement[mask, c] = matched_filter.calculate_matched_filter_columnwise(
            values, local_labels, means, covariances, target[:, None], n_classes,
        )[:, 0]
    return enhancement, labels


def detection_emit(
    radiance_file: str, observation_file: str, atmosphere_file: str,
    lut_file: str, output_dir: str, *, gas="ch4", k=1,
    min_wavelength=None, max_wavelength=None, snr_reference_path=None,
    mf_mode="srf-column", noise_backend="snr-reference", noise_parameters_path=None,
    uncertainty_method="hygas", rows=None, columns=None, export_ortho=True,
    advanced_mf_options=None, jpl_options=None, random_state=0,
):
    """Run CH4/CO2 in native detector space, then optionally export via GLT.

    Default uncertainty is shared HyGAS per-pixel Eq.(37), including its existing
    numerical floors. ``uncertainty_method='jpl'`` requires MF='jpl' and uses
    covariance-weighted propagation/sensitivity in the current shared adaptation.
    Neither route is a claim of parity with operational NASA products.
    """
    gas, lo, hi, concentrations = lut.resolve_gas_settings(gas, min_wavelength, max_wavelength)
    if not np.isfinite([lo, hi]).all() or mf_mode not in MF_MODES or k < 1:
        raise ValueError("Invalid spectral window, MF mode or cluster count.")
    if uncertainty_method not in ("hygas", "jpl") or (uncertainty_method == "jpl" and mf_mode != "jpl"):
        raise ValueError("JPL uncertainty requires the JPL MF; choose hygas or jpl explicitly.")
    if noise_backend == "snr-reference":
        snr_reference_path = snr_reference_path or os.environ.get("EMIT_SNR_REFERENCE")
        if not snr_reference_path or noise_parameters_path is not None:
            raise ValueError("Provide an explicit EMIT SNR reference, without a parametric noise file.")
        noise_path = snr_reference_path
    elif noise_backend == "jpl-parametric":
        if not noise_parameters_path or snr_reference_path:
            raise ValueError("Parametric noise requires its explicit file and no SNR reference/environment fallback.")
        noise_path = noise_parameters_path
    else:
        raise ValueError("Unknown noise backend.")
    # Select bands before reading radiance; no full visible/SWIR cube is required.
    with h5py.File(radiance_file, "r") as root:
        wl = emit_utils._numeric(root["sensor_band_parameters/wavelengths"], slice(None))
    bands = np.flatnonzero((wl >= lo) & (wl <= hi))
    if bands.size < 2:
        raise ValueError("The selected spectral window must contain at least two EMIT bands.")
    scene = emit_utils.read_emit_scene(
        [radiance_file, observation_file], rows=rows, columns=columns,
        band_indices=bands, load_glt=export_ortho,
    )
    valid = scene.valid_pixels
    if not valid.any():
        raise ValueError("No radiometrically valid spectra in the selected window.")
    water, atmosphere = emit_utils.read_emit_water_vapour(atmosphere_file, scene)
    water_valid = valid & np.isfinite(water) & (water >= 0)
    if not water_valid.any():
        raise ValueError("No valid atmospheric H2O samples on the retrieval support.")
    sza_layers = [v for name, v in scene.observation.items() if name.startswith("To-sun zenith (")]
    if len(sza_layers) != 1:
        raise ValueError("OBS must provide an unambiguous To-sun zenith layer.")
    angles = sza_layers[0][valid & np.isfinite(sza_layers[0])]
    elev = scene.location["elev"][valid & np.isfinite(scene.location["elev"])]
    if not angles.size or not elev.size:
        raise ValueError("Finite OBS solar zenith and RAD elevation are required.")
    sza = float(np.median(angles))
    water_raw = float(np.median(water[water_valid]))
    ground_raw = float(np.mean(elev) / 1000)
    if not 0 <= sza <= 80:
        raise ValueError("Solar zenith outside LUT domain [0,80] degrees.")
    ground, wv = lut.normalize_ground_km(ground_raw), lut.normalize_wv_gcm2(water_raw)
    if ground != ground_raw or wv != water_raw:
        logger.warning("Scene ground/H2O clipped to shared LUT limits; raw and used values recorded.")
    if noise_backend == "snr-reference":
        reference, supported = _reference_for_scene(noise_path, scene)
        sigma = noise.compute_sigma_map_from_reference(reference, scene.cube)
        bad_bands, bad_columns = np.where(~supported)
        sigma[bad_bands, :, bad_columns] = np.nan
        calibration = dict(metadata=reference.metadata, wavelength_tolerance_nm=1.0,
                           column_interpolation=False)
        ref_attrs = reference.metadata
        for key in ("product_version", "software_build_version"):
            if ref_attrs.get(key) != scene.metadata["product_attributes"].get(key):
                logger.warning("SNR calibration/scene %s differ; transfer is experimental.", key)
    else:
        sigma, parameters = _parametric_sigma(noise_path, scene)
        supported = np.ones((bands.size, scene.cube.shape[2]), dtype=bool)
        calibration = dict(coefficients=parameters.tolist(), units=emit_utils.RADIANCE_UNIT,
                           build_epoch_applicability="not validated", wavelength_extrapolation=False,
                           nonpositive_sqrt_argument_floor=1e-5, column_dependent=False)
    sigma[:, ~valid] = np.nan
    sim_rads, sim_wave = lut.generate_library(
        concentrations, lut_file, zenith=sza, sensor=120, ground=ground, water=wv, order=1, gas=gas,
    )
    target = targets.generate_template_from_bands(
        scene.wavelengths, scene.fwhm, sim_rads, sim_wave, concentrations,
    )[:, 1]
    if not np.all(np.isfinite(target)) or not np.any(target):
        raise ValueError("LUT produced a nonfinite or identically zero target.")
    rad = scene.cube.transpose(1, 2, 0)
    sensitivity = None
    effective_options = {}
    if mf_mode in ("srf-column", "full-column"):
        enhancement, labels = _legacy_mf(rad, valid, target, mf_mode, k, random_state)
    elif mf_mode == "advanced":
        effective_options = dict(group_min=10, group_max=30, n_clusters=k, shrinkage=0.1)
        effective_options.update(advanced_mf_options or {})
        enhancement = advanced_matched_filter.run_advanced_mf(
            rad, target, scene.wavelengths, mask=~valid, **effective_options,
        )
        labels = advanced_matched_filter.get_last_cluster_labels()
    else:
        effective_options = dict(covariance_style="looshrinkage", fixed_alpha=None)
        effective_options.update(jpl_options or {})
        if set(effective_options) != {"covariance_style", "fixed_alpha"}:
            raise ValueError("Unsupported JPL option; nodata/masks/scaling cannot be overridden.")
        enhancement, jpl_uncertainty, sensitivity = jpl_matched_filter.run_jpl_mf(
            rad, target, valid, noise_cube=sigma.transpose(1, 2, 0),
            nodata_value=np.nan, **effective_options,
        )
        labels = np.where(np.isfinite(enhancement), np.arange(rad.shape[1])[None, :], -1)
    if uncertainty_method == "hygas":
        uncertainty = noise.propagate_rmn_uncertainty_per_pixel(scene.cube, sigma, target)
    else:
        uncertainty = jpl_uncertainty
    retrieval_valid = valid & np.isfinite(enhancement)
    uncertainty_valid = (retrieval_valid & np.all(np.isfinite(sigma) & (sigma > 0), axis=0)
                         & np.isfinite(uncertainty))
    enhancement = np.where(retrieval_valid, enhancement, np.nan).astype(np.float32)
    uncertainty = np.where(uncertainty_valid, uncertainty, np.nan).astype(np.float32)
    labels = np.where(retrieval_valid, labels, -1).astype(np.int32)
    if sensitivity is not None:
        sensitivity = np.where(retrieval_valid, sensitivity, np.nan).astype(np.float32)
    output = Path(output_dir)
    basename = f"{scene.scene_id}_{gas}"
    native_path = output / f"{basename}_native.npz"
    report_path = output / f"{basename}_processing.json"
    ortho_paths = {name: output / f"{basename}_{name}_ortho.tif"
                   for name in ("MF", "MF_uncertainty", "CL", "radiance_valid", "uncertainty_valid")}
    if sensitivity is not None:
        ortho_paths["MF_sensitivity"] = output / f"{basename}_MF_sensitivity_ortho.tif"
    destinations = [native_path, report_path] + (list(ortho_paths.values()) if export_ortho else [])
    if any(p.exists() for p in destinations):
        raise FileExistsError("EMIT outputs already exist; select a new output directory, no overwrite performed.")
    inputs = {name: _identity(path) for name, path in dict(
        radiance=radiance_file, observation=observation_file, atmosphere=atmosphere_file,
        lut=lut_file, noise_source=noise_path,
    ).items()}
    output.mkdir(parents=True, exist_ok=True)
    maps = dict(MF=enhancement, MF_uncertainty=uncertainty, CL=labels,
                radiance_valid=valid, uncertainty_valid=uncertainty_valid)
    if sensitivity is not None:
        maps["MF_sensitivity"] = sensitivity
    with native_path.open("xb") as stream:
        np.savez_compressed(stream, **maps, band_indices=scene.band_indices,
                            row_indices=scene.row_indices, column_indices=scene.column_indices,
                            wavelengths=scene.wavelengths, fwhm=scene.fwhm, unit_absorption=target,
                            reference_supported=supported, latitude=scene.location["lat"],
                            longitude=scene.location["lon"])
    if export_ortho:
        for name, path in ortho_paths.items():
            units = "ppm m" if name in ("MF", "MF_uncertainty") else "1"
            emit_utils.save_ortho_map(path, maps[name].astype(np.float32), scene, units=units)
    repo = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), repo/"scripts/satellites/emit_utils.py",
                    repo/"scripts/core/noise.py", repo/"scripts/core/matched_filter.py",
                    repo/"scripts/core/jpl_matched_filter.py", repo/"scripts/core/targets.py",
                    repo/"scripts/core/lut.py", repo/"advanced_matched_filter.py"]
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(), status="experimental_not_scientifically_validated",
        sensor="emit", gas=gas, enhancement_units="ppm m", inputs=inputs,
        scene=scene.metadata, atmosphere=atmosphere, mf_mode=mf_mode, k=k,
        random_state=random_state, effective_mf_options=effective_options,
        noise_backend=noise_backend, noise_calibration=calibration,
        uncertainty_method=uncertainty_method,
        uncertainty_definition=("HyGAS per-pixel Eq.(37); not covariance-weighted MF propagation"
                                if uncertainty_method == "hygas" else "Current shared JPL MF covariance-weighted adaptation"),
        noise_numerical_guards=(dict(MIN_SIGMA=noise.MIN_SIGMA, EPS=noise.EPS, snr_scale_floor=0.1)
                                if noise_backend == "snr-reference" else {}),
        lut_parameters=dict(solar_zenith_deg=sza, ground_km_raw=ground_raw, ground_km_used=ground,
                            water_gcm2_raw=water_raw, water_gcm2_used=wv,
                            water_samples=int(water_valid.sum()), sensor_height_km=120, interpolation_order=1,
                            concentrations=concentrations.tolist(), window_nm=[lo, hi]),
        support=dict(total_pixels=int(valid.size), radiance_valid=int(valid.sum()),
                     retrieval_valid=int(retrieval_valid.sum()), uncertainty_valid=int(uncertainty_valid.sum()),
                     reference_supported_cells=int(supported.sum()), reference_cells=int(supported.size)),
        QA_exclusions_applied=[], column_interpolation=False, target_scaling=1,
        target_response="Gaussian from nominal 1D CW/FWHM; no across-track smile inferred",
        orthorectification="original GLT nearest lookup" if export_ortho else "not requested",
        sources=[_identity(p) for p in source_paths],
        environment=dict(python=platform.python_version(), packages={
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py", "scikit-learn")}),
        outputs={p.name: _identity(p) for p in destinations if p != report_path},
        limitations=["Experimental SNR calibration/noise transfer; no automatic QA selection",
                     "JPL mode is the HyGAS adaptation, not established operational parity",
                     "CO2 is not validated by the CH4 paper", "sigma_surf/sigma_tot and flux are downstream steps"],
    )
    with report_path.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    logger.info("EMIT %s: %d retrieval pixels; %d uncertainty pixels. Report: %s",
                gas.upper(), retrieval_valid.sum(), uncertainty_valid.sum(), report_path)
    return report


def process_directory_emit(root_dir, output_root_dir, *, lut_file, **kwargs):
    """Batch exact RAD/OBS/MASK triplets; reject missing/ambiguous companions."""
    root, output = Path(root_dir), Path(output_root_dir).resolve()
    radiances = sorted(p for p in root.rglob("EMIT_L1B_RAD_*.nc") if output not in p.resolve().parents)
    if not radiances:
        raise ValueError("No canonical EMIT L1B RAD products found.")
    scene_ids = [p.stem for p in radiances]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Duplicate EMIT scenes in batch tree; choose a canonical input subtree.")
    triplets = []
    for rad in radiances:
        obs = list(root.rglob(rad.name.replace("_RAD_", "_OBS_")))
        masks = list(root.rglob(rad.name.replace("_L1B_RAD_", "_L2A_MASK_")))
        if len(obs) != 1 or len(masks) != 1:
            raise ValueError(f"Require exactly one OBS and MASK companion for {rad.name}.")
        triplets.append((rad, obs[0], masks[0]))
    return [detection_emit(str(rad), str(obs), str(mask), lut_file,
                           str(output/rad.stem), **kwargs) for rad, obs, mask in triplets]
