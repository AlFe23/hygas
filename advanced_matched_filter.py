"""
Advanced matched-filter implementation with per-column grouping, PCA/k-means
background modeling, and shrinkage-regularised covariance matrices. This module
is self-contained so it can be plugged into the existing CH₄ detection
workflows without touching the legacy matched filter utilities.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:  # Optional dependency for CLI raster I/O.
    import rasterio
except ImportError:  # pragma: no cover - rasterio might be unavailable.
    rasterio = None  # type: ignore

logger = logging.getLogger(__name__)

_LAST_CLUSTER_LABELS: np.ndarray | None = None
_LAST_CLUSTER_STATS: List[ClusterStats] | None = None


@dataclass
class ClusterStats:
    """Container for per-cluster statistics computed in the spectral domain."""

    mean: np.ndarray
    covariance: np.ndarray
    sigma2: float
    sample_count: int


def run_advanced_mf(
    radiance_cube: np.ndarray,
    targets: np.ndarray,
    wavelengths: np.ndarray,
    mask: np.ndarray | None = None,
    group_min: int = 10,
    group_max: int = 30,
    n_clusters: int = 3,
    shrinkage: float = 0.1,
    per_cluster_targets: bool = False,
    adaptive_shrinkage: bool = False,
    min_clusters: int = 3,
    target_blend: float | None = None,
    adaptive_shrinkage_min: float = 0.0,
) -> np.ndarray:
    """
    Execute the enhanced matched filter as described in the requirements.

    Parameters
    ----------
    radiance_cube : np.ndarray
        Radiance hypercube shaped as (rows, columns, bands).
    targets : np.ndarray
        Target spectrum(s). Accepts a 1-D vector of length bands or a
        (bands, columns) matrix for column-wise targets.
    wavelengths : np.ndarray
        Central wavelengths for each radiance band. Used for bookkeeping and
        validation to guarantee spectral alignment.
    mask : np.ndarray | None, optional
        Boolean mask (rows, columns) where True suppresses a pixel from
        background/target estimation. Masked pixels yield NaN in the output.
    group_min : int, optional
        Minimum width (in columns) for each grouping tile.
    group_max : int, optional
        Maximum width (in columns) for each grouping tile.
    n_clusters : int, optional
        Requested number of clusters per group prior to merging.
    shrinkage : float, optional
        Shrinkage coefficient applied to each covariance matrix.
    per_cluster_targets : bool, optional
        If True, scale the target spectrum by each cluster mean instead of the
        group-wide mean radiance (mirrors the legacy matched-filter formulation).
    target_blend : float | None, optional
        If provided and per_cluster_targets is True, interpolate between group
        and cluster means (0 = group only, 1 = cluster only). Defaults to 1.
    adaptive_shrinkage : bool, optional
        Enable condition-number driven shrinkage that only regularises ill
        conditioned covariance matrices.
    min_clusters : int, optional
        Minimum number of PCA/k-means clusters per column group.
    adaptive_shrinkage_min : float, optional
        Lower bound on the shrinkage factor when adaptive shrinkage is enabled.

    Returns
    -------
    np.ndarray
        Two-dimensional matched-filter enhancement map with the same number of
        rows/columns as the radiance input.
    """

    if radiance_cube.ndim != 3:
        raise ValueError("radiance_cube must be shaped as (rows, columns, bands).")
    rows, cols, bands = radiance_cube.shape
    if bands == 0:
        raise ValueError("radiance_cube must include at least one spectral band.")
    if min_clusters < 1:
        raise ValueError("min_clusters must be a positive integer.")

    _targets = _prepare_targets(targets, bands, cols)
    _validate_wavelengths(wavelengths, bands)

    mask_array = _prepare_mask(mask, rows, cols)
    cube = np.asarray(radiance_cube, dtype=np.float64, order="C")

    striping_profile = _compute_strip_profile(cube, mask_array)
    groups = _segment_columns(striping_profile, group_min, group_max)

    logger.info(
        "Advanced MF: %d columns segmented into %d groups (min=%d, max=%d).",
        cols,
        len(groups),
        group_min,
        group_max,
    )

    result = np.full((rows, cols), np.nan, dtype=cube.dtype)
    cluster_labels = np.full((rows, cols), -1, dtype=np.int32)
    collected_models: List[ClusterStats] = []
    global_cluster_id = 0
    effective_clusters = max(n_clusters, min_clusters)
    if effective_clusters != n_clusters:
        logger.info(
            "Advanced MF: requested %d clusters but enforcing minimum %d -> %d clusters.",
            n_clusters,
            min_clusters,
            effective_clusters,
        )

    for group_index, (col_start, col_end) in enumerate(groups):
        coords, spectra = _extract_group_pixels(cube, mask_array, col_start, col_end)
        if coords.size == 0:
            logger.info("Group %d (%d:%d) contains no valid pixels.", group_index, col_start, col_end)
            continue

        standardized, group_mean, _ = _standardize_spectra(spectra)
        pca_scores = _project_pca(standardized)
        raw_labels, _ = _cluster_pca(pca_scores, effective_clusters, group_index)
        merged_labels, components = _merge_small_clusters(
            raw_labels,
            pca_scores,
            min_samples=1000,
            group_index=group_index,
        )

        cluster_models = _compute_cluster_statistics(
            spectra,
            components,
            shrinkage,
            adaptive_shrinkage=adaptive_shrinkage,
            adaptive_floor=max(0.0, adaptive_shrinkage_min),
            group_index=group_index,
        )
        collected_models.extend(cluster_models)
        unit_target = _group_unit_target(base_targets=_targets, col_start=col_start, col_end=col_end)

        _apply_matched_filter(
            result=result,
            coords=coords,
            spectra=spectra,
            merged_labels=merged_labels,
            cluster_models=cluster_models,
            unit_target=unit_target,
            group_mean=group_mean,
            per_cluster_targets=per_cluster_targets,
            target_blend=target_blend,
            cluster_labels=cluster_labels,
            global_cluster_offset=global_cluster_id,
            group_index=group_index,
        )
        global_cluster_id += len(cluster_models)

    if mask_array is not None:
        result[mask_array] = np.nan

    global _LAST_CLUSTER_LABELS, _LAST_CLUSTER_STATS
    _LAST_CLUSTER_LABELS = cluster_labels
    _LAST_CLUSTER_STATS = collected_models
    return result.astype(radiance_cube.dtype, copy=False)


def plot_column_groups(strip_profile: np.ndarray, groups: Sequence[Tuple[int, int]]) -> None:
    """
    Convenience helper to visualise the column striping profile and derived groups.
    """

    import matplotlib.pyplot as plt  # Imported lazily to avoid unnecessary dependency.

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(strip_profile, color="tab:blue", lw=1.5, label="striping profile")
    for idx, (start, end) in enumerate(groups):
        ax.axvspan(
            start - 0.5,
            end - 0.5,
            alpha=0.15,
            color="tab:orange" if idx % 2 == 0 else "tab:green",
            label="group" if idx == 0 else None,
        )
    ax.set_xlabel("Column index")
    ax.set_ylabel("Median radiance")
    ax.set_title("Column groups derived from striping profile")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def get_last_cluster_labels() -> np.ndarray | None:
    """Return a copy of the most recent cluster label map produced by run_advanced_mf."""

    if _LAST_CLUSTER_LABELS is None:
        return None
    return _LAST_CLUSTER_LABELS.copy()


def get_last_cluster_statistics() -> Tuple[np.ndarray, np.ndarray] | None:
    """Return stacked mean radiance and covariance matrices from the last run."""

    if not _LAST_CLUSTER_STATS:
        return None
    means = np.stack([stat.mean for stat in _LAST_CLUSTER_STATS], axis=0)
    covariances = np.stack([stat.covariance for stat in _LAST_CLUSTER_STATS], axis=0)
    return means, covariances


def _prepare_targets(targets: np.ndarray, bands: int, cols: int) -> np.ndarray:
    array = np.asarray(targets, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != bands:
            raise ValueError(f"1-D target spectrum must have length {bands}, received {array.shape[0]}.")
        array = array[:, None]
    elif array.ndim == 2:
        if array.shape[0] != bands:
            raise ValueError(
                f"Leading dimension of target matrix must match number of bands ({bands}). "
                f"Received {array.shape[0]}."
            )
        if array.shape[1] not in (1, cols):
            raise ValueError(
                f"Target matrix must have either 1 column or {cols} columns. Received {array.shape[1]}."
            )
    else:
        raise ValueError("targets must be a 1-D vector or a 2-D (bands, columns) matrix.")
    return array


def _validate_wavelengths(wavelengths: np.ndarray, bands: int) -> None:
    arr = np.asarray(wavelengths)
    if arr.ndim != 1 or arr.shape[0] != bands:
        raise ValueError(f"wavelengths must be a 1-D vector of length {bands}.")


def _prepare_mask(mask: np.ndarray | None, rows: int, cols: int) -> np.ndarray | None:
    if mask is None:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != (rows, cols):
        raise ValueError(f"mask must be shaped as ({rows}, {cols}). Received {mask_arr.shape}.")
    return mask_arr


def _compute_strip_profile(cube: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        masked_cube = np.where(mask[:, :, None], np.nan, cube)
    else:
        masked_cube = cube
    profile = np.nanmedian(masked_cube, axis=(0, 2))
    if np.all(~np.isfinite(profile)):
        raise RuntimeError("Striping profile could not be computed because all columns are masked.")
    fill_value = np.nanmedian(profile[np.isfinite(profile)])
    profile = np.where(np.isfinite(profile), profile, fill_value)
    return profile


def _segment_columns(strip_profile: np.ndarray, group_min: int, group_max: int) -> List[Tuple[int, int]]:
    if group_min <= 0 or group_max <= 0:
        raise ValueError("group_min and group_max must be positive integers.")
    if group_min > group_max:
        raise ValueError("group_min cannot exceed group_max.")

    n_cols = strip_profile.shape[0]
    if n_cols == 0:
        return []

    median_profile = float(np.median(strip_profile))
    mad = float(np.median(np.abs(strip_profile - median_profile)))
    threshold = 1.5 * mad if mad > 0 else math.inf
    groups: List[Tuple[int, int]] = []
    start = 0
    while start < n_cols:
        end = min(start + group_min - 1, n_cols - 1)
        if end < start:
            end = start
        while end + 1 < n_cols:
            width = end - start + 1
            if width >= group_max:
                break
            diff = abs(strip_profile[end + 1] - strip_profile[end])
            if diff > threshold and width >= group_min:
                break
            end += 1
        groups.append((start, end + 1))
        start = end + 1

    if not groups:
        groups.append((0, n_cols))
    else:
        last_start, last_end = groups[-1]
        if last_end - last_start < group_min and len(groups) > 1:
            prev_start, _ = groups[-2]
            groups[-2] = (prev_start, last_end)
            groups.pop()
        groups[-1] = (groups[-1][0], n_cols)
    return groups


def _extract_group_pixels(
    cube: np.ndarray,
    mask: np.ndarray | None,
    col_start: int,
    col_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, _, bands = cube.shape
    group_cube = cube[:, col_start:col_end, :]
    width = col_end - col_start
    pixels = group_cube.reshape(rows * width, bands)
    row_idx = np.repeat(np.arange(rows), width)
    col_idx = np.tile(np.arange(col_start, col_end), rows)
    coords = np.column_stack((row_idx, col_idx))

    valid = np.all(np.isfinite(pixels), axis=1)
    if mask is not None:
        group_mask = mask[:, col_start:col_end].reshape(rows * width)
        valid &= ~group_mask
    coords = coords[valid]
    spectra = pixels[valid]
    return coords, spectra


def _standardize_spectra(spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_spec = spectra.mean(axis=0)
    std_spec = spectra.std(axis=0, ddof=0)
    std_spec = np.where(std_spec > 0, std_spec, 1.0)
    standardized = (spectra - mean_spec) / std_spec
    return standardized, mean_spec, std_spec


def _project_pca(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 0:
        return np.empty((0, 3))
    n_components = min(3, data.shape[0], data.shape[1])
    if n_components == 0:
        return np.zeros((data.shape[0], 3))
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    reduced = pca.fit_transform(data)
    if n_components < 3:
        reduced = np.pad(reduced, ((0, 0), (0, 3 - n_components)), constant_values=0.0)
    return reduced


def _cluster_pca(pca_scores: np.ndarray, n_clusters: int, group_index: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = pca_scores.shape[0]
    if n_samples == 0:
        return np.empty((0,), dtype=int), np.empty((0, 3))
    effective_k = min(n_clusters, n_samples)
    if effective_k < n_clusters:
        logger.info(
            "Group %d: reducing clusters from %d to %d due to limited samples (%d).",
            group_index,
            n_clusters,
            effective_k,
            n_samples,
        )
    if effective_k == 1:
        return np.zeros((n_samples,), dtype=int), np.mean(pca_scores, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pca_scores)
    return labels, kmeans.cluster_centers_


def _merge_small_clusters(
    labels: np.ndarray,
    pca_scores: np.ndarray,
    min_samples: int,
    group_index: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if labels.size == 0:
        return labels, []

    clusters = {cid: np.where(labels == cid)[0] for cid in sorted(np.unique(labels))}

    def _centroid(idxs: np.ndarray) -> np.ndarray:
        return pca_scores[idxs].mean(axis=0)

    changed = True
    while changed:
        changed = False
        small = [cid for cid, idxs in clusters.items() if idxs.size < min_samples]
        if not small or len(clusters) == 1:
            break
        for cid in small:
            if cid not in clusters:
                continue
            idxs = clusters[cid]
            centroid_a = _centroid(idxs)
            best_cid = None
            best_dist = math.inf
            for other_cid, other_idxs in clusters.items():
                if other_cid == cid:
                    continue
                centroid_b = _centroid(other_idxs)
                dist = float(np.linalg.norm(centroid_a - centroid_b))
                if dist < best_dist or (math.isclose(dist, best_dist) and (best_cid is None or other_cid < best_cid)):
                    best_dist = dist
                    best_cid = other_cid
            if best_cid is None:
                continue
            merged = np.concatenate((clusters[best_cid], idxs))
            logger.info(
                "Group %d: cluster %d (n=%d) merged into cluster %d (n=%d) -> merged n=%d.",
                group_index,
                cid,
                idxs.size,
                best_cid,
                clusters[best_cid].size,
                merged.size,
            )
            clusters[best_cid] = merged
            del clusters[cid]
            changed = True
            break

    merged_labels = np.full_like(labels, fill_value=-1)
    components: List[np.ndarray] = []
    for new_id, (cid, idxs) in enumerate(sorted(clusters.items(), key=lambda item: item[0])):
        merged_labels[idxs] = new_id
        components.append(idxs)
    return merged_labels, components


def _compute_cluster_statistics(
    spectra: np.ndarray,
    components: List[np.ndarray],
    shrinkage: float,
    adaptive_shrinkage: bool,
    adaptive_floor: float,
    group_index: int,
) -> List[ClusterStats]:
    models: List[ClusterStats] = []
    if not components:
        return models
    bands = spectra.shape[1]
    for component_idx, sample_idx in enumerate(components):
        cluster_pixels = spectra[sample_idx]
        if cluster_pixels.size == 0:
            continue
        mean_spec = cluster_pixels.mean(axis=0)
        if cluster_pixels.shape[0] > 1:
            covariance = np.cov(cluster_pixels, rowvar=False)
        else:
            covariance = np.eye(bands) * 1e-6
        sigma2 = float(np.mean(np.diag(covariance)))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.finfo(cluster_pixels.dtype).eps)
        covariance = _shrink_covariance(
            covariance,
            shrinkage=shrinkage,
            sigma2=sigma2,
            adaptive_shrinkage=adaptive_shrinkage,
            adaptive_floor=adaptive_floor,
            group_index=group_index,
            cluster_index=component_idx,
        )
        models.append(
            ClusterStats(
                mean=mean_spec,
                covariance=covariance,
                sigma2=sigma2,
                sample_count=cluster_pixels.shape[0],
            )
        )
    return models


def _shrink_covariance(
    covariance: np.ndarray,
    shrinkage: float,
    sigma2: float,
    adaptive_shrinkage: bool,
    adaptive_floor: float,
    group_index: int,
    cluster_index: int,
) -> np.ndarray:
    if shrinkage <= 0:
        return covariance
    shrink_value = shrinkage
    if adaptive_shrinkage:
        eigvals = np.linalg.eigvalsh(covariance)
        eps = float(np.finfo(covariance.dtype).tiny)
        eigvals = np.clip(eigvals, eps, None)
        cond = float(np.max(eigvals) / np.min(eigvals))
        if cond < 1e4:
            logger.info(
                "Group %d cluster %d: skipping shrinkage (condition %.2e).",
                group_index,
                cluster_index,
                cond,
            )
            return covariance
        shrink_value = min(shrinkage, 1.0 - float(np.min(eigvals) / np.max(eigvals)))
        shrink_value = max(shrink_value, adaptive_floor)
        if shrink_value <= 0:
            logger.info(
                "Group %d cluster %d: adaptive shrinkage reduced weight to %.3f; skipping.",
                group_index,
                cluster_index,
                shrink_value,
            )
            return covariance
        logger.info(
            "Group %d cluster %d: adaptive shrinkage=%.3f (condition %.2e).",
            group_index,
            cluster_index,
            shrink_value,
            cond,
        )
    else:
        logger.info(
            "Group %d cluster %d: applied shrinkage=%.3f (sigma^2=%.3e).",
            group_index,
            cluster_index,
            shrink_value,
            sigma2,
        )
    identity = np.eye(covariance.shape[0], dtype=covariance.dtype)
    return (1.0 - shrink_value) * covariance + shrink_value * sigma2 * identity


def _group_unit_target(
    base_targets: np.ndarray,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    if base_targets.shape[1] == 1:
        base = base_targets[:, 0]
    else:
        base = np.nanmean(base_targets[:, col_start:col_end], axis=1)
    return np.nan_to_num(base, nan=0.0)


def _apply_matched_filter(
    result: np.ndarray,
    coords: np.ndarray,
    spectra: np.ndarray,
    merged_labels: np.ndarray,
    cluster_models: List[ClusterStats],
    unit_target: np.ndarray,
    group_mean: np.ndarray,
    per_cluster_targets: bool,
    target_blend: float | None,
    cluster_labels: np.ndarray,
    global_cluster_offset: int,
    group_index: int,
) -> None:
    if not cluster_models:
        return
    eps = np.finfo(result.dtype).eps
    for local_cluster_idx, model in enumerate(cluster_models):
        indices = np.where(merged_labels == local_cluster_idx)[0]
        if indices.size == 0:
            continue
        blend = 0.0
        if per_cluster_targets:
            blend = 1.0 if target_blend is None else float(np.clip(target_blend, 0.0, 1.0))
        effective_mean = (1.0 - blend) * group_mean + blend * model.mean
        target_vector = effective_mean * unit_target
        try:
            weight = np.linalg.solve(model.covariance, target_vector)
        except np.linalg.LinAlgError:
            jitter = model.covariance + eps * np.eye(model.covariance.shape[0])
            weight = np.linalg.solve(jitter, target_vector)
            logger.warning(
                "Group %d cluster %d: covariance inversion required jitter %.3e.",
                group_index,
                local_cluster_idx,
                eps,
            )
        denom = float(target_vector.dot(weight))
        if abs(denom) < eps:
            logger.warning(
                "Group %d cluster %d: denominator near zero (%.3e). Assigning NaN.",
                group_index,
                local_cluster_idx,
                denom,
            )
            values = np.full(indices.size, np.nan, dtype=result.dtype)
        else:
            diff = spectra[indices] - model.mean
            numerator = diff @ weight
            values = numerator / denom
        cluster_id = global_cluster_offset + local_cluster_idx
        for (row, col), value in zip(coords[indices], values):
            result[row, col] = value
            cluster_labels[row, col] = cluster_id


def _load_array(path: str) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):  # type: ignore[attr-defined]
        if "data" in data:
            return data["data"]
        first_key = sorted(data.files)[0]
        return data[first_key]
    return data


def _load_radiance(path: str) -> Tuple[np.ndarray, dict | None]:
    if rasterio is not None and path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:  # type: ignore[call-arg]
            array = src.read().transpose(1, 2, 0)
            profile = src.profile
        return array, profile
    return _load_array(path), None


def _write_output(path: str, data: np.ndarray, profile: dict | None) -> None:
    if rasterio is not None and profile is not None and path.lower().endswith((".tif", ".tiff")):
        meta = profile.copy()
        meta.update(count=1, dtype=str(data.dtype), nodata=np.nan)
        with rasterio.open(path, "w", **meta) as dst:  # type: ignore[call-arg]
            dst.write(data, 1)
        return
    np.save(path, data)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the advanced matched filter on a radiance cube.")
    parser.add_argument("--in", dest="input_path", required=True, help="Radiance cube (.tif or .npy).")
    parser.add_argument("--out", dest="output_path", required=True, help="Output MF raster (.tif or .npy).")
    parser.add_argument("--targets", dest="targets_path", help="NPY/NPZ file containing target spectra.")
    parser.add_argument("--wavelengths", dest="wavelengths_path", help="NPY/NPZ file with band wavelengths.")
    parser.add_argument("--mask", dest="mask_path", help="Optional NPY mask aligned with the radiance cube.")
    parser.add_argument("--group-min", dest="group_min", type=int, default=10, help="Minimum group width.")
    parser.add_argument("--group-max", dest="group_max", type=int, default=30, help="Maximum group width.")
    parser.add_argument("--clusters", dest="clusters", type=int, default=3, help="Initial cluster count.")
    parser.add_argument("--shrinkage", dest="shrinkage", type=float, default=0.1, help="Shrinkage coefficient.")
    parser.add_argument(
        "--per-cluster-targets",
        dest="per_cluster_targets",
        action="store_true",
        help="Scale targets by each cluster mean instead of the group-wide mean.",
    )
    parser.add_argument(
        "--target-blend",
        dest="target_blend",
        type=float,
        help="Blend factor between group and cluster means (requires --per-cluster-targets).",
    )
    parser.add_argument(
        "--adaptive-shrinkage",
        dest="adaptive_shrinkage",
        action="store_true",
        help="Only apply shrinkage when the covariance matrix is ill-conditioned.",
    )
    parser.add_argument(
        "--adaptive-shrinkage-min",
        dest="adaptive_shrinkage_min",
        type=float,
        default=0.0,
        help="Minimum shrinkage factor when adaptive shrinkage is enabled.",
    )
    parser.add_argument(
        "--min-clusters",
        dest="min_clusters",
        type=int,
        default=3,
        help="Minimum number of PCA/k-means clusters per group.",
    )
    parser.add_argument(
        "--log-level", dest="log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s - %(message)s")

    radiance_cube, profile = _load_radiance(args.input_path)
    if args.targets_path:
        targets = _load_array(args.targets_path)
    else:
        targets = np.ones(radiance_cube.shape[-1], dtype=radiance_cube.dtype)
        logger.warning("No target file supplied. Using a flat target spectrum.")

    if args.wavelengths_path:
        wavelengths = _load_array(args.wavelengths_path)
    else:
        wavelengths = np.arange(radiance_cube.shape[-1], dtype=radiance_cube.dtype)
        logger.warning("No wavelength file supplied. Using band indices as wavelengths.")

    mask = _load_array(args.mask_path) if args.mask_path else None

    mf_map = run_advanced_mf(
        radiance_cube=radiance_cube,
        targets=targets,
        wavelengths=wavelengths,
        mask=mask,
        group_min=args.group_min,
        group_max=args.group_max,
        n_clusters=args.clusters,
        shrinkage=args.shrinkage,
        per_cluster_targets=args.per_cluster_targets,
        adaptive_shrinkage=args.adaptive_shrinkage,
        min_clusters=args.min_clusters,
        target_blend=args.target_blend,
        adaptive_shrinkage_min=args.adaptive_shrinkage_min,
    )
    _write_output(args.output_path, mf_map, profile)
    logger.info("Saved matched-filter response to %s", args.output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution path.
    raise SystemExit(_cli())
