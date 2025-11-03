"""
Principal component decomposition utilities and summary plotting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib import gridspec


def pca_decompose(
    cube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    k: int = 6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Perform PCA on the cube (bands, rows, cols) and return reconstruction + residual.
    """

    if cube.ndim != 3:
        raise ValueError("Input cube must have shape (bands, rows, cols).")

    bands, rows, cols = cube.shape
    flat = cube.reshape(bands, -1).T  # (pixels, bands)
    finite = np.all(np.isfinite(flat), axis=1)

    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        if np.any(mask_flat):
            selection = finite & mask_flat
        else:
            selection = finite.copy()
    else:
        selection = finite

    if selection.sum() < max(3, k):
        selection = finite

    train_data = flat[selection]
    if train_data.shape[0] < 2:
        train_data = flat[finite]

    if train_data.shape[0] < 2:
        mean_spectrum = np.nanmean(flat[finite], axis=0) if np.any(finite) else np.zeros(cube.shape[0])
        recon = cube.copy()
        residual = np.zeros_like(cube)
        model = {
            "pca": None,
            "scores_maps": np.zeros((0, rows, cols)),
            "explained_variance_ratio": np.array([]),
            "components": np.empty((0, cube.shape[0])),
            "mean_spectrum": mean_spectrum,
            "mask": mask,
            "reference_pixel": (rows // 2, cols // 2),
            "scores": np.zeros((flat.shape[0], 0)),
        }
        return recon, residual, model

    n_components = int(min(k, train_data.shape[1], train_data.shape[0]))
    if n_components < 1:
        n_components = 1

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(train_data)

    band_mean = pca.mean_
    flat_filled = np.where(np.isfinite(flat), flat, band_mean[None, :])

    scores = pca.transform(flat_filled)
    recon_flat = pca.inverse_transform(scores)

    resid_flat = flat - recon_flat
    invalid = ~np.isfinite(flat)
    recon_flat[invalid] = np.nan
    resid_flat[invalid] = np.nan

    recon = recon_flat.T.reshape(bands, rows, cols)
    residual = resid_flat.T.reshape(bands, rows, cols)

    scores_maps = np.full((n_components, rows, cols), np.nan, dtype=float)
    for i in range(n_components):
        scores_maps[i] = scores[:, i].reshape(rows, cols)
        if mask is not None:
            scores_maps[i][~mask] = np.nan

    ref_candidates = np.flatnonzero(selection)
    if ref_candidates.size == 0:
        ref_candidates = np.flatnonzero(finite)
    ref_flat_idx = int(ref_candidates[0]) if ref_candidates.size else 0
    ref_pixel = np.unravel_index(ref_flat_idx, (rows, cols))

    model = {
        "pca": pca,
        "scores_maps": scores_maps,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "mean_spectrum": band_mean,
        "mask": mask,
        "reference_pixel": ref_pixel,
        "scores": scores,
    }

    return recon, residual, model


def plot_pca_summary(
    model: Dict[str, object],
    cube: np.ndarray,
    recon: np.ndarray,
    residual: np.ndarray,
    wl: np.ndarray,
    outpath: str,
    max_maps: int = 6,
    metadata_lines: Optional[List[str]] = None,
    reference_wavelengths: Optional[Dict[str, Optional[float]]] = None,
) -> None:
    """Generate a PCA diagnostics figure summarising variance, components and residuals."""

    pca: Optional[PCA] = model.get("pca")
    if pca is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wl, model.get("mean_spectrum", np.zeros_like(wl)), color="C0")
        ax.set_title("PCA skipped: insufficient valid pixels")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
        ax.grid(alpha=0.3)
        header_lines = list(metadata_lines) if metadata_lines else []
        scene_label = header_lines[0] if header_lines else None
        info_lines = header_lines[1:] if len(header_lines) > 1 else []
        if scene_label:
            fig.suptitle(scene_label, fontsize=11, y=0.98)
        if info_lines:
            fig.tight_layout(rect=[0, 0, 1, 0.82])
            fig.text(
                0.01,
                0.97,
                "\n".join(info_lines),
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    scores_maps = model["scores_maps"]
    mask = model.get("mask")
    n_maps = min(max_maps, scores_maps.shape[0])

    references = reference_wavelengths or {}
    header_lines = list(metadata_lines) if metadata_lines else []
    scene_label = header_lines[0] if header_lines else None
    info_lines = header_lines[1:] if len(header_lines) > 1 else []
    target_vnir_nm = references.get("vnir_nm")
    target_swir_nm = references.get("swir_nm")
    tolerance_nm = 30.0

    def _band_from_target(target_nm):
        if target_nm is None or wl.size == 0:
            return None
        finite = np.isfinite(wl)
        if not np.any(finite):
            return None
        finite_indices = np.flatnonzero(finite)
        diffs = np.abs(wl[finite] - target_nm)
        idx_local = int(np.argmin(diffs))
        global_idx = int(finite_indices[idx_local])
        if diffs[idx_local] > tolerance_nm:
            return None
        return global_idx

    swir_idx = _band_from_target(target_swir_nm)
    vnir_idx = _band_from_target(target_vnir_nm)

    fig = plt.figure(figsize=(13, 10))
    base_title = "PCA summary"
    if scene_label:
        fig.suptitle(f"{scene_label}\n{base_title}", fontsize=11, y=0.99)
    else:
        fig.suptitle(base_title, fontsize=11, y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.2])

    # Explained variance
    ax_var = fig.add_subplot(gs[0, 0])
    idx = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax_var.bar(idx, pca.explained_variance_ratio_, color="C0", alpha=0.6, label="Explained")
    ax_var.set_xlabel("Component")
    ax_var.set_ylabel("Explained variance ratio")
    ax_var.set_ylim(0, max(pca.explained_variance_ratio_) * 1.2 if len(idx) else 1)
    ax_var.grid(alpha=0.3)

    ax_var2 = ax_var.twinx()
    ax_var2.plot(idx, np.cumsum(pca.explained_variance_ratio_), color="C1", marker="o", label="Cumulative")
    ax_var2.set_ylabel("Cumulative variance")
    ax_var2.set_ylim(0, 1.05)

    handles, labels = [], []
    for ax in (ax_var, ax_var2):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_var.legend(handles, labels, loc="upper right", fontsize=8)

    # Component spectra
    ax_spec = fig.add_subplot(gs[0, 1:])
    for i, comp in enumerate(model["components"], start=1):
        ax_spec.plot(wl, comp, label=f"PC{i}")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Component loading")
    ax_spec.grid(alpha=0.3)
    ax_spec.legend(ncol=2, fontsize=8)

    # Residual maps at reference wavelengths
    ax_swir = fig.add_subplot(gs[1, 0])
    ax_vnir = fig.add_subplot(gs[1, 1])
    ax_specpix = fig.add_subplot(gs[1, 2])

    def _plot_residual_map(ax, band_idx, label):
        if band_idx is None:
            ax.axis("off")
            ax.set_title(f"{label} (not available)")
            return
        res_band = residual[band_idx]
        vmax = np.nanpercentile(np.abs(res_band), 99)
        vmax = vmax if vmax > 0 else np.nanmax(np.abs(res_band))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        im = ax.imshow(res_band, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} {wl[band_idx]:.1f} nm")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")

    _plot_residual_map(ax_swir, swir_idx, "Residual SWIR")
    _plot_residual_map(ax_vnir, vnir_idx, "Residual VNIR")

    # Spectrum comparison for reference pixel
    ref_pixel = model.get("reference_pixel", (residual.shape[1] // 2, residual.shape[2] // 2))
    rr, cc = ref_pixel
    orig_spec = cube[:, rr, cc]
    recon_spec = recon[:, rr, cc]
    resid_spec = residual[:, rr, cc]

    ax_specpix.plot(wl, orig_spec, label="Original", lw=1.2)
    ax_specpix.plot(wl, recon_spec, label="Reconstruction", lw=1.2)
    ax_specpix.plot(wl, resid_spec, label="Residual", lw=1.0)
    ax_specpix.set_xlabel("Wavelength (nm)")
    ax_specpix.set_ylabel("Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax_specpix.set_title(f"Spectrum at pixel ({rr}, {cc})")
    ax_specpix.grid(alpha=0.3)
    ax_specpix.legend(fontsize=8)

    # PC score maps
    maps_gs = gridspec.GridSpecFromSubplotSpec(1, n_maps, subplot_spec=gs[2, :])
    for i in range(n_maps):
        ax = fig.add_subplot(maps_gs[0, i])
        scores = scores_maps[i]
        vmax = np.nanpercentile(np.abs(scores), 99)
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        ax.imshow(scores, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"PC{i + 1} scores", fontsize=9)
        ax.axis("off")

    if info_lines:
        fig.tight_layout(rect=[0, 0, 1, 0.8])
        fig.text(
            0.01,
            0.96,
            "\n".join(info_lines),
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
