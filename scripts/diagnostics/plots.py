"""
Plotting helpers for striping diagnostics and SNR case comparisons.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ..core.noise import EPS


def _column_stats(band: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if mask is not None:
        values = np.where(mask, band, np.nan)
    else:
        values = np.where(np.isfinite(band), band, np.nan)
    col_mean = np.nanmean(values, axis=0)
    col_std = np.nanstd(values, axis=0)
    return col_mean, col_std


def plot_striping_diagnostics(
    plain_band: np.ndarray,
    destriped_band: np.ndarray,
    mask: np.ndarray | None,
    power_plain: Mapping[str, np.ndarray | float | None],
    power_ds: Mapping[str, np.ndarray | float | None],
    f0_plain: float | None,
    f0_ds: float | None,
    outpath: str,
    destripe_label: str,
    metadata_lines: Optional[List[str]] = None,
    radiance_unit: str = "Radiance (µW cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)",
) -> None:
    """Visualise striping mitigation before/after destriping."""

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.8, 0.6])

    vmin, vmax = np.nanpercentile(
        np.concatenate([plain_band.ravel(), destriped_band.ravel()]), (1, 99)
    )
    if not np.isfinite(vmin):
        vmin, vmax = np.nanmin(plain_band), np.nanmax(plain_band)

    ax_plain = fig.add_subplot(gs[0, 0])
    im0 = ax_plain.imshow(plain_band, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_plain.set_title("Plain radiance")
    ax_plain.axis("off")
    cbar0 = fig.colorbar(im0, ax=ax_plain, fraction=0.046, pad=0.04)
    cbar0.set_label(radiance_unit)

    ax_ds = fig.add_subplot(gs[0, 1])
    im1 = ax_ds.imshow(destriped_band, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_ds.set_title("Destriped radiance")
    ax_ds.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax_ds, fraction=0.046, pad=0.04)
    cbar1.set_label(radiance_unit)

    mean_plain, std_plain = _column_stats(plain_band, mask)
    mean_ds, std_ds = _column_stats(destriped_band, mask)
    cols = np.arange(mean_plain.size)

    ax_mean = fig.add_subplot(gs[1, 0])
    ax_mean.plot(cols, mean_plain, label="Plain", alpha=0.8)
    ax_mean.plot(cols, mean_ds, label="Destriped", alpha=0.8)
    ax_mean.set_title("Column mean (radiance)")
    ax_mean.set_xlabel("Column index")
    ax_mean.set_ylabel(radiance_unit)
    ax_mean.grid(alpha=0.3)
    ax_mean.legend()

    ax_std = fig.add_subplot(gs[1, 1])
    ax_std.plot(cols, std_plain, label="Plain", alpha=0.8)
    ax_std.plot(cols, std_ds, label="Destriped", alpha=0.8)
    ax_std.set_title("Column σ (radiance)")
    ax_std.set_xlabel("Column index")
    ax_std.set_ylabel(f"σ ({radiance_unit.split(' ',1)[1] if ' ' in radiance_unit else radiance_unit})")
    ax_std.grid(alpha=0.3)
    ax_std.legend()

    ax_pow = fig.add_subplot(gs[2, :])
    freqs_plain = np.asarray(power_plain.get("freqs"))
    power_vals_plain = np.asarray(power_plain.get("power"))
    freqs_ds = np.asarray(power_ds.get("freqs"))
    power_vals_ds = np.asarray(power_ds.get("power"))

    if power_vals_plain.size and power_vals_ds.size:
        ax_pow.plot(freqs_plain, 10 * np.log10(power_vals_plain + EPS), label="Plain", alpha=0.8)
        ax_pow.plot(freqs_ds, 10 * np.log10(power_vals_ds + EPS), label="Destriped", alpha=0.8)
    elif power_vals_plain.size:
        ax_pow.plot(freqs_plain, 10 * np.log10(power_vals_plain + EPS), label="Plain", alpha=0.8)
    elif power_vals_ds.size:
        ax_pow.plot(freqs_ds, 10 * np.log10(power_vals_ds + EPS), label="Destriped", alpha=0.8)

    for f0, style in ((f0_plain, "--"), (f0_ds, "-.")):
        if f0 is not None:
            ax_pow.axvline(f0, linestyle=style, color="k", alpha=0.5)

    ax_pow.set_xlabel("Frequency (cycles/pixel)")
    ax_pow.set_ylabel("Power (dB)")
    ax_pow.set_title("Row-wise FFT power (across-track)")
    ax_pow.grid(alpha=0.3)
    ax_pow.legend()

    header_lines = list(metadata_lines) if metadata_lines else []
    scene_label = header_lines[0] if header_lines else None
    info_lines = header_lines[1:] if len(header_lines) > 1 else []

    title_lines = [destripe_label]
    if scene_label:
        title_lines.insert(0, scene_label)
    fig.suptitle("\n".join(title_lines), fontsize=12, y=0.92)

    if info_lines:
        fig.tight_layout(rect=[0, 0, 1, 0.8])
        fig.text(
            0.01,
            0.98,
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


def plot_snr_cases(
    wl: np.ndarray,
    curves: Iterable[Mapping[str, np.ndarray | str]],
    title: str,
    outpath: str,
    metadata_lines: Optional[List[str]] = None,
) -> None:
    """Plot median and P90 SNR curves for cases A–H."""

    curves_list: List[Mapping[str, object]] = sorted(curves, key=lambda c: c.get("case", ""))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=False)
    axes = np.asarray(axes)
    n_rows, n_cols = axes.shape

    for idx, (ax, curve) in enumerate(zip(axes.ravel(), curves_list)):
        case_id = curve.get("case", "?")
        median = np.asarray(curve.get("snr_median"))
        p90 = np.asarray(curve.get("snr_p90"))
        band_nm = np.asarray(curve.get("band_nm", wl))

        ax.plot(band_nm, median, label="Median", lw=1.4)
        ax.plot(band_nm, p90, label="P90", lw=1.0, alpha=0.6)
        subtitle = f"Case {case_id} • {curve.get('sigma_kind')} • {curve.get('aggregation')}"
        ax.set_title(subtitle)
        ax.grid(alpha=0.3)

        row, col = divmod(idx, n_cols)
        if col == 0:
            ax.set_ylabel("SNR")
        if row == n_rows - 1:
            ax.set_xlabel("Wavelength (nm)")

    # Hide any unused axes when the number of cases is fewer than the grid slots.
    for ax in axes.ravel()[len(curves_list) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    header_lines = list(metadata_lines) if metadata_lines else []
    scene_label = header_lines[0] if header_lines else None
    info_lines = header_lines[1:] if len(header_lines) > 1 else []
    if scene_label:
        fig.suptitle(f"{title}\n{scene_label}", fontsize=14, y=0.9)
    else:
        fig.suptitle(title, fontsize=14, y=0.9)
    if info_lines:
        fig.tight_layout(rect=[0, 0, 1, 0.78])
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
