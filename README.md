# Methane Matched-Filter Pipelines

This repository hosts the refactored CH₄ detection workflows for PRISMA and EnMAP satellites. The legacy monolithic scripts (`scripts/PRISMA/prisma_MF.py`, `scripts/EnMAP/enmap_MF.py`) now route through a modular package under `scripts/`, with shared LUT/target/matched-filter logic and satellite-specific adapters.

## Repository Structure

- `scripts/main.py` – unified CLI entry point (single/batch, PRISMA/EnMAP).
- `scripts/core/` – reusable building blocks (I/O helpers, LUT handling, matched filter, targets).
- `scripts/pipelines/` – orchestration layers per satellite.
- `scripts/satellites/` – low-level satellite utilities (readers, georeferencing, ZIP helpers).
- `test_commands.sh` – curated examples for local end-to-end tests.

## Requirements & Setup

1. Python ≥ 3.11 with GDAL bindings installed (compiled with the same Python).
2. Project dependencies from `requirements-pip.txt`.
3. Access to:
   - PRISMA Level-1/Level-2C HE5 or ZIP archives and a DEM (NetCDF).
   - EnMAP VNIR/SWIR GeoTIFFs with matching METADATA.XML.
   - Methane LUT (`*.hdf5`) compatible with the matched filter.

Quick start with conda/mamba using the curated environment:

```bash
mamba env create -f environment.min.yml
mamba activate hygas
pip install -r requirements-pip.txt
```

The `environment.min.yml` file pins GDAL/PROJ, PyTorch (CPU build), and the essential scientific stack. If you prefer a lighter setup, you can still create your own environment manually and `pip install -r requirements-pip.txt`, but make sure GDAL is compiled against the same Python interpreter.

Confirm GDAL works by importing `osgeo.gdal` inside the environment before running the pipelines.

## CLI Overview

All executions go through:

```bash
python scripts/main.py --satellite {prisma|enmap} --mode {scene|batch} [options]
```

Global options:

- `--min-wavelength / --max-wavelength` – spectral window (nm) forwarded to both pipelines and included in the processing reports.
- `--k` – number of clusters for k-means based target estimation.
- `--log-file` – optional path to capture INFO-level logs in addition to stdout.
- `--save-rads` – PRISMA only; export the full radiance cube GeoTIFF (disabled by default to avoid multi-GB outputs).
- `--snr-reference` – path to the column-wise SNR reference `.npz` used for σ_RMN propagation and JPL MF. If omitted, the pipelines fall back to `PRISMA_SNR_REFERENCE` / `ENMAP_SNR_REFERENCE` env vars.

## PRISMA Manual

### Single Scene

```bash
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 /path/to/PRS_L1_STD_OFFL_YYYYMMDDhhmmss_xxxx.he5 \
  --l2c /path/to/PRS_L2C_STD_YYYYMMDDhhmmss_xxxx.he5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --output /path/to/output_dir \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --prisma-mf-mode srf-column \
  --log-file logs/prisma_scene.log
```

Both `--l1` and `--l2c` accept `.he5` files or ZIP archives. ZIP inputs are unpacked automatically next to the archive, processed, and deleted once the run finishes. When `--output` is omitted the pipeline writes to `<scene_dir>_output`.

`--prisma-mf-mode` mirrors the EnMAP option:

- `srf-column` (default) uses k-means clusters plus column-wise SRF targets (legacy workflow).
- `full-column` skips clustering and derives per-column mean/covariance so the matched filter fully adapts to each detector column.

### Batch Mode

```bash
python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory /path/to/prisma_root \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --output-root /path/to/output_root \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --prisma-mf-mode full-column \
  --log-file logs/prisma_batch.log
```

The batch driver scans every subfolder under `--root-directory`, pairs L1/L2C (HE5 or ZIP), and mirrors the input structure under `--output-root`. If `--output-root` is omitted, outputs land next to each scene folder (e.g., `20240911... -> 20240911..._output`).

## EnMAP Manual

### Single Scene

```bash
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir /path/to/...-SPECTRAL_IMAGE_VNIR.TIF \
  --swir /path/to/...-SPECTRAL_IMAGE_SWIR.TIF \
  --metadata /path/to/...-METADATA.XML \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --output /path/to/output_dir \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode srf-column \
  --log-file logs/enmap_scene.log
```

`--enmap-mf-mode` selects the matched-filter flavor:

- `srf-column` (default) keeps the **MF columnwise SRF with cluster tuning option**, i.e., target spectra are tiled across columns while μ/Σ come from k-means clusters.
- `full-column` activates the true column-wise implementation with per-column mean radiance and covariance (no clustering) so both SRF and statistics adapt to each detector column.

### Batch Mode

```bash
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /path/to/enmap_root \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode full-column \
  --log-file logs/enmap_batch.log
```

Each scene directory inside `--root-directory` must contain the VNIR/SWIR GeoTIFFs and the METADATA.XML file. Outputs are written to `<scene>_output` siblings, mirroring the legacy workflow.

## Outputs & Reporting

Every run produces a set of GeoTIFFs plus a text report under the chosen output directory:

- `*_MF.tif` – matched-filter response.
- `*_concentration.tif` – derived methane concentration map.
- `*_rgb.tif` – quick-look RGB composite.
- `*_classified.tif` – thresholded/classified result.
- `processing_report.txt` – provenance summary (inputs, parameters, spectral window, statistics).

PRISMA batch runs also emit `directory_process_report_<timestamp>.txt` summarizing successes/failures per scene.

## Logging & Diagnostics

- STDOUT always receives INFO logs; use `--log-file path.log` to keep a copy.
- Temporary PRISMA extractions are cleaned automatically; failures to delete are logged as warnings.
- For quick local validation, run the commands stored in `test_commands.sh` (paths assume the `test_data/` bundle under the repo).

## Tips & Troubleshooting

- DEM files are mandatory for PRISMA but ignored for EnMAP.
- When processing ZIP archives, ensure each contains exactly one `.he5` file; otherwise the CLI aborts with a clear error.
- If you see `Missing required ... arguments` revisit the per-mode required options listed above.
- Spectral window and `k` parameters are persisted in the processing report—helpful when comparing runs.

The legacy `scripts/PRISMA/prisma_MF.py` and `scripts/EnMAP/enmap_MF.py` remain callable for backwards compatibility but simply forward into the new CLI. Prefer `scripts/main.py` for all new runs.

## Standalone Smile/SNR Utilities

Beyond the end-to-end pipelines, the repo now ships lightweight analysis scripts useful for quick instrument diagnostics:

- `scripts/enmap_smile.py` – EnMAP-only explorer that reads VNIR/SWIR cubes from GeoTIFF + METADATA and produces mean spectra alongside Δλ and CW/FWHM diagnostics for user-selected local bands (radiance shown in µW·cm⁻²·sr⁻¹·nm⁻¹ to match the LUT convention).
- `scripts/prisma_smile.py` – PRISMA-focused counterpart that pulls CW/FWHM matrices straight from the Level-1 HE5 (or ZIP) archive and renders the same suite of plots for VNIR/SWIR selections.
- `scripts/SNR_enmap.py` – homogeneous-area SNR estimator for EnMAP; shares the methodology described in the matched-filter paper (auto mask, diff/high-pass sigma, optional per-column aggregation) and reports radiance in µW·cm⁻²·sr⁻¹·nm⁻¹.
- `scripts/SNR_prisma.py` – PRISMA-specific SNR estimator that loads radiance cubes via `prisma_utils` (native µW·cm⁻²·sr⁻¹·nm⁻¹ units), handles `.zip` inputs seamlessly, and mirrors the plotting/return structure of the EnMAP variant.
Run them from the repo root so the `scripts.*` imports resolve, e.g.:

```bash
PYTHONPATH=. python scripts/enmap_smile.py --help  # edit the __main__ block for your scene paths
PYTHONPATH=. python scripts/SNR_prisma.py
```

The four plotting-oriented utilities share the same 3×2 layout and rely on the existing `prisma_utils.py` / `enmap_utils.py` readers, so any improvements to the satellite helpers automatically benefit both the operational pipelines and these diagnostics.

## Notebook Guide

All notebooks live under `notebooks/` and are wired to the repository code via `PYTHONPATH=.`, so run them from the repo root (or adjust the first cell accordingly).

- `matched_filter_demo_enmap.ipynb` – runs the full EnMAP CH₄ pipeline on the bundled test scene. **Inputs:** VNIR/SWIR GeoTIFFs, METADATA.XML, CH₄ LUT, DEM/SNR references declared in the config cell. **Outputs:** RGB composite plus matched-filter concentration (`*_MF.tif`) and propagated σ₍RMN₎ rasters written to `notebooks/outputs/pipeline_demo/enmap/` and displayed inline.
- `matched_filter_demo_prisma.ipynb` – same workflow for PRISMA L1/L2C data (ZIP or HE5) alongside DEM/LUT/SNR assets. **Outputs:** RGB, concentration, and σ₍RMN₎ rasters under `notebooks/outputs/pipeline_demo/prisma/`.
- `SNR_experiments_enmap.ipynb` – orchestrates the eight-case SNR CLI runs for EnMAP. **Inputs:** scene configurations (paths/ROIs/band windows/case lists/destriping flags). **Outputs:** CLI artefacts such as `striping_diagnostics.png`, `pca_summary_*.png`, `snr_cases_*.csv|png`, and logs under `notebooks/outputs/enmap/<scene_id>/` plus inline listings.
- `SNR_experiments_prisma.ipynb` – identical orchestration for PRISMA L1/L2C scenes with the same output family stored in `notebooks/outputs/prisma/<scene_id>/`.
- `SNR_experiments_tanager.ipynb` – runs the A–H SNR pipeline for Tanager radiance HDF5 (ROI-aware; unit labels per Tanager spec).
- `diagnostics_uncertainty_enmap.ipynb` – documents how σ₍RMN₎ is produced for EnMAP by walking through band selection, k-means background stats, LUT target synthesis, SNR-reference mapping, and uncertainty propagation. **Inputs:** EnMAP scene folder, DEM, LUT, SNR reference. **Outputs:** console summaries plus an inline σ₍RMN₎ map.
- `diagnostics_uncertainty_prisma.ipynb` – mirrors the above steps for PRISMA L1/L2C inputs, yielding the propagated σ₍RMN₎ raster preview.
- `uncertainty_analysis_enmap.ipynb` – consumes finished EnMAP matched-filter concentration/uncertainty rasters (and optional plume polygons) to compute σ_tot, representative σ₍RMN₎, derived σ_Surf, and plume-level total uncertainty. **Outputs:** diagnostic figures and a JSON metrics report saved to `notebooks/outputs/uncertainty/enmap/`.
- `uncertainty_analysis_prisma.ipynb` – same clutter-versus-instrument breakdown for PRISMA products with metrics saved in `notebooks/outputs/uncertainty/prisma/`.
- `prisma_enmap_comparison.ipynb` – cross-sensor analysis that loads the reference PRISMA/EnMAP cubes plus precomputed SNR cases to compare SNR (case D), spectral smile, and striping metrics. **Outputs:** comparison tables/plots rendered inline and saved next to the configured output directories.
- `test_notebook.ipynb` – minimal placeholder to verify the notebook environment; no external inputs/outputs.
- `SNR_experiments_tanager.ipynb` – runs the A–H SNR experiment pipeline for Tanager radiance HDF5 (same CLI wrapper used for PRISMA/EnMAP).

## PRISMA HDF Exploration

Use `scripts/inspect_prisma_hdf.py` to explore the hierarchy of a Level-1 or Level-2C PRISMA product without leaving the terminal. The tool accepts both `.he5` files and the official ZIP archives, automatically extracting the embedded HE5 to a temporary directory when needed.

- Tree view (optionally capped by depth and including attributes):

  ```bash
  python scripts/inspect_prisma_hdf.py \
    test_data/prisma/20240911071151/PRS_L2C_STD_20240911071151_20240911071155_0001.zip \
    --max-depth 2 --attrs
  ```

- Focus on a specific dataset or group with a quick preview of numeric values:

  ```bash
  python scripts/inspect_prisma_hdf.py \
    test_data/prisma/20240911071151/PRS_L2C_STD_20240911071151_20240911071155_0001.zip \
    --path "HDFEOS/SWATHS/PRS_L2C_WVM/Data Fields/WVM_Map" \
    --preview 5 --attrs
  ```

Append `--output /path/to/report.txt` to save the listing to disk in addition to printing it on screen. `--path` accepts any HDF dataset/group path, `--preview` limits how many numeric values are sampled (the script only reads a thin block to avoid loading whole cubes), and `--max-members` caps how many children are listed when inspecting a group.

## Tanager HDF Exploration

Two small utilities help verify Planet Tanager Basic/Ortho HDF5 deliveries (see `product_spec_docs/tanager/Planet-UserDocumentation-Tanager.pdf` for the field definitions).

- Inspect hierarchy or a specific dataset/attribute (ZIP inputs are unpacked automatically):

  ```bash
  python scripts/inspect_tanager_hdf.py /path/to/tanager_scene.h5 --max-depth 2 --attrs
  python scripts/inspect_tanager_hdf.py /path/to/tanager_scene.zip --path "HDFEOS/SWATHS/HYP/Data Fields/toa_radiance" --preview 8
  ```

- Build a quick RGB preview from the TOA radiance cube (auto-selects 665/565/490 nm bands and applies a 2–98% stretch):

  ```bash
  python scripts/tanager_quicklook.py /path/to/tanager_scene.h5 --summary --output outputs/tanager_rgb.png
  ```

Use `--rgb-wavelengths`, `--stretch`, or `--gamma` to tweak the quicklook rendering, `--pixel r c` to print a per-pixel spectrum, and `--no-mask` to skip applying the `nodata_pixels` mask when deriving RGB.
