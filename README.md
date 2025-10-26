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
  --log-file logs/prisma_scene.log
```

Both `--l1` and `--l2c` accept `.he5` files or ZIP archives. ZIP inputs are unpacked automatically next to the archive, processed, and deleted once the run finishes. When `--output` is omitted the pipeline writes to `<scene_dir>_output`.

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
  --log-file logs/enmap_scene.log
```

### Batch Mode

```bash
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /path/to/enmap_root \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
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
