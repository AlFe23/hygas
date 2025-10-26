# Methane Matched-Filter Pipelines

This repository hosts the refactored CH₄ detection workflows for PRISMA and EnMAP satellites. The original monolithic scripts (`scripts/PRISMA/prisma_MF.py`, `scripts/EnMAP/enmap_MF.py`) now delegate to a modular package under `scripts/`, with shared LUT/target/matched-filter logic and satellite-specific adapters.

## Running The Pipelines

All executions go through the unified CLI (`scripts/main.py`). Ensure GDAL and the Python dependencies from `requirements-pip.txt` are installed.

Add `--log-file run.log` (optional) to capture the detailed INFO-level logs introduced in the pipelines.

### PRISMA – Single Scene

```bash
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 /path/to/PRS_L1_STD_OFFL_xxx.he5 \
  --l2c /path/to/PRS_L2C_STD_xxx.he5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --output /path/to/output_dir \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1

Add `--save-rads` if you need the full radiance cube GeoTIFF; it is disabled by default to avoid multi-GB outputs.
```

### PRISMA – Batch Mode

```bash
python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory /path/to/input_root \
  --output-root /path/to/output_root \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1
```

The batch driver scans every subfolder under `--root-directory`, pairing L1/L2C HE5 files (or extracting them from matching ZIP archives) and writes outputs to mirrored folders under `--output-root`.

### EnMAP – Single Scene

```bash
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir /path/to/...-SPECTRAL_IMAGE_VNIR.TIF \
  --swir /path/to/...-SPECTRAL_IMAGE_SWIR.TIF \
  --metadata /path/to/...-METADATA.XML \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --output /path/to/output_dir \
  --k 1
```

### EnMAP – Batch Mode

```bash
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /path/to/enmap_root \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --k 1
```

Each subdirectory in `--root-directory` must contain the VNIR/SWIR GeoTIFFs and METADATA.XML. Outputs go to `<scene>_output` folders alongside the source data, mirroring the legacy behavior.

### Notes

- Adjust `--k` and the wavelength window to match your study requirements.
- `--output` (scene mode) and `--output-root` (PRISMA batch) are optional; when omitted, outputs go to sibling folders named `<scene_folder>_output`, mirroring the EnMAP batch convention.
- The legacy entry points remain callable for backwards compatibility and forward their arguments to the unified CLI.
- DEMs are required only for PRISMA processing (EnMAP uses the metadata-provided mean ground elevation).
