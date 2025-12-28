#!/usr/bin/env bash
set -euo pipefail

# Percorsi base
REPO_ROOT="/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas"
PARENT_ROOT="/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection"

# SNR di riferimento (stessi file usati dai notebook, salvati in reference_snr/)
export PRISMA_SNR_REFERENCE="${REPO_ROOT}/reference_snr/prisma/20200401085313/snr_reference_columnwise.npz"
export ENMAP_SNR_REFERENCE="${REPO_ROOT}/reference_snr/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz"

DEM_FILE="${PARENT_ROOT}/DEM_1Km/srtm30plus_v11_land.nc"
LUT_FILE="${PARENT_ROOT}/LUTs/CH4_lut.hdf5"

PRISMA_ROOT="${REPO_ROOT}/test_data/prisma/Northern_State_Sudan_20200401"
ENMAP_ROOT="${REPO_ROOT}/test_data/enmap/Agadez_Niger_20220712/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z"
ENMAP_BATCH_ROGER="${PARENT_ROOT}/SNR/EnMAP_calibration_data/Roger_et_al_EnMAP_data"

# PRISMA batch
echo "Running PRISMA batch test..."
python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory "${PRISMA_ROOT}" \
  --dem "${DEM_FILE}" \
  --lut "${LUT_FILE}" \
  --snr-reference "${PRISMA_SNR_REFERENCE}" \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --prisma-mf-mode srf-column \
  --log-file logs/prisma_batch.log
# --output-root "${PRISMA_ROOT}/out" \

# PRISMA scena singola
echo "Running PRISMA scene test..."
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 "${PRISMA_ROOT}/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip" \
  --l2c "${PRISMA_ROOT}/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001.zip" \
  --dem "${DEM_FILE}" \
  --lut "${LUT_FILE}" \
  --snr-reference "${PRISMA_SNR_REFERENCE}" \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --prisma-mf-mode srf-column \
  --log-file logs/prisma_scene.log
# --output "${PRISMA_ROOT}/output_scene" \

echo "Running PRISMA scene test (full-column MF)..."
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 "${PRISMA_ROOT}/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip" \
  --l2c "${PRISMA_ROOT}/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001.zip" \
  --dem "${DEM_FILE}" \
  --lut "${LUT_FILE}" \
  --snr-reference "${PRISMA_SNR_REFERENCE}" \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --prisma-mf-mode full-column \
  --output "${PRISMA_ROOT}/output_full_column" \
  --log-file logs/prisma_scene_full_column.log


# EnMAP scena singola
echo "Running EnMAP scene test..."
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_VNIR.TIF" \
  --swir "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_SWIR.TIF" \
  --metadata "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-METADATA.XML" \
  --lut "${LUT_FILE}" \
  --snr-reference "${ENMAP_SNR_REFERENCE}" \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode srf-column \
  --log-file logs/enmap_scene.log
# --output "${REPO_ROOT}/test_data/enmap/output_enmap" \

echo "Running EnMAP scene test (full-column MF)..."
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_VNIR.TIF" \
  --swir "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-SPECTRAL_IMAGE_SWIR.TIF" \
  --metadata "${ENMAP_ROOT}/ENMAP01-____L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z-METADATA.XML" \
  --lut "${LUT_FILE}" \
  --snr-reference "${ENMAP_SNR_REFERENCE}" \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode full-column \
  --output "${REPO_ROOT}/test_data/enmap/output_full_column" \
  --log-file logs/enmap_scene_full_column.log

# EnMAP batch
echo "Running EnMAP batch test..."
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory "${REPO_ROOT}/test_data/enmap/Agadez_Niger_20220712" \
  --lut "${LUT_FILE}" \
  --snr-reference "${ENMAP_SNR_REFERENCE}" \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode srf-column \
  --log-file logs/enmap_batch.log

echo "Running EnMAP batch test..."
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory "${ENMAP_BATCH_ROGER}" \
  --lut "${LUT_FILE}" \
  --snr-reference "${ENMAP_SNR_REFERENCE}" \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --enmap-mf-mode srf-column \
  --log-file logs/enmap_batch.log

# Diagnostica radiometrica standalone
echo "Running EnMAP smile diagnostic..."
PYTHONPATH=. python scripts/enmap_smile.py || echo "enmap_smile.py skipped (missing deps?)"

echo "Running EnMAP SNR diagnostic..."
PYTHONPATH=. python scripts/SNR_enmap.py || echo "SNR_enmap.py skipped (missing deps?)"

echo "Running PRISMA smile diagnostic..."
PYTHONPATH=. python scripts/prisma_smile.py || echo "prisma_smile.py skipped (missing deps?)"

echo "Running PRISMA SNR diagnostic..."
PYTHONPATH=. python scripts/SNR_prisma.py || echo "SNR_prisma.py skipped (missing deps?)"


# Ispezione struttura HDF PRISMA
echo "Inspecting PRISMA HDF structure..."
python scripts/inspect_prisma_hdf.py \
  "${PARENT_ROOT}/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip" \
  --max-depth 5 --attrs --output "${PARENT_ROOT}/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001_structure.txt"


echo "Inspecting PRISMA HDF structure..."
python scripts/inspect_prisma_hdf.py \
  "${PARENT_ROOT}/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001.zip" \
  --max-depth 5 --attrs --output "${PARENT_ROOT}/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001_structure.txt"






  python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Articolo_confronto_sensori/extracted" \
  --lut "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5" \
  --snr-reference "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/notebooks/outputs/enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz" \
  --k 1 \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --enmap-mf-mode full-column \
  --log-file logs/enmap_batch_extracted.log


python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Articolo_confronto_sensori/ready_to_process" \
  --dem "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/DEM_1Km/srtm30plus_v11_land.nc" \
  --lut "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5" \
  --snr-reference "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/reference_snr/prisma/20200401085313/snr_reference_columnwise.npz" \
  --min-wavelength 2100 --max-wavelength 2450 --k 1 \
  --prisma-mf-mode full-column \
  --log-file logs/prisma_batch_articolo.log

