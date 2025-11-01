#!/usr/bin/env bash
set -euo pipefail

# PRISMA batch test
echo "Running PRISMA batch test..."
python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/prisma/20240911071151 \
  --dem /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/DEM_1Km/srtm30plus_v11_land.nc \
  --lut /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5 \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --log-file logs/prisma_batch.log
  #--output-root /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/prisma/out \

# PRISMA scene test
# it accepts paths to either HE5 or ZIP for L1 and L2C PRISMA files
echo "Running PRISMA scene test..."
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/prisma/20240911071151/PRS_L1_STD_OFFL_20240911071151_20240911071155_0001.zip \
  --l2c /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/prisma/20240911071151/PRS_L2C_STD_20240911071151_20240911071155_0001.zip \
  --dem /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/DEM_1Km/srtm30plus_v11_land.nc \
  --lut /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5 \
  --min-wavelength 2100 \
  --max-wavelength 2450 \
  --k 1 \
  --log-file logs/prisma_scene.log
# --output /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/prisma/output_scene \


# EnMAP scene test
echo "Running EnMAP scene test..."
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-SPECTRAL_IMAGE_VNIR.TIF \
  --swir /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-SPECTRAL_IMAGE_SWIR.TIF \
  --metadata /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/enmap/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z/ENMAP01-____L1B-DT0000092488_20240911T075547Z_001_V010502_20241207T112410Z-METADATA.XML \
  --lut /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5 \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --log-file logs/enmap_scene.log
# --output /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/enmap/output_enmap \

# EnMAP batch test
echo "Running EnMAP batch test..."
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/hygas/test_data/enmap \
  --lut /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5 \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --log-file logs/enmap_batch.log

echo "Running EnMAP batch test..."
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/EnMAP_calibration_data/Roger_et_al_EnMAP_data \
  --lut /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/Matched_filter_approach/LUTs/CH4_lut.hdf5 \
  --k 1 \
  --min-wavelength 2150 \
  --max-wavelength 2450 \
  --log-file logs/enmap_batch.log

# Standalone Radiometric diagnostics
echo "Running EnMAP smile diagnostic..."
PYTHONPATH=. python scripts/enmap_smile.py || echo "enmap_smile.py skipped (missing deps?)"

echo "Running EnMAP SNR diagnostic..."
PYTHONPATH=. python scripts/SNR_enmap.py || echo "SNR_enmap.py skipped (missing deps?)"

echo "Running PRISMA smile diagnostic..."
PYTHONPATH=. python scripts/prisma_smile.py || echo "prisma_smile.py skipped (missing deps?)"

echo "Running PRISMA SNR diagnostic..."
PYTHONPATH=. python scripts/SNR_prisma.py || echo "SNR_prisma.py skipped (missing deps?)"


# Inspect PRISMA HDF structure
echo "Inspecting PRISMA HDF structure..."
python scripts/inspect_prisma_hdf.py \
  "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001.zip" \
  --max-depth 5 --attrs --output /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L1_STD_OFFL_20200401085313_20200401085318_0001_structure.txt


echo "Inspecting PRISMA HDF structure..."
python scripts/inspect_prisma_hdf.py \
  "/mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001.zip" \
  --max-depth 5 --attrs --output /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/SNR/PRISMA_calibration_data/Northern_State_Sudan_20200401/20200401085313_20200401085318/PRS_L2C_STD_20200401085313_20200401085318_0001_structure.txt




