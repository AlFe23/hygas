# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:48:54 2025

@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:57:29 2024

@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:49:16 2024

@author: ferra


v4.3 This version is a modified version of v4.2 in order to make it ready to be compiled into an executable file with pyinstaller

--------------------------------------------------------------------------------------------
v4.4 - This version differentiates from v4.3 for the implementation of ppm_scaling factor in 'calculate_matched_filter_core' function in order to get also gas concentration map
--------------------------------------------------------------------------------------------
v4.5 - introduces the direct computation of concentration in ppmm
--------------------------------------------------------------------------------------------
v4.6 - uses gdalwarp to better project the geotiff
--------------------------------------------------------------------------------------------
v4.7 - utilizza la rads_subselection per k-means classification e tira fuori come output anche il geotiff di radiaze concatenato VNIR/SWIR
--------------------------------------------------------------------------------------------
v4.8 ??
--------------------------------------------------------------------------------------------
ctmf_newMF - si tratta di una nuova versione che implementa diverse migliorie:
    
    - formulazione del filtro adattato rivista per coerenza con letteratura che calcola concentrazione direttatmente
    - selezione dinamica della finestra spettrale da usare per target spectrum e filtro adattato
    - doppia opzione: 
        - spettro target e filtro adattato in versione column-wise
        - spettro target e filtro adattato in versione single column (meanCW, meanFWHM)
    - soluzione del salvataggiodel SR corretto nella proiezione 4326.


ctmf_newMF_auto2  -  The updated code introduces a user-specified output_root_dir to store outputs, preserving the input folder structure within this directory. It dynamically creates output directories, checks for existing outputs to avoid redundant processing, and saves the processing report in the specified output location. This enhances flexibility, organization, and scalability compared to the previous version.

--------------------------------------------------------------------------------------------

ctmf_newMF_auto3 — changes vs previous version



Bug fix (critical):
- Robust DEM handling: mean_elev_fromDEM now ignores NaNs and, when scenes are over
  water or DEM has NoData, falls back to 0.0 km instead of propagating NaN. This
  fixes the crash you observed when elevation was NaN.

Parameter normalization (to stay within LUT domain):
- Added normalize_ground_km(): returns a finite ground altitude in [0, 3] km.
- Added normalize_wv_gcm2(): returns a finite water vapor value in [0, 6] g/cm².
- ch4_detection now normalizes elevation and WV before calling generate_library.

Safer LUT interpolation:
- spline_5deg_lookup now:
  * treats any NaN lookup index as 0,
  * clips each parameter index to a valid 2-cell window (ensuring i0+1 exists),
  * builds “safe” slices to prevent out-of-bounds during interpolation.
  These changes make the 5-D→wave lookup robust to edge/border conditions.

Minor improvements:
- check_param now uses an f-string for clearer error messages.
- mean_elev_fromDEM slices DEM safely even if lat/lon are stored descending and
  always closes the dataset (try/finally).

What did NOT change:
- I/O and outputs (GeoTIFFs, report, target spectra .npy) are unchanged.
- Core MF/concentration computation and column-wise target workflow remain the same.

Net effect:
- Same science and outputs as before, but far more stable over water/NoData scenes.
--------------------------------------------------------------------------------------------

"""



import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from sklearn.cluster import KMeans
from numpy.linalg import inv
from osgeo import gdal, osr
import sys
from os.path import exists
import scipy.ndimage
import argparse
import spectral
import time
import xarray as xr
import pandas as pd
import argparse
import zipfile
import re
import subprocess
from datetime import datetime
import traceback

def generate_report(output_dir, L1_file, L2C_file, dem_file, lut_file, meanWV, SZA, mean_elevation, k, mf_output_file, concentration_output_file, rgb_output_file, classified_output_file):
    """
    Genera un report di elaborazione nella cartella di output, contenente i dettagli dei file di input e output, e i principali parametri calcolati.
    
    Parametri:
    output_dir (str): Cartella di output per salvare il report.
    L1_file (str): Path al file L1.
    L2C_file (str): Path al file L2C.
    dem_file (str): Path al file DEM.
    lut_file (str): Path al file LUT.
    meanWV (float): Valore medio del vapore acqueo.
    SZA (float): Angolo zenitale del sole.
    mean_elevation (float): Elevazione media dell'area di interesse.
    k (int): Numero di cluster utilizzato nel k-means.
    mf_output_file (str): Path al file output del matched filter.
    concentration_output_file (str): Path al file output della mappa di concentrazione.
    rgb_output_file (str): Path al file RGB di output.
    classified_output_file (str): Path al file output classificato.
    """
    # Data e ora dell'elaborazione
    processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Path per il file di report
    report_path = os.path.join(output_dir, "processing_report.txt")
    
    # Contenuto del report
    report_content = f"""
    Processing Report
    -----------------
    Date and Time of Processing: {processing_date}
    
    Input Files:
    - L1 File: {L1_file}
    - L2C File: {L2C_file}
    - DEM File: {dem_file}
    - LUT File: {lut_file}
    
    Output Files:
    - Matched Filter Output: {mf_output_file}
    - Concentration Map Output: {concentration_output_file}
    - RGB Image Output: {rgb_output_file}
    - Classified Image Output: {classified_output_file}
    
    Processing Parameters:
    - Mean Water Vapor: {meanWV} g/cm^2
    - Solar Zenith Angle: {SZA} degrees
    - Mean Elevation: {mean_elevation} km
    - Number of Clusters (k-means): {k}
    """
    
    # Scrivi il report nel file
    with open(report_path, 'w') as file:
        file.write(report_content)
    
    print(f"Report di elaborazione generato in: {report_path}")


def prismaL2C_WV_read(filename):
    
    # Open the HDF5 file
    with h5py.File(filename, 'r') as f:
        # Read the Water vapor Mask from L2C PRISMA data
        PRS_L2C_WVM = f['HDFEOS/SWATHS/PRS_L2C_WVM/Data Fields/WVM_Map'][:]
        latitude_WVM = f['HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Latitude'][:]
        longitude_WVM = f['HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Longitude'][:]

        
        # Read scale factors to transform DN (uint16) into WV physical values (g/cm2)
        L2ScaleWVMMin = f.attrs['L2ScaleWVMMin']
        L2ScaleWVMMax = f.attrs['L2ScaleWVMMax']
       
    #Convert DN to WV in g/cm^2    (pg.213 product spec doc)
    WVM_unit = L2ScaleWVMMin + PRS_L2C_WVM*(L2ScaleWVMMax-L2ScaleWVMMin)/65535
    meanWV = np.mean(WVM_unit)
    # Print the mean Water Vapor value
    print("Mean Water Vapor (g/cm^2):", meanWV)
    
    return  meanWV, PRS_L2C_WVM, latitude_WVM, longitude_WVM

def prismaL1_SZA_read(filename):
    
    # Open the HDF5 file
    with h5py.File(filename, 'r') as f:

        # Read SZA in degrees
        SZA = f.attrs['Sun_zenith_angle']

    # Print the Sun Zenith Angle
    print("Sun Zenith Angle (degrees):", SZA)
    
    return  SZA

def prismaL2C_bbox_read(filename):
    # Open the HDF5 file
    with h5py.File(filename, 'r') as f:
        # Read the geolocation fields
        latitude_WVM = f['HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Latitude'][:]
        longitude_WVM = f['HDFEOS/SWATHS/PRS_L2C_WVM/Geolocation Fields/Longitude'][:]

    # Calculate bounding box
    min_lat = np.min(latitude_WVM)
    max_lat = np.max(latitude_WVM)
    min_lon = np.min(longitude_WVM)
    max_lon = np.max(longitude_WVM)
    
    return (min_lon, max_lon, min_lat, max_lat)

# def mean_elev_fromDEM(dem_file, bbox):
#     # Load the NetCDF file
#     ds = xr.open_dataset(dem_file)
    
#     # Extract bounding box coordinates
#     min_lon, max_lon, min_lat, max_lat = bbox
    
#     # Slice the dataset to only include the area within the bounding box
#     elevation_subset = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    
#     # Calculate and print mean elevation within the bounding box expressed in Km
#     mean_elevation_subset = elevation_subset['elev'].mean() / 1000
#     print("Mean Elevation within Bounding Box in Km:", mean_elevation_subset.values)

def mean_elev_fromDEM(dem_file: str, bbox: tuple) -> float:
    """
    Return mean ground altitude in km over bbox. Falls back to 0.0 km over water/NoData.
    dem_file: NetCDF with variables lat, lon, elev [meters].
    bbox: (min_lon, max_lon, min_lat, max_lat)
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    ds = xr.open_dataset(dem_file)
    try:
        # Slice safely even if lat is descending
        lat_slice = slice(min_lat, max_lat) if ds['lat'][0] <= ds['lat'][-1] else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon) if ds['lon'][0] <= ds['lon'][-1] else slice(max_lon, min_lon)

        elevation_subset = ds.sel(lon=lon_slice, lat=lat_slice)
        if 'elev' not in elevation_subset:
            raise KeyError("Variable 'elev' not found in DEM.")

        arr = elevation_subset['elev']  # meters
        # Robust mean ignoring NaNs
        m_val = arr.where(np.isfinite(arr)).mean(skipna=True).values
        if m_val is None or not np.isfinite(m_val):
            print("Mean Elevation within Bounding Box in Km: NaN → using sea level (0 km).")
            return 0.0
        mean_km = float(m_val) / 1000.0
        print("Mean Elevation within Bounding Box in Km:", mean_km)
        return mean_km
    finally:
        ds.close()


def normalize_ground_km(ground_km: float | np.ndarray, fallback: float = 0.0) -> float:
    """Return a finite ground altitude in km. Over water/NoData → fallback (default 0)."""
    g = float(np.nan_to_num(ground_km, nan=fallback, posinf=fallback, neginf=fallback))
    # keep bounds of your LUT grid, e.g., 0–3 km
    return float(np.clip(g, 0.0, 3.0))

def normalize_wv_gcm2(wv: float | np.ndarray, fallback: float = 0.0) -> float:
    """Finite water vapor in g/cm^2, clipped to LUT domain [0,6]."""
    w = float(np.nan_to_num(wv, nan=fallback, posinf=fallback, neginf=fallback))
    return float(np.clip(w, 0.0, 6.0))


def prisma_read(filename):
    
    # Open the HDF5 file
    with h5py.File(filename, 'r') as f:
        # Read the VNIR and SWIR spectral data
        vnir_cube_DN = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube'][:]
        swir_cube_DN = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube'][:]
    
        # Read central wavelengths and FWHM for VNIR and SWIR bands
        cw_vnir = f['KDP_AUX/Cw_Vnir_Matrix'][:]
        fwhm_vnir = f['KDP_AUX/Fwhm_Vnir_Matrix'][:]
        cw_swir = f['KDP_AUX/Cw_Swir_Matrix'][:]
        fwhm_swir = f['KDP_AUX/Fwhm_Swir_Matrix'][:]
    
        # Read scale factor and offset to transform DN to radiance physical value
        offset_swir = f.attrs['Offset_Swir']
        scaleFactor_swir = f.attrs['ScaleFactor_Swir']
        offset_vnir = f.attrs['Offset_Vnir']
        scaleFactor_vnir = f.attrs['ScaleFactor_Vnir']
        
        # Read geolocation fields
        latitude_vnir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR'][:]
        longitude_vnir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR'][:]
        latitude_swir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR'][:]
        longitude_swir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR'][:]
    
    # Convert DN to radiance for VNIR and SWIR data #[W/(str*um*m^2)]
    swir_cube_rads = (swir_cube_DN / scaleFactor_swir) - offset_swir
    vnir_cube_rads = (vnir_cube_DN / scaleFactor_vnir) - offset_vnir
    
    #Convert PRISMA radiance from [W/(str*um*m^2)] to [μW*cm-2*nm-1*sr-1] in order to meet original AVIRIS radiance unit (i.e. the unit used in MODTRAN6 to build the LUT)
    swir_cube_rads = swir_cube_rads * 0.1
    vnir_cube_rads = vnir_cube_rads * 0.1
    
    # The variables swir_cube and vnir_cube now contain the physical radiance values
    
    
    #############################################################################
    # Extract the specific bands for RGB
    red_rad_channel = vnir_cube_rads[:, 29, :]   # Red channel
    green_rad_channel = vnir_cube_rads[:, 19, :] # Green channel
    blue_rad_channel = vnir_cube_rads[:, 7, :]   # Blue channel
    
    # Normalize each channel to the range [0, 1]
    red_rad_normalized = (red_rad_channel - red_rad_channel.min()) / (red_rad_channel.max() - red_rad_channel.min())
    green_rad_normalized = (green_rad_channel - green_rad_channel.min()) / (green_rad_channel.max() - green_rad_channel.min())
    blue_rad_normalized = (blue_rad_channel - blue_rad_channel.min()) / (blue_rad_channel.max() - blue_rad_channel.min())
    
    # Combine the channels into an RGB image
    rgb_image = np.stack([red_rad_normalized, green_rad_normalized, blue_rad_normalized], axis=-1)
    
    # # Create a larger figure
    # plt.figure(figsize=(10, 10))  # You can adjust the size as needed
    
    # # Plotting
    # plt.imshow(rgb_image)
    # plt.title('RGB Image from Radiance Data')
    # plt.axis('off')  # Turn off axis numbers and labels
    # plt.show(block=False)
    # plt.show()
    #############################################################################
    
    # Convert from BIL to BIP format ( BIL = M-3-N ; BIP = 3-M-N ; BSQ = M-N-3 )
    vnir_cube_bip = np.transpose(vnir_cube_rads, (0, 2, 1))
    swir_cube_bip = np.transpose(swir_cube_rads, (0, 2, 1))
    
    # Rotate 270 degrees counterclockwise (equivalent to 90 degrees clockwise) and flip horizontally
    vnir_cube_bip = np.rot90(vnir_cube_bip, k=-1, axes=(0, 1))  # Rotate 270 degrees counterclockwise
    #vnir_cube_bip = np.flip(vnir_cube_bip, axis=1)  # Flip horizontally along the columns axis
    swir_cube_bip = np.rot90(swir_cube_bip, k=-1, axes=(0, 1))  # Rotate 270 degrees counterclockwise
    #swir_cube_bip = np.flip(swir_cube_bip, axis=1)  # Flip horizontally along the columns axis
    
    # The variables swir_cube_bip and vnir_cube_bip now contain the physical radiance values in BIP format
    
    # PROBLEMA:
    # Si è trovato che le bande 1,2,3 del VNIR cube e le bande 172,173 dello SWIR cube sono azzerate:
    # Q&A PRISMA_ATBD.pdf pg.118	
    # VNIR: Remove bands 1, 2, 3 (0-indexed: 0, 1, 2), for these bands CWs and FWHMs are already set to 0.
    VNIR_cube_clean = np.delete(vnir_cube_bip, [0, 1, 2], axis=2)
    # SWIR: Remove bands 172, 173 (0-indexed: 171, 172), for these bands CWs and FWHMs are already set to 0.
    SWIR_cube_clean = np.delete(swir_cube_bip, [171, 172], axis=2)
    # SWIR: Remove bands 1, 2, 3, 4, since they are redundant with the last 4 bands of VNIR cube
    SWIR_cube_clean = np.delete(SWIR_cube_clean, [0, 1, 2, 3], axis=2)
    
    #Extract actual values of CWs and FWHMs. They are stored in standars (1000,256) arrays and have to be extracted
    cw_vnir = cw_vnir[:, 99:162]
    fwhm_vnir = fwhm_vnir[:, 99:162]
    cw_swir = cw_swir[:, 81:252]
    fwhm_swir = fwhm_swir[:, 81:252]
    
    #Reverse CWs and FWHMs vectors as they are in decrasing frequency order, but we want them opposite
    cw_vnir = cw_vnir[:, ::-1]
    fwhm_vnir = fwhm_vnir[:, ::-1]
    cw_swir = cw_swir[:, ::-1]
    fwhm_swir = fwhm_swir[:, ::-1]
    
    # SWIR: Remove bands 1, 2, 3, 4, since they are redundant with the last 4 bands of VNIR cube
    cw_swir_clean = np.delete(cw_swir, [0, 1, 2, 3], axis=1)
    fwhm_swir_clean = np.delete(fwhm_swir, [0, 1, 2, 3], axis=1)
    
    
    #Let's now concatenate the arrays for radiance cube, central wavelengths and FWHMs
    # Concatenate VNIR and SWIR cubes along the band direction
    # Make sure VNIR_cube_filtered and SWIR_cube_filtered are your actual data cubes
    concatenated_cube = np.concatenate((SWIR_cube_clean, VNIR_cube_clean), axis=2)
    # Reverse frequnecies representation according to CWs and FWHMs vectors
    concatenated_cube = concatenated_cube[:, :, ::-1]
    #####################################concatenated_cube =  np.rot90(concatenated_cube,k=3) # si è spostata questa operazione ai due cubi VNIR e SWIR direttamente
    # Concatenate CW and FWHM arrays
    # These should be the actual SRF variables
    concatenated_cw = np.concatenate((cw_vnir, cw_swir_clean), axis=1)
    concatenated_fwhm = np.concatenate((fwhm_vnir, fwhm_swir_clean), axis=1)
    
    #############################################################################
    #Plotting radiance at each band, averaged over all pixels of the image
    #let's plot the average radiance
    mean_rads_concatenated_cube = np.mean(concatenated_cube, axis=(0,1))
    
    # # Coordinates of the pixel (replace with your desired coordinates)
    # x, y = 700, 300
    # # Extracting the radiance values for the selected pixel
    # radiance_values = concatenated_cube[x, y, :]
    
    # # Plotting 
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.mean(concatenated_cw, axis=0), mean_rads_concatenated_cube, label='Radiance vs Wavelength')
    # plt.xlabel('Central Wavelength (nm)')
    # plt.ylabel('Radiance')
    # plt.title('average Radiance Spectrum (vs. mean CWs)')
    # plt.legend()
    # plt.show(block=False)
    # plt.show()
    #############################################################################
    
    #Print the RGB image
    # Assuming rads_array is your (1000, 1000, n_bands) array with radiance data

    # Extract the specific bands for RGB
    red_channel = concatenated_cube[:, :, 29]   # Red channel
    green_channel = concatenated_cube[:, :, 19] # Green channel
    blue_channel = concatenated_cube[:, :, 7]   # Blue channel
    
    # Normalize each channel to the range [0, 1]
    red_normalized = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
    green_normalized = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
    blue_normalized = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
    
    # Combine the channels into an RGB image
    rgb_image = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)
    
    # # Create a larger figure
    # plt.figure(figsize=(10, 10))  # You can adjust the size as needed
    
    # # Plotting
    # plt.imshow(rgb_image)
    # plt.title('RGB Image from Radiance Data')
    # plt.axis('off')  # Turn off axis numbers and labels
    # plt.show(block=False)
    # plt.show()
    #############################################################################
    
    return concatenated_cube, concatenated_cw, concatenated_fwhm, rgb_image, vnir_cube_bip, swir_cube_bip, latitude_vnir, longitude_vnir, latitude_swir, longitude_swir

def k_means_hyperspectral(image, k):
    """
    
    Per eseguire una classificazione non supervisionata utilizzando il metodo K-means su un'immagine iperspettrale, dovrai seguire alcuni passaggi chiave. Prima di tutto, avrai bisogno dell'immagine iperspettrale e della scelta del numero di cluster (K) per il K-means. Gli step includono:

    1. **Caricare l'Immagine Iperspettrale**: Questa immagine sarà un array tridimensionale con dimensioni (n_righe, n_colonne, n_bande), dove ogni "banda" rappresenta un diverso spettro di luce catturato.
    
    2. **Rimodellare l'Immagine per il Clustering**: Il K-means lavora su dati bidimensionali, quindi devi rimodellare l'immagine da (n_righe, n_colonne, n_bande) a (n_righe*n_colonne, n_bande). In questo modo, ogni pixel (ora una riga nel nuovo array) sarà rappresentato dal suo spettro (le bande).
    
    3. **Applicare il K-means Clustering**: Esegui l'algoritmo K-means sui dati rimodellati. Ciò raggrupperà i pixel in K gruppi basati sulle loro proprietà spettrali.
    
    4. **Rimodellare l'Output per l'Immagine Classificata**: Dopo il clustering, avrai le etichette dei cluster per ogni pixel. Queste etichette devono essere rimodellate di nuovo in un formato di immagine (n_righe, n_colonne) per visualizzare l'immagine classificata.
    
    5. **Visualizzare l'Immagine Classificata**: Ogni pixel dell'immagine può ora essere colorato in base al cluster di appartenenza, permettendoti di visualizzare i diversi cluster.
    
    Se hai l'immagine iperspettrale e vuoi procedere con la classificazione, posso aiutarti a scrivere e eseguire il codice necessario. Fammi sapere se hai l'immagine e il numero di cluster desiderato.
    
    Applica il K-means clustering a un'immagine iperspettrale.

    Parametri:
    image (numpy array): L'immagine iperspettrale con dimensioni (n_righe, n_colonne, n_bande).
    k (int): Il numero di cluster da utilizzare nel K-means.

    Ritorna:
    numpy array: Immagine classificata con dimensioni (n_righe, n_colonne).
    """

    # Ottieni le dimensioni dell'immagine
    n_righe, n_colonne, n_bande = image.shape

    # Rimodella l'immagine per il clustering
    reshaped_image = image.reshape((n_righe * n_colonne, n_bande))

    # Applica il K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(reshaped_image)

    # Ottieni le etichette dei cluster e rimodella per l'immagine classificata
    clustered_image = kmeans.labels_.reshape((n_righe, n_colonne))

    return clustered_image

def plot_classified_image(classified_image, title='Classified Image'):
    """
    Visualizza l'immagine classificata.

    Parametri:
    classified_image (numpy array): Immagine classificata con dimensioni (n_righe, n_colonne).
    title (str): Titolo del grafico.
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(classified_image, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)
    plt.show()

def calculate_statistics(image, classified_image, k):
    """
    Calcola la radianza media e la matrice di covarianza per ogni cluster.

    Parametri:
    image (numpy array): Immagine iperspettrale originale con dimensioni (n_righe, n_colonne, n_bande).
    classified_image (numpy array): Immagine classificata con dimensioni (n_righe, n_colonne).
    k (int): Numero di cluster.

    Ritorna:
    mean_radiance (list of numpy arrays): Radianza media per ogni cluster.
    covariance_matrices (list of numpy arrays): Matrici di covarianza per ogni cluster.
    """

    n_righe, n_colonne, n_bande = image.shape
    mean_radiance = []
    covariance_matrices = []

    # Rimodella l'immagine per un facile accesso ai pixel
    reshaped_image = image.reshape((n_righe * n_colonne, n_bande))

    for cluster in range(k):
        # Estrai i pixel di questo cluster
        pixels = reshaped_image[classified_image.reshape(n_righe * n_colonne) == cluster]

        # Calcola la radianza media
        mean_radiance.append(np.mean(pixels, axis=0))

        # Calcola la matrice di covarianza
        covariance_matrices.append(np.cov(pixels, rowvar=False))

    return mean_radiance, covariance_matrices

def extract_number(file_path):
    file_name = os.path.basename(file_path)  # Extract the file name from the full path
    return int(file_name.split('.')[0])

def generate_template_from_bands(centers, fwhm, simRads, simWave, concentrations):

    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers.
    :param fwhm: full width half maximum for the gaussian kernel of each band.
    :return template: the unit absorption spectum
    """
    # import scipy.stats
    #SCALING = 1e5
    SCALING = 1
    # centers = np.asarray(centers)
    # fwhm = np.asarray(fwhm)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError('Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).')
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError('Length of band center wavelengths and band fwhm arrays must be equal.')
    # lib = spectral.io.envi.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.hdr'),
    #                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.lut'))
    # ######## Run these 3 line instead of the command above if you want to run section of the code with 'F9'
    # current_dir = os.getcwd()
    # lib = spectral.io.envi.open(os.path.join(current_dir, 'ch4.hdr'),
    #                             os.path.join(current_dir, 'ch4.lut'))
    # ####################################################################
    # rads = np.asarray(lib.asarray()).squeeze()
    # wave = np.asarray(lib.bands.centers)
    #concentrations = np.asarray([0, 500, 1000, 2000, 4000, 8000, 16000])
    # sigma = fwhm / ( 2 * sqrt( 2 * ln(2) ) )  ~=  fwhm / 2.355
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
    # Evaluate normal distribution explicitly
    var = sigma ** 2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-(simWave[:, None] - centers[None, :])**2 / (2*var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(axis=0), where=response.sum(axis=0) > 0, out=response)
    # implement resampling as matrix multiply
    resampled = simRads.dot(response)
    lograd = np.log(resampled, out=np.zeros_like(resampled), where=resampled > 0)
    slope, _, _, _ = np.linalg.lstsq(np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None)
    spectrum = slope[1, :] * SCALING
    target = np.stack((centers, spectrum)).T  # np.stack((np.arange(spectrum.shape[0]), centers, spectrum)).T
    return target

def calculate_matched_filter(rads_array, classified_image, mean_radiance, covariance_matrices, target_spectra, k):
    """
    Applies the matched filter to the hyperspectral data according to the formulation in the paper.

    Parameters:
    - rads_array: The hyperspectral radiance data cube (rows x cols x bands).
    - classified_image: The classification map from k-means clustering.
    - mean_radiance: List or array of mean radiance vectors for each class.
    - covariance_matrices: List or array of covariance matrices for each class.
    - target_spectra: The unit absorption spectrum (k) for methane.
    - k: Number of clusters/classes.

    Returns:
    - concentration_map: The CH4 concentration enhancement map.
    """
    n_rows, n_columns, n_bands = rads_array.shape
    concentration_map = np.zeros((n_rows, n_columns))
    regularization = 1e-6

    for cls in range(k):
        # Inverse of the regularized covariance matrix
        inv_cov_matrix_cls = inv(covariance_matrices[cls] + regularization * np.eye(n_bands))

        # Mean radiance for the current class
        mean_radiance_cls = mean_radiance[cls]

        # Target signature including mean radiance (t = mu * k)
        target_spc_cls = mean_radiance_cls * target_spectra

        # Pixels belonging to the current class
        class_mask = (classified_image == cls)
        pixels = rads_array[class_mask]

        # Numerator and denominator of the concentration enhancement formula
        numerator = (pixels - mean_radiance_cls) @ inv_cov_matrix_cls @ target_spc_cls
        denominator = target_spc_cls.T @ inv_cov_matrix_cls @ target_spc_cls

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration_map[class_mask] = numerator / denominator

    return concentration_map


def calculate_matched_filter_columnwise(rads_array, classified_image, mean_radiance, covariance_matrices, target_spectra, k):
    """
    Calculate methane concentration maps using a matched filter approach on a column-by-column basis, 
    taking into account the spectral smile effect.

    Parameters:
    - rads_array (numpy array): Hyperspectral radiance data cube with dimensions (rows x cols x bands).
    - classified_image (numpy array): Cluster classification map of dimensions (rows x cols), where each 
      pixel's value corresponds to the cluster/class it belongs to.
    - mean_radiance (list of numpy arrays): A list containing the mean radiance spectrum for each cluster. 
      Each element in the list is a 1D array representing the mean spectrum (length = number of bands).
    - covariance_matrices (list of numpy arrays): A list of covariance matrices, one for each cluster. Each 
      covariance matrix corresponds to that cluster's pixels, size (bands x bands).
    - target_spectra (numpy array): A 2D array representing the unit absorption spectrum k for methane 
      for each column. Dimensions: (bands x columns).
      This represents the column-wise variation of the unit absorption signature due to the spectral smile.
    - k (int): Number of clusters/classes.

    Returns:
    - concentration_map (numpy array): Computed methane concentration enhancement map (rows x cols).

    Note on spectral smile:
    Spectral smile refers to the variation in the central wavelength of each band across the spatial dimension 
    of the sensor. By computing a column-wise target spectrum, we account for slight shifts in the spectral 
    bands per column, improving the accuracy of the matched filter output.

    Key difference from single-column approach:
    - We multiply the unit absorption spectrum (k) by the mean radiance vector μ of the cluster (μ * k) to 
      form the target spectrum t. In the single-column method, this was done once for all pixels. In the 
      column-wise method, we must incorporate this multiplication for each column to correctly account 
      for spectral shifts.
    """

    n_rows, n_columns, n_bands = rads_array.shape

    # Initialize output arrays
    concentration_map = np.zeros((n_rows, n_columns), dtype=np.float32)

    # Regularization constant to ensure numerical stability in matrix inversion
    regularization = 1e-6

    # Process each column independently to incorporate spectral smile
    for w in range(n_columns):
        # For each class, we need to prepare the adapted filter and normalization 
        # based on the column-specific target spectrum.
        adapted_filters = []
        target_normalizations = []

        # Extract the column-specific unit absorption spectrum (k)
        # This is the starting point for forming our target vector t = μ * k.
        column_unit_spectrum = target_spectra[:, w]  # shape: (bands,)

        for cls in range(k):
            # Retrieve the mean radiance for this class
            mean_radiance_cls = mean_radiance[cls]

            # Construct the target spectrum t by scaling the unit absorption spectrum k by μ:
            # t = μ * k
            target_spectrum = mean_radiance_cls * column_unit_spectrum

            # Retrieve and regularize the covariance matrix for numerical stability
            cov_matrix_cls = covariance_matrices[cls] + regularization * np.eye(n_bands)

            # Invert the covariance matrix
            inv_cov_matrix_cls = np.linalg.inv(cov_matrix_cls)

            # Compute the denominator of the matched filter for normalization:
            # (t^T * Σ^{-1} * t)
            target_normalization = target_spectrum.T @ inv_cov_matrix_cls @ target_spectrum

            # Compute the adapted filter for the target:
            # Σ^{-1} * t
            adapted_filter = inv_cov_matrix_cls @ target_spectrum

            # Store results for this class
            adapted_filters.append(adapted_filter)
            target_normalizations.append(target_normalization)

        # Apply the matched filter to each pixel in this column
        for i in range(n_rows):
            # Identify the class of the current pixel
            pixel_class = classified_image[i, w]
            mean_radiance_cls = mean_radiance[pixel_class]

            # Retrieve the adapted filter and normalization for this pixel's class
            adapted_filter = adapted_filters[pixel_class]
            target_normalization = target_normalizations[pixel_class]

            # Extract the pixel's measured radiance spectrum
            pixel_spectrum = rads_array[i, w, :]

            # Compute the numerator of the matched filter:
            # (r - μ)^T * Σ^{-1} * t = (pixel_spectrum - mean_radiance_cls)^T * adapted_filter
            numerator = (pixel_spectrum - mean_radiance_cls).T @ adapted_filter

            # Compute the concentration value:
            # concentration = numerator / (t^T * Σ^{-1} * t)
            # Check for zero to avoid division by zero (unlikely unless target_normalization == 0)
            concentration_value = numerator / target_normalization if target_normalization != 0 else 0.0

            # Store the concentration result
            concentration_map[i, w] = concentration_value

    return concentration_map









def plot_matched_filter_scores(scores, title='Matched Filter Scores'):
    """
    Visualizza i punteggi del filtro adattato utilizzando la palette turbo.

    Parametri:
    scores (numpy array): Array dei punteggi del filtro adattato.
    title (str): Titolo del grafico.
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(scores, cmap='turbo')
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)
    plt.show()

def save_as_geotiff_multichannel(data, output_file, latitude_vnir, longitude_vnir):
    """
    Save a multichannel numpy array as a GeoTIFF with EPSG:4326 projection and enhanced metadata.

    Parameters:
    data (numpy array): Multichannel data array (rows x cols x bands).
    output_file (str): Path to the output GeoTIFF file.
    latitude_vnir (numpy array): Latitude geolocation array.
    longitude_vnir (numpy array): Longitude geolocation array.
    """
    # Ensure the data is in float32 format
    data_float32 = data.astype(np.float32)

    # Revert the rotation applied in `prisma_read`
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))

    # Temporary files to hold intermediate GeoTIFF and VRT
    temp_file = 'temp_output.tif'
    vrt_file = 'temp_output.vrt'
    lat_file = 'latitude.tif'
    lon_file = 'longitude.tif'

    # Create temporary files for latitude and longitude
    driver = gdal.GetDriverByName('GTiff')
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)

    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))

    lat_ds = None
    lon_ds = None

    # Create a temporary GeoTIFF file without detailed geotransform and projection
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], data.shape[2], gdal.GDT_Float32)

    # Write multichannel data to each band
    for i in range(data.shape[2]):
        dataset.GetRasterBand(i + 1).WriteArray(data_float32[:, :, i])

    # Save and close the dataset
    dataset.FlushCache()
    dataset = None

    # Create the VRT file that uses latitude and longitude for georeferencing
    vrt_options = gdal.TranslateOptions(format='VRT')
    gdal.Translate(vrt_file, temp_file, options=vrt_options)

    # Open the VRT file and set geolocation metadata
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata({
        'X_DATASET': lon_file,
        'X_BAND': '1',
        'Y_DATASET': lat_file,
        'Y_BAND': '1',
        'PIXEL_OFFSET': '0',
        'LINE_OFFSET': '0',
        'PIXEL_STEP': '1',
        'LINE_STEP': '1'
    }, 'GEOLOCATION')

    # Explicitly set the CRS as EPSG:4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())

    # Metadata statement
    description = ("This product has been generated by Alvise Ferrari for School of Aerospace Engineering, "
                   "La Sapienza, under terms of license of CLEAR-UP, a project funded by the Italian Space Agency. "
                   "The dissemination of this product is closely linked to the agreements established under the CLEAR-UP project. "
                   "The authors of the code by which the product was generated cannot be held responsible for any improper use or dissemination of this product.")
    vrt_ds.SetMetadataItem('DESCRIPTION', description)
    vrt_ds = None

    # Use gdalwarp to finalize the projection and georeferencing
    subprocess.run([
        'gdalwarp',
        '-geoloc',
        '-t_srs', 'EPSG:4326',  # Ensure CRS is explicitly set
        vrt_file,
        output_file
    ], check=True)

    # Remove temporary files
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)


def save_as_geotiff_single_band(data, output_file, latitude_vnir, longitude_vnir):
    """
    Save a single-band numpy array as a GeoTIFF with EPSG:4326 projection and enhanced metadata.

    Parameters:
    data (numpy array): Single-band data array.
    output_file (str): Path to the output GeoTIFF file.
    latitude_vnir (numpy array): Latitude geolocation array.
    longitude_vnir (numpy array): Longitude geolocation array.
    """
    # Ensure the data is in float32 format
    data_float32 = data.astype(np.float32)

    # Revert the rotation applied in `prisma_read`
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))

    # Temporary files to hold intermediate GeoTIFF and VRT
    temp_file = 'temp_output.tif'
    vrt_file = 'temp_output.vrt'
    lat_file = 'latitude.tif'
    lon_file = 'longitude.tif'

    # Create temporary files for latitude and longitude
    driver = gdal.GetDriverByName('GTiff')
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)

    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))

    lat_ds = None
    lon_ds = None

    # Create a temporary GeoTIFF file without detailed geotransform and projection
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(data_float32)

    # Save and close the dataset
    dataset.FlushCache()
    dataset = None

    # Create the VRT file that uses latitude and longitude for georeferencing
    vrt_options = gdal.TranslateOptions(format='VRT')
    gdal.Translate(vrt_file, temp_file, options=vrt_options)

    # Open the VRT file and set geolocation metadata
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata({
        'X_DATASET': lon_file,
        'X_BAND': '1',
        'Y_DATASET': lat_file,
        'Y_BAND': '1',
        'PIXEL_OFFSET': '0',
        'LINE_OFFSET': '0',
        'PIXEL_STEP': '1',
        'LINE_STEP': '1'
    }, 'GEOLOCATION')

    # Explicitly set the CRS as EPSG:4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())

    # Metadata statement
    description = ("This product has been generated by Alvise Ferrari for School of Aerospace Engineering, "
                   "La Sapienza, under terms of license of CLEAR-UP, a project funded by the Italian Space Agency. "
                   "The dissemination of this product is closely linked to the agreements established under the CLEAR-UP project. "
                   "The authors of the code by which the product was generated cannot be held responsible for any improper use or dissemination of this product.")
    vrt_ds.SetMetadataItem('DESCRIPTION', description)
    vrt_ds = None

    # Use gdalwarp to finalize the projection and georeferencing
    subprocess.run([
        'gdalwarp',
        '-geoloc',
        '-t_srs', 'EPSG:4326',  # Ensure CRS is explicitly set
        vrt_file,
        output_file
    ], check=True)

    # Remove temporary files
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)

    

###################################################################################
#FUNZIONI PER ESTRARRE E DEFINIRE LE VARIABILI NECESSARIE ALL'ESTRAZIONE DELLE FIRME SPETTRALI DALLA LUT



def check_param(value, min, max, name):
    if value < min or value > max:
        raise ValueError(
            f"The value for {name} exceeds the sampled parameter space. "
            f"The limits are [{min}, {max}], requested {value}."
        )

@np.vectorize
# [0.,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
def get_5deg_zenith_angle_index(zenith_value):
    check_param(zenith_value, 0, 80, 'Zenith Angle')
    return zenith_value / 5

@np.vectorize
def get_5deg_sensor_height_index(sensor_value):  # [1, 2, 4, 10, 20, 120]
    # Only check lower bound here, atmosphere ends at 120 km so clamping there is okay.
    check_param(sensor_value, 1, np.inf, 'Sensor Height')
    # There's not really a pattern here, so just linearly interpolate between values -- piecewise linear
    if sensor_value < 1.0:
        return np.float64(0.0)
    elif sensor_value < 2.0:
        idx = sensor_value - 1.0
        return idx
    elif sensor_value < 4:
        return sensor_value / 2
    elif sensor_value < 10:
        return (sensor_value / 6) + (4.0 / 3.0)
    elif sensor_value < 20:
        return (sensor_value / 10) + 2
    elif sensor_value < 120:
        return (sensor_value / 100) + 3.8
    else:
        return 5

# @np.vectorize
# def get_5deg_ground_altitude_index(ground_value):  # [0, 0.5, 1.0, 2.0, 3.0]
#     check_param(ground_value, 0, 3, 'Ground Altitude')
#     if ground_value < 1:
#         return 2 * ground_value
#     else:
#         return 1 + ground_value
    
@np.vectorize
def get_5deg_ground_altitude_index(ground_value):
    if not np.isfinite(ground_value):
        ground_value = 0.0
    ground_value = float(np.clip(ground_value, 0.0, 3.0))
    return 2.0 * ground_value if ground_value < 1.0 else 1.0 + ground_value


@np.vectorize
def get_5deg_water_vapor_index(water_value):  # [0,1,2,3,4,5,6]
    check_param(water_value, 0, 6, 'Water Vapor')
    return water_value

@np.vectorize
# [0.0,1000,2000,4000,8000,16000,32000,64000]
def get_5deg_methane_index(methane_value):
    # the parameter clamps should rarely be calle because there are default concentrations, but the --concentraitons parameter exposes these
    check_param(methane_value, 0, 64000, 'Methane Concentration')
    if methane_value <= 0:
        return 0
    elif methane_value < 1000:
        return methane_value / 1000
    return np.log2(methane_value / 500)

@np.vectorize
def get_carbon_dioxide_index(coo_value):
    check_param(coo_value, 0, 1280000, 'Carbon Dioxode Concentration')
    if coo_value <= 0:
        return 0
    elif coo_value < 20000:
        return coo_value / 20000
    return np.log2(coo_value / 10000)

def get_5deg_lookup_index(zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4'):
    if 'ch4' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_5deg_methane_index(conc)]])
    elif 'co2' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_carbon_dioxide_index(conc)]])
    else:
        raise ValueError('Unknown gas provided.')
    return idx

def spline_5deg_lookup(grid_data, zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4', order=1):
    coords = get_5deg_lookup_index(
        zenith=zenith, sensor=sensor, ground=ground, water=water, conc=conc, gas=gas
    )

    # Common prep: split whole/fractional and build safe slices
    coords_fractional_part, coords_whole_part = np.modf(coords)
    coords_whole_part = np.nan_to_num(coords_whole_part, nan=0.0)

    # clip to valid 2-cell windows for each param dim (exclude wavelength dim)
    grid_shape_no_wave = grid_data.shape[:-1]  # (zen, sensor, ground, water, conc)
    safe_slices = []
    for dim, c in enumerate(coords_whole_part.flatten()):
        max_i = grid_shape_no_wave[dim] - 1
        i0 = int(np.clip(c, 0, max_i - 1))  # ensure i0+1 exists
        safe_slices.append(slice(i0, i0 + 2))
    coords_near_slice = tuple(safe_slices)
    near_grid_data = grid_data[coords_near_slice]  # shape (2,2,2,2,2,wave)

    if order == 1:
        # Build coordinates for scipy: first 5 rows are fractional along the 5 param dims,
        # last row is the wavelength index 0..(w-1)
        new_coord = np.concatenate(
            (coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
             np.arange(near_grid_data.shape[-1])[None, :]),
            axis=0
        )
        lookup = scipy.ndimage.map_coordinates(
            near_grid_data, coordinates=new_coord, order=1, mode='nearest'
        )
        return lookup.squeeze()

    elif order == 3:
        # 3rd order directly on param-space (no wavelength mixing window)
        lookup = np.asarray([
            scipy.ndimage.map_coordinates(
                im, coordinates=coords_fractional_part, order=3, mode='nearest'
            )
            for im in np.moveaxis(near_grid_data, 5, 0)  # iterate wavelength
        ])
        return lookup.squeeze()

    else:
        raise ValueError("order must be 1 or 3")



###################################################################################
#FUNZIONI PER LA LETTURA DELLA LUT

def load_ch4_dataset(lut_file_path):
    # Ensure the function uses the passed file path instead of a hardcoded one
    datafile = h5py.File(lut_file_path, 'r', rdcc_nbytes=4194304)
    return datafile['modtran_data'], datafile['modtran_param'], datafile['wave'], 'ch4'

def generate_library(gas_concentration_vals, lut_file, zenith=0, sensor=120, ground=0, water=0, order=1, dataset_fcn=load_ch4_dataset):
    # Use the passed `dataset_fcn` function, allowing for flexibility in data loading.
    grid, params, wave, gas = dataset_fcn(lut_file)
    rads = np.empty((len(gas_concentration_vals), grid.shape[-1]))
    for i, ppmm in enumerate(gas_concentration_vals):
        rads[i, :] = spline_5deg_lookup(
            grid, zenith=zenith, sensor=sensor, ground=ground, water=water, conc=ppmm, gas=gas, order=order)
    return rads, np.array(wave)


###################################################################################






def ch4_detection(L1_file, L2C_file, dem_file, lut_file, output_dir, min_wavelength, max_wavelength , k):
   
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read SZA from L1 image attributes
    SZA = prismaL1_SZA_read(L1_file)
    
    # Read meanWV from L2C Water Vapor Map product
    meanWV, PRS_L2C_WVM, latitude_WVM, longitude_WVM = prismaL2C_WV_read(L2C_file)
    
    # Read bounding box from PRISMA L2C file
    bbox = prismaL2C_bbox_read(L2C_file)
    
    # Analyze the DEM based on the bounding box from PRISMA
    mean_elevation = mean_elev_fromDEM(dem_file, bbox)
    
    # Extract the file name without the extension for output files
    _, full_filename = os.path.split(L1_file)
    filename_without_extension = os.path.splitext(full_filename)[0]
    
    # Define output filenames in the specified output directory
    target_spectra_export_name = os.path.join(output_dir, f"{filename_without_extension}_CH4_target_PRISMA_conv.npy")
    mf_output_file = os.path.join(output_dir, f"{filename_without_extension}_MF.tif")
    concentration_output_file = os.path.join(output_dir,  f"{filename_without_extension}_MF_concentration.tif")
    rgb_output_file = os.path.join(output_dir, f"{filename_without_extension}_rgb.tif")
    rads_output_file = os.path.join(output_dir, f"{filename_without_extension}_rads.tif")
    classified_output_file = os.path.join(output_dir, f"{filename_without_extension}_classified.tif")
    
    # Data extraction and processing
    rads_array, cw_array, fwhm_array, rgb_image, vnir_cube_bip, swir_cube_bip, latitude_vnir, longitude_vnir, latitude_swir, longitude_swir = prisma_read(L1_file)
    
    # Compute mean central wavelengths per band
    mean_cw = np.mean(cw_array, axis=0)
    mean_fwhm = np.mean(fwhm_array, axis=0)
    
    # Define the spectral window of interest
    # min_wavelength and max_wavelength are function parameters
    # For example:
    # min_wavelength = 1500  # in nm
    # max_wavelength = 2500  # in nm
    
    # Find indices of bands within the spectral window
    band_indices = np.where((mean_cw >= min_wavelength) & (mean_cw <= max_wavelength))[0]
    
    # Subselect the bands based on wavelength
    rads_array_subselection = rads_array[:, :, band_indices]
    cw_subselection = cw_array[:,band_indices]
    fwhm_subselection = fwhm_array[:,band_indices]
    
    mean_cw_subselection = mean_cw[band_indices]
    mean_fwhm_subselection = mean_fwhm[band_indices]
    
    # k-means classification using the same bands as the matched filter or all available bands
    classified_image = k_means_hyperspectral(rads_array_subselection, k)
    print("k-means classification completed")
    
    #plot_classified_image(classified_image)
    
    mean_radiance, covariance_matrices = calculate_statistics(rads_array_subselection, classified_image, k)
    
    # Target spectrum calculation and matched filter application
    concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    
    # simRads_array, simWave_array = generate_library(
    #     concentrations, lut_file, zenith=SZA, sensor=120, ground=mean_elevation, water=meanWV, order=1)
    
    ground_km = normalize_ground_km(mean_elevation)
    water_gcm2 = normalize_wv_gcm2(meanWV)
    
    simRads_array, simWave_array = generate_library(
        concentrations, lut_file, zenith=SZA, sensor=120,
        ground=ground_km, water=water_gcm2, order=1
    )

    print("Simulated radiance spectrum correctly generated from LUT for specified concentrations.")
    
    ###############################################################
    # single-column target spectrum and matched filter computation
    
    # # Generate target spectra for each pixel column (assuming no spectral smile)
    # target_i = generate_template_from_bands(
    #     mean_cw_subselection, mean_fwhm_subselection, simRads_array, simWave_array, concentrations)
    # target_spectra = target_i[:, 1]  # Extract the spectrum
    
    # # Optional: save to disk target spectra
    # np.save(target_spectra_export_name, target_spectra)
    
    # # Matched Filter application
    # concentration_map = calculate_matched_filter(
    #     rads_array_subselection, 
    #     classified_image, 
    #     mean_radiance, 
    #     covariance_matrices, 
    #     target_spectra, 
    #     k
    # )
    
    
    ###############################################################
    # column-wise target spectrum and matched filter computation
    
    # Generate target spectra for each spatial location
    target = None
    for i in range(np.size(cw_subselection, 0)):
        target_i = generate_template_from_bands(cw_subselection[i, :], fwhm_subselection[i, :], simRads_array, simWave_array, concentrations)
        if i == 0:
            target = target_i
        else:
            column_to_add = target_i[:, 1].reshape(-1, 1)
            target = np.concatenate((target, column_to_add), axis=1)
    
    target_spectra = target[:, 1:]
    # Optional: save to disk target spectra
    np.save(target_spectra_export_name, target_spectra)
    
    #Subselect target spectra based on wavelength
    #target_spectra_subselection = target_spectra[band_indices, :]
    
    # Matched Filter application
    #☺matched_filter_scores, concentration_map = calculate_matched_filter(rads_array_subselection, classified_image, mean_radiance, covariance_matrices, target_spectra, k)
    #plot_matched_filter_scores(matched_filter_scores)
    concentration_map = calculate_matched_filter_columnwise(
        rads_array_subselection,
        classified_image,
        mean_radiance,
        covariance_matrices,
        target_spectra,
        k
    )
    
    
    
    
    
    # Save results as GeoTIFF files
    #save_as_geotiff_single_band(matched_filter_scores, mf_output_file, latitude_vnir, longitude_vnir)
    save_as_geotiff_single_band(concentration_map , concentration_output_file, latitude_vnir, longitude_vnir)
    save_as_geotiff_multichannel(rgb_image, rgb_output_file, latitude_vnir, longitude_vnir)
    save_as_geotiff_multichannel(rads_array, rads_output_file, latitude_vnir, longitude_vnir)
    save_as_geotiff_single_band(classified_image, classified_output_file, latitude_vnir, longitude_vnir)
    
    print("Output files correctly generated")
    
    # Generate the processing report
    generate_report(
        output_dir=output_dir,
        L1_file=L1_file,
        L2C_file=L2C_file,
        dem_file=dem_file,
        lut_file=lut_file,
        meanWV=meanWV,
        SZA=SZA,
        mean_elevation=mean_elevation,
        k=k,
        mf_output_file=mf_output_file,
        concentration_output_file=concentration_output_file,
        rgb_output_file=rgb_output_file,
        classified_output_file=classified_output_file
    )




def extract_he5_from_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.he5'):
                zip_ref.extract(file, extract_to)
                return os.path.join(extract_to, file)
    return None

def get_date_from_filename(filename):
    match = re.search(r'(\d{8}\d{6})', filename)
    return match.group(1) if match else None

import traceback
from datetime import datetime

def process_directory(root_dir, dem_file, lut_file, min_wavelength, max_wavelength, k, output_root_dir):
    """
    Process all directories and subdirectories starting from `root_dir`,
    saving outputs in a user-specified output root directory (`output_root_dir`).

    Parameters:
        root_dir (str): Root directory containing input data.
        dem_file (str): Path to the DEM file.
        lut_file (str): Path to the LUT file.
        min_wavelength (float): Minimum wavelength for spectral selection.
        max_wavelength (float): Maximum wavelength for spectral selection.
        k (int): Number of clusters for k-means.
        output_root_dir (str): Root directory for saving outputs.
    """
    # Prepare a list to keep track of the processing results
    processing_log = []

    # We'll store the start time for naming the global report file
    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    for root, dirs, files in os.walk(root_dir):
        L1_zip = None
        L2C_zip = None
        L1_file = None
        L2C_file = None

        # Determine relative path from root_dir
        relative_path = os.path.relpath(root, root_dir)

        # Define the corresponding output directory
        output_dir = os.path.join(output_root_dir, relative_path + "_output")

        # Check if output directory already exists and is not empty
        if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Output directory {output_dir} already exists and is not empty. Skipping reprocessing of {root}.")
            processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                   "Skipped", f"Output directory {output_dir} not empty"))
            continue

        # Identify files
        for file in files:
            if file.startswith('PRS_L1_STD_OFFL') and file.endswith('.zip'):
                L1_zip = os.path.join(root, file)
            elif file.startswith('PRS_L2C_STD') and file.endswith('.zip'):
                L2C_zip = os.path.join(root, file)
            elif file.startswith('PRS_L1_STD_OFFL') and file.endswith('.he5'):
                L1_file = os.path.join(root, file)
            elif file.startswith('PRS_L2C_STD') and file.endswith('.he5'):
                L2C_file = os.path.join(root, file)

        extracted_from_zip = False
        status = "Unknown"
        details = ""

        # Try direct he5 files first
        if L1_file and L2C_file:
            L1_date = get_date_from_filename(L1_file)
            L2C_date = get_date_from_filename(L2C_file)

            if L1_date == L2C_date:
                os.makedirs(output_dir, exist_ok=True)
                try:
                    ch4_detection(L1_file, L2C_file, dem_file, lut_file, output_dir, min_wavelength, max_wavelength, k)
                    status = "Success"
                    details = "Processed successfully"
                except Exception as e:
                    print(f"Error processing {root}: {e}")
                    traceback.print_exc()
                    status = "Failed"
                    details = f"Error encountered: {str(e)}"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

            else:
                print(f"Date mismatch: {L1_file} and {L2C_file}")
                status = "Failed"
                details = f"Date mismatch between L1 and L2C files: {L1_file}, {L2C_file}"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

        # If direct he5 files not found, try extracting from zip
        elif L1_zip and L2C_zip:
            L1_date = get_date_from_filename(L1_zip)
            L2C_date = get_date_from_filename(L2C_zip)

            if L1_date == L2C_date:
                try:
                    L1_file = extract_he5_from_zip(L1_zip, root)
                    L2C_file = extract_he5_from_zip(L2C_zip, root)
                except Exception as e:
                    print(f"Error extracting from zip in {root}: {e}")
                    traceback.print_exc()
                    status = "Failed"
                    details = f"Extraction failed: {str(e)}"
                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
                    continue

                if L1_file and L2C_file:
                    extracted_from_zip = True
                    os.makedirs(output_dir, exist_ok=True)
                    try:
                        ch4_detection(L1_file, L2C_file, dem_file, lut_file, output_dir, min_wavelength, max_wavelength, k)
                        status = "Success"
                        details = "Processed successfully from extracted zip files"
                    except Exception as e:
                        print(f"Error processing {root}: {e}")
                        traceback.print_exc()
                        status = "Failed"
                        details = f"Error encountered: {str(e)}"

                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

                    # Remove extracted .he5 files
                    try:
                        os.remove(L1_file)
                        os.remove(L2C_file)
                        print(f"Removed extracted files: {L1_file}, {L2C_file}")
                    except Exception as e:
                        print(f"Error removing extracted files: {e}")
                        details += f" | Warning: Could not remove extracted files: {str(e)}"
                else:
                    print(f"Failed to extract .he5 files from {L1_zip} or {L2C_zip}")
                    status = "Failed"
                    details = "Extraction of .he5 files from zip failed or no he5 files found in the zip"
                    processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))
            else:
                print(f"Date mismatch: {L1_zip} and {L2C_zip}")
                status = "Failed"
                details = "Date mismatch between L1 and L2C zip files"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

        else:
            # Missing required files
            if files:
                print(f"Missing required files in {root}")
                status = "Failed"
                details = "Missing required files (L1 and/or L2C in either .he5 or .zip format)"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

    # After processing all directories, write a global report in the output root directory
    report_filename = f"directory_process_report_{start_time_str}.txt"
    report_filepath = os.path.join(output_root_dir, report_filename)
    with open(report_filepath, 'w') as report_file:
        report_file.write("Directory Processing Report\n")
        report_file.write("-----------------------------------\n")
        for entry in processing_log:
            folder_processed, time_processed, status, details = entry
            report_file.write(f"Folder: {folder_processed}\n")
            report_file.write(f"Time Processed: {time_processed}\n")
            report_file.write(f"Status: {status}\n")
            if details:
                report_file.write(f"Details: {details}\n")
            report_file.write("-----------------------------------\n")

    print(f"Global process report saved at: {report_filepath}")




if __name__ == "__main__":
    
    root_directory = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\PRISMA_calibration_data\to_process"
    dem_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\Matched_filter_approach\codici_PRISMA\CTMF\DEM_1Km\srtm30plus_v11_land.nc"
    lut_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\Matched_filter_approach\codici_PRISMA\CTMF\LUTs\dataset_ch4_full.hdf5"
    k = 1
    min_wavelength=2100
    max_wavelength=2450
    output_root_dir =r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\PRISMA_calibration_data\to_process"
    
    process_directory(
    root_directory,
    dem_file,
    lut_file,
    min_wavelength,
    max_wavelength,
    k,
    output_root_dir
    )
