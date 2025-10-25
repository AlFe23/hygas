# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 13:49:20 2025

Modifiche per risolvere problema di apertura dell'output MF tiff'

Updates (2025-10-03)
- Improved GeoTIFF writing to ensure QGIS compatibility:
  • Added gdal.UseExceptions(), tiled + LZW compression, BIGTIFF=IF_SAFER
  • Explicitly closed band and dataset handles to avoid file locks
  • Enforced Float32 + NoData (single-band) and UInt8 (RGB) with shape checks
- Introduced metadata-driven basenames for outputs:
  • Format: <level>_<datatake>_<tile>_<start>_<stop> (start/stop from METADATA.XML)
  • Ensures shorter, standardized filenames while preserving acquisition info
- Output products renamed consistently: _MF.tif, _RGB.tif, _CL.tif, _CH4_target.npy
- Changes solve previous issues with long UNC paths, renaming restrictions,
  and QGIS “unsupported TIFF” errors.


@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:48:24 2024

@author: ferra
"""



# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:45:13 2024

@author: Alvise Ferrari

Prototipo di codice CH4 Detection per immagini EnMAP.
Ottenuto dall'adattamento del codice ctmf_v4_7_automatic.py scritto per le immagini PRISMA'



"""



import os
import numpy as np
import xml.etree.ElementTree as ET
from osgeo import gdal, osr
from numpy.linalg import inv
import h5py
import scipy.ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

gdal.UseExceptions()



def enmap_read(vnir_file, swir_file, metadata_file):
    """
    Reads EnMAP VNIR and SWIR data cubes and associated metadata.

    Parameters:
    - vnir_file: Path to the VNIR GeoTIFF file.
    - swir_file: Path to the SWIR GeoTIFF file.
    - metadata_file: Path to the XML metadata file.

    Returns:
    - concatenated_cube: Combined VNIR and SWIR radiance data cube in BIP format.
    - concatenated_cw: Central wavelengths for each band.
    - concatenated_fwhm: FWHM for each band.
    - rgb_image: RGB image created from the radiance data.
    - latitude: Latitude array (if available).
    - longitude: Longitude array (if available).
    """
    # Read VNIR data cube
    vnir_dataset = gdal.Open(vnir_file)
    vnir_cube_DN = vnir_dataset.ReadAsArray()  # Shape: (bands, rows, cols)
    vnir_cube_DN = np.transpose(vnir_cube_DN, (1, 2, 0))  # Shape: (rows, cols, bands)

    # Read SWIR data cube
    swir_dataset = gdal.Open(swir_file)
    swir_cube_DN = swir_dataset.ReadAsArray()
    swir_cube_DN = np.transpose(swir_cube_DN, (1, 2, 0))

    # Read metadata from XML file
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    # Extract Gain, Offset, CW, and FWHM for each band
    band_info = []
    # Adjust the XPath according to the XML structure
    for band in root.findall(".//specific/bandCharacterisation/bandID"):
        band_number = int(band.get('number'))
        wavelength_center = float(band.find('wavelengthCenterOfBand').text)
        fwhm = float(band.find('FWHMOfBand').text)
        gain = float(band.find('GainOfBand').text)
        offset = float(band.find('OffsetOfBand').text)
        band_info.append({
            'number': band_number,
            'wavelength_center': wavelength_center,
            'fwhm': fwhm,
            'gain': gain,
            'offset': offset
        })

    # Sort band_info by band number
    band_info.sort(key=lambda x: x['number'])

    # Ensure the number of bands matches the data cubes
    num_vnir_bands = vnir_cube_DN.shape[2]
    num_swir_bands = swir_cube_DN.shape[2]
    total_bands = num_vnir_bands + num_swir_bands

    if len(band_info) != total_bands:
        print(f"Warning: Number of bands in metadata ({len(band_info)}) does not match total bands in data cubes ({total_bands})")

    vnir_band_info = band_info[:num_vnir_bands]
    swir_band_info = band_info[num_vnir_bands:]

    # Apply radiance conversion: Radiance = Gain * DN + Offset
    # Ensure DN data is in float32 to prevent overflow/underflow
    vnir_cube_DN = vnir_cube_DN.astype(np.float32)
    swir_cube_DN = swir_cube_DN.astype(np.float32)

    # VNIR
    vnir_radiance = np.zeros_like(vnir_cube_DN, dtype=np.float32)
    for i, band in enumerate(vnir_band_info):
        gain = band['gain']
        offset = band['offset']
        vnir_radiance[:, :, i] = vnir_cube_DN[:, :, i] * gain + offset

    # SWIR
    swir_radiance = np.zeros_like(swir_cube_DN, dtype=np.float32)
    for i, band in enumerate(swir_band_info):
        gain = band['gain']
        offset = band['offset']
        swir_radiance[:, :, i] = swir_cube_DN[:, :, i] * gain + offset

    # Convert radiance units from [W/(sr*nm*m^2)] to [μW/(sr*nm*cm^2)]
    vnir_radiance *= 1e2
    swir_radiance *= 1e2

    # Concatenate VNIR and SWIR radiance cubes
    concatenated_cube = np.concatenate((vnir_radiance, swir_radiance), axis=2)

    # Collect CW and FWHM
    concatenated_cw = np.array([band['wavelength_center'] for band in band_info])
    concatenated_fwhm = np.array([band['fwhm'] for band in band_info])

    # Create an RGB image for visualization
    # Adjust band indices to match EnMAP's spectral bands
    # Let's select bands close to 650 nm (Red), 550 nm (Green), and 450 nm (Blue)
    red_wavelength = 650
    green_wavelength = 550
    blue_wavelength = 450

    # Find indices of bands closest to these wavelengths
    red_band_idx = np.argmin(np.abs(concatenated_cw - red_wavelength))
    green_band_idx = np.argmin(np.abs(concatenated_cw - green_wavelength))
    blue_band_idx = np.argmin(np.abs(concatenated_cw - blue_wavelength))

    red_band = concatenated_cube[:, :, red_band_idx]
    green_band = concatenated_cube[:, :, green_band_idx]
    blue_band = concatenated_cube[:, :, blue_band_idx]

    # Normalize bands
    red_norm = (red_band - red_band.min()) / (red_band.max() - red_band.min())
    green_norm = (green_band - green_band.min()) / (green_band.max() - green_band.min())
    blue_norm = (blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())

    rgb_image = np.stack((red_norm, green_norm, blue_norm), axis=-1)

    # Latitude and Longitude arrays can be extracted if available (EnMAP provides geolocation)
    # For now, we set them to None
    latitude = None
    longitude = None

    return concatenated_cube, concatenated_cw, concatenated_fwhm, rgb_image, latitude, longitude

def enmap_metadata_read(metadata_file):
    """
    Reads SZA and mean WV from EnMAP metadata XML file.

    Parameters:
    - metadata_file: Path to the XML metadata file.

    Returns:
    - SZA: Solar Zenith Angle in degrees.
    - meanWV: Mean Water Vapor in g/cm^2.
    """
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    # Extract SZA
    sza_elem = root.find(".//specific/qualityFlag/sceneSZA")
    if sza_elem is not None and sza_elem.text is not None:
        SZA = float(sza_elem.text)
    else:
        raise ValueError("Solar Zenith Angle (sceneSZA) not found in metadata.")

    # Extract mean WV
    wv_elem = root.find(".//specific/qualityFlag/sceneWV")
    if wv_elem is not None and wv_elem.text is not None:
        scene_wv = float(wv_elem.text)
        meanWV = scene_wv / 1000  # Convert from [cm * 1000] to [cm]
    else:
        raise ValueError("Mean Water Vapor (sceneWV) not found in metadata.")


    # Alternatively, extract meanGroundElevation
    mean_ground_elevation_elem = root.find(".//specific/meanGroundElevation")
    if mean_ground_elevation_elem is not None and mean_ground_elevation_elem.text is not None:
        mean_ground_elevation = float(mean_ground_elevation_elem.text)
    else:
        print("Warning: meanGroundElevation not found in metadata.")
        mean_ground_elevation = None  # Handle appropriately

    print(f"Sun Zenith Angle (degrees): {SZA}")
    print(f"Mean Water Vapor (g/cm^2): {meanWV}")
    print(f"Mean Ground Elevation (m): {mean_ground_elevation}")

    return SZA, meanWV, mean_ground_elevation

def k_means_hyperspectral(rads_array, k):
    """
    Performs k-means clustering on the hyperspectral data.
    """
    n_rows, n_columns, n_bands = rads_array.shape
    reshaped_data = rads_array.reshape(-1, n_bands)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(reshaped_data)
    classified_image = kmeans.labels_.reshape(n_rows, n_columns)
    return classified_image

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

def calculate_statistics(rads_array, classified_image, k):
    """
    Calculates mean radiance and covariance matrices for each cluster.

    Calcola la radianza media e la matrice di covarianza per ogni cluster.

    Parametri:
    image (numpy array): Immagine iperspettrale originale con dimensioni (n_righe, n_colonne, n_bande).
    classified_image (numpy array): Immagine classificata con dimensioni (n_righe, n_colonne).
    k (int): Numero di cluster.

    Ritorna:
    mean_radiance (list of numpy arrays): Radianza media per ogni cluster.
    covariance_matrices (list of numpy arrays): Matrici di covarianza per ogni cluster.

    """
    n_rows, n_columns, n_bands = rads_array.shape
    mean_radiance = []
    covariance_matrices = []

    for cls in range(k):
        class_mask = (classified_image == cls)
        class_pixels = rads_array[class_mask]
        if class_pixels.size == 0:
            # Handle empty clusters
            mean_radiance.append(np.zeros(n_bands))
            covariance_matrices.append(np.zeros((n_bands, n_bands)))
            continue
        mean_radiance.append(np.mean(class_pixels, axis=0))
        covariance_matrices.append(np.cov(class_pixels, rowvar=False))
    mean_radiance = np.array(mean_radiance)
    covariance_matrices = np.array(covariance_matrices)
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

###################################################################################
#FUNZIONI PER ESTRARRE E DEFINIRE LE VARIABILI NECESSARIE ALL'ESTRAZIONE DELLE FIRME SPETTRALI DALLA LUT

def check_param(value, min, max, name):
    """
    Ensures that a parameter is within the specified range. If it exceeds the range, it is clamped.
    """
    if value < min:
        print(f"Warning: {name} value ({value}) is below the minimum ({min}). Setting to {min}.")
        return min
    elif value > max:
        print(f"Warning: {name} value ({value}) exceeds the maximum ({max}). Setting to {max}.")
        return max
    return value



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

@np.vectorize
def get_5deg_ground_altitude_index(ground_value):  # [0, 0.5, 1.0, 2.0, 3.0]
    check_param(ground_value, 0, 3, 'Ground Altitude')
    if ground_value < 1:
        return 2 * ground_value
    else:
        return 1 + ground_value

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
        zenith=zenith, sensor=sensor, ground=ground, water=water, conc=conc, gas=gas)
    # correct_lookup = np.asarray([scipy.ndimage.map_coordinates(
    #     im, coordinates=coords, order=order, mode='nearest') for im in np.moveaxis(grid_data, 5, 0)])
    if order == 1:
        coords_fractional_part, coords_whole_part = np.modf(coords)
        #coords_near_slice = tuple((slice(int(c), int(c+2)) for c in coords_whole_part))  #This line gives a warning, so it as been modified as below
        coords_near_slice = tuple((slice(int(c[0]), int(c[0]+2)) if isinstance(c, np.ndarray) else slice(int(c), int(c+2)) for c in coords_whole_part))
        near_grid_data = grid_data[coords_near_slice]
        new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
                                    np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
        lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    elif order == 3:
        lookup = np.asarray([scipy.ndimage.map_coordinates(
            im, coordinates=coords_fractional_part, order=order, mode='nearest') for im in np.moveaxis(near_grid_data, 5, 0)])
    return lookup.squeeze()


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




def save_as_geotiff_single_band_enmap(data, output_file, reference_dataset):
    """
    Saves a single-band array as a GeoTIFF file with EnMAP georeferencing.
    Robust version: closes all handles, uses compression, BigTIFF if safer,
    and guarantees Float32 + NoData.
    """
    # Ensure 2D float32 and replace non-finite
    arr = np.asarray(data, dtype=np.float32)
    nodata_value = np.float32(-9999.0)
    np.copyto(arr, nodata_value, where=~np.isfinite(arr))

    # Dimensions from reference
    xsize = reference_dataset.RasterXSize
    ysize = reference_dataset.RasterYSize
    if arr.shape != (ysize, xsize):
        raise ValueError(f"Array shape {arr.shape} does not match reference raster size {(ysize, xsize)}")

    # Creation options: tiled, compressed, BigTIFF if needed
    driver = gdal.GetDriverByName('GTiff')
    opts = [
        'TILED=YES',
        'COMPRESS=LZW',
        'BIGTIFF=IF_SAFER',      # avoids >4GB classic-TIFF issues
        'BLOCKXSIZE=256',
        'BLOCKYSIZE=256'
    ]
    ds = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_Float32, options=opts)
    if ds is None:
        raise RuntimeError(f"Could not create {output_file}")

    # GeoTransform / Projection
    ds.SetGeoTransform(reference_dataset.GetGeoTransform())
    ds.SetProjection(reference_dataset.GetProjection())

    # Write band + NoData
    band1 = ds.GetRasterBand(1)
    band1.WriteArray(arr)
    band1.SetNoDataValue(float(nodata_value))
    band1.FlushCache()
    band1 = None   # CRITICAL: release band handle

    # Flush and close
    ds.FlushCache()
    ds = None       # CRITICAL: release dataset handle



def save_as_geotiff_rgb_enmap(rgb_data, output_file, reference_dataset):
    """
    Saves an RGB array as a GeoTIFF with EnMAP georeferencing.
    """
    # Scale and convert to uint8
    rgb = np.clip(rgb_data * 255.0, 0, 255).astype(np.uint8)

    ysize, xsize, bands = rgb.shape
    if bands != 3:
        raise ValueError("RGB array must have 3 bands")

    driver = gdal.GetDriverByName('GTiff')
    opts = [
        'TILED=YES',
        'COMPRESS=LZW',
        'BIGTIFF=IF_SAFER',
        'BLOCKXSIZE=256',
        'BLOCKYSIZE=256'
    ]
    ds = driver.Create(output_file, xsize, ysize, 3, gdal.GDT_Byte, options=opts)
    if ds is None:
        raise RuntimeError(f"Could not create {output_file}")

    ds.SetGeoTransform(reference_dataset.GetGeoTransform())
    ds.SetProjection(reference_dataset.GetProjection())

    # Write bands and set color interpretation
    for i, interp in enumerate([gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand], start=1):
        b = ds.GetRasterBand(i)
        b.WriteArray(rgb[:, :, i-1])
        b.SetColorInterpretation(interp)
        b.FlushCache()
        b = None  # CRITICAL

    ds.FlushCache()
    ds = None    # CRITICAL



def ch4_detection_enmap(vnir_file, swir_file, metadata_file, lut_file, output_dir, k=10):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read SZA, meanWV, and mean_ground_elevation from EnMAP metadata
    SZA, meanWV, mean_ground_elevation = enmap_metadata_read(metadata_file)

    # Use the extracted mean ground elevation
    if mean_ground_elevation is None:
        # Handle the case where mean ground elevation is not provided
        print("Mean ground elevation not found in metadata. Using default value of 0 m.")
        mean_elevation = 0  # You can set a default value or handle this case differently
    else:
        mean_elevation = mean_ground_elevation  # Elevation in meters

    # Convert mean elevation to kilometers if needed
    mean_elevation_km = mean_elevation / 1000.0  # Convert from meters to kilometers

    # Extract the file name without the extension for output files
    _, full_filename = os.path.split(vnir_file)
    filename_without_extension = os.path.splitext(full_filename)[0]

    # # Define output filenames in the specified output directory
    # target_spectra_export_name = os.path.join(output_dir, f"{filename_without_extension}_CH4_target_EnMAP_conv.npy")
    # concentration_output_file = os.path.join(output_dir, f"{filename_without_extension}_MF_concentration.tif")
    # rgb_output_file = os.path.join(output_dir, f"{filename_without_extension}_rgb.tif")
    # classified_output_file = os.path.join(output_dir, f"{filename_without_extension}_classified.tif")
    
    # NEW (metadata-driven):
    output_basename = derive_basename_from_metadata(metadata_file)
    
    target_spectra_export_name = os.path.join(output_dir, f"{output_basename}_CH4_target.npy")
    concentration_output_file   = os.path.join(output_dir, f"{output_basename}_MF.tif")
    rgb_output_file             = os.path.join(output_dir, f"{output_basename}_RGB.tif")
    classified_output_file      = os.path.join(output_dir, f"{output_basename}_CL.tif")
       
    # Data extraction and processing
    rads_array, cw_array, fwhm_array, rgb_image, latitude, longitude = enmap_read(vnir_file, swir_file, metadata_file)

    # Subselect the bands between gas-specific absorption window (adjust indices based on EnMAP bands)
    
    band_indices = np.where((cw_array >= 2100) & (cw_array <= 2450))[0]
    rads_array_subselection = rads_array[:, :, band_indices]
    cw_subselection = cw_array[band_indices]
    fwhm_subselection = fwhm_array[band_indices]

    # k-means classification using the same bands as the matched filter
    classified_image = k_means_hyperspectral(rads_array_subselection, k)
    print("k-means classification completed")

    # Calculate statistics
    mean_radiance, covariance_matrices = calculate_statistics(rads_array_subselection, classified_image, k)

    # Target spectrum calculation and matched filter application
    concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    simRads_array, simWave_array = generate_library(
        concentrations, lut_file, zenith=SZA, sensor=120, ground=mean_elevation_km, water=meanWV, order=1)
    print("Simulated radiance spectrum generated from LUT for CH4 column enhancements.")

    # Generate target spectra for each pixel column (assuming no spectral smile)
    target_i = generate_template_from_bands(
        cw_subselection, fwhm_subselection, simRads_array, simWave_array, concentrations)
    target_spectra = target_i[:, 1]  # Extract the spectrum

    # Optional: save to disk target spectra
    np.save(target_spectra_export_name, target_spectra)

    # Matched Filter application
    concentration_map = calculate_matched_filter(
        rads_array_subselection, classified_image, mean_radiance, covariance_matrices, target_spectra, k)

    # Save results as GeoTIFF files
    reference_dataset = gdal.Open(swir_file)  # Use SWIR dataset as reference for georeferencing

    # Save results
    save_as_geotiff_single_band_enmap(concentration_map, concentration_output_file, reference_dataset)
    save_as_geotiff_rgb_enmap(rgb_image, rgb_output_file, reference_dataset)
    save_as_geotiff_single_band_enmap(classified_image, classified_output_file, reference_dataset)
    
    # Close the reference dataset to release file locks
    reference_dataset = None
    
    print("Output files successfully generated.")




# # vnir_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-SPECTRAL_IMAGE_VNIR.TIF"
# # swir_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-SPECTRAL_IMAGE_SWIR.TIF"
# # metadata_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-METADATA.XML"


# # vnir_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-SPECTRAL_IMAGE_VNIR.TIF"
# # swir_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-SPECTRAL_IMAGE_SWIR.TIF"
# # metadata_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\EnMAP\EnMAP_data\Buenos_Aires\20240112T144653\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z\ENMAP01-____L1B-DT0000058121_20240112T144653Z_002_V010401_20240312T140539Z-METADATA.XML"



# vnir_file = r"\\EOSIAL_NAS4\data2\Alvise\CH4\Comparazione_Acquisizioni\matching_acquisitions\EnMAP\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z-SPECTRAL_IMAGE_VNIR.TIF"
# swir_file = r"\\EOSIAL_NAS4\data2\Alvise\CH4\Comparazione_Acquisizioni\matching_acquisitions\EnMAP\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z-SPECTRAL_IMAGE_SWIR.TIF"
# metadata_file = r"\\EOSIAL_NAS4\data2\Alvise\CH4\Comparazione_Acquisizioni\matching_acquisitions\EnMAP\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z\ENMAP01-____L1B-DT0000090108_20240830T074507Z_001_V010502_20241207T112758Z-METADATA.XML"
# output_dir = os.path.join(os.path.dirname(vnir_file), 'MF_out')

# lut_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\Matched_filter_approach\UtahUni_LUTs\dataset_ch4_full.hdf5"


# # concatenated_cube, concatenated_cw, concatenated_fwhm, rgb_image, latitude, longitude = enmap_read(vnir_file, swir_file, metadata_file)
# # SZA, meanWV, mean_ground_elevation = enmap_metadata_read(metadata_file)


# ch4_detection_enmap(vnir_file, swir_file, metadata_file, lut_file, output_dir, k=1)


import os
import traceback
from datetime import datetime

def extract_enmap_files_from_folder(folder_path):
    """
    Identify the VNIR, SWIR, and METADATA files in the given folder.
    Returns (vnir_file, swir_file, metadata_file) or (None, None, None) if not found.
    """
    vnir_file = None
    swir_file = None
    metadata_file = None

    for file in os.listdir(folder_path):
        if file.endswith("SPECTRAL_IMAGE_VNIR.TIF"):
            vnir_file = os.path.join(folder_path, file)
        elif file.endswith("SPECTRAL_IMAGE_SWIR.TIF"):
            swir_file = os.path.join(folder_path, file)
        elif file.endswith("METADATA.XML"):
            metadata_file = os.path.join(folder_path, file)

    return vnir_file, swir_file, metadata_file


def derive_output_basename(vnir_file):
    """
    Derive a simplified output basename from the VNIR file name.
    For example:
    ENMAP01-____L1B-DT0000090108_20240112T144653Z_002_V010401_...-SPECTRAL_IMAGE_VNIR.TIF
    -> L1B_20240112T144653Z
    """
    base = os.path.basename(vnir_file)
    parts = base.split('_')
    # parts[0]: something like "ENMAP01-____L1B-DT0000090108"
    # parts[1]: date/time "20240112T144653Z"
    
    # Extract 'L1B' from parts[0]
    # It's always there after 'ENMAP01-____'
    # We'll just hardcode L1B since we know it's an L1B product
    product_level = "L1B"
    date_str = parts[1]  # The date/time string
    output_basename = f"{product_level}_{date_str}"
    return output_basename


def process_directory_enmap(root_dir, lut_file, k=1):
    # Prepare a list to keep track of the processing results
    # Each entry will be (folder_processed, time_processed, status, details)
    processing_log = []

    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    for root, dirs, files in os.walk(root_dir):
        # Skip the main directory itself if it has no VNIR/SWIR/metadata
        if root == root_dir:
            continue

        vnir_file, swir_file, metadata_file = extract_enmap_files_from_folder(root)

        # Determine output directory name at the same level as 'root'
        folder_name = os.path.basename(root)
        if folder_name == '':
            folder_name = os.path.basename(os.path.normpath(root))
        parent_dir = os.path.dirname(root)
        output_dir = os.path.join(parent_dir, folder_name + "_output")

        # Check if output directory already exists and not empty
        if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Output directory {output_dir} already exists and is not empty. Skipping reprocessing of {root}.")
            processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                   "Skipped", f"Output directory {output_dir} not empty"))
            continue

        if vnir_file and swir_file and metadata_file:
            # Derive output basename
            output_basename = derive_output_basename(vnir_file)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Attempt to run the ch4_detection_enmap process
            try:
                ch4_detection_enmap(vnir_file, swir_file, metadata_file, lut_file, output_dir, k=k)

                status = "Success"
                details = f"Processed successfully: {output_basename}"
            except Exception as e:
                print(f"Error processing {root}: {e}")
                traceback.print_exc()
                status = "Failed"
                details = f"Error encountered: {str(e)}"
            processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

        else:
            # Missing one or more required files
            if files:  # Only log if there's something in the folder
                print(f"Missing required files in {root}. VNIR: {vnir_file}, SWIR: {swir_file}, METADATA: {metadata_file}")
                status = "Failed"
                details = "Missing required VNIR, SWIR or METADATA file"
                processing_log.append((root, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, details))

    # After processing all directories, write a global report in the root directory
    report_filename = f"directory_process_report_{start_time_str}.txt"
    report_filepath = os.path.join(root_dir, report_filename)
    with open(report_filepath, 'w') as report_file:
        report_file.write("Directory Processing Report (EnMAP)\n")
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



import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# --- Utilities ---

def _to_yyyymmddThhmmssZ(dt_text: str) -> str:
    """
    Normalize an EnMAP time string like '2025-06-23T11:01:29.036499Z'
    to 'YYYYMMDDTHHMMSSZ' with seconds precision.
    """
    # Remove trailing Z for parsing, keep fractional seconds if present
    t = dt_text.strip()
    assert t.endswith('Z'), f"Expected Zulu time, got: {dt_text}"
    t_noz = t[:-1]
    # Try with microseconds then without
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(t_noz, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            dt = None
    if dt is None:
        raise ValueError(f"Unrecognized datetime format: {dt_text}")
    return dt.strftime("%Y%m%dT%H%M%SZ")

def _first_text(elem, path):
    e = elem.find(path)
    return e.text.strip() if (e is not None and e.text) else None

# --- Metadata-driven basename builder ---

def derive_basename_from_metadata(metadata_file: str) -> str:
    """
    Build a compact, informative basename using the METADATA.XML file:
      <level>_<datatake>_<tile>_<start>_<stop>

    Example: L1B_DT0000137438_010_20250623T110129Z_20250623T110133Z
    """
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    # Level (processing level)
    level = _first_text(root, ".//metadata/schema/processingLevel") or \
            _first_text(root, ".//base/level") or "L1B"

    # temporal coverage start/stop
    start_raw = _first_text(root, ".//base/temporalCoverage/startTime")
    stop_raw  = _first_text(root, ".//base/temporalCoverage/stopTime")
    if not start_raw or not stop_raw:
        raise ValueError("startTime/stopTime not found in METADATA.XML")

    start_z = _to_yyyymmddThhmmssZ(start_raw)
    stop_z  = _to_yyyymmddThhmmssZ(stop_raw)

    # Datatake + tile:
    # Robust way: parse from the product name inside <metadata><name>
    name_txt = _first_text(root, ".//metadata/name") or ""
    # Example name contains "...-DT0000137438_20250623T110129Z_010_..."
    m_dt = re.search(r"(DT\d{10,})", name_txt)  # DT + at least 10 digits
    m_tile = re.search(r"_(\d{3})_", name_txt)
    datatake = m_dt.group(1) if m_dt else "DTUNKNOWN"
    tile = m_tile.group(1) if m_tile else "000"

    # Assemble
    return f"{level}_{datatake}_{tile}_{start_z}_{stop_z}"




if __name__ == "__main__":
    main_directory = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\EnMAP_calibration_data\articolo_LARS\20240614T061335"
    lut_file = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\Matched_filter_approach\UtahUni_LUTs\dataset_ch4_full.hdf5"
    k = 1  # number of clusters for k-means

    process_directory_enmap(main_directory, lut_file, k)