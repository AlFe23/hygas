# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:27:40 2025

@author: ferra
"""
import numpy as np
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    
    #Convert PRISMA radiance from [W/(str*um*m^2)] to [μW*cm-2*nm-1*sr-1] in order to meet original AVIRIS radiance unit.
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



def prisma_noise_analysis(filename, n_components=5, homogeneous_subset=None):
    """
    Process PRISMA data with noise estimation using PCA decomposition
    
    Parameters:
    - filename: PRISMA HDF5 file path
    - n_components: Number of PCA components for denoising
    - homogeneous_subset: Tuple of slices (x_slice, y_slice) for homogeneous area selection
    
    Returns:
    - Dictionary containing all PRISMA data and noise analysis results
    """
    # Load original data using your existing function
    results = {}
    (results['concatenated_cube'], results['concatenated_cw'],
     results['concatenated_fwhm'], results['rgb_image'],
     results['vnir_cube_bip'], results['swir_cube_bip'],
     results['latitude_vnir'], results['longitude_vnir'],
     results['latitude_swir'], results['longitude_swir']) = prisma_read(filename)

    # Select homogeneous area if specified
    if homogeneous_subset:
        x_slice, y_slice = homogeneous_subset
        analysis_cube = results['concatenated_cube'][x_slice, y_slice, :]
    else:
        analysis_cube = results['concatenated_cube']

    # Reshape for PCA (pixels x bands)
    original_shape = analysis_cube.shape
    data_2d = analysis_cube.reshape(-1, original_shape[2])

    # Perform PCA decomposition
    pca = PCA(n_components=n_components)
    reconstructed_2d = pca.inverse_transform(pca.fit_transform(data_2d))
    
    # Calculate noise spectra
    noise_2d = data_2d - reconstructed_2d
    noise_cube = noise_2d.reshape(original_shape)
    
    # Store results
    results['noise_cube'] = noise_cube
    results['reconstructed_cube'] = reconstructed_2d.reshape(original_shape)
    results['pca_model'] = pca
    results['explained_variance'] = pca.explained_variance_ratio_
    
    return results

def plot_highres_pca(results, pixel_loc=(50, 50), band=100, figsize=(24, 16)):
    """
    High-resolution visualization of PCA components and RGB comparison
    
    Parameters:
    - results: Dictionary from prisma_noise_analysis()
    - pixel_loc: Tuple (x,y) for spectral profile
    - band: Band index for spatial noise
    - figsize: Figure size in inches
    """
    plt.figure(figsize=figsize, dpi=300)
    
    # 1. Original RGB
    plt.subplot(3, 4, 1)
    plt.imshow(results['rgb_image'])
    plt.title('Original RGB Image', fontsize=10)
    plt.axis('off')
    
    # 2. Reconstructed RGB
    rec_rgb = generate_rgb(results['reconstructed_cube'])
    plt.subplot(3, 4, 2)
    plt.imshow(rec_rgb)
    plt.title('PCA Reconstructed RGB', fontsize=10)
    plt.axis('off')
    
    # 3. Noise RGB (absolute value)
    noise_rgb = generate_noise_rgb(results['noise_cube'])
    plt.subplot(3, 4, 3)
    plt.imshow(noise_rgb, cmap='gray', vmin=-3*np.std(noise_rgb), 
               vmax=3*np.std(noise_rgb))
    plt.title('Noise RGB (Absolute Values)', fontsize=10)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 4. Spatial noise pattern at band (grayscale)
    plt.subplot(3, 4, 4)
    noise_band = results['noise_cube'][:, :, band]
    plt.imshow(noise_band, cmap='gray', 
               vmin=-3*np.std(noise_band), 
               vmax=3*np.std(noise_band))
    plt.title(f'Band {band} Noise Pattern', fontsize=10)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 5. Principal Components (grayscale)
    pc_maps = results['pca_model'].transform(
        results['concatenated_cube'].reshape(-1, results['concatenated_cube'].shape[2])
    ).reshape(results['concatenated_cube'].shape[0], 
             results['concatenated_cube'].shape[1], 
             results['pca_model'].n_components)
    
    for i in range(4):
        plt.subplot(3, 4, 5+i)
        if i < pc_maps.shape[2]:
            pc_map = pc_maps[:, :, i]
            pc_map = (pc_map - pc_map.min()) / (pc_map.max() - pc_map.min())
            plt.imshow(pc_map, cmap='gray')
            plt.title(f'PC {i+1} Spatial Pattern', fontsize=10)
            plt.axis('off')
    
    # 9. Spectral profiles
    plt.subplot(3, 4, 9)
    x, y = pixel_loc
    plt.plot(results['concatenated_cube'][x, y], 'b', label='Original')
    plt.plot(results['reconstructed_cube'][x, y], 'r--', label='Reconstructed')
    plt.xlabel('Band Number', fontsize=9)
    plt.ylabel('Radiance', fontsize=9)
    plt.title('Spectral Profile Comparison', fontsize=10)
    plt.legend(fontsize=8)
    
    # 10. Noise spectrum
    plt.subplot(3, 4, 10)
    plt.plot(results['noise_cube'][x, y])
    plt.xlabel('Band Number', fontsize=9)
    plt.ylabel('Noise Value', fontsize=9)
    plt.title('Noise Spectrum', fontsize=10)
    
    # 11. Explained variance
    plt.subplot(3, 4, 11)
    plt.plot(np.cumsum(results['pca_model'].explained_variance_ratio_), 'g-')
    plt.xlabel('Components', fontsize=9)
    plt.ylabel('Explained Variance', fontsize=9)
    plt.title('Cumulative Explained Variance', fontsize=10)
    plt.grid(True)
    
    # 12. Component spectra
    plt.subplot(3, 4, 12)
    components = results['pca_model'].components_
    for i in range(min(4, components.shape[0])):
        plt.plot(components[i], label=f'PC {i+1}')
    plt.xlabel('Band Number', fontsize=9)
    plt.ylabel('Weight', fontsize=9)
    plt.title('Principal Component Spectra', fontsize=10)
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

def generate_rgb(cube):
    """Generate RGB from hyperspectral cube"""
    red = cube[:, :, 29]  # Same bands as original RGB
    green = cube[:, :, 19]
    blue = cube[:, :, 7]
    
    red_norm = (red - red.min()) / (red.max() - red.min())
    green_norm = (green - green.min()) / (green.max() - green.min())
    blue_norm = (blue - blue.min()) / (blue.max() - blue.min())
    
    return np.stack([red_norm, green_norm, blue_norm], axis=-1)

def generate_noise_rgb(noise_cube):
    """Generate noise RGB visualization"""
    red_noise = noise_cube[:, :, 29]
    green_noise = noise_cube[:, :, 19]
    blue_noise = noise_cube[:, :, 7]
    
    # Normalize to 3 sigma range
    max_val = max(np.abs(red_noise).max(), 
                 np.abs(green_noise).max(), 
                 np.abs(blue_noise).max())
    
    red_norm = np.clip(red_noise / max_val, -1, 1)
    green_norm = np.clip(green_noise / max_val, -1, 1)
    blue_norm = np.clip(blue_noise / max_val, -1, 1)
    
    return np.stack([red_norm, green_norm, blue_norm], axis=-1)

# Usage example:
if __name__ == "__main__":
    results = prisma_noise_analysis(r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\data\PRS_L1_STD_OFFL_20240227071508_20240227071512_0001.he5", 
                                  n_components=5,
                                  homogeneous_subset=(slice(450,550), slice(450,550)))
    
    plot_highres_pca(results, pixel_loc=(50, 50), band=150, figsize=(24, 16))