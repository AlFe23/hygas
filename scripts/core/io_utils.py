"""
General-purpose helpers shared across pipelines (report generation, logging utils).
"""

import os
from datetime import datetime


def generate_prisma_report(
    output_dir,
    L1_file,
    L2C_file,
    dem_file,
    lut_file,
    meanWV,
    SZA,
    mean_elevation,
    k,
    mf_output_file,
    concentration_output_file,
    rgb_output_file,
    classified_output_file,
    min_wavelength,
    max_wavelength,
):
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
    - Spectral Window: {min_wavelength} - {max_wavelength} nm
    """

    # Scrivi il report nel file
    with open(report_path, "w") as file:
        file.write(report_content)

    print(f"Report di elaborazione PRISMA generato in: {report_path}")


def generate_enmap_report(
    output_dir,
    vnir_file,
    swir_file,
    metadata_file,
    lut_file,
    mean_wv,
    SZA,
    mean_elevation_km,
    k,
    min_wavelength,
    max_wavelength,
    concentration_output_file,
    rgb_output_file,
    classified_output_file,
    target_spectra_file,
):
    """Create an EnMAP-specific processing report mirroring the PRISMA flow."""

    processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "processing_report.txt")

    report_content = f"""
    Processing Report (EnMAP)
    -------------------------
    Date and Time of Processing: {processing_date}

    Input Files:
    - VNIR File: {vnir_file}
    - SWIR File: {swir_file}
    - Metadata File: {metadata_file}
    - LUT File: {lut_file}

    Output Files:
    - Matched Filter Output: {concentration_output_file}
    - RGB Image Output: {rgb_output_file}
    - Classified Image Output: {classified_output_file}
    - Target Spectrum (NPY): {target_spectra_file}

    Processing Parameters:
    - Mean Water Vapor: {mean_wv} g/cm^2
    - Solar Zenith Angle: {SZA} degrees
    - Mean Elevation: {mean_elevation_km} km
    - Number of Clusters (k-means): {k}
    - Spectral Window: {min_wavelength} - {max_wavelength} nm
    """

    with open(report_path, "w") as file:
        file.write(report_content)

    print(f"Report di elaborazione EnMAP generato in: {report_path}")
