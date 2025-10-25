# -*- coding: utf-8 -*-
"""

EnMAP L1B VNIR/SWIR Radiance and Spectral Smile Analysis
--------------------------------------------------------

This script reads and analyses EnMAP Level-1B products containing separate
VNIR and SWIR hyperspectral cubes together with the corresponding metadata XML.
It performs radiometric conversion and spectral smile characterization for
each instrument independently, following the definitions in the EnMAP L1B
Processor documentation (EN-PCV-TN-4006 and EN-PCV-ICD-2009-2).

Main features
--------------
1. **Independent VNIR/SWIR handling**
   - Reads VNIR and SWIR GeoTIFF cubes separately.
   - Extracts global band metadata (<bandCharacterisation>) and the respective
     <smileCorrection><VNIR> / <SWIR> subsections with local band numbering.
   - Associates each local band with its specific 4th-order polynomial smile
     coefficients (if available).

2. **Radiometric calibration**
   - Converts digital numbers (DN) to top-of-atmosphere radiance using
     per-band calibration coefficients:
       Radiance = Gain * DN + Offset.

3. **Spectral smile computation**
   - For a selected band, computes wavelength shifts across the detector
     columns as:
         Δλ(x) = c0 + c1·x + c2·x² + c3·x³ + c4·x⁴  [nm]
       where coefficients c0…c4 are read from the corresponding smile section.
   - Derives column-wise center wavelengths:
         λ(x) = λ_nominal + Δλ(x)
   - (In current L1B products, FWHM(x) is constant across track.)

4. **Outputs and plots**
   - Mean top-of-atmosphere radiance spectra for VNIR and SWIR separately.
   - Spectral smile curves Δλ(x) for user-selected VNIR and SWIR bands.
   - CW(x) and FWHM(x) plots across the detector for the same bands.

Usage
-----
    python enmap_vnir_swir_smile_fixed.py

or within another script:

    run_vnir_swir_independent(
        in_dir=<path_to_L1B_directory>,
        vnir_local_band_to_plot=<VNIR band index>,
        swir_local_band_to_plot=<SWIR band index>
    )

The input directory must contain:
    - *SPECTRAL_IMAGE_VNIR*.TIF
    - *SPECTRAL_IMAGE_SWIR*.TIF
    - *METADATA.XML*

Dependencies
------------
    - Python ≥ 3.8
    - GDAL
    - NumPy
    - Matplotlib

References
----------
    - EN-PCV-ICD-2009-2 : EnMAP HSI Product Specification Level-1/2
    - EN-PCV-TN-4006   : Level-1B Processor – Systematic and Radiometric Correction
    - DLR (2020–2024)  : EnMAP Processor and Calibration Documents

Author
------
    Alvise Ferrari, Sapienza University
    Date: 2025-10-05


"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import xml.etree.ElementTree as ET

gdal.UseExceptions()

# --------------------------- Utilities ---------------------------

def strip_namespaces(root):
    """Remove namespaces so we can use simple tag paths."""
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return root

def find_enmap_files(in_dir):
    """Locate VNIR, SWIR GeoTIFFs and METADATA.XML in a folder."""
    vnir = sorted(glob.glob(os.path.join(in_dir, "*SPECTRAL_IMAGE_VNIR*.TIF"))) \
        or sorted(glob.glob(os.path.join(in_dir, "*VNIR*.TIF")))
    swir = sorted(glob.glob(os.path.join(in_dir, "*SPECTRAL_IMAGE_SWIR*.TIF"))) \
        or sorted(glob.glob(os.path.join(in_dir, "*SWIR*.TIF")))
    xml  = sorted(glob.glob(os.path.join(in_dir, "*METADATA.XML")))
    if not vnir or not swir or not xml:
        raise FileNotFoundError("VNIR/SWIR TIF or METADATA.XML not found in directory")
    return vnir[0], swir[0], xml[0]

def read_cube_gdal(path_tif):
    """Read multi-band GeoTIFF into (bands, rows, cols) float32 array of DN."""
    ds = gdal.Open(path_tif, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open {path_tif}")
    bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    arr = np.empty((bands, rows, cols), dtype=np.float32)
    for b in range(1, bands+1):
        arr[b-1] = ds.GetRasterBand(b).ReadAsArray().astype(np.float32)
    ds = None
    return arr

# --------------------------- Metadata parsing ---------------------------

def parse_metadata_vnir_swir(xml_path):
    """
    Parse METADATA.XML and return VNIR and SWIR lists with LOCAL indexing:
      vnir_meta[i]: dict for local VNIR band (i = 0..Nvnir-1)
      swir_meta[i]: dict for local SWIR band (i = 0..Nswir-1)
    Each dict has:
      'global_id', 'cw_nm', 'fwhm_nm', 'gain', 'offset', 'smile_coeffs' (or None)
    """
    root = ET.parse(xml_path).getroot()
    strip_namespaces(root)

    # --- 1) Global bandCharacterisation (1..N_total) ---
    global_bands = []
    for bn in root.findall(".//bandCharacterisation/bandID"):
        gid = int(bn.attrib["number"])
        global_bands.append({
            'global_id': gid,
            'cw_nm':   float(bn.findtext("wavelengthCenterOfBand")),
            'fwhm_nm': float(bn.findtext("FWHMOfBand")),
            'gain':    float(bn.findtext("GainOfBand")),
            'offset':  float(bn.findtext("OffsetOfBand")),
        })
    global_bands.sort(key=lambda d: d['global_id'])

    # --- 2) SmileCorrection subsections with LOCAL numbering ---
    smile = root.find(".//smileCorrection")
    if smile is None:
        raise RuntimeError("smileCorrection not found in metadata.")

    vnir_smile = smile.find("VNIR")
    swir_smile = smile.find("SWIR")
    if vnir_smile is None or swir_smile is None:
        raise RuntimeError("Expected <smileCorrection><VNIR> and <SWIR> subsections.")

    # Collect local smile coeffs
    def read_smile_section(sec):
        coeffs_by_local = {}  # local_idx (1..) -> np.array([c0..c4])
        wl_by_local = {}      # optional: wavelength tag inside smile (for sanity)
        for bn in sec.findall(".//bandID"):
            local_idx = int(bn.attrib["number"])  # LOCAL numbering
            c = []
            for k in range(5):
                t = bn.findtext(f"coeff{k}")
                if t is None:
                    c = []
                    break
                c.append(float(t))
            if len(c) == 5:
                coeffs_by_local[local_idx] = np.array(c, dtype=float)
            wtxt = bn.findtext("wavelength")
            if wtxt is not None:
                wl_by_local[local_idx] = float(wtxt)
        return coeffs_by_local, wl_by_local

    vnir_coeffs_by_local, vnir_wl_smile = read_smile_section(vnir_smile)
    swir_coeffs_by_local, swir_wl_smile = read_smile_section(swir_smile)

    # --- 3) Split global bands into VNIR / SWIR using LOCAL counts ---
    Nvnir = len(vnir_coeffs_by_local)  # e.g., 91
    Nswir = len(swir_coeffs_by_local)  # e.g., 133

    if Nvnir + Nswir != len(global_bands):
        # still usable: many products have exactly this sum; warn if not
        print(f"[w] NVNIR({Nvnir}) + NSWIR({Nswir}) != Ntotal({len(global_bands)}). Proceeding with slice by counts.")

    vnir_global = global_bands[:Nvnir]
    swir_global = global_bands[Nvnir:Nvnir+Nswir]

    # --- 4) Attach LOCAL smile coeffs to VNIR/SWIR lists ---
    vnir_meta = []
    for i, gb in enumerate(vnir_global, start=1):  # local index 1..Nvnir
        m = dict(gb)  # copy
        m['smile_coeffs'] = vnir_coeffs_by_local.get(i, None)
        m['local_idx'] = i
        vnir_meta.append(m)

    swir_meta = []
    for i, gb in enumerate(swir_global, start=1):  # local index 1..Nswir
        m = dict(gb)
        m['smile_coeffs'] = swir_coeffs_by_local.get(i, None)
        m['local_idx'] = i
        swir_meta.append(m)

    # Optional sanity: compare CW from bandCharacterisation vs <wavelength> in smile
    def check_alignment(meta_list, wl_smile_dict, label):
        diffs = []
        for m in meta_list:
            li = m['local_idx']
            if li in wl_smile_dict:
                diffs.append(abs(m['cw_nm'] - wl_smile_dict[li]))
        if diffs:
            dmax = max(diffs)
            if dmax > 0.5:
                print(f"[w] Max |CW_meta - wavelength_smile| in {label}: {dmax:.3f} nm")
    check_alignment(vnir_meta, vnir_wl_smile, "VNIR")
    check_alignment(swir_meta, swir_wl_smile, "SWIR")

    return vnir_meta, swir_meta

# --------------------------- Calibration ---------------------------

def dn_to_radiance(dn_cube, meta_list):
    """Radiance = gain*DN + offset per band, meta_list is local VNIR or SWIR list."""
    nb = dn_cube.shape[0]
    if nb != len(meta_list):
        print(f"[w] DN bands={nb} vs metadata={len(meta_list)}")
    nb = min(nb, len(meta_list))
    rad = np.empty_like(dn_cube[:nb], dtype=np.float32)
    for i in range(nb):
        g = meta_list[i]['gain']
        o = meta_list[i]['offset']
        rad[i] = g * dn_cube[i] + o
    return rad

# --------------------------- Analysis & Plots ---------------------------

def mean_spectrum(rad_cube, cw_vec):
    return cw_vec, rad_cube.reshape(rad_cube.shape[0], -1).mean(axis=1)

def plot_mean_spectrum(cw, spec, title):
    plt.figure(figsize=(9,4))
    plt.plot(cw, spec, lw=1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def delta_lambda_across_track(band_meta, ncols):
    x = np.arange(1, ncols+1, dtype=np.float64)
    c = band_meta['smile_coeffs']
    if c is None:
        return x, np.zeros_like(x, dtype=float)
    # Δλ(x) [nm] = c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4
    delta = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4
    return x, delta

def plot_smile_delta_lambda(band_meta, ncols, prefix=""):
    x, delta = delta_lambda_across_track(band_meta, ncols)
    note = "" if band_meta['smile_coeffs'] is not None else " (no coeffs in metadata)"
    plt.figure(figsize=(9,4))
    plt.plot(x, delta, lw=1)
    plt.axhline(0, color='k', lw=0.8, alpha=0.4)
    plt.xlabel("Across-track column (x)")
    plt.ylabel("Δλ(x) = CW(x) − CW_nominal  [nm]")
    plt.title(f"{prefix}Spectral smile Δλ(x) — local band {band_meta['local_idx']} (global {band_meta['global_id']}){note}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cw_and_fwhm_across_track(band_meta, ncols, prefix=""):
    x = np.arange(1, ncols+1, dtype=np.float64)
    lam0 = band_meta['cw_nm']
    c = band_meta['smile_coeffs']
    if c is None:
        cw_x = np.full_like(x, lam0, dtype=float)
    else:
        cw_x = lam0 + (c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4)
    fwhm_x = np.full_like(x, band_meta['fwhm_nm'], dtype=float)  # constant in L1B
    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(x, cw_x, lw=1)
    ax1.set_xlabel("Across-track column (x)")
    ax1.set_ylabel("Center wavelength λ(x) [nm]")
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{prefix}CW(x) & FWHM(x) — local band {band_meta['local_idx']} (global {band_meta['global_id']})")
    ax2 = ax1.twinx()
    ax2.plot(x, fwhm_x, lw=1, ls="--", color="orange")
    ax2.set_ylabel("FWHM(x) [nm]")
    plt.tight_layout()
    plt.show()

# --------------------------- Main pipeline ---------------------------

def run_vnir_swir_independent(in_dir, vnir_local_band_to_plot=None, swir_local_band_to_plot=None):
    """Treat VNIR and SWIR independently and use LOCAL band indices for smile."""
    vnir_path, swir_path, xml_path = find_enmap_files(in_dir)
    print(f"[i] VNIR: {os.path.basename(vnir_path)}")
    print(f"[i] SWIR: {os.path.basename(swir_path)}")
    print(f"[i] META: {os.path.basename(xml_path)}")

    # VNIR/SWIR meta with local smile indices
    vnir_meta, swir_meta = parse_metadata_vnir_swir(xml_path)
    print(f"[i] VNIR meta: {len(vnir_meta)} bands (CW {vnir_meta[0]['cw_nm']:.2f}–{vnir_meta[-1]['cw_nm']:.2f} nm)")
    print(f"[i] SWIR meta: {len(swir_meta)} bands (CW {swir_meta[0]['cw_nm']:.2f}–{swir_meta[-1]['cw_nm']:.2f} nm)")

    # Read DN cubes
    vnir_dn = read_cube_gdal(vnir_path)
    swir_dn = read_cube_gdal(swir_path)

    # DN -> Radiance (separate)
    vnir_rad = dn_to_radiance(vnir_dn, vnir_meta)
    swir_rad = dn_to_radiance(swir_dn, swir_meta)

    # Mean spectra
    cw_vnir = np.array([m['cw_nm'] for m in vnir_meta], dtype=float)
    cw_swir = np.array([m['cw_nm'] for m in swir_meta], dtype=float)
    wl_vnir, mean_vnir = cw_vnir, vnir_rad.reshape(vnir_rad.shape[0], -1).mean(axis=1)
    wl_swir, mean_swir = cw_swir, swir_rad.reshape(swir_rad.shape[0], -1).mean(axis=1)

    plot_mean_spectrum(wl_vnir, mean_vnir, title="Mean TOA Radiance — VNIR")
    plot_mean_spectrum(wl_swir, mean_swir, title="Mean TOA Radiance — SWIR")

    # Smile plots — VNIR
    if vnir_local_band_to_plot is not None:
        if not (1 <= vnir_local_band_to_plot <= len(vnir_meta)):
            raise ValueError(f"VNIR local band must be in 1..{len(vnir_meta)}")
        meta_b = vnir_meta[vnir_local_band_to_plot - 1]
        ncols = vnir_rad.shape[2]
        plot_smile_delta_lambda(meta_b, ncols, prefix="VNIR — ")
        plot_cw_and_fwhm_across_track(meta_b, ncols, prefix="VNIR — ")

    # Smile plots — SWIR
    if swir_local_band_to_plot is not None:
        if not (1 <= swir_local_band_to_plot <= len(swir_meta)):
            raise ValueError(f"SWIR local band must be in 1..{len(swir_meta)}")
        meta_b = swir_meta[swir_local_band_to_plot - 1]
        ncols = swir_rad.shape[2]
        plot_smile_delta_lambda(meta_b, ncols, prefix="SWIR — ")
        plot_cw_and_fwhm_across_track(meta_b, ncols, prefix="SWIR — ")

# --------------------------- Example entry ---------------------------

if __name__ == "__main__":
    # main_dir = r"D:\...\L1B-DT0000004147_20221002T074828Z_001_V010501_20241110T222720Z"
    main_dir = r"D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\SNR\codes\test_data\20221002T074828\L1B-DT0000004147_20221002T074828Z_001_V010501_20241110T222720Z"

    # Choose LOCAL band indices for VNIR and SWIR (1-based within each sensor)
    run_vnir_swir_independent(
        main_dir,
        vnir_local_band_to_plot=60,   # e.g., VNIR local band #60
        swir_local_band_to_plot=100   # e.g., SWIR local band #100
    )
