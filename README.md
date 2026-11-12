# HyGAS (Hyperspectral Gas Analysis Suite)

<p align="center">
  <img src="logo/hygas_logo.png" alt="HyGAS logo" width="260" />
</p>

HyGAS is an open-source framework for analysing gas plumes from imaging spectroscopy. Its published, validated workflow is for **methane (CH₄)**: it retrieves path-integrated enhancements (ΔX, **ppm·m**) from Level-1 radiances, characterises uncertainty, and supports consistent plume segmentation, Integrated Mass Enhancement (IME), and flux analysis across multiple satellite products.

The repository also contains an **experimental CO₂ retrieval path for PRISMA, EnMAP, Tanager-1, and EMIT**, and an experimental EMIT Level-1 CH₄ path. Their scope and current limitations are stated explicitly below.

<p align="center">
  <img src="GA_repo_fixed.JPG" alt="Published HyGAS CH₄ framework graphical abstract" width="700" />
</p>

The graphical abstract above is the published visual summary of the HyGAS CH₄ framework.

## What HyGAS supports

| Input / sensor | Gas | Entry point | Retrieval outputs | Common plume analysis | Status |
| --- | --- | --- | --- | --- | --- |
| PRISMA Level-1 + L2C | CH₄ | CLI, scene or batch | ΔX, instrument uncertainty (`σ_RMN`), class map, RGB | Yes, through analysis notebooks | Published workflow |
| EnMAP Level-1B radiances | CH₄ | CLI, scene or batch | ΔX, `σ_RMN`, class map, RGB | Yes, through analysis notebooks | Published workflow |
| Tanager-1 Basic radiance + surface reflectance | CH₄ | CLI, scene | ΔX, `σ_RMN`, class map, RGB | Yes, through analysis notebooks | Published workflow |
| EMIT Level-2B CH₄ enhancement | CH₄ | Analysis notebooks | Provider ΔX and uncertainty are ingested | Yes | Published downstream workflow |
| EMIT Level-1B RAD + OBS + paired L2A MASK atmosphere | CH₄ / CO₂ | CLI, scene or batch | Native ΔX, instrument uncertainty, class/support maps; optional GLT GeoTIFFs | Not yet validated end-to-end | Experimental L1B integration; explicit noise calibration required |
| GHGSat Level-2 CH₄ product | CH₄ | Analysis notebooks | Provider ΔX and uncertainty are ingested | Yes | Published downstream workflow |
| PRISMA Level-1 + L2C | CO₂ | CLI, scene or batch | ΔX, `σ_RMN`, class map, RGB | Segmentation QA notebooks only | Experimental retrieval path |
| EnMAP Level-1B radiances | CO₂ | CLI, scene or batch | ΔX, `σ_RMN`, class map, RGB | Segmentation QA notebooks only | Experimental retrieval path |
| Tanager-1 Basic radiance + surface reflectance | CO₂ | CLI, scene | ΔX, `σ_RMN`, class map, RGB | Segmentation QA notebooks only | Experimental retrieval path |

`σ_RMN` is propagated instrument-noise uncertainty. The background/clutter term (`σ_surf`), total uncertainty (`σ_tot`), segmentation, IME, and flux steps are the **downstream scientific-analysis workflow** used by the case-study notebooks; they are not produced by the retrieval CLI alone.

## How the CH₄ framework works

HyGAS uses the same physical and statistical chain for Level-1 CH₄ retrievals from PRISMA, EnMAP, and Tanager-1:

1. A scene-specific, LUT-derived radiance target represents a 1 ppm·m CH₄ enhancement, including the sensor spectral response.
2. A matched filter estimates a ΔX map and propagates radiometric noise into a per-pixel `σ_RMN` map.
3. Semi-automatic, scale-aware segmentation defines comparable plume masks at each sensor's native resolution.
4. Plume-free pixels with continuum spectra similar to the background beneath each mask are selected to estimate surface/background clutter (`σ_surf`); it combines with `σ_RMN` to form `σ_tot`.
5. The common IME and effective-wind formulation yields plume mass, flux, and their uncertainty.

### CH₄ uncertainty treatment

For PRISMA, EnMAP, and Tanager-1 Level-1 retrievals, HyGAS first propagates the sensor's radiometric noise through the matched filter to obtain the per-pixel instrument term `σ_RMN`. It then estimates a scene-level clutter term, `σ_surf`, from matched-filter variability over plume-free pixels selected for their continuum spectral similarity to the background beneath the plume. The background-aware uncertainty is calculated per pixel as `σ_tot = √(σ_RMN² + σ_surf²)` and is propagated to IME and flux in the downstream analysis. This separates detector/radiometric precision from surface-driven and residual background variability.

For the complete uncertainty ATBD—noise model, spectrally matched-background selection, assumptions, and propagation to IME and flux—refer to the published paper and the [paper-to-code map](docs/paper-notebook-map.md). EMIT and GHGSat use their provider-derived Level-2 uncertainty fields in the downstream HyGAS workflow. The paper is about **CH₄**; it does not validate the repository's later CO₂ extension.

### Reference SNR used for `σ_RMN`

The Level-1 uncertainty workflow uses a reference for each spectral band and
detector column. The four selected desert scenes and their **ready-to-use NPZ
files are included in [reference_snr/](reference_snr/README.md)**, with dates,
product identifiers, spatial selections, hashes and known limitations in the
[scene catalog](reference_snr/scenes.json). EMIT uses the experimental full-scene
Libya-4 acquisition `20240624T091109_2417606_010`.

The numerator is mean observed radiance along the selected rows. Random noise
is estimated from along-track first differences of the **residual after a
four-component PCA reconstruction**, divided by √2. This is not the standard
deviation of the fourth component. Reference SNR is mean radiance divided by
this noise estimate. For positive target-scene radiance, the photon-noise
approximation scales SNR by `√(L_scene / L_ref)` before propagation through the
gas target to `σ_RMN`. The same radiometric NPZ supplies CH4 and CO2; only the
spectral selection and gas target change.

[Generation methods and commands](reference_snr/README.md#generate-references)
distinguish historical notebook writers from A–H sensitivity diagnostics. EMIT
has a [standalone CLI](scripts/SNR_emit_reference.py) and a
[short notebook](notebooks/emit_snr_reference.ipynb), with no private-cache
dependency. The historical Tanager NPZ is unchanged: its recorded 234-row crop
and native radiance scale differ from the later 220-row calibration analysis and
common-unit comparison. These pending Tanager reference issues are documented
in the catalog; they are not silently recalibrated by the EMIT integration.

## Common HyGAS workflow

HyGAS currently combines **automated retrieval** with **notebook-guided plume analysis**. The CLI is suitable for repeatable single-scene and, where supported, batch retrievals; the final plume masks, total-uncertainty derivation, and IME/flux products are produced through the example notebooks so that image-specific statistics and quality-control choices remain visible and reproducible.

1. **Prepare the input product.** Use a Level-1 radiance product for PRISMA, EnMAP, or Tanager-1, with the required LUT, DEM where applicable, and SNR reference. The experimental EMIT L1B route additionally requires paired OBS and atmospheric H2O from L2A MASK. The published EMIT and GHGSat examples instead begin from provider Level-2 CH₄ enhancement and uncertainty fields.
2. **Run or ingest the CH₄ retrieval.** Run `scripts/main.py` for Level-1 scenes to create ΔX and `σ_RMN` rasters, or load Level-2 products in the relevant analysis notebook. Select and record the matched-filter configuration; use multiple configurations when assessing retrieval sensitivity.
3. **Define the analysis domain.** Inspect the enhancement and uncertainty maps. For a multi-sensor comparison, clip every image to the common footprint before calculating segmentation statistics.
4. **Segment plumes semi-automatically.** In the case-study notebook, choose the threshold multiplier and physical morphology settings. The notebook calculates robust background statistics separately for every clipped image, converts the shared physical settings to each sensor's pixel grid, and creates candidate masks. Review the masks, remove implausible isolated objects, and merge fragments belonging to the same source and wind-aligned plume where needed.
5. **Derive total uncertainty.** Use the reviewed plume mask to select spectrally matched, plume-free background pixels. For Level-1 HyGAS retrievals, derive `σ_surf` and combine it with `σ_RMN` to produce `σ_tot`; for Level-2 products, use the uncertainty treatment supplied and documented for that product.
6. **Calculate and report IME and flux.** Provide the final plume polygons and the selected wind input (including its uncertainty) to the IME/flux workflow. Report ΔX statistics, plume area, IME and `σ_IME`, flux and `σ_flux`, together with the retrieval, segmentation, and wind settings.

This is intentionally not a one-click plume-to-flux batch processor yet: the notebook stage exposes the scene-dependent choices that most affect plume extent and quantitative uncertainty. The [notebook index](notebooks/README.md) and the case studies linked below provide executable examples of this workflow.

## Multi-sensor consistency

For each matchup, HyGAS clips all enhancement maps to their shared footprint, then runs the same downstream IME and flux workflow. The plume segmentation is **semi-automatic**:

- For every sensor image, robust background statistics (`μ_bg` and `σ_bg`) are calculated independently from that image's clipped enhancement map. Candidate plume pixels are selected with the same statistical rule, `ΔX > μ_bg + n·σ_bg`.
- Smoothing, morphology, maximum gap, and minimum plume area are defined once in physical units. They are converted to pixel units using each sensor's ground sampling distance, so the geometric criteria remain comparable at different resolutions.
- Connected components and morphological filtering create the initial masks. A final review discards isolated structures inconsistent with the prevailing wind direction and can merge adjacent fragments that share a source and advective direction.

This preserves image-specific background behaviour while applying common physical segmentation criteria. The resulting masks feed the same IME and effective-wind flux equations. For Level-1 PRISMA, EnMAP, and Tanager-1 products, HyGAS also separates instrument noise from spectrally matched background clutter; EMIT and GHGSat enter with provider-derived Level-2 enhancement and uncertainty fields.

The published examples are:

- [Buenos Aires: EnMAP, EMIT, and GHGSat](notebooks/BA_plume_detection_scaled.ipynb) with [IME/flux analysis](notebooks/BA_plume_analysis_enmap_ghgsat_emit.ipynb)
- [Turkmenistan: PRISMA, EnMAP, and GHGSat](notebooks/Turkmenistan_plume_detection_scaled.ipynb) with [IME/flux analysis](notebooks/Turkmenistan_plume_analysis_enmap_prisma_ghgsat.ipynb)
- [Pakistan: EMIT and GHGSat](notebooks/Pakistan_plume_detection_scaled.ipynb) with [IME/flux analysis](notebooks/Pakistan_plume_analysis_emit_ghgsat.ipynb)
- [EnMAP and Tanager-1 revisit comparison](notebooks/BA2_plume_detection_single_MF.ipynb) with [IME/flux analysis](notebooks/BA2_plume_analysis_single_MF.ipynb)

## Reference paper

The CH₄ framework implemented here is described in:

Ferrari, A.; Pampanoni, V.; Laneve, G.; Carvajal Tellez, R.A.; Saquella, S. *A Multi-Sensor Framework for Methane Detection and Flux Estimation with Scale-Aware Plume Segmentation and Uncertainty Propagation from High-Resolution Spaceborne Imaging Spectrometers*. **Methane** 2026, 5(1), 10. DOI: [10.3390/methane5010010](https://doi.org/10.3390/methane5010010).

The exact relationship between paper sections, figures, code, and notebooks is documented in [docs/paper-notebook-map.md](docs/paper-notebook-map.md).

## Installation and inputs

HyGAS requires Python 3.11 or later and GDAL bindings compiled for the same Python interpreter. Create the curated environment with:

```bash
mamba env create -f environment.min.yml
mamba activate hygas
pip install -r requirements-pip.txt
```

Confirm that `from osgeo import gdal` works before running a pipeline.

You will need the following data, as applicable:

- PRISMA Level-1 and Level-2C HE5 files (or official ZIP archives).
- EnMAP VNIR/SWIR GeoTIFFs with their matching `METADATA.XML`.
- Tanager Basic radiance and companion surface-reflectance HDF5 products.
- A CH₄ LUT, `dataset_ch4_full.hdf5`, from the [University of Utah HIVE dataset](https://hive.utah.edu/concern/datasets/9w0323039) (or the [repository mirror](https://drive.google.com/file/d/196adGp_XCcTXAk3SRjiOnBJxUhDANNvn/view?usp=sharing)). Please cite Foote et al., *Impact of Scene-Specific Enhancement Spectra on Matched Filter GreenhouseGas Retrievals from Imaging Spectroscopy*.
- A DEM NetCDF file for PRISMA and Tanager. It must provide `lat`, `lon`, and `elev` in metres; the [SRTM30 Global 1 km DEM](https://catalog.data.gov/dataset/srtm30-global-1-km-digital-elevation-model-dem-version-11-land-surface) is one suitable source.
- A column-wise SNR-reference `.npz` for uncertainty propagation, supplied in [reference_snr/](reference_snr/README.md). Pass `--snr-reference` or set `PRISMA_SNR_REFERENCE`, `ENMAP_SNR_REFERENCE`, `TANAGER_SNR_REFERENCE`, or `EMIT_SNR_REFERENCE`.
- For experimental EMIT L1B retrievals, exactly paired RAD/OBS NetCDFs and the same-scene, same-version L2A MASK providing atmospheric water vapour.
- For the CO₂ path only, a compatible `dataset_co2_full.hdf5` LUT with the same `modtran_data`, `modtran_param`, and `wave` structure as the CH₄ LUT.

### Optional EMIT downloads through NASA Earthdata

Acquisition is separate from retrieval. Install the optional dependencies in your
HyGAS environment (the tested combination pins `earthaccess` to 0.16.0):

```bash
python -m pip install -r requirements-earthdata.txt
```

Generate a personal token through [Earthdata Login](https://urs.earthdata.nasa.gov/).
For persistent configuration, run this once from an interactive terminal:

```bash
python scripts/configure_earthdata.py
# Optional: check the local file without showing its contents or contacting NASA.
python scripts/configure_earthdata.py --check
```

The prompt hides the token and stores it in `~/.config/hygas/earthdata-token`,
outside this repository. On WSL/Linux the file is created with mode `600` and
its new parent directory with mode `700`. The EMIT downloader automatically
uses it in future terminals. Exported token or username/password credentials take
precedence; no shell startup file needs changing. Use
`python scripts/configure_earthdata.py --replace` to update a token.

Alternatively, in Bash/WSL, set a token for the current terminal session only:

```bash
read -r -s -p "Earthdata token: " EARTHDATA_TOKEN
printf '\n'
export EARTHDATA_TOKEN
```

Use an explicit collection, version, scene and product selection. Public preview
does not authenticate or create files; remove `--dry-run` to download. For the
selected SNR scene, first obtain RAD/OBS:

```bash
python scripts/download_emit.py \
  --collection EMITL1BRAD --version 001 \
  --scene 20240624T091109_2417606_010 --products RAD OBS \
  --output downloads/emit/20240624T091109_2417606_010 --dry-run
```

For retrieval, also obtain the paired atmospheric MASK:

```bash
python scripts/download_emit.py \
  --collection EMITL2ARFL --version 001 \
  --scene 20240624T091109_2417606_010 --products MASK \
  --output downloads/emit/20240624T091109_2417606_010 --dry-run
```

For L1B radiance and geometry, use `--collection EMITL1BRAD --version 001`
and `--products RAD OBS`. These are native scene NetCDF downloads, not plume
complex GeoTIFFs. Legacy V001 MASK is part of `EMITL2ARFL`; newer standalone
`EMITL2AMASK` collections have their own versions. No version pairing or upgrade
is inferred. Query each collection explicitly.

The downloader uses [earthaccess search and download APIs](https://earthaccess.readthedocs.io/en/v0.16.0/api/).
It checks NetCDF/HDF5 readability and declared product version (or GeoTIFF readability
and georeferencing for scene enhancement products), exact file sizes
and CMR checksums when supplied, and records SHA-256, source identity and receipts.
Existing files are never overwritten; interrupted downloads remain in `.partial`
for inspection and are not automatically resumed. Repeated runs verify receipts
before reusing files. Authentication defaults to environment credentials, falling
back to the configured local HyGAS token file, with
`persist=False`; `--auth interactive` and `--auth netrc` require explicit selection.
No token, password or signed temporary URL is written to the manifests.

Run all commands in the same terminal: exported variables do not automatically
reach other terminals or an already-running agent. Use `unset EARTHDATA_TOKEN`
when finished. `downloads/`, local `.env*` and `.netrc` files are excluded from Git;
Git exclusion is not a substitute for protecting credentials. Downloading EMIT
L1B data does **not** validate a calibration or establish equivalence to NASA
operational retrievals. The experimental main-CLI path is described below.

## Retrieval quick start

All retrievals use the unified entry point:

```bash
python scripts/main.py --satellite {prisma|enmap|tanager|emit} --mode {scene|batch} --lut /path/to/lut.hdf5 [options]
```

Tanager currently supports `--mode scene` only. PRISMA, EnMAP and experimental EMIT accept both scene and batch modes.

Common options are `--min-wavelength` / `--max-wavelength` (spectral window), `--k` (cluster count for `srf-column`), `--snr-reference` (or its sensor-specific environment-variable fallback), and `--log-file`. `--save-rads` additionally exports the full PRISMA radiance cube and is intentionally unavailable for the other sensors.

### CH₄: EnMAP single scene

```bash
python scripts/main.py \
  --satellite enmap --mode scene \
  --vnir /path/to/...-SPECTRAL_IMAGE_VNIR.TIF \
  --swir /path/to/...-SPECTRAL_IMAGE_SWIR.TIF \
  --metadata /path/to/...-METADATA.XML \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --enmap-mf-mode full-column
```

The default CH₄ window is 2100–2450 nm. Use `--min-wavelength` and `--max-wavelength` to override it.

### CO₂: EnMAP single scene

```bash
python scripts/main.py \
  --satellite enmap --mode scene --gas co2 \
  --vnir /path/to/...-SPECTRAL_IMAGE_VNIR.TIF \
  --swir /path/to/...-SPECTRAL_IMAGE_SWIR.TIF \
  --metadata /path/to/...-METADATA.XML \
  --lut /path/to/dataset_co2_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --enmap-mf-mode full-column
```

### CO₂: PRISMA single scene

```bash
python scripts/main.py \
  --satellite prisma --mode scene --gas co2 \
  --l1 /path/to/PRS_L1_STD_OFFL_YYYYMMDDhhmmss_xxxx.he5 \
  --l2c /path/to/PRS_L2C_STD_YYYYMMDDhhmmss_xxxx.he5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_co2_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --prisma-mf-mode full-column
```

### CO₂: Tanager-1 single scene

```bash
python scripts/main.py \
  --satellite tanager --mode scene --gas co2 \
  --tanager-rad /path/to/basic_radiance.h5 \
  --tanager-sr /path/to/surface_reflectance.h5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_co2_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --tanager-mf-mode full-column
```

### Experimental EMIT L1B: CH₄ or CO₂

```bash
python scripts/main.py \
  --satellite emit --mode scene --gas ch4 \
  --emit-rad /path/to/EMIT_L1B_RAD_001_20240112T185917_2401212_016.nc \
  --emit-obs /path/to/EMIT_L1B_OBS_001_20240112T185917_2401212_016.nc \
  --emit-atmosphere /path/to/EMIT_L2A_MASK_001_20240112T185917_2401212_016.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference reference_snr/emit/20240624T091109_2417606_010/snr_reference_columnwise_experimental.npz \
  --emit-mf-mode full-column \
  --output /path/to/new_output
```

For CO₂, use `--gas co2` and its own compatible CO₂ LUT. Default windows are
2100–2450nm (CH₄) and 1900–2100nm (CO₂), as for the other sensors. The paired
L2A MASK supplies **only atmospheric H2O**, identified by its explicit g/cm² label;
its cloud, Aggregate, AOD and band-interpolation exclusion rules are not applied.
Native dimensions, identity and selected lat/lon must agree exactly. Solar
zenith comes from OBS; mean ground altitude comes from RAD elevation.

The supplied reference contains some nonpositive cells in the default CO₂
window; affected column uncertainties remain unavailable. The four-sensor CO₂
experiment explicitly used `--min-wavelength 1920 --max-wavelength 2100` for all
sensors. This narrower choice is documented in the [EMIT reference notes](reference_snr/emit/20240624T091109_2417606_010/README.md)
and is not silently applied to other runs.

The EMIT choices are deliberately distinct:

- `--emit-mf-mode {srf-column,full-column,advanced,jpl}` selects the existing
  shared MF implementation. Nominal 1D wavelength/FWHM is used; no unprovided
  across-track smile calibration is invented.
- `--emit-noise-backend snr-reference` (default) requires an explicit reference
  or `EMIT_SNR_REFERENCE`. Bands are selected by native ID and columns by physical
  detector index, including crops. No column interpolation or gap filling occurs.
  References must document EMIT, radiance units, full native columns and no
  interpolation. Unsupported/nonpositive reference cells make uncertainty
  unavailable for the affected pixels, without automatically masking enhancement.
- Alternatively, `--emit-noise-backend jpl-parametric --emit-noise-parameters
  /path/to/emit_noise.txt` uses the explicit wavelength/a/b/c EMIT noise model
  in native radiance units, independently of the MF choice. Do not also provide
  an SNR reference. Calibration applicability to a particular build/epoch is
  not established by selecting the file; no wavelength extrapolation is performed.
- `--emit-uncertainty-method hygas` (default, including with JPL MF) retains the
  shared per-pixel HyGAS formula and existing numerical guards. Choosing `jpl`
  instead requires JPL MF and uses its current covariance-weighted adaptation.
  Neither JPL option asserts full numerical parity with the operational pipeline.

Native windows `--emit-rows START STOP` and `--emit-columns START STOP` are
zero-based, contiguous and stop-exclusive. `--emit-native-only` skips GLT export.
Outputs contain native NPZ maps/indices/target/geolocation/support and a JSON
report with input/output/source hashes, atmosphere, noise method, numerical
guards and effective configuration. Default GeoTIFFs have explicit `_ortho`
names and use the original 1-based full-scene GLT/CRS/geotransform; crop-external
pixels stay nodata. Native data are never presented as already orthorectified.
No RGB quicklook or total plume/flux uncertainty is generated by this route yet.
Existing scene outputs are not overwritten.

EMIT batch mode requires `--root-directory` and `--output-root`. Each RAD must
have exactly one same-version OBS and MASK companion below the input root;
duplicate scenes or incomplete pairings are rejected before retrievals start.

CO₂ defaults to 1900–2100 nm. This path has been exercised on the Tanager-1 Korba/Kusmunda scene (`20250219_053251_31_4001`) with 40 selected bands, but it is an engineering test of data loading, target synthesis, uncertainty propagation, and export—not a validation of CO₂ plume detection or retrieved magnitude.

## Matched-filter modes

| Paper term | CLI selection | Intended use |
| --- | --- | --- |
| CMF (scene-wide) | `--*-mf-mode srf-column --k 1` | Scene-wide background statistics with column-wise targets |
| CTMF (k = 3) | `--*-mf-mode srf-column --k 3` | Cluster-tuned background statistics |
| CWCMF (column-wise) | `--*-mf-mode full-column` | Per-detector-column target/statistics; `--k` is ignored |

`advanced` (grouped PCA plus shrinkage) and `jpl` (JPL/EMIT-style adaptation) are additional research modes. The CLI default is `srf-column`; in the published multi-sensor CH₄ case studies, **CWCMF / `full-column`** was the primary reference configuration because it most consistently reduced detector-column artefacts and stabilised IME/flux estimates.

## Sensor-specific commands

### PRISMA

```bash
python scripts/main.py \
  --satellite prisma --mode scene \
  --l1 /path/to/PRS_L1_STD_OFFL_YYYYMMDDhhmmss_xxxx.he5 \
  --l2c /path/to/PRS_L2C_STD_YYYYMMDDhhmmss_xxxx.he5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --prisma-mf-mode full-column
```

PRISMA accepts HE5 files or official ZIP archives. Batch mode scans scene folders under `--root-directory` and writes a corresponding output tree under `--output-root` (or adjacent to each input scene when omitted).

```bash
python scripts/main.py \
  --satellite prisma --mode batch \
  --root-directory /path/to/prisma_root \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output-root /path/to/output_root \
  --prisma-mf-mode full-column
```

### EnMAP

```bash
python scripts/main.py \
  --satellite enmap --mode batch \
  --root-directory /path/to/enmap_root \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --enmap-mf-mode full-column
```

Each EnMAP scene directory must contain its VNIR and SWIR GeoTIFFs and `METADATA.XML`. Scene mode is shown in the quick-start example above.

For case studies organised as EnMAP Level-1B ZIP archives, `scripts/run_enmap_borger2025_co2.py` selects the newest archive revision per acquisition and resumes from `co2_batch_manifest.json`:

```bash
python scripts/run_enmap_borger2025_co2.py \
  --root /path/to/enmap_case_studies \
  --lut /path/to/dataset_co2_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz
```

### Tanager-1 CH₄

```bash
python scripts/main.py \
  --satellite tanager --mode scene \
  --tanager-rad /path/to/basic_radiance.h5 \
  --tanager-sr /path/to/surface_reflectance.h5 \
  --dem /path/to/dem.nc \
  --lut /path/to/dataset_ch4_full.hdf5 \
  --snr-reference /path/to/snr_reference_columnwise.npz \
  --output /path/to/output_dir \
  --tanager-mf-mode full-column
```

## Retrieval outputs and scope

PRISMA, EnMAP and Tanager CLI retrievals write sensor-specific GeoTIFF names under the output directory (experimental EMIT uses the explicit native/ortho convention described above):

| Output | Meaning |
| --- | --- |
| `*_MF.tif` | Matched-filter gas enhancement, ΔX (ppm·m) |
| `*_MF_uncertainty.tif` | Propagated instrument-noise uncertainty, `σ_RMN` |
| `*_CL.tif` / `*_classified.tif` | Technical classification map used by the selected MF mode |
| `*_RGB.tif` / `*_rgb.tif` | RGB quicklook |
| `*_MF_sensitivity.tif` | Sensitivity map exported by `jpl` mode for PRISMA and EnMAP |
| `processing_report.txt` | Inputs, parameters, spectral window, and processing provenance |

PRISMA batch runs also create a `directory_process_report_<timestamp>.txt` summary. `*_MF_uncertainty.tif` is **not** a total plume-analysis uncertainty: it excludes the spectrally matched clutter term, segmentation effects, and flux/wind contributions.

### CO₂ limitations

The experimental CO₂ path is implemented for PRISMA, EnMAP, Tanager-1 and EMIT. The first three create the same four basic retrieval outputs (ΔX, `σ_RMN`, classification, RGB); EMIT currently exports enhancement, instrument uncertainty and native/ortho support maps, without RGB. The CO₂ notebooks provide full-scene segmentation quality assurance and reviewed plume masks, but the repository does **not** yet implement CO₂ background-clutter quantification, `σ_tot`, IME, flux, or validation against an independent CO₂ reference. Do not treat the outputs as quantitatively validated CO₂ plume products.

## Notebooks and examples

The paper notebook index is in [notebooks/README.md](notebooks/README.md). Useful entry points are:

- [CH₄ radiance windows and LUT targets](notebooks/ch4_radiance_windows.ipynb)
- [EnMAP uncertainty walkthrough](notebooks/diagnostics_uncertainty_enmap.ipynb)
- [PRISMA uncertainty walkthrough](notebooks/diagnostics_uncertainty_prisma.ipynb)
- [Spectrally matched background selection](notebooks/plume_analysis_enmap.ipynb)
- [PRISMA, EnMAP, and Tanager SNR experiments](notebooks/SNR_experiments_prisma.ipynb), [EnMAP](notebooks/SNR_experiments_enmap.ipynb), and [Tanager](notebooks/SNR_experiments_tanager.ipynb)
- [Cross-sensor SNR comparison](notebooks/tanager_prisma_enmap_SNR_comparison.ipynb) and [striping diagnostics](notebooks/striping_sweep_diagnostics_cal_scenes_triple.ipynb)
- [CO₂ segmentation and quality assurance](notebooks/co2_plume_segmentation.ipynb) and [CO₂ retrieval showcase](notebooks/co2_retrieval_showcase.ipynb)

Notebook outputs are deliberately excluded from version control. When adding a new README example figure, create a clean, captioned, unit-checked derivative under `docs/assets/` rather than linking directly to `notebooks/outputs/`.

## Utilities and troubleshooting

- `scripts/snr_experiment.py` runs the A–H SNR experiment for PRISMA, EnMAP, Tanager, or EMIT inputs.
- `scripts/enmap_smile.py` and `scripts/prisma_smile.py` provide spectral-smile diagnostics.
- `scripts/plumes_analyzer.py` computes IME, flux, and uncertainty on supplied plume polygons.
- `scripts/inspect_prisma_hdf.py` and `scripts/inspect_tanager_hdf.py` inspect product structures; `scripts/tanager_quicklook.py` creates an RGB preview.
- `scripts/ghgsat_catalog_to_geojson.py` converts GHGSat catalogue CSV exports to point/buffer GeoJSON.

Run utilities from the repository root. The legacy `scripts/PRISMA/prisma_MF.py` and `scripts/EnMAP/enmap_MF.py` entry points remain for backwards compatibility, but new retrievals should use `scripts/main.py`.

If a pipeline reports missing required inputs, check the sensor-specific command above. PRISMA and Tanager require a DEM; EnMAP ignores it. Record the chosen spectral window, MF mode, SNR reference, and `k` when comparing products.

## Repository layout

- `scripts/main.py` — unified retrieval CLI.
- `scripts/pipelines/` — PRISMA, EnMAP, and Tanager orchestration.
- `scripts/core/` — LUT, targets, matched filters, noise, and reporting utilities.
- `scripts/satellites/` — sensor readers and adapters.
- `scripts/plumes_analyzer.py` — downstream IME/flux utilities.
- `notebooks/` — published-workflow and development notebooks.
- `docs/paper-notebook-map.md` — paper section/figure to implementation mapping.

## License

See [LICENSE](LICENSE).
