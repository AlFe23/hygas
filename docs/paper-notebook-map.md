# HyGAS paper → code / notebook map (methane-4061227)

This repository implements the end-to-end methane enhancement, uncertainty, plume segmentation, IME, and flux framework described in:

- **A Multi-Sensor Framework for Methane Detection and Flux Estimation with Scale-Aware Plume Segmentation and Uncertainty Propagation from High-Resolution Spaceborne Imaging Spectrometers**
- Manuscript ID: `methane-4061227` (MDPI *Methane*)
- Local proof PDF: `product_spec_docs/methane-4061227 - send to 2nd proof_VP_AF.pdf` *(not tracked in git; see `.gitignore`)*

## Paper naming vs HyGAS CLI naming

The paper discusses three matched-filter “flavours” for PRISMA/EnMAP/Tanager:

- **CMF** (scene-wide statistics) → `--*-mf-mode srf-column --k 1`
- **CTMF (k=3)** (cluster-tuned statistics) → `--*-mf-mode srf-column --k 3`
- **CWCMF** (per-detector-column statistics) → `--*-mf-mode full-column` (no clustering; `--k` ignored)

HyGAS also exposes two additional modes that are useful for research/diagnostics:

- `advanced`: grouped PCA + shrinkage workflow (`advanced_matched_filter.py:1`)
- `jpl`: JPL/EMIT MF adaptation (`scripts/core/jpl_matched_filter.py:1`)

## Section → code map (ATBD)

### §2 Multi-Sensor CH₄ Retrieval and Flux Framework

- **Radiative basis, LUT simulation, κ(λ) / t(λ)=μ⊙κ** (§2.1–2.2; Figs 1–2)
  - `scripts/core/lut.py:1`
  - `scripts/core/targets.py:1`
  - Notebook: `notebooks/ch4_radiance_windows.ipynb:1`

- **Matched filter + variants** (§2.3–2.5)
  - `scripts/core/matched_filter.py:1`
  - Pipelines: `scripts/pipelines/enmap_pipeline.py:1`, `scripts/pipelines/prisma_pipeline.py:1`, `scripts/pipelines/tanager_pipeline.py:1`
  - CLI entrypoint: `scripts/main.py:1`

- **Instrument noise, SNR references, σ_RMN propagation** (§2.6.1; Fig 11)
  - `scripts/core/noise.py:1`
  - Diagnostics notebooks: `notebooks/diagnostics_uncertainty_enmap.ipynb:1`, `notebooks/diagnostics_uncertainty_prisma.ipynb:1`
  - SNR experiments CLI + notebooks: `scripts/snr_experiment.py:1`, `notebooks/SNR_experiments_enmap.ipynb:1`, `notebooks/SNR_experiments_prisma.ipynb:1`, `notebooks/SNR_experiments_tanager.ipynb:1`

- **Spectrally matched background selection, σ_surf** (§2.6.2; Fig 4)
  - Notebook (walkthrough + plotting): `notebooks/plume_analysis_enmap.ipynb:1`
  - Used inside multi-sensor case-study notebooks as a helper for EnMAP/PRISMA background-matched clutter.

- **Scale-aware segmentation** (§2.7; Figs 6/17/19/21; Tables 6/9/12)
  - Notebooks:
    - `notebooks/BA_plume_detection_scaled.ipynb:1`
    - `notebooks/Turkmenistan_plume_detection_scaled.ipynb:1`
    - `notebooks/Pakistan_plume_detection_scaled.ipynb:1`
    - `notebooks/BA2_plume_detection_single_MF.ipynb:1`

- **IME + flux computation and uncertainty propagation** (§2.8–2.9; Tables 7/10/13/14)
  - Core implementation: `scripts/plumes_analyzer.py:1`
  - Applied by the case-study notebooks listed in §5 below.

### §3 Satellite Products and Specific Implementations

- **PRISMA / EnMAP / Tanager (Level-1 radiances)**: `scripts/pipelines/prisma_pipeline.py:1`, `scripts/pipelines/enmap_pipeline.py:1`, `scripts/pipelines/tanager_pipeline.py:1` (driven via `scripts/main.py:1`)
- **EMIT (JPL MF, Level-2 enhancement usage)**: `scripts/core/jpl_matched_filter.py:1` (and downstream segmentation/IME/flux via notebooks)
- **GHGSat products (Level-2 enhancement usage)**: downstream segmentation/IME/flux via notebooks (no end-to-end CLI pipeline)

### §4 Radiometric Comparison (SNR / smile / striping)

- **Reference SNR estimation, A–H SNR experiment** (Figs 7–10)
  - CLI: `scripts/snr_experiment.py:1`
  - Runner notebooks: `notebooks/SNR_experiments_enmap.ipynb:1`, `notebooks/SNR_experiments_prisma.ipynb:1`, `notebooks/SNR_experiments_tanager.ipynb:1`
  - Cross-sensor comparison: `notebooks/tanager_prisma_enmap_SNR_comparison.ipynb:1`

- **Spectral smile metrics** (Fig 12; Table 2)
  - `scripts/enmap_smile.py:1`
  - `scripts/prisma_smile.py:1`

- **Across-track striping metrics + ratio maps** (Figs 13–15; Tables 3–4)
  - Notebook: `notebooks/striping_sweep_diagnostics_cal_scenes_triple.ipynb:1`
  - Shared striping utilities: `scripts/diagnostics/striping.py:1`
  - Paper-ready exports land under: `notebooks/outputs/paper_figs/` (see notebook config).

## §5 Case studies → notebooks

- **Buenos Aires (2024-01-12, EnMAP + GHGSat + EMIT)** (Figs 16–17; Tables 5–7)
  - Segmentation: `notebooks/BA_plume_detection_scaled.ipynb:1`
  - IME/flux + recap tables/mosaics: `notebooks/BA_plume_analysis_enmap_ghgsat_emit.ipynb:1`

- **Turkmenistan (2024-09-11, EnMAP + PRISMA + GHGSat)** (Figs 18–19; Tables 8–10)
  - Segmentation: `notebooks/Turkmenistan_plume_detection_scaled.ipynb:1`
  - IME/flux + recap tables/mosaics: `notebooks/Turkmenistan_plume_analysis_enmap_prisma_ghgsat.ipynb:1`

- **Pakistan (2024-02-28, EMIT + GHGSat)** (Figs 20–21; Tables 11–13)
  - Segmentation: `notebooks/Pakistan_plume_detection_scaled.ipynb:1`
  - IME/flux + recap tables/mosaics: `notebooks/Pakistan_plume_analysis_emit_ghgsat.ipynb:1`

- **Non-simultaneous revisit (EnMAP vs Tanager)** (Figs 22–23; Table 14)
  - Segmentation: `notebooks/BA2_plume_detection_single_MF.ipynb:1`
  - IME/flux + recap tables/mosaics: `notebooks/BA2_plume_analysis_single_MF.ipynb:1`

## Notes on data / paths

- Large inputs under `case_studies_data/`, `test_data/`, and `product_spec_docs/` are intentionally excluded from version control (see `.gitignore`).
- Many notebooks currently contain absolute local paths; the recommended long-term cleanup is to move these to a small YAML/JSON config per case study and keep notebook paths relative to `REPO_ROOT`.
