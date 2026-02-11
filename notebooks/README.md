# Notebook index

This folder contains the notebooks used to develop and validate HyGAS (methane enhancement retrieval, uncertainty propagation, scale-aware segmentation, IME, and flux estimation).

Notes:
- **`notebooks/to_rev/` is intentionally excluded from this index** (ongoing work).
- Outputs are generally written under `notebooks/outputs/` (also excluded from git; see `.gitignore`).
- Some notebooks are legacy/prototyping artefacts; the top-level `README.md` lists only the notebooks used for the paper.

## Notebooks used for the paper (methane-4061227)

Radiative / MF fundamentals:
- `notebooks/ch4_radiance_windows.ipynb`: LUT → radiance windows + κ(λ) and t(λ)=μ⊙κ (paper Figs 1–2).

Radiometric comparison (SNR / striping):
- `notebooks/SNR_experiments_enmap.ipynb`: runs the A–H SNR experiment CLI for EnMAP calibration scenes.
- `notebooks/SNR_experiments_prisma.ipynb`: runs the A–H SNR experiment CLI for PRISMA calibration scenes.
- `notebooks/SNR_experiments_tanager.ipynb`: runs the A–H SNR experiment CLI for Tanager calibration scenes.
- `notebooks/tanager_prisma_enmap_SNR_comparison.ipynb`: cross-sensor SNR comparisons (brightness-normalised).
- `notebooks/striping_sweep_diagnostics_cal_scenes_triple.ipynb`: PRISMA vs EnMAP vs Tanager striping sweep; exports paper-ready figures to `notebooks/outputs/paper_figs/`.
- `notebooks/diagnostics_uncertainty_enmap.ipynb`: matched-filter + σ_RMN derivation walkthrough for EnMAP.
- `notebooks/diagnostics_uncertainty_prisma.ipynb`: matched-filter + σ_RMN derivation walkthrough for PRISMA.
- `notebooks/plume_analysis_enmap.ipynb`: plume-level workflow including spectrally matched background selection (SAD) for clutter estimation.

Case studies (segmentation → IME/flux):
- `notebooks/BA_plume_detection_scaled.ipynb`: scale-aware plume segmentation across EnMAP/GHGSat/EMIT (Buenos Aires 2024-01-12).
- `notebooks/BA_plume_analysis_enmap_ghgsat_emit.ipynb`: IME/flux + uncertainty comparison (Buenos Aires 2024-01-12).
- `notebooks/Turkmenistan_plume_detection_scaled.ipynb`: scale-aware plume segmentation across PRISMA/EnMAP/GHGSat (Turkmenistan 2024-09-11).
- `notebooks/Turkmenistan_plume_analysis_enmap_prisma_ghgsat.ipynb`: IME/flux + uncertainty comparison (Turkmenistan 2024-09-11).
- `notebooks/Pakistan_plume_detection_scaled.ipynb`: scale-aware plume segmentation across EMIT/GHGSat (Pakistan 2024-02-28).
- `notebooks/Pakistan_plume_analysis_emit_ghgsat.ipynb`: IME/flux + uncertainty comparison (Pakistan 2024-02-28).
- `notebooks/BA2_plume_detection_single_MF.ipynb`: single-MF segmentation for EnMAP vs Tanager revisit.
- `notebooks/BA2_plume_analysis_single_MF.ipynb`: EnMAP vs Tanager revisit analysis (non-simultaneous).

See `docs/paper-notebook-map.md` for a figure/table-to-code mapping.

## Pipeline demos (operational CLI parity)

These run the same code path as `python scripts/main.py ...` and are useful sanity checks.

- `notebooks/matched_filter_demo_enmap.ipynb`: EnMAP end-to-end matched-filter demo (ΔX + σ_RMN outputs).
- `notebooks/matched_filter_demo_prisma.ipynb`: PRISMA end-to-end matched-filter demo.
- `notebooks/matched_filter_demo_tanager.ipynb`: Tanager end-to-end matched-filter demo.
- `notebooks/matched_filter_tanager_all_modes.ipynb`: quick sweep of Tanager MF modes.

## Per-sensor plume analysis (single-sensor workflows)

These notebooks are useful when working on one sensor at a time (often superseded by the multi-sensor case-study notebooks).

- `notebooks/plume_analysis_enmap.ipynb`: EnMAP plume analysis + background-matched clutter workflow (spectral SAD selection).
- `notebooks/plume_analysis_prisma.ipynb`: PRISMA plume analysis walkthrough.
- `notebooks/plume_analysis_emit.ipynb`: EMIT L2B plume analysis walkthrough.
- `notebooks/plume_analysis_ghgsat.ipynb`: GHGSat plume analysis walkthrough.

## Uncertainty diagnostics (standalone)

- `notebooks/uncertainty_analysis_enmap.ipynb`: derive σ_tot / σ_surf by masking plumes on EnMAP outputs.
- `notebooks/uncertainty_analysis_prisma.ipynb`: same for PRISMA outputs.
- `notebooks/uncertainty_analysis_enmap_spectral.ipynb`: EnMAP background-matched clutter workflow (continuum SAD selection).
- `notebooks/uncertainty_analysis_prisma_spectral.ipynb`: PRISMA background-matched clutter workflow.

## Radiometry / calibration utilities

- `notebooks/tanager_calibration_selection.ipynb`: identifies homogeneous row subsets for Tanager calibration scenes.
- `notebooks/tanager_hdf_explorer.ipynb`: browse Tanager HDF5 content and quicklook.
- `notebooks/prisma_preprocess_diagnostics.ipynb`: PRISMA preprocessing checks (radiance cube sanity, masks, etc.).
- `notebooks/prisma_enmap_SNRcomparison.ipynb`: earlier PRISMA vs EnMAP SNR comparison notebook (superseded by `notebooks/tanager_prisma_enmap_SNR_comparison.ipynb`).

## Legacy / development notebooks

- `notebooks/plume_detection.ipynb`: early plume detection prototype notebook (kept for reference).
- `notebooks/plume_detection_Old.ipynb`: older version of plume detection prototype.
- `notebooks/striping_sweep_diagnostics.ipynb`: PRISMA vs EnMAP striping sweep (generic).
- `notebooks/striping_sweep_diagnostics_cal_scenes.ipynb`: PRISMA vs EnMAP striping sweep on calibration scenes.
- `notebooks/striping_sweep_diagnostics_turk_scenes.ipynb`: PRISMA vs EnMAP striping sweep on Turkmenistan scenes.

## Additional case-study notebooks (not used in the paper)

- `notebooks/PRISMA_HassiMessaoud_plume_detection_scaled.ipynb`: PRISMA case study (Hassi Messaoud).
- `notebooks/PRISMA_HassiMessaoud_plume_analysis_prisma.ipynb`: PRISMA case study analysis (Hassi Messaoud).
- `notebooks/Tanager_HoChiMin_plume_detection_single_MF.ipynb`: Tanager case study (Ho Chi Minh).
- `notebooks/Tanager_HoChiMin_plume_analysis_single_MF.ipynb`: Tanager case study analysis (Ho Chi Minh).
- `notebooks/SNR_experiments_emit.ipynb`: EMIT radiometric/SNR experiments (development).
