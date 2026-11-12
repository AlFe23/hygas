# Column-wise reference SNR

These are the radiometric references selected for HyGAS. They are included in
Git, so users can propagate instrument noise without downloading the calibration
scenes or running PCA again. The selections are project references; they are not
mission-wide noise calibrations endorsed by the data providers.

The machine-readable [scene catalog](scenes.json) gives product identifiers,
acquisition dates, spatial selections, generation entry points, file hashes and
known limitations. Satellite source products and LUTs are not distributed here.

| Sensor | Acquisition (UTC) | Locality | Reference domain | NPZ |
| --- | --- | --- | --- | --- |
| PRISMA | 2020-04-01 08:53:13 | Northern State, Sudan | All 1000 rows × 1000 columns | [Historical reference](prisma/20200401085313/snr_reference_columnwise.npz) |
| EnMAP | 2022-07-12 10:43:02 | Agadez region, Niger | All 1024 rows × 1000 columns, segment 001 | [Historical reference](enmap/L1B-DT0000001584_20220712T104302Z_001_V010502_20251017T093724Z/snr_reference_columnwise.npz) |
| Tanager-1 | 2025-05-09 09:03:23 | Northern State, Sudan | Historical NPZ: rows 0:234 × 607 columns, automatic mask fraction 0.12 | [Historical reference](tanager/snr_reference_columnwise.npz) |
| EMIT | 2024-06-24 09:11:09 | Libya-4 area, Libya | All 2176 rows × 1242 native columns | [Experimental reference](emit/20240624T091109_2417606_010/snr_reference_columnwise_experimental.npz), [details](emit/20240624T091109_2417606_010/README.md) |

Rows/columns use zero-based indices and exclusive stop bounds. In the later
Tanager calibration analysis only the upper **220 rows** were judged sufficiently
homogeneous. The unchanged historical NPZ instead records `roi=0:607,0:234`, as
does its generation command. The 220-row analysis and the distributed 234-row
reference must not be described as identical. Tanager reference-unit consistency
also remains a separate maintenance item: its legacy generator stores the native
radiance scale, whereas comparisons convert mean radiance by ×0.1 to
µW cm⁻² sr⁻¹ nm⁻¹. This EMIT release does not recalibrate the Tanager file or
establish its absolute uncertainty calibration.

## Method and use

For a spectral band `b` and physical detector column `c`, take the mean of the
**observed radiance along the selected rows**, `μ_ref(b,c)`. Estimate a global
four-component PCA reconstruction of the selected SWIR spectra and subtract it
from the observations to obtain residuals `R`. The random-noise estimator is:

$$
\sigma_{\mathrm{ref}}(b,c)=\frac{\operatorname{std}_{\mathrm{ddof}=1}
\left[R(b,r+1,c)-R(b,r,c)\right]}{\sqrt{2}},
\qquad
\mathrm{SNR}_{\mathrm{ref}}(b,c)=\frac{\mu_{\mathrm{ref}}(b,c)}
{\sigma_{\mathrm{ref}}(b,c)+10^{-12}}.
$$

Only pairs of consecutive valid native rows contribute. The denominator is the
noise estimated from the **residual after four components**, not the standard
deviation of the fourth component. PCA is fitted once over the selected spatial
domain and spectral interval, not independently per column. A constant column
offset disappears in row differences; reference SNR is therefore not a complete
measure of striping. PCA retention and along-track noise correlation influence
the estimate.

The legacy PRISMA/EnMAP generation block uses `frac_keep=1`; Tanager uses
`mask_frac=0.12` within its recorded crop. Those legacy writers can interpolate
missing columns. The EMIT writer uses all finite selected spectra and **never
interpolates columns**. These differences are explicit rather than hidden by a
single label for all four historical files.

For positive radiance, the shared uncertainty workflow approximately scales
`SNR_scene = SNR_ref × sqrt(L_scene / μ_ref)` and computes `σ_MN = L_scene /
SNR_scene`, with the numerical floors in `scripts/core/noise.py`. The gas target
then propagates this instrument term to enhancement uncertainty. A reference is
**radiometric**, not gas-specific: the same NPZ supplies both CH4 and CO2 by
selecting the appropriate detector bands. The reference does not include target
error, surface clutter, or total plume/flux uncertainty.

Use `--snr-reference <npz>` or the corresponding `PRISMA_SNR_REFERENCE`,
`ENMAP_SNR_REFERENCE`, `TANAGER_SNR_REFERENCE`, `EMIT_SNR_REFERENCE` environment
variable. EMIT preserves native band/column identities and rejects resizing.
Missing or nonpositive reference support leaves uncertainty unavailable; no
positive replacement is invented. Transfer between different processing builds
or epochs requires separate scientific assessment.

## Generate references

- **PRISMA and EnMAP:** the `build_columnwise_snr_reference` block in
  [prisma_enmap_SNRcomparison.ipynb](../notebooks/prisma_enmap_SNRcomparison.ipynb).
  Its `SCENE_CONFIG` identifies the selected products. Adapt local paths before
  running it; its output path is separate from the published references.
- **Tanager-1:** `python -m scripts.SNR_tanager_reference`; the generation cell in
  [tanager_prisma_enmap_SNR_comparison.ipynb](../notebooks/tanager_prisma_enmap_SNR_comparison.ipynb)
  records the historical `--roi 0:607,0:234 --mask-frac 0.12 --k-pca 4 --sigma-mode diff`.
- **EMIT:** [scripts/SNR_emit_reference.py](../scripts/SNR_emit_reference.py) is a
  standalone radiance-to-NPZ CLI. The short
  [EMIT reference notebook](../notebooks/emit_snr_reference.ipynb) calls the same
  script and plots its outputs. Neither requires private analysis caches.

The historical `SNR_experiments_*` notebooks invoke A–H diagnostics through
`scripts.snr_experiment`; those diagnostics do not themselves export the final
`ColumnwiseSNRReference` NPZ.

From the repository root, after obtaining the exact EMIT product:

```bash
python scripts/SNR_emit_reference.py \
  --input /path/to/EMIT_L1B_RAD_001_20240624T091109_2417606_010.nc \
  --output-dir outputs/emit_reference_rebuilt \
  --bands 1000 2500 --k-pca 4 \
  --compare-reference reference_snr/emit/20240624T091109_2417606_010/snr_reference_columnwise_experimental.npz
```

Defaults use all native rows/columns and two numerical-library threads. The
full-scene SVD can require about 12 GiB of RAM; it is not a streaming PCA. An
explicit `--rows START STOP` is available for a different experiment, but changes
the PCA basis and reference and will generally fail comparison with this baseline.
For another scene, omit `--compare-reference` and validate its suitability first.
The command refuses an existing output directory.

Outputs are the experimental reference NPZ, `support.npz` (counts, adjacent-row
pairs, measured sigma, positive-support mask and native indices), `pca.npz`
(basis, mean and explained variance), and `generation_report.json` (input/source/
output hashes, selected window, numerical settings, environment and comparison).
Only RAD is needed for SNR; OBS and atmospheric MASK are needed for retrieval.
The generator records validity but does not assess scene homogeneity for the user.
