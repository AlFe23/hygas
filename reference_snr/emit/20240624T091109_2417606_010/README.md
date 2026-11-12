# Experimental EMIT reference: Libya-4, 24 June 2024

This is the **unchanged** HyGAS experimental baseline
`emit_D_full2176_v1_20261023`. It supports explicit EMIT L1B CH4/CO2 instrument-
uncertainty experiments. It is not an operational NASA/JPL noise calibration.

| Property | Selection |
| --- | --- |
| RAD | `EMIT_L1B_RAD_001_20240624T091109_2417606_010.nc` |
| Product / software build | V001 / 010618 |
| Native radiance image | 2176 rows × 1242 columns × 285 bands |
| Reference | 201 bands × 1242 columns; native band IDs 84:285 |
| Spectral selection | 1000–2500 nm inclusive; band centres 1006.834–2492.924 nm |
| Spatial selection | All native rows and columns; no rotation or resampling |
| Method | Global PCA, four components, full SVD; along-track residual first differences, ddof=1, divided by √2 |
| Radiance | µW cm⁻² sr⁻¹ nm⁻¹; multiplier 1 |
| Validity | Finite selected spectra after declared fill/range rules; no automatic homogeneity ranking |
| Preprocessing | No cloud/Aggregate/bandmask exclusion, destriping, equalisation, notch or column interpolation |

Every column has 2176 contributing spectra and 2175 adjacent-row pairs. Of
249642 band/column cells, **41 have nonpositive mean/SNR**; they are retained
for traceability and cannot support positive-radiance uncertainty scaling.
In 1900–2100 nm, 10 cells affect 8 columns; in 2100–2450 nm all reference cells
have positive support. The EMIT pipeline reports support and leaves affected
uncertainties NaN instead of filling columns. The four-sensor CO2 experiment used
the explicitly narrower 1920–2100 nm window; it does not change the CLI default.

Known diagnostics include anomalous rows 114–115, sensitivity of edge column
1241 to the PCA domain, and provider band-interpolation flags. Their presence was
documented without changing the agreed full-scene baseline. All-column spatial
coverage does not imply valid positive radiometric support in every band.

## Files and integrity

- `snr_reference_columnwise_experimental.npz`: directly usable by the retrieval
  CLI via `--snr-reference`. SHA-256:
  `977c02481454ac15b6d52c4f1b5a2b7852c4d32a901368449bb83be5067bd954`.
- `support_and_QA.npz`: unchanged companion with float64 measured sigma, counts,
  adjacent-pair counts, positive-support mask, native indices, FWHM and provider
  interpolation-flag row counts. `QA_applied=False`; flags are diagnostic.
  SHA-256: `1a18351b0edcc1920dc2837998167663105d2f988525eaca42e594cea5f0f7f8`.
- `provenance.json`: portable source identities and publication verification.

The historical NPZ metadata retains paths into the original analysis archive.
These strings are provenance only; the loader does not open those paths. The
portable files in this directory and the public generator are sufficient.
The new generator writes its own `support.npz`, without requiring the historical
provider-QA diagnostic archive. New metadata/timestamps and compression can
change the NPZ hash; compare numeric arrays, not byte identity, when regenerating.

See the [method, scene catalog and regeneration command](../../README.md).
Changing input build, scene, row selection, PCA setting or QA policy creates a
different reference and requires scientific review before replacing this one.
