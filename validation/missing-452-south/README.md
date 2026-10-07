# South missing-conditioning reproduction for PR #452

This package reproduces the design of the [September 10 pilot for PR #452](https://github.com/alan-turing-institute/cryocast/pull/452#issuecomment-5622298939) on the CryoCast implementation shared by commits `393157f8152ef068bbf8d42198f0d2aac7641637` and `71dd142e9c6f939e0fdc335d39a447f7f49a261b`. A separately labelled extension evaluates missing conditioning at inference. The historical script is preserved for comparison. Another working session added northern validation artifacts during startup; the complete `cryocast` source/configuration tree is identical at the two commits (tree `228900623d72f38ca5eb1fd50984b8f0239055e6`). The runner recorded the latter commit before training. Its source file was frozen before launch; `provenance.json` records the timestamps and hashes.

## Experimental design

| Setting | Value |
| --- | --- |
| Observations | South OSISAF SIC, Argo PSAL/TEMP, ERA5 10u/10v/2t/msl |
| Native grid | 432 × 432 |
| Training dates | January 2020 through December 2021 |
| Held-out dates | January and July 2024 |
| Forecast windows | Three history days, two forecast days |
| Normalisation | Existing per-variable source statistics with recorded period January 1, 2020 through November 14, 2021, within training |
| Training missingness | Withhold Argo date indices satisfying `i % 5 == 2` before constructing windows |
| Training windows | 290 strict; 727 fill |
| Common complete-input test windows | 54 |
| Seeds | 123 and 456 |
| Initialisation | Identical complete model state within each seed; hashes recorded |
| Model | Native `EncodeProcessDecode`, naive linear encoders/decoder, 32 × 32 latent grid, UNet with eight starting channels; 174,271 parameters |
| Optimisation | Adam at 0.001; 500 updates per arm; batch size two; fixed final checkpoint |
| Compute | CPU, four PyTorch threads, one inter-op thread |

Each arm receives 500 optimiser updates and 1,000 sampled examples. The strict and fill policies have different available training sets and different epoch exposure. This measures data-handling policies under the same optimiser-step budget.

The original complete-input evaluation is unchanged. The additional missing-conditioning evaluation withholds every fifth selected held-out Argo date. Both trained models then use fill-enabled inference on the same 54 test windows, allowing a matched comparison of robustness by training policy. Operational loader coverage is reported separately: the strict loader retains 20 of those 54 windows, while the fill-enabled loader retains all 54.

## Results

### Prediction quality

Errors below are means across two seeds and two forecast leads, in loader-normalised SIC units.

| Held-out evaluation | Strict-trained model | Fill-trained model |
| --- | ---: | ---: |
| Complete-input MAE | 0.046647 | 0.047032 |
| Complete-input mean per-lead RMSE | 0.083301 | 0.083341 |
| Missing-input MAE, both using fill inference | 0.049352 | 0.046993 |
| Missing-input mean per-lead RMSE, both using fill inference | 0.087878 | 0.083257 |

The complete-input results reproduce the September comment to six decimal places. In this controlled missing-input extension, the fill-trained model has lower mean error. The two-seed pilot does not establish statistical significance or production forecasting skill.

### Training and complete-input timing

| Training measure, mean across two seeds | Strict policy | Fill policy |
| --- | ---: | ---: |
| Wall seconds for 500 updates | 103.707 | 100.100 |
| Process CPU seconds for 500 updates | 81.537 | 85.003 |

| Complete-input inference, fixed shared model | Strict loader | Fill loader |
| --- | ---: | ---: |
| Model-only median ms/sample | 3.391 | 3.148 |
| Loading plus inference median ms/sample | 16.181 | 15.940 |

The complete run finished successfully in 443.780 seconds with a peak parent-process RSS of 805,814,272 bytes (805.8 MB). The machine had 15 logical CPUs and 24 GiB RAM. The process used four CPU threads within a 1,200-second budget. Raw process snapshots, individual timings and matched-state hashes are included.

MAE and RMSE are reported in loader-normalised SIC units, computed over every grid cell for each forecast lead. Summary error values average the two seeds and two per-lead scores. In particular, the reported mean RMSE is the mean of four per-lead RMSE values, not an RMSE pooled across seeds and leads. Raw per-seed and per-lead values are in `current-results.json`.

## Timing interpretation

Training timing includes the loader and optimiser loop. Wall time and process CPU time are recorded separately, with process snapshots before and after each arm. Policy order is reversed for the second seed.

Inference timing uses one fixed model and complete-input windows. Every tensor produced by the strict and fill loaders is verified bit-identical. Model-only timing uses five warmups and 30 interleaved measurements; loading plus inference uses four alternating complete passes. These measurements concern complete-input overhead. Model calls receive identical tensors and weights, so small timing differences cannot establish a different model computation cost.

These are local CPU measurements with shared filesystem and operating-system activity. They are not hardware-independent throughput guarantees.

## Legacy calendar compatibility

The recovered source stores contain 855 observed dates distributed over a 1,674-day calendar, with 819 missing dates. Anemoi 0.5.45 rejects this older sparse encoding by default. Its native `ZarrWithMissingDatesFix` reader supports an in-memory index reconstruction through the process-local setting `ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL=1`.

Before training, an independent read-only verification checked the complete stored-date/missing-date partition and compared every observed value in all three sources against the adapted reader: 5,265,596,160 float values. The original and adapted streams have matching SHA-256 hashes. All stored statistics and source metadata hashes also match. No source cache was edited, no observations were invented, and no dataset copy or new download was required. See `calendar-verification.json` and the portable verifier.

Both verification and training use a proper `__main__` guard. Anemoi usage analytics was already disabled by its default setting. The runner checks that state without changing user configuration and finishes the library worker's startup before timed training.

## Reproduction

Use a checkout of CryoCast commit `393157f8152ef068bbf8d42198f0d2aac7641637` (the base of this evidence branch) or `71dd142e9c6f939e0fdc335d39a447f7f49a261b` and its installed dependencies. `environment.json` records the versions used. Place the following three existing Zarr stores in one data directory:

- `samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1.zarr`
- `samp-floatsouth-argo-25p0km-2020-2024-24h-v3.zarr`
- `samp-weathersouth-era5-25p0km-2020-2024-24h-v4.zarr`

Run from this evidence directory, using that environment's Python executable:

```bash
export PYTHONPATH=/path/to/cryocast-checkout
export ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python verify_legacy_calendar.py \
  --data /path/to/sample-stores \
  --output calendar-verification.json

python run_current_south_pilot.py \
  --data /path/to/sample-stores \
  --output current-results.json \
  --steps 500
```

The runner checks the expected 290/727 training windows and 54 common held-out windows, and requires the successful calendar verification record. It writes compact raw results and local checkpoints. Checkpoints and datasets are excluded from this package.

## Limits of the evidence

This is a fixed-budget reduced-model pilot with two seeds. It does not establish convergence, representative annual forecasting skill, or statistical significance. Metrics include land and inactive grid cells because the original all-grid design is preserved. No ocean-mask or production-skill conclusion follows from these results. The missingness pattern is controlled rather than a full model of operational sensor outages. The additional missing-input evaluation uses fill-enabled inference for both trained models; it does not imply that the strict operational loader forecasts windows it rejects.

The ten-day northern pilot in another working session uses different data and settings. Its results are not mixed with this reproduction.

## Source specimens and code checks

The runnable scripts add command-line portability, formatting, type annotations and explanatory docstrings to the recovered evidence. The exact executed runner is preserved as executed_south_pilot.py.txt, and the verbatim September comment script is preserved as historical_september_pilot.py.txt. The text specimens are archival records, not current package entry points.

The published runner has the same computational AST as the executed specimen after removing annotations and docstrings and normalising import order and equivalent import grouping. No calculation, data selection, timing loop, random seed, optimisation budget or evaluation operation was changed for publication. The portable verifier retains all original calendar, value and statistics comparisons; its data/output locations are CLI parameters.

The repository Ruff configuration is unchanged. These driver-local exceptions are explicit:

| Codes | Reason |
| --- | --- |
| S101, PT018 | Assertions, including compound invariants, are required verification checks; run without Python optimisation flags. |
| T201 | Structured progress and result output is part of the captured experiment record. |
| PLC0415 | Delayed Anemoi imports preserve the guarded multiprocessing and analytics startup sequence. |
| C901, PLR0912, PLR0915 | Preserve the executed experiment's sequential driver and timing boundaries instead of refactoring after measurement. |
| FBT001, FBT002, PLR2004 | Keep the original boolean policy arguments and explicit fixed experiment constants. |
| S607 | Fixed git and ps commands only collect local provenance; no shell or user-supplied command is executed. |

The verifier uses only PLC0415, PLR0915, PT018, S101 and T201 from this list. The hash manifest records every included file. Dataset caches, checkpoints, environments, credentials and large runtime logs are excluded.
