# Notebook data compatibility

The legacy IceNet notebooks use CMIP-style variable names, while IceNet-MP dataset descriptors use ERA5 short names and OSI SAF variable names. The table below records the direct mapping for the variables used by the three forecast notebooks.

| Notebook name | IceNet-MP source | Meaning |
| --- | --- | --- |
| `tas` | ERA5 `2t` | 2 m air temperature |
| `zg250` | ERA5 `z` at 250 hPa | geopotential at 250 hPa |
| `zg500` | ERA5 `z` at 500 hPa | geopotential at 500 hPa |
| `uas` | ERA5 `10u` | 10 m eastward wind |
| `vas` | ERA5 `10v` | 10 m northward wind |
| `SIC` | OSI SAF `ice_conc` | sea-ice concentration |

The current southern sample descriptors already contain these fields:

- `samp_weathersouth_era5_25p0km_2020_2024_24h_v4` contains `2t`, `10u`, `10v`, and pressure-level `z` at both 250 and 500 hPa. It also provides the current time forcings `cos_julian_day`, `sin_julian_day`, and `insolation`.
- `samp_sicsouth_osisaf_25p0km_2020_2024_24h_v1` provides `ice_conc` on the 25 km EASE2 grid.

This means the raw meteorological and SIC fields used by `0_notebook_tf.ipynb`, `1_icenet_forecast_unet.ipynb`, and `2_icenet_forecast_cgan.ipynb` are represented in the pipeline dataset descriptors.

## Important difference from original IceNet inputs

The original IceNet workflow derives anomalies and other model-specific features from several raw variables. A direct short-name mapping does not imply that those derived features are reproduced automatically. When comparing against the original model, treat anomaly generation, linear-trend features, and monthly initialisation encodings as separate preprocessing/model-design choices rather than silently assuming equivalence.

For normal IceNet-MP runs, prefer the current dataset groups in `icenet_mp/config/data/` rather than copying dataset-download code out of the notebooks.
