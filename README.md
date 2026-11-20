# Surface–air temperature difference and vegetation productivity

ΔT is land surface temperature minus near-surface air temperature. The observational regressions use daytime ΔT. The CMIP6 projections use annual, all-day ΔT.

## Running the analyses

Install the Python dependencies used for these analyses.

```sh
python -m pip install numpy==2.1.3 pandas==2.2.3 rasterio==1.4.3 pyarrow==19.0.0 scikit-learn==1.6.1 scipy==1.15.3 statsmodels==0.14.4 xgboost==1.6.2
```

For each script you plan to run, create a UTF-8 JSON file beside it named `NN_data_paths.txt`, where `NN` is the script number (01–07). Use the input keys shown in the script’s `if __name__ == "__main__":` block and set `OUTPUT_DIR` to your chosen results directory. Paths can be absolute or relative to the TXT file. For example, `04_data_paths.txt` can contain

```json
{
  "TRAINING_TABLE_PATH": "data/training_table.csv",
  "OUTPUT_DIR": "results/04"
}
```

In 01, `RAW_8DAY_FILES` and `RAW_MONTHLY_FILES` map each raster variable to its filename pattern. Use `{year}` for the year and `{period:03d}` (8-day starting day) or `{period:02d}` (month) for the period. Both dictionaries use `DeltaT`, `Ta`, `TP`, `SSRD` and `VPD`. The 8-day dictionary also uses `GOSIF_GPP` and `GOSIF_SIF`. The monthly dictionary also uses `GLOFLUX_GPP` and `LHSIF_SIF`. Supply CO₂ separately through `CO2_SOURCE_PATH`.

Run 01 and then 02 for the observational grid sensitivities. Set `OUTPUT_DIR` in 01 and `STANDARDIZED_RASTER_DIR` in 02 to the same folder, for example `results/01_standardized`. The FLUXNET part of 02 and scripts 03–07 read the prepared inputs specified in their own path files.

| Script | What it reads and calculates | What it writes |
|---|---|---|
| 01 | Reads the 1° observations for 2001–2023 and the NOAA CO₂ latitude–time table. Screens the four productivity products for temporal outliers. Removes each cell’s linear trend across years separately for each 8-day period or month. Divides the centred residuals by their sample standard deviation and applies the vegetation mask. | A global 180 × 360 GeoTIFF per variable and period. |
| 02 | Reads the rasters written by 01. Fits each productivity response on ΔT, air temperature, precipitation, solar radiation, CO₂ and vapour pressure deficit. Fits daily GPP and canopy conductance separately at each FLUXNET site from the corresponding prepared station tables. | Four grid tables and one station table, with the ΔT coefficient, sample size, intercept and selected ridge penalty. Grid tables also contain the whole-year bootstrap P value. |
| 03 | Reads two paper inventories of standardized rasters, including sensible and latent heat flux, and the complete daily station anomaly tables. Fits the two flux responses to ΔT and the partial GPP responses to the fluxes on dates shared by all equations. | Grid and station tables of slopes, the two pathway products, their sum and the directly fitted sensitivity. |
| 04 | Reads the prepared predictor table and the mean of the four **monthly** product sensitivities. Fits three XGBoost models with 4, 8 and 12 predictors using the original training/test split. | Predictions, held-out scores, bootstrap scores, TreeSHAP values, feature importance and dependence curves. |
| 05 | Uses half-hourly FLUXNET flux and meteorology files, matched surface temperatures, site heights, local-solar-time offsets and archived growing-season activity flags. Inverts Penman–Monteith at each interval. Forms a daily mean on active days with at least four eligible daytime intervals. | One CSV per station containing daily ΔT (°C), canopy conductance (m s⁻¹), GPP (µmol m⁻² s⁻¹), the interval count and the dominant energy-input tier. |
| 06 | Reads the prepared FLUXNET site-year anomaly table. Fits the Figure 2 equations with five controls and equal total weight per site, including H → Gc and Gc → LE. Cross-resamples sites and years for interval estimates. | Ridge penalties, standardized coefficients, five path products, their sum, equation R² and intervals. |
| 07 | Reads prepared annual fields for 12 CMIP6 models under SSP2-4.5 and SSP5-8.5. Estimates pixel-level ΔT sensitivity and uses the annual departure from the model’s 2001–2014 baseline to calculate its contribution to GPP. | Pixel sensitivities, global and regional annual series, model and ensemble endpoints, and recent temperature trends. |

Run each script with Python, for example `python 01_detrend_and_standardize_inputs.py`. The ridge regressions include an unpenalized intercept. Resampling uses 2,000 draws in 02, 03, 04 and 06.

## Observational grids

GOSIF GPP and GOSIF SIF use 8-day inputs. GloFlux GPP and LHSIF use monthly inputs. For each productivity series with at least five finite years at the same calendar position, 01 excludes values more than three scaled median absolute deviations from the median (scale factor 1.4826, positive dispersion required). The 8-day precipitation and radiation inputs also retain their archived physical range checks.

For CO₂, 01 integrates the original NOAA time series over each 8-day interval or calendar month, interpolates the 41 sine-latitude nodes to the 180 latitude centres and repeats each latitude value across 360 longitudes. The station regressions use site CO₂ measurements.

| Time scale | Output path under `OUTPUT_DIR` | Example filename |
|---|---|---|
| 8-day | `8day/<variable>/` | `CO2_2001_DOY009_zscore.tif` |
| Monthly | `monthly/<variable>/` | `CO2_2001_M02_zscore.tif` |

`DOY` is the starting day of the 8-day period (001, 009, …, 361). `M` is the calendar month (01–12). Output values are dimensionless standardized anomalies on the EPSG:4326 grid. Missing values and cells outside the vegetation mask are NaN.

## Output variables

| Field | Quantity |
|---|---|
| `S_std` | Standardized sensitivity to ΔT for the named response. The station table in 02 identifies the response and reports its coefficient in `estimate`. |
| `a_H`, `a_LE`, `b_H`, `b_LE` | Standardized flux responses to ΔT and partial GPP responses to sensible and latent heat flux. |
| `C_H`, `C_LE`, `S_recon` | The products a_H × b_H and a_LE × b_LE, and their sum. |
| `delta_t_c`, `gc_m_s` | Physical daytime ΔT in °C and canopy conductance in m s⁻¹ in 05. |
| `DeltaT_anomaly_2001_2014_C` | Annual all-day ΔT minus the corresponding 2001–2014 mean, in °C. |
| `DeltaT_anomaly_2081_2100_vs_2001_2014_C` | Pixel mean ΔT in 2081–2100 minus its 2001–2014 mean, in °C. |
| `S_phys_gC_m2_yr_K`, `GPP_contribution_pct` | Physical GPP sensitivity in g C m⁻² yr⁻¹ K⁻¹ and area-weighted contribution as a percentage of baseline GPP. |

