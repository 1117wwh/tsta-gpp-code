# Surface-air temperature difference and vegetation productivity

This release contains the seven Python scripts used to estimate vegetation productivity sensitivity to the daytime land surface minus air temperature difference (DeltaT), quantify sensible-heat and latent-heat contributions, and project changes in GPP under two CMIP6 scenarios.

## Where to start

Run the commands below from the directory containing these scripts. Each script writes to its own numbered folder under `results/`. Runs can replace existing output files. The observational grid analyses use 2001-2023 data at 1 degree, with 8-day GOSIF GPP and SIF and monthly GloFlux GPP and LHSIF.

- **Grid sensitivity and pathways.** Scripts 02 and 03 can start directly from archived annual cross-products. To rebuild from rasters, run both modes of 01 before using the prepared-input commands below. Script 03 fits H and LE on DeltaT and five controls, then fits GPP on both fluxes and the controls. Multiplying the corresponding slopes gives the H and LE contributions.
- **FLUXNET.** Run 05 for all sites to produce the daily canopy conductance needed by the station mode of 02 or the daily-input mode of 06. Script 03 also has a station pathway analysis. Script 06 can run independently from the archived annual anomalies for the selected Figure 2 model.
- **Spatial drivers and projections.** Script 04 reads the prepared grid-cell training table. Script 07 reads prepared annual CMIP6 fields, fits model-specific sensitivities and calculates the contribution of DeltaT to projected GPP relative to 2001-2014.

## Software and data layout

Use Python with NumPy and pandas. Raster processing requires rasterio, FLUXNET conductance requires PyArrow, and the machine-learning analysis uses scikit-learn, statsmodels and XGBoost 1.6.2. This is a code release. Source observations, prepared rasters, annual matrices, station records, model inputs and reference tables must be available separately.

The scripts use the study archive layout. On another computer, update the project-relative and absolute data paths in the scripts and any paths recorded in the CSV or JSON input inventories. Keep time periods, spatial masks and variable order consistent with those inventories when reproducing the manuscript.

## Run individual analyses

### 01 Prepare Figure 1 daytime temperature inputs from unstandardized 1-degree rasters.

The temperature stage and vegetation index are set by stage and mask_indices. Other
products use the raster inventory in input_index. Requires rasterio.

```bash
python 01_detrend_and_standardize_inputs.py
python 01_detrend_and_standardize_inputs.py --other-inputs
python 01_detrend_and_standardize_inputs.py --check-only
```

**Output directory** `results/01/`

Standardized rasters and inventory CSVs. Use --verify to compare while writing, or
--check-only to compare without writing rasters. Add --other-inputs to check
productivity and control fields.

### 02 Estimate standardized productivity sensitivity to daytime DeltaT (Fig. 1 and S1).

The default grid fit uses archived yearly matrices and the vegetation pixel index.
For prepared rasters, first run script 01 both with and without --other-inputs. For
station fits, run script 05 for all sites.

```bash
python 02_estimate_productivity_sensitivity_to_delta_t.py
python 02_estimate_productivity_sensitivity_to_delta_t.py --from-rasters --prepared-root results/01
python 02_estimate_productivity_sensitivity_to_delta_t.py --stations results/05
```

**Output directory** `results/02/`

One CSV per productivity product, or stations.csv for the station fit. In grid mode
--point-only skips bootstrap P values.

### 03 Estimate sensible-heat and latent-heat pathways (Fig. 3 and S5–S8).

Use the archived annual matrices, or run both preprocessing modes of script 01 and
read H and LE from the raster inventory. Station mode uses the project daily records
directly.

```bash
python 03_decompose_sensible_and_latent_heat_pathways.py
python 03_decompose_sensible_and_latent_heat_pathways.py --from-rasters --prepared-root results/01
python 03_decompose_sensible_and_latent_heat_pathways.py --stations
```

**Output directory** `results/03/`

One CSV per GPP product, or stations.csv for the station fit. In grid mode --point-
only skips bootstrap P values.

### 04 Explain observed productivity sensitivity with XGBoost (S12 and S13).

The inputs directory must contain training_table.csv and training_manifest.json. Use
XGBoost 1.6.2 with pandas, NumPy, scikit-learn and statsmodels.

```bash
python 04_explain_productivity_sensitivity_with_xgboost_shap.py
```

**Output directory** `results/04/`

Test predictions and metrics, bootstrap scores, SHAP values and importance, and
LOWESS curves.

### 05 Calculate daytime FLUXNET canopy conductance for the selected 154 sites.

The source and archive directories provide measurement heights, site identifiers and
inventories pointing to the half-hourly records and daily GPP. Requires pandas,
NumPy and PyArrow.

```bash
python 05_estimate_fluxnet_canopy_conductance.py --site AT-Neu
python 05_estimate_fluxnet_canopy_conductance.py
```

**Output directory** `results/05/`

One daily CSV per processed site. Omit --site to process the full cohort before
using station mode in 02 or rebuilding annual inputs in 06.

### 06 Fit standardized annual FLUXNET ridge path coefficients (Fig. 2e–f).

The default input is locked_input.csv from the Figure 2 release. To rebuild site-
year inputs, run script 05 for all sites and retain the project daily station
records.

```bash
python 06_estimate_fluxnet_sem_path_coefficients.py
python 06_estimate_fluxnet_sem_path_coefficients.py --gc-folder results/05
```

**Output directory** `results/06/`

annual_inputs.csv, penalties.csv and coefficients.csv with point estimates and
bootstrap intervals.

### 07 Reproduce annual CMIP6 DeltaT sensitivities and GPP contributions (Fig. 5, S9, S14, S20–S23).

The inputs directory contains prepared annual historical and scenario NPZ files. The
selected path identifies the model selection JSON. Regional summaries and recent
trends require the project spatial and ERA5 reference files.

```bash
python 07_project_cmip6_delta_t_and_gpp_contributions.py
```

**Output directory** `results/07/`

Per-model pixel tables, annual.csv, model_endpoints.csv, ensemble_endpoints.csv,
regional.csv and recent_trends.csv.

## Numerical checks

The held-out test R-squared values for the three XGBoost models round to 0.71, 0.77 and 0.81. For the twelve-model CMIP6 ensemble, the mean DeltaT contributions to GPP in 2081-2100 relative to 2001-2014 round to -2.89% under SSP2-4.5 and -5.91% under SSP5-8.5. These endpoints are available in `results/07/ensemble_endpoints.csv`.
