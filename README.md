# TSS and productivity analysis

| File | Analysis | Data format |
|---|---|---|
| `01_detrend_standardize.py` | Temporal detrending and standardization | 1-degree GeoTIFF |
| `02_sensitivity.py` | Pixelwise sensitivity | 1-degree GeoTIFF |
| `03_pathway_decomposition.py` | Water and temperature pathways | 1-degree GeoTIFF |
| `04_machine_learning.py` | XGBoost and SHAP | 1-degree GeoTIFF |
| `05_canopy_conductance.py` | Station canopy conductance | Station CSV |

Each script contains its own calculation functions.
TSS is surface temperature minus air temperature; GPP is gross primary
productivity, VPD is vapour pressure deficit, SM is soil moisture, and Ta is
air temperature.

## Setup and execution

Use Python 3.13 and install the packages below:

```bash
python -m pip install numpy==2.1.3 scipy==1.15.3 rasterio==1.4.3 pandas==2.2.3 scikit-learn==1.6.1 xgboost==1.6.2
```

Set the input paths and parameters at the top of each script. The default raw
data paths use `D:/DATA`. Paths relative to `CODE_DIR` refer to the directory
containing the scripts. Set `OVERWRITE=True` to replace existing outputs.

Run 01 before 02 and 03:

```bash
python 01_detrend_standardize.py
python 02_sensitivity.py
python 03_pathway_decomposition.py
```

Scripts 04 and 05 use their own prepared inputs:

```bash
python 04_machine_learning.py
python 05_canopy_conductance.py
```

## Raster inputs

| Property | Value |
|---|---|
| Format | Single-band GeoTIFF |
| CRS | EPSG:4326 |
| Shape | 180 rows by 360 columns |
| Pixel size | 1 degree |
| Upper-left corner | Longitude -180, latitude 90 |
| Row order | North to south |
| Missing values | Declared NoData or NaN |

Prepare all raster inputs on this grid before running the scripts. Spatial
aggregation and product quality control must precede temporal standardization.
Raster outputs retain this grid and use NaN for missing values.

For each configured year, provide 46 8-day rasters with start DOY 001, 009,
..., 361, or 12 monthly rasters. Use an all-NoData raster for a date without
observations. Missing files and duplicate dates are rejected.

## 01_detrend_standardize.py

**Inputs:** raw raster series listed in `DATASETS` and
`inputs/vegetation_mask_1deg.tif`. Mask value 1 identifies retained vegetation.
The mask represents MCD12Q1 classes 1-4 with vegetation fraction at least 0.5.

The default series are 8-day GOSIF GPP, ERA5 TSS, VPD, SM and Ta, plus monthly
ERA5 TSS for 2001-2023. Within each DOY or calendar month, fit a cross-year
linear trend with an intercept, subtract the trend, and standardize the
centered residuals using the sample standard deviation (`ddof=1`).

| Dataset | Minimum samples: trend / standardization | Residual SD | When a trend cannot be fitted |
|---|---|---|---|
| GOSIF GPP, 8-day | 5 / 3 | >= 1e-6 | Center and standardize the available raw values |
| ERA5 TSS, 8-day and monthly | 2 / 2 | > 0 | Missing |
| VPD, SM and Ta, 8-day | 5 / 3 | >= 1e-6 | Center and standardize the available raw values |

Negative GOSIF GPP and fill values 65534/65535 are treated as missing.
Standardized raster outputs use float32.

**Output:** `outputs/01_standardized/<dataset>/<year>/`, with filenames such as
`GOSIF_GPP_8day_z_2001_001_1deg.tif`. The last date field is a three-digit
DOY or month; monthly `001` means January. `run_summary.json` records the
dataset settings and valid observation counts.

## 02_sensitivity.py

**Inputs:** standardized TSS and response rasters listed in `DATASETS`.
The 8-day inputs come from 01. Monthly TSS also comes from 01; monthly
GloFlux GPP is read from the standardized directory defined by `MONTHLY_DIR`.

For each pixel, fit `response = intercept + sensitivity * TSS` using at least
30 finite pairs. Sensitivity is in response standard deviations per TSS
standard deviation. Heteroskedasticity and autocorrelation consistent (HAC)
uncertainty uses Bartlett weights, 6 lags for 8-day data or 2 for monthly data,
the `n/(n-k)` correction and a Student t reference with `n-k` degrees of freedom.
Here n is the number of finite pairs and k includes the intercept.
Missing dates remain in the time axis. P values are unadjusted.

**Output:** `outputs/02_sensitivity/<dataset>/`.

| File | Value |
|---|---|
| `sensitivity_1deg.tif` | TSS regression slope |
| `intercept_1deg.tif` | Regression intercept |
| `standard_error_HAC_1deg.tif` | HAC standard error of the slope |
| `p_HAC_1deg.tif` | Two-sided HAC P value of the slope |
| `n_pairs_1deg.tif` | Number of finite TSS-response pairs |
| `r_squared_1deg.tif` | Regression R-squared |

## 03_pathway_decomposition.py

**Inputs:** standardized TSS, response, VPD, SM and Ta rasters listed in
`DATASETS`. The 8-day inputs come from 01. Monthly GloFlux GPP, VPD, SM and Ta
are read from `MONTHLY_DIR`; monthly TSS comes from 01.

Fit the following OLS regressions with intercepts on complete five-variable
observations:

```text
VPD = intercept + a_VPD * TSS
SM  = intercept + a_SM  * TSS
Ta  = intercept + a_Ta  * TSS
response = intercept + b_VPD * VPD + b_SM * SM + b_Ta * Ta

C_VPD     = a_VPD * b_VPD
C_SM      = a_SM  * b_SM
C_thermal = a_Ta  * b_Ta
C_water   = C_VPD + C_SM
S_recon   = C_water + C_thermal
```

At least five complete observations, positive TSS variation and a full-rank
environmental predictor matrix are required. `S_direct` uses all finite
TSS-response pairs. `S_direct_matched` uses the same complete observations
as the pathways, and `S_residual = S_direct_matched - S_recon`.
The coefficients describe associations among standardized anomalies.

Uncertainty uses 1,999 paired calendar-year bootstrap draws with seed
20260917. All variables use the same resampled years; standardized inputs
are held fixed. `PIXELS_PER_BATCH` controls memory use.
The centered two-sided bootstrap P value is:

```text
(1 + count(abs(bootstrap_estimate - estimate) >= abs(estimate)))
/ (1 + number_of_valid_bootstrap_estimates)
```

**Output:** `outputs/03_pathways/<dataset>/`.

| Filename suffix | Value |
|---|---|
| `_estimate_1deg.tif` | Coefficient or pathway contribution |
| `_ci_low_1deg.tif`, `_ci_high_1deg.tif` | Percentile 95% bootstrap interval |
| `_p_bootstrap_1deg.tif` | Centered bootstrap P value |
| `_valid_fraction_1deg.tif` | Fraction of finite bootstrap estimates |
| `_p_HAC_1deg.tif` | HAC P value for a regression coefficient or `S_direct` |

Metric names are listed in `METRICS`. Additional rasters contain
`S_direct_matched`, `S_residual`, `n_complete` and `n_pairs`.
Use HAC P values for regression coefficients and bootstrap P values for
pathway contributions. P values are unadjusted.

## 04_machine_learning.py

**Inputs:** `inputs/04_machine_learning/observed_sensitivity_monthly_1deg.tif`
and one `<feature>_1deg.tif` for each name in `FEATURES`. The target is the
pixelwise mean monthly sensitivity from two GPP and two SIF products.
Prepare this target separately from the single-product outputs of 02.
AI inputs contain `log10(AI)` and root-zone storage inputs contain `log1p(RZSC)`.

| Model | Predictors |
|---|---|
| M1 | VPD, SM, Ta and aridity index |
| M2 | M1 plus biomass, root-zone storage, LAI and canopy height |
| M3 | M2 plus soil organic carbon, pH, clay content and elevation |

All models use the same complete pixels and a 70:30 random pixel training/test
split with seed 42. XGBoost parameters are in `MODEL_PARAMETERS`.
TreeSHAP is calculated for M3 on test pixels. Performance intervals use
2,000 paired test-pixel bootstrap draws with seed 20260928 and fixed fitted models.

**Output:** `outputs/04_machine_learning/`.

- `sample_split_1train_2test_1deg.tif`: 1 for training pixels, 2 for test pixels.
- `excluded_missing_predictor_1deg.tif`: pixels excluded because a predictor is missing.
- `<model>/prediction_1deg.tif`: predictions for all complete pixels.
- `<model>/test_residual_observed_minus_predicted_1deg.tif`: test-pixel residuals.
- `<model>/model.json`: fitted model.
- `M3_all_predictors/SHAP/<feature>_1deg.tif` and `base_value_1deg.tif`: test-pixel SHAP values.
- `model_results.json`: model performance, confidence intervals, SHAP importance and parameters.

Residual and SHAP maps have NoData outside the test pixels. SHAP feature values
plus the base value equal the M3 prediction.

## 05_canopy_conductance.py

**Inputs:** `inputs/05_canopy_conductance/site_inputs.csv`.

| Column | Meaning |
|---|---|
| `site_id` | Unique station identifier |
| `fluxmet_file` | Full path to the half-hour FLUXMET CSV |
| `lst_file` | Full path to the half-hour LST CSV, with LST in degrees C |
| `wind_height_m` | Wind measurement height in meters |
| `canopy_height_bif_m` | BIF canopy height in meters, used as a fallback; may be missing |
| `AI` | Site aridity index |
| `vegetation_class` | Site vegetation class |

Both half-hour files use `TIMESTAMP_START` in `YYYYMMDDHHMM` format and the
same time convention. The FLUXMET fields are listed in `FLUX_COLUMNS`.

Gc is bulk canopy/surface conductance. Canopy height is the site median
estimated from dry, near-neutral records (`abs(z_m/L) <= 0.1`), after trimming
the height candidates to their 1st-99th percentile range. At least 20 candidates
are required; otherwise a physically compatible BIF canopy height is used.

```text
h   = z_m / (0.6 + 0.1 * exp(0.40 * U / u_star))
r_a = log((z_m - 0.67*h) / (0.10*h))^2 / (U * 0.40^2)
g_a = 1 / r_a
Gc  = gamma * g_a * LE
      / (delta * energy + rho * cp * VPD * g_a - LE * (delta + gamma))
```

Here `z_m` is wind measurement height, `U` is wind speed, and `u_star` is
friction velocity. Delta is the saturation vapour-pressure slope, gamma is
the psychrometric constant, rho is air density, and cp is air heat capacity.
VPD is converted from hPa to Pa and pressure from kPa to Pa.
LE and energy are in W/m2.

For each half-hour, select the first available complete energy pair:

1. `LE_CORR`, `NETRAD - G_F_MDS`.
2. `LE_F_MDS`, `NETRAD - G_F_MDS`.
3. `LE_CORR`, `LE_CORR + H_CORR`.
4. `LE_F_MDS`, `LE_F_MDS + H_F_MDS`.

Fallback occurs only when inputs are missing. Apply the physical filters to
the selected pair. Retain timestamps from 10:00 inclusive to 14:00 exclusive,
Ta above 15 degrees C, positive VPD and pressure, wind and friction velocity
above 0.1 m/s, and a dry window of 49 consecutive half-hour records including
the current record. Applicable QC flags must be at most 2, LE must be positive,
energy must exceed 50 W/m2, geometry and the PM denominator must be valid,
and `0 < Gc <= 0.05 m/s`. TSS and GPP must be finite.

**Output:** `outputs/05_canopy_conductance/`.

- `halfhour/<site_id>_Gc_halfhour.csv`: matched half-hour Gc, TSS, GPP and physical inputs.
- `daily/<site_id>_Gc_daily.csv`: daily means from at least four matched half-hours.
- `daily_Gc_all_sites.csv`: combined daily values, with within-site TSS percentile bins.
- `site_coverage_and_heights.csv`: sample counts, canopy heights and source labels.
- `energy_tier_counts.csv`: counts for each energy-input pair.
- `run_summary.json`: site and observation counts.

Gc is saved in m/s and mol/m2/s, TSS in degrees C, and GPP in micromoles CO2/m2/s.
Daily TSS, Gc and GPP use the same matched half-hours.
