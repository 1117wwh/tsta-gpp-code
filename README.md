# TSS and vegetation productivity

This code examines how vegetation productivity varies with the temperature
difference between the land surface and near-surface air (TSS). The analyses
estimate gross primary productivity (GPP) sensitivity to TSS, separate the contributions associated with
water availability and air temperature, and examine the environmental
characteristics associated with spatial differences in sensitivity. A
separate analysis estimates canopy conductance from flux-station observations.

## Analysis workflow

### 1. Prepare comparable anomalies

`01_detrend_standardize.py` removes long-term linear trends and standardizes
each variable separately for each calendar month or 8-day period. This
produces anomalies that can be compared across variables and used in the
sensitivity and pathway analyses.

### 2. Estimate GPP sensitivity to TSS

`02_sensitivity.py` regresses standardized GPP against standardized TSS at
each pixel. The regression slope describes the direction and strength of
their association. Outputs include a sensitivity map, regression uncertainty
and goodness of fit.

### 3. Separate water and thermal contributions

`03_pathway_decomposition.py` first relates vapour pressure deficit (VPD),
soil moisture (SM) and air temperature (Ta) to TSS. It then relates GPP to
these three environmental variables together. Combining the two regression
stages gives a contribution for each pathway. VPD and SM contributions are
combined as the water component, while Ta represents the thermal component.
The script compares their sum with the directly estimated GPP sensitivity
and uses year-based bootstrap resampling to estimate uncertainty. These are
statistical contributions, rather than experimental estimates of causality.

### 4. Examine spatial differences in sensitivity

`04_machine_learning.py` uses XGBoost to relate the sensitivity map to
environmental characteristics. Three models successively include climate,
vegetation, and soil and topographic predictors. Their predictive performance
is compared on the same held-out pixels. TreeSHAP describes how individual
predictors contribute to predictions from the full model.

### 5. Estimate canopy conductance

`05_canopy_conductance.py` uses half-hourly flux and meteorological
observations to estimate aerodynamic resistance and invert the
Penman-Monteith equation. It produces half-hourly and daily canopy
conductance, with daily GPP and TSS calculated from matching observations.
This station-based analysis is separate from the raster workflow. Its
outputs precede the growing-season and final station selection used in the
manuscript's figure analyses.

## Data needed

The raster analyses use aligned global 1-degree GeoTIFFs in EPSG:4326
(180 rows by 360 columns, extending from 180 degrees west to 180 degrees east
and from 90 degrees north to 90 degrees south). Prepare spatial aggregation
and data quality control before running the scripts. The default study
period is 2001-2023.

- **Anomalies, sensitivity and pathways:** GPP, TSS, VPD, SM and Ta time
  series, together with the natural-vegetation mask. The supplied workflows
  use 8-day GOSIF GPP and monthly GloFlux GPP. Script 01 prepares the 8-day
  variables and monthly TSS; standardized monthly GloFlux GPP, VPD, SM and
  Ta must be supplied separately. All variables in an analysis must cover
  matching dates and use the same standardization period.
- **Machine learning:** a prepared sensitivity map and climate, vegetation,
  soil and topographic maps. The default target is the mean monthly
  sensitivity across GOSIF GPP, GloFlux GPP, GOSIF SIF and LHSIF. Prepare this
  ensemble separately. The aridity index and root-zone water storage inputs
  use log10 and log(1 + value) transformations, respectively.
- **Canopy conductance:** half-hourly flux, meteorological and land-surface
  temperature records, with station information including measurement and
  canopy heights. Input fields are identified in script 05.

## Running the code

Each Python file is self-contained. The software environment used was
Python 3.13 with the following packages:

```bash
python -m pip install numpy==2.1.3 scipy==1.15.3 rasterio==1.4.3 pandas==2.2.3 scikit-learn==1.6.1 xgboost==1.6.2
```

Set the input paths and analysis period at the bottom of each script.
In scripts 01-03, `RUN_DATASETS` selects the variables or GPP products to
analyse; the available choices and their time resolutions appear directly
below it in `DATASETS`. Monthly and 8-day analyses require inputs at the
corresponding time resolution. Changing the study period also requires
preparing the standardized inputs for that period.

Run script 01 first, followed by scripts 02 and 03. Scripts 04 and 05 use
their own prepared inputs and can be run independently. For example:

```bash
python 01_detrend_standardize.py
```

Use the corresponding filename to run the other analyses. In script 04,
select the target map and predictors at the bottom of the file; its year
and time-resolution settings label the supplied maps, rather than changing
their contents.

## Results

Outputs are saved under `outputs/`, with a separate folder for each analysis:

| Folder | Main results |
|---|---|
| `01_standardized/` | Standardized anomaly time series |
| `02_sensitivity/` | GPP sensitivity and regression uncertainty |
| `03_pathways/` | Water and thermal contributions and their uncertainty |
| `04_machine_learning/` | Fitted models, predictions, performance and SHAP maps |
| `05_canopy_conductance/` | Half-hourly and daily conductance tables and station summaries |

Raster results retain the 1-degree input grid. Dataset names, time
resolution and study period distinguish the output folders and files.
Station results are saved as CSV tables. Existing results are protected
from replacement unless `OVERWRITE` is enabled in the script.
