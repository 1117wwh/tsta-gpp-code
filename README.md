# Analysis code

This repository contains the core analysis code used to quantify the sensitivity
of gross primary productivity (GPP) to the surface–air temperature difference
(Ts−Ta), to decompose that sensitivity into environmental pathways, and to
estimate Ts−Ta-associated GPP change. Each script is self-contained and reproduces
one stage of the analysis described in the Methods.

The scripts are written in Python and operate on gridded GeoTIFF rasters
(0.1° × 0.1°, 8-day composites, 2001–2023) and the publicly available input
datasets listed under **Input data** below.

---

## Repository contents

| Script | Stage | Manuscript |
|--------|-------|------------|
| `01a_preprocessing_detrend_zscore.py` | Detrend + deseasonalize + standardize → standardized anomalies (Z-scores) | inputs for Figs 1, 3, 4 |
| `01b_preprocessing_deseasonalize.py`  | Deseasonalize only (no detrend, physical units) → anomalies | inputs for Fig 5 |
| `02_sensitivity_gpp_tsta.py`          | Per-pixel standardized GPP sensitivity to Ts−Ta (S_std) | **Fig 1** |
| `03_pathway_decomposition.py`         | VPD / SM / T2M pathway decomposition of the sensitivity | **Figs 3, 4** |
| `04_first_difference_beta.py`         | Interannual (first-difference) sensitivity β of GPP to Ts−Ta | input for Fig 5 |
| `05_tsta_gpp_change.py`               | Ts−Ta-associated GPP change vs the 2001–2003 baseline (β × ΔTs−Ta) | **Fig 5a,b** |

There are **two preprocessing scripts** because the two families of analyses
require different anomalies:

* **01a** removes the long-term linear trend and standardizes to dimensionless
  Z-scores. Used by the standardized-sensitivity (02) and pathway (03) analyses.
* **01b** removes only the seasonal cycle and keeps physical units. Used by the
  first-difference β (04), where the trend must **not** be removed beforehand
  (first-differencing already suppresses trends) and physical units are required
  for β (g C m⁻² K⁻¹).

---

## Pipeline and execution order

```
                     ┌─────────────────────────────┐
 raw 8-day rasters ─▶│ 01a  detrend + z-score       │─▶ *_ZSCORE ─┬─▶ 02  sensitivity  (Fig 1)
 (GOSIF GPP, ERA5-   │      (GPP,TSTA,T2M,SM,VPD)    │             └─▶ 03  pathways     (Figs 3,4)
  Land, MODIS veg)   └─────────────────────────────┘
                     ┌─────────────────────────────┐
                    ▶│ 01b  deseasonalize only      │─▶ *_ANOM ─▶ 04  first-diff β ─▶ 05  GPP change (Fig 5a,b)
                     │      (GPP, TSTA)             │            (beta_TSTA.tif)
                     └─────────────────────────────┘
```

Run `01a` before `02`/`03`, and `01b` before `04` before `05`. Steps `02`, `03`
and the `04→05` chain are independent of one another.

> **Note on Fig 5.** `04` produces only the sensitivity map `beta_TSTA.tif`;
> `05` converts it into the Ts−Ta-associated GPP change relative to the 2001–2003
> baseline (the quantity plotted in Fig 5a,b). The regional extreme-event panels
> (Fig 5c–g) are spatial subsets of the per-year maps written by `05` and are not
> reproduced separately.

---

## Method summary

**Preprocessing (01a / 01b).** For each variable, all 8-day composites sharing the
same day-of-year (DOY) are grouped across years. 01a removes the per-DOY linear
trend and standardizes each recurring interval to a Z-score; 01b subtracts only
the per-DOY multi-year mean (seasonal climatology), retaining physical units. For
GPP, quality control removes fill/negative values, restricts to vegetated pixels
(MODIS IGBP classes), and applies a robust per-DOY outlier rejection: within each
pixel and DOY, values with |value − median| > 3 × 1.4826 × MAD are discarded
(median/MAD is used instead of mean/std so that extreme spikes cannot inflate the
spread and hide themselves).

**Sensitivity (02).** For each pixel, standardized GPP anomalies are regressed on
standardized Ts−Ta anomalies; the slope is the dimensionless standardized
sensitivity S_std. Multi-threshold significance masks (P < 0.1, 0.05, 0.01, 0.001)
are also produced.

**Pathway decomposition (03).** The sensitivity is partitioned into VPD, SM and
T2M pathways. For each pixel, α_E is the covariation of each mediator with Ts−Ta
(simple regression), β_E its partial effect on GPP (one multiple regression
GPP ~ VPD + SM + T2M), and the pathway contribution is C_E = α_E × β_E. The
reconstructed sensitivity is C_VPD + C_SM + C_T2M. Collinearity is screened with
variance inflation factors (VIF).

**First-difference sensitivity (04).** Annual means of GPP and Ts−Ta are
differenced between adjacent years and regressed pixel-by-pixel
(dGPP = intercept + β·dTs−Ta); β is the interannual sensitivity in g C m⁻² K⁻¹.

**Ts−Ta-associated GPP change (05).** For each year, dGPP = β × [Ts−Ta_annual −
baseline], where the baseline is the 2001–2003 mean of annual Ts−Ta. Per-pixel
maps are written and summarized as cos-latitude area-weighted means over global
vegetated land, water-limited (AI < 1) and energy-limited (AI ≥ 1) regions, with
linear trends and the 2023 total in Pg C.

---

## Input data

All input datasets are publicly available (see the manuscript Data Availability
statement):

* **GOSIF GPP** — https://globalecology.unh.edu/data/GOSIF-GPP.html
* **ERA5-Land** (skin temperature, 2 m temperature, dew point, soil water) — https://cds.climate.copernicus.eu
* **MODIS MCD12Q1** land cover (vegetation mask) — https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12q1-061
* **TerraClimate** (aridity index) — https://www.climatologylab.org/terraclimate.html

Ts−Ta is computed from ERA5-Land skin temperature and 2 m air temperature over the
10:00–14:00 local-time (midday) window and aggregated to 8-day composites; VPD is
derived from 2 m temperature and dew point. These variable-preparation steps
upstream of `01a`/`01b` follow the Methods and are not reproduced here.

---

## Requirements

Tested with:

```
python 3.13
numpy    2.1.3
scipy    1.15.3
rasterio 1.4.3
joblib   1.4.2
tqdm     4.67.1
```

Install with:

```bash
pip install numpy scipy rasterio joblib tqdm
```

---

## Usage

Each script exposes its configuration (input/output directories, parameters) in a
block at the top of the file. The absolute paths reflect the authors' local data
layout and should be edited to match your environment. After adjusting the paths,
run a script directly, e.g.:

```bash
python 01a_preprocessing_detrend_zscore.py
python 02_sensitivity_gpp_tsta.py
```

The preprocessing scripts and 03/04 use joblib for parallelism; set `N_JOBS` /
`NUM_JOBS` to the number of CPU cores to use.
