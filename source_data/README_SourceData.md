# Source Data — README

**Manuscript:** Surface–air temperature difference signals water–energy constraints on vegetation carbon uptake

This folder contains the numerical source data underlying the plotted elements of the main-text and supplementary figures. Each file corresponds to one figure or set of panels, as listed below. Panels shown as spatial maps (Fig. 1a; Fig. 3a,c,e; Fig. 5c,d,f; Supplementary Fig. 1a–c, 2, 3a, 6a,c,e) are derived from the gridded input datasets listed in the manuscript **Data availability** section and are not tabulated here.

## Conventions (apply to all files)

- **Sensitivity values** (`sensitivity`, `median`, `q25`, `q75`, pathway contributions) are **standardized and therefore dimensionless**: GPP and each predictor were converted to standardized anomalies (z-scores) before regression.
- **Aridity index (AI)** is dimensionless; drylands are AI < 0.65 and humid regions AI ≥ 0.65.
- **Coordinates**: `lat` in degrees north, `lon` in degrees east.
- **Time range**: gridded GOSIF/ERA5-Land analyses cover 2001–2023 (8-day); the GloFlux robustness analyses (Supplementary Fig. 3, 6) cover 2001–2023 (monthly); FLUXNET site analyses cover 2001–2022 (limited by GLASS broadband emissivity availability); Fig. 5 time series cover 2001–2023.
- **Ts−Ta** is the midday (10:00–14:00 local time) surface–air temperature difference; gridded resolution is 0.1°.
- **Missing values**: empty cell (NaN).
- **Encoding**: UTF-8 (with BOM); comma-separated.
- FLUXNET-derived variables are reported here in the units used in the figures: H in W m⁻²; EF dimensionless; VPD in kPa; SM (volumetric soil water content) in m³ m⁻³; air temperature and Ts−Ta in °C. (Note: the raw ONEFlux/FLUXNET product reports VPD in hPa; the values here are in kPa, i.e. hPa ÷ 10.)

---

## File list

### Fig1b_area_weighted_sensitivity_distributions.csv — Fig. 1b
Area-weighted distribution of GOSIF GPP sensitivity to Ts−Ta over gridded vegetated land, by hydroclimatic regime.
| Column | Description | Unit |
|---|---|---|
| region | Drylands / Humid regions | — |
| AI_definition | aridity-index range defining the regime | — |
| n_pixels | number of 0.1° vegetated pixels | count |
| q25, median, q75 | area-weighted quartiles of standardized sensitivity | dimensionless |
| IQR | interquartile range (q75 − q25) | dimensionless |

### Fig1cd_FLUXNET_site_sensitivity.csv — Fig. 1c (map) and Fig. 1d (distributions)
Site-level GPP sensitivity to Ts−Ta at 240 FLUXNET sites (8-day aggregation).
| Column | Description | Unit |
|---|---|---|
| site_id | FLUXNET site code | — |
| lat, lon | site coordinates | °N, °E |
| AI | aridity index | dimensionless |
| regime | Drylands / Humid regions | — |
| sensitivity | standardized GPP sensitivity to Ts−Ta | dimensionless |
| n_obs | number of valid 8-day observations | count |
| n_years | number of calendar years | count |

### Fig2_FLUXNET_TsTa_percentile_bins.csv — Fig. 2a–f
Across-site medians of within-site Ts−Ta percentile-bin values (240 FLUXNET sites, midday 10:00–14:00).
| Column | Description | Unit |
|---|---|---|
| panel | figure panel (a–f) | — |
| variable_name | Ts-Ta (a), H (b), EF (c), VPD (d), SM (e), Ta (f) | — |
| unit | variable unit | degC / W m-2 / dimensionless / kPa / m3 m-3 |
| bin_label | within-site Ts−Ta percentile bin (P0-10 … P90-100) | — |
| n_sites | number of contributing sites | count |
| median | across-site median of site-level bin medians | see unit |
| ci95_low, ci95_high | 95% bootstrap confidence interval (5,000 site resamples) | see unit |

### Fig3bdf_pathway_AI_gradient.csv — Fig. 3b,d,f
Binned median pathway contributions to GPP sensitivity to Ts−Ta along the aridity gradient (GOSIF 8-day).
| Column | Description | Unit |
|---|---|---|
| panel | figure panel (b=VPD, d=SM, f=Ta) | — |
| pathway | VPD / SM / Ta | — |
| bin_index | aridity bin index | — |
| log10_ai_center, ai_center | bin centre on log10(AI) and linear AI | dimensionless |
| n_pixels | number of pixels in the bin | count |
| q25, median, q75 | quartiles of the standardized pathway contribution | dimensionless |

### Fig4b_site_observed_vs_reconstructed.csv — Fig. 4b
Observed versus reconstructed site-level sensitivity at 240 FLUXNET sites.
| Column | Description | Unit |
|---|---|---|
| site_id | FLUXNET site code | — |
| lat, lon | site coordinates | °N, °E |
| AI | aridity index | dimensionless |
| regime | Drylands / Humid regions (derived from AI) | — |
| observed_sensitivity | observed standardized sensitivity | dimensionless |
| reconstructed_sensitivity | sum of the VPD, SM and Ta pathway contributions | dimensionless |
| c_vpd, c_sm, c_ta | individual pathway contributions | dimensionless |
| n_years | number of valid observation years | count |

### Fig4_reconstruction_metrics.csv — Fig. 4a,b
Regression statistics for observed versus reconstructed sensitivity.
| Column | Description | Unit |
|---|---|---|
| scale | gridded (Fig. 4a) or site (Fig. 4b) | — |
| n | number of pixels (gridded) or sites (site) | count |
| r, p | Pearson correlation and P value | — |
| slope, intercept | linear regression parameters | — |
| rmse | root-mean-square error | dimensionless |

### Fig5a_global_TsTa_timeseries.csv — Fig. 5a
Annual global mean midday Ts−Ta over vegetated land.
| Column | Description | Unit |
|---|---|---|
| Year | 2001–2023 | year |
| global_mean_TsTa_degC | area-weighted annual mean midday Ts−Ta | °C |

### Fig5b_regional_GPP_change_timeseries.csv — Fig. 5b
Annual Ts−Ta-associated GPP change, expressed as anomalies relative to the 2001–2003 baseline mean.
| Column | Description | Unit |
|---|---|---|
| Year | 2001–2023 | year |
| global_GPP_change | global area-weighted change | g C m-2 |
| dryland_GPP_change | dryland (AI < 0.65) change | g C m-2 |
| humid_GPP_change | humid (AI ≥ 0.65) change | g C m-2 |

### Fig5eg_Australia_Europe_GPP_change_timeseries.csv — Fig. 5e (Australia) and Fig. 5g (Europe)
Annual area-weighted Ts−Ta-associated GPP change, relative to the 2001–2003 baseline mean.
| Column | Description | Unit |
|---|---|---|
| Year | 2001–2023 | year |
| australia_GPP_change | Australia area-weighted change | g C m-2 |
| europe_GPP_change | Europe area-weighted change | g C m-2 |

---

## Supplementary figures

### FigS1d_zonal_absr_profile.csv — Supplementary Fig. 1d
Zonal mean \|r\| profiles for the three averaging windows.
| Column | Description | Unit |
|---|---|---|
| window | Daytime / Noon / All-day | — |
| latitude | zonal band centre | °N |
| zonal_mean_absr | zonal mean \|r\| | dimensionless |

### FigS3b_gloflux_sensitivity_region_summary.csv — Supplementary Fig. 3b
Area-weighted median GloFlux (monthly) GPP sensitivity to Ts−Ta by regime.
| Column | Description | Unit |
|---|---|---|
| scale | data source/resolution | — |
| region | Drylands / Humid regions | — |
| n | number of pixels | count |
| median | area-weighted median standardized sensitivity | dimensionless |

### FigS4_dryland_humid_percentile_bins.csv — Supplementary Fig. 4a–f
Within-site Ts−Ta percentile-bin medians, stratified by regime (dryland n = 73, humid n = 167).
| Column | Description | Unit |
|---|---|---|
| regime | dryland / humid | — |
| panel | figure panel (a–f) | — |
| variable_name | Ts-Ta (a), H (b), EF (c), VPD (d), SM (e), Ta (f) | — |
| unit | variable unit | degC / W m-2 / dimensionless / kPa / m3 m-3 |
| bin_label | within-site Ts−Ta percentile bin (P0-10 … P90-100) | — |
| n_sites | number of contributing sites | count |
| median | across-site median of site-level bin medians | see unit |
| ci95_low, ci95_high | 95% bootstrap confidence interval (5,000 site resamples) | see unit |

### FigS5_collinearity_diagnostics.csv — Supplementary Fig. 5
Collinearity diagnostics for the pathway-decomposition predictors (panels: a = mean_r, b = mean_abs_r, c = vif).
| Column | Description | Unit |
|---|---|---|
| variable | VPD / SM / Ta | — |
| mean_r | spatial-mean correlation of the predictor with Ts−Ta | dimensionless |
| mean_abs_r | spatial-mean absolute correlation | dimensionless |
| vif | variance inflation factor in GPP ~ VPD + SM + Ta | dimensionless |

### FigS6bdf_gloflux_pathway_AI_gradient.csv — Supplementary Fig. 6b,d,f
Binned median GloFlux (monthly) pathway contributions along the aridity gradient.
| Column | Description | Unit |
|---|---|---|
| panel | figure panel (b=VPD, d=SM, f=Ta) | — |
| pathway | VPD / SM / Ta | — |
| bin_index | aridity bin index | — |
| log10_ai_center | bin centre on log10(AI) | dimensionless |
| n | number of pixels in the bin | count |
| q25, median, q75 | quartiles of the standardized pathway contribution | dimensionless |

---

## Notes
- Spatial-map panels (Fig. 1a; Fig. 3a,c,e; Fig. 5c,d,f) are not tabulated; they are reproducible from the gridded datasets in the manuscript Data availability section.
- Site metadata and per-site derived sensitivities for all 240 FLUXNET sites are also provided in **Supplementary Data 1** (separate file).
