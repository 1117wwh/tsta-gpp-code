# -*- coding: utf-8 -*-
r"""
Ts-Ta-associated GPP change relative to the 2001-2003 baseline (Fig. 5a,b).

Using the interannual sensitivity beta from 04 and the annual Ts-Ta means, the
Ts-Ta-associated GPP change in each year is

    dGPP(year) = beta * [ Ts-Ta_annual(year) - baseline_mean ]

where baseline_mean is the 2001-2003 mean of annual Ts-Ta. Per-pixel maps are
written for every year, and area-weighted (cos-latitude) means are summarized for
global vegetated land, drylands (AI < 0.65) and humid regions (AI >= 0.65),
together with linear trends and the 2023 total in Pg C. This reproduces the data
underlying Fig. 5a,b. (The regional extreme-event panels, Fig. 5c-g, are simple
spatial subsets of the same per-year maps and are not reproduced here.)

Inputs :
  - beta_TSTA.tif                         (from 04_first_difference_beta.py)
  - annual_means/TSTA_annual_mean_{year}.tif  (from 04; annual means of Ts-Ta)
  - TS_TA_Sensitivity.tif                 (from 02; defines the analysis footprint)
  - aridity index raster                  (TerraClimate AI, for the dryland/humid split)
Outputs:
  - {OUT_DIR}/TSTA_driven_GPP_change_{year}.tif   (per-year maps)
  - {OUT_DIR}/tsta_gpp_change_timeseries.csv       (global/dryland/humid series)
"""

import os
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import linregress


# ==================== Configuration ====================
# Outputs of 04_first_difference_beta.py (its OUT_DIR):
BASE = Path(r"C:\Users\dell\Desktop\EIS\AnnualScale_TSTA_NOON_Sensitivity_YearDiff")
BETA_PATH = BASE / "beta_TSTA.tif"
TSTA_ANNUAL_DIR = BASE / "annual_means"

# Sensitivity map from 02 (used only as the spatial footprint / mask):
SENS_MASK_PATH = Path(r"C:\Users\dell\Desktop\EIS\TS_TA_Sensitivity_FullData2\TS_TA_Sensitivity.tif")
# Aridity index (TerraClimate) for the dryland/humid split:
AI_PATH = Path(r"D:\DATA\FLUXNET2015\TerraClimate_AI_2001_2023_aligned_to_TS_TA_2001_001_v2.tif")

OUT_DIR = BASE / "tsta_gpp_change_baseline_2001_2003"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2001, 2024))
BASELINE_YEARS = [2001, 2002, 2003]
AI_THRESHOLD = 0.65
EARTH_R = 6371000.0


# ==================== Utilities ====================

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    arr[np.abs(arr) > 1e30] = np.nan
    return arr, profile


def save_raster(arr, path, ref_profile):
    p = ref_profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)


def reproject_to(path, ref_profile, resampling=Resampling.nearest):
    """Reproject a raster onto the reference grid (used for the sens/AI masks)."""
    dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float64)
    with rasterio.open(path) as src:
        source = src.read(1).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            source[source == src.nodata] = np.nan
        reproject(source=source, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return dst


def cos_weights(ref_profile):
    """cos(latitude) area weights on the reference grid."""
    rows = np.arange(ref_profile["height"], dtype=float)
    lats = ref_profile["transform"].f + (rows + 0.5) * ref_profile["transform"].e
    return np.cos(np.deg2rad(lats))[:, None] * np.ones((1, ref_profile["width"]))


def pixel_area_m2(ref_profile):
    """Per-pixel area (m^2) on the reference grid, for Pg C totals."""
    rows = np.arange(ref_profile["height"], dtype=float)
    lat_c = ref_profile["transform"].f + (rows + 0.5) * ref_profile["transform"].e
    dlat = abs(ref_profile["transform"].e)
    dlon = abs(ref_profile["transform"].a)
    lat_top = np.deg2rad(lat_c + dlat / 2)
    lat_bot = np.deg2rad(lat_c - dlat / 2)
    band = EARTH_R ** 2 * np.deg2rad(dlon) * (np.sin(lat_top) - np.sin(lat_bot))
    return np.abs(band)[:, None] * np.ones((1, ref_profile["width"]))


def weighted_mean(arr, mask, weights):
    v = mask & np.isfinite(arr)
    return float(np.average(arr[v], weights=weights[v])) if np.any(v) else np.nan


def main():
    # Reference grid = the beta grid (same grid as the annual Ts-Ta means, both
    # produced by 04). beta and Ts-Ta are combined element-wise by array index;
    # only the sensitivity and AI rasters are reprojected onto this grid.
    beta, ref = read_raster(BETA_PATH)
    ref_profile = ref
    W = cos_weights(ref_profile)
    area = pixel_area_m2(ref_profile)

    # Analysis footprint: valid beta pixels that also fall within the sensitivity map.
    valid_beta = np.isfinite(beta)
    sens = reproject_to(SENS_MASK_PATH, ref_profile, Resampling.nearest)
    footprint = valid_beta & np.isfinite(sens)

    # Dryland / humid split from the aridity index.
    ai = reproject_to(AI_PATH, ref_profile, Resampling.nearest)
    ai_ok = np.isfinite(ai) & (ai > 0)
    masks = {
        "Global": footprint,
        "Drylands": footprint & ai_ok & (ai < AI_THRESHOLD),
        "Humid": footprint & ai_ok & (ai >= AI_THRESHOLD),
    }
    print(f"Footprint pixels: {footprint.sum():,}  "
          f"(drylands {masks['Drylands'].sum():,}, humid {masks['Humid'].sum():,})")

    # Annual Ts-Ta means and the 2001-2003 baseline.
    tsta = {y: read_raster(TSTA_ANNUAL_DIR / f"TSTA_annual_mean_{y}.tif")[0] for y in YEARS}
    baseline = np.nanmean(np.stack([tsta[y] for y in BASELINE_YEARS]), axis=0)

    # Per-year Ts-Ta-associated GPP change: maps + area-weighted regional means.
    series = {name: [] for name in masks}
    effect_last = None
    for y in YEARS:
        effect = beta * (tsta[y] - baseline)
        effect[~footprint] = np.nan
        save_raster(effect, OUT_DIR / f"TSTA_driven_GPP_change_{y}.tif", ref_profile)
        for name, mask in masks.items():
            series[name].append(weighted_mean(effect, mask, W))
        if y == 2023:
            effect_last = effect

    series = {name: np.asarray(vals) for name, vals in series.items()}
    years = np.array(YEARS, dtype=float)

    # 2023 totals in Pg C (per region).
    print("\n=== 2023 Ts-Ta-associated GPP change ===")
    for name, mask in masks.items():
        v = mask & np.isfinite(effect_last)
        total_PgC = float(np.sum(effect_last[v] * area[v])) * 1e-15
        print(f"  {name:9s}: {series[name][-1]:+7.1f} g C m-2   total {total_PgC:+.3f} Pg C")

    # Linear trends.
    print("\n=== 2001-2023 trends ===")
    for name in ("Global", "Drylands", "Humid"):
        fit = linregress(years, series[name])
        p = "P<0.001" if fit.pvalue < 0.001 else f"P={fit.pvalue:.3f}"
        print(f"  {name:9s}: {fit.slope * 10:+.1f} g C m-2 decade-1   {p}")

    # Save the timeseries.
    csv_path = OUT_DIR / "tsta_gpp_change_timeseries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "Global_gCm2", "Drylands_gCm2", "Humid_gCm2"])
        for i, y in enumerate(YEARS):
            writer.writerow([y, round(series["Global"][i], 4),
                             round(series["Drylands"][i], 4),
                             round(series["Humid"][i], 4)])
    print(f"\nSaved per-year maps and timeseries to: {OUT_DIR}")


if __name__ == "__main__":
    main()
