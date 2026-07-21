# -*- coding: utf-8 -*-
r"""
Data preprocessing 01a: DETREND + deseasonalize + STANDARDIZE (Z-score).

==============================================================================
 This is 01a, one of two preprocessing scripts. Do not confuse it with 01b.
   - 01a (this file): DETRENDS, then standardizes -> dimensionless Z-scores.
                      Feeds 02 (GPP-Ts-Ta sensitivity) and 03 (pathway
                      decomposition), which regress standardized anomalies.
   - 01b            : deseasonalizes ONLY (NO detrend, NO standardize),
                      keeping physical units. Feeds 04 (first-difference beta).
 Key difference: 01a removes the long-term trend; 01b does not.
==============================================================================

For each variable, all 8-day composites sharing the same day-of-year (DOY) are
grouped across years. Within each DOY group we (1) remove the long-term linear
trend and (2) convert the detrended values to standardized anomalies (Z-scores)
by subtracting the multi-year mean and dividing by the standard deviation of that
recurring interval. This yields the detrended, deseasonalized and standardized
anomalies used in the standardized-sensitivity and pathway analyses.

Variables processed:
    - GPP   : GOSIF GPP (primary product; vegetation-masked, per-DOY 3-sigma
              outlier rejection)
    - TSTA  : surface-air temperature difference, Ts-Ta (10:00-14:00, "noon")
    - T2M   : 2 m air temperature
    - SM    : volumetric soil water (ERA5-Land layer 2)
    - VPD   : vapour pressure deficit
Optional (used for the temporal-window robustness check):
    - TSTA_DAYTIME, TSTA_ALLDAY, TS_ALLDAY

Outputs (one GeoTIFF per timestep):
    - {PREPROC_ROOT}\{VAR}_DETRENDED   (detrended, physical units)
    - {PREPROC_ROOT}\{VAR}_ZSCORE      (detrended + deseasonalized + standardized)
"""

import gc
import os
import tempfile
import warnings
from typing import List, Tuple

import numpy as np
import rasterio
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==================== Variables to process ====================
# Edit this list to select which variables to run.
# The main analysis uses: GPP, TSTA, T2M, SM, VPD.
# The remaining Ts-Ta windows support the Supplementary Fig. 1 window comparison.
VARIABLES = ["GPP", "TSTA", "T2M", "SM", "VPD"]


# ==================== Year ranges ====================
YEAR_RANGE = {
    "GPP": range(2001, 2024),
    "TSTA": range(2001, 2024),
    "TSTA_DAYTIME": range(2001, 2024),
    "TSTA_ALLDAY": range(2001, 2024),
    "T2M": range(2001, 2024),
    "SM": range(2001, 2024),
    "VPD": range(2001, 2024),
    "TS_ALLDAY": range(2001, 2024),
}


# ==================== Input directories ====================
# Absolute paths reflect the authors' local data layout; adjust to your own
# environment before running.
VAR_DIRS = {
    "GPP": r"D:\DATA\GOSIF_GPP_8day_0.1deg",
    "TSTA": r"D:\DATA\ECMWF\8days_TS_TA_multiwindow\TS-TA_noon",
    "TSTA_DAYTIME": r"D:\DATA\ECMWF\8days_TS_TA_multiwindow\TS-TA_daytime",
    "TSTA_ALLDAY": r"D:\DATA\ECMWF\8days_TS_TA_multiwindow\TS-TA_allday",
    "T2M": r"D:\DATA\ECMWF\8days\t2m",
    "SM": r"D:\DATA\ECMWF\8days\swvl2",
    "VPD": r"D:\DATA\ECMWF\8days\vpd",
    "TS_ALLDAY": r"D:\DATA\ECMWF\8days_TS_TA_multiwindow\TS_allday",
}

VAR_FILENAME = {
    "GPP": "GOSIF_GPP_{year}{doy:03d}_Mean.tif",
    "TSTA": os.path.join("{year}", "TS_TA_{year}_{doy:03d}.tif"),
    "TSTA_DAYTIME": os.path.join("{year}", "TS_TA_{year}_{doy:03d}.tif"),
    "TSTA_ALLDAY": os.path.join("{year}", "TS_TA_{year}_{doy:03d}.tif"),
    "T2M": "T2M-{year}-{doy:03d}.tif",
    "SM": "swvl2_{year}{doy:03d}_8day.tif",
    "VPD": "VPD-{year}-{doy:03d}.tif",
    "TS_ALLDAY": os.path.join("{year}", "TS_{year}_{doy:03d}.tif"),
}

PREPROC_ROOT = r"D:\DATA\FDI_PREPROC2"
TEMP_DIR = tempfile.gettempdir()
VEG_PATH = r"C:\Users\dell\Downloads\GEE_MCD12Q1-20260304T024357Z-3-001\GEE_MCD12Q1\Vegetation_Merged\MCD12Q1_2012_0p1deg_veg.tif"
VALID_VEG_IDS = (1, 2, 3, 4, 5)


# ==================== Parameters ====================
MIN_TREND_SAMPLES = 5        # minimum valid years required to fit the linear trend
MIN_ZSCORE_SAMPLES = 3       # minimum valid years required to compute mean/std
N_JOBS = 20

# Per-DOY outlier rejection: within each DOY group, a robust (median / MAD)
# n-sigma rule removes values that deviate more than N_SIGMA standard-deviation
# equivalents from the pixel's multi-year median. The median and MAD are used
# instead of the mean and standard deviation because a few extreme spikes cannot
# inflate them and hide themselves (the failure mode of mean/std clipping).
# Applied to GPP raw values (together with the fill/negative masking) before
# detrending.
N_SIGMA = 3.0
OUTLIER_MIN_SAMPLES = 5      # minimum valid years required to estimate the median/MAD
OUTLIER_CLIP_VARS = ("GPP",)  # variables to which the outlier rule is applied


def load_vegetation_mask(shape: Tuple[int, int]) -> np.ndarray:
    with rasterio.open(VEG_PATH) as ds:
        veg = ds.read(1)
    if veg.shape != shape:
        raise ValueError(f"Vegetation shape {veg.shape} does not match data shape {shape}")
    return np.isin(veg, VALID_VEG_IDS)


def get_file_path(var: str, year: int, doy: int) -> str:
    return os.path.join(VAR_DIRS[var], VAR_FILENAME[var].format(year=year, doy=doy))


def build_time_list(var: str) -> List[Tuple[str, int, int]]:
    time_list = []
    for year in YEAR_RANGE[var]:
        for doy in range(1, 367, 8):
            fp = get_file_path(var, year, doy)
            if os.path.exists(fp):
                time_list.append((fp, year, doy))
    if not time_list:
        raise RuntimeError(f"[{var}] No valid files found.")
    return time_list


def save_raster(arr: np.ndarray, path: str, profile: dict) -> None:
    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32), 1)


def read_raster_as_float32(path: str, var: str, vegetation_mask: np.ndarray = None) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            arr[arr == ds.nodata] = np.nan
        arr[arr == 65534] = np.nan
        arr[arr == 65535] = np.nan
        if var == "GPP":
            # GPP quality control: drop negatives and restrict to vegetated
            # pixels.
            arr[arr < 0] = np.nan
            if vegetation_mask is not None:
                arr[~vegetation_mask] = np.nan
    return arr


def robust_linear_regression(x: np.ndarray, y: np.ndarray, min_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(x)
    x_centered = x - x_mean
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype(np.float64)

    slope = np.full(y.shape[1:], np.nan, dtype=np.float64)
    intercept = np.full(y.shape[1:], np.nan, dtype=np.float64)

    sum_xx = np.where(valid, x_centered[:, None, None] ** 2, 0.0).sum(axis=0)
    sum_xy = np.where(valid, x_centered[:, None, None] * y, 0.0).sum(axis=0)
    sum_y = np.where(valid, y, 0.0).sum(axis=0)

    good = (n >= min_samples) & (sum_xx > 1e-12)
    slope[good] = sum_xy[good] / sum_xx[good]
    y_mean = np.where(n > 0, sum_y / n, np.nan)
    intercept[good] = y_mean[good] - slope[good] * x_mean
    return slope, intercept, good


def reject_3sigma_doy(sub_stack: np.ndarray, n_sigma: float = N_SIGMA,
                      min_samples: int = OUTLIER_MIN_SAMPLES) -> np.ndarray:
    """Robust per-pixel outlier rejection (median / MAD) across years within one DOY.

    For each pixel, the multi-year median and median absolute deviation (MAD) of
    that DOY are computed. The MAD is scaled to a standard-deviation equivalent
    (1.4826 * MAD, exact for Gaussian data), and values with
    |value - median| > n_sigma * 1.4826 * MAD are set to NaN. Using the median and
    MAD (rather than the mean and standard deviation) makes the test insensitive to
    the very spikes being detected, which would otherwise inflate the spread and
    hide themselves. Grouping by DOY removes the seasonal cycle, so only
    within-season (interannual) outliers are affected. Pixels with fewer than
    min_samples valid years, or with MAD == 0, are left unchanged.
    """
    sub = sub_stack.astype(np.float64)
    n_valid = np.sum(np.isfinite(sub), axis=0)
    enough = n_valid >= min_samples

    median = np.full(sub.shape[1:], np.nan, dtype=np.float64)
    mad = np.full(sub.shape[1:], np.nan, dtype=np.float64)
    if np.any(enough):
        median[enough] = np.nanmedian(sub[:, enough], axis=0)
        mad[enough] = np.nanmedian(np.abs(sub[:, enough] - median[enough][None, :]), axis=0)

    robust_std = 1.4826 * mad
    valid_scale = np.isfinite(robust_std) & (robust_std > 0)
    dev = np.abs(sub - median[None, :, :])
    outlier = np.isfinite(dev) & valid_scale[None, :, :] & (dev > n_sigma * robust_std[None, :, :])

    out = sub.copy()
    out[outlier] = np.nan
    return out.astype(np.float32)


def detrend_one_doy(sub_stack: np.ndarray, years_sub: np.ndarray, min_samples: int) -> np.ndarray:
    sub = sub_stack.astype(np.float64)
    x = years_sub.astype(np.float64)
    slope, intercept, good = robust_linear_regression(x, sub, min_samples)

    trend = slope[None, :, :] * x[:, None, None] + intercept[None, :, :]
    out = np.full(sub.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(sub)

    use_trend = valid & good[None, :, :]
    out[use_trend] = sub[use_trend] - trend[use_trend]

    # Pixels with too few years to fit a trend keep their raw (valid) values.
    fallback = valid & (~good[None, :, :])
    out[fallback] = sub[fallback]
    return out.astype(np.float32)


def zscore_one_doy(sub_stack: np.ndarray, min_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = sub_stack.astype(np.float64)
    n_valid = np.sum(np.isfinite(sub), axis=0)
    sufficient = n_valid >= min_samples

    mu = np.full(sub.shape[1:], np.nan, dtype=np.float64)
    sigma = np.full(sub.shape[1:], np.nan, dtype=np.float64)

    if np.any(sufficient):
        mu[sufficient] = np.nanmean(sub[:, sufficient], axis=0)
        sigma[sufficient] = np.nanstd(sub[:, sufficient], axis=0, ddof=1)

    sigma[(sigma < 1e-6) | ~np.isfinite(sigma)] = np.nan

    zscore = np.full(sub.shape, np.nan, dtype=np.float64)
    valid_sigma = np.isfinite(sigma)
    if np.any(valid_sigma):
        zscore[:, valid_sigma] = (sub[:, valid_sigma] - mu[None, valid_sigma]) / sigma[None, valid_sigma]

    return zscore.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def process_one_doy(
    doy: int,
    stack_memmap_path: str,
    shape: Tuple[int, int, int],
    calendar_index_arr: np.ndarray,
    year_index_arr: np.ndarray,
    min_trend_samples: int,
    min_zscore_samples: int,
    apply_outlier: bool = False,
):
    stack = np.memmap(stack_memmap_path, dtype=np.float32, mode="r", shape=shape)
    idx = np.where(calendar_index_arr == doy)[0]
    sub_stack = stack[idx].copy()
    years_sub = year_index_arr[idx]
    del stack

    if apply_outlier:
        sub_stack = reject_3sigma_doy(sub_stack)

    detrended = detrend_one_doy(sub_stack, years_sub, min_trend_samples)
    zscore, mu, sigma = zscore_one_doy(detrended, min_zscore_samples)

    del sub_stack
    return doy, detrended, zscore, mu, sigma


def process_variable(var: str, n_jobs: int = -1) -> None:
    print(f"\n{'=' * 68}")
    print(f"Processing variable: {var}")
    print(f"Years: {YEAR_RANGE[var].start}-{YEAR_RANGE[var].stop - 1}")
    if var == "GPP":
        print("GPP quality control: remove values < 0 and mask non-vegetation pixels")
    if var in OUTLIER_CLIP_VARS:
        print(f"Per-DOY outlier rejection: |value - DOY mean| > {N_SIGMA} sigma")
    print(f"{'=' * 68}")

    detrended_dir = os.path.join(PREPROC_ROOT, f"{var}_DETRENDED")
    zscore_dir = os.path.join(PREPROC_ROOT, f"{var}_ZSCORE")
    os.makedirs(detrended_dir, exist_ok=True)
    os.makedirs(zscore_dir, exist_ok=True)

    time_list = build_time_list(var)
    print(f"Valid timesteps: {len(time_list)}")

    with rasterio.open(time_list[0][0]) as ds:
        profile = ds.profile.copy()
        height = ds.height
        width = ds.width
    vegetation_mask = load_vegetation_mask((height, width)) if var == "GPP" else None

    time_count = len(time_list)
    shape = (time_count, height, width)
    year_index_arr = np.array([item[1] for item in time_list], dtype=np.int32)
    calendar_index_arr = np.array([item[2] for item in time_list], dtype=np.int16)
    source_files = [item[0] for item in time_list]
    temp_mmap_path = os.path.join(TEMP_DIR, f"{var}_stack_temp.dat")

    try:
        print(f"\nStep 1/4: Build memmap stack (~{time_count * height * width * 4 / 1024 ** 3:.2f} GB)")
        stack_mmap = np.memmap(temp_mmap_path, dtype=np.float32, mode="w+", shape=shape)
        for idx, fp in enumerate(tqdm(source_files, desc=f"Reading {var}")):
            try:
                stack_mmap[idx] = read_raster_as_float32(fp, var, vegetation_mask)
            except Exception as exc:
                print(f"  Warning: failed to read {fp}: {exc}")
                stack_mmap[idx] = np.nan
        stack_mmap.flush()
        del stack_mmap
        gc.collect()

        print(f"\nStep 2/4: Detrend + Z-score (n_jobs={n_jobs if n_jobs > 0 else cpu_count()})")
        unique_doys = np.unique(calendar_index_arr)
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_one_doy)(
                doy,
                temp_mmap_path,
                shape,
                calendar_index_arr,
                year_index_arr,
                MIN_TREND_SAMPLES,
                MIN_ZSCORE_SAMPLES,
                var in OUTLIER_CLIP_VARS,
            )
            for doy in unique_doys
        )

        print("\nStep 3/4: Reassemble outputs")
        detrended_full = np.full(shape, np.nan, dtype=np.float32)
        zscore_full = np.full(shape, np.nan, dtype=np.float32)
        for doy, det_chunk, zsc_chunk, _, _ in tqdm(results, desc="Reassembling"):
            idx = np.where(calendar_index_arr == doy)[0]
            detrended_full[idx] = det_chunk
            zscore_full[idx] = zsc_chunk
        del results
        gc.collect()

        print("\nStep 4/4: Save outputs")
        for t, (_, year, doy) in enumerate(tqdm(time_list, desc=f"Saving {var}_DETRENDED")):
            out_path = os.path.join(detrended_dir, f"{var}_DETRENDED_{year}_{doy:03d}.tif")
            save_raster(detrended_full[t], out_path, profile)

        for t, (_, year, doy) in enumerate(tqdm(time_list, desc=f"Saving {var}_ZSCORE")):
            out_path = os.path.join(zscore_dir, f"{var}_ZSCORE_{year}_{doy:03d}.tif")
            save_raster(zscore_full[t], out_path, profile)

        del detrended_full, zscore_full
        gc.collect()

    finally:
        if os.path.exists(temp_mmap_path):
            os.remove(temp_mmap_path)
            print(f"Removed temp file: {temp_mmap_path}")

    print(f"[{var}] Done")
    print(f"  DETRENDED -> {detrended_dir}")
    print(f"  ZSCORE    -> {zscore_dir}")


def main():
    print("=" * 68)
    print("Preprocess: detrend by DOY across years + z-score")
    print("=" * 68)

    os.makedirs(PREPROC_ROOT, exist_ok=True)

    print(f"Variables to process: {VARIABLES}")
    for var in VARIABLES:
        process_variable(var, n_jobs=N_JOBS)

    print("\n" + "=" * 68)
    print("All done")
    print(f"Output root: {PREPROC_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
