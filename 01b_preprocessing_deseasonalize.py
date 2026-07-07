# -*- coding: utf-8 -*-
r"""
Data preprocessing 01b: deseasonalize ONLY (NO detrend, NO standardize).

==============================================================================
 This is 01b, one of two preprocessing scripts. Do not confuse it with 01a.
   - 01b (this file): deseasonalizes ONLY -> anomalies in PHYSICAL UNITS,
                      long-term variability retained. Feeds 04 (first-difference
                      beta, Fig. 5). Pre-detrending must be avoided because
                      first-differencing already removes persistent trends, and
                      beta needs physical units (g C m^-2 K^-1).
   - 01a            : DETRENDS, then standardizes -> dimensionless Z-scores,
                      for 02 (sensitivity) and 03 (pathway decomposition).
 Key difference: 01b does NOT detrend and does NOT standardize; 01a does both.
==============================================================================

For each variable, all 8-day composites sharing the same day-of-year (DOY) are
grouped across years, and the multi-year DOY mean (the seasonal climatology) is
subtracted to form anomalies. No linear trend is removed and no standardization
is applied.

Variables processed:
    - GPP  : GOSIF GPP (primary product; vegetation-masked, per-DOY 3-sigma
             outlier rejection)
    - TSTA : surface-air temperature difference, Ts-Ta (10:00-14:00, "noon")

Output (one GeoTIFF per timestep):
    - {PREPROC_ROOT}\{VAR}_ANOM        (deseasonalized only, physical units)
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
VARIABLES = ["GPP", "TSTA"]


# ==================== Year ranges ====================
YEAR_RANGE = {
    "GPP": range(2001, 2024),
    "TSTA": range(2001, 2024),
}


# ==================== Input directories ====================
# Absolute paths reflect the authors' local data layout; adjust to your own
# environment before running.
VAR_DIRS = {
    "GPP": r"D:\DATA\GOSIF_GPP_8day_0.1deg",
    "TSTA": r"D:\DATA\ECMWF\8days_TS_TA_multiwindow\TS-TA_noon",
}

VAR_FILENAME = {
    "GPP": "GOSIF_GPP_{year}{doy:03d}_Mean.tif",
    "TSTA": os.path.join("{year}", "TS_TA_{year}_{doy:03d}.tif"),
}

PREPROC_ROOT = r"D:\DATA\FDI_PREPROC_TIME"
TEMP_DIR = tempfile.gettempdir()
VEG_PATH = r"C:\Users\dell\Downloads\GEE_MCD12Q1-20260304T024357Z-3-001\GEE_MCD12Q1\Vegetation_Merged\MCD12Q1_2012_0p1deg_veg.tif"
VALID_VEG_IDS = (1, 2, 3, 4, 5)


# ==================== Parameters ====================
MIN_CLIM_SAMPLES = 5   # minimum valid years required to compute a DOY climatology
N_JOBS = 20

# Per-DOY outlier rejection: within each DOY group, a robust (median / MAD)
# n-sigma rule removes values that deviate more than N_SIGMA standard-deviation
# equivalents from the pixel's multi-year median. The median and MAD are used
# instead of the mean and standard deviation because a few extreme spikes cannot
# inflate them and hide themselves (the failure mode of mean/std clipping).
# Applied to GPP raw values (together with the fill/negative masking) before
# deseasonalizing.
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


def deseasonalize_one_doy(sub_stack: np.ndarray, min_samples: int) -> np.ndarray:
    """Subtract the multi-year DOY mean (seasonal climatology) from each year."""
    sub = sub_stack.astype(np.float64)
    n_valid = np.sum(np.isfinite(sub), axis=0)
    enough = n_valid >= min_samples

    clim_mean = np.full(sub.shape[1:], np.nan, dtype=np.float64)
    if np.any(enough):
        clim_mean[enough] = np.nanmean(sub[:, enough], axis=0)

    anomaly = sub - clim_mean
    valid = np.isfinite(sub) & np.isfinite(clim_mean)
    anomaly[~valid] = np.nan
    return anomaly.astype(np.float32)


def process_one_doy(
    doy: int,
    stack_memmap_path: str,
    shape: Tuple[int, int, int],
    calendar_index_arr: np.ndarray,
    min_clim_samples: int,
    apply_outlier: bool = False,
):
    stack = np.memmap(stack_memmap_path, dtype=np.float32, mode="r", shape=shape)
    idx = np.where(calendar_index_arr == doy)[0]
    sub_stack = stack[idx].copy()
    del stack

    if apply_outlier:
        sub_stack = reject_3sigma_doy(sub_stack)

    anomaly = deseasonalize_one_doy(sub_stack, min_clim_samples)
    del sub_stack
    return doy, idx, anomaly


def process_variable(var: str, n_jobs: int = -1) -> None:
    print(f"\n{'=' * 68}")
    print(f"Processing variable: {var}")
    print(f"Years: {YEAR_RANGE[var].start}-{YEAR_RANGE[var].stop - 1}")
    if var == "GPP":
        print("GPP quality control: remove values < 0 and mask non-vegetation pixels")
    if var in OUTLIER_CLIP_VARS:
        print(f"Per-DOY outlier rejection: |value - DOY mean| > {N_SIGMA} sigma")
    print(f"{'=' * 68}")

    anom_dir = os.path.join(PREPROC_ROOT, f"{var}_ANOM")
    os.makedirs(anom_dir, exist_ok=True)

    time_list = build_time_list(var)
    print(f"Valid timesteps: {len(time_list)}")

    with rasterio.open(time_list[0][0]) as ds:
        profile = ds.profile.copy()
        height = ds.height
        width = ds.width
    vegetation_mask = load_vegetation_mask((height, width)) if var == "GPP" else None

    time_count = len(time_list)
    shape = (time_count, height, width)
    calendar_index_arr = np.array([item[2] for item in time_list], dtype=np.int16)
    source_files = [item[0] for item in time_list]
    temp_mmap_path = os.path.join(TEMP_DIR, f"{var}_time_stack.dat")

    try:
        print(f"\nStep 1/3: Build memmap stack (~{time_count * height * width * 4 / 1024 ** 3:.2f} GB)")
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

        print(f"\nStep 2/3: Deseasonalize by DOY (n_jobs={n_jobs if n_jobs > 0 else cpu_count()})")
        unique_doys = np.unique(calendar_index_arr)
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_one_doy)(
                doy, temp_mmap_path, shape, calendar_index_arr, MIN_CLIM_SAMPLES,
                var in OUTLIER_CLIP_VARS,
            )
            for doy in unique_doys
        )

        print("\nStep 3/3: Reassemble and save anomalies")
        anomaly_full = np.full(shape, np.nan, dtype=np.float32)
        for doy, idx, anomaly_chunk in tqdm(results, desc="Reassembling"):
            anomaly_full[idx] = anomaly_chunk
        del results
        gc.collect()

        for t, (_, year, doy) in enumerate(tqdm(time_list, desc=f"Saving {var}_ANOM")):
            out_path = os.path.join(anom_dir, f"{var}_ANOM_{year}_{doy:03d}.tif")
            save_raster(anomaly_full[t], out_path, profile)

        del anomaly_full
        gc.collect()

    finally:
        if os.path.exists(temp_mmap_path):
            os.remove(temp_mmap_path)
            print(f"Removed temp file: {temp_mmap_path}")

    print(f"[{var}] Done")
    print(f"  ANOM -> {anom_dir}")


def main():
    print("=" * 68)
    print("Preprocess (b): deseasonalize by DOY only (no detrending)")
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
