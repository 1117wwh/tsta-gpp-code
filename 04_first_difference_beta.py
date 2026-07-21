# -*- coding: utf-8 -*-
r"""
First-difference sensitivity of GPP to Ts-Ta (beta), used for Fig. 5.

Annual means of GPP and Ts-Ta are formed for each year, differenced between
adjacent years, and regressed pixel-by-pixel:

    dGPP = intercept + beta * d(Ts-Ta)

where d denotes the year-to-year first difference. The slope beta is the
interannual sensitivity of GPP to Ts-Ta, in g C m^-2 K^-1. Negative beta means an
increase in Ts-Ta is associated with a GPP decrease. First-differencing suppresses
persistent pixel-level trends and isolates interannual covariation.

This script produces ONLY the sensitivity map (beta_TSTA.tif). Converting beta
into a Ts-Ta-associated GPP change relative to a baseline period is a separate
step and is intentionally not included here.

Inputs : deseasonalized anomalies (physical units) at 8-day resolution
         ({PREPROC_ROOT}\GPP_ANOM, {PREPROC_ROOT}\TSTA_ANOM)
Output : {OUT_DIR}\beta_TSTA.tif
"""

import gc
import os
import warnings

import numpy as np
import rasterio
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==================== Configuration ====================
PREPROC_ROOT = r"D:\DATA\FDI_PREPROC_TIME"
OUT_DIR = r"C:\Users\dell\Desktop\EIS\AnnualScale_TSTA_NOON_Sensitivity_YearDiff"

YEARS = list(range(2001, 2024))
DIFF_YEARS = list(range(2002, 2024))   # each year differenced against the previous
DOYS = list(range(1, 367, 8))          # 8-day composites

MIN_8DAY_PER_YEAR = 30   # minimum 8-day steps required to form an annual mean
MIN_DIFF_YEARS = 12      # minimum first-difference samples required per pixel
NUM_JOBS = 20
ROW_BLOCK_SIZE = 10

# GPP (GOSIF GPP product) is the response; TSTA (Ts-Ta, noon window) is the
# single predictor.
VARS = {
    "GPP": {"dir": "GPP_ANOM", "prefix": "GPP"},
    "TSTA": {"dir": "TSTA_ANOM", "prefix": "TSTA"},
}

ANNUAL_MEAN_DIR = os.path.join(OUT_DIR, "annual_means")
ANNUAL_DIFF_DIR = os.path.join(OUT_DIR, "annual_diffs")

for folder in [OUT_DIR, ANNUAL_MEAN_DIR, ANNUAL_DIFF_DIR]:
    os.makedirs(folder, exist_ok=True)


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    arr[arr == 65534] = np.nan
    arr[arr == 65535] = np.nan
    arr[np.abs(arr) > 1e30] = np.nan
    return arr, profile


def save_raster(arr, path, profile):
    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32), 1)


def var_file(var_name, year, doy):
    cfg = VARS[var_name]
    return os.path.join(PREPROC_ROOT, cfg["dir"], f"{cfg['prefix']}_ANOM_{year}_{doy:03d}.tif")


def annual_mean_path(var_name, year):
    return os.path.join(ANNUAL_MEAN_DIR, f"{var_name}_annual_mean_{year}.tif")


def annual_count_path(var_name, year):
    return os.path.join(ANNUAL_MEAN_DIR, f"{var_name}_annual_count_{year}.tif")


def annual_diff_path(var_name, year):
    return os.path.join(ANNUAL_DIFF_DIR, f"{var_name}_annual_diff_{year}.tif")


def compute_annual_mean(var_name, year):
    out_mean = annual_mean_path(var_name, year)
    out_count = annual_count_path(var_name, year)
    if os.path.exists(out_mean) and os.path.exists(out_count):
        return

    files = [var_file(var_name, year, doy) for doy in DOYS if os.path.exists(var_file(var_name, year, doy))]
    if len(files) < MIN_8DAY_PER_YEAR:
        raise RuntimeError(f"{var_name} {year}: too few 8-day files ({len(files)})")

    sample, profile = read_raster(files[0])
    sums = np.zeros(sample.shape, dtype=np.float64)
    counts = np.zeros(sample.shape, dtype=np.uint16)

    for path in files:
        arr, _ = read_raster(path)
        valid = np.isfinite(arr)
        sums[valid] += arr[valid]
        counts[valid] += 1

    mean = np.full(sample.shape, np.nan, dtype=np.float32)
    valid_count = counts > 0
    mean[valid_count] = (sums[valid_count] / counts[valid_count]).astype(np.float32)

    save_raster(mean, out_mean, profile)
    save_raster(counts.astype(np.float32), out_count, profile)


def compute_annual_diff(var_name, year):
    out_path = annual_diff_path(var_name, year)
    if os.path.exists(out_path):
        return

    curr, profile = read_raster(annual_mean_path(var_name, year))
    prev, _ = read_raster(annual_mean_path(var_name, year - 1))
    diff = curr - prev
    diff[~np.isfinite(curr) | ~np.isfinite(prev)] = np.nan
    save_raster(diff, out_path, profile)


def prepare_annual_inputs():
    print("\nStep 1/3: annual means")
    for var_name in VARS:
        for year in tqdm(YEARS, desc=f"Annual mean {var_name}"):
            compute_annual_mean(var_name, year)

    print("\nStep 2/3: year-to-year differences")
    for var_name in VARS:
        for year in tqdm(DIFF_YEARS, desc=f"Annual diff {var_name}"):
            compute_annual_diff(var_name, year)


def load_diff_stack(var_name):
    stack = []
    profile = None
    for year in DIFF_YEARS:
        arr, profile = read_raster(annual_diff_path(var_name, year))
        stack.append(arr)
    return np.stack(stack).astype(np.float32), profile


def beta_block_worker(row_start, row_stop, dgpp_block, dtsta_block):
    """OLS slope of dGPP on dTSTA for one block of image rows."""
    _, n_rows, width = dgpp_block.shape
    beta = np.full((n_rows, width), np.nan, dtype=np.float32)

    for local_row in range(n_rows):
        for col in range(width):
            y = dgpp_block[:, local_row, col]
            x = dtsta_block[:, local_row, col]
            valid = np.isfinite(y) & np.isfinite(x)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < MIN_DIFF_YEARS:
                continue

            yv = y[valid].astype(np.float64)
            xv = x[valid].astype(np.float64)
            x_reg = np.column_stack([np.ones(n_valid), xv])
            try:
                coefs, _, _, _ = np.linalg.lstsq(x_reg, yv, rcond=None)
            except Exception:
                continue
            beta[local_row, col] = coefs[1]

    return row_start, row_stop, beta


def compute_beta(dgpp, dtsta, profile):
    _, height, width = dgpp.shape
    print("\nStep 3/3: pixel-wise first-difference regression (dGPP ~ dTSTA)")
    row_blocks = [(start, min(start + ROW_BLOCK_SIZE, height)) for start in range(0, height, ROW_BLOCK_SIZE)]
    results = Parallel(n_jobs=NUM_JOBS, backend="loky")(
        delayed(beta_block_worker)(
            row_start, row_stop,
            dgpp[:, row_start:row_stop, :],
            dtsta[:, row_start:row_stop, :],
        )
        for row_start, row_stop in tqdm(row_blocks, desc="Beta regression")
    )

    beta_map = np.full((height, width), np.nan, dtype=np.float32)
    for row_start, row_stop, block in results:
        beta_map[row_start:row_stop] = block

    del results
    gc.collect()
    return beta_map


def main():
    prepare_annual_inputs()

    print("\nLoading annual-difference stacks")
    dgpp, profile = load_diff_stack("GPP")
    dtsta, _ = load_diff_stack("TSTA")

    beta_map = compute_beta(dgpp, dtsta, profile)

    out_path = os.path.join(OUT_DIR, "beta_TSTA.tif")
    save_raster(beta_map, out_path, profile)

    valid = np.isfinite(beta_map)
    print("\nDone.")
    print(f"  Valid pixels : {int(valid.sum()):,}")
    print(f"  beta mean    : {np.nanmean(beta_map):.4f} g C m^-2 K^-1")
    print(f"  beta median  : {np.nanmedian(beta_map):.4f} g C m^-2 K^-1")
    print(f"  Output       : {out_path}")


if __name__ == "__main__":
    main()
