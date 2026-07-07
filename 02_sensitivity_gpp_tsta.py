# -*- coding: utf-8 -*-
r"""
Standardized GPP sensitivity to Ts-Ta (Fig. 1).

For each pixel, standardized GPP anomalies (GPP*) are regressed on standardized
Ts-Ta anomalies (TSTA*) using ordinary least squares:

    GPP* = slope * TSTA* + intercept

Because both variables are standardized (Z-scores) beforehand, the slope is the
dimensionless standardized sensitivity S_std. Negative values indicate lower GPP
under higher Ts-Ta, positive values the opposite. No filtering is applied to the
Ts-Ta range: all valid, temporally overlapping GPP*/TSTA* pairs are used.

Inputs  : standardized anomalies produced by 01_preprocessing_detrend_zscore.py
          ({preproc_root}\GPP_ZSCORE, {preproc_root}\{TSTA_VAR}_ZSCORE)
Outputs : per-pixel regression maps (slope = TS_TA_Sensitivity.tif, intercept,
          p-value, R^2, sample size, standard error, t-statistic), multi-threshold
          significance masks, and a metadata summary.
"""

import numpy as np
import rasterio
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ==================== Configuration ====================
NUM_JOBS = 12

# Input / output roots
preproc_root = r"D:\DATA\FDI_PREPROC2"
output_root = r"C:\Users\dell\Desktop\EIS"

# Ts-Ta temporal window. NOON is the main analysis; DAYTIME / ALLDAY are used for
# the temporal-window robustness check (Supplementary Fig. 1). The names map to
# the variable folders written by the preprocessing script.
TSTA_WINDOW = "NOON"
TSTA_VAR_BY_WINDOW = {
    "NOON": "TSTA",
    "DAYTIME": "TSTA_DAYTIME",
    "ALLDAY": "TSTA_ALLDAY",
}
tsta_var_name = TSTA_VAR_BY_WINDOW[TSTA_WINDOW]
tsta_zscore_dir = os.path.join(preproc_root, f"{tsta_var_name}_ZSCORE")

# GPP product (GOSIF GPP is the primary product used in the paper).
gpp_var_name = "GPP"
gpp_zscore_dir = os.path.join(preproc_root, f"{gpp_var_name}_ZSCORE")

years = range(2001, 2024)

out_dir = os.path.join(output_root, f"TS_TA_Sensitivity_{TSTA_WINDOW}")
os.makedirs(out_dir, exist_ok=True)

# Minimum number of valid time steps required to fit a pixel-level regression.
MIN_SAMPLES = 10

# P-value thresholds used to build significance masks.
P_THRESHOLDS = {
    'P10': 0.10,
    'P05': 0.05,
    'P01': 0.01,
    'P001': 0.001,
}


# ==================== Core functions ====================

def read_raster_as_float32(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        if src.nodata is not None and np.isfinite(src.nodata):
            arr[arr == src.nodata] = np.nan
        arr[arr == 65534] = np.nan
        return arr, profile


def build_time_list():
    """List (GPP*, TSTA*, year, doy) tuples for all overlapping time steps."""
    time_list = []
    for year in years:
        for doy in range(1, 367, 8):  # 8-day composites
            gpp_f = os.path.join(gpp_zscore_dir, f"{gpp_var_name}_ZSCORE_{year}_{doy:03d}.tif")
            tsta_f = os.path.join(tsta_zscore_dir, f"{tsta_var_name}_ZSCORE_{year}_{doy:03d}.tif")
            if os.path.exists(gpp_f) and os.path.exists(tsta_f):
                time_list.append((gpp_f, tsta_f, year, doy))
    return time_list


def read_stacks(time_list, H, W):
    """Read the GPP* and TSTA* stacks into memory."""
    T = len(time_list)
    print(f"Reading data: {T} time steps, grid {H}x{W}")

    gpp_stack = np.full((T, H, W), np.nan, dtype=np.float32)
    tsta_stack = np.full((T, H, W), np.nan, dtype=np.float32)

    for t, (gpp_f, tsta_f, _, _) in enumerate(tqdm(time_list, desc="Reading")):
        gpp_stack[t] = read_raster_as_float32(gpp_f)[0]
        tsta_stack[t] = read_raster_as_float32(tsta_f)[0]

    print(f"  GPP valid values : {np.isfinite(gpp_stack).sum():,}")
    print(f"  TSTA valid values: {np.isfinite(tsta_stack).sum():,}")

    return gpp_stack, tsta_stack


def _row_sensitivity_worker(row_idx, gpp_row, tsta_row):
    """Per-pixel regression GPP* ~ TSTA* for one image row (no range filtering)."""
    T, W = gpp_row.shape

    slope_row = np.full(W, np.nan, dtype=np.float32)
    intercept_row = np.full(W, np.nan, dtype=np.float32)
    pvalue_row = np.full(W, np.nan, dtype=np.float32)
    r2_row = np.full(W, np.nan, dtype=np.float32)
    n_row = np.full(W, np.nan, dtype=np.float32)
    std_err_row = np.full(W, np.nan, dtype=np.float32)   # standard error of slope
    t_stat_row = np.full(W, np.nan, dtype=np.float32)     # t-statistic

    for col in range(W):
        x = tsta_row[:, col]  # TSTA*
        y = gpp_row[:, col]   # GPP*

        valid = np.isfinite(x) & np.isfinite(y)
        n = valid.sum()
        if n < MIN_SAMPLES:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        t_stat = slope / std_err if std_err > 0 else np.nan

        slope_row[col] = slope
        intercept_row[col] = intercept
        pvalue_row[col] = p_value
        r2_row[col] = r_value ** 2
        n_row[col] = n
        std_err_row[col] = std_err
        t_stat_row[col] = t_stat

    return row_idx, slope_row, intercept_row, pvalue_row, r2_row, n_row, std_err_row, t_stat_row


def compute_sensitivity_parallel(gpp_stack, tsta_stack):
    """Compute the sensitivity maps in parallel over image rows."""
    T, H, W = gpp_stack.shape
    print(f"\nComputing Ts-Ta sensitivity (all data, min samples {MIN_SAMPLES})...")

    tasks = [
        (row_idx, gpp_stack[:, row_idx, :], tsta_stack[:, row_idx, :])
        for row_idx in range(H)
    ]

    results = Parallel(n_jobs=NUM_JOBS, backend="loky")(
        delayed(_row_sensitivity_worker)(row_idx, gpp_row, tsta_row)
        for row_idx, gpp_row, tsta_row in tqdm(tasks, desc="Computing")
    )

    slope_map = np.full((H, W), np.nan, dtype=np.float32)
    intercept_map = np.full((H, W), np.nan, dtype=np.float32)
    pvalue_map = np.full((H, W), np.nan, dtype=np.float32)
    r2_map = np.full((H, W), np.nan, dtype=np.float32)
    n_map = np.full((H, W), np.nan, dtype=np.float32)
    std_err_map = np.full((H, W), np.nan, dtype=np.float32)
    t_stat_map = np.full((H, W), np.nan, dtype=np.float32)

    for (row_idx, slope_row, intercept_row, pvalue_row,
         r2_row, n_row, std_err_row, t_stat_row) in results:
        slope_map[row_idx] = slope_row
        intercept_map[row_idx] = intercept_row
        pvalue_map[row_idx] = pvalue_row
        r2_map[row_idx] = r2_row
        n_map[row_idx] = n_row
        std_err_map[row_idx] = std_err_row
        t_stat_map[row_idx] = t_stat_row

    return slope_map, intercept_map, pvalue_map, r2_map, n_map, std_err_map, t_stat_map


def create_p_masks(pvalue_map):
    """Build significance masks at several p-value thresholds."""
    masks = {}
    for name, threshold in P_THRESHOLDS.items():
        mask = (pvalue_map < threshold).astype(np.float32)
        mask[np.isnan(pvalue_map)] = np.nan
        masks[name] = mask
        print(f"  {name} (P<{threshold}): {(mask == 1).sum():,} pixels "
              f"({(mask == 1).sum() / np.isfinite(mask).sum() * 100:.1f}%)")
    return masks


def save_raster(arr, name, profile, output_dir):
    path = os.path.join(output_dir, name)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress='lzw')
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)
    print(f"  Saved: {name}")


def save_metadata(slope_map, pvalue_map, r2_map, n_map, profile):
    """Write a JSON + text summary of the sensitivity results."""
    valid = np.isfinite(slope_map)
    valid_p = np.isfinite(pvalue_map)

    metadata = {
        "method": "Linear regression: GPP* ~ TSTA* (all data, no filtering)",
        "filter": "None (all valid TSTA* and GPP* data)",
        "min_samples": MIN_SAMPLES,
        "total_pixels": int(valid.sum()),
        "total_valid": int(np.isfinite(n_map).sum()),
        "sensitivity_stats": {
            "mean": float(np.nanmean(slope_map)),
            "std": float(np.nanstd(slope_map)),
            "median": float(np.nanmedian(slope_map)),
            "min": float(np.nanmin(slope_map)),
            "max": float(np.nanmax(slope_map)),
            "percentile_5": float(np.nanpercentile(slope_map, 5)),
            "percentile_95": float(np.nanpercentile(slope_map, 95)),
        },
        "significance_by_threshold": {
            name: {
                "count": int((pvalue_map < thresh).sum()),
                "percentage": float((pvalue_map < thresh).sum() / valid_p.sum() * 100)
            }
            for name, thresh in P_THRESHOLDS.items()
        },
        "r2_stats": {
            "mean": float(np.nanmean(r2_map)),
            "median": float(np.nanmedian(r2_map)),
            "min": float(np.nanmin(r2_map)),
            "max": float(np.nanmax(r2_map)),
        },
        "sample_size_stats": {
            "mean_n": float(np.nanmean(n_map)),
            "median_n": float(np.nanmedian(n_map)),
            "min_n": float(np.nanmin(n_map)),
            "max_n": float(np.nanmax(n_map)),
        }
    }

    json_path = os.path.join(out_dir, "sensitivity_metadata_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(out_dir, "summary_full.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Ts-Ta sensitivity analysis (all-data version)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method: {gpp_var_name}* = slope * TSTA* + intercept\n")
        f.write("Filter: none (all valid data)\n")
        f.write(f"Minimum samples: {MIN_SAMPLES}\n\n")

        f.write(f"Valid pixels: {metadata['total_pixels']:,}\n")
        f.write(f"Successful regressions: {metadata['total_valid']:,}\n\n")

        f.write("Sensitivity statistics:\n")
        for key, val in metadata['sensitivity_stats'].items():
            f.write(f"  {key}: {val:.4f}\n")

        f.write("\nSignificance (by threshold):\n")
        for name, st in metadata['significance_by_threshold'].items():
            f.write(f"  {name}: {st['count']:,} pixels ({st['percentage']:.1f}%)\n")

        f.write("\nGoodness of fit (R^2):\n")
        for key, val in metadata['r2_stats'].items():
            f.write(f"  {key}: {val:.4f}\n")

        f.write("\nSample-size statistics:\n")
        for key, val in metadata['sample_size_stats'].items():
            f.write(f"  {key}: {val:.1f}\n")

    print(f"\nMetadata summary saved: {json_path}")
    return metadata


def main():
    print("=" * 60)
    print("Ts-Ta sensitivity computation (all-data version)")
    print("=" * 60)
    print("Notes:")
    print("  - No Ts-Ta range filtering (all valid data used)")
    print("  - Multi-threshold p-value masks (P<0.1, 0.05, 0.01, 0.001)")
    print("=" * 60)

    print("\nStep 1: Build time steps...")
    time_list = build_time_list()
    if not time_list:
        raise RuntimeError("No valid data found")
    print(f"  Time steps: {len(time_list)}")

    with rasterio.open(time_list[0][0]) as ds:
        profile = ds.profile.copy()
        H, W = ds.height, ds.width
    print(f"  Grid size: {H}x{W}")

    print(f"\nStep 2: Read {gpp_var_name}* and TSTA* data...")
    gpp_stack, tsta_stack = read_stacks(time_list, H, W)

    print("\nStep 3: Compute sensitivity...")
    (slope_map, intercept_map, pvalue_map, r2_map,
     n_map, std_err_map, t_stat_map) = compute_sensitivity_parallel(gpp_stack, tsta_stack)

    print("\nStep 4: Build multi-threshold p-value masks...")
    p_masks = create_p_masks(pvalue_map)

    print(f"\nStep 5: Save results to: {out_dir}")
    save_raster(slope_map, "TS_TA_Sensitivity.tif", profile, out_dir)
    save_raster(intercept_map, "Intercept.tif", profile, out_dir)
    save_raster(pvalue_map, "P_Value.tif", profile, out_dir)
    save_raster(r2_map, "R_Squared.tif", profile, out_dir)
    save_raster(n_map, "Sample_Size.tif", profile, out_dir)
    save_raster(std_err_map, "Std_Error.tif", profile, out_dir)
    save_raster(t_stat_map, "T_Statistic.tif", profile, out_dir)

    for name, mask in p_masks.items():
        save_raster(mask, f"Mask_{name}.tif", profile, out_dir)

    # Combined significance levels (for visualization):
    # P<0.10 = 1, P<0.05 = 2, P<0.01 = 3, P<0.001 = 4
    combined_mask = np.full_like(pvalue_map, np.nan)
    combined_mask[pvalue_map < 0.10] = 1
    combined_mask[pvalue_map < 0.05] = 2
    combined_mask[pvalue_map < 0.01] = 3
    combined_mask[pvalue_map < 0.001] = 4
    save_raster(combined_mask, "Mask_Combined_Levels.tif", profile, out_dir)

    metadata = save_metadata(slope_map, pvalue_map, r2_map, n_map, profile)

    print("\n" + "=" * 60)
    print("Result summary")
    print("=" * 60)
    print(f"Valid pixels: {metadata['total_valid']:,}")
    print(f"Sensitivity mean  : {metadata['sensitivity_stats']['mean']:.4f}")
    print(f"Sensitivity median: {metadata['sensitivity_stats']['median']:.4f}")
    print(f"R^2 median        : {metadata['r2_stats']['median']:.4f}")
    print("=" * 60)
    print("Done!")
    print(f"Output directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
