"""Prepare Figure 1 daytime temperature inputs from unstandardized 1-degree rasters.

Process DeltaT and Ta separately at each 8-day period or calendar month.
Each pixel is detrended across years and scaled using its sample standard deviation.
Use --other-inputs to process the four productivity products and the atmospheric
controls with each product-specific rule and native temporal resolution.
Run with --check-only to compare every output pixel to the release without writing rasters.

Inputs
------
The temperature stage and vegetation index are set by stage and mask_indices. Other
products use the raster inventory in input_index. Requires rasterio.

Run from the directory containing this script
--------------------------------------------
    python 01_detrend_and_standardize_inputs.py
    python 01_detrend_and_standardize_inputs.py --other-inputs
    python 01_detrend_and_standardize_inputs.py --check-only

Outputs in results/01
---------------------
Standardized rasters and inventory CSVs. Use --verify to compare while writing, or
--check-only to compare without writing rasters. Add --other-inputs to check
productivity and control fields.
"""

from pathlib import Path
import argparse, re, warnings
import numpy as np
import pandas as pd
import rasterio


def detrend_and_zscore(stack: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Fit each pixel on finite years, detrend, then use sample Z-scores."""
    if stack.ndim != 3 or stack.shape[0] != years.size:
        raise ValueError(f"Expected (time, rows, cols), got {stack.shape}")
    y = stack.astype(np.float64, copy=False)
    finite = np.isfinite(y)
    n = finite.sum(axis=0).astype(np.float64)
    t = years.astype(np.float64) - float(np.mean(years))
    t3 = t[:, None, None]
    sum_y = np.where(finite, y, 0.0).sum(axis=0)
    sum_t = np.where(finite, t3, 0.0).sum(axis=0)
    sum_tt = np.where(finite, t3 * t3, 0.0).sum(axis=0)
    sum_ty = np.where(finite, t3 * y, 0.0).sum(axis=0)
    denominator = n * sum_tt - sum_t * sum_t
    good_trend = (n >= 2.0) & (denominator > 0.0)
    slope = np.zeros_like(sum_y)
    intercept = np.full_like(sum_y, np.nan)
    slope[good_trend] = (
        n[good_trend] * sum_ty[good_trend] - sum_t[good_trend] * sum_y[good_trend]
    ) / denominator[good_trend]
    intercept[good_trend] = (sum_y[good_trend] - slope[good_trend] * sum_t[good_trend]) / n[
        good_trend
    ]
    fitted = intercept[None, :, :] + slope[None, :, :] * t3
    residual = np.where(finite & good_trend[None, :, :], y - fitted, np.nan)
    n_residual = np.isfinite(residual).sum(axis=0).astype(np.float64)
    sum_residual = np.nansum(residual, axis=0)
    mean_residual = np.divide(
        sum_residual,
        n_residual,
        out=np.full_like(sum_residual, np.nan),
        where=n_residual > 0,
    )
    centered = residual - mean_residual[None, :, :]
    sum_squared = np.nansum(centered * centered, axis=0)
    variance = np.divide(
        sum_squared,
        n_residual - 1.0,
        out=np.full_like(sum_squared, np.nan),
        where=n_residual > 1,
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    use = (n_residual >= 2.0) & np.isfinite(std) & (std > 0.0)
    zscore = np.full_like(centered, np.nan)
    zscore[:, use] = centered[:, use] / std[use]
    return zscore.astype(np.float32)


def reject_outliers(stack: np.ndarray) -> tuple[np.ndarray, int]:
    """Mask temporal outliers using the scaled median absolute deviation and return the
    count removed.
    """
    valid_n = np.isfinite(stack).sum(axis=0)
    enough = valid_n >= OUTLIER_MIN_SAMPLES
    median = np.full(stack.shape[1:], np.nan)
    mad = np.full(stack.shape[1:], np.nan)
    median[enough] = np.nanmedian(stack[:, enough], axis=0)
    mad[enough] = np.nanmedian(np.abs(stack[:, enough] - median[enough][None]), axis=0)
    scale = 1.4826 * mad
    outlier = np.isfinite(stack) & np.isfinite(scale)[None] & (scale[None] > 0)
    outlier &= np.abs(stack - median[None]) > OUTLIER_N_SIGMA * scale[None]
    result = stack.copy()
    result[outlier] = np.nan
    return result, int(outlier.sum())


def detrend_zscore(stack: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Standardize across years at each pixel, retaining raw values where too few
    observations support a trend.
    """
    y = stack.astype(np.float64)
    x = years.astype(np.float64)[:, None, None]
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype(float)
    sx = np.where(valid, x, 0).sum(axis=0)
    sy = np.where(valid, y, 0).sum(axis=0)
    sxx = np.where(valid, x * x, 0).sum(axis=0)
    sxy = np.where(valid, x * y, 0).sum(axis=0)
    denominator = n * sxx - sx * sx
    good = (n >= MIN_TREND_SAMPLES) & (denominator > 1e-12)
    slope = np.full(y.shape[1:], np.nan)
    intercept = np.full_like(slope, np.nan)
    slope[good] = (n[good] * sxy[good] - sx[good] * sy[good]) / denominator[good]
    intercept[good] = (sy[good] - slope[good] * sx[good]) / n[good]
    residual = np.full_like(y, np.nan)
    fitted = intercept[None] + slope[None] * x
    residual[valid & good[None]] = (y - fitted)[valid & good[None]]
    residual[valid & ~good[None]] = y[valid & ~good[None]]
    count = np.isfinite(residual).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(residual, axis=0)
        std = np.nanstd(residual, axis=0, ddof=1)
    use = (count >= MIN_ZSCORE_SAMPLES) & np.isfinite(std) & (std >= 1e-6)
    z = np.full_like(residual, np.nan)
    z[:, use] = (residual[:, use] - mean[use]) / std[use]
    return z.astype(np.float32)


def linear_detrend(stack: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Subtract the fitted temporal trend at each pixel and keep finite values where the
    fit is unavailable.
    """
    y = stack.astype(np.float64)
    x = years.astype(np.float64)
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype(np.float64)
    xx = x[:, None, None]
    sum_x = np.where(valid, xx, 0.0).sum(axis=0)
    sum_y = np.where(valid, y, 0.0).sum(axis=0)
    sum_xx = np.where(valid, xx * xx, 0.0).sum(axis=0)
    sum_xy = np.where(valid, xx * y, 0.0).sum(axis=0)
    denominator = n * sum_xx - sum_x * sum_x
    good = (n >= MIN_TREND_SAMPLES) & (denominator > 1e-12)
    slope = np.full(y.shape[1:], np.nan, np.float64)
    intercept = np.full_like(slope, np.nan)
    slope[good] = (n[good] * sum_xy[good] - sum_x[good] * sum_y[good]) / denominator[good]
    intercept[good] = (sum_y[good] - slope[good] * sum_x[good]) / n[good]
    trend = slope[None] * x[:, None, None] + intercept[None]
    result = np.full_like(y, np.nan)
    use = valid & good[None]
    result[use] = y[use] - trend[use]
    fallback = valid & ~good[None]
    result[fallback] = y[fallback]
    return result.astype(np.float32)


def zscore(detrended: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel standardized anomalies, temporal means and sample standard
    deviations.
    """
    y = detrended.astype(np.float64)
    n = np.isfinite(y).sum(axis=0)
    enough = n >= MIN_ZSCORE_SAMPLES
    mean = np.full(y.shape[1:], np.nan, np.float64)
    std = np.full_like(mean, np.nan)
    if np.any(enough):
        mean[enough] = np.nanmean(y[:, enough], axis=0)
        std[enough] = np.nanstd(y[:, enough], axis=0, ddof=1)
    std[(std < 1e-6) | ~np.isfinite(std)] = np.nan
    out = np.full_like(y, np.nan)
    good = np.isfinite(std)
    out[:, good] = (y[:, good] - mean[None, good]) / std[None, good]
    return out.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def screen_detrend_zscore(stack, years):
    """Screen temporal outliers, fit trends with at least five valid years and return
    Z-scores and the count removed.
    """
    valid_n = np.isfinite(stack).sum(0)
    enough = valid_n >= 5
    median = np.full(stack.shape[1:], np.nan, np.float64)
    mad = np.full_like(median, np.nan)
    median[enough] = np.nanmedian(stack[:, enough], axis=0)
    mad[enough] = np.nanmedian(np.abs(stack[:, enough] - median[enough][None]), axis=0)
    scale = 1.4826 * mad
    outlier = (
        np.isfinite(stack) & (scale[None] > 0) & (np.abs(stack - median[None]) > 3 * scale[None])
    )
    y = stack.astype(np.float64)
    y[outlier] = np.nan
    x = years.astype(np.float64)[:, None, None]
    valid = np.isfinite(y)
    n = valid.sum(0).astype(float)
    sx = np.where(valid, x, 0).sum(0)
    sy = np.where(valid, y, 0).sum(0)
    sxx = np.where(valid, x * x, 0).sum(0)
    sxy = np.where(valid, x * y, 0).sum(0)
    den = n * sxx - sx * sx
    good = (n >= 5) & (den > 1e-12)
    slope = np.full(y.shape[1:], np.nan)
    intercept = np.full_like(slope, np.nan)
    slope[good] = (n[good] * sxy[good] - sx[good] * sy[good]) / den[good]
    intercept[good] = (sy[good] - slope[good] * sx[good]) / n[good]
    resid = np.where(valid & good[None], y - (intercept[None] + slope[None] * x), np.nan)
    count = np.isfinite(resid).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(resid, axis=0)
        std = np.nanstd(resid, axis=0, ddof=1)
    usable = (count >= 3) & np.isfinite(std) & (std >= 1e-6)
    z = np.full_like(resid, np.nan)
    z[:, usable] = (resid[:, usable] - mean[usable]) / std[usable]
    return z.astype(np.float32), int(outlier.sum())


def monthly_environment_zscore(stack: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Standardize monthly control residuals using the full year axis for the trend
    denominator.
    """
    data = stack.astype(np.float64, copy=False)
    x = years.astype(np.float64)
    valid = np.isfinite(data)
    n = valid.sum(axis=0)
    x_mean = float(x.mean())
    x_center = x - x_mean
    y_sum = np.nansum(np.where(valid, data, 0.0), axis=0)
    y_mean = np.divide(y_sum, n, out=np.full(n.shape, np.nan), where=n > 0)
    covariance = np.nansum(
        np.where(valid, x_center[:, None, None] * (data - y_mean[None]), 0.0), axis=0
    )
    denominator = np.sum(x_center**2)
    slope = np.divide(
        covariance,
        denominator,
        out=np.zeros(y_mean.shape, dtype=np.float64),
        where=(n >= 2),
    )
    fitted = y_mean[None] + slope[None] * x_center[:, None, None]
    residual = np.where(valid, data - fitted, np.nan)
    # With fewer than two observations, retain no standardized value.  This
    # is conservative and never invents a temporal value.
    residual = np.where(n[None] >= 2, residual, np.nan)
    count = np.isfinite(residual).sum(axis=0)
    mean = np.nanmean(residual, axis=0)
    std = np.nanstd(residual, axis=0, ddof=1)
    zscore = (residual - mean[None]) / std[None]
    zscore[(count[None] < 3) | ~np.isfinite(zscore) | (std[None] <= 1e-12)] = np.nan
    return zscore.astype(np.float32)


def other_figure1_inputs():
    """Use the published preprocessing rule for each productivity or control product."""
    index = pd.read_csv(input_index)
    index = index[~index.variable.isin(["DeltaT", "Ta"])]
    reports = []
    if options.scale:
        index = index[index.scale.eq(options.scale)]
    vegetation = np.load(mask_indices)
    for (scale, variable, period), group in index.groupby(["scale", "variable", "period"]):
        # Monthly radiation and precipitation archives retain their full native land domain.
        flat = (
            np.arange(180 * 360)
            if scale == "monthly" and variable in ("TP", "SSRD")
            else vegetation
        )
        if period == 1:
            print(scale, variable, flush=True)
        group = group.sort_values("year")
        np.testing.assert_array_equal(group.year, years)
        arrays = []
        for row in group.itertuples():
            reference = Path(row.path)
            directory = Path(str(reference).replace("standardized", "raw")).parent
            slot = f"{period:03d}" if scale == "8day" else f"{period:02d}"
            # Products encode YYYY_DDD, YYYYDDD, YYYY_MM or YYYYMM.
            tokens = [f"{row.year}_{slot}", f"{row.year}{slot}"]
            matches = [
                p
                for p in directory.glob("*.tif")
                if any(re.search(token + r"(?:_|\.|-)", p.name) for token in tokens)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{variable} {row.year} {period}: expected one raw raster in {directory}"
                )
            with rasterio.open(matches[0]) as image:
                profile = image.profile.copy()
                arrays.append(
                    image.read(1, masked=True).astype("float32").filled(np.nan).ravel()[flat]
                )
        data = np.stack(arrays)[:, :, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if variable in ["GOSIF_GPP", "GOSIF_SIF"]:
                clean, _ = reject_outliers(data)
                z = detrend_zscore(clean, years)
            elif variable in ["GLOFLUX_GPP", "LHSIF_SIF"]:
                z, _ = screen_detrend_zscore(data, years)
            elif variable == "CO2":
                z = detrend_and_zscore(data, years)
            elif scale == "monthly" and variable == "VPD":
                z = monthly_environment_zscore(data, years)
            else:
                if scale == "8day" and variable in ["TP", "SSRD"]:
                    upper = 2 if variable == "TP" else 1e9
                    data[(data < 0) | (data > upper)] = np.nan
                z = zscore(linear_detrend(data, years))[0]
        for row, array in zip(group.itertuples(), z):
            reference = Path(row.path)
            values = np.full((180, 360), np.nan, dtype="float32")
            values.ravel()[flat] = array.ravel()
            max_error = np.nan
            if verify:
                with rasterio.open(reference) as image:
                    expected = image.read(1, masked=True).astype("float32").filled(np.nan)
                    np.testing.assert_array_equal(
                        np.isfinite(values), np.isfinite(expected), err_msg=str(reference)
                    )
                    np.testing.assert_allclose(
                        values, expected, rtol=0, atol=2e-5, equal_nan=True, err_msg=str(reference)
                    )
                    max_error = np.nanmax(np.abs(values - expected))
            if save:
                suffix = reference.parts[reference.parts.index("standardized") + 1 :]
                dest = output.joinpath(scale, *suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                profile.update(dtype="float32", nodata=np.nan)
                with rasterio.open(dest, "w", **profile) as raster:
                    raster.write(values, 1)
            reports.append((scale, variable, period, row.year, max_error))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports, columns=["scale", "variable", "period", "year", "max_abs_error"]).to_csv(
        output / f"product_inventory_{options.scale or 'all'}.csv", index=False
    )


def main():
    """Prepare temperature rasters for each temporal scale and write the processing
    inventory.
    """
    vegetation = np.zeros((180, 360), dtype=bool)
    vegetation.ravel()[np.load(mask_indices)] = True
    reports = []
    for scale in ["8day", "monthly"]:
        for variable, source in [("DeltaT", "ERA5Land_SKT_minus_ERA5Land_T2M"), ("Ta", "ERA5Land")]:
            folder = stage / scale / "raw" / variable / source / "daytime_LMST_06-18"
            for first in sorted((folder / "2001").glob("*.tif")):
                name = first.name
                arrays = []
                for year in years:
                    path = folder / str(year) / name.replace("_2001_", f"_{year}_")
                    with rasterio.open(path) as dataset:
                        arrays.append(dataset.read(1))
                        profile = dataset.profile.copy()
                standardized = detrend_and_zscore(np.stack(arrays), years)
                standardized[:, ~vegetation] = np.nan
                for year, values in zip(years, standardized):
                    filename = name.replace("_2001_", f"_{year}_").replace(
                        "_1deg_K", "_1deg_zscore"
                    )
                    pair = Path(variable) / source / "daytime_LMST_06-18" / str(year) / filename
                    if verify:
                        with rasterio.open(stage / scale / "standardized" / pair) as reference:
                            np.testing.assert_equal(values, reference.read(1))
                    if save:
                        path = output / scale / pair
                        path.parent.mkdir(parents=True, exist_ok=True)
                        profile.update(dtype="float32", nodata=np.nan, compress="deflate")
                        with rasterio.open(path, "w", **profile) as dest:
                            dest.write(values, 1)
                    reports.append(
                        (scale, variable, year, filename, int(np.isfinite(values).sum()))
                    )
    if len(reports) != 2 * (46 + 12) * len(years):
        raise ValueError("Incomplete input raster inventory")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports, columns=["scale", "variable", "year", "file", "valid_pixels"]).to_csv(
        output / "inventory.csv", index=False
    )


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    years = np.arange(2001, 2024)
    stage = Path("D:/DATA/DeltaT_GPP/metadata/temperature_rows20_stage")
    mask_indices = folder.parent / "main_figures/monthly_tp_ssrd_20260917/source_data/flat.npy"
    output = folder / "results/01"
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--other-inputs",
        action="store_true",
        help="Process Fig. 1 productivity, CO2, VPD, precipitation and radiation inputs",
    )
    input_index = Path(
        "D:/DATA/DeltaT_GPP_Analysis/Results/Temporary/20261024_F01_daytime_VPD_ridge_native/input_index.csv"
    )
    MIN_TREND_SAMPLES, MIN_ZSCORE_SAMPLES = 5, 3
    OUTLIER_MIN_SAMPLES, OUTLIER_N_SIGMA = 5, 3.0
    parser.add_argument(
        "--scale", choices=["8day", "monthly"], help="Optional subset for --other-inputs"
    )
    options = parser.parse_args()
    verify, save = options.verify or options.check_only, not options.check_only
    if options.other_inputs:
        other_figure1_inputs()
    else:
        main()
