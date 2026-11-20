"""Estimate GPP sensitivity to all-day annual CMIP6 ΔT and its contribution relative to 2001–2014."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


PATHS_FILE = Path(__file__).resolve().parent / "07_data_paths.txt"


def solve_ridge(moment, predictors, response, penalty):
    """Fit slopes from cross-products, leaving the intercept unpenalized."""
    count = moment[..., 0, 0]
    sums = moment[..., predictors, 0]
    gram = (
        moment[..., predictors, :][..., :, predictors]
        - sums[..., None] * sums[..., None, :] / count[..., None, None]
    )
    cross_covariance = (
        moment[..., predictors, response] - sums * (moment[..., 0, response] / count)[..., None]
    )
    system = gram + (count * penalty)[..., None, None] * np.eye(len(predictors))
    if np.isscalar(penalty) and penalty == 0:
        valid = np.linalg.matrix_rank(gram) == len(predictors)
        slopes = np.full(cross_covariance.shape, np.nan)
        slopes[valid] = np.linalg.solve(system[valid], cross_covariance[valid, ..., None])[..., 0]
    else:
        slopes = np.linalg.solve(system, cross_covariance[..., None])[..., 0]
    intercept = (moment[..., 0, response] - (slopes * sums).sum(-1)) / count
    return np.concatenate([intercept[..., None], slopes], axis=-1)


def detrend_and_zscore(stack: np.ndarray, years: np.ndarray, detrend=True):
    """Return Z-scores, scales, centers and slopes. Set detrend=False to standardize with
    trends retained.
    """
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
    if not detrend:
        slope[good_trend] = 0.0
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
    return zscore, std, intercept + mean_residual, slope


def standardized(raw, training, detrend=True):
    """Fit trends and scales on training years, then apply them to all years at each
    pixel.
    """
    z = np.full_like(raw, np.nan)
    scale = np.empty(raw.shape[1:])
    for column in range(raw.shape[2]):
        fitted, sd, center, slope = detrend_and_zscore(
            raw[training, None, :, column], training.astype(float), detrend
        )
        scale[:, column] = sd[0]
        time = np.arange(len(raw)) - training.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            z[:, :, column] = (raw[:, :, column] - center - slope * time[:, None]) / sd
        z[training, :, column] = fitted[:, 0]
    return z, scale


def fitted_domain(raw, detrend):
    """Identify pixels usable in every validation fold with at least 30 complete years in
    the full fit.
    """
    years = np.arange(len(raw))
    valid = np.ones(raw.shape[1], dtype=bool)
    for fold in np.array_split(years, 5):
        train = np.setdiff1d(years, fold)
        z, _ = standardized(raw, train, detrend)
        complete = np.isfinite(z).all(axis=-1)
        valid &= (complete[train].sum(0) > 1) & complete[fold].any(0)
    z, _ = standardized(raw, years, detrend)
    return valid & (np.isfinite(z).all(axis=-1).sum(0) >= 30)


def cross_products(z):
    """Sum per-pixel design cross-products over years with complete observations,
    including an intercept column.
    """
    valid = np.isfinite(z).all(-1)
    design = np.concatenate([np.ones((*z.shape[:2], 1)), z], axis=-1)
    design = np.where(valid[..., None], design, 0)
    return np.einsum("tpi,tpj->pij", design, design)


def annual_fit(raw):
    """Choose the penalty by year-block validation, then convert the ΔT slope to g C m⁻² yr⁻¹ K⁻¹."""
    predictors = [1, 3, 4, 5, 6, 7]  # DeltaT, Ta, TP, SSRD, CO2, VPD
    years = np.arange(len(raw))
    grid = np.r_[0, np.logspace(-5, 2, 29)]
    score = np.zeros((len(grid), raw.shape[1]))
    eligible = np.ones(raw.shape[1], dtype=bool)
    for fold in np.array_split(years, 5):
        train = np.setdiff1d(years, fold)
        z, _ = standardized(raw, train)
        training_moments, test_moments = cross_products(z[train]), cross_products(z[fold])
        good = (training_moments[:, 0, 0] > 1) & (test_moments[:, 0, 0] > 0)
        eligible &= good
        columns = [0] + predictors
        test = test_moments[good]
        gram = test[:, columns, :][:, :, columns]
        for i, penalty in enumerate(grid):
            coef = solve_ridge(training_moments[good], predictors, 2, penalty)
            squared_error = (
                test[:, 2, 2]
                - 2 * (coef * test[:, columns, 2]).sum(-1)
                + np.einsum("pi,pij,pj->p", coef, gram, coef)
            )
            score[i, good] += np.where(np.isfinite(squared_error), squared_error, np.inf)
    z, sd = standardized(raw, years)
    full = cross_products(z)
    eligible &= (full[:, 0, 0] >= 30) & np.isfinite(score).any(0)
    penalty = grid[score.argmin(0)]
    slopes = solve_ridge(full[eligible], predictors, 2, penalty[eligible])[:, 1]
    physical = slopes * sd[eligible, 1] / sd[eligible, 0]
    return eligible, slopes, physical


def area_totals(delta_t_anomaly, historical_gpp, physical_slope, pixels):
    """Sum ΔT anomalies and their GPP contributions with cosine-latitude weights."""
    weight = np.cos(np.deg2rad(89.5 - pixels // 360))
    totals = np.empty((len(delta_t_anomaly), 4))
    totals[:, 0] = delta_t_anomaly @ weight
    totals[:, 1] = (delta_t_anomaly * physical_slope) @ weight
    totals[:, 2] = weight @ historical_gpp
    totals[:, 3] = weight.sum()
    return totals


def model_projections(model, inputs, aridity, output):
    """Use the same eligible cells in the historical period and both scenarios."""
    scenarios = ["ssp245", "ssp585"]
    groups = ["Global", "Water_limited", "Energy_limited"]
    totals = {(scenario, group): np.zeros((100, 4)) for scenario in scenarios for group in groups}
    pixel_tables = {scenario: [] for scenario in scenarios}
    for path in sorted(inputs.glob(model + "_historical_*.npz")):
        with np.load(path) as historical:
            flat = historical["flat"]
            history = historical["raw"]
        # Raw columns: ΔT, GPP, Ta, TP, SSRD, CO2 and VPD. Rows are calendar years.
        series = {}
        for scenario in scenarios:
            future_path = path.with_name(path.name.replace("_historical_", f"_{scenario}_"))
            with np.load(future_path) as future:
                if not np.array_equal(flat, future["flat"]):
                    raise ValueError(f"Different historical and scenario grids in {future_path}")
                if history.shape != (14, len(flat), 7) or future["raw"].shape != (86, len(flat), 7):
                    raise ValueError(
                        f"Expected 14 historical and 86 scenario years in {future_path}"
                    )
                series[scenario] = np.concatenate([history, future["raw"]])
        complete = np.isfinite(series["ssp245"]).all(axis=(0, 2)) & np.isfinite(
            series["ssp585"]
        ).all(axis=(0, 2))
        if not complete.any():
            continue
        # Membership is common to both scenarios and to fits with and without detrending.
        shared = np.ones(complete.sum(), dtype=bool)
        for data in series.values():
            for detrend in (False, True):
                shared &= fitted_domain(data[:, complete], detrend)
        pixels = flat[complete][shared]
        if len(pixels) == 0:
            continue
        for scenario, data in series.items():
            raw = data[:, complete][:, shared]
            eligible, standardized_slope, physical_slope = annual_fit(raw)
            raw = raw[:, eligible]
            fitted_pixels = pixels[eligible]
            baseline = raw[:14].mean(axis=0)
            delta_t_anomaly = raw[:, :, 0] - baseline[:, 0]
            pixel_tables[scenario].append(
                pd.DataFrame(
                    {
                        "flat": fitted_pixels,
                        "S_std": standardized_slope,
                        "S_phys_gC_m2_yr_K": physical_slope,
                        "DeltaT_anomaly_2081_2100_vs_2001_2014_C": delta_t_anomaly[-20:].mean(
                            axis=0
                        ),
                    }
                )
            )
            totals[scenario, "Global"] += area_totals(
                delta_t_anomaly, baseline[:, 1], physical_slope, fitted_pixels
            )
            for group, mask in [
                ("Water_limited", aridity[fitted_pixels] < 1),
                ("Energy_limited", aridity[fitted_pixels] >= 1),
            ]:
                totals[scenario, group] += area_totals(
                    delta_t_anomaly[:, mask],
                    baseline[mask, 1],
                    physical_slope[mask],
                    fitted_pixels[mask],
                )

    global_series, regional_series = [], []
    for scenario in scenarios:
        pixel_table = pd.concat(pixel_tables[scenario], ignore_index=True)
        pixel_table.to_csv(output / f"{model}_{scenario}_pixels.csv", index=False)
        for group in groups:
            temperature_sum, contribution_sum, gpp_sum, area = totals[scenario, group].T
            series = pd.DataFrame(
                {
                    "model": model,
                    "scenario": scenario,
                    "year": np.arange(2001, 2101),
                    "DeltaT_anomaly_2001_2014_C": temperature_sum / area,
                    "GPP_baseline_2001_2014_gC_m2_yr": gpp_sum / area,
                    "GPP_contribution_pct": 100 * contribution_sum / gpp_sum,
                }
            )
            if group == "Global":
                series.insert(3, "pixels", len(pixel_table))
                series.insert(5, "GPP_contribution_gC_m2_yr", contribution_sum / area)
                global_series.append(series)
            else:
                series.insert(2, "group", group)
                regional_series.append(series)
    return pd.concat(global_series), pd.concat(regional_series)


def recent_trends(annual, era_file, output):
    """Fit decadal ΔT anomaly trends in the three ERA5–CMIP6 comparison windows."""
    # ERA5 stores the 2001–2014 anomaly under the original column heading DeltaT.
    era5 = pd.read_csv(era_file).groupby("year")["DeltaT"].mean()
    rows = []
    for scenario, members in annual.groupby("scenario"):
        cmip = members.groupby("year")["DeltaT_anomaly_2001_2014_C"].mean()
        for start, end in [(2001, 2014), (2015, 2023), (2001, 2023)]:
            for source, data in [("ERA5", era5), ("CMIP6", cmip)]:
                period = data.loc[start:end]
                slope = np.polyfit(period.index, period.values, 1)[0] * 10
                rows.append(
                    dict(
                        scenario=scenario,
                        start=start,
                        end=end,
                        source=source,
                        trend_C_per_decade=slope,
                    )
                )
    pd.DataFrame(rows).to_csv(output / "recent_trends.csv", index=False)


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    CMIP6_ANNUAL_INPUT_DIR = PATHS_FILE.parent / paths["CMIP6_ANNUAL_INPUT_DIR"]
    ARIDITY_INDEX_PATH = PATHS_FILE.parent / paths["ARIDITY_INDEX_PATH"]
    ERA5_REFERENCE_PATH = PATHS_FILE.parent / paths["ERA5_REFERENCE_PATH"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    models = [
        "ACCESS-ESM1-5",
        "CMCC-ESM2",
        "CanESM5",
        "INM-CM5-0",
        "IPSL-CM6A-LR",
        "UKESM1-0-LL",
        "CESM2",
        "CNRM-ESM2-1",
        "EC-Earth3-Veg",
        "CNRM-CM6-1-HR",
        "INM-CM4-8",
        "CMCC-CM2-SR5",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ARIDITY_INDEX_PATH) as raster:
        aridity = raster.read(1, masked=True).astype(float).filled(np.nan).ravel()
    global_series, regional_series = [], []
    for model in models:
        annual_global, annual_regional = model_projections(model, CMIP6_ANNUAL_INPUT_DIR, aridity, OUTPUT_DIR)
        global_series.append(annual_global)
        regional_series.append(annual_regional)
    annual = pd.concat(global_series, ignore_index=True)
    annual.to_csv(OUTPUT_DIR / "annual.csv", index=False)
    pd.concat(regional_series).to_csv(OUTPUT_DIR / "regional.csv", index=False)
    endpoints = (
        annual[annual.year >= 2081].groupby(["scenario", "model"])["GPP_contribution_pct"].mean()
    )
    endpoints.to_csv(OUTPUT_DIR / "model_endpoints.csv")
    endpoints.groupby("scenario").mean().to_csv(OUTPUT_DIR / "ensemble_endpoints.csv")
    recent_trends(annual, ERA5_REFERENCE_PATH, OUTPUT_DIR)
