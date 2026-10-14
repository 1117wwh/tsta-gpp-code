"""Decompose productivity sensitivity into VPD, soil-moisture and temperature pathways."""

from pathlib import Path
import json
import warnings
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import t as student_t


def calculate_pathways(sums, pair_sums):
    """Fit the two regression stages and combine their pathway coefficients."""
    original_shape = sums["n"].shape
    data = {}
    pairs = {}
    for name in sums:
        data[name] = sums[name].flatten()
    for name in pair_sums:
        pairs[name] = pair_sums[name].flatten()

    n = data["n"]
    divisor = np.maximum(n, 1)
    TSS_variation = data["TSS_squared"] - data["TSS"] * data["TSS"] / divisor
    TSS_denominator = np.maximum(TSS_variation, 1e-30)

    # Stage 1: VPD, SM and Ta regressed separately on TSS, with intercepts.
    a_VPD = (data["TSS_VPD"] - data["TSS"] * data["VPD"] / divisor) / TSS_denominator
    a_SM = (data["TSS_SM"] - data["TSS"] * data["SM"] / divisor) / TSS_denominator
    a_Ta = (data["TSS_Ta"] - data["TSS"] * data["Ta"] / divisor) / TSS_denominator

    # Stage 2: GPP regressed jointly on VPD, SM and Ta, with an intercept.
    predictor_covariance = np.empty((len(n), 3, 3))
    predictor_covariance[:, 0, 0] = data["VPD_squared"] - data["VPD"]**2 / divisor
    predictor_covariance[:, 1, 1] = data["SM_squared"] - data["SM"]**2 / divisor
    predictor_covariance[:, 2, 2] = data["Ta_squared"] - data["Ta"]**2 / divisor
    predictor_covariance[:, 0, 1] = data["VPD_SM"] - data["VPD"] * data["SM"] / divisor
    predictor_covariance[:, 0, 2] = data["VPD_Ta"] - data["VPD"] * data["Ta"] / divisor
    predictor_covariance[:, 1, 2] = data["SM_Ta"] - data["SM"] * data["Ta"] / divisor
    predictor_covariance[:, 1, 0] = predictor_covariance[:, 0, 1]
    predictor_covariance[:, 2, 0] = predictor_covariance[:, 0, 2]
    predictor_covariance[:, 2, 1] = predictor_covariance[:, 1, 2]
    GPP_covariance = np.column_stack([
        data["GPP_VPD"] - data["VPD"] * data["GPP"] / divisor,
        data["GPP_SM"] - data["SM"] * data["GPP"] / divisor,
        data["GPP_Ta"] - data["Ta"] * data["GPP"] / divisor,
    ])

    eigenvalues = np.linalg.eigvalsh(predictor_covariance)
    full_rank = eigenvalues[:, 0] > 1e-10 * np.maximum(eigenvalues[:, -1], 1)
    valid = (n >= 5) & (TSS_variation > 1e-12) & full_rank
    predictor_covariance[~valid] = np.eye(3)
    coefficients = np.linalg.solve(predictor_covariance, GPP_covariance.reshape(len(n), 3, 1))
    b_VPD = coefficients[:, 0, 0]
    b_SM = coefficients[:, 1, 0]
    b_Ta = coefficients[:, 2, 0]

    C_VPD = a_VPD * b_VPD
    C_SM = a_SM * b_SM
    C_thermal = a_Ta * b_Ta
    C_water = C_VPD + C_SM
    S_recon = C_water + C_thermal

    # Direct sensitivity uses all finite TSS-GPP pairs.
    with np.errstate(divide="ignore", invalid="ignore"):
        paired_covariance = pairs["TSS_GPP"] - pairs["TSS"] * pairs["GPP"] / pairs["n"]
        paired_variation = pairs["TSS_squared"] - pairs["TSS"]**2 / pairs["n"]
        S_direct = paired_covariance / paired_variation

    results = {
        "a_VPD": a_VPD, "a_SM": a_SM, "a_Ta": a_Ta,
        "b_VPD": b_VPD, "b_SM": b_SM, "b_Ta": b_Ta,
        "C_VPD": C_VPD, "C_SM": C_SM, "C_thermal": C_thermal,
        "C_water": C_water, "S_recon": S_recon, "S_direct": S_direct,
        "C_water_minus_thermal": C_water - C_thermal,
    }
    for name in results:
        results[name][~valid] = np.nan
        results[name] = results[name].reshape(original_shape)
    return results


def yearly_regression_sums(TSS, GPP, VPD, SM, Ta, periods_per_year):
    """Keep named annual sums so whole years can be resampled together."""
    sums = {}
    pair_sums = {}
    for year_index in range(len(YEARS)):
        first_date = year_index * periods_per_year
        last_date = first_date + periods_per_year
        tss = np.array(TSS[first_date:last_date], dtype=float, order="C")
        gpp = np.array(GPP[first_date:last_date], dtype=float, order="C")
        vpd = np.array(VPD[first_date:last_date], dtype=float, order="C")
        sm = np.array(SM[first_date:last_date], dtype=float, order="C")
        ta = np.array(Ta[first_date:last_date], dtype=float, order="C")

        paired = np.isfinite(tss) & np.isfinite(gpp)
        paired_tss = np.where(paired, tss, 0)
        paired_gpp = np.where(paired, gpp, 0)
        this_year_pairs = {
            "n": paired.sum(axis=0),
            "TSS": paired_tss.sum(axis=0),
            "GPP": paired_gpp.sum(axis=0),
            "TSS_squared": (paired_tss * paired_tss).sum(axis=0),
            "TSS_GPP": (paired_tss * paired_gpp).sum(axis=0),
        }

        complete = paired & np.isfinite(vpd) & np.isfinite(sm) & np.isfinite(ta)
        tss[~complete] = 0
        gpp[~complete] = 0
        vpd[~complete] = 0
        sm[~complete] = 0
        ta[~complete] = 0
        this_year = {
            "n": complete.sum(axis=0),
            "TSS": tss.sum(axis=0), "GPP": gpp.sum(axis=0),
            "VPD": vpd.sum(axis=0), "SM": sm.sum(axis=0), "Ta": ta.sum(axis=0),
            "TSS_squared": (tss * tss).sum(axis=0),
            "TSS_GPP": (tss * gpp).sum(axis=0),
            "TSS_VPD": (tss * vpd).sum(axis=0),
            "TSS_SM": (tss * sm).sum(axis=0),
            "TSS_Ta": (tss * ta).sum(axis=0),
            "GPP_VPD": (gpp * vpd).sum(axis=0),
            "GPP_SM": (gpp * sm).sum(axis=0),
            "GPP_Ta": (gpp * ta).sum(axis=0),
            "VPD_squared": (vpd * vpd).sum(axis=0),
            "VPD_SM": (vpd * sm).sum(axis=0),
            "VPD_Ta": (vpd * ta).sum(axis=0),
            "SM_squared": (sm * sm).sum(axis=0),
            "SM_Ta": (sm * ta).sum(axis=0),
            "Ta_squared": (ta * ta).sum(axis=0),
        }
        for name in this_year:
            if year_index == 0:
                sums[name] = np.empty((len(YEARS), TSS.shape[1]))
            sums[name][year_index] = this_year[name]
        for name in this_year_pairs:
            if year_index == 0:
                pair_sums[name] = np.empty((len(YEARS), TSS.shape[1]))
            pair_sums[name][year_index] = this_year_pairs[name]
    return sums, pair_sums


def bootstrap_pathways(yearly_sums, yearly_pairs, estimates, year_counts, output_directory, pixels):
    """Resample calendar years and save uncertainty for each contribution."""
    lower = {}
    upper = {}
    p_values = {}
    valid_fractions = {}
    for name in METRICS:
        lower[name] = np.full(len(pixels), np.nan)
        upper[name] = np.full(len(pixels), np.nan)
        p_values[name] = np.full(len(pixels), np.nan)
        valid_fractions[name] = np.full(len(pixels), np.nan)

    for first_pixel in range(0, len(pixels), PIXELS_PER_BATCH):
        last_pixel = min(first_pixel + PIXELS_PER_BATCH, len(pixels))
        sampled_sums = {}
        sampled_pairs = {}
        for name in yearly_sums:
            annual_values = yearly_sums[name][:, first_pixel:last_pixel]
            sampled_sums[name] = np.dot(year_counts, annual_values)
        for name in yearly_pairs:
            annual_values = yearly_pairs[name][:, first_pixel:last_pixel]
            sampled_pairs[name] = np.dot(year_counts, annual_values)
        bootstrap_estimates = calculate_pathways(sampled_sums, sampled_pairs)

        for name in METRICS:
            draws = bootstrap_estimates[name]
            estimate = estimates[name][first_pixel:last_pixel]
            valid = np.isfinite(draws)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                interval = np.nanquantile(draws, [0.025, 0.975], axis=0)
            lower[name][first_pixel:last_pixel] = interval[0]
            upper[name][first_pixel:last_pixel] = interval[1]
            extreme = (abs(draws - estimate) >= abs(estimate)) & valid
            p_values[name][first_pixel:last_pixel] = (1 + extreme.sum(axis=0)) / (1 + valid.sum(axis=0))
            valid_fractions[name][first_pixel:last_pixel] = valid.mean(axis=0)
        if first_pixel % 1024 == 0:
            print(f"Bootstrap: {last_pixel}/{len(pixels)} pixels", flush=True)

    for name in METRICS:
        invalid = ~np.isfinite(estimates[name]) | (valid_fractions[name] == 0)
        p_values[name][invalid] = np.nan
        save_pixel_map(output_directory, name + "_ci_low", lower[name], pixels)
        save_pixel_map(output_directory, name + "_ci_high", upper[name], pixels)
        save_pixel_map(output_directory, name + "_p_bootstrap", p_values[name], pixels)
        save_pixel_map(output_directory, name + "_valid_fraction", valid_fractions[name], pixels)


def save_hac_tests(TSS, GPP, VPD, SM, Ta, estimates, lags, directory, pixels):
    p_values = {}
    for name in ["a_VPD", "a_SM", "a_Ta", "b_VPD", "b_SM", "b_Ta", "S_direct"]:
        p_values[name] = np.full(len(pixels), np.nan)

    for pixel in range(len(pixels)):
        if not np.isfinite(estimates["S_recon"][pixel]):
            continue
        tss = TSS[:, pixel].astype(float)
        gpp = GPP[:, pixel].astype(float)
        environment = np.column_stack([VPD[:, pixel], SM[:, pixel], Ta[:, pixel]]).astype(float)
        complete = np.isfinite(tss) & np.isfinite(gpp) & np.isfinite(environment).all(axis=1)
        matched_tss = tss.copy()
        matched_gpp = gpp.copy()
        matched_tss[~complete] = np.nan
        matched_gpp[~complete] = np.nan
        environment[~complete] = np.nan

        for column, name in enumerate(["a_VPD", "a_SM", "a_Ta"]):
            fitted = fit_regression(matched_tss.reshape(-1, 1), environment[:, column], lags, 5)
            coefficients, standard_errors, probabilities, n, r_squared = fitted
            p_values[name][pixel] = probabilities[1]
        fitted = fit_regression(environment, matched_gpp, lags, 5)
        coefficients, standard_errors, probabilities, n, r_squared = fitted
        p_values["b_VPD"][pixel] = probabilities[1]
        p_values["b_SM"][pixel] = probabilities[2]
        p_values["b_Ta"][pixel] = probabilities[3]
        fitted = fit_regression(tss.reshape(-1, 1), gpp, lags, 5)
        coefficients, standard_errors, probabilities, n, r_squared = fitted
        p_values["S_direct"][pixel] = probabilities[1]

    for name in p_values:
        save_pixel_map(directory, name + "_p_HAC", p_values[name], pixels)


def fit_regression(predictors, response, lags, minimum_samples):
    """Fit OLS with an intercept and calculate HAC uncertainty."""
    valid = np.isfinite(response) & np.isfinite(predictors).all(axis=1)
    sample_count = int(valid.sum())
    design_matrix = np.column_stack([np.ones(len(response)), predictors])
    valid_predictors = design_matrix[valid]
    observed = response[valid]
    coefficient_count = design_matrix.shape[1]
    if sample_count < max(minimum_samples, coefficient_count + 1):
        return None
    if np.linalg.matrix_rank(valid_predictors) < coefficient_count:
        return None

    coefficients = np.linalg.lstsq(valid_predictors, observed, rcond=None)[0]
    fitted_values = np.dot(valid_predictors, coefficients)
    residuals = observed - fitted_values
    predictor_products = np.dot(valid_predictors.T, valid_predictors)
    inverse_products = np.linalg.inv(predictor_products)

    # Missing dates keep their positions so HAC lags follow the original time axis.
    weighted_residuals = np.zeros_like(design_matrix)
    residual_column = residuals.reshape(sample_count, 1)
    weighted_residuals[valid] = valid_predictors * residual_column
    residual_covariance = np.dot(weighted_residuals.T, weighted_residuals)
    for lag in range(1, min(lags, len(response) - 1) + 1):
        later_values = weighted_residuals[lag:]
        earlier_values = weighted_residuals[:-lag]
        lagged_covariance = np.dot(later_values.T, earlier_values)
        bartlett_weight = 1 - lag / (lags + 1)
        residual_covariance += bartlett_weight * (lagged_covariance + lagged_covariance.T)

    coefficient_covariance = np.dot(inverse_products, residual_covariance)
    coefficient_covariance = np.dot(coefficient_covariance, inverse_products)
    coefficient_covariance = coefficient_covariance * sample_count / (sample_count - coefficient_count)
    standard_errors = np.sqrt(np.maximum(np.diag(coefficient_covariance), 0))
    degrees_of_freedom = sample_count - coefficient_count
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = coefficients / standard_errors
        p_values = 2 * student_t.sf(abs(t_statistics), degrees_of_freedom)

    total_variation = np.sum((observed - observed.mean()) ** 2)
    if total_variation > 0:
        r_squared = 1 - np.sum(residuals**2) / total_variation
    else:
        r_squared = np.nan
    return coefficients, standard_errors, p_values, sample_count, r_squared


def read_raster(path):
    with rasterio.open(path) as source:
        if (
            source.shape != GRID_SHAPE
            or source.count != 1
            or source.crs != rasterio.crs.CRS.from_epsg(4326)
            or not np.allclose(source.transform, GRID_TRANSFORM, rtol=0, atol=1e-9)
        ):
            raise ValueError(f"Expected an aligned global 1-degree EPSG:4326 raster: {path}")
        values = source.read(1, masked=True).astype("float64").filled(np.nan)
    values[~np.isfinite(values)] = np.nan
    return values


def write_raster(path, values, description):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=180,
        width=360,
        count=1,
        dtype=str(values.dtype),
        crs="EPSG:4326",
        transform=GRID_TRANSFORM,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as output:
        output.write(values, 1)
        output.set_band_description(1, description)
        output.update_tags(years=f"{YEARS[0]}-{YEARS[-1]}", resolution="1 degree")


def prepare_output(directory):
    if directory.exists() and any(directory.iterdir()) and not OVERWRITE:
        raise FileExistsError(f"Results already exist. Change OUTPUT_DIR or set OVERWRITE=True: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def index_rasters(directory, time_scale):
    if time_scale not in ["8day", "monthly"]:
        raise ValueError("Time scale must be '8day' or 'monthly'")
    if time_scale == "8day":
        periods = range(1, 366, 8)
    else:
        periods = range(1, 13)

    paths = {}
    for path in sorted(directory.rglob("*.tif")):
        # Example: GOSIF_GPP_8day_z_2001_001_1deg.tif
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[-1] != "1deg":
            continue
        if not parts[-3].isdigit() or not parts[-2].isdigit():
            continue
        year = int(parts[-3])
        period = int(parts[-2])
        if year not in YEARS:
            continue
        date = (year, period)
        if period not in periods or date in paths:
            raise ValueError(f"Invalid or duplicate date: {path}")
        paths[date] = path

    ordered_paths = {}
    for year in YEARS:
        for period in periods:
            if (year, period) not in paths:
                raise ValueError(f"Missing raster for {year}, period {period}: {directory}")
            ordered_paths[year, period] = paths[year, period]
    return ordered_paths


def load_series(directory, time_scale):
    paths = index_rasters(directory, time_scale)
    # Rows are dates; columns are pixels in north-to-south raster order.
    values = np.empty((len(paths), 180 * 360), dtype=np.float32)
    for index, path in enumerate(paths.values()):
        raster = read_raster(path)
        values[index] = raster.flatten()
    print(f"Read {len(paths)} full rasters: {directory}", flush=True)
    return values


def save_pixel_map(directory, name, values, pixels):
    flat_grid = np.full(180 * 360, np.nan, dtype=np.float64)
    flat_grid[pixels] = values
    grid = flat_grid.reshape(180, 360)
    write_raster(directory / f"{directory.name}_{name}_1deg.tif", grid, name)


def main():
    if not YEARS:
        raise ValueError("START_YEAR must not exceed END_YEAR")
    if not RUN_DATASETS or len(RUN_DATASETS) != len(set(RUN_DATASETS)):
        raise ValueError("RUN_DATASETS must contain unique dataset names")
    for name in RUN_DATASETS:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS)}")
        if DATASETS[name]["scale"] not in ["8day", "monthly"] or not name.endswith("_" + DATASETS[name]["scale"]):
            raise ValueError(f"Dataset name and time scale disagree: {name}")
        prepare_output(OUTPUT_DIR / f"{name}_{YEARS[0]}-{YEARS[-1]}")
    random_generator = np.random.default_rng(BOOTSTRAP_SEED)
    year_probabilities = np.full(len(YEARS), 1 / len(YEARS))
    year_counts = random_generator.multinomial(len(YEARS), year_probabilities, size=BOOTSTRAP_REPEATS)

    for dataset_name in RUN_DATASETS:
        dataset = DATASETS[dataset_name]
        directory = OUTPUT_DIR / f"{dataset_name}_{YEARS[0]}-{YEARS[-1]}"
        GPP_grid = load_series(dataset["GPP"], dataset["scale"])
        pixels = np.flatnonzero(np.isfinite(GPP_grid).any(axis=0))
        GPP = GPP_grid[:, pixels]
        del GPP_grid
        TSS = load_series(dataset["TSS"], dataset["scale"])[:, pixels]
        VPD = load_series(dataset["VPD"], dataset["scale"])[:, pixels]
        SM = load_series(dataset["SM"], dataset["scale"])[:, pixels]
        Ta = load_series(dataset["Ta"], dataset["scale"])[:, pixels]
        if dataset["scale"] == "8day":
            periods_per_year = 46
            hac_lags = 6
        else:
            periods_per_year = 12
            hac_lags = 2

        yearly_sums, yearly_pairs = yearly_regression_sums(TSS, GPP, VPD, SM, Ta, periods_per_year)
        total_sums = {}
        total_pairs = {}
        for name in yearly_sums:
            total_sums[name] = yearly_sums[name].sum(axis=0)
        for name in yearly_pairs:
            total_pairs[name] = yearly_pairs[name].sum(axis=0)

        estimates = calculate_pathways(total_sums, total_pairs)
        for name in METRICS:
            save_pixel_map(directory, name + "_estimate", estimates[name], pixels)
        bootstrap_pathways(yearly_sums, yearly_pairs, estimates, year_counts, directory, pixels)
        save_hac_tests(TSS, GPP, VPD, SM, Ta, estimates, hac_lags, directory, pixels)

        n = total_sums["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = total_sums["TSS_GPP"] - total_sums["TSS"] * total_sums["GPP"] / n
            variation = total_sums["TSS_squared"] - total_sums["TSS"]**2 / n
            S_direct_matched = covariance / variation
        S_residual = S_direct_matched - estimates["S_recon"]
        save_pixel_map(directory, "S_direct_matched", S_direct_matched, pixels)
        save_pixel_map(directory, "S_residual", S_residual, pixels)
        save_pixel_map(directory, "n_complete", n, pixels)
        save_pixel_map(directory, "n_pairs", total_pairs["n"], pixels)
        result = {
            "candidate_pixels": len(pixels),
            "fitted_pixels": int(np.isfinite(estimates["S_recon"]).sum()),
            "time_steps": len(TSS), "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED, "HAC_lags": hac_lags,
        }
        (directory / "run_summary.json").write_text(
            json.dumps({dataset_name: result}, indent=2), encoding="utf-8"
        )
        print(f"{dataset_name}: complete", flush=True)


# Input paths and run settings
RUN_DATASETS = ["GOSIF_GPP_8day", "GloFlux_GPP_monthly"]
START_YEAR = 2001
END_YEAR = 2023
YEARS = list(range(START_YEAR, END_YEAR + 1))
CODE_DIR = Path(__file__).resolve().parent
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
OVERWRITE = False
STANDARDIZED_DIR = CODE_DIR / "outputs/01_standardized"
MONTHLY_DIR = Path("D:/DATA/TTS-GPP/STANDARDIZED/1deg_monthly")
DATASETS = {
    "GOSIF_GPP_8day": {
        "scale": "8day",
        "TSS": STANDARDIZED_DIR / f"TSS_8day_{START_YEAR}-{END_YEAR}",
        "GPP": STANDARDIZED_DIR / f"GOSIF_GPP_8day_{START_YEAR}-{END_YEAR}",
        "VPD": STANDARDIZED_DIR / f"VPD_8day_{START_YEAR}-{END_YEAR}",
        "SM": STANDARDIZED_DIR / f"SM_8day_{START_YEAR}-{END_YEAR}",
        "Ta": STANDARDIZED_DIR / f"Ta_8day_{START_YEAR}-{END_YEAR}",
    },
    "GloFlux_GPP_monthly": {
        "scale": "monthly",
        "TSS": STANDARDIZED_DIR / f"TSS_monthly_{START_YEAR}-{END_YEAR}",
        "GPP": MONTHLY_DIR / "GLOFLUX_GPP_MONTH_ZSCORE",
        "VPD": MONTHLY_DIR / "ERA5Land_VPD_MONTH_ZSCORE",
        "SM": MONTHLY_DIR / "ERA5Land_SM_MONTH_ZSCORE",
        "Ta": MONTHLY_DIR / "ERA5Land_TA_ALLDAY_MONTH_ZSCORE",
    },
}
OUTPUT_DIR = CODE_DIR / "outputs/03_pathways"
BOOTSTRAP_REPEATS = 1999
BOOTSTRAP_SEED = 20260917
PIXELS_PER_BATCH = 64
VARIABLES = ["TSS", "GPP", "VPD", "SM", "Ta"]
METRICS = [
    "a_VPD",
    "a_SM",
    "a_Ta",
    "b_VPD",
    "b_SM",
    "b_Ta",
    "C_VPD",
    "C_SM",
    "C_thermal",
    "C_water",
    "S_recon",
    "S_direct",
    "C_water_minus_thermal",
]

if __name__ == "__main__":
    main()
