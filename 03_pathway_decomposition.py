"""Calculate VPD, soil-moisture and temperature pathways with a paired year bootstrap."""

from pathlib import Path
import json
import re
import warnings
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import t as student_t

CODE_DIR = Path(__file__).resolve().parent
YEARS = list(range(2001, 2024))
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
OVERWRITE = False

STANDARDIZED_DIR = CODE_DIR / "outputs/01_standardized"
MONTHLY_DIR = Path("D:/DATA/TTS-GPP/STANDARDIZED/1deg_monthly")
DATASETS = {
    "GOSIF_GPP_8day": {
        "scale": "8day",
        "TSS": STANDARDIZED_DIR / "TSS_8day",
        "response": STANDARDIZED_DIR / "GOSIF_GPP_8day",
        "VPD": STANDARDIZED_DIR / "VPD_8day",
        "SM": STANDARDIZED_DIR / "SM_8day",
        "Ta": STANDARDIZED_DIR / "Ta_8day",
    },
    "GloFlux_GPP_monthly": {
        "scale": "monthly",
        "TSS": STANDARDIZED_DIR / "TSS_monthly",
        "response": MONTHLY_DIR / "GLOFLUX_GPP_MONTH_ZSCORE",
        "VPD": MONTHLY_DIR / "ERA5Land_VPD_MONTH_ZSCORE",
        "SM": MONTHLY_DIR / "ERA5Land_SM_MONTH_ZSCORE",
        "Ta": MONTHLY_DIR / "ERA5Land_TA_ALLDAY_MONTH_ZSCORE",
    },
}

OUTPUT_DIR = CODE_DIR / "outputs/03_pathways"
BOOTSTRAP_REPEATS = 1999
BOOTSTRAP_SEED = 20260917
PIXELS_PER_BATCH = 64
VARIABLES = ["TSS", "response", "VPD", "SM", "Ta"]
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
PRODUCT_PAIRS = [(first, second) for first in range(5) for second in range(first, 5)]


def read_raster(path):
    with rasterio.open(path) as source:
        if (
            source.shape != GRID_SHAPE
            or source.count != 1
            or source.crs != rasterio.crs.CRS.from_epsg(4326)
            or not np.allclose(tuple(source.transform), tuple(GRID_TRANSFORM), rtol=0, atol=1e-9)
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


def index_rasters(directory, pattern, time_scale):
    periods = list(range(1, 366, 8)) if time_scale == "8day" else list(range(1, 13))
    expected = [(year, period) for year in YEARS for period in periods]
    paths = {}
    for path in sorted(directory.rglob("*.tif")):
        match = re.search(pattern, path.name)
        if match is None:
            continue
        year, period = int(match["year"]), int(match["period"])
        if year not in YEARS:
            continue
        if period not in periods or (year, period) in paths:
            raise ValueError(f"Invalid or duplicate date: {path}")
        paths[year, period] = path
    missing = sorted(set(expected) - set(paths))
    if missing:
        raise ValueError(f"Missing {len(missing)} raster dates in {directory}: {missing[:5]}")
    return {key: paths[key] for key in expected}


def load_series(directory, time_scale):
    paths = index_rasters(directory, r"_(?P<year>20\d{2})_(?P<period>\d{2,3})_1deg\.tif$", time_scale)
    values = np.empty((len(paths), 180 * 360), dtype=np.float32)
    for index, path in enumerate(paths.values()):
        values[index] = read_raster(path).ravel()
    print(f"Read {len(paths)} full rasters: {directory}", flush=True)
    return values


def save_pixel_map(directory, name, values, pixels):
    grid = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
    grid.ravel()[pixels] = values
    write_raster(directory / f"{name}_1deg.tif", grid, name)


def fit_regression(predictors, response, lags, minimum_samples):
    valid = np.isfinite(response) & np.isfinite(predictors).all(axis=1)
    count = int(valid.sum())
    design = np.column_stack([np.ones(len(response)), predictors])
    fitted_design = design[valid]
    observed = response[valid]
    parameters = design.shape[1]
    if count < max(minimum_samples, parameters + 1) or np.linalg.matrix_rank(fitted_design) < parameters:
        return None
    coefficients = np.linalg.lstsq(fitted_design, observed, rcond=None)[0]
    residual = observed - fitted_design @ coefficients
    inverse_gram = np.linalg.inv(fitted_design.T @ fitted_design)
    # Keep missing dates in the time axis when computing HAC lagged products.
    scores = np.zeros_like(design)
    scores[valid] = fitted_design * residual[:, None]
    score_covariance = scores.T @ scores
    for lag in range(1, min(lags, len(response) - 1) + 1):
        cross_product = scores[lag:].T @ scores[:-lag]
        weight = 1 - lag / (lags + 1)
        score_covariance += weight * (cross_product + cross_product.T)
    covariance = inverse_gram @ score_covariance @ inverse_gram * count / (count - parameters)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_value = 2 * student_t.sf(abs(coefficients / standard_error), count - parameters)
    total_variation = np.sum((observed - observed.mean()) ** 2)
    r_squared = 1 - np.sum(residual**2) / total_variation if total_variation > 0 else np.nan
    return coefficients, standard_error, p_value, count, r_squared


def annual_statistics(values, periods_per_year, paired_only=False):
    # Each year has a count, five variable sums and 15 unique cross-products.
    statistics = np.zeros((len(YEARS), values.shape[1], 21))
    for index, year in enumerate(YEARS):
        yearly = values[index * periods_per_year : (index + 1) * periods_per_year].astype(float)
        checked = yearly[:, :, :2] if paired_only else yearly
        valid = np.isfinite(checked).all(axis=2)
        yearly = np.where(valid[:, :, None], np.nan_to_num(yearly, nan=0), 0)
        statistics[index, :, 0] = valid.sum(axis=0)
        statistics[index, :, 1:6] = yearly.sum(axis=0)
        for column, (first, second) in enumerate(PRODUCT_PAIRS):
            statistics[index, :, 6 + column] = (yearly[:, :, first] * yearly[:, :, second]).sum(axis=0)
    return statistics


def fit_pathways(statistics, paired_statistics):
    count = statistics[..., 0]
    sums = statistics[..., 1:6]
    centered_products = np.empty((*count.shape, 5, 5))
    for column, (first, second) in enumerate(PRODUCT_PAIRS):
        centered = statistics[..., 6 + column] - sums[..., first] * sums[..., second] / np.maximum(count, 1)
        centered_products[..., first, second] = centered
        centered_products[..., second, first] = centered
    environmental_products = centered_products[..., 2:, 2:]
    eigenvalues = np.linalg.eigvalsh(environmental_products)
    valid = (
        (count >= 5)
        & (centered_products[..., 0, 0] > 1e-12)
        & (eigenvalues[..., 0] > 1e-10 * np.maximum(eigenvalues[..., -1], 1))
    )
    safe_products = np.where(valid[..., None, None], environmental_products, np.eye(3))
    beta = np.linalg.solve(safe_products, centered_products[..., 2:, 1, None])[..., 0]
    alpha = centered_products[..., 0, 2:] / np.maximum(centered_products[..., 0, 0, None], 1e-30)
    contribution = alpha * beta
    pair_count = paired_statistics[..., 0]
    pair_sums = paired_statistics[..., 1:6]
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = (paired_statistics[..., 7] - pair_sums[..., 0] * pair_sums[..., 1] / pair_count) / (
            paired_statistics[..., 6] - pair_sums[..., 0] ** 2 / pair_count
        )
    water = contribution[..., :2].sum(axis=-1)
    reconstruction = contribution.sum(axis=-1)
    results = np.concatenate(
        [
            alpha,
            beta,
            contribution,
            water[..., None],
            reconstruction[..., None],
            direct[..., None],
            (water - contribution[..., 2])[..., None],
        ],
        axis=-1,
    )
    return np.where(valid[..., None], results, np.nan)


def main():
    prepare_output(OUTPUT_DIR)
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(YEARS), np.full(len(YEARS), 1 / len(YEARS)), size=BOOTSTRAP_REPEATS
    )
    summary = {}
    for name, settings in DATASETS.items():
        response = load_series(settings["response"], settings["scale"])
        pixels = np.flatnonzero(np.isfinite(response).any(axis=0))
        values = np.empty((len(response), len(pixels), 5), dtype=np.float32)
        values[:, :, 1] = response[:, pixels]
        del response
        for column, variable in enumerate(VARIABLES):
            if variable != "response":
                values[:, :, column] = load_series(settings[variable], settings["scale"])[:, pixels]
        periods = 46 if settings["scale"] == "8day" else 12
        statistics = annual_statistics(values, periods)
        paired_statistics = annual_statistics(values, periods, paired_only=True)
        total = statistics.sum(axis=0)
        estimates = fit_pathways(total, paired_statistics.sum(axis=0))
        lower = np.full_like(estimates, np.nan)
        upper = lower.copy()
        probability = lower.copy()
        valid_fraction = lower.copy()
        for start in range(0, len(pixels), PIXELS_PER_BATCH):
            end = min(start + PIXELS_PER_BATCH, len(pixels))
            batch_size = end - start
            sampled = (draws @ statistics[:, start:end].reshape(len(YEARS), -1)).reshape(
                BOOTSTRAP_REPEATS, batch_size, 21
            )
            sampled_pairs = (draws @ paired_statistics[:, start:end].reshape(len(YEARS), -1)).reshape(
                BOOTSTRAP_REPEATS, batch_size, 21
            )
            bootstrap = fit_pathways(sampled, sampled_pairs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                lower[start:end], upper[start:end] = np.nanquantile(bootstrap, [0.025, 0.975], axis=0)
            valid = np.isfinite(bootstrap)
            extreme = (abs(bootstrap - estimates[None, start:end]) >= abs(estimates[None, start:end])) & valid
            probability[start:end] = (1 + extreme.sum(axis=0)) / (1 + valid.sum(axis=0))
            valid_fraction[start:end] = valid.mean(axis=0)
            if start % 1024 == 0:
                print(
                    f"{name}: {end}/{len(pixels)} pixels, {BOOTSTRAP_REPEATS} year resamples each", flush=True
                )
        probability[~np.isfinite(estimates) | (valid_fraction == 0)] = np.nan
        directory = OUTPUT_DIR / name
        for column, metric in enumerate(METRICS):
            for suffix, array in [
                ("estimate", estimates),
                ("ci_low", lower),
                ("ci_high", upper),
                ("p_bootstrap", probability),
                ("valid_fraction", valid_fraction),
            ]:
                save_pixel_map(directory, f"{metric}_{suffix}", array[:, column], pixels)
        count = total[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            matched = (total[:, 7] - total[:, 1] * total[:, 2] / count) / (
                total[:, 6] - total[:, 1] ** 2 / count
            )
        save_pixel_map(directory, "S_direct_matched", matched, pixels)
        save_pixel_map(directory, "S_residual", matched - estimates[:, 10], pixels)
        save_pixel_map(directory, "n_complete", count, pixels)
        save_pixel_map(directory, "n_pairs", paired_statistics[:, :, 0].sum(axis=0), pixels)

        # Use HAC for regression coefficients and the year bootstrap for pathway contributions.
        hac_probability = np.full((len(pixels), 7), np.nan)
        lags = 6 if settings["scale"] == "8day" else 2
        for index in range(len(pixels)):
            if not np.isfinite(estimates[index]).all():
                continue
            pixel = values[:, index].astype(float)
            complete = np.isfinite(pixel).all(axis=1)
            matched_pixel = np.where(complete[:, None], pixel, np.nan)
            for environmental in range(3):
                fitted = fit_regression(matched_pixel[:, :1], matched_pixel[:, environmental + 2], lags, 5)
                hac_probability[index, environmental] = fitted[2][1]
            fitted = fit_regression(matched_pixel[:, 2:], matched_pixel[:, 1], lags, 5)
            hac_probability[index, 3:6] = fitted[2][1:]
            fitted = fit_regression(pixel[:, :1], pixel[:, 1], lags, 5)
            hac_probability[index, 6] = fitted[2][1]
        for column, metric in enumerate(METRICS[:6] + ["S_direct"]):
            save_pixel_map(directory, f"{metric}_p_HAC", hac_probability[:, column], pixels)
        summary[name] = {
            "candidate_pixels": len(pixels),
            "fitted_pixels": int(np.isfinite(estimates).all(axis=1).sum()),
            "time_steps": len(values),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "HAC_lags": lags,
        }
        print(f"{name}: complete", flush=True)
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
