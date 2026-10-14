"""Regress productivity on TSS at each pixel and estimate HAC uncertainty."""

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import t as student_t

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
    for name in RUN_DATASETS:
        dataset = DATASETS[name]
        directory = OUTPUT_DIR / f"{name}_{YEARS[0]}-{YEARS[-1]}"
        TSS = load_series(dataset["TSS"], dataset["scale"])
        GPP = load_series(dataset["GPP"], dataset["scale"])
        pixels = np.flatnonzero(np.isfinite(GPP).any(axis=0))
        maps = {}
        output_names = ["sensitivity", "intercept", "standard_error_HAC", "p_HAC", "n_pairs", "r_squared"]
        for output_name in output_names:
            maps[output_name] = np.full(len(pixels), np.nan)
        if dataset["scale"] == "8day":
            lags = 6
        else:
            lags = 2
        for index, pixel in enumerate(pixels):
            maps["n_pairs"][index] = np.sum(np.isfinite(TSS[:, pixel]) & np.isfinite(GPP[:, pixel]))
            fitted = fit_regression(TSS[:, pixel].reshape(-1, 1), GPP[:, pixel], lags, MINIMUM_SAMPLES)
            if fitted is None:
                continue
            coefficients, standard_error, p_value, count, r_squared = fitted
            maps["sensitivity"][index] = coefficients[1]
            maps["intercept"][index] = coefficients[0]
            maps["standard_error_HAC"][index] = standard_error[1]
            maps["p_HAC"][index] = p_value[1]
            maps["r_squared"][index] = r_squared
        for metric, values in maps.items():
            save_pixel_map(directory, metric, values, pixels)
        result = {
            "candidate_pixels": len(pixels),
            "fitted_pixels": int(np.isfinite(maps["sensitivity"]).sum()),
            "time_steps": len(TSS),
            "HAC_lags": lags,
            "minimum_samples": MINIMUM_SAMPLES,
        }
        print(f"{name}: {result}", flush=True)
        (directory / "run_summary.json").write_text(json.dumps({name: result}, indent=2), encoding="utf-8")

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
    },
    "GloFlux_GPP_monthly": {
        "scale": "monthly",
        "TSS": STANDARDIZED_DIR / f"TSS_monthly_{START_YEAR}-{END_YEAR}",
        "GPP": MONTHLY_DIR / "GLOFLUX_GPP_MONTH_ZSCORE",
    },
}
OUTPUT_DIR = CODE_DIR / "outputs/02_sensitivity"
MINIMUM_SAMPLES = 30

if __name__ == "__main__":
    main()
