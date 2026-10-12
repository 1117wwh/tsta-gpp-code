"""Calculate pixelwise sensitivity and HAC uncertainty from standardized rasters."""

from pathlib import Path
import json
import re
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

OUTPUT_DIR = CODE_DIR / "outputs/02_sensitivity"
MINIMUM_SAMPLES = 30


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


def main():
    prepare_output(OUTPUT_DIR)
    summary = {}
    for name, settings in DATASETS.items():
        tss = load_series(settings["TSS"], settings["scale"])
        response = load_series(settings["response"], settings["scale"])
        pixels = np.flatnonzero(np.isfinite(response).any(axis=0))
        maps = {
            key: np.full(len(pixels), np.nan)
            for key in ["sensitivity", "intercept", "standard_error_HAC", "p_HAC", "n_pairs", "r_squared"]
        }
        lags = 6 if settings["scale"] == "8day" else 2
        for index, pixel in enumerate(pixels):
            maps["n_pairs"][index] = np.sum(np.isfinite(tss[:, pixel]) & np.isfinite(response[:, pixel]))
            fitted = fit_regression(tss[:, pixel, None], response[:, pixel], lags, MINIMUM_SAMPLES)
            if fitted is None:
                continue
            coefficients, standard_error, p_value, count, r_squared = fitted
            maps["sensitivity"][index] = coefficients[1]
            maps["intercept"][index] = coefficients[0]
            maps["standard_error_HAC"][index] = standard_error[1]
            maps["p_HAC"][index] = p_value[1]
            maps["r_squared"][index] = r_squared
        for metric, values in maps.items():
            save_pixel_map(OUTPUT_DIR / name, metric, values, pixels)
        summary[name] = {
            "candidate_pixels": len(pixels),
            "fitted_pixels": int(np.isfinite(maps["sensitivity"]).sum()),
            "time_steps": len(tss),
            "HAC_lags": lags,
            "minimum_samples": MINIMUM_SAMPLES,
        }
        print(f"{name}: {summary[name]}", flush=True)
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
