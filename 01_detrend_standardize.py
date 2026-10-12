"""Detrend and standardize 1-degree raster time series within each DOY or month."""

from pathlib import Path
import json
import re
import numpy as np
import rasterio
from rasterio.transform import from_origin

CODE_DIR = Path(__file__).resolve().parent
YEARS = list(range(2001, 2024))
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
OVERWRITE = False

OUTPUT_DIR = CODE_DIR / "outputs/01_standardized"
MASK_FILE = CODE_DIR / "inputs/vegetation_mask_1deg.tif"
RAW_ROOT = Path("D:/DATA")

DATASETS = {
    "GOSIF_GPP_8day": {
        "directory": RAW_ROOT / "GOSIF_GPP_8day_1deg",
        "pattern": r"GOSIF_GPP_(?P<year>20\d{2})(?P<period>\d{3})_Mean_1deg\.tif$",
        "scale": "8day",
        "min_trend": 5,
        "min_zscore": 3,
        "sd_floor": 1e-6,
        "sd_inclusive": True,
        "fallback": "retain",
        "residual_float32": False,
        "center_years": False,
        "minimum": 0,
    },
    "TSS_8day": {
        "directory": RAW_ROOT / "TSS/TSS/8day/ERA5_LST__ERA5_TA",
        "pattern": r"_(?P<year>20\d{2})_(?P<period>\d{3})_1deg_K\.tif$",
        "scale": "8day",
        "min_trend": 2,
        "min_zscore": 2,
        "sd_floor": 0,
        "sd_inclusive": False,
        "fallback": "missing",
        "residual_float32": False,
        "center_years": True,
        "minimum": None,
    },
    "VPD_8day": {
        "directory": RAW_ROOT / "FDI_8DAY_1DEG/VPD",
        "pattern": r"_(?P<year>20\d{2})_(?P<period>\d{3})_8day_1deg\.tif$",
        "scale": "8day",
        "min_trend": 5,
        "min_zscore": 3,
        "sd_floor": 1e-6,
        "sd_inclusive": True,
        "fallback": "retain",
        "residual_float32": True,
        "center_years": False,
        "minimum": None,
    },
    "SM_8day": {
        "directory": RAW_ROOT / "FDI_8DAY_1DEG/SM",
        "pattern": r"_(?P<year>20\d{2})_(?P<period>\d{3})_8day_1deg\.tif$",
        "scale": "8day",
        "min_trend": 5,
        "min_zscore": 3,
        "sd_floor": 1e-6,
        "sd_inclusive": True,
        "fallback": "retain",
        "residual_float32": True,
        "center_years": False,
        "minimum": None,
    },
    "Ta_8day": {
        "directory": RAW_ROOT / "FDI_8DAY_1DEG/TA_ALLDAY",
        "pattern": r"_(?P<year>20\d{2})_(?P<period>\d{3})_8day_1deg\.tif$",
        "scale": "8day",
        "min_trend": 5,
        "min_zscore": 3,
        "sd_floor": 1e-6,
        "sd_inclusive": True,
        "fallback": "retain",
        "residual_float32": True,
        "center_years": False,
        "minimum": None,
    },
    "TSS_monthly": {
        "directory": RAW_ROOT / "TSS/TSS/monthly/ERA5_LST__ERA5_TA",
        "pattern": r"_(?P<year>20\d{2})_(?P<period>\d{2})_1deg_K\.tif$",
        "scale": "monthly",
        "min_trend": 2,
        "min_zscore": 2,
        "sd_floor": 0,
        "sd_inclusive": False,
        "fallback": "missing",
        "residual_float32": False,
        "center_years": True,
        "minimum": None,
    },
}


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


def standardize_period(values, settings):
    values = values.astype(np.float64)
    year = np.asarray(YEARS, dtype=float)
    if settings["center_years"]:
        year = year - year.mean()
    year = year[:, None, None]
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    sum_year = np.where(valid, year, 0).sum(axis=0)
    sum_value = np.where(valid, values, 0).sum(axis=0)
    sum_year_squared = np.where(valid, year * year, 0).sum(axis=0)
    sum_year_value = np.where(valid, year * values, 0).sum(axis=0)
    denominator = count * sum_year_squared - sum_year * sum_year
    can_fit = (count >= settings["min_trend"]) & (denominator > 1e-12)
    slope = np.zeros(GRID_SHAPE)
    slope[can_fit] = (count * sum_year_value - sum_year * sum_value)[can_fit] / denominator[can_fit]
    intercept = np.divide(
        sum_value - slope * sum_year, count, out=np.full(GRID_SHAPE, np.nan), where=count > 0
    )
    residual = values - (intercept + slope * year)
    if settings["fallback"] == "retain":
        residual[:, ~can_fit] = values[:, ~can_fit]
    else:
        residual[:, ~can_fit] = np.nan
    if settings["residual_float32"]:
        residual = residual.astype(np.float32).astype(np.float64)

    count = np.isfinite(residual).sum(axis=0)
    enough = count >= settings["min_zscore"]
    mean = np.full(GRID_SHAPE, np.nan)
    sd = mean.copy()
    mean[enough] = np.nanmean(residual[:, enough], axis=0)
    sd[enough] = np.nanstd(residual[:, enough], axis=0, ddof=1)
    if settings["sd_inclusive"]:
        usable = enough & (sd >= settings["sd_floor"])
    else:
        usable = enough & (sd > settings["sd_floor"])
    standardized = np.full_like(values, np.nan)
    standardized[:, usable] = (residual[:, usable] - mean[usable]) / sd[usable]
    return standardized.astype(np.float32)


def main():
    prepare_output(OUTPUT_DIR)
    retained = read_raster(MASK_FILE) == 1
    report = {}
    for name, settings in DATASETS.items():
        paths = index_rasters(settings["directory"], settings["pattern"], settings["scale"])
        periods = sorted({period for year, period in paths})
        valid_observations = 0
        for period in periods:
            values = np.stack([read_raster(paths[year, period]) for year in YEARS])
            values[:, ~retained] = np.nan
            values[(values == 65534) | (values == 65535)] = np.nan
            if settings["minimum"] is not None:
                values[values < settings["minimum"]] = np.nan
            standardized = standardize_period(values, settings)
            valid_observations += int(np.isfinite(standardized).sum())
            for index, year in enumerate(YEARS):
                path = OUTPUT_DIR / name / str(year) / f"{name}_z_{year}_{period:03d}_1deg.tif"
                write_raster(path, standardized[index], f"{name} standardized anomaly")
            print(f"{name}: period {period:03d}, all {len(YEARS)} years complete", flush=True)
        report[name] = {"rasters": len(paths), "valid_observations": valid_observations, "settings": settings}
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
