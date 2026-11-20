"""Prepare 8-day and monthly observations and write standardized, vegetation-masked 1° grids."""

import json
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


PATHS_FILE = Path(__file__).resolve().parent / "01_data_paths.txt"
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
TIME_ORIGIN = date(2000, 1, 1)


def read_co2_source(path):
    """Read NOAA decimal years and CO₂ concentrations (ppm) at 41 sine-latitude nodes."""
    table = np.loadtxt(path)
    decimal_years = table[:, 0]
    concentrations = table[:, 1::2]  # Each concentration is followed by its uncertainty.
    if table.shape[1] != 83 or not np.all(np.diff(decimal_years) > 0):
        raise ValueError("Expected ordered NOAA dates and 41 concentration–uncertainty pairs.")

    days = []
    for decimal_year in decimal_years:
        year = int(decimal_year)
        january_1 = date(year, 1, 1)
        days_in_year = (date(year + 1, 1, 1) - january_1).days
        days.append((january_1 - TIME_ORIGIN).days + (decimal_year - year) * days_in_year)
    return np.array(days), concentrations


def prepare_co2_grids(source_days, concentrations, years, scale, period):
    """Return one global CO₂ grid per year for the same 8-day period or calendar month."""
    source_sin_lat = np.linspace(-1, 1, 41)
    target_sin_lat = np.sin(np.deg2rad(89.5 - np.arange(180)))
    grids = []
    for year in years:
        year = int(year)
        if scale == "8day":
            start = date(year, 1, 1) + timedelta(days=period - 1)
            end = min(start + timedelta(days=8), date(year + 1, 1, 1))
        else:
            start = date(year, period, 1)
            end = start + timedelta(days=monthrange(year, period)[1])
        first_day = (start - TIME_ORIGIN).days
        end_day = (end - TIME_ORIGIN).days
        if first_day < source_days[0] or end_day > source_days[-1]:
            raise ValueError(f"NOAA CO₂ data do not cover {start} through {end}.")

        # Include both boundaries and every native observation between them.
        within = (source_days > first_day) & (source_days < end_day)
        days = np.r_[first_day, source_days[within], end_day]
        interval_values = np.column_stack([
            np.interp(days, source_days, series) for series in concentrations.T
        ])
        mean_ppm = np.trapezoid(interval_values, x=days, axis=0) / (end_day - first_day)

        latitude_ppm = np.interp(target_sin_lat, source_sin_lat, mean_ppm)
        grid = np.repeat(latitude_ppm[:, None], 360, axis=1)
        grids.append(grid.astype("float32"))  # Match the precision of the 1° raster inputs.
    return np.stack(grids)


def read_rasters(file_pattern, years, period):
    """Read one global field per year for the same month or 8-day period."""
    values = []
    for year in years:
        path = PATHS_FILE.parent / file_pattern.format(year=year, period=period)
        with rasterio.open(path) as raster:
            if raster.shape != GRID_SHAPE or raster.transform != GRID_TRANSFORM or raster.crs != "EPSG:4326":
                raise ValueError(f"Expected the global 1° grid in {path}.")
            field = raster.read(1, masked=True).astype("float64").filled(np.nan)
        values.append(field)
    return np.stack(values)


def screen_productivity(values):
    """Remove outliers across years for the same grid cell and calendar period."""
    screened = values.copy()
    enough = np.isfinite(values).sum(axis=0) >= 5
    series = values[:, enough]
    median = np.nanmedian(series, axis=0)
    deviation = np.abs(series - median)
    mad = np.nanmedian(deviation, axis=0)
    outliers = (mad > 0) & (deviation > 3 * 1.4826 * mad)
    screened[:, enough] = np.where(outliers, np.nan, series)
    return screened


def remove_linear_trend(values, years):
    """Fit a separate least-squares line to each cell's valid years and return residuals."""
    values = np.asarray(values, dtype="float64")
    valid = np.isfinite(values)
    # A line has two coefficients. At least three years leave residual variation to estimate.
    usable = valid.sum(axis=0) >= 3
    residuals = np.full(values.shape, np.nan)
    if not usable.any():
        return residuals

    observed = np.where(valid[:, usable], values[:, usable], np.nan)
    elapsed_years = years - years[0]
    observed_years = np.where(valid[:, usable], elapsed_years[:, None], np.nan)
    year_offset = observed_years - np.nanmean(observed_years, axis=0)
    value_offset = observed - np.nanmean(observed, axis=0)
    slope = np.nansum(year_offset * value_offset, axis=0) / np.nansum(year_offset**2, axis=0)
    residuals[:, usable] = value_offset - slope * year_offset
    return residuals


def standardize(residuals):
    """Subtract the residual mean and divide by the sample standard deviation."""
    usable = np.isfinite(residuals).any(axis=0)
    anomalies = np.full(residuals.shape, np.nan, dtype="float32")
    if usable.any():
        series = residuals[:, usable]
        mean = np.nanmean(series, axis=0)
        std = np.nanstd(series, axis=0, ddof=1)
        std[std == 0] = np.nan
        anomalies[:, usable] = (series - mean) / std
    return anomalies


def write_masked_rasters(anomalies, vegetation_mask, output_dir, scale, variable, years, period):
    """Apply the vegetation mask after standardization and retain the full global grid."""
    profile = dict(
        driver="GTiff", count=1, height=GRID_SHAPE[0], width=GRID_SHAPE[1],
        crs="EPSG:4326", transform=GRID_TRANSFORM,
        dtype="float32", nodata=np.nan, compress="deflate",
    )
    folder = output_dir / scale / variable
    folder.mkdir(parents=True, exist_ok=True)
    interval = f"DOY{period:03d}" if scale == "8day" else f"M{period:02d}"
    for year, field in zip(years, anomalies):
        grid = np.where(vegetation_mask, field, np.nan)
        path = folder / f"{variable}_{year}_{interval}_zscore.tif"
        with rasterio.open(path, "w", **profile) as raster:
            raster.write(grid, 1)


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    VEG_MASK_PATH = PATHS_FILE.parent / paths["VEG_MASK_PATH"]
    CO2_SOURCE_PATH = PATHS_FILE.parent / paths["CO2_SOURCE_PATH"]
    raw_files = {"8day": paths["RAW_8DAY_FILES"], "monthly": paths["RAW_MONTHLY_FILES"]}
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    years = np.arange(2001, 2024)

    co2_days, co2_ppm = read_co2_source(CO2_SOURCE_PATH)
    vegetation_mask = np.zeros(GRID_SHAPE, dtype=bool)
    vegetation_mask.ravel()[np.load(VEG_MASK_PATH)] = True
    shared_variables = ["DeltaT", "Ta", "TP", "SSRD", "VPD", "CO2"]
    products = {"8day": ["GOSIF_GPP", "GOSIF_SIF"], "monthly": ["GLOFLUX_GPP", "LHSIF_SIF"]}

    for scale, productivity_products in products.items():
        periods = range(1, 362, 8) if scale == "8day" else range(1, 13)
        for variable in shared_variables + productivity_products:
            for period in periods:
                if variable == "CO2":
                    values = prepare_co2_grids(co2_days, co2_ppm, years, scale, period)
                else:
                    values = read_rasters(raw_files[scale][variable], years, period)
                if variable in productivity_products:
                    values = screen_productivity(values)
                if scale == "8day" and variable in ("TP", "SSRD"):
                    upper_limit = 2 if variable == "TP" else 1e9
                    values[(values < 0) | (values > upper_limit)] = np.nan
                residuals = remove_linear_trend(values, years)
                anomalies = standardize(residuals)
                write_masked_rasters(anomalies, vegetation_mask, OUTPUT_DIR, scale, variable, years, period)
