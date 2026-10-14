"""Remove linear trends and standardize each DOY/month across years."""

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin


def detrend_series(years, observations, variable):
    """Remove the linear trend from one pixel at one DOY or calendar month."""
    valid = np.isfinite(observations)
    valid_years = years[valid]
    valid_values = observations[valid]
    n = len(valid_values)

    if variable == "TSS":
        valid_years = valid_years - years.mean()
        minimum_samples = 2
    else:
        minimum_samples = 5

    if n < minimum_samples:
        if variable == "TSS":
            return np.full(len(years), np.nan)
        return observations.copy()

    sum_year = sum(valid_years)
    sum_value = sum(valid_values)
    denominator = n * sum(valid_years**2) - sum_year**2
    if denominator <= 1e-12:
        if variable == "TSS":
            return np.full(len(years), np.nan)
        return observations.copy()

    slope = (n * sum(valid_years * valid_values) - sum_year * sum_value) / denominator
    intercept = (sum_value - slope * sum_year) / n
    trend = intercept + slope * valid_years
    detrended = np.full(len(years), np.nan)
    detrended[valid] = valid_values - trend
    return detrended


def standardize_series(detrended, variable):
    """Subtract the period mean and divide by its sample standard deviation."""
    if variable in ["VPD", "SM", "Ta"]:
        # Environmental residuals are stored with single precision.
        detrended = detrended.astype(np.float32)
        detrended = detrended.astype(np.float64)

    valid = np.isfinite(detrended)
    standardized = np.full(len(detrended), np.nan)
    if variable == "TSS":
        minimum_samples = 2
    else:
        minimum_samples = 3
    if valid.sum() < minimum_samples:
        return standardized

    period_mean = np.nanmean(detrended)
    period_sd = np.nanstd(detrended, ddof=1)
    if variable == "TSS":
        if period_sd <= 0:
            return standardized
    else:
        if period_sd < 1e-6:
            return standardized
    standardized[valid] = (detrended[valid] - period_mean) / period_sd
    return standardized


def standardize_period(values, dataset_name):
    """Apply the same calculation to every pixel of a complete raster stack."""
    if dataset_name.startswith("TSS"):
        variable = "TSS"
    elif dataset_name.startswith("GOSIF_GPP"):
        variable = "GPP"
    else:
        variable = dataset_name.split("_")[0]

    years = np.asarray(YEARS, dtype=float)
    standardized = np.full(values.shape, np.nan, dtype=np.float32)
    rows, columns = np.where(np.isfinite(values).any(axis=0))
    for row, column in zip(rows, columns):
        observations = values[:, row, column].astype(np.float64)
        detrended = detrend_series(years, observations, variable)
        standardized[:, row, column] = standardize_series(detrended, variable)
    return standardized


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


def index_rasters(directory, filename_type, time_scale):
    if time_scale not in ["8day", "monthly"]:
        raise ValueError("Time scale must be '8day' or 'monthly'")
    if filename_type not in ["gosif_raw", "tss_raw", "environment_raw", "dated_raster"]:
        raise ValueError(f"Unknown filename convention: {filename_type}")
    if filename_type in ["gosif_raw", "environment_raw"] and time_scale != "8day":
        raise ValueError(f"{filename_type} filenames require the 8day time scale")
    if time_scale == "8day":
        periods = list(range(1, 366, 8))
    else:
        periods = list(range(1, 13))

    paths = {}
    for path in sorted(directory.rglob("*.tif")):
        parts = path.stem.split("_")
        if filename_type == "gosif_raw":
            # GOSIF_GPP_2001001_Mean_1deg.tif
            if not path.name.startswith("GOSIF_GPP_") or not path.name.endswith("_Mean_1deg.tif"):
                continue
            date_text = parts[2]
            if len(date_text) != 7 or not date_text.isdigit():
                continue
            year_text = date_text[:4]
            period_text = date_text[4:]
        elif filename_type == "tss_raw":
            # ERA5_LST__ERA5_TA_2001_001_1deg_K.tif
            if len(parts) < 5 or parts[-2:] != ["1deg", "K"]:
                continue
            year_text = parts[-4]
            period_text = parts[-3]
        elif filename_type == "environment_raw":
            # VPD_2001_001_8day_1deg.tif
            if len(parts) < 5 or parts[-2:] != ["8day", "1deg"]:
                continue
            year_text = parts[-4]
            period_text = parts[-3]
        else:
            # GOSIF_GPP_8day_z_2001_001_1deg.tif
            if len(parts) < 4 or parts[-1] != "1deg":
                continue
            year_text = parts[-3]
            period_text = parts[-2]

        if not year_text.isdigit() or not period_text.isdigit():
            continue
        year = int(year_text)
        period = int(period_text)
        if year not in YEARS:
            continue
        date = (year, period)
        if period not in periods or date in paths:
            raise ValueError(f"Invalid or duplicate date: {path}")
        paths[date] = path

    ordered_paths = {}
    for year in YEARS:
        for period in periods:
            date = (year, period)
            if date not in paths:
                raise ValueError(f"Missing raster for {year}, period {period}: {directory}")
            ordered_paths[date] = paths[date]
    return ordered_paths


def process_dataset(dataset_name, input_directory, filename_type, time_scale, vegetation_mask):
    paths = index_rasters(input_directory, filename_type, time_scale)
    if time_scale == "8day":
        periods = range(1, 366, 8)
    else:
        periods = range(1, 13)

    valid_observations = 0
    for period in periods:
        yearly_rasters = []
        for year in YEARS:
            yearly_rasters.append(read_raster(paths[year, period]))
        values = np.stack(yearly_rasters)
        values[:, ~vegetation_mask] = np.nan
        values[(values == 65534) | (values == 65535)] = np.nan
        if dataset_name == "GOSIF_GPP_8day":
            values[values < 0] = np.nan

        standardized = standardize_period(values, dataset_name)
        valid_observations += int(np.isfinite(standardized).sum())
        for index, year in enumerate(YEARS):
            if time_scale == "8day":
                date = f"{year}_{period:03d}"
            else:
                date = f"{year}_{period:02d}"
            run_name = f"{dataset_name}_{YEARS[0]}-{YEARS[-1]}"
            filename = f"{run_name}_z_{date}_1deg.tif"
            output_file = OUTPUT_DIR / run_name / str(year) / filename
            write_raster(output_file, standardized[index], f"{dataset_name} standardized anomaly")
        print(f"{dataset_name}: period {period:03d}, all {len(YEARS)} years complete", flush=True)

    return {"rasters": len(paths), "valid_observations": valid_observations}


def main():
    if len(YEARS) < 2:
        raise ValueError("Select at least two years for detrending")
    available = [dataset[0] for dataset in DATASETS]
    if not RUN_DATASETS or len(RUN_DATASETS) != len(set(RUN_DATASETS)):
        raise ValueError("RUN_DATASETS must contain unique dataset names")
    for name in RUN_DATASETS:
        if name not in available:
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        prepare_output(OUTPUT_DIR / f"{name}_{YEARS[0]}-{YEARS[-1]}")
    vegetation_mask = read_raster(MASK_FILE) == 1
    for dataset_name, input_directory, filename_type, time_scale in DATASETS:
        if dataset_name not in RUN_DATASETS:
            continue
        result = process_dataset(
            dataset_name, input_directory, filename_type, time_scale, vegetation_mask
        )
        summary_file = OUTPUT_DIR / f"{dataset_name}_{YEARS[0]}-{YEARS[-1]}" / "run_summary.json"
        summary_file.write_text(json.dumps({dataset_name: result}, indent=2), encoding="utf-8")


# Input paths and run settings
RUN_DATASETS = ["GOSIF_GPP_8day", "TSS_8day", "VPD_8day", "SM_8day", "Ta_8day", "TSS_monthly"]
START_YEAR = 2001
END_YEAR = 2023
YEARS = list(range(START_YEAR, END_YEAR + 1))
CODE_DIR = Path(__file__).resolve().parent
RAW_ROOT = Path("D:/DATA")
MASK_FILE = CODE_DIR / "inputs/vegetation_mask_1deg.tif"
OUTPUT_DIR = CODE_DIR / "outputs/01_standardized"
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
OVERWRITE = False

# Dataset name, input directory, filename convention, time step
DATASETS = [
    ("GOSIF_GPP_8day", RAW_ROOT / "GOSIF_GPP_8day_1deg", "gosif_raw", "8day"),
    ("TSS_8day", RAW_ROOT / "TSS/TSS/8day/ERA5_LST__ERA5_TA", "tss_raw", "8day"),
    ("VPD_8day", RAW_ROOT / "FDI_8DAY_1DEG/VPD", "environment_raw", "8day"),
    ("SM_8day", RAW_ROOT / "FDI_8DAY_1DEG/SM", "environment_raw", "8day"),
    ("Ta_8day", RAW_ROOT / "FDI_8DAY_1DEG/TA_ALLDAY", "environment_raw", "8day"),
    ("TSS_monthly", RAW_ROOT / "TSS/TSS/monthly/ERA5_LST__ERA5_TA", "tss_raw", "monthly"),
]

if __name__ == "__main__":
    main()
