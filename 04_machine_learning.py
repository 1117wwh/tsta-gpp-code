"""Fit XGBoost models for monthly sensitivity and calculate test-set TreeSHAP values."""

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor, DMatrix

CODE_DIR = Path(__file__).resolve().parent
YEARS = list(range(2001, 2024))
GRID_SHAPE = (180, 360)
GRID_TRANSFORM = from_origin(-180, 90, 1, 1)
OVERWRITE = False

INPUT_DIR = CODE_DIR / "inputs/04_machine_learning"
OUTPUT_DIR = CODE_DIR / "outputs/04_machine_learning"
TARGET_FILE = INPUT_DIR / "observed_sensitivity_monthly_1deg.tif"
FEATURES = [
    "vpd_mean",
    "sm_mean",
    "ta_mean",
    "aridity_index_log10",
    "agb_2010",
    "rootzone_storage_log1p",
    "lai_2002_2023_mean",
    "canopy_height_2020_eth",
    "soil_soc_0_30cm",
    "soil_ph_0_30cm",
    "soil_clay_0_30cm",
    "merit_dem_mean",
]
MODEL_FEATURES = {
    "M1_climate": FEATURES[:4],
    "M2_climate_vegetation": FEATURES[:8],
    "M3_all_predictors": FEATURES,
}
TEST_FRACTION = 0.30
SPLIT_SEED = 42
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260928
MODEL_PARAMETERS = {
    "n_estimators": 700,
    "learning_rate": 0.035,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
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


def save_pixel_map(directory, name, values, pixels):
    grid = np.full(GRID_SHAPE, np.nan, dtype=np.float64)
    grid.ravel()[pixels] = values
    write_raster(directory / f"{name}_1deg.tif", grid, name)


def main():
    prepare_output(OUTPUT_DIR)
    target_grid = read_raster(TARGET_FILE).ravel()
    predictor_grids = np.column_stack(
        [read_raster(INPUT_DIR / f"{feature}_1deg.tif").ravel() for feature in FEATURES]
    )
    valid = np.isfinite(target_grid) & np.isfinite(predictor_grids).all(axis=1)
    pixels = np.flatnonzero(valid)
    if len(pixels) < 10 or np.var(target_grid[pixels]) == 0:
        raise ValueError("Insufficient complete pixels or constant target")
    target = target_grid[pixels]
    predictors = pd.DataFrame(predictor_grids[pixels], columns=FEATURES)
    train, test = train_test_split(np.arange(len(pixels)), test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    split = np.ones(len(pixels))
    split[test] = 2
    save_pixel_map(OUTPUT_DIR, "sample_split_1train_2test", split, pixels)
    excluded = np.flatnonzero(np.isfinite(target_grid) & ~valid)
    save_pixel_map(OUTPUT_DIR, "excluded_missing_predictor", np.ones(len(excluded)), excluded)
    performance = []
    predictions_by_model = {}
    for name, features in MODEL_FEATURES.items():
        model = XGBRegressor(**MODEL_PARAMETERS)
        model.fit(predictors.loc[train, features], target[train])
        predicted = model.predict(predictors[features])
        predictions_by_model[name] = predicted
        save_pixel_map(OUTPUT_DIR / name, "prediction", predicted, pixels)
        save_pixel_map(
            OUTPUT_DIR / name,
            "test_residual_observed_minus_predicted",
            target[test] - predicted[test],
            pixels[test],
        )
        for split_name, indices in [("train", train), ("test", test)]:
            observed = target[indices]
            fitted = predicted[indices]
            performance.append(
                {
                    "model": name,
                    "split": split_name,
                    "n": len(indices),
                    "r2": r2_score(observed, fitted),
                    "rmse": float(np.sqrt(mean_squared_error(observed, fitted))),
                    "mae": mean_absolute_error(observed, fitted),
                }
            )
        model.save_model(str(OUTPUT_DIR / name / "model.json"))
        if name == "M3_all_predictors":
            contributions = model.get_booster().predict(
                DMatrix(predictors.loc[test, features]), pred_contribs=True
            )
            np.testing.assert_allclose(contributions.sum(axis=1), predicted[test], rtol=1e-5, atol=2e-6)
            importance = {}
            for column, feature in enumerate(FEATURES):
                save_pixel_map(OUTPUT_DIR / name / "SHAP", feature, contributions[:, column], pixels[test])
                importance[feature] = float(np.abs(contributions[:, column]).mean())
            save_pixel_map(OUTPUT_DIR / name / "SHAP", "base_value", contributions[:, -1], pixels[test])
        print(f"{name}: {performance[-1]}", flush=True)

    # Resample the same test pixels for all models; fitted models stay fixed.
    ordered_test = np.sort(test)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_scores = np.full((BOOTSTRAP_REPEATS, len(MODEL_FEATURES)), np.nan)
    for repeat in range(BOOTSTRAP_REPEATS):
        sample = ordered_test[generator.integers(0, len(test), len(test))]
        observed = target[sample]
        total_variation = np.sum((observed - observed.mean()) ** 2)
        if total_variation == 0:
            continue
        for column, name in enumerate(MODEL_FEATURES):
            squared_error = np.sum((observed - predictions_by_model[name][sample]) ** 2)
            bootstrap_scores[repeat, column] = 1 - squared_error / total_variation
    intervals = []
    previous_score = 0
    previous_bootstrap = np.zeros(BOOTSTRAP_REPEATS)
    for column, name in enumerate(MODEL_FEATURES):
        score = next(row["r2"] for row in performance if row["model"] == name and row["split"] == "test")
        lower, upper = np.nanquantile(bootstrap_scores[:, column], [0.025, 0.975])
        gain_lower, gain_upper = np.nanquantile(
            bootstrap_scores[:, column] - previous_bootstrap, [0.025, 0.975]
        )
        intervals.append(
            {
                "model": name,
                "r2": score,
                "r2_ci_low": lower,
                "r2_ci_high": upper,
                "gain": score - previous_score,
                "gain_ci_low": gain_lower,
                "gain_ci_high": gain_upper,
            }
        )
        previous_score = score
        previous_bootstrap = bootstrap_scores[:, column]
    report = {
        "candidate_pixels": int(np.isfinite(target_grid).sum()),
        "complete_pixels": len(pixels),
        "excluded_pixels": len(excluded),
        "train_pixels": len(train),
        "test_pixels": len(test),
        "performance": performance,
        "test_bootstrap_intervals": intervals,
        "mean_absolute_test_SHAP": importance,
        "model_parameters": MODEL_PARAMETERS,
        "split_seed": SPLIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
    }
    (OUTPUT_DIR / "model_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Complete: {len(train)} training pixels, {len(test)} test pixels", flush=True)


if __name__ == "__main__":
    main()
