"""Decompose GPP sensitivity to ΔT into sensible and latent heat pathways in gridded products and FLUXNET observations."""

import json
from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


PATHS_FILE = Path(__file__).resolve().parent / "03_data_paths.txt"


# Matrix columns: intercept, ΔT, GPP, H, LE, Ta, TP, SSRD, CO2, VPD.
delta_t, gpp, sensible_heat, latent_heat = 1, 2, 3, 4
controls = [5, 6, 7, 8, 9]
equations = [
    ("lambda_A_H", [delta_t] + controls, sensible_heat),
    ("lambda_A_LE", [delta_t] + controls, latent_heat),
    ("lambda_B", [sensible_heat, latent_heat] + controls, gpp),
    ("lambda_direct", [delta_t] + controls, gpp),
]
metrics = ["a_H", "a_LE", "b_H", "b_LE", "C_H", "C_LE", "S_recon", "S_std"]


def solve_ridge(moment, predictors, response, penalty):
    """Fit the ridge slopes from cross-products. Column zero holds the unpenalized intercept."""
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
        # An unpenalized fit requires a full-rank design.
        valid = np.linalg.matrix_rank(gram) == len(predictors)
        slopes = np.full(cross_covariance.shape, np.nan)
        slopes[valid] = np.linalg.solve(system[valid], cross_covariance[valid, ..., None])[..., 0]
    else:
        slopes = np.linalg.solve(system, cross_covariance[..., None])[..., 0]
    intercept = (moment[..., 0, response] - (slopes * sums).sum(-1)) / count
    return np.concatenate([intercept[..., None], slopes], axis=-1)


def choose_penalty(annual, predictors, response):
    """Minimize the sum of held-out squared errors across five contiguous year folds."""
    total = annual.sum(0)
    grid = np.r_[0, np.logspace(-5, 2, 29)]
    scores = np.zeros((len(grid), len(total)))
    columns = [0] + predictors
    for fold in np.array_split(np.arange(len(annual)), 5):
        test = annual[fold].sum(0)
        for i, penalty in enumerate(grid):
            coef = solve_ridge(total - test, predictors, response, penalty)
            cross = test[:, columns, response]
            gram = test[:, columns, :][:, :, columns]
            residual_ss = (
                test[:, response, response]
                - 2 * (coef * cross).sum(-1)
                + np.einsum("pi,pij,pj->p", coef, gram, coef)
            )
            scores[i] += np.where(np.isfinite(residual_ss), residual_ss, np.inf)
    return grid[scores.argmin(axis=0)]


def grid_pathways(annual, flat):
    """Fit the four equations and calculate two-sided P values by resampling whole years."""
    total = annual.sum(0)
    penalties = [choose_penalty(annual, inputs, target) for _, inputs, target in equations]
    estimates = calculate_metrics(total, penalties)
    table = pd.DataFrame(estimates, columns=metrics)
    table.insert(0, "flat", flat)
    table["n"] = total[:, 0, 0].astype(int)
    for (name, _, _), penalty in zip(equations, penalties):
        table[name] = penalty
    count = len(annual)
    n_draws = 2000
    draws = np.random.default_rng(20261021).multinomial(count, np.full(count, 1 / count), size=n_draws)
    probabilities = np.empty_like(estimates)
    for start in range(0, len(flat), 12):
        block = annual[:, start : start + 12]
        bootstrap_sum = (draws @ block.reshape(count, -1)).reshape(n_draws, *block.shape[1:])
        samples = calculate_metrics(bootstrap_sum, [v[start : start + 12] for v in penalties])
        observed = estimates[start : start + 12]
        extreme = np.abs(samples - observed) >= np.abs(observed)
        probabilities[start : start + 12] = (1 + extreme.sum(axis=0)) / (n_draws + 1)
    for i, name in enumerate(metrics):
        table["p_" + name] = probabilities[:, i]
    return table


def calculate_metrics(total, penalties):
    """Return flux responses, partial GPP slopes, their pathway products, their sum and
    the direct-fit sensitivity.
    """
    coefs = [
        solve_ridge(total, inputs, target, penalty)
        for (_, inputs, target), penalty in zip(equations, penalties)
    ]
    # ΔT-to-flux slopes (a) times partial flux-to-GPP slopes (b).
    a_h, a_le = coefs[0][..., 1], coefs[1][..., 1]
    b_h, b_le = coefs[2][..., 1], coefs[2][..., 2]
    c_h, c_le = a_h * b_h, a_le * b_le
    return np.stack([a_h, a_le, b_h, b_le, c_h, c_le, c_h + c_le, coefs[3][..., 1]], axis=-1)


def annual_pixel_blocks(index_path, variables, pixels, periods, scale):
    """Build yearly cross-products from jointly valid periods for each grid cell."""
    inventory = pd.read_csv(index_path)
    if "scale" in inventory:
        inventory = inventory.loc[inventory.scale.eq(scale)]
    inventory = inventory.rename(columns={"month": "period"})
    if scale == "8day":
        inventory["variable"] = inventory["variable"].replace({"GPP": "GOSIF_GPP"})
    inventory = inventory[inventory.variable.isin(variables)]
    files = inventory.pivot(
        index=["year", "period"], columns="variable", values="path"
    ).sort_index()
    expected = pd.MultiIndex.from_product([range(2001, 2024), periods], names=["year", "period"])
    files = files.reindex(index=expected, columns=variables)
    if files.isna().any().any():
        raise ValueError("Missing year, period, or variable in the input index")
    width = len(variables) + 1
    # Each year and pixel stores the joint products of the intercept and all variables.
    moments = np.zeros((23, len(pixels), width, width))
    for index, year in enumerate(range(2001, 2024)):
        values = np.ones((len(periods), len(pixels), width))
        for period_index, period in enumerate(periods):
            for column, variable in enumerate(variables):
                path = index_path.parent / files.at[(year, period), variable]
                with rasterio.open(path) as raster:
                    if (
                        raster.shape != (180, 360)
                        or raster.crs.to_epsg() != 4326
                        or tuple(raster.transform)[:6] != (1, 0, -180, 0, -1, 90)
                    ):
                        raise ValueError(f"Unexpected grid: {path}")
                    data = raster.read(1, masked=True).astype(float).filled(np.nan).ravel()[pixels]
                    values[period_index, :, column + 1] = data
        valid = np.isfinite(values).all(axis=-1)
        values[~valid] = 0
        moments[index] = np.einsum("tpi,tpj->pij", values, values)
    per_year = moments[:, :, 0, 0]
    total = per_year.sum(0)
    keep = total >= 30
    for fold in np.array_split(np.arange(23), 5):
        keep &= total - per_year[fold].sum(0) > 1
    for start in range(0, len(pixels), 512):
        select = np.flatnonzero(keep[start : start + 512]) + start
        if len(select):
            yield moments[:, select], pixels[select]


def station_pathways(cohort_path, daily_anomaly_dir):
    """Refit the H, LE and GPP equations using the same daily rows for each site."""
    cohort = pd.read_csv(cohort_path, usecols=["site", "AI"])
    variables = ["DeltaT", "GPP", "H", "LE", "Ta", "TP", "SSRD", "CO2", "VPD"]
    rows = []
    for station in cohort.itertuples(index=False):
        data = pd.read_csv(daily_anomaly_dir / f"{station.site}.csv", float_precision="round_trip")
        matrices = []
        for _, group in data.groupby("year"):
            values = group[[name + "_z" for name in variables]].to_numpy()
            design = np.column_stack([np.ones(len(group)), values])
            matrices.append(design.T @ design)
        moments = np.stack(matrices)[:, None]
        penalties = [choose_penalty(moments, inputs, target) for _, inputs, target in equations]
        estimates = calculate_metrics(moments.sum(0), penalties)[0]
        rows.append(
            dict(
                site=station.site,
                AI_station=station.AI,
                n=len(data),
                years=len(matrices),
                **dict(zip(metrics, estimates)),
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    VEG_MASK_PATH = PATHS_FILE.parent / paths["VEG_MASK_PATH"]
    GOSIF_RASTER_INDEX_PATH = PATHS_FILE.parent / paths["GOSIF_RASTER_INDEX_PATH"]
    GLOFLUX_RASTER_INDEX_PATH = PATHS_FILE.parent / paths["GLOFLUX_RASTER_INDEX_PATH"]
    STATION_COHORT_PATH = PATHS_FILE.parent / paths["STATION_COHORT_PATH"]
    STATION_DAILY_ANOMALY_DIR = PATHS_FILE.parent / paths["STATION_DAILY_ANOMALY_DIR"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pixels = np.load(VEG_MASK_PATH)  # Flat indices of vegetated 1° grid cells.
    products = [
        ("GOSIF_GPP", "8day", GOSIF_RASTER_INDEX_PATH),
        ("GLOFLUX_GPP", "monthly", GLOFLUX_RASTER_INDEX_PATH),
    ]
    for product, scale, input_index in products:
        periods = range(1, 362, 8) if scale == "8day" else range(1, 13)
        variables = ["DeltaT", product, "H", "LE", "Ta", "TP", "SSRD", "CO2", "VPD"]
        blocks = annual_pixel_blocks(input_index, variables, pixels, periods, scale)
        tables = [grid_pathways(annual, cell_ids) for annual, cell_ids in blocks]
        pd.concat(tables, ignore_index=True).to_csv(OUTPUT_DIR / f"{product}.csv", index=False)
    stations = station_pathways(STATION_COHORT_PATH, STATION_DAILY_ANOMALY_DIR)
    stations.to_csv(OUTPUT_DIR / "stations.csv", index=False)
