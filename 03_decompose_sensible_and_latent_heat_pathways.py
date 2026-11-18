"""Estimate sensible-heat and latent-heat pathways (Fig. 3 and S5–S8).

Fit each flux on DeltaT and five controls. Fit GPP on both fluxes and the same
controls, then multiply the corresponding coefficients. All equations at a
pixel use the same observations. The direct GPP regression uses DeltaT and
the controls. Run --from-rasters to start from the input inventories, or use
the default annual cross-products. Run --stations for the shared daily site fits.
Inputs and the final run settings are at the end of this script.

Inputs
------
Use the archived annual matrices, or run both preprocessing modes of script 01 and
read H and LE from the raster inventory. Station mode uses the project daily records
directly.

Run from the directory containing this script
--------------------------------------------
    python 03_decompose_sensible_and_latent_heat_pathways.py
    python 03_decompose_sensible_and_latent_heat_pathways.py --from-rasters --prepared-root results/01
    python 03_decompose_sensible_and_latent_heat_pathways.py --stations

Outputs in results/03
---------------------
One CSV per GPP product, or stations.csv for the station fit. In grid mode --point-
only skips bootstrap P values.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def solve_ridge(moment, predictors, response, penalty):
    # Matrices store sums of cross products; column zero is the intercept.
    """Fit ridge regression from cross-products and return the unpenalized intercept
    followed by predictor slopes.
    """
    count = moment[..., 0, 0]
    sums = moment[..., predictors, 0]
    gram = (
        moment[..., predictors, :][..., :, predictors]
        - sums[..., None] * sums[..., None, :] / count[..., None, None]
    )
    rhs = moment[..., predictors, response] - sums * (moment[..., 0, response] / count)[..., None]
    system = gram + (count * penalty)[..., None, None] * np.eye(len(predictors))
    if np.isscalar(penalty) and penalty == 0:
        valid = np.linalg.matrix_rank(gram) == len(predictors)
        slopes = np.full(rhs.shape, np.nan)
        slopes[valid] = np.linalg.solve(system[valid], rhs[valid, ..., None])[..., 0]
    else:
        slopes = np.linalg.solve(system, rhs[..., None])[..., 0]
    intercept = (moment[..., 0, response] - (slopes * sums).sum(-1)) / count
    return np.concatenate([intercept[..., None], slopes], axis=-1)


def choose_penalty(annual, predictors, response):
    """Select each pixel's penalty by minimizing prediction error across five contiguous
    year folds.
    """
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


def calculate_block(annual, flat, bootstrap=True):
    """Return pixel estimates, sample counts and penalties, with year-bootstrap P values
    when requested.
    """
    total = annual.sum(0)
    penalties = [choose_penalty(annual, inputs, target) for _, inputs, target in equations]
    estimates = calculate_metrics(total, penalties)
    table = pd.DataFrame(estimates, columns=metrics)
    table.insert(0, "flat", flat)
    table["n"] = total[:, 0, 0].astype(int)
    for (name, _, _), penalty in zip(equations, penalties):
        table[name] = penalty
    if bootstrap:
        count = len(annual)
        draws = np.random.default_rng(20261021).multinomial(
            count, np.full(count, 1 / count), size=1999
        )
        probabilities = np.empty_like(estimates)
        for start in range(0, len(flat), 12):
            block = annual[:, start : start + 12]
            bootstrap_sum = (draws @ block.reshape(count, -1)).reshape(1999, *block.shape[1:])
            samples = calculate_metrics(bootstrap_sum, [v[start : start + 12] for v in penalties])
            observed = estimates[start : start + 12]
            extreme = np.abs(samples - observed) >= np.abs(observed)
            probabilities[start : start + 12] = (1 + extreme.sum(axis=0)) / 2000
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
    a_h, a_le = coefs[0][..., 1], coefs[1][..., 1]
    b_h, b_le = coefs[2][..., 1], coefs[2][..., 2]
    c_h, c_le = a_h * b_h, a_le * b_le
    return np.stack([a_h, a_le, b_h, b_le, c_h, c_le, c_h + c_le, coefs[3][..., 1]], axis=-1)


def from_rasters(
    index_path, variables, pixels, periods, scale, temperature_root=None, prepared_root=None
):
    """Build annual cross products on the joint finite sample for this equation set."""
    import rasterio

    inventory = pd.read_csv(index_path)
    if "scale" in inventory:
        inventory = inventory.loc[inventory.scale.eq(scale)]
    # Monthly GloFlux used common dates with the two other monthly products.
    mask_products = ["GOSIF_GPP", "LHSIF_SIF"] if scale == "monthly" else []
    indexed_variables = variables + mask_products
    inventory = inventory[inventory.variable.isin(indexed_variables)]
    files = inventory.pivot(
        index=["year", "period"], columns="variable", values="path"
    ).sort_index()
    expected = pd.MultiIndex.from_product([range(2001, 2024), periods], names=["year", "period"])
    files = files.reindex(index=expected, columns=indexed_variables)
    if files.isna().any().any():
        raise ValueError("Missing year, period, or variable in the input index")
    width = len(variables) + 1
    moments = np.zeros((23, len(pixels), width, width))
    for index, (_, group) in enumerate(files.groupby(level="year")):
        values = np.ones((len(group), len(pixels), width))
        shared_dates = np.ones((len(group), len(pixels)), dtype=bool)
        for t, paths in enumerate(group.itertuples(index=False, name=None)):
            for j, path in enumerate(paths):
                path = Path(path)
                if temperature_root and indexed_variables[j] in ["Ta", "DeltaT"]:
                    tail = path.parts[path.parts.index("standardized") + 1 :]
                    path = temperature_root.joinpath(scale, *tail)
                if prepared_root and indexed_variables[j] not in ["H", "LE"]:
                    tail = path.parts[path.parts.index("standardized") + 1 :]
                    path = prepared_root.joinpath(scale, *tail)
                with rasterio.open(path) as raster:
                    if (
                        raster.shape != (180, 360)
                        or raster.crs.to_epsg() != 4326
                        or tuple(raster.transform)[:6] != (1, 0, -180, 0, -1, 90)
                    ):
                        raise ValueError(f"Unexpected grid: {path}")
                    data = raster.read(1, masked=True).astype(float).filled(np.nan).ravel()[pixels]
                    if j < len(variables):
                        values[t, :, j + 1] = data
                    else:
                        shared_dates[t] &= np.isfinite(data)
        valid = shared_dates & np.isfinite(values).all(axis=-1)
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


def process_product(source, pattern, index, variables, pixels, periods, scale, args):
    """Read or rebuild annual cross-products for one product and combine the resulting
    pixel estimates into a table.
    """
    tables = []
    if args.from_rasters:
        blocks = from_rasters(
            index, variables, pixels, periods, scale, args.temperature_root, args.prepared_root
        )
        for annual, flat in blocks:
            tables.append(calculate_block(annual, flat, not args.point_only))
    else:
        for path in sorted(source.glob(pattern)):
            support = path.with_name(path.name.replace("moments", "support")).with_suffix(".npz")
            with np.load(support) as data:
                tables.append(calculate_block(np.load(path), data["flat"], not args.point_only))
    return pd.concat(tables, ignore_index=True)


def station_pathways():
    """Refit the H, LE and GPP equations using the same daily rows for each site."""
    cohort = pd.read_csv(
        folder.parent
        / "Figure_releases_20261029/Figure_02/20261029_154sites/inputs/station_A_coefficients.csv",
        usecols=["site", "AI"],
    )
    directory = (
        folder.parent / "Figure2_preparation_20261028/20261028_A_maps_OLS_v1/station_fitted_inputs"
    )
    variables = ["DeltaT", "GPP", "H", "LE", "Ta", "TP", "SSRD", "CO2", "VPD"]
    rows = []
    for station in cohort.itertuples(index=False):
        data = pd.read_csv(directory / f"{station.site}.csv", float_precision="round_trip")
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
                AI=station.AI,
                n=len(data),
                years=len(matrices),
                **dict(zip(metrics, estimates)),
            )
        )
    pd.DataFrame(rows).to_csv(output / "stations.csv", index=False)


if __name__ == "__main__":
    controls = [5, 6, 7, 8, 9]
    equations = [
        ("lambda_A_H", [1] + controls, 3),
        ("lambda_A_LE", [1] + controls, 4),
        ("lambda_B", [3, 4] + controls, 2),
        ("lambda_direct", [1] + controls, 2),
    ]
    metrics = ["a_H", "a_LE", "b_H", "b_LE", "C_H", "C_LE", "reconstructed", "direct"]
    folder = Path(__file__).resolve().parent
    archive = Path("D:/DATA/DeltaT_GPP_Analysis/Results/Temporary") / "Figure_03"
    output = folder / "results/03"
    output.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-rasters",
        action="store_true",
        help="Build yearly matrices from standardized rasters instead of reading their archived cross-products",
    )
    parser.add_argument(
        "--temperature-root",
        type=Path,
        help="Optional output directory of script 01 for raster input",
    )
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument(
        "--stations", action="store_true", help="Run the 154 site-level pathway models"
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Use script 01 outputs for all productivity and control inputs; H and LE use the inventory",
    )
    args = parser.parse_args()
    if args.temperature_root and args.prepared_root:
        parser.error("Choose either temperature-only or all prepared inputs")
    if args.stations:
        station_pathways()
        raise SystemExit(0)
    pixels = np.load(folder.parent / "main_figures/monthly_tp_ssrd_20260917/source_data/flat.npy")
    products = [("GOSIF_GPP", "8day"), ("GLOFLUX_GPP", "monthly")]
    for product, scale in products:
        periods = range(1, 362, 8) if scale == "8day" else range(1, 13)
        variables = ["DeltaT", product, "H", "LE", "Ta", "TP", "SSRD", "CO2", "VPD"]
        if product == "GOSIF_GPP":
            source = archive / "20261024_GOSIF_8day_daytime_HLE_ridge_VPD/rows"
            index = source.parent / "input_index.csv"
        else:
            source = archive / "20261026_T02_native_product_pathways/monthly_GLOFLUX_GPP/rows"
            index = source.parent.parent / "input_index_monthly.csv"
        pattern = "moments_*.npy"
        if product == "GOSIF_GPP":
            variables[1] = "GPP"
        result = process_product(source, pattern, index, variables, pixels, periods, scale, args)
        filename = product + ("_point.csv" if args.point_only else ".csv")
        result.to_csv(output / filename, index=False)
