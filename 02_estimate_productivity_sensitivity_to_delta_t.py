"""Estimate standardized productivity sensitivity to daytime DeltaT (Fig. 1 and S1).

Use each product's native time scale: 8-day GOSIF GPP/SIF and monthly GloFlux/LHSIF.
Five contiguous calendar-year folds select the ridge penalty. A fixed-penalty
1999-draw year bootstrap gives a two-sided centered P value at each pixel.
The default input is archived annual cross-products. Use --from-rasters to start
from standardized input rasters and optionally --temperature-root results/01.
Use --stations results/05 for the separate FLUXNET GPP and canopy-conductance analyses.
The flat pixel index is row * 360 + column on the 180 by 360 north-up grid.

Inputs
------
The default grid fit uses archived yearly matrices and the vegetation pixel index.
For prepared rasters, first run script 01 both with and without --other-inputs. For
station fits, run script 05 for all sites.

Run from the directory containing this script
--------------------------------------------
    python 02_estimate_productivity_sensitivity_to_delta_t.py
    python 02_estimate_productivity_sensitivity_to_delta_t.py --from-rasters --prepared-root results/01
    python 02_estimate_productivity_sensitivity_to_delta_t.py --stations results/05

Outputs in results/02
---------------------
One CSV per productivity product, or stations.csv for the station fit. In grid mode
--point-only skips bootstrap P values.
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
    """Extract the standardized DeltaT coefficient from the productivity regression for
    each pixel.
    """
    name, predictors, response = equations[0]
    slopes = solve_ridge(total, predictors, response, penalties[0])
    return slopes[..., 1:2]


def from_rasters(
    index_path, variables, pixels, periods, scale, temperature_root=None, prepared_root=None
):
    """Build annual cross products on the joint finite sample for this equation set."""
    import rasterio

    inventory = pd.read_csv(index_path)
    if "scale" in inventory:
        inventory = inventory.loc[inventory.scale.eq(scale)]
    inventory = inventory[inventory.variable.isin(variables)]
    files = inventory.pivot(
        index=["year", "period"], columns="variable", values="path"
    ).sort_index()
    expected = pd.MultiIndex.from_product([range(2001, 2024), periods], names=["year", "period"])
    files = files.reindex(index=expected, columns=variables)
    if files.isna().any().any():
        raise ValueError("Missing year, period, or variable in the input index")
    width = len(variables) + 1
    moments = np.zeros((23, len(pixels), width, width))
    for index, (_, group) in enumerate(files.groupby(level="year")):
        values = np.ones((len(group), len(pixels), width))
        for t, paths in enumerate(group.itertuples(index=False, name=None)):
            for j, path in enumerate(paths):
                path = Path(path)
                if temperature_root and variables[j] in ["Ta", "DeltaT"]:
                    tail = path.parts[path.parts.index("standardized") + 1 :]
                    path = temperature_root.joinpath(scale, *tail)
                if prepared_root and variables[j] not in ["H", "LE"]:
                    tail = path.parts[path.parts.index("standardized") + 1 :]
                    path = prepared_root.joinpath(scale, *tail)
                with rasterio.open(path) as raster:
                    if (
                        raster.shape != (180, 360)
                        or raster.crs.to_epsg() != 4326
                        or tuple(raster.transform)[:6] != (1, 0, -180, 0, -1, 90)
                    ):
                        raise ValueError(f"Unexpected grid: {path}")
                    values[t, :, j + 1] = (
                        raster.read(1, masked=True).astype(float).filled(np.nan).ravel()[pixels]
                    )
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


def station_sensitivities(gc_folder):
    """Use response-specific day sets for FLUXNET GPP and active-day conductance."""
    sites = pd.read_csv(
        folder.parent
        / "Figure_releases_20261029/Figure_01/20261029_154sites/station_analysis/station_GPP_Gc_AI.csv"
    ).site
    rows = []
    predictors = ["DeltaT_z", "Ta_z", "TP_z", "SSRD_z", "CO2_z", "VPD_z"]
    for site in sites:
        path = (
            folder.parent
            / f"Figure2_preparation_20261028/20261028_shared_processed_batch/stations/{site}.csv"
        )
        days = pd.read_csv(path, float_precision="round_trip")
        daily = pd.read_csv(gc_folder / f"{site}.csv", float_precision="round_trip").set_index(
            "date"
        )
        conductance = days.date.map(daily.gc_m_s)
        trend = conductance.copy()
        for _, group in days.loc[conductance.notna()].groupby("doy"):
            y = conductance.loc[group.index].to_numpy()
            x = group.year.to_numpy(dtype=float)
            if len(y) >= 5 and np.var(x) > 1e-12:
                x -= x.mean()
                centered = y - y.mean()
                trend.loc[group.index] = centered - x * np.dot(x, centered) / np.dot(x, x)
        month_day = pd.to_datetime(days.date).dt.strftime("%m-%d")
        seasonal = trend.groupby(month_day)
        sd = seasonal.transform("std")
        days["Gc_z"] = ((trend - seasonal.transform("mean")) / sd).where(
            (sd >= 1e-6) & (seasonal.transform("count") >= 3)
        )
        for response in ["GPP", "Gc"]:
            cols = [predictors[0], response + "_z"] + predictors[1:]
            valid = days.loc[np.isfinite(days[cols]).all(axis=1)]
            matrices = []
            for _, group in valid.groupby("year"):
                design = np.column_stack([np.ones(len(group)), group[cols].to_numpy()])
                matrices.append(design.T @ design)
            if len(valid) < 30 or len(matrices) < 2:
                continue
            annual = np.stack(matrices)[:, None]
            inputs = equations[0][1]
            penalty = choose_penalty(annual, inputs, 2)
            beta = solve_ridge(annual.sum(0), inputs, 2, penalty)[0, 1]
            rows.append(
                dict(
                    site=site,
                    response=response,
                    n=len(valid),
                    years=len(matrices),
                    sensitivity=beta,
                    lambda_ridge=penalty[0],
                )
            )
    pd.DataFrame(rows).to_csv(output / "stations.csv", index=False)


if __name__ == "__main__":
    equations = [("lambda_ridge", [1, 3, 4, 5, 6, 7], 2)]
    metrics = ["sensitivity"]
    folder = Path(__file__).resolve().parent
    archive = (
        Path("D:/DATA/DeltaT_GPP_Analysis/Results/Temporary")
        / "20261024_F01_daytime_VPD_ridge_native"
    )
    output = folder / "results/02"
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
        "--stations",
        type=Path,
        help="Run station models from the folder of daily Gc files instead of the raster models",
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
        station_sensitivities(args.stations)
        raise SystemExit(0)
    pixels = np.load(folder.parent / "main_figures/monthly_tp_ssrd_20260917/source_data/flat.npy")
    products = [
        ("GOSIF_GPP", "8day"),
        ("GOSIF_SIF", "8day"),
        ("GLOFLUX_GPP", "monthly"),
        ("LHSIF_SIF", "monthly"),
    ]
    for product, scale in products:
        periods = range(1, 362, 8) if scale == "8day" else range(1, 13)
        variables = ["DeltaT", product, "Ta", "TP", "SSRD", "CO2", "VPD"]
        source = archive / "rows" / scale
        index = archive / "input_index.csv"
        pattern = product + "_*_moments.npy"
        result = process_product(source, pattern, index, variables, pixels, periods, scale, args)
        filename = product + ("_point.csv" if args.point_only else ".csv")
        result.to_csv(output / filename, index=False)
