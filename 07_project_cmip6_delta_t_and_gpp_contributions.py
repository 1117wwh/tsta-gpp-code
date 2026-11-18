"""Reproduce annual CMIP6 DeltaT sensitivities and GPP contributions (Fig. 5, S9, S14, S20–S23).

Inputs are physical annual arrays in year x pixel x variable order:
DeltaT, GPP, Ta, precipitation, radiation, CO2, VPD. They are prepared upstream
with each model calendar: sum GPP over the year and day-weight the other fields.
Annualized GPP is g C m-2 year-1. For each validation fold estimate trend and
scale on training years. Use the accepted complete support common to both
scenarios and both trend specifications. Report the detrended sensitivity.
Convert its coefficient to physical GPP per degree before multiplying by
the DeltaT anomaly. Normalize by historical GPP on the same spatial domain;
calculate model endpoints first, then weight the twelve models equally.

Inputs
------
The inputs directory contains prepared annual historical and scenario NPZ files. The
selected path identifies the model selection JSON. Regional summaries and recent
trends require the project spatial and ERA5 reference files.

Run from the directory containing this script
--------------------------------------------
    python 07_project_cmip6_delta_t_and_gpp_contributions.py

Outputs in results/07
---------------------
Per-model pixel tables, annual.csv, model_endpoints.csv, ensemble_endpoints.csv,
regional.csv and recent_trends.csv.
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


def detrend_and_zscore(stack: np.ndarray, years: np.ndarray, detrend=True):
    """Return Z-scores, scales, centers and slopes. Set detrend=False to standardize with
    trends retained.
    """
    if stack.ndim != 3 or stack.shape[0] != years.size:
        raise ValueError(f"Expected (time, rows, cols), got {stack.shape}")
    y = stack.astype(np.float64, copy=False)
    finite = np.isfinite(y)
    n = finite.sum(axis=0).astype(np.float64)
    t = years.astype(np.float64) - float(np.mean(years))
    t3 = t[:, None, None]
    sum_y = np.where(finite, y, 0.0).sum(axis=0)
    sum_t = np.where(finite, t3, 0.0).sum(axis=0)
    sum_tt = np.where(finite, t3 * t3, 0.0).sum(axis=0)
    sum_ty = np.where(finite, t3 * y, 0.0).sum(axis=0)
    denominator = n * sum_tt - sum_t * sum_t
    good_trend = (n >= 2.0) & (denominator > 0.0)
    slope = np.zeros_like(sum_y)
    intercept = np.full_like(sum_y, np.nan)
    slope[good_trend] = (
        n[good_trend] * sum_ty[good_trend] - sum_t[good_trend] * sum_y[good_trend]
    ) / denominator[good_trend]
    if not detrend:
        slope[good_trend] = 0.0
    intercept[good_trend] = (sum_y[good_trend] - slope[good_trend] * sum_t[good_trend]) / n[
        good_trend
    ]
    fitted = intercept[None, :, :] + slope[None, :, :] * t3
    residual = np.where(finite & good_trend[None, :, :], y - fitted, np.nan)
    n_residual = np.isfinite(residual).sum(axis=0).astype(np.float64)
    sum_residual = np.nansum(residual, axis=0)
    mean_residual = np.divide(
        sum_residual,
        n_residual,
        out=np.full_like(sum_residual, np.nan),
        where=n_residual > 0,
    )
    centered = residual - mean_residual[None, :, :]
    sum_squared = np.nansum(centered * centered, axis=0)
    variance = np.divide(
        sum_squared,
        n_residual - 1.0,
        out=np.full_like(sum_squared, np.nan),
        where=n_residual > 1,
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    use = (n_residual >= 2.0) & np.isfinite(std) & (std > 0.0)
    zscore = np.full_like(centered, np.nan)
    zscore[:, use] = centered[:, use] / std[use]
    return zscore, std, intercept + mean_residual, slope


def standardized(raw, training, detrend=True):
    """Fit trends and scales on training years, then apply them to all years at each
    pixel.
    """
    z = np.full_like(raw, np.nan)
    scale = np.empty(raw.shape[1:])
    for column in range(raw.shape[2]):
        fitted, sd, center, slope = detrend_and_zscore(
            raw[training, None, :, column], training.astype(float), detrend
        )
        scale[:, column] = sd[0]
        time = np.arange(len(raw)) - training.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            z[:, :, column] = (raw[:, :, column] - center - slope * time[:, None]) / sd
        z[training, :, column] = fitted[:, 0]
    return z, scale


def fitted_domain(raw, detrend):
    """Identify pixels usable in every validation fold with at least 30 complete years in
    the full fit.
    """
    years = np.arange(len(raw))
    valid = np.ones(raw.shape[1], dtype=bool)
    for fold in np.array_split(years, 5):
        train = np.setdiff1d(years, fold)
        z, _ = standardized(raw, train, detrend)
        complete = np.isfinite(z).all(axis=-1)
        valid &= (complete[train].sum(0) > 1) & complete[fold].any(0)
    z, _ = standardized(raw, years, detrend)
    return valid & (np.isfinite(z).all(axis=-1).sum(0) >= 30)


def cross_products(z):
    """Sum per-pixel design cross-products over years with complete observations,
    including an intercept column.
    """
    valid = np.isfinite(z).all(-1)
    design = np.concatenate([np.ones((*z.shape[:2], 1)), z], axis=-1)
    design = np.where(valid[..., None], design, 0)
    return np.einsum("tpi,tpj->pij", design, design)


def annual_fit(raw):
    """Return the eligibility mask and standardized and physical DeltaT slopes after
    year-block penalty selection.
    """
    predictors = [1, 3, 4, 5, 6, 7]  # DeltaT, Ta, TP, SSRD, CO2, VPD
    years = np.arange(len(raw))
    grid = np.r_[0, np.logspace(-5, 2, 29)]
    score = np.zeros((len(grid), raw.shape[1]))
    eligible = np.ones(raw.shape[1], dtype=bool)
    for fold in np.array_split(years, 5):
        train = np.setdiff1d(years, fold)
        z, _ = standardized(raw, train)
        train_m, test_m = cross_products(z[train]), cross_products(z[fold])
        good = (train_m[:, 0, 0] > 1) & (test_m[:, 0, 0] > 0)
        eligible &= good
        columns = [0] + predictors
        test = test_m[good]
        gram = test[:, columns, :][:, :, columns]
        for i, penalty in enumerate(grid):
            coef = solve_ridge(train_m[good], predictors, 2, penalty)
            sse = (
                test[:, 2, 2]
                - 2 * (coef * test[:, columns, 2]).sum(-1)
                + np.einsum("pi,pij,pj->p", coef, gram, coef)
            )
            score[i, good] += np.where(np.isfinite(sse), sse, np.inf)
    z, sd = standardized(raw, years)
    full = cross_products(z)
    eligible &= (full[:, 0, 0] >= 30) & np.isfinite(score).any(0)
    penalty = grid[score.argmin(0)]
    slopes = solve_ridge(full[eligible], predictors, 2, penalty[eligible])[:, 1]
    physical = slopes * sd[eligible, 1] / sd[eligible, 0]
    return eligible, slopes, physical


def calculate(model, scenario):
    """Write model-scenario pixel sensitivities and return the annual area-weighted
    series.
    """
    history = sorted(inputs.glob(model + "_historical_*.npz"))
    if not history:
        raise FileNotFoundError(model)
    sums = np.zeros((100, 2))
    total_weight = historical_gpp = pixels = 0
    maps = []
    for path in history:
        datasets = []
        for period in ["historical", "ssp245", "ssp585"]:
            with np.load(path.with_name(path.name.replace("_historical_", f"_{period}_"))) as item:
                flat = item["flat"]
                datasets.append(item["raw"])
        common = np.logical_and.reduce([np.isfinite(array).all(axis=(0, 2)) for array in datasets])
        if not common.any():
            continue
        # The accepted maps and denominator require eligible fits in both scenarios
        # under retained-trend and detrended preprocessing.
        shared = np.ones(common.sum(), dtype=bool)
        for future in datasets[1:]:
            candidate = np.concatenate([datasets[0], future])[:, common, :]
            for detrend in [False, True]:
                shared &= fitted_domain(candidate, detrend)
        common[common] = shared
        if not common.any():
            continue
        raw = np.concatenate([datasets[0], datasets[1 if scenario == "ssp245" else 2]])[
            :, common, :
        ]
        eligible, beta, physical = annual_fit(raw)
        raw = raw[:, eligible]
        flat = flat[common][eligible]
        baseline = raw[:14].mean(0)
        anomaly = raw[:, :, 0] - baseline[:, 0]
        weight = np.cos(np.deg2rad(89.5 - flat // 360))
        sums[:, 0] += anomaly @ weight
        sums[:, 1] += (anomaly * physical) @ weight
        historical_gpp += np.sum(weight * baseline[:, 1])
        total_weight += weight.sum()
        pixels += len(flat)
        maps.append(
            pd.DataFrame(
                dict(
                    flat=flat,
                    sensitivity=beta,
                    physical_slope=physical,
                    delta_t_change=anomaly[-20:].mean(0),
                )
            )
        )
    pd.concat(maps).to_csv(output / f"{model}_{scenario}_pixels.csv", index=False)
    return pd.DataFrame(
        dict(
            model=model,
            scenario=scenario,
            year=np.arange(2001, 2101),
            pixels=pixels,
            DeltaT=sums[:, 0] / total_weight,
            annual_change=sums[:, 1] / total_weight,
            baseline_GPP=historical_gpp / total_weight,
            percent=100 * sums[:, 1] / historical_gpp,
        )
    )


def regional_tables(annual):
    """Use each model's matched pixel support and the same regional GPP denominator."""
    import rasterio

    config = json.loads(
        (folder.parent / "supplementary_materials/config/datasets.json").read_text(encoding="utf8")
    )
    with rasterio.open(config["datasets"]["main_figure1_ai"]["files"][0]) as image:
        ai = image.read(1, masked=True).astype(float).filled(np.nan).ravel()
    rows = []
    for model, scenario in (
        annual[["model", "scenario"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        slopes = pd.read_csv(
            output / f"{model}_{scenario}_pixels.csv", float_precision="round_trip"
        ).set_index("flat")
        sums = {key: np.zeros((100, 4)) for key in ["Water_limited", "Energy_limited"]}
        for path in sorted(inputs.glob(model + "_historical_*.npz")):
            with np.load(path) as record:
                flat = record["flat"]
                selected_pixels = np.isin(flat, slopes.index)
                baseline = record["raw"][:, selected_pixels].mean(0)
                historical_dt = record["raw"][:, selected_pixels, 0]
            flat = flat[selected_pixels]
            with np.load(
                path.with_name(path.name.replace("_historical_", f"_{scenario}_"))
            ) as future:
                anomaly = (
                    np.concatenate([historical_dt, future["raw"][:, selected_pixels, 0]])
                    - baseline[:, 0]
                )
            beta = slopes.loc[flat, "physical_slope"].to_numpy()
            for group, selected_cells in [
                ("Water_limited", ai[flat] < 1),
                ("Energy_limited", ai[flat] >= 1),
            ]:
                weight = np.cos(np.deg2rad(89.5 - flat[selected_cells] // 360))
                change = anomaly[:, selected_cells]
                sums[group][:, 0] += change @ weight
                sums[group][:, 1] += (change * beta[selected_cells]) @ weight
                sums[group][:, 2] += weight @ baseline[selected_cells, 1]
                sums[group][:, 3] += weight.sum()
        for group, values in sums.items():
            rows.append(
                pd.DataFrame(
                    dict(
                        model=model,
                        scenario=scenario,
                        group=group,
                        year=np.arange(2001, 2101),
                        DeltaT=values[:, 0] / values[:, 3],
                        percent=100 * values[:, 1] / values[:, 2],
                        baseline_GPP=values[:, 2] / values[:, 3],
                    )
                )
            )
    pd.concat(rows).to_csv(output / "regional.csv", index=False)


def recent_trends(annual):
    """Fit the three S14 windows to the original fixed-baseline annual anomalies."""
    era_file = selected.parent / "ERA5_reference_original.csv"
    era5 = pd.read_csv(era_file).groupby("year").DeltaT.mean()
    rows = []
    for scenario, members in annual.groupby("scenario"):
        cmip = members.groupby("year").DeltaT.mean()
        for start, end in [(2001, 2014), (2015, 2023), (2001, 2023)]:
            for source, data in [("ERA5", era5), ("CMIP6", cmip)]:
                period = data.loc[start:end]
                slope = np.polyfit(period.index, period.values, 1)[0] * 10
                rows.append(
                    dict(
                        scenario=scenario,
                        start=start,
                        end=end,
                        source=source,
                        trend_C_per_decade=slope,
                    )
                )
    pd.DataFrame(rows).to_csv(output / "recent_trends.csv", index=False)


if __name__ == "__main__":
    import json

    folder = Path(__file__).resolve().parent
    inputs = Path(
        "D:/DATA/DeltaT_GPP_Analysis/Results/Temporary/cmip6_overall_review_20261030/figure5_recalculation/annual_inputs"
    )
    selected = (
        folder.parent
        / "Figure5_release_20261102_12models/Code/frozen_inputs/analysis/selection.json"
    )
    output = folder / "results/07"
    output.mkdir(exist_ok=True, parents=True)
    tables = []
    for model in json.loads(selected.read_text())["models"]:
        for scenario in ["ssp245", "ssp585"]:
            tables.append(calculate(model, scenario))
    annual = pd.concat(tables, ignore_index=True)
    annual.to_csv(output / "annual.csv", index=False)
    late = annual[annual.year >= 2081].groupby(["scenario", "model"]).percent.mean()
    late.to_csv(output / "model_endpoints.csv")
    late.groupby("scenario").mean().to_csv(output / "ensemble_endpoints.csv")
    regional_tables(annual)
    recent_trends(annual)
