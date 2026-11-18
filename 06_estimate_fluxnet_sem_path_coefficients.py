"""Fit standardized annual FLUXNET ridge path coefficients (Fig. 2e–f).

TSO in the input denotes DeltaT. H and LE predict Gc; GPP depends on
DeltaT, H, LE and Gc. All four equations include five controls. Fit each
equation to equal-site-weight anomalies, then estimate path products and the
H–LE residual correlation. Crossed site/year bootstrap intervals condition
on the selected penalties. Inputs are the accepted annual anomalies, or run
--gc-folder results/05 to rebuild them from the daily physical variables.

Inputs
------
The default input is locked_input.csv from the Figure 2 release. To rebuild site-
year inputs, run script 05 for all sites and retain the project daily station
records.

Run from the directory containing this script
--------------------------------------------
    python 06_estimate_fluxnet_sem_path_coefficients.py
    python 06_estimate_fluxnet_sem_path_coefficients.py --gc-folder results/05

Outputs in results/06
---------------------
annual_inputs.csv, penalties.csv and coefficients.csv with point estimates and
bootstrap intervals.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def standardize(values, weights):
    """Center and scale each variable across site-years using the supplied observation
    weights.
    """
    mean = np.average(values, weights=weights, axis=0)
    residual = values - mean
    variance = np.average(residual**2, weights=weights, axis=0)
    return residual / np.sqrt(variance)


def ridge(x, y, weights, penalty):
    """Estimate weighted ridge slopes and the unpenalized intercept for one response."""
    mass = weights.sum()
    x_mean = np.average(x, axis=0, weights=weights)
    y_mean = np.average(y, weights=weights)
    centered = x - x_mean
    weighted = centered.T * weights
    slopes = np.linalg.solve(
        weighted @ centered + penalty * mass * np.eye(x.shape[1]), weighted @ (y - y_mean)
    )
    return slopes, y_mean - x_mean @ slopes


def select_penalty(x, y, weights, year):
    """Choose the penalty with the smallest weighted prediction error over five held-out
    year blocks.
    """
    penalties = np.r_[0, np.logspace(-5, 2, 29)]
    scores = np.zeros(len(penalties))
    for test_years in np.array_split(np.unique(year), 5):
        test = np.isin(year, test_years)
        for i, penalty in enumerate(penalties):
            slopes, intercept = ridge(x[~test], y[~test], weights[~test], penalty)
            error = y[test] - intercept - x[test] @ slopes
            scores[i] += np.sum(weights[test] * error**2)
    return penalties[np.argmin(scores)]


def estimate(data, weights, penalties):
    """Return standardized paths, path products, equation R-squared values and the H-LE
    residual correlation.
    """
    z = pd.DataFrame(standardize(data.to_numpy(), weights), columns=data.columns)
    results = {}
    residuals = {}
    for target, parents in equations.items():
        x = z[parents + controls].to_numpy()
        y = z[target].to_numpy()
        slopes, intercept = ridge(x, y, weights, penalties[target])
        for predictor, slope in zip(parents + controls, slopes):
            results[f"{predictor}_to_{target}"] = slope
        error = y - intercept - x @ slopes
        residuals[target] = error
        results["R2_" + target] = 1 - np.average(error**2, weights=weights)
    covariance = np.average(residuals["H"] * residuals["LE"], weights=weights)
    results["H_LE_residual_correlation"] = covariance / np.sqrt(
        (1 - results["R2_H"]) * (1 - results["R2_LE"])
    )
    paths = [
        ("TSO", "H", "Gc", "GPP"),
        ("TSO", "H", "GPP"),
        ("TSO", "LE", "Gc", "GPP"),
        ("TSO", "LE", "GPP"),
        ("TSO", "GPP"),
    ]
    for path in paths:
        results["C_" + "_".join(path)] = np.prod(
            [results[f"{a}_to_{b}"] for a, b in zip(path, path[1:])]
        )
    results["specified_path_sum"] = sum(results["C_" + "_".join(path)] for path in paths)
    return results


def annual_anomalies(gc_folder):
    """Match the physical variables by day before station-year averaging."""
    variables = ["TSO", "GPP", "H", "LE", "Gc"] + controls
    stage = root / "Figure2_preparation_20261028"
    original_years = pd.read_csv(
        stage / "20261028_all_daylight_annual_ridge_S27/annual_raw_common_day_means.csv",
        usecols=["site", "year"],
    )
    sites = pd.read_csv(
        root
        / "Figure_releases_20261029/Figure_01/20261029_154sites/station_analysis/station_GPP_Gc_AI.csv"
    ).site
    records = []
    for site in sorted(sites):
        days = pd.read_csv(
            stage / f"20261028_all_daylight_processed_S27/stations/{site}.csv",
            float_precision="round_trip",
        )
        gc = pd.read_csv(gc_folder / f"{site}.csv", float_precision="round_trip").set_index("date")
        days["Gc"] = days.date.map(gc.gc_m_s)
        days = days.rename(columns={"DeltaT": "TSO"}).merge(
            original_years, on=["site", "year"], validate="many_to_one"
        )
        retained = days.loc[np.isfinite(days[variables]).all(axis=1)]
        groups = retained.groupby("year")
        annual = groups[variables].mean()
        annual["n_days"] = groups.size()
        annual = annual[annual.n_days >= 2].copy()
        if len(annual) < 2:
            continue
        dates = pd.to_datetime(annual.index.astype(str) + "-07-01")
        elapsed = np.asarray((dates - dates.min()).days) / 365.2425
        time = (elapsed - elapsed.mean()) / elapsed.std(ddof=1)
        design = np.column_stack([np.ones(len(time)), time])
        values = annual[variables].to_numpy()
        residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        scale = residual.std(axis=0, ddof=1)
        if (scale < 1e-12).any():
            continue
        annual[variables] = (residual - residual.mean(0)) / scale
        records.append(annual.assign(site=site).reset_index())
    return pd.concat(records, ignore_index=True)


def main():
    """Fit four annual equations and write the input table, penalties, coefficients and
    crossed site/year bootstrap intervals.
    """
    frame = (
        annual_anomalies(gc_folder)
        if gc_folder
        else pd.read_csv(input_file, float_precision="round_trip")
    )
    data = frame[["TSO", "GPP", "H", "LE", "Gc"] + controls]
    year, year_id = np.unique(frame.year, return_inverse=True)
    site, site_id, counts = np.unique(frame.site, return_inverse=True, return_counts=True)
    weights = 1 / counts[site_id]
    z = pd.DataFrame(standardize(data.to_numpy(), weights), columns=data.columns)
    penalties = {}
    for target, parents in equations.items():
        penalties[target] = select_penalty(
            z[parents + controls].values, z[target].values, weights, frame.year
        )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "annual_inputs.csv", index=False)
    point = estimate(data, weights, penalties)
    draws = []
    rng = np.random.default_rng(20261018)
    for _ in range(1999):
        years = rng.multinomial(len(year), np.full(len(year), 1 / len(year)))
        sites = rng.multinomial(len(site), np.full(len(site), 1 / len(site)))
        draws.append(estimate(data, weights * years[year_id] * sites[site_id], penalties))
    limits = pd.DataFrame(draws).quantile([0.025, 0.975]).T
    limits.columns = ["ci_low", "ci_high"]
    limits["estimate"] = pd.Series(point)
    output.mkdir(parents=True, exist_ok=True)
    limits.rename_axis("metric").to_csv(output / "coefficients.csv")
    pd.Series(penalties, name="lambda").to_csv(output / "penalties.csv")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    input_file = (
        root
        / "Figure_releases_20261029/Figure_02/20261029_154sites/inputs/SEM/controlled_ridge/locked_input.csv"
    )
    output = Path(__file__).parent / "results/06"
    controls = ["Ta", "TP", "SSRD", "CO2", "VPD"]
    equations = {"H": ["TSO"], "LE": ["TSO"], "Gc": ["H", "LE"], "GPP": ["TSO", "H", "LE", "Gc"]}
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gc-folder", type=Path, help="Rebuild annual inputs from the daily results of script 05"
    )
    gc_folder = parser.parse_args().gc_folder
    main()
