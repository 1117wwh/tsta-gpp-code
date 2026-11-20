"""Estimate the Figure 2 paths from FLUXNET site-year anomalies, with equal weight per site and crossed site–year bootstrap intervals."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


PATHS_FILE = Path(__file__).resolve().parent / "06_data_paths.txt"


controls = ["Ta", "TP", "SSRD", "CO2", "VPD"]
equations = {
    "H": ["DeltaT"],
    "LE": ["DeltaT", "Gc"],
    "Gc": ["H"],
    "GPP": ["DeltaT", "H", "LE", "Gc"],
}


def standardize(values, weights):
    """Center and scale across site-years with equal total weight per site."""
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
    """Return standardized paths, five path products and equation R-squared values."""
    z = pd.DataFrame(standardize(data.to_numpy(), weights), columns=data.columns)
    results = {}
    for target, parents in equations.items():
        x = z[parents + controls].to_numpy()
        y = z[target].to_numpy()
        slopes, intercept = ridge(x, y, weights, penalties[target])
        for predictor, slope in zip(parents + controls, slopes):
            results[f"{predictor}_to_{target}"] = slope
        error = y - intercept - x @ slopes
        results["R2_" + target] = 1 - np.average(error**2, weights=weights)
    # Figure 2f, bars A to E, in displayed order.
    paths = [
        ("DeltaT", "GPP"),
        ("DeltaT", "H", "GPP"),
        ("DeltaT", "LE", "GPP"),
        ("DeltaT", "H", "Gc", "GPP"),
        ("DeltaT", "H", "Gc", "LE", "GPP"),
    ]
    for path in paths:
        results["C_" + "_".join(path)] = np.prod(
            [results[f"{a}_to_{b}"] for a, b in zip(path, path[1:])]
        )
    results["specified_path_sum"] = sum(results["C_" + "_".join(path)] for path in paths)
    return results


def fit_paths(frame, output):
    """Fit four equations and resample sites and years independently with fixed penalties."""
    variables = ["DeltaT", "GPP", "H", "LE", "Gc"] + controls
    data = frame[variables]
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
    point = estimate(data, weights, penalties)
    draws = []
    rng = np.random.default_rng(20261018)
    # Crossed bootstrap: multiply each site-year weight by its site and year draw counts.
    for _ in range(2000):
        years = rng.multinomial(len(year), np.full(len(year), 1 / len(year)))
        sites = rng.multinomial(len(site), np.full(len(site), 1 / len(site)))
        draws.append(estimate(data, weights * years[year_id] * sites[site_id], penalties))
    limits = pd.DataFrame(draws).quantile([0.025, 0.975]).T
    limits.columns = ["ci_low", "ci_high"]
    limits["estimate"] = pd.Series(point)
    limits.rename_axis("metric").to_csv(output / "coefficients.csv")
    pd.Series(penalties, name="lambda").to_csv(output / "penalties.csv")


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    STATION_ANNUAL_ANOMALIES_PATH = PATHS_FILE.parent / paths["STATION_ANNUAL_ANOMALIES_PATH"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    annual = pd.read_csv(STATION_ANNUAL_ANOMALIES_PATH, float_precision="round_trip")
    fit_paths(annual, OUTPUT_DIR)
