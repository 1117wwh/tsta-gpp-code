"""Estimate ΔT sensitivity from standardized gridded products and FLUXNET site records."""

# Gridded NOAA background CO₂ is already aligned to each 8-day period or calendar month:
# its 41 sine-latitude nodes have been interpolated to 180 rows and repeated across 360 longitudes.
# FLUXNET regressions use each site's own CO₂ series (CO2_F_MDS).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PATHS_FILE = Path(__file__).resolve().parent / "02_data_paths.txt"
PREDICTORS = ["DeltaT", "Ta", "TP", "SSRD", "CO2", "VPD"]
PENALTIES = np.r_[0.0, np.logspace(-5, 2, 29)]
N_BOOTSTRAP = 2000


def year_cross_products(values):
    """Sum cross-products over complete dates. Input axes are date, cell, variable."""
    complete = np.isfinite(values).all(axis=-1)
    # Columns are the intercept, the six predictors, and the response.
    design = np.concatenate([np.ones_like(values[..., :1]), values], axis=-1)
    design[~complete] = 0
    return np.einsum("tpi,tpj->pij", design, design)


def fit_ridge(products, penalty):
    """Fit an intercept and six slopes by minimizing mean squared error plus the ridge penalty."""
    xtx = products[..., :-1, :-1]
    xty = products[..., :-1, -1]
    count = products[..., 0, 0]
    ridge_weights = np.diag([0.0] + [1.0] * len(PREDICTORS))  # The intercept is unpenalized.
    system = xtx + (count * penalty)[..., None, None] * ridge_weights

    valid = (count > 0) & np.isfinite(penalty)
    unpenalized = valid & (penalty == 0)
    if unpenalized.any():
        valid[unpenalized] = np.linalg.matrix_rank(xtx[unpenalized]) == xtx.shape[-1]
    coefficients = np.full(xty.shape, np.nan)
    coefficients[valid] = np.linalg.solve(system[valid], xty[valid][..., None])[..., 0]
    return coefficients


def choose_penalty(annual):
    """Select a separate ridge penalty for each cell or site using five blocks of years."""
    total = annual.sum(axis=0)
    scores = np.zeros((len(PENALTIES), len(total)))
    for held_out in np.array_split(np.arange(len(annual)), 5):
        test = annual[held_out].sum(axis=0)
        train = total - test
        for i, penalty in enumerate(PENALTIES):
            coefficients = fit_ridge(train, penalty)
            # The test residual is y minus the intercept and fitted predictor terms.
            weights = np.column_stack([-coefficients, np.ones(len(test))])
            squared_error = np.einsum("pi,pij,pj->p", weights, test, weights)
            scores[i] += np.where(np.isfinite(squared_error), squared_error, np.inf)
    best = PENALTIES[scores.argmin(axis=0)]
    best[~np.isfinite(scores).any(axis=0)] = np.nan
    return best


def bootstrap_p_value(annual, penalty, sensitivity):
    """Resample entire years, hold the penalty fixed, and calculate a two-sided P value."""
    n_years, n_cells = annual.shape[:2]
    rng = np.random.default_rng(20261021)
    draws = rng.multinomial(n_years, np.full(n_years, 1 / n_years), size=N_BOOTSTRAP)
    p_value = np.full(n_cells, np.nan)
    for start in range(0, n_cells, 12):  # Process a few cells at a time to limit memory use.
        cells = slice(start, start + 12)
        block = annual[:, cells]
        resampled = np.einsum("by,ypij->bpij", draws, block, optimize=True)
        slopes = fit_ridge(resampled, penalty[cells])[..., 1]
        observed = sensitivity[cells]
        extreme = np.abs(slopes - observed) >= np.abs(observed)
        p = (1 + extreme.sum(axis=0)) / (N_BOOTSTRAP + 1)
        p[~np.isfinite(observed) | ~np.isfinite(slopes).all(axis=0)] = np.nan
        p_value[cells] = p
    return p_value


def read_grid_years(raster_dir, product, scale, pixels):
    """Read prepared anomalies and collect one cross-product matrix per year and cell."""
    fields = PREDICTORS + [product]
    periods = range(1, 362, 8) if scale == "8day" else range(1, 13)
    annual = []
    for year in range(2001, 2024):
        values = np.empty((len(periods), len(pixels), len(fields)))
        for t, period in enumerate(periods):
            interval = f"DOY{period:03d}" if scale == "8day" else f"M{period:02d}"
            for j, variable in enumerate(fields):
                path = raster_dir / scale / variable / f"{variable}_{year}_{interval}_zscore.tif"
                with rasterio.open(path) as raster:
                    if (
                        raster.shape != (180, 360)
                        or raster.crs is None
                        or raster.crs.to_epsg() != 4326
                        or tuple(raster.transform)[:6] != (1, 0, -180, 0, -1, 90)
                    ):
                        raise ValueError(f"Expected the study's 1° grid in {path}")
                    values[t, :, j] = raster.read(1, masked=True).filled(np.nan).ravel()[pixels]
        annual.append(year_cross_products(values))
    return np.stack(annual)


def grid_sensitivities(annual, pixels):
    counts = annual[:, :, 0, 0]
    total_count = counts.sum(axis=0)
    eligible = total_count >= 30
    for held_out in np.array_split(np.arange(len(annual)), 5):
        eligible &= total_count - counts[held_out].sum(axis=0) > 1

    annual = annual[:, eligible]
    penalty = choose_penalty(annual)
    coefficients = fit_ridge(annual.sum(axis=0), penalty)
    sensitivity = coefficients[:, 1]  # ΔT is the first predictor.
    return pd.DataFrame({
        "flat": pixels[eligible],
        "S_std": sensitivity,
        "n": total_count[eligible].astype(int),
        "lambda_ridge": penalty,
        "p_S_std": bootstrap_p_value(annual, penalty, sensitivity),
        "intercept": coefficients[:, 0],
    })


def station_sensitivities(sites, response_dirs):
    """Fit daily GPP and Gc separately using the valid dates for each response."""
    rows = []
    for site in sites:
        for response, directory in response_dirs.items():
            fields = [name + "_z" for name in PREDICTORS + [response]]
            days = pd.read_csv(
                directory / f"{site}.csv", usecols=["year"] + fields, float_precision="round_trip"
            )
            days = days.loc[np.isfinite(days[fields]).all(axis=1)]
            if len(days) < 30 or days.year.nunique() < 2:
                continue
            yearly = []
            for _, group in days.groupby("year"):
                yearly.append(year_cross_products(group[fields].to_numpy()[:, None, :]))
            annual = np.stack(yearly)
            penalty = choose_penalty(annual)
            coefficients = fit_ridge(annual.sum(axis=0), penalty)[0]
            rows.append({
                "site": site,
                "response": response,
                "n": len(days),
                "years": len(annual),
                "coefficient": "S_std" if response == "GPP" else "Gc_sensitivity_to_DeltaT_std",
                "estimate": coefficients[1],
                "lambda_ridge": penalty[0],
                "intercept": coefficients[0],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    VEG_MASK_PATH = PATHS_FILE.parent / paths["VEG_MASK_PATH"]
    STANDARDIZED_RASTER_DIR = PATHS_FILE.parent / paths["STANDARDIZED_RASTER_DIR"]
    STATION_COHORT_PATH = PATHS_FILE.parent / paths["STATION_COHORT_PATH"]
    STATION_GPP_DIR = PATHS_FILE.parent / paths["STATION_GPP_DIR"]
    STATION_GC_DIR = PATHS_FILE.parent / paths["STATION_GC_DIR"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pixels = np.load(VEG_MASK_PATH)  # Flat indices of vegetated 1° grid cells.

    products = {"8day": ["GOSIF_GPP", "GOSIF_SIF"], "monthly": ["GLOFLUX_GPP", "LHSIF_SIF"]}
    for scale, names in products.items():
        for product in names:
            annual = read_grid_years(STANDARDIZED_RASTER_DIR, product, scale, pixels)
            grid_sensitivities(annual, pixels).to_csv(OUTPUT_DIR / f"{product}.csv", index=False)

    sites = pd.read_csv(STATION_COHORT_PATH, usecols=["site"]).site
    response_dirs = {"GPP": STATION_GPP_DIR, "Gc": STATION_GC_DIR}
    station_sensitivities(sites, response_dirs).to_csv(OUTPUT_DIR / "stations.csv", index=False)
