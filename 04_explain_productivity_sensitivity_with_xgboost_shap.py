"""Explain observed productivity sensitivity with XGBoost (S12 and S13).

Input is one row per grid cell, including the observed mean standardized
sensitivity of four products and the environmental predictors. Use XGBoost
1.6.2 to reproduce the release. Compare the three predictor sets using the
same training/test split. TreeSHAP and LOWESS use the held-out cells of the
12-predictor model; precipitation is one of the four climate predictors.

Inputs
------
The inputs directory must contain training_table.csv and training_manifest.json. Use
XGBoost 1.6.2 with pandas, NumPy, scikit-learn and statsmodels.

Run from the directory containing this script
--------------------------------------------
    python 04_explain_productivity_sensitivity_with_xgboost_shap.py

Outputs in results/04
---------------------
Test predictions and metrics, bootstrap scores, SHAP values and importance, and
LOWESS curves.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, DMatrix
from statsmodels.nonparametric.smoothers_lowess import lowess


def main():
    """Compare three predictor sets on a common split and write test, bootstrap and SHAP
    summaries.
    """
    table = pd.read_csv(inputs / "training_table.csv")
    settings = json.loads((inputs / "training_manifest.json").read_text())
    features = [
        "vpd_mean",
        "tp_mean",
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
    train, test = train_test_split(np.arange(len(table)), test_size=0.3, random_state=42)
    assert np.array_equal(np.sort(test), np.flatnonzero(table.split.eq("test")))
    target = table.observed
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions = [], {}
    for name, count in [("M1", 4), ("M2", 8), ("M3", 12)]:
        x = table[features[:count]]
        model = XGBRegressor(**settings["parameters"])
        model.fit(x.iloc[train], target.iloc[train])
        predicted = model.predict(x)
        predictions[name] = predicted
        metrics.append(
            dict(
                model=name,
                r2=r2_score(target.iloc[test], predicted[test]),
                rmse=mean_squared_error(target.iloc[test], predicted[test]) ** 0.5,
                mae=mean_absolute_error(target.iloc[test], predicted[test]),
            )
        )
    key = table[["sample_index", "flat", "split"]].copy()
    key.assign(**predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    contributions = model.get_booster().predict(DMatrix(x.iloc[test]), pred_contribs=True)
    np.testing.assert_allclose(
        contributions.sum(axis=1), predictions["M3"][test], rtol=1e-5, atol=2e-6
    )
    shap = pd.DataFrame(
        contributions[:, :-1], columns=features, index=table.sample_index.iloc[test]
    )
    np.save(output / "shap_values.npy", contributions)
    shap.to_csv(output / "shap.csv")
    shap.abs().mean().sort_values(ascending=False).to_csv(output / "importance.csv")
    curves = []
    for feature in features:
        values = x.iloc[test][feature].to_numpy()
        curve = lowess(shap[feature], values, frac=0.3, it=0)
        lo, hi = np.quantile(values, [0.01, 0.99])
        curve = curve[(curve[:, 0] >= lo) & (curve[:, 0] <= hi)]
        axis_values = curve[:, 0]
        if feature == "ta_mean":
            axis_values = axis_values - 273.15
        elif feature == "aridity_index_log10":
            axis_values = 10**axis_values
        elif feature == "rootzone_storage_log1p":
            axis_values = np.expm1(axis_values)
        curves.append(pd.DataFrame(dict(feature=feature, x=axis_values, shap=curve[:, 1])))
    pd.concat(curves).to_csv(output / "lowess.csv", index=False)
    test = np.sort(test)
    rng = np.random.default_rng(20260928)
    scores = []
    for _ in range(2000):
        sample = test[rng.integers(0, len(test), len(test))]
        scores.append(
            {
                name: r2_score(target.iloc[sample], values[sample])
                for name, values in predictions.items()
            }
        )
    pd.DataFrame(scores).to_csv(output / "bootstrap.csv", index=False)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    inputs = folder.parent / "Supplementary_ML_TP_trial_20261111/frozen_baseline"
    output = folder / "results/04"
    main()
