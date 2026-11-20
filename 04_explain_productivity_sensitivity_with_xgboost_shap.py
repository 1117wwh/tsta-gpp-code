"""Fit three XGBoost models of mean monthly productivity sensitivity and explain their test predictions with TreeSHAP."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, DMatrix
from statsmodels.nonparametric.smoothers_lowess import lowess


PATHS_FILE = Path(__file__).resolve().parent / "04_data_paths.txt"


def fit_models(input_table, output):
    # The response is the mean sensitivity across the four monthly productivity products.
    table = pd.read_csv(input_table).rename(columns={"observed": "S_std_monthly_product_mean"})
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
    target = table["S_std_monthly_product_mean"]
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions = [], {}
    # M1 uses climate, M2 adds vegetation and water storage, M3 adds soils and elevation.
    for name, count in [("M1", 4), ("M2", 8), ("M3", 12)]:
        predictor_table = table[features[:count]]
        model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
        )
        model.fit(predictor_table.iloc[train], target.iloc[train])
        predicted = model.predict(predictor_table)
        predictions[name] = predicted
        metrics.append(
            dict(
                model=name,
                r2=r2_score(target.iloc[test], predicted[test]),
                rmse=mean_squared_error(target.iloc[test], predicted[test]) ** 0.5,
                mae=mean_absolute_error(target.iloc[test], predicted[test]),
            )
        )
    prediction_table = table[["sample_index", "flat", "S_std_monthly_product_mean"]].copy()
    prediction_table.insert(2, "split", "train")
    prediction_table.loc[test, "split"] = "test"
    for name, values in predictions.items():
        prediction_table[f"predicted_S_std_{name}"] = values
    prediction_table.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    explain_test_predictions(
        model, predictor_table.iloc[test], table.sample_index.iloc[test], output
    )
    bootstrap_test_scores(test, target, predictions, output)


def explain_test_predictions(model, test_features, sample_ids, output):
    """Compute SHAP values and smooth each feature dependence across held-out cells."""
    contributions = model.get_booster().predict(DMatrix(test_features), pred_contribs=True)
    shap = pd.DataFrame(contributions[:, :-1], columns=test_features.columns, index=sample_ids)
    np.save(output / "shap_values.npy", contributions)
    shap.to_csv(output / "shap.csv")
    importance = shap.abs().mean().sort_values(ascending=False)
    importance.rename_axis("feature").rename("mean_abs_SHAP").to_csv(output / "importance.csv")
    curves = []
    for feature in test_features:
        values = test_features[feature].to_numpy()
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
        curves.append(
            pd.DataFrame(dict(feature=feature, display_value=axis_values, SHAP=curve[:, 1]))
        )
    pd.concat(curves).to_csv(output / "lowess.csv", index=False)


def bootstrap_test_scores(test, target, predictions, output):
    """Resample the same test cells for all three models in each of 2,000 draws."""
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
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    TRAINING_TABLE_PATH = PATHS_FILE.parent / paths["TRAINING_TABLE_PATH"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    fit_models(TRAINING_TABLE_PATH, OUTPUT_DIR)
