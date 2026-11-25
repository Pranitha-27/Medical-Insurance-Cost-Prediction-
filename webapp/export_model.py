"""
Train the deployed model bundle for the web app.

Reproduces the winning model of the case study -- degree-2 polynomial features,
StandardScaler, Ridge(alpha=1.0) -- on exactly the same 64/16/20 split
(random_state = 42) and saves everything the Flask app needs into
webapp/model/model_bundle.joblib:

    model      : fitted sklearn Pipeline
    columns    : feature order expected by the pipeline
    row_ref    : median training row, used to draw the SHAP-style contribution bars
    metadata   : metrics, ranges, dataset means, top drivers

Run from the repository root:

    python webapp/export_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "webapp", "model")
os.makedirs(HERE, exist_ok=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Same encoding + feature engineering as src/medical_insurance_ml.py."""
    d = df.copy()
    d["sex"] = (d["sex"] == "male").astype(int)
    d["smoker"] = (d["smoker"] == "yes").astype(int)
    d = pd.get_dummies(d, columns=["region"], drop_first=True)
    for col in ("region_northwest", "region_southeast", "region_southwest"):
        if col not in d:
            d[col] = 0
    d["bmi_smoker_interaction"] = d["bmi"] * d["smoker"]
    d["age_smoker_interaction"] = d["age"] * d["smoker"]
    d["is_obese"] = (d["bmi"] >= 30).astype(int)
    return d


def main() -> None:
    raw = pd.read_csv(os.path.join(ROOT, "data", "insurance.csv")).drop_duplicates()
    raw = raw.reset_index(drop=True)
    data = engineer(raw)

    columns = ["age", "sex", "bmi", "children", "smoker",
               "region_northwest", "region_southeast", "region_southwest",
               "bmi_smoker_interaction", "age_smoker_interaction", "is_obese"]
    X, y = data[columns], data["charges"]

    # identical partition to the report: 64 % train / 16 % val / 20 % test.
    # The polynomial model in the report is fitted on the 855 training rows only,
    # so the deployed model reproduces R2 = 0.9031 exactly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]).fit(X_fit, y_fit)

    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    pred = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, pred)), 2),
        "n_test": int(len(y_test)),
        "n_train": int(len(X_fit)),
    }

    # median training profile -> reference point for the contribution bars
    row_ref = X_fit.median().to_dict()

    results = json.load(open(os.path.join(ROOT, "src", "results.json")))
    bundle = {
        "model": model,
        "columns": columns,
        "row_ref": row_ref,
        "metadata": {
            "metrics": metrics,
            "dataset": {
                "n_rows": int(len(raw)),
                "charges_mean": round(float(raw["charges"].mean()), 2),
                "charges_median": round(float(raw["charges"].median()), 2),
                "charges_min": round(float(raw["charges"].min()), 2),
                "charges_max": round(float(raw["charges"].max()), 2),
                "non_smoker_mean": results["mean_charges_by_smoker"].get("no"),
                "smoker_mean": results["mean_charges_by_smoker"].get("yes"),
            },
            "ranges": {
                "age": [18, 64], "bmi": [15.96, 53.13], "children": [0, 5],
            },
            "importance": dict(list(results["rf_importance"].items())[:5]),
            "naive_baseline_mae": round(float(np.abs(y_test - y_test.mean()).mean()), 2),
        },
    }

    out = os.path.join(HERE, "model_bundle.joblib")
    joblib.dump(bundle, out)
    print("wrote", out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
