"""
Flask web app for the Medical Insurance Cost Prediction case study.

Serves the trained degree-2 polynomial (Ridge) model so that an applicant's
details can be entered in a browser and the predicted annual charge returned,
together with a contribution breakdown and the study's metrics.

Run from the repository root:

    python webapp/app.py            # http://localhost:5000

The app never re-trains: it loads webapp/model/model_bundle.joblib, which is
produced by webapp/export_model.py.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, "webapp", "model", "model_bundle.joblib")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Load the model once at import time
# --------------------------------------------------------------------------- #
BUNDLE = joblib.load(BUNDLE_PATH)
MODEL: Pipeline = BUNDLE["model"]
COLUMNS = BUNDLE["columns"]
ROW_REF = BUNDLE["row_ref"]
META = BUNDLE["metadata"]

# The study's four-model comparison (test set, n = 268) -- read from results.json
RESULTS = json.load(open(os.path.join(ROOT, "src", "results.json")))
MODEL_TABLE = RESULTS["metrics_table"]
BEST_MODEL = RESULTS["best_model"]

LIMITS = META["ranges"]
REGIONS = ["northeast", "northwest", "southeast", "southwest"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_feature_row(age: float, sex: str, bmi: float, children: int,
                      smoker: str, region: str) -> pd.DataFrame:
    """Raw inputs -> the 11 engineered features, in the model's column order."""
    row = {
        "age": float(age),
        "sex": 1 if sex == "male" else 0,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": 1 if smoker == "yes" else 0,
        # northeast is the reference level (drop_first=True), so it is all zeros
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    row["bmi_smoker_interaction"] = row["bmi"] * row["smoker"]
    row["age_smoker_interaction"] = row["age"] * row["smoker"]
    row["is_obese"] = 1 if row["bmi"] >= 30 else 0
    return pd.DataFrame([row])[COLUMNS]


def contributions(features: pd.DataFrame) -> list:
    """
    Exact decomposition of the linear prediction into per-feature terms.

    The winning model is linear in its 77 polynomial terms, so
        prediction = intercept + sum(coef_j * scaled_term_j)
    and each of the 11 original features can be grouped with its own polynomial
    terms to give a contribution in USD. Contributions therefore add up to
    (prediction - intercept) exactly -- no approximation, no sampling.
    """
    poly: PolynomialFeatures = MODEL.named_steps["poly"]
    scaler = MODEL.named_steps["scaler"]
    ridge = MODEL.named_steps["ridge"]

    z = scaler.transform(poly.transform(features))[0]
    terms = ridge.coef_ * z

    names = poly.get_feature_names_out(COLUMNS)
    grouped = {c: 0.0 for c in COLUMNS}
    for name, value in zip(names, terms):
        base = name.split(" ")[0].split("^")[0]
        grouped[base] = grouped.get(base, 0.0) + float(value)

    labels = {
        "age": "Age",
        "sex": "Sex (male)",
        "bmi": "BMI",
        "children": "Children",
        "smoker": "Smoker",
        "region_northwest": "Region: northwest",
        "region_southeast": "Region: southeast",
        "region_southwest": "Region: southwest",
        "bmi_smoker_interaction": "BMI × smoker",
        "age_smoker_interaction": "Age × smoker",
        "is_obese": "Obese (BMI ≥ 30)",
    }
    out = [{"feature": labels.get(k, k), "value": round(v, 2)}
           for k, v in grouped.items()]
    out.sort(key=lambda d: -abs(d["value"]))
    return out


def validate(payload: dict) -> dict:
    """Return a cleaned payload or raise ValueError with a readable message."""
    clean = {}
    try:
        clean["age"] = int(payload.get("age", ""))
        clean["children"] = int(payload.get("children", 0) or 0)
        clean["bmi"] = float(payload.get("bmi", ""))
    except (TypeError, ValueError):
        raise ValueError("Age, BMI and children must be numbers.")

    clean["sex"] = str(payload.get("sex", "female")).lower()
    clean["smoker"] = str(payload.get("smoker", "no")).lower()
    clean["region"] = str(payload.get("region", "northeast")).lower()
    clean["age_smoker_income"] = 0  # placeholder to keep the dict explicit

    if clean["sex"] not in ("female", "male"):
        raise ValueError("Sex must be female or male.")
    if clean["smoker"] not in ("no", "yes"):
        raise ValueError("Smoking status must be no or yes.")
    if clean["region"] not in REGIONS:
        raise ValueError("Region must be one of " + ", ".join(REGIONS) + ".")
    if not LIMITS["age"][0] <= clean["age"] <= LIMITS["age"][1]:
        raise ValueError(f"Age must be between {LIMITS['age'][0]} and {LIMITS['age'][1]} "
                         "(the range present in the training data).")
    if not LIMITS["bmi"][0] <= clean["bmi"] <= LIMITS["bmi"][1]:
        raise ValueError(f"BMI must be between {LIMITS['bmi'][0]} and {LIMITS['bmi'][1]} "
                         "(the range present in the training data).")
    if not LIMITS["children"][0] <= clean["children"] <= LIMITS["children"][1]:
        raise ValueError("Children must be between 0 and 5.")
    clean.pop("age_smoker_income")
    return clean


def predict(clean: dict) -> dict:
    features = build_feature_row(**clean)
    value = float(MODEL.predict(features)[0])

    # an honest uncertainty band: one test-set RMSE, clipped at zero
    sd = META["metrics"]["rmse"]
    contrib = contributions(features)
    biggest = next((c for c in contrib if c["value"] > 0), None)

    if value < META["dataset"]["non_smoker_mean"]:
        band = "below the dataset average"
    elif value < META["dataset"]["smoker_mean"]:
        band = "between the non-smoker and smoker averages"
    else:
        band = "above the dataset average"

    return {
        "prediction": round(value, 2),
        "baseline": round(float(MODEL.named_steps["ridge"].intercept_), 2),
        "low": max(0.0, round(value - sd, 2)),
        "high": round(value + sd, 2),
        "contributions": contrib,
        "biggest_driver": biggest,
        "band": band,
        "inputs": clean,
        "metrics": META["metrics"],
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template(
        "index.html",
        meta=META,
        results=RESULTS,
        model_table=MODEL_TABLE,
        best_model=BEST_MODEL,
        limits=LIMITS,
        regions=REGIONS,
        presets=[
            {"label": "24-year-old non-smoker, BMI 22.5",
             "payload": {"age": 24, "sex": "female", "bmi": 22.5, "children": 0,
                         "smoker": "no", "region": "southwest"}},
            {"label": "45-year-old smoker, BMI 34.8",
             "payload": {"age": 45, "sex": "male", "bmi": 34.8, "children": 2,
                         "smoker": "yes", "region": "northeast"}},
            {"label": "60-year-old non-smoker, BMI 28.1",
             "payload": {"age": 60, "sex": "male", "bmi": 28.1, "children": 3,
                         "smoker": "no", "region": "southeast"}},
        ],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        clean = validate(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, **predict(clean)})


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "best_model": BEST_MODEL,
        "metrics": META["metrics"],
        "model_table": MODEL_TABLE,
        "dataset": META["dataset"],
        "ranges": META["ranges"],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": BEST_MODEL,
                    "r2": META["metrics"]["r2"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
