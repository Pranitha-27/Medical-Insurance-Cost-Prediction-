# Steps to Execute the Project

**Medical Insurance Cost Prediction Using Machine Learning Regression Models** — Case Study 14
Everything below was run on **Python 3.13, scikit-learn 1.6, pandas 2.2, matplotlib 3.11** on a laptop CPU.
Total runtime of the full pipeline is about **100 seconds**; there is no GPU or internet dependency once
`data/insurance.csv` is present.

---

## 0. What you need

| Requirement | Value used in this project |
|---|---|
| Python | 3.10 or newer (tested on 3.13) |
| RAM | 2 GB is plenty (the dataset is 1,338 rows × 7 columns) |
| Packages | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` |
| Optional (only to rebuild the report) | `reportlab`, `python-docx`, `pypdf`, `pillow` |
| Internet | Not needed — the dataset is committed at `data/insurance.csv` |

---

## 1. Get the code

```bash
# option A: clone from GitHub
git clone https://github.com/Pranitha-27/Medical-Insurance-Cost-Prediction-.git
cd Medical-Insurance-Cost-Prediction-

# option B: unzip the submitted archive
unzip medical-insurance-cost-prediction.zip
cd medical-insurance-cost-prediction
```

---

## 2. Create an isolated environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the minimum versions the code was tested with:

```
numpy>=2.0
pandas>=2.2
scikit-learn>=1.6
matplotlib>=3.9
seaborn>=0.13
jupyter>=1.0
python-docx>=1.1
reportlab>=4.0
pypdf>=5.0
pillow>=10.0
```

---

## 3. Run the full pipeline (the main step)

```bash
python src/medical_insurance_ml.py
```

This single script performs the entire study and takes about **1.5 minutes**:

1. loads `data/insurance.csv`, reports shape, dtypes, missing values (`0`) and duplicates (`1` removed →
   1,337 rows);
2. computes all EDA statistics (means by smoking status / sex / region / BMI class, correlation with the
   target, IQR outlier counts);
3. encodes `sex`/`smoker` (binary) and `region` (one-hot, `drop_first=True`) and engineers
   `bmi_smoker_interaction`, `age_smoker_interaction`, `is_obese`;
4. splits 64 % / 16 % / 20 % (`random_state = 42`) → 855 / 214 / 268 rows;
5. fits **Multiple Linear Regression**, **Polynomial Regression (degree 2 + StandardScaler + Ridge α = 1.0)**,
   **Decision Tree** and **Random Forest**, tuning the two tree models with 5-fold `GridSearchCV` on
   neg-RMSE, scoring the validation set, refitting on train+validation and evaluating on the untouched
   test set;
6. reports R², MAE, MSE, RMSE and MAPE per model, plus five-fold CV, subgroup MAE by smoking status, five
   sample predictions, three new-applicant predictions and a learning curve;
7. writes the two artefacts the report and the figures read from:
   * `src/results.json` — every number quoted in the report,
   * `src/predictions.json` — test-set actual vs predicted values.

Expected terminal tail (abridged):

```
"metrics_table": {
  "Multiple Linear Regression":   {"R2": 0.8069, "RMSE": 5956.34, "MAPE_pct": 41.40, ...},
  "Polynomial Regression (deg 2)":{"R2": 0.9031, "RMSE": 4218.88, "MAPE_pct": 30.08, ...},
  "Decision Tree Regression":     {"R2": 0.8943, "RMSE": 4407.90, "MAPE_pct": 33.50, ...},
  "Random Forest Regression":     {"R2": 0.8962, "RMSE": 4366.72, "MAPE_pct": 35.29, ...}
},
"best_model": "Polynomial Regression (deg 2)",
"mae_by_smoker": {"no": 2364.47, "yes": 2315.48},
"new_customer_predictions": [3317.16, 43090.16, 15314.56]
```

**Check that it reproduces:** the best model must give **R² = 0.9031** and **RMSE = 4,218.88**. Because the
split and every estimator seed are fixed, the numbers are identical on any machine.

---

## 4. Regenerate the report figures

```bash
python src/make_figures_refstyle.py
```

Prints one line per figure and writes the ten PNGs used in the report into `figures/`:

```
wrote fig3_1_charges_distribution.png
wrote fig3_2_smoker_children.png
wrote fig3_3_scatter.png            <- this is Fig. 3.3 in the report
wrote fig5_1_workflow.png
wrote fig6_1_model_comparison.png
wrote fig6_2_actual_vs_predicted.png
wrote fig6_3_residuals.png
wrote fig6_4_feature_importance.png
wrote fig6_5_error_analysis.png
wrote figB1_learning_curve.png
```

Run step 3 first — this script reads `results.json` and `predictions.json`, so the plots always match the
numbers. `FIGURE_GUIDE.md` explains each figure and shows its exact code.

---

## 5. Run the notebook (alternative, cell-by-cell walkthrough)

```bash
jupyter notebook medical_insurance_cost_prediction.ipynb
```

Then **Kernel → Restart & Run All**. Same results as step 3 with the EDA and diagnostics visible inline.
In Google Colab: upload the notebook and `data/insurance.csv`, then *Runtime → Run all*.

---

## 6. Rebuild the report (optional — only if you change text or data)

```bash
python src/make_pdf.py       # -> docs/Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf
python src/make_docx.py      # -> docs/Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.docx
```

Important: **`make_pdf.py` reads the page map written by the previous run** (`src/pgmap.json`), so that the
Table of Contents, List of Figures and List of Tables show real page numbers. Run it **twice** after any
edit that changes pagination:

```bash
python src/make_pdf.py && python src/make_pdf.py
```

The DOCX lists are plain text (no Word fields), so `make_docx.py` only needs one run and picks up the
finished `pgmap.json`. If you edit the DOCX by hand and Word offers to update fields, choose **No** —
these lists are already correct and Word would replace them with `Error! Bookmark not defined`.

Report text lives in `src/report_content.py` (single source of truth), and the rubric pages are appended
by both `make_pdf.py` and `make_docx.py` after the final appendix.

---

## 7. Predict for a new applicant (standalone, no retraining needed for the pattern)

`src/medical_insurance_ml.py` already prints three unseen-applicant predictions at the end of step 3
(`new_customer_predictions`: 3,317.16 / 43,090.16 / 15,314.56 USD). To run that logic on your own applicant,
save this as `predict_applicant.py` in the repository root and run it:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 1. the applicant -- edit these six values
new = pd.DataFrame([{"age": 45, "sex": "male", "bmi": 34.8, "children": 2,
                     "smoker": "yes", "region": "northeast"}])

# 2. exact same encoding and feature engineering as the pipeline
df = pd.read_csv("data/insurance.csv").drop_duplicates()
d = pd.concat([df, new], ignore_index=True)
d["sex"] = (d.sex == "male").astype(int)
d["smoker"] = (d.smoker == "yes").astype(int)
d = pd.get_dummies(d, columns=["region"], drop_first=True)
d["bmi_smoker_interaction"] = d.bmi * d.smoker
d["age_smoker_interaction"] = d.age * d.smoker
d["is_obese"] = (d.bmi >= 30).astype(int)

X = d.drop(columns=["charges"]).iloc[:-1]
y = d.charges.iloc[:-1]

# 3. the same split, so the training rows are identical to the report
X_tv, _, y_tv, _ = train_test_split(X, y, test_size=0.20, random_state=42)
X_tr, _, y_tr, _ = train_test_split(X_tv, y_tv, test_size=0.20, random_state=42)

# 4. the winning model: degree-2 polynomial features + scaling + Ridge(alpha=1.0)
model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("scaler", StandardScaler()),
                  ("ridge", Ridge(alpha=1.0, random_state=42))]).fit(X_tr, y_tr)

print("Predicted annual charge: USD %.2f"
      % model.predict(d.drop(columns=["charges"]).iloc[[-1]])[0])
```

Output for the applicant above:

```
Predicted annual charge: USD 43090.16
```

which is exactly the value in the report's demonstration table — a useful self-check that your environment
reproduces the study. Two cautions when you change the inputs: the region must be one of
`northeast / northwest / southeast / southwest` (`northeast` is the reference level, so it becomes all-zero
dummies), and a prediction is only meaningful inside the training ranges (age 18-64, BMI 16-53, children 0-5).
The model output is a **point estimate in USD for a US-style benchmark dataset — not a quote, and not valid
for underwriting.**

## 8. Launch the web app

```bash
python webapp/export_model.py     # optional: the trained bundle is already committed
python webapp/app.py              # -> http://localhost:5000
```

Open the URL, adjust the sliders / pills / drop-downs and press **Predict annual charge**. The page returns
the predicted annual premium, a likely range of ± 1 test RMSE, and a bar chart of how each input moves the
estimate. Three preset buttons reproduce the demonstration cases from the report
(USD 3,317.16 / 43,090.16 / 15,314.56). Use `PORT=8000 python webapp/app.py` for a different port, or serve
it with `gunicorn -w 2 -b 0.0.0.0:8000 webapp.app:app`.

Command-line equivalent of the same endpoint:

```bash
curl -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' \
     -d '{"age":45,"sex":"male","bmi":34.8,"children":2,"smoker":"yes","region":"northeast"}'
```

If you change the model, re-run `python webapp/export_model.py` so the bundle and the app stay in sync.

---

## 9. Common problems

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: sklearn` | the virtual environment is not active — re-run step 2 |
| `FileNotFoundError: data/insurance.csv` | run the commands from the **repository root**, not from `src/` |
| Figures look different from the report | run `python src/medical_insurance_ml.py` first; the figure script only reads saved results |
| TOC page numbers show `—` in a rebuilt PDF | run `python src/make_pdf.py` a second time |
| `ModuleNotFoundError: flask` | the web app needs Flask — `pip install flask` (it is in `requirements.txt`) |
| Web page loads but prediction fails | the bundle is missing: run `python webapp/export_model.py` |
| `matplotlib` shows no window | expected: `Agg` backend is set on purpose so the scripts run headless |
| Numbers differ from the report | a modified dataset or `random_state` — restore the original CSV and keep `RANDOM_STATE = 42` |

---

## 10. One-line summary of the run order

```bash
pip install -r requirements.txt
python src/medical_insurance_ml.py      # results.json + predictions.json
python src/make_figures_refstyle.py     # figures/*.png
python src/make_pdf.py && python src/make_pdf.py   # docs/*.pdf   (optional)
python src/make_docx.py                            # docs/*.docx (optional)
python webapp/app.py                               # web demo on :5000
```
