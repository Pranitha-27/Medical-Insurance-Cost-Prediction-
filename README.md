# 🏥 Medical Insurance Cost Prediction Using Machine Learning

Predicting annual medical insurance charges from demographic and lifestyle attributes, and comparing four
regression models — built as **Case Study 14** for the Machine Learning course (V Semester, B.E. AI & ML).

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/task-regression-informational">
  <img alt="Best R2" src="https://img.shields.io/badge/best%20R%C2%B2-0.9031-success">
</p>

---

## 📌 Problem statement

> Predict annual medical insurance charges using age, gender, BMI, number of children, smoking habits and
> residential region. Investigate relationships between attributes and insurance costs and compare
> regression models.

Formally, learn a function `f: (age, sex, bmi, children, smoker, region) → charges (USD)` from 1,337 historical
records, and evaluate it with **R², MAE, MSE, RMSE and MAPE**.

---

## 📊 Results (held-out test set, n = 268, `random_state = 42`)

| Model | R² | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| Multiple Linear Regression | 0.8069 | 4,177.05 | 5,956.34 | 41.40 |
| **Polynomial Regression (deg 2, Ridge)** | **0.9031** | **2,353.50** | **4,218.88** | 30.08 |
| Decision Tree Regression | 0.8943 | 2,649.80 | 4,407.90 | 33.50 |
| Random Forest Regression | 0.8962 | 2,435.54 | 4,366.72 | **35.29** |

**Best model:** regularised degree-2 polynomial regression — **29.2 % lower RMSE** than the linear baseline.
Five-fold cross-validation reproduced the ranking (CV R² 0.740 → 0.847 for MLR → Poly).

### What drives the cost?

* **Smoking** is the dominant discriminator: mean charges **USD 8,440** (non-smoker) vs **USD 32,050** (smoker).
* The effect of **BMI is conditional on smoking**: rising from normal to obese BMI adds **USD 1,132** for a
  non-smoker but **USD 21,751** for a smoker.
* Random-forest importance confirms it: `bmi × smoker` (0.344) + `age × smoker` (0.250) + `smoker` (0.188) +
  `age` (0.104) ≈ **89 %** of predictive power. Sex and region are negligible (< 1 % combined).

---

## 🗂 Repository structure

```
medical-insurance-cost-prediction/
├── data/
│   └── insurance.csv                  # Medical Cost Personal Dataset (1,338 rows)
├── docs/
│   ├── Medical_Insurance_..._Report.pdf     # final report (44 pages)
│   └── Medical_Insurance_..._Report.docx    # editable Word version
├── figures/                           # all 13 auto-generated plots
├── src/
│   ├── medical_insurance_ml.py        # ⭐ full pipeline: EDA → models → figures → results.json
│   ├── results.json                   # every number quoted in the report
│   ├── report_content.py              # report text (single source of truth)
│   ├── make_pdf.py                    # renders the PDF (reportlab, two-pass TOC)
│   └── make_docx.py                   # renders the DOCX (python-docx)
├── medical_insurance_cost_prediction.ipynb   # executed walkthrough notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
pip install -r requirements.txt

# 1. Reproduce every result, figure and results.json  (~60 s on a laptop CPU)
python src/medical_insurance_ml.py

# 2. (optional) rebuild the report from those results
python src/make_pdf.py     # needs reportlab
python src/make_docx.py    # needs python-docx
```

Or just open **`medical_insurance_cost_prediction.ipynb`** in Jupyter/Colab and run all cells.

---

## 🔬 Methodology

| Stage | What was done |
|---|---|
| **Cleaning** | 0 missing values; 1 exact duplicate removed (1,338 → 1,337); outliers **retained** (139 genuine high-cost claims — deleting them would bias predictions) |
| **Encoding** | binary map for `sex` / `smoker`; one-hot for `region` (`drop_first=True`) |
| **Scaling** | `StandardScaler` **inside** the pipeline for the polynomial model only; trees are scale-invariant; MLR left unscaled for interpretable USD coefficients |
| **Feature engineering** | `bmi × smoker`, `age × smoker`, `is_obese` (BMI ≥ 30) — motivated by EDA, not guesswork |
| **Split** | 64 % train / 16 % validation / 20 % test, `random_state = 42` |
| **Leakage control** | dedup before splitting; scaler + polynomial expansion fitted only on training folds; `GridSearchCV` never sees the test set |
| **Tuning** | 5-fold `GridSearchCV` (neg-RMSE) on decision tree and random forest |
| **Error analysis** | residual plots, absolute error by smoking status, learning curve vs training size |

### Algorithms

1. **Multiple Linear Regression** — OLS baseline, interpretable USD-per-unit coefficients.
2. **Polynomial Regression (deg 2)** — 11 → 77 terms + L2 (Ridge, α = 1.0) to control variance.
3. **Decision Tree Regression** — tuned to depth 5 / 21 leaves.
4. **Random Forest Regression** — tuned (200 trees, depth 10, `max_features="sqrt"`).

---

## 📈 Figures

| | |
|---|---|
| `fig3_1_charges_distribution.png` | target is right-skewed (skew 1.52) and multimodal |
| `fig3_4_scatter.png` | the smoking bands + BMI interaction |
| `fig5_1_workflow.png` | end-to-end pipeline flowchart |
| `fig6_1_model_comparison.png` | R² and RMSE by model |
| `fig6_2_actual_vs_predicted.png` | linear model's horizontal banding vs polynomial fit |
| `fig6_3_residuals.png` | heteroscedasticity of the linear baseline |
| `fig6_4_feature_importance.png` | interaction features dominate |
| `fig6_5_error_by_smoker.png` | subgroup error analysis |
| `figB1_learning_curve.png` | performance saturates at the available sample size |

---

## ⚠️ Limitations

* Only 6 attributes and 1,337 rows — no diagnosis codes, prior claims, income or plan type. The residual
  RMSE (~USD 4,219, ~30 % of mean charge) is largely a **data** limitation.
* Single random split, no out-of-time validation; US-derived benchmark, so conclusions do not transfer to
  the Indian market without external validation.
* Relative error is much larger for low-cost non-smokers than for high-cost smokers.
* Point estimates only — no prediction intervals. **Not for premium setting, underwriting or clinical use.**

---

## 🔭 Future work

Gradient boosting (XGBoost/LightGBM), SVR-RBF, quantile/Tweedie distributional models for prediction
intervals, SHAP-based explainability, formal fairness audit, log-target or two-part models, external
validation, and a small web form for interactive demonstration.

---

## 📚 Dataset & references

**Dataset:** Medical Cost Personal Dataset — [Kaggle: mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(public, anonymised, educational/research use). Originally distributed with *Machine Learning with R* by Brett Lantz.

Key literature (full IEEE list in `docs/` → References):

1. K. Kaushik *et al.*, "Machine learning-based regression framework to predict health insurance premiums," *IJERPH*, 19(13):7898, 2022.
2. S. Panda *et al.*, "Health insurance cost prediction using regression models," *IEEE COM-IT-CON*, 2022.
3. J. Ruth Sandra *et al.*, "Prediction of cost for medical care insurance by using regression models," *Springer ERCICA*, 2024.
4. U. Orji, E. Ukwandu, "Machine learning for an explainable cost prediction of medical insurance," *ML with Applications*, 15:100516, 2024.
5. M. Hanafy, O. M. A. Mahmoud, "Predict health insurance cost by using ML and DNN regression models," *IJITEE*, 10(3):137–143, 2021.

---

## 📄 License

Code released under the MIT License (see `LICENSE`). The dataset remains subject to its original Kaggle
terms; the report is submitted for academic evaluation.
