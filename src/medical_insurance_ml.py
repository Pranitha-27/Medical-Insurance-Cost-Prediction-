"""
Medical Insurance Cost Prediction - Machine Learning Case Study
Dataset : Medical Cost Personal Dataset (Kaggle: mirichoi0218/insurance)
Task    : Regression - predict annual medical insurance charges (USD)

Trains and compares: Multiple Linear Regression, Polynomial Regression (deg 2),
Decision Tree Regression and Random Forest Regression.
Produces all figures used in the report + results.json with every number.
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)

RANDOM_STATE = 42

# resolve paths so the script works from the repo root or from src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_PATH = os.path.join(_ROOT, "data", "insurance.csv")
FIG = os.path.join(_ROOT, "figures")
RESULTS_PATH = os.path.join(_ROOT, "src", "results.json")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titlesize": 11,
                     "axes.labelsize": 10})

R = {}   # results container -> results.json

# ----------------------------------------------------------------------------
# 1. LOAD + OVERVIEW
# ----------------------------------------------------------------------------
df_raw = pd.read_csv(DATA_PATH)
n0 = len(df_raw)
df = df_raw.drop_duplicates().reset_index(drop=True)
R["shape"] = list(df.shape)
R["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
R["head"] = df.head(5).round(3).to_dict(orient="records")
R["describe"] = df.describe().round(3).to_dict()
R["describe_full"] = df.describe(include="all").round(3).to_string()
R["missing_total"] = int(df.isna().sum().sum())
R["duplicates"] = int(df_raw.duplicated().sum())
R["n_raw"] = int(n0)
R["n_clean"] = int(len(df))

for c in ["sex", "smoker", "region"]:
    R[f"counts_{c}"] = df[c].value_counts().to_dict()

# charges by group (for EDA narrative)
R["mean_charges_by_smoker"] = df.groupby("smoker")["charges"].mean().round(2).to_dict()
R["mean_charges_by_sex"] = df.groupby("sex")["charges"].mean().round(2).to_dict()
R["mean_charges_by_region"] = df.groupby("region")["charges"].mean().round(2).to_dict()
R["mean_charges_by_children"] = df.groupby("children")["charges"].mean().round(2).to_dict()
R["median_charges_by_smoker"] = df.groupby("smoker")["charges"].median().round(2).to_dict()
R["bmi_smoker_means"] = df.groupby(["smoker", pd.cut(df.bmi, [0, 18.5, 25, 30, 100],
                            labels=["Under", "Normal", "Over", "Obese"])], observed=True
                            )["charges"].mean().round(2).to_dict()
R["bmi_smoker_means"] = {f"{k[0]}|{k[1]}": v for k, v in R["bmi_smoker_means"].items()}
R["corr_matrix"] = df[["age", "bmi", "children", "charges"]].corr().round(4).to_dict()
R["corr_with_target"] = df[["age", "bmi", "children", "charges"]].corr()["charges"].round(4).to_dict()

# Outlier check by IQR
iqr_report = {}
for c in ["age", "bmi", "children", "charges"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_report[c] = {"q1": round(q1, 3), "q3": round(q3, 3),
                     "lower": round(lo, 3), "upper": round(hi, 3),
                     "n_outliers": int(((df[c] < lo) | (df[c] > hi)).sum()),
                     "skew": round(float(df[c].skew()), 3)}
R["iqr"] = iqr_report

# ----------------------------------------------------------------------------
# 2. EDA FIGURES
# ----------------------------------------------------------------------------
# Fig 3.1 charges distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.histplot(df["charges"], kde=True, ax=ax[0], color="#2a6f97", bins=35)
ax[0].set_title("Distribution of Annual Charges")
ax[0].set_xlabel("Charges (USD)"); ax[0].set_ylabel("Frequency")
sns.boxplot(x=df["charges"], ax=ax[1], color="#89c2d9")
ax[1].set_title("Box Plot of Annual Charges")
ax[1].set_xlabel("Charges (USD)")
plt.savefig(f"{FIG}/fig3_1_charges_distribution.png"); plt.close()

# Fig 3.2 smoker vs charges
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.boxplot(data=df, x="smoker", y="charges", hue="smoker", legend=False,
            palette=["#4c956c", "#c1121f"], ax=ax[0])
ax[0].set_title("Charges by Smoking Status"); ax[0].set_xlabel("Smoker"); ax[0].set_ylabel("Charges (USD)")
sns.boxplot(data=df, x="children", y="charges", ax=ax[1], color="#f4a261")
ax[1].set_title("Charges by Number of Children"); ax[1].set_xlabel("Children"); ax[1].set_ylabel("Charges (USD)")
plt.savefig(f"{FIG}/fig3_2_smoker_children.png"); plt.close()

# Fig 3.3 numerical feature distributions
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for a, c in zip(ax, ["age", "bmi", "children"]):
    sns.histplot(df[c], kde=(c != "children"), ax=a, color="#343a40", bins=25)
    a.set_title(f"Distribution of {c.capitalize()}")
    a.set_xlabel(c); a.set_ylabel("Frequency")
plt.savefig(f"{FIG}/fig3_3_numeric_distributions.png"); plt.close()

# Fig 3.4 scatter age/bmi vs charges
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", s=16, ax=ax[0])
ax[0].set_title("Age vs Charges (coloured by Smoker)"); ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("Charges (USD)")
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", s=16, ax=ax[1])
ax[1].set_title("BMI vs Charges (coloured by Smoker)"); ax[1].set_xlabel("BMI"); ax[1].set_ylabel("Charges (USD)")
plt.savefig(f"{FIG}/fig3_4_scatter.png"); plt.close()

# Fig 3.5 correlation heatmap
plt.figure(figsize=(5.2, 4.2))
sns.heatmap(df[["age", "bmi", "children", "charges"]].corr(), annot=True,
            fmt=".3f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix (Numerical Features)")
plt.savefig(f"{FIG}/fig3_5_corr.png"); plt.close()

# Fig 3.6 region / sex charges
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.boxplot(data=df, x="region", y="charges", ax=ax[0], color="#6d597a")
ax[0].set_title("Charges by Region"); ax[0].tick_params(axis="x", rotation=15)
sns.boxplot(data=df, x="sex", y="charges", ax=ax[1], color="#457b9d")
ax[1].set_title("Charges by Sex")
plt.savefig(f"{FIG}/fig3_6_region_sex.png"); plt.close()

# ----------------------------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------------------------
data = df.copy()
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data = pd.get_dummies(data, columns=["region"], drop_first=True, dtype=int)

# engineered features
data["bmi_smoker_interaction"] = data["bmi"] * data["smoker"]
data["age_smoker_interaction"] = data["age"] * data["smoker"]
data["is_obese"] = (data["bmi"] >= 30).astype(int)

X = data.drop(columns=["charges"])
y = data["charges"]
R["engineered_features"] = ["bmi_smoker_interaction", "age_smoker_interaction", "is_obese"]
R["final_feature_list"] = list(X.columns)
R["n_features"] = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=RANDOM_STATE)
R["split"] = {"train": int(len(X_tr)), "val": int(len(X_val)), "test": int(len(X_test)),
              "train_pct": round(100 * len(X_tr) / len(X), 1),
              "val_pct": round(100 * len(X_val) / len(X), 1),
              "test_pct": round(100 * len(X_test) / len(X), 1)}

# ----------------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"R2": round(float(r2_score(y_true, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "MSE": round(float(mean_squared_error(y_true, y_pred)), 2),
            "RMSE": round(rmse, 2),
            "MAPE_pct": round(100 * float(mean_absolute_percentage_error(y_true, y_pred)), 2)}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models, preds = {}, {}

# --- 5.2.1 Multiple Linear Regression (baseline, raw features) ---------------
Xb = df[["age", "bmi", "children"]].copy()
Xb["sex"] = (df["sex"] == "male").astype(int)
Xb["smoker"] = (df["smoker"] == "yes").astype(int)
Xb = pd.concat([Xb, pd.get_dummies(df["region"], prefix="region", drop_first=True, dtype=int)], axis=1)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, y, test_size=0.20, random_state=RANDOM_STATE)
lin = LinearRegression().fit(Xb_tr, yb_tr)
p = lin.fit(Xb_tr, yb_tr).predict(Xb_te)
models["Multiple Linear Regression"] = lin
preds["Multiple Linear Regression"] = (yb_te, p)
R["linear_coefs"] = {k: round(float(v), 2) for k, v in zip(Xb.columns, lin.coef_)}
R["linear_intercept"] = round(float(lin.intercept_), 2)
R["linear_cv_r2"] = round(float(cross_val_score(LinearRegression(), Xb, y, cv=kf, scoring="r2").mean()), 4)
R["lin_metrics"] = metrics(yb_te, p)
# simple single-feature LR for illustration
simple = LinearRegression().fit(df[["age"]], y)
R["simple_age_r2"] = round(float(simple.score(df[["age"]], y)), 4)
R["simple_age_coef"] = round(float(simple.coef_[0]), 2)

# --- 5.2.2 Polynomial Regression (degree 2, scaled) -------------------------
poly_pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("scaler", StandardScaler()),
                      ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
poly_pipe.fit(X_tr, y_tr)
p = poly_pipe.predict(X_test)
models["Polynomial Regression (deg 2)"] = poly_pipe
preds["Polynomial Regression (deg 2)"] = (y_test, p)
R["poly_n_features"] = int(poly_pipe.named_steps["poly"].n_output_features_)
R["poly_cv_r2"] = round(float(cross_val_score(poly_pipe, X_train, y_train, cv=kf, scoring="r2").mean()), 4)
R["poly_metrics"] = metrics(y_test, p)

# --- 5.2.3 Decision Tree Regression (tuned) --------------------------------
dt_grid = {"max_depth": [3, 4, 5, 6, 8, 10, 12, None],
           "min_samples_leaf": [1, 2, 3, 5, 10, 20],
           "min_samples_split": [2, 5, 10, 20]}
dt_search = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                         dt_grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
dt_search.fit(X_tr, y_tr)
dt = dt_search.best_estimator_
p = dt.predict(X_test)
models["Decision Tree Regression"] = dt
preds["Decision Tree Regression"] = (y_test, p)
R["dt_best_params"] = {k: (v if v is not None else "None") for k, v in dt_search.best_params_.items()}
R["dt_cv_rmse"] = round(float(-dt_search.best_score_), 2)
R["dt_metrics"] = metrics(y_test, p)
R["dt_depth"] = int(dt.get_depth()); R["dt_leaves"] = int(dt.get_n_leaves())

# --- 5.2.4 Random Forest Regression (tuned) -------------------------------
rf_grid = {"n_estimators": [200, 400],
           "max_depth": [6, 10, 15, None],
           "min_samples_leaf": [1, 2, 3, 5],
           "max_features": ["sqrt", 1.0]}
rf_search = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE),
                         rf_grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_search.fit(X_tr, y_tr)
rf = rf_search.best_estimator_
p = rf.predict(X_test)
models["Random Forest Regression"] = rf
preds["Random Forest Regression"] = (y_test, p)
R["rf_best_params"] = rf_search.best_params_
R["rf_cv_rmse"] = round(float(-rf_search.best_score_), 2)
R["rf_metrics"] = metrics(y_test, p)
R["rf_importance"] = {k: round(float(v), 4) for k, v in
                      sorted(zip(X.columns, rf.feature_importances_), key=lambda t: -t[1])}

R["metrics_table"] = {name: metrics(yt, yp) for name, (yt, yp) in preds.items()}
R["best_model"] = min(R["metrics_table"], key=lambda m: R["metrics_table"][m]["RMSE"])
R["test_y_stats"] = {"min": round(float(y_test.min()), 2), "max": round(float(y_test.max()), 2),
                     "mean": round(float(y_test.mean()), 2)}
R["nontest_mae_best"] = {}

# ----------------------------------------------------------------------------
# 5. VISUAL RESULTS
# ----------------------------------------------------------------------------
best_name = R["best_model"]
best = models[best_name]
y_test_best, p_best = preds[best_name]

# Fig 6.1 model comparison bar chart
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
names = list(R["metrics_table"]); short = ["MLR", "Poly-2", "DTR", "RFR"]
r2s = [R["metrics_table"][n]["R2"] for n in names]
rmses = [R["metrics_table"][n]["RMSE"] for n in names]
b1 = ax[0].bar(short, r2s, color="#2a6f97"); ax[0].set_title("Test R² by Model")
ax[0].set_ylabel("R²"); ax[0].set_ylim(0, 1)
ax[0].bar_label(b1, fmt="%.3f", fontsize=8)
b2 = ax[1].bar(short, rmses, color="#bc4749"); ax[1].set_title("Test RMSE by Model")
ax[1].set_ylabel("RMSE (USD)")
ax[1].bar_label(b2, fmt="%.0f", fontsize=8)
plt.savefig(f"{FIG}/fig6_1_model_comparison.png"); plt.close()

# Fig 6.2 actual vs predicted
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for a, (name, (yt, yp)) in zip(ax, [("Multiple Linear Regression", preds["Multiple Linear Regression"]),
                                    (best_name, preds[best_name])]):
    a.scatter(yt, yp, s=14, alpha=.75, color="#2a6f97", edgecolor="w", linewidth=.3)
    lim = [0, max(yt.max(), yp.max()) * 1.03]
    a.plot(lim, lim, "r--", lw=1.2, label="Perfect prediction")
    a.set_xlabel("Actual Charges (USD)"); a.set_ylabel("Predicted Charges (USD)")
    a.set_title(f"{name}\nR² = {r2_score(yt, yp):.3f}"); a.legend(fontsize=8)
plt.savefig(f"{FIG}/fig6_2_actual_vs_predicted.png"); plt.close()

# Fig 6.3 residual plots
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for a, (name, (yt, yp)) in zip(ax, [("Multiple Linear Regression", preds["Multiple Linear Regression"]),
                                    (best_name, preds[best_name])]):
    res = yt - yp
    a.scatter(yp, res, s=14, alpha=.75, color="#6a4c93", edgecolor="w", linewidth=.3)
    a.axhline(0, color="crimson", ls="--", lw=1.2)
    a.set_xlabel("Predicted Charges (USD)"); a.set_ylabel("Residual (USD)")
    a.set_title(f"Residuals – {name}")
plt.savefig(f"{FIG}/fig6_3_residuals.png"); plt.close()

# Fig 6.4 feature importance
imp = pd.Series(R["rf_importance"]).head(10)[::-1]
plt.figure(figsize=(6.4, 3.9))
plt.barh(imp.index, imp.values, color="#386641")
plt.xlabel("Importance (mean decrease in impurity)")
plt.title("Random Forest – Top 10 Feature Importances")
plt.savefig(f"{FIG}/fig6_4_feature_importance.png"); plt.close()

# Fig 6.5 error by smoker group (fairness / error analysis)
err_df = pd.DataFrame({"actual": y_test.values, "pred": np.asarray(preds[best_name][1])})
err_df["smoker"] = df.loc[y_test.index, "smoker"].values
err_df["abs_err"] = (err_df.actual - err_df.pred).abs()
R["mae_by_smoker"] = err_df.groupby("smoker")["abs_err"].mean().round(2).to_dict()
plt.figure(figsize=(4.6, 3.4))
sns.boxplot(data=err_df, x="smoker", y="abs_err", hue="smoker", legend=False,
            palette=["#4c956c", "#c1121f"])
plt.ylabel("Absolute Error (USD)"); plt.xlabel("Smoker")
plt.title(f"Absolute Error by Smoking Status\n({best_name})")
plt.savefig(f"{FIG}/fig6_5_error_by_smoker.png"); plt.close()

# Fig 5.1 workflow flowchart (drawn with matplotlib)
fig, ax = plt.subplots(figsize=(4.6, 10.2))
ax.axis("off")
steps = ["Dataset\n(Medical Cost Personal, n=1338)", "Exploratory Data\nAnalysis (EDA)",
         "Preprocessing\n(encoding, duplicates, outliers)", "Feature Engineering\n(interactions, is_obese)",
         "Train / Validation / Test\nSplit (64 / 16 / 20)",
         "Model Training\nMLR · Poly-2 · DTR · RFR",
         "Evaluation\n(R², MAE, RMSE, MAPE)", "Model Comparison\n& Selection",
         "Prediction / Demonstration"]
y0, h, w = 9.55, 0.72, 0.66
for i, s in enumerate(steps):
    y = y0 - i * 1.06
    box = plt.Rectangle((0.17, y - h / 2), w, h, facecolor="#e8f1f8" if i % 2 == 0 else "#dcebdc",
                        edgecolor="#1d3557", lw=1.1, zorder=2)
    ax.add_patch(box)
    ax.text(0.17 + w / 2, y, s, ha="center", va="center", fontsize=8.6, zorder=3,
            fontweight="bold" if i in (0, 8) else "normal")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 1.06 + h / 2), xytext=(0.5, y - h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#1d3557", lw=1.3))
ax.set_xlim(0, 1); ax.set_ylim(0.1, 10)
ax.set_title("Fig. 5.1  End-to-End ML Workflow for the Case Study", fontsize=10)
plt.savefig(f"{FIG}/fig5_1_workflow.png"); plt.close()

# ----------------------------------------------------------------------------
# 6. SAMPLE PREDICTIONS (5 representative test cases)
# ----------------------------------------------------------------------------
sample_idx = y_test.index[:5]
sample_rows = df.loc[sample_idx, ["age", "sex", "bmi", "children", "smoker", "region", "charges"]]
sample_pred = best.predict(X.loc[sample_idx])
R["sample_predictions"] = []
for i, idx in enumerate(sample_idx):
    r = sample_rows.loc[idx].to_dict()
    r["actual"] = round(float(r.pop("charges")), 2)
    r["predicted"] = round(float(sample_pred[i]), 2)
    r["error"] = round(r["predicted"] - r["actual"], 2)
    r["abs_pct_err"] = round(100 * abs(r["error"]) / r["actual"], 2)
    R["sample_predictions"].append(r)

# New unseen customers demonstration
new = pd.DataFrame([
    {"age": 24, "sex": 0, "bmi": 22.5, "children": 0, "smoker": 0, "region_northwest": 0,
     "region_southeast": 0, "region_southwest": 1},
    {"age": 45, "sex": 1, "bmi": 34.8, "children": 2, "smoker": 1, "region_northwest": 0,
     "region_southeast": 0, "region_southwest": 0},
    {"age": 60, "sex": 1, "bmi": 28.1, "children": 3, "smoker": 0, "region_northwest": 0,
     "region_southeast": 1, "region_southwest": 0},
])
for c in X.columns:
    if c not in new.columns:
        new[c] = 0
new = new[X.columns]
new["bmi_smoker_interaction"] = new["bmi"] * new["smoker"]
new["age_smoker_interaction"] = new["age"] * new["smoker"]
new["is_obese"] = (new["bmi"] >= 30).astype(int)
R["new_customer_predictions"] = [round(float(v), 2) for v in best.predict(new[X.columns])]

# ----------------------------------------------------------------------------
# 7. LEARNING CURVE / DATA SIZE SENSITIVITY (appendix)
# ----------------------------------------------------------------------------
sizes, r2s_lc = [], []
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    n = int(len(X_tr) * frac)
    m = RandomForestRegressor(**{**rf_search.best_params_, "random_state": RANDOM_STATE})
    m.fit(X_tr.iloc[:n], y_tr.iloc[:n])
    sizes.append(n); r2s_lc.append(round(float(r2_score(y_test, m.predict(X_test))), 4))
R["learning_curve"] = {"train_size": sizes, "test_r2": r2s_lc}
plt.figure(figsize=(5.2, 3.4))
plt.plot(sizes, r2s_lc, "o-", color="#2a6f97")
plt.xlabel("Number of Training Records"); plt.ylabel("Test R²")
plt.title("Random Forest – Test R² vs Training Set Size")
plt.savefig(f"{FIG}/figB1_learning_curve.png"); plt.close()

with open(RESULTS_PATH, "w") as f:
    json.dump(R, f, indent=2, default=str)

print(json.dumps({k: R[k] for k in ["shape", "missing_total", "duplicates", "split",
                                    "metrics_table", "best_model", "rf_best_params",
                                    "dt_best_params", "mae_by_smoker",
                                    "sample_predictions", "new_customer_predictions"]}, indent=2, default=str))
