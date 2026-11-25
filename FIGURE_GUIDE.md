# Figure Guide - Medical Insurance Cost Prediction

Every figure in `docs/Medical_Insurance_Cost_Prediction_ML_Case_Study_Report.pdf` is reproduced here with
(a) what it plots, (b) where it appears in the report, (c) how to read it and what it proves, and (d) the
exact code that generates it.

All figures are produced by **one script**, `src/make_figures_refstyle.py`, which reads the saved
results (`src/results.json`, `src/predictions.json`) and the dataset (`data/insurance.csv`), so the plots
always agree with the numbers in the report. The code shown for each figure is the exact block from that
script. To regenerate everything:

```bash
python src/medical_insurance_ml.py        # trains the models, writes results.json + predictions.json
python src/make_figures_refstyle.py       # redraws all figures into figures/
```

The report contains **10 figures**: Fig. 3.1, 3.2, 3.3, 5.1, 6.1, 6.2, 6.3, 6.4, 6.5 and Fig. B.1, and
`figures/` holds exactly those ten PNGs. Three exploratory plots from an earlier draft (numeric
distributions, correlation heat map, region/sex split) were merged into the prose of Section 3.4 during
the revision that shortened the report, so their code blocks no longer exist.

| Figure | File | Report section |
|---|---|---|
| Fig. 3.1 | `figures/fig3_1_charges_distribution.png` | Section 3.4 |
| Fig. 3.2 | `figures/fig3_2_smoker_children.png` | Section 3.4 |
| Fig. 3.3 | `figures/fig3_3_scatter.png` | Section 3.4 |
| Fig. 5.1 | `figures/fig5_1_workflow.png` | Section 5.1 |
| Fig. 6.1 | `figures/fig6_1_model_comparison.png` | Section 6.3 |
| Fig. 6.2 | `figures/fig6_2_actual_vs_predicted.png` | Section 6.3 |
| Fig. 6.3 | `figures/fig6_3_residuals.png` | Section 6.3 |
| Fig. 6.4 | `figures/fig6_4_feature_importance.png` | Section 6.3 |
| Fig. 6.5 | `figures/fig6_5_error_analysis.png` | Section 6.3 |
| Fig. B.1 | `figures/figB1_learning_curve.png` | Section B.2 |

---

## 1. Fig. 3.1 - Distribution of Annual Medical Insurance Charges

**File:** `figures/fig3_1_charges_distribution.png`  
**Appears in:** Section 3.4, Chapter 3 (Exploratory Data Analysis)

### What the figure plots

A 30-bin histogram of the `charges` column with alternating grey/hatched bars. The x-axis is annual charges in USD and the y-axis is the number of policyholders in each bin.

### How to read it, and what it establishes

It answers the first question of any regression study: what shape is the target? The plot shows a dense low-cost mode below USD 15,000, a middle band near USD 20,000-35,000 and a small high-cost cluster above USD 40,000. That multimodality plus the long right tail (skewness 1.52) proves that a single linear function of the inputs cannot fit the data, which is why the study goes on to fit polynomial and tree models. It is also why 139 high-cost records were kept rather than trimmed: they are a real sub-population, not noise.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 3.1 charges
fig, ax = plt.subplots(figsize=(6.3, 3.5))
counts, bins, patches = ax.hist(df["charges"], bins=30, edgecolor="black", linewidth=0.8)
for i, p in enumerate(patches):
    p.set_facecolor("white" if i % 2 else GREY)
    p.set_hatch("//" if i % 2 else "")
ax.set_xlabel("Annual Insurance Charges (USD)")
ax.set_ylabel("Number of Policyholders")
ax.set_title("Distribution of Annual Medical Insurance Charges")
save(fig, "fig3_1_charges_distribution.png")
```

---

## 2. Fig. 3.2 - Charges by Smoking Status and by Number of Children

**File:** `figures/fig3_2_smoker_children.png`  
**Appears in:** Section 3.4, Chapter 3 (Exploratory Data Analysis)

### What the figure plots

Two side-by-side box plots. The left panel splits charges into non-smoker and smoker groups; the right panel splits charges by the number of children (0-5). Boxes are grey/hatched, whiskers and outliers are drawn to the 1.5 x IQR rule.

### How to read it, and what it establishes

The left panel is the single most important plot in the EDA: the two smoking groups barely overlap, with mean charges of USD 8,440.66 for non-smokers against USD 32,050.23 for smokers (a factor of about 3.8). This explains most of the multimodality seen in Fig. 3.1. The right panel shows that family size is a weak, non-monotonic predictor, so it is expected to be a minor feature - a prediction the random forest importance in Fig. 6.4 confirms (children = 0.0154).

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 3.2 smoker / children
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
data = [df.loc[df.smoker == "no", "charges"], df.loc[df.smoker == "yes", "charges"]]
bp = axes[0].boxplot(data, tick_labels=["Non-smoker", "Smoker"], patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", lw=1.4))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor("white" if i else GREY); b.set_hatch("//" if i else "")
    b.set_edgecolor("black")
axes[0].set_ylabel("Charges (USD)")
axes[0].set_title("Charges by Smoking Status")
groups = [df.loc[df.children == k, "charges"] for k in range(6)]
bp2 = axes[1].boxplot(groups, tick_labels=[str(k) for k in range(6)], patch_artist=True, widths=0.6,
                      medianprops=dict(color="black", lw=1.2))
for i, b in enumerate(bp2["boxes"]):
    b.set_facecolor("white" if i % 2 else GREY); b.set_hatch("//" if i % 2 else "")
    b.set_edgecolor("black")
axes[1].set_xlabel("Number of Children"); axes[1].set_ylabel("Charges (USD)")
axes[1].set_title("Charges by Number of Children")
fig.tight_layout()
save(fig, "fig3_2_smoker_children.png")
```

---

## 3. Fig. 3.3 - Age and BMI versus Charges, Separated by Smoking Status

**File:** `figures/fig3_3_scatter.png`  
**Appears in:** Section 3.4, Chapter 3 (Exploratory Data Analysis)

### What the figure plots

Two scatter plots. Left: age (years) against charges. Right: BMI against charges. In both panels, non-smokers are hollow circles and smokers are filled triangles, so the two sub-populations can be compared within each panel.

### How to read it, and what it establishes

This figure reveals the interaction that drives the whole modelling strategy. Against age, non-smokers form a slowly rising lower band while smokers occupy two higher bands, so the smoker flag behaves as a band selector rather than an additive shift. In the BMI panel the non-smoker relationship is almost flat, whereas for smokers cost rises steeply once BMI passes about 30. Because the effect of BMI depends on smoking status, an additive model cannot represent it - which is exactly why the three interaction features (bmi x smoker, age x smoker, is_obese) were engineered in Section 4.3.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 3.3 scatters  (file name fig3_3_scatter.png kept; report label is Fig. 3.3)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
for ax, col, lab in zip(axes, ["age", "bmi"], ["Age (Years)", "Body Mass Index"]):
    ns = df[df.smoker == "no"]
    sm = df[df.smoker == "yes"]
    ax.scatter(ns[col], ns["charges"], s=11, facecolor="white", edgecolor="black",
               linewidth=0.5, label="Non-smoker")
    ax.scatter(sm[col], sm["charges"], s=14, facecolor="black", edgecolor="black",
               linewidth=0.4, marker="^", label="Smoker")
    ax.set_xlabel(lab); ax.set_ylabel("Charges (USD)")
    ax.legend(fontsize=8, frameon=True, edgecolor="black")
axes[0].set_title("Age vs Charges by Smoking Status")
axes[1].set_title("BMI vs Charges by Smoking Status")
fig.tight_layout()
save(fig, "fig3_3_scatter.png")
```

---

## 4. Fig. 5.1 - End-to-End Machine Learning Workflow for Insurance Cost Prediction

**File:** `figures/fig5_1_workflow.png`  
**Appears in:** Section 5.1, Chapter 5 (Machine Learning Model Development)

### What the figure plots

A vertical flowchart of nine boxes connected by arrows: dataset -> EDA -> data cleaning -> encoding and feature engineering -> train/validation/test split -> model training -> evaluation and tuning -> model comparison and error analysis -> prediction/demonstration. Alternating boxes are shaded grey for legibility.

### How to read it, and what it establishes

The flowchart documents the exact order of operations so the study is reproducible and so the reader can see where leakage is prevented. Two boxes carry the methodological guarantees: the split box fixes the 64/16/20 partition with random_state = 42, and the evaluation box states that grid search runs with five-fold cross-validation inside the training subset only. The cleaning box notes that the duplicate was removed and outliers were deliberately retained.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 5.1 workflow
fig, ax = plt.subplots(figsize=(5.0, 8.2))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 10)
steps = ["Medical Cost Personal Dataset\n(1,338 Raw Records)",
         "Exploratory Data Analysis\n(Distribution, Group and Interaction Checks)",
         "Data Cleaning\n(1 Duplicate Removed, Outliers Retained)",
         "Encoding and Feature Engineering\n(Binary/One-Hot, BMI x Smoker,\nAge x Smoker, is_obese)",
         "Train / Validation / Test Split\n(64% / 16% / 20%, random_state = 42)",
         "Model Training\n(MLR, Polynomial Deg-2, Decision Tree,\nRandom Forest)",
         "Evaluation and Tuning\n(R2, MAE, MSE, RMSE, MAPE; 5-Fold GridSearchCV)",
         "Model Comparison and Error Analysis\n(Residuals, Subgroup Error, Importance)",
         "Prediction / Demonstration\n(Held-Out Records and New Applicants)"]
ys = np.linspace(9.25, 0.75, len(steps))
H = 0.70
for i, (s_, ycen) in enumerate(zip(steps, ys)):
    box = Rectangle((0.05, ycen - H / 2), 0.90, H,
                    facecolor="white" if i % 2 == 0 else "#ececec",
                    edgecolor="black", linewidth=1.0, zorder=2)
    ax.add_patch(box)
    ax.text(0.5, ycen, s_, ha="center", va="center", fontsize=7.3, zorder=3,
            fontweight="bold" if i in (0, 8) else "normal")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, ys[i + 1] + H / 2 + 0.02), xytext=(0.5, ycen - H / 2 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0,
                                    mutation_scale=11))
ax.set_title("End-to-End Machine Learning Workflow\nfor Insurance Cost Prediction",
             fontweight="bold", fontsize=9.5)
save(fig, "fig5_1_workflow.png")
```

---

## 5. Fig. 6.1 - Comparison of Test R2 and Test RMSE Across the Four Regression Models

**File:** `figures/fig6_1_model_comparison.png`  
**Appears in:** Section 6.3, Chapter 6 (Results and Performance Evaluation)

### What the figure plots

Two bar charts sharing the same four models. The left chart plots test R2 (higher is better, scale 0-1); the right chart plots test RMSE in USD (lower is better). Each bar is labelled with its exact value, and the polynomial model's bar is shaded to draw the eye to the winner.

### How to read it, and what it establishes

This is the headline result of the study. Polynomial regression (degree 2, Ridge alpha = 1.0) is best on both metrics with R2 = 0.9031 and RMSE = USD 4,218.88, followed by random forest (0.8962 / USD 4,366.72), decision tree (0.8943 / USD 4,407.90) and the linear baseline (0.8069 / USD 5,956.34). The large gap between the linear bar and the other three shows that the interaction structure found in the EDA matters; the small gap among the three non-linear models shows that representing the interaction is what counts, not the choice of algorithm.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 6.1 model comparison
M = R["metrics_table"]
names = ["Multiple\nLinear Reg.", "Polynomial\nReg. (Deg 2)",
         "Decision\nTree Reg.", "Random\nForest Reg."]
r2 = [M["Multiple Linear Regression"]["R2"], M["Polynomial Regression (deg 2)"]["R2"],
      M["Decision Tree Regression"]["R2"], M["Random Forest Regression"]["R2"]]
rmse = [M["Multiple Linear Regression"]["RMSE"], M["Polynomial Regression (deg 2)"]["RMSE"],
        M["Decision Tree Regression"]["RMSE"], M["Random Forest Regression"]["RMSE"]]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
for ax, vals, title, ylab, fmt in [
        (axes[0], r2, "Test R\u00b2 Across Implemented Regression Models", "R\u00b2 (Coefficient of Determination)", "{:.4f}"),
        (axes[1], rmse, "Test RMSE Across Implemented Regression Models", "RMSE (USD)", "{:,.0f}")]:
    bars = ax.bar(range(4), vals, edgecolor="black", linewidth=0.9, width=0.62,
                  facecolor=["white", GREY, "white", GREY])
    for b, hatch in zip(bars, ["", "//", "", "//"]):
        b.set_hatch(hatch)
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
fig.tight_layout()
save(fig, "fig6_1_model_comparison.png")
```

---

## 6. Fig. 6.2 - Predicted versus Actual Charges for the Linear Baseline and the Best Model

**File:** `figures/fig6_2_actual_vs_predicted.png`  
**Appears in:** Section 6.3, Chapter 6 (Results and Performance Evaluation)

### What the figure plots

Two scatter plots of predicted charge (y-axis) against actual charge (x-axis), one for multiple linear regression and one for the degree-2 polynomial model, with a dashed 45-degree line marking perfect prediction. Points come from the saved test-set predictions in src/predictions.json.

### How to read it, and what it establishes

The shapes of the point clouds diagnose the models. The linear model produces nearly horizontal bands: because a linear function can only shift its output by a few thousand dollars, it over-predicts cheap individuals and severely under-predicts expensive ones. The polynomial model's points track the diagonal closely, with only a handful of very expensive non-smokers still under-predicted because such cases are rare in training. This visual evidence corroborates the R2 and RMSE numbers in Table 6.1.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 6.2 actual vs predicted
import pickle  # noqa
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
pred_store = json.load(open(os.path.join(ROOT, "src", "predictions.json")))
for ax, key, title in [(axes[0], "mlr", "Multiple Linear Regression"),
                       (axes[1], "poly", "Polynomial Regression (Deg 2)")]:
    yt = np.array(pred_store[key]["y_true"]); yp = np.array(pred_store[key]["y_pred"])
    ax.scatter(yt, yp, s=12, facecolor="white", edgecolor="black", linewidth=0.5)
    lim = [0, max(yt.max(), yp.max()) * 1.04]
    ax.plot(lim, lim, "--", color="black", lw=1.0)
    ax.set_xlabel("Actual Charges (USD)"); ax.set_ylabel("Predicted Charges (USD)")
    ax.set_title(f"{title}\n(R\u00b2 = {R['metrics_table']['Multiple Linear Regression' if key=='mlr' else 'Polynomial Regression (deg 2)']['R2']:.4f})", fontsize=8.8)
fig.tight_layout()
save(fig, "fig6_2_actual_vs_predicted.png")
```

---

## 7. Fig. 6.3 - Residual Plots for the Linear Baseline and the Best Model

**File:** `figures/fig6_3_residuals.png`  
**Appears in:** Section 6.3, Chapter 6 (Results and Performance Evaluation)

### What the figure plots

Two scatter plots of residual (actual minus predicted, y-axis) against predicted charge (x-axis), one per model, with a dashed horizontal line at zero.

### How to read it, and what it establishes

Residual plots test an assumption that a single R2 value cannot: constant error variance. For the linear model the residuals are strongly patterned and heteroscedastic - truncated below and fanning out at higher predictions - so ordinary least squares' constant-variance assumption is violated. For the polynomial model the cloud is far more symmetric around zero and the funnel pattern largely disappears, although a few large positive residuals remain at mid-range predictions, corresponding to expensive non-smokers whose cost is driven by variables the dataset does not contain.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 6.3 residuals
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
for ax, key, title in [(axes[0], "mlr", "Multiple Linear Regression"),
                       (axes[1], "poly", "Polynomial Regression (Deg 2)")]:
    yt = np.array(pred_store[key]["y_true"]); yp = np.array(pred_store[key]["y_pred"])
    ax.scatter(yp, yt - yp, s=12, facecolor="white", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", ls="--", lw=1.0)
    ax.set_xlabel("Predicted Charges (USD)"); ax.set_ylabel("Residual (USD)")
    ax.set_title(f"Residual Plot \u2013 {title}", fontsize=8.8)
fig.tight_layout()
save(fig, "fig6_3_residuals.png")
```

---

## 8. Fig. 6.4 - Random Forest Feature Importances (Top Ten Features)

**File:** `figures/fig6_4_feature_importance.png`  
**Appears in:** Section 6.3, Chapter 6 (Results and Performance Evaluation)

### What the figure plots

A horizontal bar chart of the ten largest impurity-based feature importances from the tuned random forest (mean decrease in impurity), sorted so the most important feature is at the bottom.

### How to read it, and what it establishes

The chart converts the model from a black box into an explanation. The BMI x smoker interaction dominates (0.3442), followed by age x smoker (0.2501), the raw smoker flag (0.1877) and age (0.1036). Together these four account for about 89% of predictive power, while sex (0.0039) and the three region indicators (each below 0.004) are negligible. This is quantitative confirmation of the EDA finding in Fig. 3.3 and of the interaction features engineered in Section 4.3, and it is the evidence behind the claim that cost is driven by smoking and its interaction with BMI and age rather than by geography.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 6.4 importance
imp = pd.Series(R["rf_importance"]).head(10)[::-1]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
bars = ax.barh(range(len(imp)), imp.values, edgecolor="black", linewidth=0.8,
               facecolor=["white" if i % 2 else GREY for i in range(len(imp))])
for b, hatch in zip(bars, ["" if i % 2 else "//" for i in range(len(imp))]):
    b.set_hatch(hatch)
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=8)
ax.set_xlabel("Importance (Mean Decrease in Impurity)")
ax.set_title("Random Forest \u2013 Top 10 Feature Importances")
save(fig, "fig6_4_feature_importance.png")
```

---

## 9. Fig. 6.5 - Prediction Error Analysis: Absolute Error by Smoking Status and Signed Error versus Actual Charge

**File:** `figures/fig6_5_error_analysis.png`  
**Appears in:** Section 6.3, Chapter 6 (Results and Performance Evaluation)

### What the figure plots

Two panels for the best model. Left: a box plot of absolute error (USD) split by smoking status. Right: a scatter of signed error (predicted minus actual) against actual charge, with a dashed zero line.

### How to read it, and what it establishes

The figure answers the fairness question that aggregate metrics hide. In absolute terms the model is not biased against either group - mean absolute error is USD 2,364.47 for non-smokers and USD 2,315.48 for smokers - but the smoker distribution has a longer upper tail of large misses. The signed-error panel shows why: errors cluster tightly around zero for expensive individuals (who are mostly smokers) and fan out downwards for cheap individuals, meaning the model over-predicts a subset of low-cost non-smokers. That asymmetry is the main caveat discussed in Sections 6.4 and 7.2.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig 6.5 combined error analysis
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
err_df = pd.DataFrame({"abs_err": np.abs(np.array(pred_store["poly"]["y_true"]) -
                                        np.array(pred_store["poly"]["y_pred"])),
                       "smoker": pred_store["poly"]["smoker"]})
data = [err_df.loc[err_df.smoker == "no", "abs_err"], err_df.loc[err_df.smoker == "yes", "abs_err"]]
bp = axes[0].boxplot(data, tick_labels=["Non-smoker", "Smoker"], patch_artist=True,
                     medianprops=dict(color="black", lw=1.3), widths=0.55)
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor("white" if i else GREY); b.set_hatch("//" if i else "")
    b.set_edgecolor("black")
axes[0].set_ylabel("Absolute Error (USD)")
axes[0].set_title("Absolute Error by Smoking Status", fontsize=9)

yt = np.array(pred_store["poly"]["y_true"]); yp = np.array(pred_store["poly"]["y_pred"])
axes[1].scatter(yt, yp - yt, s=12, facecolor="white", edgecolor="black", linewidth=0.5)
axes[1].axhline(0, color="black", ls="--", lw=1.0)
axes[1].set_xlabel("Actual Charges (USD)"); axes[1].set_ylabel("Signed Error (USD)")
axes[1].set_title("Signed Error versus Actual Charge", fontsize=9)
fig.tight_layout()
save(fig, "fig6_5_error_analysis.png")
```

---

## 10. Fig. B.1 - Random Forest Test R2 versus Training Set Size

**File:** `figures/figB1_learning_curve.png`  
**Appears in:** Section B.2, Appendix B (Additional Results)

### What the figure plots

A line plot with markers. The x-axis is the number of training records used to refit the tuned random forest (171, 342, 513, 684, 855) and the y-axis is the resulting test R2.

### How to read it, and what it establishes

This tests whether the study is limited by data volume or by data content. Test R2 rises steeply between 171 and 513 records and then flattens, gaining only about 0.5 percentage points over the last 342 records. Performance is therefore close to saturation: collecting more rows with the same six attributes would buy little, whereas adding genuinely new informative attributes (claims history, diagnosis codes, plan type) would not face the same ceiling. That is the empirical basis for the first future-work recommendation in Section 7.3.

### Code that generates it

```python
# ---------------------------------------------------------------- Fig B.1 learning curve
lc = R["learning_curve"]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(lc["train_size"], lc["test_r2"], "o-", color="black", lw=1.2, ms=5)
ax.set_xlabel("Number of Training Records"); ax.set_ylabel("Test R\u00b2")
ax.set_title("Random Forest \u2013 Test R\u00b2 vs Training Set Size")
save(fig, "figB1_learning_curve.png")
```

---

## Shared boilerplate (top of `src/make_figures_refstyle.py`)

All figures share one plotting setup - a monochrome, serif, hatched house style that matches the report.
This is the header that every figure block above relies on:

```python
"""Regenerate all report figures in the reference report's house style:
monochrome, hatched bars, bold serif titles, dashed grey gridlines, black axes.
Numbers are read from results.json so the figures match the report text exactly.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "src", "results.json")))
df = pd.read_csv(os.path.join(ROOT, "data", "insurance.csv")).drop_duplicates().reset_index(drop=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#b0b0b0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
GREY = "#8c8c8c"
DARK = "#3f3f3f"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)
```
