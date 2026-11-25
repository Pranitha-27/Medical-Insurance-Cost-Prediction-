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

# ---------------------------------------------------------------- Fig B.1 learning curve
lc = R["learning_curve"]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(lc["train_size"], lc["test_r2"], "o-", color="black", lw=1.2, ms=5)
ax.set_xlabel("Number of Training Records"); ax.set_ylabel("Test R\u00b2")
ax.set_title("Random Forest \u2013 Test R\u00b2 vs Training Set Size")
save(fig, "figB1_learning_curve.png")
