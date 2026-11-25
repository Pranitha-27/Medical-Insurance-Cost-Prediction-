# Viva Questions and Answers

**Case Study 14 — Medical Insurance Cost Prediction Using Machine Learning Regression Models**

Forty-two questions grouped from basic to advanced, each answered in the terms used in the report. The
**numbers are the project's own** (test set, n = 268, `random_state = 42`), so they can be quoted directly.
Questions 40–41 cover the Flask web app (`webapp/`).

---

## A. Problem, data and basic concepts

**1. What is the problem statement?**
Predict the annual medical insurance charge (in USD) of a policyholder from six attributes — age, sex, BMI,
number of children, smoking status and residential region — and compare regression models to find the best
one.

**2. Is this classification or regression? Why?**
Regression. The target `charges` is a continuous quantity (USD 1,121.87 to 63,770.43), so we minimise a
squared-error loss and report R², MAE, RMSE and MAPE rather than accuracy, precision or F1.

**3. Which dataset did you use?**
The **Medical Cost Personal Dataset** (Kaggle `mirichoi0218/insurance`), originally distributed with
*Machine Learning with R* by Brett Lantz. 1,338 rows × 7 columns, US policyholders, publicly available for
research and teaching. Access date **22 November 2026**.

**4. What are the features and the target?**
Features: `age` (18–64), `sex` (female/male), `bmi` (15.96–53.13), `children` (0–5), `smoker` (yes/no),
`region` (northeast/northwest/southeast/southwest). Target: `charges` in USD. There are 3 categorical and
3 numeric features.

**5. How many rows after cleaning, and what did you clean?**
1,338 → **1,337**. There were no missing values; exactly one duplicate row was removed. Outliers were
**deliberately kept** — 139 by the IQR rule on charges plus 9 by the BMI rule — because they are genuine
high-cost individuals, and deleting them would make the model systematically under-predict expensive cases.

**6. Why did you not normalise or remove the outliers?**
Two reasons. First, they are real observations, not measurement errors, so removing them biases the target.
Second, tree-based models are insensitive to monotone transformations and to the magnitude of outliers,
and the linear models were left unscaled on purpose so their coefficients stay interpretable in USD.

**7. What is the distribution of the target?**
Right-skewed and multimodal: mean USD 13,279.12, median 9,386.16, standard deviation 12,110.36, skewness
**1.515**. The modes correspond to smoking status (see Q11), which is why a single straight line cannot fit
the data.

---

## B. Exploratory data analysis

**8. Which feature is most associated with charges?**
Smoking. Mean charge is **USD 8,440.66 for non-smokers vs USD 32,050.23 for smokers** — a factor of about
3.8. It is the single strongest discriminator in the dataset.

**9. What did you find about BMI?**
On its own it is a weak linear correlate of charges (r = 0.1984), but its effect is **conditional on smoking**
— an interaction. Mean charges by BMI class: non-smokers 5,485 / 7,735 / 8,226 / 8,866; smokers 18,810 /
19,942 / 22,491 / **41,693** USD. Going from normal to obese BMI adds only **USD 1,131.51** for a
non-smoker but **USD 21,750.59** for a smoker.

**10. What about sex and region?**
Both are weak. Mean charges are USD 13,975.00 (male) vs 12,569.58 (female), and by region 13,406.38
(northeast), 12,450.84 (northwest), 14,735.41 (southeast), 12,346.94 (southwest). Random-forest importance
confirms it: sex 0.0039 and the three region dummies each below 0.004.

**11. Why does the histogram show three bumps?**
Because the population is really several sub-populations: cheap non-smokers, mid-range smokers and a small
group of very high-cost obese smokers. The target histogram is therefore a **mixture distribution**, and
mixtures are exactly where additive linear models fail.

**12. Which figure was most useful in the EDA, and why?**
Fig. 3.3, the scatter of age and BMI against charges separated by smoking status. It shows non-smokers as a
slowly rising band, smokers in two higher bands, and a steep BMI effect only among smokers — the visual proof
that an interaction term is needed.

---

## C. Preprocessing and feature engineering

**13. How did you encode the categorical variables?**
Binary map for `sex` (male = 1) and `smoker` (yes = 1); one-hot encoding with `drop_first=True` for
`region`, giving 3 dummies instead of 4 to avoid the **dummy-variable trap** (perfect multicollinearity with
the intercept).

**14. Which features did you engineer and why?**
Three: `bmi_smoker_interaction = bmi × smoker`, `age_smoker_interaction = age × smoker`, and
`is_obese = 1 if bmi ≥ 30`. They encode the interactions seen in Fig. 3.3 so that even the linear and
tree models can represent them. Final feature count: 11.

**15. What is the effect of the engineered features?**
They are the top two features of the random forest: `bmi × smoker` 0.3442 and `age × smoker` 0.2501.
Together with the raw `smoker` flag (0.1877) and `age` (0.1036) they account for roughly **89 %** of the
model's predictive power.

**16. Why scale only the polynomial model?**
`StandardScaler` is necessary there because polynomial terms have wildly different units (age² vs bmi²), and
Ridge's single α penalises all coefficients equally. Trees split on thresholds and are scale-invariant, and
multiple linear regression was left unscaled so a coefficient reads as "USD per year of age".

**17. How did you prevent data leakage?**
The duplicate was removed **before** splitting; the scaler and polynomial expansion are inside a `Pipeline`
so they are fitted only on training folds; `GridSearchCV` runs on the training subset only; the test set was
touched once, for the final numbers; and the model that was evaluated on test was refitted on
train + validation so the validation rows are not wasted.

**18. What is the split ratio and why?**
**64 % / 16 % / 20 % = 855 / 214 / 268** rows, `random_state = 42`. Training fits the models, validation
scores candidate configurations (through 5-fold CV), and the 20 % test set gives an unbiased final estimate —
the standard way to avoid optimistically reporting a tuned model's score.

---

## D. Models

**19. Which four models did you compare?**
Multiple Linear Regression (OLS baseline), Polynomial Regression of degree 2 with `StandardScaler` and
Ridge (α = 1.0), Decision Tree Regression (tuned), and Random Forest Regression (tuned).

**20. How many terms does the polynomial model have?**
11 original features expand to **77** degree-2 terms (11 linear + 66 interactions/squares), which is why the
L2 penalty is needed.

**21. Why Ridge and not plain OLS on the polynomial features?**
With 77 correlated terms and only 855 training rows, OLS overfits and its coefficients become unstable.
Ridge adds α·Σβ² to the loss, shrinking and stabilising them; α = 1.0 was small enough to keep the curvature
and large enough to control variance.

**22. How did you tune the tree models?**
Exhaustive `GridSearchCV` with 5-fold `KFold` on the training set, scoring `neg_root_mean_squared_error`.
Search space: depth {3,4,5,6,8,10,12,None}, leaf {1,2,3,5,10,20}, split {2,5,10,20} for the tree; and
`n_estimators` {200,400}, depth {6,10,15,None}, `min_samples_leaf` {1,2,3,5}, `max_features` {sqrt, 1.0}
for the forest.

**23. What were the best hyper-parameters?**
Decision tree: **max_depth = 5, min_samples_leaf = 5, min_samples_split = 20** → 21 leaves.
Random forest: **200 trees, max_depth = 10, max_features = "sqrt", min_samples_leaf = 3**.

**24. Why do the trees beat the linear baseline?**
Because they can split on `smoker` first and then model a different relationship inside each branch — an
interaction, found automatically. A linear model assigns one coefficient to `bmi` for everybody, which is
exactly the assumption Fig. 3.3 contradicts.

**25. Why does polynomial regression still edge out the random forest?**
The true structure is smooth and multiplicative (charge ≈ base + f(age) + g(bmi)·smoker). A degree-2
expansion with 77 terms represents that almost exactly, whereas a forest approximates it with a
**piecewise-constant** function — a step function cannot follow a straight line inside a leaf. With only
1,337 rows the forest also cannot average away enough variance.

---

## E. Results and evaluation metrics

**26. State your final results.**
Test set, n = 268:

| Model | R² | MAE | MSE | RMSE | MAPE |
|---|---|---|---|---|---|
| Multiple Linear Regression | 0.8069 | 4,177.05 | 35,478,020.68 | 5,956.34 | 41.40 % |
| **Polynomial (deg 2, Ridge)** | **0.9031** | **2,353.50** | 17,798,941.23 | **4,218.88** | **30.08 %** |
| Decision Tree | 0.8943 | 2,649.80 | 19,429,611.74 | 4,407.90 | 33.50 % |
| Random Forest | 0.8962 | 2,435.54 | 19,068,226.55 | 4,366.72 | 35.29 % |

Best model: **degree-2 polynomial regression** — 29.2 % lower RMSE than the linear baseline.

(Note the small trade-off: the random forest has a slightly worse MAPE than the polynomial model on this
split; MAPE is dominated by cheap non-smokers, where absolute errors are small but relative errors are
large. The ranking by R², MAE and RMSE — the metrics that matter for a USD cost target — is consistent.)

**27. What does R² = 0.9031 mean?**
The model explains **90.31 %** of the variance in test-set charges; equivalently, it reduces squared error
by 90.31 % relative to always predicting the mean charge (USD 14,272.01 on the test set).

**28. Difference between MAE, MSE and RMSE?**
MAE is the average absolute error in USD — robust and easy to explain. MSE squares the errors, so it
punishes large misses heavily, and its units are USD². RMSE is the square root of MSE, back in USD, so it
reads like a typical error but still penalises big misses. The report quotes all three plus MAPE (% error),
which is scale-free but unstable when actual values are small.

**29. Why is MAPE as high as 30 %?**
Because of the low-cost sub-population: a miss of USD 1,600 on a USD 4,463 charge is 36 % relative error but
trivial in absolute terms. That is why we do not select the model on MAPE alone.

**30. Did cross-validation agree with the test ranking?**
Yes. Five-fold CV R²: MLR **0.7390**, Polynomial **0.8468**; CV RMSE: Decision Tree 4,753.24, Random Forest
4,570.16. The ranking (linear worst, polynomial best) is stable, so the test result is not a lucky split.

**31. How do you know the model is not just overfitting?**
Three checks: (i) test metrics are close to CV metrics; (ii) the polynomial model was regularised and still
generalised; (iii) the learning curve (Fig. B.1) rises then plateaus — test R² goes 0.8814 → 0.8932 →
0.8951 → 0.8955 → 0.8962 as training rows go 171 → 855, so more of the same data adds almost nothing, which
means the model is close to the achievable ceiling rather than starving for data.

**32. What kind of errors does the model still make?**
Mean absolute error is almost identical by smoking status (USD 2,364.47 for non-smokers vs 2,315.48 for
smokers), so there is no systematic group bias, but the smoker group has a longer upper tail of large misses.
The signed-error panel of Fig. 6.5 shows the model over-predicting a subset of **cheap non-smokers** and
tracking expensive cases closely.

**33. Give an example prediction.**
A 45-year-old male smoker, BMI 34.8, 2 children, northeast region → **USD 43,090.16** predicted.
A 24-year-old female non-smoker, BMI 22.5, no children, southwest → **USD 3,317.16**.
A 60-year-old male non-smoker, BMI 28.1, 3 children, southeast → **USD 15,314.56**.

**34. Interpret the linear regression coefficients.**
Baseline MLR: age +USD 248.21 per year, BMI +318.70 per unit, children +533.01 each, male −101.54,
**smoker +23,077.76**, northwest −391.76, southeast −838.92, southwest −659.14, intercept −11,092.65
(all relative to the reference profile: female, non-smoker, northeast). The smoker coefficient is 50×
larger than any other — the same conclusion as the EDA, now in dollars.

---

## F. Critical thinking, limitations and extensions

**35. What are the main limitations of your study?**
(a) Only six attributes — no diagnosis codes, prior claims, income or plan type, so part of the residual
RMSE (~USD 4,219) is irreducible with this data. (b) A single random split, no out-of-time validation.
(c) US-derived data, so it does not transfer to the Indian market without re-validation. (d) Point estimates
only, no prediction intervals. (e) No formal fairness audit beyond the smoking subgroup.

**36. Can you deploy this model to price insurance policies?**
**No.** It is an academic benchmark on anonymised historical data, trained on a US sample with no regulatory
review. Pricing or underwriting decisions would need actuarial validation, fairness testing on protected
attributes, prediction intervals and a governed model lifecycle. It is suitable for education and for
illustrating the drivers of cost.

**37. What would you do to improve accuracy?**
Add genuinely new informative features first (claims history, chronic conditions, plan type), because the
learning curve shows the sample size is already saturated. Then try gradient boosting (XGBoost / LightGBM),
SVR with an RBF kernel, and quantile or Tweedie regression to model the skew and produce intervals. On the
modelling side, target transformation (log or two-part model) and SHAP-based explanation would help.

**38. Why not use a neural network?**
With 1,337 rows and 11 tabular features, a multi-layer network has far more parameters than the data can
constrain and would need heavy regularisation, while offering no representational advantage over a
degree-2 expansion or a forest. Deep models pay off on large, high-cardinality or unstructured data — nothing
here suggests that.

**39. Your target is skewed — why not train on log(charges)?**
It is a reasonable extension: a log target stabilises variance and makes errors multiplicative. It was
not chosen for the main comparison because the study's objective is to compare standard regressors on their
native scale and to report errors in USD, and because log-training biases the back-transformed prediction
downwards without a correction factor. It is listed as future work.

**40. You built a website — how does the model reach the browser?**
`webapp/export_model.py` re-trains the winning pipeline on the same 855 rows and saves it with `joblib` as
`webapp/model/model_bundle.joblib`. The Flask app loads that bundle **once at start-up** and exposes
`POST /api/predict`; the page sends the six inputs as JSON and receives the prediction, a ± 1 RMSE band and a
per-feature contribution breakdown. Nothing is re-trained per request (`webapp/app.py`), so latency is a few
milliseconds and the numbers can never drift from the report. Inputs outside the training ranges
(age 18–64, BMI 15.96–53.13, children 0–5) are rejected with HTTP 400 instead of being extrapolated.

**41. How are the contribution bars in the web app computed — is that SHAP?**
No, and it doesn't need to be. The deployed model is **linear in its 77 polynomial terms**, so
`prediction = intercept + Σ coef_j · z_j`. Each of the 77 terms is attributed to the original feature it came
from (for example every term containing `bmi` or `bmi_smoker_interaction` is grouped under BMI), which makes
the bars an **exact** decomposition: they sum to `prediction − intercept` to within rounding. SHAP would be
needed for a non-linear model such as the random forest, where it can only be sampled approximately.

**42. What did you personally take away from this case study?**
That the EDA decided the modelling. The smoking–BMI interaction visible in a scatter plot is the whole
story: once it was engineered into the features, four very different algorithms all landed within one
percentage point of each other, and the linear baseline without it stayed 10 points behind. Methodologically,
the discipline of splitting before scaling, tuning only inside the training folds, and reporting CV alongside
a single held-out test score is what makes the R² = 0.9031 believable.

---

## Quick number sheet (memorise these)

| Item | Value |
|---|---|
| Rows raw → clean | 1,338 → 1,337 |
| Train / val / test | 855 / 214 / 268 |
| Features → after engineering | 11 |
| Polynomial terms | 77 |
| Best model | Polynomial deg 2 + StandardScaler + Ridge(α = 1.0) |
| Best test R² / RMSE / MAE / MAPE | 0.9031 / 4,218.88 / 2,353.50 / 30.08 % |
| Linear baseline R² / RMSE | 0.8069 / 5,956.34 |
| CV R² (MLR / Poly) | 0.7390 / 0.8468 |
| Mean charge, non-smoker vs smoker | 8,440.66 vs 32,050.23 |
| Top RF features | bmi×smoker 0.3442, age×smoker 0.2501, smoker 0.1877 |
| Skewness of charges | 1.515 |
