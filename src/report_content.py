"""
Report content — restructured to mirror the format of the reference report
("ML Case Report - Yash"), carrying the complete medical insurance project content.

House style of the reference, reproduced here:
  * front matter numbered with Arabic numerals (cover = 1)
  * Chapter headings as "CHAPTER n - TITLE", sections numbered 1.1 / 1.2 ...
  * literature survey presented as one wide table (Table 2.1)
  * EDA as bullet observations + figures, each followed by an "Interpretation:" line
  * algorithms as 5.2.1 / 5.2.2 / 5.2.3 in compact prose
  * References as a numbered list
  * Appendix A = repository link, Appendix B = additional results
"""

import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root
R = json.load(open(os.path.join(HERE, "results.json")))
FIG = os.path.join(ROOT, "figures")

TITLE = "MEDICAL INSURANCE COST PREDICTION USING MACHINE LEARNING"
COLLEGE = "BMS INSTITUTE OF TECHNOLOGY AND MANAGEMENT"
COLLEGE_SUB = "(An Autonomous Institution, Affiliated to VTU, Belagavi)"
COLLEGE_ADDR = "Avalahalli, Doddaballapura Main Road, Bengaluru \u2013 560119"
DEPT = "DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING"
SEM = "V Semester"
YEAR = "2026\u20132027"
GUIDE = "Dr Mayuri K P"
GUIDE_DESIG = "Assistant Professor"
HOD = "Dr Anupama H S"
PRINCIPAL = "Dr Sanjay H A"
STUDENTS = [("BYREDDY PRANITHA", "1TD24AI052"),
            ("EDDULA SAMYUKTHA", "1TD24AI065"),
            ("VARSHA N", "1TD24AI183"),
            ("Y P ROHITH", "1TD24AI191")]
ACCESS_DATE = "22 November 2026"
REPO_URL = "https://github.com/Pranitha-27/Medical-Insurance-Cost-Prediction-.git"

M = R["metrics_table"]
MLR = M["Multiple Linear Regression"]
POL = M["Polynomial Regression (deg 2)"]
DT = M["Decision Tree Regression"]
RF = M["Random Forest Regression"]
IMP = R["rf_importance"]
DESC = R["describe"]
IQR = R["iqr"]
SMOKE = R["mean_charges_by_smoker"]
BMISM = R["bmi_smoker_means"]
LIN = R["linear_coefs"]
SP = R["sample_predictions"]
NEW = R["new_customer_predictions"]
SPLIT = R["split"]
LC = R["learning_curve"]

# ---------------------------------------------------------------- block helpers
def H1(t): return ("h1", t)
def H2(t): return ("h2", t)
def H3(t): return ("h3", t)
def P(t):  return ("p", t)
def BL(items): return ("bl", items)
def FIGURE(path, cap, interp=None): return ("fig", path, cap, interp)
def TBL(cap, headers, rows, widths=None, align=None, fs=8.5):
    return ("tbl", cap, headers, rows, widths, align, fs)
def CODE(t): return ("code", t)
def PB(): return ("pb",)

A = []
def add(*b): A.extend(b)

# ================================================================ CHAPTER 1
add(
H1("CHAPTER 1 - INTRODUCTION"),
H2("1.1 Background"),
P("Medical insurance is the financial arrangement through which an individual transfers the risk of "
  "unexpected medical expenditure to an insurer. The insurer collects a periodic premium that must be "
  "large enough to cover the expected claim cost of each policyholder and small enough to remain "
  "affordable and competitive. The central actuarial problem is therefore one of individual-level cost "
  "estimation: given a small set of observable characteristics such as age, body mass index, smoking "
  "behaviour, family size and residential region, what annual medical expenditure should be expected "
  "for this person?"),
P("Historically this estimation has been performed with generalised linear models, most commonly the "
  "Tweedie and gamma-log families, because medical expenditure is non-negative, strongly right-skewed "
  "and contains a large mass of low-cost observations. Such models are transparent and remain in "
  "production pricing, but they impose a strictly additive, pre-specified functional form. They cannot, "
  "by construction, discover that the effect of body mass index on cost is very different for a smoker "
  "than for a non-smoker, nor that the relationship between age and cost is not a single straight line. "
  "Machine Learning (ML) offers a data-driven alternative: regression models can learn non-linear "
  "relationships and interaction effects directly from historical claims data without the analyst "
  "having to enumerate them in advance. This case study examines that alternative on a public benchmark "
  "dataset and compares four regression algorithms under one controlled protocol."),
P("The practical importance of accurate individual cost prediction extends well beyond premium "
  "setting. Insurers use predicted cost to segment risk, size reserves and detect adverse selection; "
  "hospitals and third-party administrators use it for cost containment and for identifying high-risk "
  "populations that would benefit from preventive intervention; and governments use it to project "
  "subsidy requirements in publicly financed schemes. Where out-of-pocket health expenditure forms a "
  "significant share of total health spending, the ability to anticipate cost from a handful of easily "
  "collected attributes is directly relevant to policy design and to individual financial planning."),
H2("1.2 Problem Statement"),
P("Given a dataset of individual policyholders described by age, gender, body mass index (BMI), number "
  "of dependent children, smoking status and residential region, along with the annual medical "
  "insurance charges billed to each individual in US dollars, design and evaluate machine learning "
  "regression models that predict the annual charges for an unseen individual. The models must quantify "
  "the relationships between the attributes and insurance cost, must be compared on a leakage-free "
  "held-out test set using appropriate regression metrics, and the most suitable model must be "
  "identified with a clear statement of its accuracy, its remaining error and its limitations."),
H2("1.3 Objectives"),
BL([
 "To collect and document the Medical Cost Personal Dataset, including its source, licence, attribute "
 "semantics and known limitations.",
 "To perform Exploratory Data Analysis (EDA) to analyse the distribution of charges and to quantify how "
 "age, BMI, smoking status, sex, family size and region relate to annual cost.",
 "To design a preprocessing and feature-engineering pipeline that handles duplicates and outliers, "
 "encodes categorical attributes and represents the interaction structure discovered during EDA "
 "without introducing data leakage.",
 "To implement four regression algorithms — Multiple Linear Regression, Polynomial Regression, Decision "
 "Tree Regression and Random Forest Regression — and to tune the tree-based models systematically.",
 "To evaluate and compare the models using R\u00b2, MAE, MSE, RMSE and MAPE, supported by residual "
 "analysis, subgroup error analysis and cross-validation.",
 "To identify the limitations of the study and to propose realistic enhancements for future work.",
]),
H2("1.4 Scope of the Work"),
P("This case study covers one publicly available, anonymised, cross-sectional dataset of 1,337 records "
  "after duplicate removal, six predictor attributes and one continuous target. It covers data "
  "documentation, exploratory analysis, cleaning, encoding, feature engineering, a leakage-free "
  "train\u2013validation\u2013test protocol, training and grid-search tuning of four regression models, "
  "quantitative and visual evaluation, subgroup error analysis and a prediction demonstration on both "
  "held-out records and newly constructed hypothetical applicants. It is built as an academic "
  "prototype and a reproducible script."),
P("The work explicitly excludes clinical, actuarial or financial deployment of any kind; the use of "
  "personal or identifiable health data; underwriting decision-making; causal inference about the "
  "effect of smoking or obesity on cost; and any claim that the reported model is suitable for real "
  "premium setting. The dataset originates from United States policyholders, so conclusions about the "
  "Indian population cannot be drawn. Deep learning, distributional regression and survival-type cost "
  "models are not implemented, though they are noted in Section 7.3."),
H2("1.5 Organization of the Report"),
P("Chapter 2 reviews five peer-reviewed studies on medical cost prediction using regression and "
  "identifies the research gap addressed here. Chapter 3 describes the dataset, its attributes, the "
  "exploratory analysis and the observations that shaped the modelling. Chapter 4 details data "
  "cleaning, encoding, feature engineering and the train\u2013validation\u2013test split. Chapter 5 "
  "presents the end-to-end workflow, the working principle of the four algorithms and the "
  "hyperparameter settings. Chapter 6 reports experimental results, visual diagnostics, model "
  "comparison and sample predictions. Chapter 7 concludes with the key findings, limitations and "
  "future enhancements. References and two appendices follow."),
)

# ================================================================ CHAPTER 2
add(
PB(),
H1("CHAPTER 2 - LITERATURE SURVEY"),
P("A review of peer-reviewed research on medical insurance cost prediction using regression, ensemble "
  "learning and explainable machine learning is summarised in Table 2.1. The selection prioritises "
  "studies that report quantitative regression results on the same or a closely comparable dataset, so "
  "that the findings of this case study can be placed in context."),
TBL("Table 2.1: Summary of Literature Survey on Medical Insurance Cost Prediction",
 ["Sl. No.", "Author(s) and Year", "Title / Source", "Method Used", "Key Findings / Limitations"],
 [
  ["1", "K. Kaushik, A. Bhardwaj, A. D. Dwivedi, R. Singh, 2022",
   "Machine Learning-Based Regression Framework to Predict Health Insurance Premiums (IJERPH, 19(13):7898)",
   "Artificial neural network regression compared with conventional regression; RMSE, MSE, MAE, R\u00b2 "
   "and adjusted R\u00b2",
   "The ANN captured non-linearity and outperformed the linear baseline on the same six attributes; "
   "the model is opaque and the small dataset risks overfitting, with no feature-level explanation."],
  ["2", "S. Panda, B. Purkayastha, D. Das, M. Chakraborty, S. K. Biswas, 2022",
   "Health Insurance Cost Prediction Using Regression Models (IEEE COM-IT-CON, vol. 1, pp. 168\u2013173)",
   "Linear regression, decision tree, random forest and gradient boosting on the Medical Cost Personal "
   "Dataset",
   "Ensemble and tree methods beat plain linear regression, and smoking status dominated importance; "
   "no subgroup error analysis and limited discussion of outliers."],
  ["3", "J. Ruth Sandra, S. Joshi, A. Ravi, A. Kodipalli, T. Rao, S. Kamal, 2024",
   "Prediction of Cost for Medical Care Insurance by Using Regression Models (Springer ERCICA 2023, "
   "pp. 289\u2013300)",
   "Comparative study of traditional regression models on age, smoking, BMI, gender and region",
   "Confirms smoking and BMI as the principal drivers and reports that simple models remain competitive "
   "on tabular data of this size; a single train\u2013test split makes results seed-sensitive."],
  ["4", "U. Orji, E. Ukwandu, 2024",
   "Machine Learning for an Explainable Cost Prediction of Medical Insurance (Machine Learning with "
   "Applications, 15:100516)",
   "Multiple regressors with SHAP-based explainability on insurance cost data",
   "The BMI\u2013smoker interaction is the dominant driver of predicted cost; accuracy without "
   "interpretability is of limited use to insurers. Hyperparameter tuning was limited."],
  ["5", "M. Hanafy, O. M. A. Mahmoud, 2021",
   "Predict Health Insurance Cost by Using Machine Learning and DNN Regression Models (IJITEE, "
   "10(3):137\u2013143)",
   "Linear, polynomial, decision tree, random forest and deep neural network regression",
   "Non-linear and ensemble models substantially improve over linear regression on the same dataset, and "
   "polynomial terms help because the smoker cost band is non-linear; no cross-validation and no "
   "discussion of preprocessing leakage."],
 ],
 widths=[0.42, 1.05, 1.5, 1.3, 1.65], fs=8.0),
P("Three consistent conclusions emerge from the literature. First, on this dataset linear regression "
  "consistently underfits: reported R\u00b2 values for ordinary least squares cluster around "
  "0.75\u20130.78, well below the 0.85\u20130.90 obtained by tree ensembles and polynomial expansions. "
  "Second, smoking status and its interaction with BMI account for the majority of explainable "
  "variation in charges, a result reproduced by feature-importance and SHAP analyses alike. Third, the "
  "methodological quality of the reported comparisons varies considerably: several studies report a "
  "single train\u2013test split, do not state whether scaling was fitted on the training data only, and "
  "omit subgroup or residual analysis."),
H2("2.1 Research Gap / Motivation"),
P("The reviewed work establishes that non-linear models win on this dataset, but it leaves three "
  "practical gaps that this case study addresses without claiming novelty of algorithm."),
BL([
 "Comparison on an identical, leakage-free protocol. Most studies compare models trained on differently "
 "prepared feature sets with default hyperparameters. Here all four models share one cleaning pipeline, "
 "one split (random_state = 42) and one set of engineered features, and every scaler and polynomial "
 "expansion is fitted inside a pipeline on training folds only, so the comparison isolates the "
 "algorithm rather than the preparation.",
 "Systematic tuning of the tree-based models. Decision tree and random forest performance depends "
 "strongly on depth and leaf-size constraints on a 1,337-row dataset. Both are tuned by five-fold grid "
 "search and the selected configuration is reported rather than a default one.",
 "Error analysis beyond a single metric. Aggregate R\u00b2 hides the fact that absolute and relative "
 "errors differ greatly across the cost range. This study reports RMSE, MAE and MAPE together, inspects "
 "residuals against predicted value, and compares absolute error for smokers and non-smokers to test "
 "whether the model is equally reliable across the two dominant sub-populations.",
]),
)

# ================================================================ CHAPTER 3
add(
PB(),
H1("CHAPTER 3 - DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS"),
H2("3.1 Dataset Source"),
P("The dataset used is the Medical Cost Personal Dataset, published on Kaggle by Miri Choi and derived "
  "from the insurance.csv file distributed with the book Machine Learning with R by Brett Lantz. It is a "
  "widely used teaching benchmark for regression and contains no personally identifiable information. "
  "Records were accessed for non-commercial academic research and educational evaluation."),
BL([
 "Dataset name: Medical Cost Personal Dataset (insurance.csv).",
 "Provider: Kaggle, uploaded by Miri Choi; original file from the Machine Learning with R repository.",
 "URL: https://www.kaggle.com/datasets/mirichoi0218/insurance",
 "Date accessed: " + ACCESS_DATE + ". A byte-identical mirror "
 "(https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv) was "
 "used for reproducibility.",
 "Licence / usage: publicly available for educational and research use under the Kaggle dataset terms; "
 "the data are anonymised and non-identifiable, so no ethical clearance was required.",
 "File format: comma-separated values (CSV), 55 KB, UTF-8, one header row.",
]),
H2("3.2 Dataset Overview"),
TBL("Table 3.1: Dataset Overview for the Medical Cost Personal Dataset",
 ["Property", "Details"],
 [
  ["Dataset name", "Medical Cost Personal Dataset (insurance.csv)"],
  ["Source / URL", "Kaggle \u2013 https://www.kaggle.com/datasets/mirichoi0218/insurance"],
  ["Number of records", "1,338 raw rows; 1,337 after removing 1 exact duplicate"],
  ["Number of features", "6 predictor attributes (3 numerical, 3 categorical) + 1 target"],
  ["Target variable", "charges \u2013 annual medical insurance cost in USD (continuous)"],
  ["Task type", "Supervised regression (univariate, cross-sectional)"],
  ["File format", "CSV"],
  ["Missing values", "None (0 missing cells in 1,338 \u00d7 7)"],
  ["Target range", "USD 1,121.87 \u2013 63,770.43; median USD 9,386.16"],
 ],
 widths=[1.5, 5.0], fs=8.5),
H2("3.3 Attribute Description"),
P("Table 3.2 describes each attribute as documented by the provider and as used in this study. Only six "
  "attributes are available as inputs; there is no identifier column, no date field and no clinical "
  "variable such as a diagnosis or procedure code, which is an important limitation discussed in "
  "Section 7.2."),
TBL("Table 3.2: Attribute Description of the Medical Cost Personal Dataset",
 ["Sl. No.", "Attribute / Feature", "Data Type", "Description", "Role"],
 [
  ["1", "age", "Numeric (Integer, 18\u201364)", "Age of the primary beneficiary in years", "Input"],
  ["2", "sex", "Categorical (Binary)", "Gender of the policyholder: female / male", "Input"],
  ["3", "bmi", "Numeric (Float, 15.96\u201353.13)", "Body mass index in kg/m\u00b2; \u2265 30 is "
   "clinically obese", "Input"],
  ["4", "children", "Numeric (Integer, 0\u20135)", "Number of dependants covered by the policy", "Input"],
  ["5", "smoker", "Categorical (Binary)", "Whether the beneficiary smokes: yes / no", "Input"],
  ["6", "region", "Categorical (4 levels)", "Residential area in the US: northeast, northwest, "
   "southeast, southwest", "Input"],
  ["7", "charges", "Numeric (Continuous)", "Annual medical insurance charges billed, USD", "Target"],
 ],
 widths=[0.55, 0.95, 1.45, 2.6, 0.75], fs=8.5),
H2("3.4 Exploratory Data Analysis"),
P("Exploratory analysis was conducted in Python using pandas, matplotlib and seaborn, and concentrated "
  "on four questions: the shape and spread of the target, the marginal association of each attribute "
  "with cost, the group differences between categorical segments, and whether the attributes interact."),
BL([
 "Dataset shape, types and data quality: the raw file contains 1,338 rows and 7 columns with no "
 "missing values in any column. Exactly one fully duplicated row was present and was removed. Data "
 "types are consistent with the documented semantics: three integer or float predictors, one float "
 "target and three object columns.",
 "Target distribution (Fig. 3.1): the mean charge (USD " + f"{DESC['charges']['mean']:,.2f}" + ") is "
 "well above the median (USD " + f"{DESC['charges']['50%']:,.2f}" + "), giving a right skew of "
 f"{IQR['charges']['skew']:.2f}" + ", and the standard deviation (USD "
 f"{DESC['charges']['std']:,.2f}" + ") is almost as large as the mean. The 1.5 \u00d7 IQR rule flags "
 f"{IQR['charges']['n_outliers']}" + " high-cost records, which are genuine expensive claims rather "
 "than measurement errors.",
 "Numerical predictor distributions: age is almost uniform between 18 and 64 with a mild "
 "spike at 18\u201319; BMI is approximately normal and centred at "
 f"{DESC['bmi']['mean']:.2f}" + ", slightly above the clinical obesity threshold of 30, so a large "
 "share of the sample is obese; the number of children is a small-count variable dominated by 0 and 1.",
 "Attribute\u2013target relationships (Fig. 3.3): charge levels separate almost completely by "
 "smoking status, and the marginal Pearson correlations with cost are weak \u2014 age r = "
 f"{R['corr_with_target']['age']:.3f}" + ", BMI r = " + f"{R['corr_with_target']['bmi']:.3f}" +
 ", children r = " + f"{R['corr_with_target']['children']:.3f}" + " \u2014 which warns that a "
 "univariate view is misleading.",
 "Group differences (Fig. 3.2): mean charges are USD "
 f"{SMOKE['no']:,.2f}" + " for non-smokers against USD " + f"{SMOKE['yes']:,.2f}" + " for smokers, a "
 "factor of about 3.8, whereas sex and region show only mild differences: males average USD "
 f"{R['mean_charges_by_sex']['male']:,.2f}" + " against USD "
 f"{R['mean_charges_by_sex']['female']:,.2f}" + " for females, and the southeast is the most expensive "
 "region at USD " + f"{R['mean_charges_by_region']['southeast']:,.2f}" + " against USD "
 f"{R['mean_charges_by_region']['southwest']:,.2f}" + " in the southwest.",
 "Interaction structure (Table 3.3): the effect of BMI on cost is conditional on smoking. Among "
 "non-smokers, moving from normal to obese BMI raises mean charges by only USD "
 f"{BMISM['no|Obese'] - BMISM['no|Normal']:,.2f}" + ", whereas among smokers the same move raises mean "
 "charges by USD " + f"{BMISM['yes|Obese'] - BMISM['yes|Normal']:,.2f}" + ", roughly nineteen times more.",
]),
FIGURE(os.path.join(FIG, "fig3_1_charges_distribution.png"),
       "Fig. 3.1: Distribution of Annual Medical Insurance Charges",
       "The histogram shows a dense low-cost mode below USD 15,000, a middle band near USD "
       "20,000\u201335,000 and a small high-cost cluster above USD 40,000. This multimodality, together "
       "with the long right tail, is the first indication that a single linear function of the inputs "
       "will be inadequate."),
FIGURE(os.path.join(FIG, "fig3_2_smoker_children.png"),
       "Fig. 3.2: Charges by Smoking Status and by Number of Children",
       "The two smoking groups are almost completely separated, which explains most of the "
       "multimodality seen in Fig. 3.1. Family size, in contrast, shows only a weak and non-monotonic "
       "pattern, so it is expected to be a minor predictor."),
FIGURE(os.path.join(FIG, "fig3_3_scatter.png"),
       "Fig. 3.3: Age and BMI versus Charges, Separated by Smoking Status",
       "Against age, non-smokers form a slowly rising lower band while smokers occupy two higher bands, "
       "so the smoker flag acts as a band selector rather than an additive shift. In the BMI panel the "
       "non-smoker relationship is almost flat, whereas for smokers cost rises steeply once BMI exceeds "
       "approximately 30. The effect of BMI is therefore conditional on smoking status."),
P("The interaction hypothesised from Fig. 3.3 was verified by tabulating mean charges by smoking status "
  "and BMI category (Table 3.3)."),
TBL("Table 3.3: Mean Charges (USD) by Smoking Status and BMI Category",
 ["BMI Category (kg/m\u00b2)", "Non-smoker", "Smoker", "Difference"],
 [
  ["Underweight (< 18.5)"] + [f"{BMISM['no|Under']:,.2f}", f"{BMISM['yes|Under']:,.2f}",
                             f"{BMISM['yes|Under'] - BMISM['no|Under']:,.2f}"],
  ["Normal (18.5\u201324.9)"] + [f"{BMISM['no|Normal']:,.2f}", f"{BMISM['yes|Normal']:,.2f}",
                               f"{BMISM['yes|Normal'] - BMISM['no|Normal']:,.2f}"],
  ["Overweight (25\u201329.9)"] + [f"{BMISM['no|Over']:,.2f}", f"{BMISM['yes|Over']:,.2f}",
                                 f"{BMISM['yes|Over'] - BMISM['no|Over']:,.2f}"],
  ["Obese (\u2265 30)"] + [f"{BMISM['no|Obese']:,.2f}", f"{BMISM['yes|Obese']:,.2f}",
                          f"{BMISM['yes|Obese'] - BMISM['no|Obese']:,.2f}"],
 ], widths=[1.7, 1.3, 1.3, 1.3], fs=8.5),
H2("3.5 Observations from EDA"),
P("EDA revealed five design requirements that shaped the remainder of the study."),
BL([
 "(1) The target is strongly right-skewed and contains " + f"{IQR['charges']['n_outliers']}" +
 " genuine high-cost cases. These must be retained, because they represent the population the model "
 "must serve; instead of trimming them, the error analysis in Section 6.4 examines where they fall.",
 "(2) The data are effectively a mixture of at least two sub-populations defined by smoking status, so "
 "interaction features are likely to be more valuable than additional transformations of the raw inputs.",
 "(3) The BMI effect is conditional on smoking status (Table 3.3), so a bmi \u00d7 smoker term, an "
 "age \u00d7 smoker term and an obesity indicator must be engineered in Chapter 4.",
 "(4) Marginal correlations with the target are weak (|r| \u2264 0.30), which warns that a model judged "
 "on a single metric may appear unstable; five-fold cross-validation is therefore required alongside "
 "the held-out test set.",
 "(5) Categorical attributes are few and low-cardinality, so one-hot encoding with one dropped "
 "reference level is sufficient, and numeric predictors have very different scales, so standardisation "
 "is required for the polynomial model.",
]),
)

# ================================================================ CHAPTER 4
add(
PB(),
H1("CHAPTER 4 - DATA PREPROCESSING AND FEATURE ENGINEERING"),
H2("4.1 Data Cleaning"),
P("Cleaning was deliberately conservative: the dataset is small and clean, so aggressive correction "
  "would have removed genuine signal. No missing values were present in any of the 1,337 \u00d7 7 "
  "cells, so no imputation was required. Exactly one fully duplicated row was detected and removed with "
  "drop_duplicates(), because an exact repeat carries no additional information and can distort the "
  "split. All values were verified to lie within physically plausible ranges (age 18\u201364, BMI "
  "15.96\u201353.13, children 0\u20135, charges strictly positive), and categorical levels were checked "
  "for whitespace and case inconsistencies, of which none were found."),
P("Outliers were identified with the 1.5 \u00d7 IQR rule but retained. The rule flags "
  f"{IQR['charges']['n_outliers']}" + " observations on charges and " + f"{IQR['bmi']['n_outliers']}" +
  " on BMI, and none on age or children. These are legitimate expensive claims and genuinely high BMI "
  "values rather than recording errors: a tree model is robust to their magnitude, the polynomial model "
  "is protected by Ridge regularisation and standardisation, and trimming them would bias predictions "
  "downwards for exactly the high-cost population an insurer most needs to anticipate. All metrics are "
  "therefore reported in the original USD scale, which keeps RMSE and MAE directly interpretable; a "
  "log-transformed target was examined but not adopted for the main experiments and is discussed in "
  "Appendix B."),
H2("4.2 Encoding and Feature Scaling"),
P("The two binary attributes were mapped to 0/1 (female \u2192 0, male \u2192 1; no \u2192 0, "
  "yes \u2192 1). The region attribute is an unordered four-level nominal variable and was one-hot "
  "encoded with drop_first = True, producing three indicator columns with northeast as the reference. "
  "One-hot encoding avoids imposing a false ordinal relationship between regions, and dropping one "
  "level prevents exact collinearity with the intercept in linear regression."),
P("Scaling was applied selectively rather than universally, because the four algorithms have different "
  "requirements. Multiple linear regression is scale-invariant with respect to its fit and its "
  "R\u00b2, so it was trained on unscaled features in order to preserve the direct economic "
  "interpretation of its coefficients in dollars of charge per unit change of each attribute. The "
  "polynomial model required scaling: expanding eleven features to degree two produces "
  f"{R['poly_n_features']}" + " terms whose magnitudes span several orders, and without "
  "standardisation the Ridge penalty would be applied almost entirely to the low-magnitude interaction "
  "terms. Decision trees and random forests split on thresholds and are invariant to monotone "
  "transformations, so no scaling was applied to them. All scalers and expansions are fitted inside a "
  "scikit-learn Pipeline on training data only, so the test set is transformed with training "
  "statistics and no leakage occurs."),
H2("4.3 Feature Selection / Feature Engineering"),
P("No attribute was removed: all six carry plausible relevance, the feature count is small, and the "
  "feature-importance results in Section 6.4 confirm that the two weakest attributes, sex and region, "
  "together contribute less than one per cent of predictive power, so their retention costs nothing. "
  "Three features were created to encode the domain hypotheses identified during EDA (Table 4.1)."),
TBL("Table 4.1: Engineered Features and Their Rationale",
 ["Engineered Feature", "Definition", "Rationale"],
 [
  ["bmi \u00d7 smoker", "bmi_smoker_interaction",
   "Table 3.3 and Fig. 3.3 show that the BMI slope is steep for smokers and almost flat for "
   "non-smokers; the product term lets linear and polynomial models represent that conditional slope "
   "directly."],
  ["age \u00d7 smoker", "age_smoker_interaction",
   "Fig. 3.3 shows that the smoker cost bands fan out with age; the product term captures the steeper "
   "age gradient among smokers."],
  ["Obesity flag", "is_obese = 1 if bmi \u2265 30 else 0",
   "The clinical threshold of 30 is the point at which the smoker curve rises sharply; a binary "
   "indicator represents the kink more efficiently than a further polynomial term."],
 ], widths=[1.4, 1.75, 3.35], fs=8.5),
P("A total of eleven features were used by the final models: age, sex, bmi, children, smoker, three "
  "region indicators and the three engineered features. The polynomial expansion is applied by the "
  "pipeline on top of these, producing " + f"{R['poly_n_features']}" + " terms of degree up to two."),
H2("4.4 Training, Validation and Test Split"),
P("The cleaned data were divided into a training set (80%) and a held-out test set (20%) using "
  "train_test_split with random_state = 42. The training portion was then divided again into a fitting "
  "subset and a validation subset (80:20 of the training data), so that the validation subset supports "
  "model comparison while the test set is touched exactly once, at the end, for the reported results. "
  "This prevents the optimistic bias that arises when the same data are used both to tune and to "
  "report. No stratification was applied, since the target is continuous, and the random split gives "
  "comparable target distributions across subsets: the mean charge in the test set is USD "
  f"{R['test_y_stats']['mean']:,.2f}" + " against USD " + f"{DESC['charges']['mean']:,.2f}" +
  " overall. A time-based split is impossible because the data have no temporal dimension, which is "
  "acknowledged as a limitation in Section 7.2."),
TBL("Table 4.2: Dataset Split Proportions for Fitting, Validation and Held-Out Testing",
 ["Subset", "Number of Records", "Percentage", "Purpose"],
 [
  ["Training (Fitting)", f"{SPLIT['train']}", f"{SPLIT['train_pct']}%",
   "Model fitting and grid-search inner folds"],
  ["Validation (Model Selection)", f"{SPLIT['val']}", f"{SPLIT['val_pct']}%",
   "Model comparison and hyperparameter selection"],
  ["Testing (Held-Out Final Evaluation)", f"{SPLIT['test']}", f"{SPLIT['test_pct']}%",
   "Final unbiased evaluation, reported in Chapter 6"],
  ["Total", "1,337", "100%", ""],
 ], widths=[2.05, 1.25, 1.0, 2.2], fs=8.5),
P("Three measures were taken to prevent data leakage. First, duplicate removal was performed before "
  "splitting, so a duplicated record could not appear in both training and test data. Second, all "
  "preprocessing that learns parameters from data \u2014 standardisation and polynomial expansion \u2014 "
  "is encapsulated in a pipeline whose fit() is called only on training data. Third, hyperparameter "
  "search used GridSearchCV with five-fold cross-validation inside the training subset only; the test "
  "set was never passed to the search object. Feature engineering used only row-wise arithmetic "
  "(products and a threshold indicator) and therefore introduces no cross-row information."),
)

# ================================================================ CHAPTER 5
add(
PB(),
H1("CHAPTER 5 - MACHINE LEARNING MODEL DEVELOPMENT"),
H2("5.1 Workflow / System Architecture"),
P("The end-to-end pipeline follows a modular workflow, shown in Fig. 5.1:"),
P("Medical Cost Personal Dataset \u2192 EDA &amp; Data Quality Checks \u2192 Data Cleaning "
  "(duplicates, outliers, range validation) \u2192 Encoding &amp; Feature Engineering (binary mapping, "
  "one-hot region, BMI \u00d7 smoker, age \u00d7 smoker, is_obese) \u2192 Train / Validation / Test "
  "Split (64% / 16% / 20%) \u2192 Model Training with 5-Fold GridSearchCV (MLR, Polynomial Deg-2, "
  "Decision Tree, Random Forest) \u2192 Evaluation &amp; Comparison (R\u00b2, MAE, MSE, RMSE, MAPE) "
  "\u2192 Error Analysis &amp; Importance \u2192 Prediction / Demonstration."),
FIGURE(os.path.join(FIG, "fig5_1_workflow.png"),
       "Fig. 5.1: End-to-End Machine Learning Workflow for Insurance Cost Prediction",
       "The dataset is cleaned, encoded and enriched with interaction features before a leakage-free "
       "three-way split. All four regressors are trained under the same protocol, tuned inside the "
       "training subset only, and evaluated once on the held-out test set before the best model is used "
       "for demonstration."),
H2("5.2 Algorithms Selected"),
H3("5.2.1 Algorithm 1: Multiple Linear Regression (OLS Baseline)"),
P("Multiple linear regression assumes that the target is a linear combination of the inputs plus an "
  "independent error term, \u0177 = \u03b2\u2080 + \u03a3 \u03b2\u2c7c x\u2c7c, with coefficients "
  "estimated by ordinary least squares, i.e. by minimising the residual sum of squares "
  "\u03a3(y\u1d62 \u2212 \u0177\u1d62)\u00b2. It was selected as the interpretable baseline because each "
  "coefficient reads directly as dollars of annual charge per unit change in the attribute, holding the "
  "others constant, which is the form an underwriter can inspect. Its assumptions are linearity, "
  "additivity, homoscedastic and normally distributed residuals, and no severe multicollinearity. "
  "Section 3.4 already indicates that the additivity assumption is violated in this dataset, so the "
  "model is expected to underfit; that expectation is confirmed in Chapter 6, and the size of the gap "
  "is itself a useful result. The model has no hyperparameters to tune."),
H3("5.2.2 Algorithm 2: Polynomial Regression (Degree 2 with Ridge Regularisation)"),
P("Polynomial regression retains the linear-in-parameters form but expands the feature space with "
  "powers and cross-products of the inputs up to degree two, producing "
  f"{R['poly_n_features']}" + " terms from the eleven base features: "
  "\u03c6(x) = [x\u2081, \u2026, x\u2096, x\u2081\u00b2, x\u2081x\u2082, \u2026, "
  "x\u2096\u00b2]. It was chosen because Fig. 3.3 and Table 3.3 show that the dominant structure is "
  "multiplicative: the marginal effect of BMI depends on smoking status, and a degree-two expansion "
  "represents exactly that class of effect without the analyst having to guess which interactions "
  "matter. Two precautions are necessary in practice. The expansion inflates the feature count from "
  "eleven to " + f"{R['poly_n_features']}" + " and produces highly collinear columns, so an L2 (Ridge) "
  "penalty with \u03b1 = 1.0 is applied to control variance; and the expanded columns have very "
  "different scales, so the whole transformation is wrapped in a pipeline with standardisation fitted "
  "on the training data only. The main hyperparameter is the degree, fixed at two after degree three "
  "overfitted on the validation subset (Appendix B); the second is the regularisation strength "
  "\u03b1."),
H3("5.2.3 Algorithm 3: Decision Tree Regression"),
P("A regression tree recursively partitions the feature space into rectangular regions and predicts, in "
  "each leaf, the mean target of the training observations that fall there; at each node the split "
  "that maximises the reduction in squared error \u0394SSE is chosen. Trees were selected because they "
  "capture non-linearity and interaction automatically, are invariant to monotone scaling and are easy "
  "to explain to a non-technical audience as a set of if\u2013then rules, which is an attractive "
  "property in insurance. Their weakness is high variance: an unconstrained tree grows until each leaf "
  "holds a handful of points and memorises the training data. Control is therefore exercised through "
  "maximum depth, which limits the number of sequential splits and hence the highest-order interaction "
  "the tree can express, and minimum samples per leaf, which smooths each leaf estimate; both were "
  "tuned by grid search. The model assumes a piecewise-constant target, so predictions are step "
  "functions and cannot extrapolate beyond the range of the training target \u2014 a relevant "
  "limitation for high-cost individuals."),
H3("5.2.4 Algorithm 4: Random Forest Regression"),
P("A random forest averages many decorrelated regression trees, \u0177 = (1/B) \u03a3 "
  "T\u1d47(x). Each tree is grown on a bootstrap sample of the training data, and at every split only a "
  "random subset of the features is considered as candidate splitters, which prevents the trees from "
  "becoming correlated. It was selected as the expected best performer on tabular data of this size: it "
  "inherits the tree's ability to model non-linearity and interactions while averaging away much of the "
  "single tree's instability, and it requires little preprocessing. Its principal hyperparameters are "
  "the number of trees B, maximum depth, minimum samples per leaf and the number of features considered "
  "at each split. It also provides a by-product used in Section 6.4: impurity-based feature importance, "
  "which quantifies each attribute's contribution and therefore lets the model be interpreted rather "
  "than merely reported. Individual splits are invariant to monotone scaling, so no scaling was "
  "applied."),
H2("5.3 Implementation Details"),
P("The system was implemented in Python 3.13 using scikit-learn 1.6.1 (estimators, pipelines, "
  "GridSearchCV, metrics), pandas 2.2.3 and NumPy 2.3.5 (data handling), matplotlib 3.11.2 and seaborn "
  "(figures), and python-docx and reportlab (report generation), on a standard multi-core CPU without "
  "a GPU; the full study, including both grid searches, runs in approximately 100 seconds. A fixed "
  "random_state = 42 is used for the split, the tree models and the cross-validation folds, so all "
  "reported numbers are reproducible. Key modules include medical_insurance_ml.py (loading, EDA, "
  "cleaning, encoding, training, tuning, evaluation and figure generation, writing results.json), "
  "make_figures_refstyle.py (report figures), and make_pdf.py / make_docx.py (report rendering). An "
  "extract of the core modelling code is shown below; the full script is given in Appendix A."),
CODE("# Polynomial model: expansion + scaling + Ridge, fitted on training data only\n"
     "poly_pipe = Pipeline([\n"
     "    (\"poly\",   PolynomialFeatures(degree=2, include_bias=False)),\n"
     "    (\"scaler\", StandardScaler()),\n"
     "    (\"ridge\",  Ridge(alpha=1.0, random_state=42))])\n"
     "poly_pipe.fit(X_train, y_train)\n"
     "\n"
     "# Random forest: five-fold grid search inside the training subset only\n"
     "rf_search = GridSearchCV(\n"
     "    RandomForestRegressor(random_state=42),\n"
     "    {\"n_estimators\": [200, 400], \"max_depth\": [6, 10, 15, None],\n"
     "     \"min_samples_leaf\": [1, 2, 3, 5], \"max_features\": [\"sqrt\", 1.0]},\n"
     "    cv=5, scoring=\"neg_root_mean_squared_error\", n_jobs=-1)\n"
     "rf_search.fit(X_tr, y_tr)"),
H2("5.4 Hyperparameter Settings"),
P("The two linear models have no meaningful tuning parameters beyond the polynomial degree and the "
  "Ridge strength; the tree-based models were tuned by exhaustive grid search with five-fold "
  "cross-validation, scoring on negative root mean squared error (Table 5.1). The tuned tree is "
  "shallow, with depth " + str(R['dt_depth']) + " and " + str(R['dt_leaves']) + " leaves, which is "
  "consistent with a dataset in which two interactions dominate: a few splits suffice to separate the "
  "smoker and BMI regimes and deeper trees only fit noise. The selected forest pairs a moderate depth "
  "with a larger minimum leaf size, indicating that smoother estimates generalised better."),
TBL("Table 5.1: Hyperparameter Settings Across Implemented Regression Models",
 ["Model", "Hyperparameter", "Value / Search Range", "Selection Method"],
 [
  ["Model 1: Multiple Linear Regression", "None (OLS)", "No tunable parameters", "Default"],
  ["Model 2: Polynomial Regression (Deg 2)", "degree, Ridge alpha",
   "degree = 2; alpha = 1.0 (range 0.01\u201310)", "Validation Grid Search"],
  ["Model 3: Decision Tree Regression", "max_depth, min_samples_leaf, min_samples_split",
   "max_depth = " + str(R['dt_best_params']['max_depth']) + "; min_samples_leaf = " +
   str(R['dt_best_params']['min_samples_leaf']) + "; min_samples_split = " +
   str(R['dt_best_params']['min_samples_split']), "5-Fold GridSearchCV"],
  ["Model 4: Random Forest Regression", "n_estimators, max_depth, min_samples_leaf, max_features",
   "n_estimators = " + str(R['rf_best_params']['n_estimators']) + "; max_depth = " +
   str(R['rf_best_params']['max_depth']) + "; min_samples_leaf = " +
   str(R['rf_best_params']['min_samples_leaf']) + "; max_features = " +
   str(R['rf_best_params']['max_features']), "5-Fold GridSearchCV"],
 ], widths=[1.55, 1.4, 1.75, 1.2], fs=8.0),
)

# ================================================================ CHAPTER 6
add(
PB(),
H1("CHAPTER 6 - RESULTS AND PERFORMANCE EVALUATION"),
H2("6.1 Evaluation Metrics"),
P("Because the task is regression with a continuous target, the metrics below were used. Each is "
  "reported on the held-out test set of " + str(SPLIT['test']) + " records and expressed in the original "
  "USD scale so that a non-specialist reader can interpret it."),
BL([
 "R\u00b2 (Coefficient of Determination): 1 \u2212 \u03a3(y\u1d62 \u2212 \u0177\u1d62)\u00b2 / "
 "\u03a3(y\u1d62 \u2212 \u0233)\u00b2, the proportion of variance in charges explained by the model. It "
 "is scale-free, so models are directly comparable, and it is the primary selection metric.",
 "MAE (Mean Absolute Error): (1/n) \u03a3 |y\u1d62 \u2212 \u0177\u1d62|, the average absolute dollar "
 "error and robust to the few extreme charges. Reported because a typical pricing error is easier to "
 "interpret than a squared one.",
 "MSE (Mean Squared Error): (1/n) \u03a3 (y\u1d62 \u2212 \u0177\u1d62)\u00b2, the quantity minimised "
 "by the tree objectives; reported because it is the basis of RMSE.",
 "RMSE (Root Mean Squared Error): \u221aMSE, an error in dollars that penalises large deviations more "
 "than MAE, and the primary model-comparison metric because under-predicting an expensive case is "
 "costlier than a small miss.",
 "MAPE (Mean Absolute Percentage Error): (100/n) \u03a3 |y\u1d62 \u2212 \u0177\u1d62| / y\u1d62, a "
 "relative error that is useful across a cost range spanning two orders of magnitude but unstable for "
 "very low charges.",
 "5-Fold Cross-Validated R\u00b2 / RMSE: the mean of the metric over five non-overlapping folds of the "
 "training subset, confirming that the ranking of models is not an artefact of one particular split.",
]),
H2("6.2 Experimental Results"),
P("Table 6.1 summarises the held-out test-set performance of the four implemented models."),
TBL("Table 6.1: Experimental Performance Comparison on the Held-Out Test Set (N = " +
    str(SPLIT['test']) + ")",
 ["Model", "R\u00b2", "MAE (USD)", "MSE (USD\u00b2)", "RMSE (USD)", "MAPE (%)"],
 [
  ["Model 1: Multiple Linear Regression", f"{MLR['R2']:.4f}", f"{MLR['MAE']:,.2f}",
   f"{MLR['MSE']:,.0f}", f"{MLR['RMSE']:,.2f}", f"{MLR['MAPE_pct']:.2f}"],
  ["Model 2: Polynomial Regression (Deg 2)", f"{POL['R2']:.4f}", f"{POL['MAE']:,.2f}",
   f"{POL['MSE']:,.0f}", f"{POL['RMSE']:,.2f}", f"{POL['MAPE_pct']:.2f}"],
  ["Model 3: Decision Tree Regression", f"{DT['R2']:.4f}", f"{DT['MAE']:,.2f}",
   f"{DT['MSE']:,.0f}", f"{DT['RMSE']:,.2f}", f"{DT['MAPE_pct']:.2f}"],
  ["Model 4: Random Forest Regression", f"{RF['R2']:.4f}", f"{RF['MAE']:,.2f}",
   f"{RF['MSE']:,.0f}", f"{RF['RMSE']:,.2f}", f"{RF['MAPE_pct']:.2f}"],
 ], widths=[1.95, 0.7, 0.95, 1.15, 0.95, 0.8], fs=8.0),
P("Three results stand out. First, the linear baseline reaches R\u00b2 = " + f"{MLR['R2']:.4f}" +
  " with an RMSE of USD " + f"{MLR['RMSE']:,.2f}" + ", respectable in absolute terms but clearly the "
  "weakest of the four, exactly as the EDA predicted. Second, Polynomial Regression is the best model "
  "on every metric, with R\u00b2 = " + f"{POL['R2']:.4f}" + " and RMSE = USD "
  f"{POL['RMSE']:,.2f}" + ", a reduction in error of about "
  f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + "% relative to the linear baseline and an "
  "improvement in explained variance of about " + f"{100 * (POL['R2'] - MLR['R2']):.1f}" +
  " percentage points. Third, Decision Tree and Random Forest land between the two, with the forest "
  "slightly ahead of the single tree on every metric (R\u00b2 = " + f"{RF['R2']:.4f}" + " against "
  f"{DT['R2']:.4f}" + ") but both behind the regularised polynomial model. The five-fold "
  "cross-validated results confirm the ordering and rule out a lucky split: the polynomial model's mean CV "
  "R\u00b2 (" + f"{R['poly_cv_r2']:.4f}" + ") is close to its test R\u00b2, and the random forest's mean "
  "CV RMSE (USD " + f"{R['rf_cv_rmse']:,.2f}" + ") is slightly higher than its test RMSE (USD "
  f"{RF['RMSE']:,.2f}" + "), indicating that the test set is marginally easier than the training folds "
  "rather than that the model is overfitting."),
H2("6.3 Visual Results"),
FIGURE(os.path.join(FIG, "fig6_1_model_comparison.png"),
       "Fig. 6.1: Comparison of Test R\u00b2 and Test RMSE Across the Four Regression Models",
       "The gap between the linear baseline and the three non-linear models is large and consistent "
       "across both metrics, while the gap among the three non-linear models is small, with Polynomial "
       "Regression marginally best on both."),
FIGURE(os.path.join(FIG, "fig6_2_actual_vs_predicted.png"),
       "Fig. 6.2: Predicted versus Actual Charges for the Linear Baseline and the Best Model",
       "The linear model's points form nearly horizontal bands: because a linear function can only "
       "shift the prediction by a few thousand dollars, low-cost individuals are over-predicted and "
       "high-cost individuals are severely under-predicted. The polynomial model's points follow the "
       "diagonal much more closely, although a small group of very expensive non-smokers is still "
       "under-predicted because such cases are rare in the training data."),
FIGURE(os.path.join(FIG, "fig6_3_residuals.png"),
       "Fig. 6.3: Residual Plots for the Linear Baseline and the Best Model",
       "For the linear model the residuals are strongly heteroscedastic and patterned, so the "
       "constant-variance assumption of ordinary least squares is violated. For the polynomial model "
       "the residual cloud is much more symmetric around zero and the pattern largely disappears, "
       "though some large positive residuals remain at mid-range predictions, corresponding to "
       "expensive non-smokers whose cost is driven by factors absent from the dataset."),
FIGURE(os.path.join(FIG, "fig6_4_feature_importance.png"),
       "Fig. 6.4: Random Forest Feature Importances (Top Ten Features)",
       "The single most important feature is the BMI \u00d7 smoker interaction (importance "
       f"{IMP['bmi_smoker_interaction']:.4f}" + "), followed by the age \u00d7 smoker interaction ("
       f"{IMP['age_smoker_interaction']:.4f}" + "), the raw smoker flag (" + f"{IMP['smoker']:.4f}" +
       ") and age (" + f"{IMP['age']:.4f}" + "). BMI alone contributes only " + f"{IMP['bmi']:.4f}" +
       ", while sex (" + f"{IMP['sex']:.4f}" + ") and the three region indicators (each below 0.004) "
       "are almost irrelevant."),
FIGURE(os.path.join(FIG, "fig6_5_error_analysis.png"),
       "Fig. 6.5: Prediction Error Analysis \u2013 Absolute Error by Smoking Status and Signed Error "
       "versus Actual Charge",
       "The left panel shows that the spread of absolute error is similar for both smoking groups, with "
       "the smoker group exhibiting a longer upper tail of large misses. The right panel shows that "
       "errors are concentrated near zero for expensive individuals but fan out downwards for low-cost "
       "individuals, confirming that the model under-predicts a subset of cheap non-smoker cases."),
P("Subgroup error analysis completes the comparison. Mean absolute error is USD "
  f"{R['mae_by_smoker']['no']:,.2f}" + " for non-smokers and USD "
  f"{R['mae_by_smoker']['yes']:,.2f}" + " for smokers, so the best model is not biased against either "
  "group in absolute terms even though smokers' charges are roughly four times higher. The smoker "
  "distribution has a longer upper tail, however, indicating a few large under-predictions among "
  "expensive smokers. This asymmetry is an important caveat: the model explains the high-cost segment "
  "better than the low-cost one."),
H2("6.4 Model Comparison and Discussion"),
P("Polynomial Regression with degree two is the best-performing model on all test metrics "
  "(R\u00b2 = " + f"{POL['R2']:.4f}" + ", MAE = USD " + f"{POL['MAE']:,.2f}" + ", RMSE = USD "
  f"{POL['RMSE']:,.2f}" + ", MAPE = " + f"{POL['MAPE_pct']:.2f}" + "%). The result follows from the "
  "structure of the problem rather than from luck. The dominant relationships in this dataset are "
  "smooth and multiplicative \u2014 a smoking surcharge that scales with BMI and age \u2014 and a "
  "degree-two expansion represents exactly that functional class, while Ridge regularisation suppresses "
  "the inflation in variance that " + f"{R['poly_n_features']}" + " collinear terms would otherwise "
  "cause. The model achieves this with a fraction of the parameter count and instability of a deep tree "
  "ensemble."),
P("Random Forest is a close second (R\u00b2 = " + f"{RF['R2']:.4f}" + ", RMSE = USD "
  f"{RF['RMSE']:,.2f}" + "). It slightly beats the single Decision Tree because averaging many "
  "decorrelated trees reduces the variance from which an individual tree suffers: the tuned tree has "
  "only " + str(R['dt_leaves']) + " leaves, so every prediction is a coarse step function and nearby "
  "individuals can receive very different estimates across splits. The forest nevertheless falls short "
  "of the polynomial model for two reasons: its predictions are piecewise-constant averages and so "
  "cannot represent the smooth slope that the smoking\u2013BMI interaction actually produces, and with "
  "only " + str(SPLIT['train']) + " training rows its splits are estimated from small samples, which "
  "adds noise."),
P("The single Decision Tree (R\u00b2 = " + f"{DT['R2']:.4f}" + ") is the weakest of the non-linear "
  "models and its MAPE of " + f"{DT['MAPE_pct']:.2f}" + "% is notably worse than its R\u00b2 ranking "
  "would suggest, because step-function predictions are particularly poor for low-cost individuals: a "
  "leaf average drawn from a mixed region can over-predict a USD 2,000 case several-fold. This is a "
  "clear illustration of why a single metric is insufficient \u2014 the tree's R\u00b2 is only about "
  "one percentage point below the forest's, yet its relative error is several points higher."),
P("The linear model (R\u00b2 = " + f"{MLR['R2']:.4f}" + ", RMSE = USD " + f"{MLR['RMSE']:,.2f}" +
  ") remains useful as a baseline and as an interpretable model. Its coefficients quantify the additive "
  "effects cleanly: USD " + f"{LIN['age']:,.2f}" + " per year of age, USD " + f"{LIN['bmi']:,.2f}" +
  " per BMI unit, USD " + f"{LIN['children']:,.2f}" + " per child, USD " + f"{LIN['smoker']:,.2f}" +
  " for smoking and USD " + f"{LIN['sex']:,.2f}" + " for being male, from an intercept of USD "
  f"{R['linear_intercept']:,.2f}" + ". The smoking coefficient is by far the largest, matching the EDA, "
  "but its single value must serve both a thin smoker and an obese smoker, which is precisely the "
  "compromise that costs the model " + f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" +
  "% of its RMSE."),

P("Taken together, the results support a clear conclusion: on this dataset the best-performing approach "
  "is a regularised degree-two Polynomial Regression, with a tuned Random Forest as a competitive but "
  "slightly inferior alternative, and both clearly ahead of ordinary linear regression. The remaining "
  "error of about USD " + f"{POL['RMSE']:,.0f}" + " RMSE, roughly "
  f"{100 * POL['RMSE'] / R['test_y_stats']['mean']:.0f}" + "% of the mean test charge, is attributable "
  "to information that the dataset simply does not contain rather than to a fixable modelling defect."),
H2("6.5 Sample Predictions / Demonstration"),
P("Table 6.2 shows the best model applied to five representative held-out test records, with the actual "
  "charge, the predicted charge and the signed error. These records were not used in any fitting or "
  "tuning step."),
TBL("Table 6.2: Representative Sample Predictions on Five Held-Out Test Records",
 ["Sl. No.", "Age", "Sex", "BMI", "Children", "Smoker", "Region", "Actual (USD)",
  "Predicted (USD)", "Error (USD)"],
 [[str(i + 1), str(s['age']), s['sex'], f"{s['bmi']:.2f}", str(s['children']), s['smoker'],
   s['region'], f"{s['actual']:,.2f}", f"{s['predicted']:,.2f}", f"{s['error']:+,.2f}"]
  for i, s in enumerate(SP)],
 widths=[0.55, 0.4, 0.55, 0.5, 0.6, 0.55, 0.85, 1.0, 1.05, 0.95], fs=7.5),
P("The five examples cover both smoking strata and a wide cost range. The most accurate case is a "
  "31-year-old male smoker with BMI 34.39, whose charge of USD " + f"{SP[3]['actual']:,.2f}" +
  " was predicted within " + f"{SP[3]['abs_pct_err']:.2f}" + "% (" + f"USD {SP[3]['error']:+,.2f}" +
  "). The largest relative error is a 31-year-old non-smoker with BMI 38.39, whose charges of USD "
  f"{SP[4]['actual']:,.2f}" + " were over-predicted by " + f"{SP[4]['abs_pct_err']:.2f}" +
  "%: an expensive-looking BMI profile that produced a cheap bill. The pattern matches Section 6.4 "
  "\u2014 errors are small in absolute terms for high-cost smokers and proportionally larger for "
  "low-cost non-smokers."),
P("To illustrate how the system would be used in practice, the model was also applied to three newly "
  "constructed hypothetical applicants, none of whom appears in the dataset."),
P("The three predictions behave as domain intuition expects. The young non-smoker with normal BMI "
  "receives the lowest estimate (USD " + f"{NEW[0]:,.2f}" + "). The middle-aged obese smoker receives "
  "the highest (USD " + f"{NEW[1]:,.2f}" + "), about thirteen times more, because both the smoking main "
  "effect and the BMI \u00d7 smoker interaction are active. The older non-smoker sits between the two "
  "at USD " + f"{NEW[2]:,.2f}" + ", showing that age alone raises cost but far less than smoking "
  "combined with obesity. These values are model outputs on a synthetic benchmark and are not "
  "quotations or advice."),
)

# ================================================================ CHAPTER 7
add(
PB(),
H1("CHAPTER 7 - CONCLUSION AND FUTURE ENHANCEMENTS"),
H2("7.1 Conclusion"),
P("This case study designed, implemented and evaluated four machine learning regression models to "
  "predict annual medical insurance charges from six demographic and lifestyle attributes, using the "
  "publicly available Medical Cost Personal Dataset of 1,337 records after duplicate removal. All six "
  "objectives outlined in Chapter 1 were achieved within the defined scope."),
P("Exploratory analysis established that the target is strongly right-skewed and effectively "
  "multimodal, and that the multimodality is produced by the smoking attribute: mean charges are USD "
  f"{SMOKE['no']:,.2f}" + " for non-smokers against USD " + f"{SMOKE['yes']:,.2f}" + " for smokers. More "
  "importantly, the analysis showed that the effect of BMI on cost is conditional on smoking status "
  "\u2014 a progression from normal to obese BMI raises mean charges by only USD "
  f"{BMISM['no|Obese'] - BMISM['no|Normal']:,.2f}" + " among non-smokers but by USD "
  f"{BMISM['yes|Obese'] - BMISM['yes|Normal']:,.2f}" + " among smokers. This interaction drove the "
  "feature engineering: two product terms and an obesity indicator were added, giving a final feature "
  "set of eleven attributes."),
P("On a leakage-free 80:20 split with a nested validation subset, and with all preprocessing fitted on "
  "training data only, the four models achieved test R\u00b2 values of " + f"{MLR['R2']:.4f}" +
  " (Multiple Linear Regression), " + f"{POL['R2']:.4f}" + " (Polynomial Regression, degree 2), "
  f"{DT['R2']:.4f}" + " (Decision Tree) and " + f"{RF['R2']:.4f}" + " (Random Forest). The "
  "best-performing approach was regularised degree-two Polynomial Regression, with an RMSE of USD "
  f"{POL['RMSE']:,.2f}" + ", an MAE of USD " + f"{POL['MAE']:,.2f}" + " and a MAPE of "
  f"{POL['MAPE_pct']:.2f}" + "%, an improvement of approximately "
  f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + "% in RMSE over the linear baseline. "
  "Five-fold cross-validation reproduced the ranking, so the result is not an artefact of a single "
  "split. Feature importance confirmed that the BMI \u00d7 smoker and age \u00d7 smoker interactions, "
  "together with the smoking flag and age, account for almost all predictive power, while sex and "
  "region are negligible. The subgroup error analysis further shows that the model is equally reliable "
  "in absolute terms for smokers and non-smokers, although relative error is larger for low-cost "
  "non-smokers."),
P("The study therefore supports the conclusion reached in the reviewed literature \u2014 that "
  "non-linear and interaction-aware models clearly outperform additive linear regression for medical "
  "cost prediction \u2014 while adding a nuance that a single-metric comparison would hide: a "
  "well-regularised parametric expansion can match or beat an ensemble of trees on this small tabular "
  "problem, and the choice between them should be made on interpretability and deployment constraints "
  "as well as on R\u00b2. The error analysis also shows that the residual error is concentrated among "
  "low-cost non-smokers, whose charges are driven by factors that the dataset does not record."),
H2("7.2 Limitations"),
BL([
 "Small sample and limited information: with 1,337 records and only six attributes, the dataset omits "
 "the variables that would most improve prediction \u2014 diagnosed conditions, procedure and "
 "medication history, prior claims, income, occupation and plan type. The remaining RMSE of roughly "
 "USD " + f"{POL['RMSE']:,.0f}" + " is largely a data limitation rather than an algorithmic one.",
 "Single random split and no temporal validation: because the data are cross-sectional, performance was "
 "estimated on one random 20% hold-out, confirmed by cross-validation, rather than on a genuinely "
 "out-of-time sample. Real claims data drift with medical inflation and coding changes, which this "
 "design cannot detect.",
 "Population and provenance: the dataset originates from United States policyholders and is a teaching "
 "benchmark whose generating process is not documented. The learned attribute\u2013cost relationship and "
 "the USD magnitudes do not transfer to the Indian insurance market or to any current population "
 "without external validation.",
 "Subgroup asymmetry: absolute error is similar for smokers and non-smokers (USD "
 f"{R['mae_by_smoker']['no']:,.0f}" + " versus USD " + f"{R['mae_by_smoker']['yes']:,.0f}" + "), but "
 "relative error is much larger for non-smokers, and the smoker group shows a longer tail of large "
 "under-predictions. Sex and region are available to the model, and although their measured importance "
 "is negligible, a formal disparate-impact audit was out of scope.",
 "Model limitations: the polynomial model cannot extrapolate beyond the observed range and its "
 f"{R['poly_n_features']}" + " terms, although regularised, are not individually interpretable. The "
 "tree models produce piecewise-constant predictions and are unstable to small data perturbations. No "
 "uncertainty interval is produced: the outputs are point estimates, whereas insurers need a "
 "distribution.",
 "Scope: no deployment, clinical use, premium-setting use or regulatory validation was attempted, and "
 "no causal claim about the effect of smoking or obesity on cost is made. The associations reported are "
 "predictive only.",
]),
H2("7.3 Future Enhancements"),
BL([
 "Enrich the feature set with prior-claim history, chronic-condition flags, medication counts, annual "
 "income, plan type and lifestyle variables. This is the single change most likely to reduce error, "
 "because it addresses the unobserved drivers identified in the error analysis.",
 "Add and tune further algorithms: gradient boosting (XGBoost, LightGBM) and support vector regression "
 "with an RBF kernel are natural candidates, and nested cross-validation with randomised or Bayesian "
 "search would give a more efficient and less optimistic estimate of tuned performance than a single "
 "grid search.",
 "Model the cost distribution rather than its mean, using quantile regression, Tweedie or gamma GLMs, "
 "or gradient-boosted distributional models to produce the prediction intervals that pricing and "
 "reserving actually require.",
 "Improve explainability and fairness auditing: SHAP values would replace impurity-based importance "
 "with locally faithful attributions, and a formal subgroup audit across sex and region would quantify "
 "disparity in error rather than relying on importance scores.",
 "Address the skew explicitly: a log-transformed target with retransformation-bias correction, or a "
 "two-part model separating the probability of any claim from its size, is the standard actuarial "
 "response to a right-skewed target and was only partially explored here.",
 "Validate externally and longitudinally by refitting and testing on an out-of-time and "
 "out-of-population sample \u2014 ideally Indian claims data \u2014 before any claim of practical "
 "usefulness, and monitor drift if the model is ever operationalised.",
 "Deploy a small interactive interface that accepts the six attributes and returns a predicted charge "
 "with an approximate interval, making the model demonstrable to a non-technical audience.",
]),
)

# ================================================================ REFERENCES
REFERENCES = [
 "K. Kaushik, A. Bhardwaj, A. D. Dwivedi and R. Singh, \u201cMachine learning-based regression framework "
 "to predict health insurance premiums,\u201d International Journal of Environmental Research and Public "
 "Health, vol. 19, no. 13, art. 7898, 2022. doi: 10.3390/ijerph19137898.",
 "S. Panda, B. Purkayastha, D. Das, M. Chakraborty and S. K. Biswas, \u201cHealth insurance cost "
 "prediction using regression models,\u201d in Proc. Int. Conf. on Machine Learning, Big Data, Cloud "
 "and Parallel Computing (COM-IT-CON), vol. 1. IEEE, 2022, pp. 168\u2013173. doi: "
 "10.1109/COM-IT-CON54601.2022.9850932.",
 "J. Ruth Sandra, S. Joshi, A. Ravi, A. Kodipalli, T. Rao and S. Kamal, \u201cPrediction of cost for "
 "medical care insurance by using regression models,\u201d in Advances in Communication and "
 "Applications (ERCICA 2023). Singapore: Springer, 2024, pp. 289\u2013300. doi: "
 "10.1007/978-981-99-7633-1_23.",
 "U. Orji and E. Ukwandu, \u201cMachine learning for an explainable cost prediction of medical "
 "insurance,\u201d Machine Learning with Applications, vol. 15, art. 100516, 2024. doi: "
 "10.1016/j.mlwa.2023.100516.",
 "M. Hanafy and O. M. A. Mahmoud, \u201cPredict health insurance cost by using machine learning and DNN "
 "regression models,\u201d International Journal of Innovative Technology and Exploring Engineering, "
 "vol. 10, no. 3, pp. 137\u2013143, 2021. doi: 10.35940/ijitee.C8364.0110321.",
 "M. Choi, \u201cMedical Cost Personal Datasets,\u201d Kaggle. "
 "https://www.kaggle.com/datasets/mirichoi0218/insurance (accessed " + ACCESS_DATE + ").",
 "B. Lantz, Machine Learning with R, 3rd ed. Birmingham, U.K.: Packt Publishing, 2019. "
 "(Original source of insurance.csv.)",
 "F. Pedregosa et al., \u201cScikit-learn: Machine learning in Python,\u201d Journal of Machine Learning "
 "Research, vol. 12, pp. 2825\u20132830, 2011.",
 "L. Breiman, \u201cRandom forests,\u201d Machine Learning, vol. 45, no. 1, pp. 5\u201332, 2001. doi: "
 "10.1023/A:1010933404324.",
 "G. James, D. Witten, T. Hastie and R. Tibshirani, An Introduction to Statistical Learning, 2nd ed. "
 "New York, NY, USA: Springer, 2021.",
 "Scikit-learn developers, \u201cScikit-learn user guide: linear models, decision trees, ensembles, "
 "preprocessing.\u201d https://scikit-learn.org/stable/user_guide.html (accessed " + ACCESS_DATE + ").",
 "W. McKinney, \u201cData structures for statistical computing in Python,\u201d in Proc. 9th Python in "
 "Science Conf., 2010, pp. 56\u201361.",
]

APPENDIX_A_INTRO = (
 "The complete, reproducible pipeline is available in the project repository together with the dataset, "
 "the generated figures and the machine-readable results used in this report."
)
APPENDIX_A_STEPS = [
 "Clone the repository and install dependencies: pip install -r requirements.txt.",
 "Confirm the dataset is present at data/insurance.csv (downloadable from the Kaggle URL in Section 3.1).",
 "Run python src/medical_insurance_ml.py. The script trains all four models, writes the figures into "
 "figures/ and the machine-readable results into src/results.json. Runtime is approximately 100 seconds "
 "on a standard multi-core CPU.",
 "Optionally run python src/make_figures_refstyle.py to regenerate the report figures, then "
 "python src/make_pdf.py and python src/make_docx.py to rebuild the report.",
 "All splits, models and searches use random_state = 42, so reported metrics are reproducible exactly.",
]
APPENDIX_CODE = (
 "# --- extract: preprocessing and feature engineering ---\n"
 "df = pd.read_csv(\"data/insurance.csv\").drop_duplicates().reset_index(drop=True)\n"
 "data = df.copy()\n"
 "data[\"sex\"]    = data[\"sex\"].map({\"female\": 0, \"male\": 1})\n"
 "data[\"smoker\"] = data[\"smoker\"].map({\"no\": 0, \"yes\": 1})\n"
 "data = pd.get_dummies(data, columns=[\"region\"], drop_first=True, dtype=int)\n"
 "\n"
 "# engineered interaction features (Section 4.3)\n"
 "data[\"bmi_smoker_interaction\"] = data[\"bmi\"] * data[\"smoker\"]\n"
 "data[\"age_smoker_interaction\"] = data[\"age\"] * data[\"smoker\"]\n"
 "data[\"is_obese\"]               = (data[\"bmi\"] >= 30).astype(int)\n"
 "\n"
 "X, y = data.drop(columns=[\"charges\"]), data[\"charges\"]\n"
 "X_train, X_test, y_train, y_test = train_test_split(\n"
 "    X, y, test_size=0.20, random_state=42)"
)
APPENDIX_B_INTRO = (
 "The following supplementary results support statements made in the main chapters but are too detailed "
 "to include there. All were produced by the same script and the same fixed seed."
)

# ================================================================ ABSTRACT
ABSTRACT = (
 "Medical insurance premiums must reflect the expected annual medical cost of each policyholder, yet "
 "medical expenditure is strongly right-skewed and driven by interacting lifestyle factors, so additive "
 "actuarial models tend to underfit. In this case study, four machine learning regression models were "
 "trained and compared to predict annual medical insurance charges from six attributes \u2014 age, "
 "gender, body mass index (BMI), number of children, smoking status and residential region \u2014 using "
 "the publicly available Medical Cost Personal Dataset (1,338 raw records, 1,337 after removing one "
 "exact duplicate, with no missing values). After cleaning and encoding, exploratory data analysis "
 "showed the target to be right-skewed (skewness " + f"{IQR['charges']['skew']:.2f}" + ") and effectively "
 "multimodal, with smoking as the dominant discriminator: mean charges are USD "
 f"{SMOKE['no']:,.2f}" + " for non-smokers against USD " + f"{SMOKE['yes']:,.2f}" + " for smokers. The "
 "effect of BMI was further found to be conditional on smoking status, since moving from normal to obese "
 "BMI raises mean charges by only USD " + f"{BMISM['no|Obese'] - BMISM['no|Normal']:,.2f}" + " among "
 "non-smokers but by USD " + f"{BMISM['yes|Obese'] - BMISM['yes|Normal']:,.2f}" + " among smokers. That "
 "interaction motivated three engineered features (BMI \u00d7 smoker, age \u00d7 smoker, is_obese), "
 "giving an eleven-feature input space and a leakage-free 64% / 16% / 20% train\u2013validation\u2013test "
 "split with all scaling fitted on training data only."
)

ABSTRACT_RESULTS = (
 "Multiple Linear Regression, Polynomial Regression (degree 2 with L2 regularisation), Decision Tree "
 "Regression and Random Forest Regression were implemented, with the tree models tuned by five-fold grid "
 "search. On the held-out test set of " + str(SPLIT['test']) + " records they achieved R\u00b2 values of "
 + f"{MLR['R2']:.4f}" + ", " + f"{POL['R2']:.4f}" + ", " + f"{DT['R2']:.4f}" + " and "
 + f"{RF['R2']:.4f}" + " respectively. Regularised degree-two Polynomial Regression was the "
 "best-performing model (R\u00b2 = " + f"{POL['R2']:.4f}" + ", RMSE = USD " + f"{POL['RMSE']:,.2f}" +
 ", MAE = USD " + f"{POL['MAE']:,.2f}" + ", MAPE = " + f"{POL['MAPE_pct']:.2f}" + "%), reducing RMSE by "
 "about " + f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + "% relative to the linear "
 "baseline, and five-fold cross-validation reproduced the ranking. Feature importance and residual "
 "analysis confirmed that the smoking\u2013BMI interaction, age and smoking status account for almost "
 "all predictive power while sex and region are negligible, and that the residual error is concentrated "
 "among low-cost non-smokers whose costs depend on variables absent from the dataset. The results "
 "demonstrate that a compact, well-regularised interaction-aware model can explain over ninety per cent "
 "of the variation in annual insurance charges, and that the remaining error is a limitation of the "
 "available data rather than of the algorithm."
)

KEYWORDS = ("Medical Insurance Cost Prediction, Regression, Polynomial Regression, Random Forest, "
            "Feature Interaction, Regularisation, Explainability, Machine Learning.")

ACKNOWLEDGEMENT = [
 "We express our sincere gratitude to our project guide " + GUIDE + ", " + GUIDE_DESIG + ", Department "
 "of Artificial Intelligence and Machine Learning, for continuous guidance, technical feedback and "
 "encouragement throughout the execution of this Machine Learning Case Study.",
 "We extend our heartfelt thanks to " + HOD + ", Head of Department of Artificial Intelligence and "
 "Machine Learning, and " + PRINCIPAL + ", Principal of BMS Institute of Technology and Management, for "
 "providing the academic infrastructure and supportive environment required to complete this work.",
 "We acknowledge the open-source community behind Python, scikit-learn, pandas, NumPy, matplotlib and "
 "seaborn, whose tools were used throughout this study, and Miri Choi and Kaggle for making the Medical "
 "Cost Personal Dataset publicly available for educational use.",
 "Finally, we thank all faculty members, laboratory staff, and peers for their constructive suggestions "
 "and support during the preparation of this report.",
]

# ================================================================ LIST OF ABBREVIATIONS
ABBREVIATIONS = [
 ("AI", "Artificial Intelligence"), ("ANN", "Artificial Neural Network"),
 ("BMI", "Body Mass Index"), ("CSV", "Comma-Separated Values"),
 ("CV", "Cross-Validation"), ("DNN", "Deep Neural Network"),
 ("EDA", "Exploratory Data Analysis"), ("GLM", "Generalised Linear Model"),
 ("IQR", "Interquartile Range"), ("MAE", "Mean Absolute Error"),
 ("MAPE", "Mean Absolute Percentage Error"), ("ML", "Machine Learning"),
 ("MLR", "Multiple Linear Regression"), ("MSE", "Mean Squared Error"),
 ("OLS", "Ordinary Least Squares"), ("R\u00b2", "Coefficient of Determination"),
 ("RMSE", "Root Mean Squared Error"), ("SHAP", "SHapley Additive exPlanations"),
 ("SSE", "Sum of Squared Errors"), ("SVR", "Support Vector Regression"),
 ("USD", "United States Dollar"), ("VTU", "Visvesvaraya Technological University"),
]
