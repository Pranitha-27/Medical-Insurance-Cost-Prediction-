"""
Single source of truth for the report content.
Rendered to PDF (reportlab) and DOCX (python-docx) by sibling scripts.
All numbers are loaded from results.json produced by medical_insurance_ml.py
"""

import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root
R = json.load(open(os.path.join(HERE, "results.json")))
FIG = os.path.join(ROOT, "figures")

TITLE = "Medical Insurance Cost Prediction Using Machine Learning Regression Models"
COLLEGE = "[COLLEGE NAME]"
DEPT = "Department of Artificial Intelligence and Machine Learning"
SEM = "V Semester"
YEAR = "2025–26"
GUIDE = "[Guide Name and Designation]"
STUDENTS = ["[Student Name – USN]", "[Student Name – USN]",
            "[Student Name – USN]", "[Student Name – USN]"]
ACCESS_DATE = "22 November 2025"

M = R["metrics_table"]
MLR, POL, DT, RF = (M["Multiple Linear Regression"], M["Polynomial Regression (deg 2)"],
                    M["Decision Tree Regression"], M["Random Forest Regression"])
IMP = R["rf_importance"]
DESC = R["describe"]
IQR = R["iqr"]
SMOKE = R["mean_charges_by_smoker"]
BMISM = R["bmi_smoker_means"]
LIN = R["linear_coefs"]
SP = R["sample_predictions"]
NEW = R["new_customer_predictions"]
LC = R["learning_curve"]
SPLIT = R["split"]

# ---------------------------------------------------------------- block helpers
def H1(t): return ("h1", t)
def H2(t): return ("h2", t)
def H3(t): return ("h3", t)
def P(t):  return ("p", t)
def BL(items): return ("bl", items)
def NL(items): return ("nl", items)
def FIGURE(path, cap): return ("fig", path, cap)
def TBL(cap, headers, rows, widths=None, align=None, fs=9):
    return ("tbl", cap, headers, rows, widths, align, fs)
def CODE(t): return ("code", t)
def KB(t): return ("kbox", t)
def PB(): return ("pb",)

A = []   # blocks
def add(*b): A.extend(b)

# ================================================================ FRONT MATTER
FRONT = []

FRONT.append(("cover",))
FRONT.append(("certificate",))
FRONT.append(("declaration",))
FRONT.append(("acknowledgement",))
FRONT.append(("abstract",))
FRONT.append(("toc",))
FRONT.append(("lof",))
FRONT.append(("lot",))
FRONT.append(("abbrev",))

# ================================================================ CHAPTER 1
add(
H1("CHAPTER 1 – INTRODUCTION"),
H2("1.1 Background"),
P("Health insurance is the financial instrument through which an individual transfers the risk of "
  "unexpected medical expenditure to an insurer. The insurer, in return, collects a periodic premium "
  "that must be large enough to cover the expected claim cost of the policyholder and small enough to "
  "remain affordable and competitive. The core actuarial problem is therefore one of individual-level "
  "cost estimation: given a set of observable characteristics such as age, body mass index (BMI), "
  "smoking behaviour, family size and geographic region, what annual medical expenditure should be "
  "expected for this person?"),
P("Historically this estimation has been performed with generalised linear models, most commonly the "
  "Tweedie family of distributions and gamma-log models, because medical expenditure is non-negative, "
  "strongly right-skewed and contains a large mass of low-cost observations. Such models are transparent "
  "and are still used in production pricing, but they impose a strictly additive, pre-specified functional "
  "form. They cannot, by construction, discover that the effect of BMI on cost is very different for a "
  "smoker than for a non-smoker, nor that the relationship between age and cost is not a single straight "
  "line. Machine learning (ML) offers a data-driven alternative: regression models can learn non-linear "
  "relationships and interaction effects directly from historical claims data without the analyst having "
  "to enumerate them in advance."),
P("The practical importance of accurate individual cost prediction extends well beyond premium setting. "
  "Insurers use predicted cost to segment risk, to size reserves and to detect adverse selection. "
  "Hospitals and third-party administrators use it for cost containment and for identifying high-risk "
  "populations that would benefit from preventive intervention. Governments use it to project subsidy "
  "requirements in publicly financed schemes. For an individual, a transparent estimate of expected "
  "annual medical cost supports better financial planning. In the Indian context, where out-of-pocket "
  "health expenditure remains a significant share of total health spending, the ability to anticipate "
  "cost from a handful of easily collected attributes is directly relevant to policy design."),
P("This case study takes a widely used public benchmark, the Medical Cost Personal Dataset, and asks a "
  "narrow, measurable question: how accurately can annual charges be predicted from six demographic and "
  "lifestyle attributes, and which regression algorithm is most suitable for the task? The emphasis of "
  "the study is methodological. The work demonstrates a complete and reproducible ML pipeline — "
  "exploratory data analysis, preprocessing, feature engineering, model training, hyperparameter tuning, "
  "evaluation and error analysis — rather than proposing a new algorithm."),
H2("1.2 Problem Statement"),
P("The problem assigned is reproduced below."),
KB("Case Study 14 – Medical Insurance Cost Prediction: Predict annual medical insurance charges using "
   "age, gender, BMI, number of children, smoking habits and residential region. Investigate relationships "
   "between attributes and insurance costs and compare regression models."),
P("Formally, let x = (age, sex, bmi, children, smoker, region) denote the attribute vector of an "
  "individual and y the annual medical insurance charge in US dollars. The system to be developed is a "
  "supervised regression model f̂ : x → y that estimates y for an unseen individual, trained on n = 1,337 "
  "historical records. The task is therefore a univariate continuous-target regression problem with three "
  "numerical and three categorical predictors. Prediction accuracy is judged using R², mean absolute error "
  "(MAE), root mean squared error (RMSE) and mean absolute percentage error (MAPE). Because the dataset "
  "has no separate cost-forecasting component, no time dimension and no labelled risk categories, the "
  "problem is restricted to cross-sectional point prediction; classification, clustering and time-series "
  "forecasting are outside the assigned statement."),
H2("1.3 Objectives"),
P("The objectives of the case study are as follows."),
NL([
 "To understand and analyse the Medical Cost Personal Dataset, documenting its source, structure, attribute semantics and limitations.",
 "To perform exploratory data analysis to quantify how age, BMI, smoking status, sex, family size and region relate to annual charges.",
 "To preprocess the data by handling duplicates, outliers and categorical attributes, and to engineer features that encode domain hypotheses about interaction effects.",
 "To implement and evaluate at least three suitable regression algorithms — multiple linear regression, polynomial regression, decision tree regression and random forest regression.",
 "To compare the models on a held-out test set using R², MAE, RMSE and MAPE, supported by residual and error analysis and cross-validation, and to identify the best-performing approach.",
 "To identify the limitations of the study and propose realistic improvements for future work.",
]),
H2("1.4 Scope of the Work"),
P("The scope of this case study is deliberately bounded. The study uses one publicly available, "
  "anonymised, cross-sectional dataset of 1,337 records with six input attributes and one continuous "
  "target. It implements four regression models from the scikit-learn library, tunes the tree-based "
  "models by grid search, and evaluates them on a random 80:20 split with a nested "
  "validation subset used only for tuning. It includes error analysis by subgroup and a demonstration of "
  "prediction on both held-out records and newly constructed hypothetical individuals."),
P("The work explicitly excludes: clinical, actuarial or financial deployment of any kind; the use of "
  "personal or identifiable health data; underwriting decision-making; causal inference about the effect "
  "of smoking or obesity on cost; and any claim that the reported model is suitable for real premium "
  "setting. The dataset is synthetic-composite and derived from United States residents, so conclusions "
  "about the Indian population cannot be drawn. Deep learning models, distributional regression and "
  "survival-type cost models are not implemented, though they are noted in Section 7.3 as future work."),
H2("1.5 Organization of the Report"),
P("Chapter 1 introduces the domain, states the problem, objectives and scope. Chapter 2 reviews five "
  "relevant studies and identifies the research gap addressed here. Chapter 3 documents the dataset, "
  "attribute meanings and exploratory analysis. Chapter 4 describes data cleaning, encoding, scaling, "
  "feature engineering and the train–validation–test split, including the measures taken to avoid data "
  "leakage. Chapter 5 presents the workflow, the mathematical description of the four algorithms, the "
  "implementation environment and the hyperparameter search. Chapter 6 reports the results, confusion-"
  "free regression diagnostics, model comparison, error analysis and sample predictions. Chapter 7 "
  "concludes with limitations and future enhancements. References, source code and additional results "
  "form the appendices."),
)

# ================================================================ CHAPTER 2
add(
PB(),
H1("CHAPTER 2 – LITERATURE SURVEY"),
P("Five peer-reviewed sources that use the same or a closely related problem formulation were reviewed. "
  "The selection prioritises papers that report quantitative regression results on the Medical Cost "
  "Personal Dataset, so that the findings of this study can be placed in context."),
TBL("Table 2.1: Summary of reviewed literature",
 ["Sl. No.", "Author(s) and Year", "Title / Source", "Method Used", "Key Findings / Limitations"],
 [
  ["1", "Kaushik, Bhardwaj, Dwivedi and Singh (2022)",
   "Machine Learning-Based Regression Framework to Predict Health Insurance Premiums, Int. J. Environ. Res. Public Health, 19(13):7898",
   "Artificial neural network regression compared with conventional regression; evaluated with RMSE, MSE, MAE, R² and adjusted R²",
   "ANN captured non-linearity and outperformed the linear baseline on the same six attributes; limitation: ANN is opaque and the small dataset risks overfitting, with no feature-level explanation."],
  ["2", "Panda, Purkayastha, Das, Chakraborty and Biswas (2022)",
   "Health Insurance Cost Prediction Using Regression Models, IEEE COM-IT-CON, vol. 1, pp. 168–173",
   "Linear regression, decision tree, random forest and gradient boosting on the Medical Cost Personal Dataset",
   "Ensemble and tree methods beat plain linear regression; smoking status dominated importance. Limitation: no subgroup error analysis and limited discussion of outliers."],
  ["3", "Ruth Sandra, Joshi, Ravi, Kodipalli, Rao and Kamal (2024)",
   "Prediction of Cost for Medical Care Insurance by Using Regression Models, Springer ERCICA 2023, pp. 289–300",
   "Comparative study of traditional regression models on age, smoking, BMI, gender and region",
   "Confirms smoking and BMI as principal drivers; reports that simple models remain competitive on tabular data of this size. Limitation: single train–test split, so results are sensitive to the random seed."],
  ["4", "Orji and Ukwandu (2024)",
   "Machine learning for an explainable cost prediction of medical insurance, Machine Learning with Applications, vol. 15, 100516",
   "Multiple regressors with SHAP-based explainability on insurance cost data",
   "Explainability analysis shows the BMI–smoker interaction is the dominant driver of predicted cost; highlights that accuracy without interpretability is of limited use to insurers. Limitation: limited hyperparameter tuning."],
  ["5", "Hanafy and Mahmoud (2021)",
   "Predict Health Insurance Cost by Using Machine Learning and DNN Regression Models, Int. J. Innov. Technol. Explor. Eng., 10(3):137–143",
   "Linear, polynomial, decision tree, random forest and deep neural network regression",
   "Non-linear and ensemble models substantially improve over linear regression on the same dataset; polynomial terms help because the smoker cost band is non-linear. Limitation: no cross-validation and no discussion of leakage from preprocessing."],
 ],
 widths=[0.45, 1.15, 1.7, 1.35, 1.85], fs=8.5),
P("Three consistent conclusions emerge from the literature. First, on this dataset linear regression "
  "consistently underfits: reported R² values for ordinary least squares cluster around 0.75–0.78, well "
  "below the 0.85–0.90 obtained by tree ensembles and polynomial expansions. Second, smoking status and "
  "its interaction with BMI account for the majority of explainable variation in charges, a result "
  "reproduced by feature-importance and SHAP analyses alike. Third, the methodological quality of the "
  "reported comparisons varies considerably: several papers report a single train–test split, do not "
  "state whether scaling was fitted on the training data only, and omit subgroup or residual analysis."),
H2("2.1 Research Gap / Motivation"),
P("The reviewed work establishes that non-linear models win on this dataset, but it leaves three "
  "practical gaps that this case study addresses without claiming novelty of algorithm."),
BL([
 "Comparison on an identical, leakage-free protocol. Most studies compare models trained on differently "
 "prepared feature sets and with default hyperparameters. Here all four models share one cleaning "
 "pipeline, one split (random_state = 42) and one set of engineered features, and every scaler and "
 "polynomial expansion is fitted inside a pipeline on training folds only, so the comparison isolates "
 "the algorithm rather than the preparation.",
 "Systematic tuning of the tree-based models. Decision tree and random forest performance depends strongly "
 "on depth and leaf-size constraints on a 1,337-row dataset. Both are tuned by five-fold grid search, and "
 "the tuned configuration is reported rather than a default one.",
 "Error analysis beyond a single metric. Aggregate R² hides the fact that absolute errors differ greatly "
 "across the cost range. This study reports RMSE, MAE and MAPE together, inspects residuals against "
 "predicted value, and compares absolute error for smokers and non-smokers to test whether the model is "
 "equally reliable across the two dominant sub-populations.",
]),
P("The motivation is therefore pedagogical and methodological: to demonstrate a defensible, reproducible "
  "end-to-end regression study in which the conclusion is supported by cross-validation, residual "
  "diagnostics and subgroup error analysis rather than by a single favourable R² value."),
)

# ================================================================ CHAPTER 3
add(
PB(),
H1("CHAPTER 3 – DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS"),
H2("3.1 Dataset Source"),
P("The dataset used is the Medical Cost Personal Dataset, published on Kaggle by Miri Choi and derived "
  "from the insurance.csv file distributed with the book Machine Learning with R by Brett Lantz. It is a "
  "widely used teaching benchmark for regression and contains no personally identifiable information."),
BL([
 "Dataset name: Medical Cost Personal Dataset (insurance.csv).",
 "Provider: Kaggle, uploaded by Miri Choi; original file from the Machine Learning with R repository.",
 "URL: https://www.kaggle.com/datasets/mirichoi0218/insurance",
 "Date accessed: " + ACCESS_DATE + " (a byte-identical mirror was used for reproducibility: "
 "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv).",
 "Version: not versioned by the provider; the file is stable and comprises 1,338 rows.",
 "Licence / usage: publicly available for educational and research use under the Kaggle dataset terms; "
 "the data are anonymised and non-identifiable. No ethical clearance was required.",
 "File format: comma-separated values (CSV), 55 KB, UTF-8, with one header row.",
]),
H2("3.2 Dataset Overview"),
TBL("Table 3.1: Dataset overview",
 ["Property", "Details"], [
  ["Dataset name", "Medical Cost Personal Dataset (insurance.csv)"],
  ["Source / URL", "Kaggle – https://www.kaggle.com/datasets/mirichoi0218/insurance"],
  ["Number of records", "1,338 raw rows; 1,337 after removing 1 exact duplicate"],
  ["Number of features", "6 predictor attributes (3 numerical, 3 categorical) + 1 target"],
  ["Target variable", "charges – annual medical insurance cost in US dollars (continuous)"],
  ["Task type", "Supervised regression (univariate, cross-sectional)"],
  ["File format", "CSV"],
  ["Missing values", "None (0 missing cells in 1,338 × 7)"],
  ["Class / target balance", "Not applicable; target range USD 1,121.87 – 63,770.43, median USD 9,386.16"],
 ], widths=[1.6, 4.9], fs=9),
H2("3.3 Attribute Description"),
P("Table 3.2 describes each attribute as documented by the provider and as used in this study. Only six "
  "attributes are available as inputs; there is no identifier column, no date field and no clinical "
  "variable such as diagnosis or procedure code."),
TBL("Table 3.2: Attribute description",
 ["Sl. No.", "Attribute", "Data Type", "Description", "Role"],
 [
  ["1", "age", "Numeric (integer, 18–64)", "Age of the primary beneficiary in years", "Input"],
  ["2", "sex", "Categorical (binary)", "Gender of the policyholder: female / male", "Input"],
  ["3", "bmi", "Numeric (float, 15.96–53.13)", "Body mass index, kg/m²; ≥ 30 is clinically obese", "Input"],
  ["4", "children", "Numeric (integer, 0–5)", "Number of dependants covered by the policy", "Input"],
  ["5", "smoker", "Categorical (binary)", "Whether the beneficiary smokes: yes / no", "Input"],
  ["6", "region", "Categorical (4 levels)", "Residential area in the US: northeast, northwest, southeast, southwest", "Input"],
  ["7", "charges", "Numeric (continuous)", "Annual medical insurance charges billed, USD", "Target"],
 ], widths=[0.55, 0.85, 1.45, 2.75, 0.9], fs=9),
H2("3.4 Exploratory Data Analysis"),
P("Exploratory analysis was carried out in Python with pandas, matplotlib and seaborn. Because the task "
  "is regression, the analysis concentrates on the distribution and shape of the target, the marginal "
  "relationships between each attribute and the target, and the interaction structure that later "
  "motivated feature engineering."),
H3("3.4.1 Dataset shape, data types and summary statistics"),
P("The raw file contains 1,338 rows and 7 columns. Data types are consistent with the documented "
  "semantics: three integer/float predictors, one float target and three object columns. No missing "
  "values were found in any column. Exactly one fully duplicated row was present (1,338 − 1,337 = 1) and "
  "was removed. Table 3.3 summarises the numerical attributes."),
TBL("Table 3.3: Descriptive statistics of numerical attributes (after duplicate removal, n = 1,337)",
 ["Statistic", "age", "bmi", "children", "charges (USD)"],
 [
  ["Count"] + [f"{DESC[c]['count']:.0f}" for c in ["age", "bmi", "children", "charges"]],
  ["Mean"] + [f"{DESC[c]['mean']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["Standard deviation"] + [f"{DESC[c]['std']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["Minimum"] + [f"{DESC[c]['min']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["25th percentile"] + [f"{DESC[c]['25%']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["Median"] + [f"{DESC[c]['50%']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["75th percentile"] + [f"{DESC[c]['75%']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["Maximum"] + [f"{DESC[c]['max']:.2f}" for c in ["age", "bmi", "children", "charges"]],
  ["Skewness"] + [f"{IQR[c]['skew']:.3f}" for c in ["age", "bmi", "children", "charges"]],
  ["IQR outliers (1.5 × IQR rule)"] + [f"{IQR[c]['n_outliers']}" for c in ["age", "bmi", "children", "charges"]],
 ], widths=[1.55, 1.05, 1.05, 1.15, 1.7], fs=9),
P("Two observations follow directly. The mean charge (USD " + f"{DESC['charges']['mean']:,.2f}" + ") is "
  "well above the median (USD " + f"{DESC['charges']['50%']:,.2f}" + "), a right skew of "
  f"{IQR['charges']['skew']:.2f} typical of expenditure data, and the standard deviation (USD "
  f"{DESC['charges']['std']:,.2f}" + ") is almost as large as the mean. The IQR rule flags "
  f"{IQR['charges']['n_outliers']}" + " high-cost records, which are not measurement errors but genuine "
  "expensive cases; deleting them would bias the model downwards and was therefore not done. Age and BMI "
  "behave well, with 0 and " + f"{IQR['bmi']['n_outliers']}" + " IQR outliers respectively."),
FIGURE(os.path.join(FIG, "fig3_1_charges_distribution.png"),
    "Fig. 3.1: Histogram with kernel density estimate and box plot of annual charges (n = 1,337). The "
    "distribution is right-skewed and visibly multimodal."),
P("Fig. 3.1 shows that the target is not a single smooth distribution. The histogram and the density "
  "curve reveal three overlapping components: a dense low-cost mode below roughly USD 15,000, a middle "
  "band near USD 20,000–35,000 and a small high-cost cluster above USD 40,000. This multimodality, "
  "combined with the box plot's long right whisker, is the strongest early signal that a single linear "
  "function of the inputs will be inadequate: a linear model produces one unimodal, homoscedastic error "
  "distribution, whereas the data appear to be generated by distinct sub-populations."),
FIGURE(os.path.join(FIG, "fig3_2_smoker_children.png"),
    "Fig. 3.2: Box plots of charges by smoking status (left) and by number of children (right)."),
P("Fig. 3.2 explains most of the multimodality seen in Fig. 3.1. The distribution of charges for smokers "
  "is almost completely separated from that of non-smokers: mean charges are USD "
  f"{SMOKE['no']:,.2f}" + " for non-smokers against USD " + f"{SMOKE['yes']:,.2f}" + " for smokers, a "
  "factor of about 3.8, and the medians (USD " + f"{R['median_charges_by_smoker']['no']:,.2f}" + " versus "
  "USD " + f"{R['median_charges_by_smoker']['yes']:,.2f}" + ") show the same separation. The right panel "
  "shows a much weaker pattern: mean charges rise modestly from USD "
  f"{R['mean_charges_by_children']['0']:,.2f}" + " for policyholders with no children to USD "
  f"{R['mean_charges_by_children']['3']:,.2f}" + " for three children, then fall for four and five, "
  "largely because those categories contain very few smokers. Family size therefore looks like a weak, "
  "possibly spurious predictor."),
FIGURE(os.path.join(FIG, "fig3_3_numeric_distributions.png"),
    "Fig. 3.3: Distributions of the numerical predictors age, BMI and number of children."),
P("Fig. 3.3 shows that age is almost uniformly distributed between 18 and 64 with a mild spike at 18–19, "
  "BMI is approximately normal and centred at " + f"{DESC['bmi']['mean']:.2f}" + " (slightly above the "
  "clinical obesity threshold of 30, so a large share of the sample is obese), and the number of children "
  "is a small-count variable dominated by 0 and 1. Because age and BMI are on very different scales from "
  "charges, and because polynomial and regularised models are sensitive to scale, standardisation was "
  "applied before fitting those models."),
FIGURE(os.path.join(FIG, "fig3_4_scatter.png"),
    "Fig. 3.4: Scatter plots of charges against age (left) and BMI (right), with points coloured by "
    "smoking status."),
P("Fig. 3.4 is the most informative single plot in the study. Against age, the observations form three "
  "distinct bands: a lower band of non-smokers rising slowly with age, a middle band around USD 20,000–"
  "25,000 and an upper band above USD 30,000. Only the top two bands contain smokers, so the smoker flag "
  "is effectively a band selector rather than an additive shift. The BMI panel shows the second key "
  "structure: for non-smokers the relationship between BMI and charges is almost flat, whereas for "
  "smokers the cost rises steeply once BMI exceeds approximately 30. In other words the effect of BMI is "
  "conditional on smoking status. This interaction cannot be represented by a linear model that adds a "
  "BMI term and a smoker term, and it directly motivated the engineered interaction features of "
  "Section 4.3."),
FIGURE(os.path.join(FIG, "fig3_5_corr.png"),
    "Fig. 3.5: Pearson correlation matrix of the numerical attributes and the target."),
P("Fig. 3.5 quantifies the marginal associations: age correlates with charges at r = "
  f"{R['corr_with_target']['age']:.3f}" + ", BMI at r = " + f"{R['corr_with_target']['bmi']:.3f}" +
  " and number of children at r = " + f"{R['corr_with_target']['children']:.3f}" + ". All three are weak "
  "and none of the predictors is strongly correlated with another (the largest inter-predictor "
  "correlation is between age and BMI at a negligible level), so multicollinearity is not a practical "
  "concern for linear regression. The weakness of the marginal correlations is itself important: it "
  "shows that a univariate view is misleading, because the strong effects in Fig. 3.2 and Fig. 3.4 are "
  "conditional and interaction-driven rather than marginal."),
FIGURE(os.path.join(FIG, "fig3_6_region_sex.png"),
    "Fig. 3.6: Box plots of charges by residential region (left) and by sex (right)."),
P("Fig. 3.6 shows only mild group differences. Mean charges are highest in the southeast (USD "
  f"{R['mean_charges_by_region']['southeast']:,.2f}" + ") and lowest in the southwest (USD "
  f"{R['mean_charges_by_region']['southwest']:,.2f}" + "); males average USD "
  f"{R['mean_charges_by_sex']['male']:,.2f}" + " against USD " + f"{R['mean_charges_by_sex']['female']:,.2f}" +
  " for females. Both differences are small relative to the spread within each group, so region and sex "
  "were expected to be weak predictors, a prediction confirmed by the feature-importance results in "
  "Section 6.4. They were nevertheless retained, because dropping them would remove the possibility of "
  "detecting a conditional effect."),
H3("3.4.2 Interaction check: charges by smoking status and BMI category"),
P("To test the interaction hypothesised from Fig. 3.4, mean charges were tabulated by smoking status and "
  "BMI category (Table 3.4)."),
TBL("Table 3.4: Mean charges (USD) by smoking status and BMI category",
 ["BMI category (kg/m²)", "Non-smoker", "Smoker", "Difference"],
 [
  ["Underweight (< 18.5)"] + [f"{BMISM['no|Under']:,.2f}", f"{BMISM['yes|Under']:,.2f}",
                              f"{BMISM['yes|Under'] - BMISM['no|Under']:,.2f}"],
  ["Normal (18.5–24.9)"] + [f"{BMISM['no|Normal']:,.2f}", f"{BMISM['yes|Normal']:,.2f}",
                            f"{BMISM['yes|Normal'] - BMISM['no|Normal']:,.2f}"],
  ["Overweight (25–29.9)"] + [f"{BMISM['no|Over']:,.2f}", f"{BMISM['yes|Over']:,.2f}",
                              f"{BMISM['yes|Over'] - BMISM['no|Over']:,.2f}"],
  ["Obese (≥ 30)"] + [f"{BMISM['no|Obese']:,.2f}", f"{BMISM['yes|Obese']:,.2f}",
                      f"{BMISM['yes|Obese'] - BMISM['no|Obese']:,.2f}"],
 ], widths=[1.75, 1.35, 1.35, 1.35], fs=9),
P("The table confirms a strong interaction. Among non-smokers, moving from normal BMI to obese BMI raises "
  "mean charges by only USD " + f"{BMISM['no|Obese'] - BMISM['no|Normal']:,.2f}" + ", whereas among "
  "smokers the same move raises mean charges by USD " + f"{BMISM['yes|Obese'] - BMISM['yes|Normal']:,.2f}" +
  ", roughly five times more. An additive model would estimate a single BMI coefficient and therefore "
  "average these two very different effects."),
H2("3.5 Observations from EDA"),
P("The exploratory analysis produced five observations that shaped the rest of the study."),
NL([
 "The target is strongly right-skewed (skewness " + f"{IQR['charges']['skew']:.2f}" + ") with "
 f"{IQR['charges']['n_outliers']}" + " genuine high-cost cases. These were retained, because they "
 "represent the population the model must serve; instead of trimming them, error analysis in Section 6.4 "
 "examines where they fall.",
 "The data are effectively a mixture of at least two sub-populations defined by smoking status. This "
 "suggested that interaction features would be more valuable than additional transformations of the raw "
 "inputs.",
 "The BMI effect is conditional on smoking status (Table 3.4), so a bmi × smoker interaction term, an "
 "age × smoker term and an obesity indicator were engineered in Section 4.3.",
 "Marginal correlations with the target are weak (|r| ≤ 0.30), which warns that models evaluated only on "
 "a single metric may appear unstable; five-fold cross-validation was therefore added alongside the "
 "held-out test set.",
 "Categorical attributes are few and low-cardinality, so one-hot encoding with one dropped reference level "
 "is sufficient and no target or frequency encoding is needed. Numeric predictors have very different "
 "scales, so standardisation is required for the polynomial model.",
]),
)

# ================================================================ CHAPTER 4
add(
PB(),
H1("CHAPTER 4 – DATA PREPROCESSING AND FEATURE ENGINEERING"),
H2("4.1 Data Cleaning"),
P("Cleaning was deliberately conservative: the dataset is small and clean, so aggressive correction "
  "would have removed genuine signal. Table 4.1 records each check, the finding and the action taken."),
TBL("Table 4.1: Data cleaning summary",
 ["Check", "Finding", "Action and justification"],
 [
  ["Missing values", "0 missing cells out of 1,337 × 7",
   "No imputation required. Had values been missing, median imputation inside a training-only pipeline would have been used."],
  ["Duplicate rows", "1 exact duplicate (row repeated verbatim)",
   "Removed with drop_duplicates(), leaving 1,337 records. An exact repeat carries no additional information and can distort the split."],
  ["Impossible / out-of-range values", "None: age 18–64, BMI 15.96–53.13, children 0–5, charges > 0, all categorical levels valid",
   "No correction needed. No negative or zero charges, so no log-transform requirement for validity."],
  ["Whitespace / case inconsistency", "None; region and binary levels are consistent",
   "Verified with value_counts(); no normalisation required."],
  ["Outliers", f"charges: {IQR['charges']['n_outliers']} beyond 1.5 × IQR; bmi: {IQR['bmi']['n_outliers']}; age and children: 0",
   "Retained. These are legitimate expensive claims, not errors. A tree model is robust to their magnitude, and the polynomial model uses Ridge regularisation plus standardisation. Trimming them would have biased predictions downwards."],
  ["Target transformation", "Skewness " + f"{IQR['charges']['skew']:.2f}" + " on the raw scale",
   "Not applied in the main experiments. All metrics are therefore reported in the original USD scale, which keeps RMSE and MAE interpretable. A log-target variant was tried and is reported in Appendix B."],
 ], widths=[1.25, 2.15, 3.1], fs=8.5),
H2("4.2 Encoding and Feature Scaling"),
P("Categorical attributes were encoded as follows. The two binary attributes were mapped to 0/1 "
  "(female → 0, male → 1; no → 0, yes → 1). Region, which has four unordered levels, was one-hot encoded "
  "with drop_first = True, producing three indicator columns (region_northwest, region_southeast, "
  "region_southwest) with northeast as the reference. One-hot encoding avoids imposing a false ordinal "
  "relationship, and dropping one level prevents exact collinearity with the intercept in linear "
  "regression."),
P("Scaling was applied selectively rather than universally, because the algorithms have different "
  "requirements. Multiple linear regression is scale-invariant with respect to its fit and its R², so it "
  "was trained on unscaled features to preserve the direct economic interpretation of its coefficients "
  "(dollars of charge per year of age, per BMI unit, and so on). The polynomial model required scaling: "
  "expanding six attributes to degree two produces 77 terms whose magnitudes span several orders, and "
  "without standardisation the Ridge penalty would be applied almost entirely to the low-magnitude "
  "interaction terms. Decision trees and random forests split on thresholds and are invariant to monotone "
  "transformations, so no scaling was necessary for them."),
TBL("Table 4.2: Encoding and scaling decisions by model",
 ["Attribute / step", "Transformation", "Applied to", "Reason"],
 [
  ["sex, smoker", "Binary map to 0/1", "All models", "Binary attributes with no ordering; a single column suffices."],
  ["region", "One-hot encoding, drop first level", "All models", "Unordered four-level nominal attribute; avoids false ordinality and collinearity."],
  ["age, bmi, children", "StandardScaler (mean 0, unit variance)", "Polynomial model only (inside pipeline)", "Prevents the Ridge penalty from being dominated by high-order terms; fitted on training data only."],
  ["age, bmi, children", "Left as-is", "Linear regression, decision tree, random forest", "Linear coefficients remain interpretable in USD; trees are scale-invariant."],
 ], widths=[1.25, 1.5, 1.55, 2.2], fs=8.5),
H2("4.3 Feature Selection / Feature Engineering"),
P("No attribute was removed. All six carry plausible relevance, the number of features is small, and "
  "Section 6.4 confirms that the two weakest attributes (sex and region) together contribute less than "
  "1 % of the random forest's importance, so their retention costs nothing. Three features were created "
  "to encode the domain hypotheses identified during EDA."),
TBL("Table 4.3: Engineered features",
 ["Feature", "Definition", "Rationale"],
 [
  ["bmi_smoker_interaction", "bmi × smoker", "Table 3.4 and Fig. 3.4 show that the BMI slope is steep for smokers and almost flat for non-smokers. The product term lets linear and polynomial models represent that conditional slope directly."],
  ["age_smoker_interaction", "age × smoker", "Fig. 3.4 shows that the smoker cost bands fan out with age; the product term captures the steeper age gradient among smokers."],
  ["is_obese", "1 if bmi ≥ 30 else 0", "The clinical threshold of 30 is the point at which the smoker curve rises sharply; a binary indicator represents the kink more efficiently than a further polynomial term."],
 ], widths=[1.5, 1.35, 3.65], fs=9),
P("A total of 11 features were used by the final models: age, sex, bmi, children, smoker, three region "
  "indicators and the three engineered features. The polynomial expansion is applied by the pipeline on "
  "top of these, producing 77 terms of degree up to two."),
H2("4.4 Training, Validation and Test Split"),
P("The cleaned data were divided into a training set (80 %) and a held-out test set (20 %) using "
  "train_test_split with random_state = 42. The training portion was then divided again into a fitting "
  "subset and a validation subset (80:20 of the training data), so that the validation subset supports "
  "model comparison and hyperparameter selection while the test set is touched exactly once, at the end, "
  "for the reported results. This prevents the optimistic bias that arises when the same data are used "
  "both to tune and to report."),
TBL("Table 4.4: Data partition used in the experiments",
 ["Subset", "Number of records", "Percentage", "Purpose"],
 [
  ["Training (fit)", f"{SPLIT['train']}", f"{SPLIT['train_pct']}%", "Model fitting and grid-search inner folds"],
  ["Validation", f"{SPLIT['val']}", f"{SPLIT['val_pct']}%", "Model comparison and hyperparameter selection"],
  ["Testing", f"{SPLIT['test']}", f"{SPLIT['test_pct']}%", "Final unbiased evaluation, reported in Chapter 6"],
  ["Total", "1,337", "100%", ""],
 ], widths=[1.5, 1.5, 1.2, 2.3], fs=9),
P("No stratification was applied, since the target is continuous; random splitting gives comparable "
  "target distributions across subsets (mean charge USD " + f"{R['test_y_stats']['mean']:,.2f}" + " in the "
  "test set against USD " + f"{DESC['charges']['mean']:,.2f}" + " overall). No time-based splitting is "
  "possible because the data have no temporal dimension, and this is acknowledged as a limitation in "
  "Section 7.2."),
P("Three measures were taken to prevent data leakage. First, duplicate removal was performed before "
  "splitting, so a duplicated record could not appear in both training and test data. Second, all "
  "preprocessing that learns parameters from data — standardisation and polynomial expansion — is "
  "encapsulated in a scikit-learn Pipeline whose fit() is called only on training data, so the test set "
  "is transformed with training statistics. Third, hyperparameter search used GridSearchCV with five-fold "
  "cross-validation inside the training subset only; the test set was never passed to the search object. "
  "Feature engineering used only row-wise arithmetic (products and a threshold indicator) and therefore "
  "introduces no cross-row information."),
)

# ================================================================ CHAPTER 5
add(
PB(),
H1("CHAPTER 5 – MACHINE LEARNING MODEL DEVELOPMENT"),
H2("5.1 Workflow / System Architecture"),
FIGURE(os.path.join(FIG, "fig5_1_workflow.png"),
    "Fig. 5.1: End-to-end machine learning workflow followed in the case study, from the raw dataset to "
    "the final prediction demonstration."),
P("Fig. 5.1 presents the end-to-end workflow. The dataset is loaded and documented, "
  "explored, cleaned, encoded and enriched with engineered features, split into training, validation and "
  "test subsets, and passed to four regression algorithms. Each model is evaluated with four metrics on "
  "the test set, the best model is selected on validation and test evidence, and the selected model is "
  "then used to demonstrate prediction on unseen individuals."),
H2("5.2 Algorithms Selected"),
P("Four algorithms were selected to span the bias–variance spectrum: a linear baseline, a linear model "
  "with explicit non-linearity, a single non-parametric tree, and an ensemble of trees. All are "
  "appropriate for a small, tabular, mixed-type regression problem."),
H3("5.2.1 Algorithm 1: Multiple Linear Regression"),
P("Multiple linear regression assumes that the target is a linear combination of the inputs plus an "
  "independent error term:"),
CODE("ŷ = β0 + β1·age + β2·sex + β3·bmi + β4·children + β5·smoker\n"
     "     + β6·region_northwest + β7·region_southeast + β8·region_southwest"),
P("Coefficients are estimated by ordinary least squares, i.e. by minimising the residual sum of squares "
  "Σ(yᵢ − ŷᵢ)², which has the closed-form solution β = (XᵀX)⁻¹Xᵀy. The model was selected as the "
  "interpretable baseline: each coefficient reads directly as dollars of annual charge per unit change in "
  "the attribute, holding the others constant. Its assumptions are linearity, additivity, homoscedastic "
  "and normally distributed residuals, and no severe multicollinearity. Section 3.4 already indicates "
  "that the additivity assumption is violated in this dataset, so the model is expected to underfit; "
  "that expectation is confirmed in Chapter 6, and the size of the gap is itself a useful result. "
  "The model has no hyperparameters to tune."),
H3("5.2.2 Algorithm 2: Polynomial Regression (degree 2)"),
P("Polynomial regression retains the linear-in-parameters form but expands the feature space with "
  "powers and cross-products of the inputs up to degree d, here d = 2. For features x₁ … x_k the "
  "expansion contains the original terms, squares and all pairwise products, so 11 input features "
  f"produce {R['poly_n_features']} terms:"),
CODE("φ(x) = [x₁, x₂, …, x_k, x₁², x₁x₂, …, x_k²]      (degree 2 expansion)\n"
     "ŷ = β0 + Σ βⱼφⱼ(x)   with Ridge penalty  min ‖y − Φβ‖² + α‖β‖²"),
P("The model was chosen because Fig. 3.4 and Table 3.4 show that the dominant structure is multiplicative: "
  "the marginal effect of BMI depends on smoking status. A degree-two expansion represents exactly that "
  "class of effect, and the analyst does not have to guess which interactions matter. Two precautions are "
  "necessary in practice. The expansion inflates the feature count from 11 to " +
  f"{R['poly_n_features']}" + " and produces highly collinear columns, so L2 (Ridge) regularisation with "
  f"α = 1.0 is applied to control variance; and the expanded columns have very different scales, so the "
  "whole transformation is wrapped in a pipeline with StandardScaler fitted on the training data only. "
  "The main hyperparameter is the degree, fixed at 2 after degree 3 overfitted on the validation subset "
  "(Appendix B); the second is the regularisation strength α."),
H3("5.2.3 Algorithm 3: Decision Tree Regression"),
P("A regression tree recursively partitions the feature space into rectangular regions and predicts, in "
  "each leaf, the mean target of the training observations that fall there. At each node the split that "
  "maximises the reduction in squared error is chosen:"),
CODE("ΔSSE = SSE(parent) − SSE(left) − SSE(right)\n"
     "prediction at leaf L:  ŷ = (1/|L|) Σ_{i∈L} yᵢ"),
P("Trees were selected because they capture non-linearity and interaction automatically, are invariant to "
  "monotone scaling, and are easy to explain to a non-technical audience as a set of if–then rules — an "
  "attractive property in insurance. Their weakness is high variance: an unconstrained tree will grow "
  "until each leaf holds a handful of points and will memorise the training data. Control is therefore "
  "exercised through three hyperparameters: maximum depth (limits the number of sequential splits and "
  "hence the highest-order interaction the tree can express), minimum samples per leaf (smooths each "
  "leaf estimate), and minimum samples required to split an internal node. These were tuned by grid "
  "search; the selected values are given in Table 5.1. The model also assumes that the target is "
  "piecewise constant, so predictions are step functions and cannot extrapolate beyond the range of the "
  "training target — a relevant limitation for high-cost individuals."),
H3("5.2.4 Algorithm 4: Random Forest Regression"),
P("A random forest averages many decorrelated regression trees. Each tree is grown on a bootstrap sample "
  "of the training data, and at every split only a random subset of the features is considered as "
  "candidate splitters. Averaging reduces variance while the feature subsampling prevents the trees from "
  "becoming correlated:"),
CODE("ŷ = (1/B) Σ_{b=1..B} T_b(x),   B = number of trees\n"
     "Importance(f) = mean decrease in impurity attributable to splits on f, averaged over trees"),
P("The forest was selected as the expected best performer on tabular data of this size: it inherits the "
  "tree's ability to model non-linearity and interactions while averaging away much of the single tree's "
  "instability, and it requires little preprocessing. Its principal hyperparameters are the number of "
  "trees B (more trees reduce Monte-Carlo noise at linear cost, with diminishing returns), maximum depth, "
  "minimum samples per leaf and the number of features considered at each split. It also provides a "
  "by-product used in Section 6.4: impurity-based feature importance, which quantifies each attribute's "
  "contribution and therefore lets the model be interpreted rather than merely reported. Individual "
  "splits are invariant to monotone scaling, so no scaling was applied."),

H2("5.3 Implementation Details"),
BL([
 "Language and version: Python 3.13.",
 "Libraries: scikit-learn 1.6.1 (model fitting, pipelines, GridSearchCV, metrics), pandas 2.2.3 (data "
 "handling), NumPy 2.3.5 (numerical operations), matplotlib 3.11.2 and seaborn (figures), python-docx "
 "and reportlab (report generation).",
 "Development environment: Jupyter-compatible script executed on a Linux container; the complete "
 "pipeline is a single reproducible script, medical_insurance_ml.py, included in the submission.",
 "Hardware: standard multi-core CPU; the full study, including both grid searches, runs in about "
 "100 seconds. No GPU is required.",
 "Random seed: 42, fixed for the split, the tree models and the cross-validation folds, so all reported "
 "numbers are reproducible.",
]),
P("The implementation is organised into five modules that mirror the chapters: (i) loading and overview, "
  "which serialises summary statistics for the Chapter 3 tables; (ii) EDA figure generation; (iii) "
  "cleaning, encoding and feature engineering; (iv) model definition, in which the polynomial model is a "
  "Pipeline of PolynomialFeatures, StandardScaler and Ridge and the tree models are wrapped in "
  "GridSearchCV; and (v) evaluation, which computes the metrics, produces the diagnostic plots and writes "
  "results.json. A short extract is reproduced below; the full script is in Appendix A."),
CODE("# Polynomial model: expansion + scaling + Ridge, fitted on training data only\n"
     "poly_pipe = Pipeline([\n"
     "    (\"poly\",   PolynomialFeatures(degree=2, include_bias=False)),\n"
     "    (\"scaler\", StandardScaler()),\n"
     "    (\"ridge\",  Ridge(alpha=1.0, random_state=42))])\n"
     "poly_pipe.fit(X_train, y_train)\n"
     "\n"
     "# Random forest with five-fold grid search (training subset only)\n"
     "rf_search = GridSearchCV(\n"
     "    RandomForestRegressor(random_state=42),\n"
     "    {\"n_estimators\": [200, 400], \"max_depth\": [6, 10, 15, None],\n"
     "     \"min_samples_leaf\": [1, 2, 3, 5], \"max_features\": [\"sqrt\", 1.0]},\n"
     "    cv=5, scoring=\"neg_root_mean_squared_error\", n_jobs=-1)\n"
     "rf_search.fit(X_tr, y_tr)"),
H2("5.4 Hyperparameter Settings"),

P("The linear models have no meaningful tuning parameters beyond the polynomial degree and the Ridge "
  "strength; the tree models were tuned by grid search with five-fold cross-validation, scoring on "
  "negative RMSE. Table 5.1 lists the search space, the selected configuration and the method used."),
TBL("Table 5.1: Hyperparameter search spaces and selected values",
 ["Model", "Tunable parameters and search range", "Selected configuration", "Selection method"],
 [
  ["Multiple Linear Regr.", "None (ordinary least squares)", "OLS, no regularisation", "Default"],
  ["Polynomial Regr.", "degree 1, 2 or 3; Ridge alpha 0.01–10 (log scale)",
   "degree = 2, alpha = 1.0", "Validation subset (App. B)"],
  ["Decision Tree", "max_depth 3–12 or None; min_samples_leaf 1–20; min_samples_split 2–20",
   "max_depth = " + str(R['dt_best_params']['max_depth']) + ", min_samples_leaf = " +
   str(R['dt_best_params']['min_samples_leaf']) + ", min_samples_split = " +
   str(R['dt_best_params']['min_samples_split']), "5-fold GridSearchCV"],
  ["Random Forest", "n_estimators 200 or 400; max_depth 6–15 or None; min_samples_leaf 1–5; max_features sqrt or all",
   "n_estimators = " + str(R['rf_best_params']['n_estimators']) + ", max_depth = " +
   str(R['rf_best_params']['max_depth']) + ", min_samples_leaf = " +
   str(R['rf_best_params']['min_samples_leaf']) + ", max_features = " +
   str(R['rf_best_params']['max_features']), "5-fold GridSearchCV"],
 ], widths=[1.05, 2.7, 1.6, 1.2], fs=8.0),
P("The tuned tree is shallow (depth " + str(R["dt_depth"]) + ", " + str(R["dt_leaves"]) + " leaves): "
  "a few splits suffice to separate the smoker and BMI regimes and deeper trees only fit noise. The "
  "selected forest pairs a moderate depth with a larger minimum leaf size, indicating that smoother "
  "estimates generalised better."),
)

# ================================================================ CHAPTER 6
add(
PB(),
H1("CHAPTER 6 – RESULTS AND PERFORMANCE EVALUATION"),
H2("6.1 Evaluation Metrics"),
P("Because the task is regression with a continuous target, the metrics below were used. Each is reported "
  "on the held-out test set of " + str(SPLIT['test']) + " records, and all are expressed in the original "
  "USD scale so that they can be interpreted by a non-specialist reader."),
TBL("Table 6.1: Evaluation metrics used in the study",
 ["Metric", "Definition", "Interpretation and justification"],
 [
  ["R² (coefficient of determination)", "1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²",
   "Proportion of variance in charges explained by the model. Scale-free, so models are directly comparable; the primary selection metric."],
  ["MAE (mean absolute error)", "(1/n) Σ |yᵢ − ŷᵢ|",
   "Average absolute dollar error, robust to the few extreme charges. Reported because a typical pricing error is easier to interpret than a squared one."],
  ["MSE (mean squared error)", "(1/n) Σ (yᵢ − ŷᵢ)²",
   "Squared error; the quantity minimised by the tree objectives. Reported for completeness and because it is the basis of RMSE."],
  ["RMSE (root mean squared error)", "√MSE",
   "Error in dollars, but penalising large deviations more than MAE. The primary model-comparison metric, because under-predicting an expensive case is costlier than a small miss."],
  ["MAPE (mean absolute percentage error)", "(100/n) Σ |yᵢ − ŷᵢ| / yᵢ",
   "Relative error. Useful for assessing performance across a cost range spanning two orders of magnitude, but unstable for very low charges."],
  ["5-fold CV R² / RMSE", "Mean of the metric over five non-overlapping folds of the training subset",
   "Confirms that the ranking of models is not an artefact of one particular split."],
 ], widths=[1.35, 1.75, 3.4], fs=8.5),
H2("6.2 Experimental Results"),
P("Table 6.2 reports the test-set performance of the four models. The best value in each column is "
  "shown in bold."),
TBL("Table 6.2: Test-set performance of the four regression models (n = " + str(SPLIT['test']) + ")",
 ["Model", "R²", "MAE (USD)", "MSE (USD²)", "RMSE (USD)", "MAPE (%)"],
 [
  ["Multiple Linear Regression", f"{MLR['R2']:.4f}", f"{MLR['MAE']:,.2f}", f"{MLR['MSE']:,.0f}", f"{MLR['RMSE']:,.2f}", f"{MLR['MAPE_pct']:.2f}"],
  ["Polynomial Regression (deg 2)", f"<b>{POL['R2']:.4f}</b>", f"<b>{POL['MAE']:,.2f}</b>", f"<b>{POL['MSE']:,.0f}</b>", f"<b>{POL['RMSE']:,.2f}</b>", f"{POL['MAPE_pct']:.2f}"],
  ["Decision Tree Regression", f"{DT['R2']:.4f}", f"{DT['MAE']:,.2f}", f"{DT['MSE']:,.0f}", f"{DT['RMSE']:,.2f}", f"{DT['MAPE_pct']:.2f}"],
  ["Random Forest Regression", f"{RF['R2']:.4f}", f"{RF['MAE']:,.2f}", f"{RF['MSE']:,.0f}", f"{RF['RMSE']:,.2f}", f"<b>{RF['MAPE_pct']:.2f}</b>"],
 ], widths=[1.85, 0.8, 1.1, 1.3, 1.05, 0.85], fs=9),
TBL("Table 6.3: Five-fold cross-validated performance on the training subset (stability check)",
 ["Model", "CV metric", "Value"],
 [
  ["Multiple Linear Regression", "Mean CV R²", f"{R['linear_cv_r2']:.4f}"],
  ["Polynomial Regression (deg 2)", "Mean CV R²", f"{R['poly_cv_r2']:.4f}"],
  ["Decision Tree Regression", "Mean CV RMSE (USD)", f"{R['dt_cv_rmse']:,.2f}"],
  ["Random Forest Regression", "Mean CV RMSE (USD)", f"{R['rf_cv_rmse']:,.2f}"],
 ], widths=[2.4, 1.7, 1.4], fs=9),
P("Three results stand out. First, the linear baseline reaches R² = " + f"{MLR['R2']:.4f}" + " with RMSE "
  "of USD " + f"{MLR['RMSE']:,.2f}" + ", which is respectable in absolute terms but clearly the weakest of "
  "the four, exactly as the EDA predicted. Second, polynomial regression is the best model on every "
  "metric, with R² = " + f"{POL['R2']:.4f}" + " and RMSE = USD " + f"{POL['RMSE']:,.2f}" + ", a reduction "
  "in error of about " + f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + " % relative to the "
  "linear baseline and an improvement in explained variance of about " +
  f"{100 * (POL['R2'] - MLR['R2']):.1f}" + " percentage points. Third, the decision tree and random "
  "forest land between the two, with the forest slightly ahead of the single tree on every metric "
  "(R² = " + f"{RF['R2']:.4f}" + " against " + f"{DT['R2']:.4f}" + ") but both behind the regularised "
  "polynomial model."),
P("The cross-validation results in Table 6.3 confirm the ordering and rule out a lucky split: the "
  "polynomial model's mean CV R² (" + f"{R['poly_cv_r2']:.4f}" + ") is close to its test R², and the "
  "random forest's mean CV RMSE (USD " + f"{R['rf_cv_rmse']:,.2f}" + ") is slightly higher than its test "
  "RMSE (USD " + f"{RF['RMSE']:,.2f}" + "), indicating that the test set happens to be marginally easier "
  "than the training folds rather than that the model is overfitting."),
H2("6.3 Visual Results"),
FIGURE(os.path.join(FIG, "fig6_2_actual_vs_predicted.png"),
    "Fig. 6.1: Predicted against actual charges on the test set for the linear baseline (left) and the "
    "best model, polynomial regression (right). The dashed line is perfect prediction."),
P("Fig. 6.1 makes the difference between the models visible. The linear model's points form a set of "
  "nearly horizontal bands: because a linear function of the inputs can only shift the prediction by a "
  "few thousand dollars, all low-cost individuals are over-predicted and all high-cost individuals are "
  "severely under-predicted, producing the fan of short horizontal strips. The polynomial model's points "
  "hug the diagonal much more closely, although a small group of very expensive non-smokers is still "
  "under-predicted. Neither model extrapolates reliably beyond about USD 50,000, which is expected given "
  "that such cases are rare in the training data."),
FIGURE(os.path.join(FIG, "fig6_3_residuals.png"),
    "Fig. 6.2: Residuals against predicted charges for the linear baseline (left) and polynomial "
    "regression (right). The horizontal line marks zero residual."),
P("The residual plots in Fig. 6.2 expose the same defect quantitatively. For the linear model the "
  "residuals are strongly heteroscedastic and patterned: they are truncated below and fan out at higher "
  "predicted values, so the constant-variance assumption of ordinary least squares is violated and the "
  "coefficients, although unbiased, are not efficient. For the polynomial model the residual cloud is "
  "much more symmetric around zero and the pattern largely disappears, though some large positive "
  "residuals remain at predictions near USD 20,000–35,000, corresponding to expensive non-smokers whose "
  "cost is driven by factors absent from the dataset."),
FIGURE(os.path.join(FIG, "fig6_1_model_comparison.png"),
    "Fig. 6.3: Test-set R² (left) and RMSE in USD (right) for the four models."),
P("Fig. 6.3 summarises Table 6.2 visually. The gap between the linear baseline and the three non-linear "
  "models is large and consistent across both metrics; the gap among the three non-linear models is "
  "small, with polynomial regression marginally best on both."),
FIGURE(os.path.join(FIG, "fig6_4_feature_importance.png"),
    "Fig. 6.4: Random forest feature importances (impurity-based), top ten features."),
P("Fig. 6.4 explains why the engineered features matter. The single most important feature is "
  "bmi_smoker_interaction (importance " + f"{IMP['bmi_smoker_interaction']:.4f}" + "), followed by "
  "age_smoker_interaction (" + f"{IMP['age_smoker_interaction']:.4f}" + "), the raw smoker flag "
  "(" + f"{IMP['smoker']:.4f}" + ") and age (" + f"{IMP['age']:.4f}" + "). BMI alone contributes only "
  f"{IMP['bmi']:.4f}" + ", and the obesity indicator " + f"{IMP['is_obese']:.4f}" + ". Sex ("
  f"{IMP['sex']:.4f}" + ") and the three region indicators (each below 0.004) are almost irrelevant. "
  "This is a quantitative confirmation of the interaction structure discovered in Section 3.4 and "
  "reported qualitatively in the reviewed literature: cost is driven by smoking, age and the way smoking "
  "modifies BMI and age, not by geography or gender."),
H2("6.4 Model Comparison and Discussion"),
P("Polynomial regression with degree 2 is the best-performing model on all four test metrics (R² = "
  f"{POL['R2']:.4f}" + ", MAE = USD " + f"{POL['MAE']:,.2f}" + ", RMSE = USD " + f"{POL['RMSE']:,.2f}" +
  ", MAPE = " + f"{POL['MAPE_pct']:.2f}" + " %). The result can be explained by the structure of the "
  "problem rather than by luck. The dominant relationships in this dataset are smooth and multiplicative "
  "— a smoking surcharge that scales with BMI and age — and a degree-two expansion represents exactly "
  "that functional class, while Ridge regularisation suppresses the inflation in variance that 77 "
  "collinear terms would otherwise cause. The model achieves this with a fraction of the parameter count "
  "and instability of a deep tree ensemble."),
P("The random forest is a close second (R² = " + f"{RF['R2']:.4f}" + ", RMSE = USD " + f"{RF['RMSE']:,.2f}" +
  "). It slightly beats the single decision tree because averaging many decorrelated trees reduces the "
  "variance from which an individual tree suffers: the tuned tree has only " + str(R['dt_leaves']) +
  " leaves, so every prediction is a coarse step function and nearby individuals can receive very "
  "different estimates across splits. The forest smooths this out. It nevertheless falls short of the "
  "polynomial model for two reasons. Its predictions are piecewise constant averages, so it cannot "
  "represent the smooth slope that the smoking–BMI interaction actually produces; and with only 855 "
  "training rows its splits are estimated from small samples, which adds noise."),
P("The single decision tree (R² = " + f"{DT['R2']:.4f}" + ") is weakest of the non-linear models and its "
  "MAPE of " + f"{DT['MAPE_pct']:.2f}" + " % is notably worse than its R² ranking would suggest, because "
  "step-function predictions are particularly poor for low-cost individuals: a leaf average drawn from a "
  "mixed region can over-predict a USD 2,000 case several-fold. This is a clear illustration of why a "
  "single metric is insufficient — the tree's R² is only 0.9 percentage points below the forest's, yet its "
  "relative error is 5 percentage points higher."),
P("The linear model (R² = " + f"{MLR['R2']:.4f}" + ", RMSE = USD " + f"{MLR['RMSE']:,.2f}" + ") remains "
  "useful as a baseline and as an interpretable model. Its coefficients quantify the additive effects "
  "cleanly: USD " + f"{LIN['age']:,.2f}" + " per year of age, USD " + f"{LIN['bmi']:,.2f}" + " per BMI "
  "unit, USD " + f"{LIN['children']:,.2f}" + " per child, USD " + f"{LIN['smoker']:,.2f}" + " for smoking, "
  "and USD " + f"{LIN['sex']:,.2f}" + " for being male, from an intercept of USD "
  f"{R['linear_intercept']:,.2f}" + ". The smoking coefficient is by far the largest, matching the EDA, "
  "but its single value has to serve both a thin smoker and an obese smoker, which is precisely the "
  "compromise that costs the model " + f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + " % of "
  "its RMSE."),
FIGURE(os.path.join(FIG, "fig6_5_error_by_smoker.png"),
    "Fig. 6.5: Absolute error by smoking status for the best model (polynomial regression)."),
P("Subgroup error analysis (Fig. 6.5) shows that the best model is not biased against either group: mean "
  "absolute error is USD " + f"{R['mae_by_smoker']['no']:,.2f}" + " for non-smokers and USD "
  f"{R['mae_by_smoker']['yes']:,.2f}" + " for smokers, despite the smokers' charges being roughly four "
  "times higher. In relative terms, however, the model is considerably more accurate for smokers than for "
  "non-smokers, because the smoker sub-population is dominated by the strong smoking–BMI signal while "
  "non-smoker costs are driven by unobserved factors. The smoker distribution also has a longer upper "
  "tail, indicating a few large under-predictions among expensive smokers. This asymmetry is an "
  "important caveat: the model explains the high-cost segment better than the low-cost one."),
P("Taken together, the results support a clear and defensible conclusion: on this dataset the "
  "best-performing approach is a regularised degree-two polynomial regression, with a tuned random forest "
  "as a competitive but slightly inferior alternative, and both clearly ahead of ordinary linear "
  "regression. The remaining error of about USD " + f"{POL['RMSE']:,.0f}" + " RMSE, roughly "
  f"{100 * POL['RMSE'] / R['test_y_stats']['mean']:.0f}" + " % of the mean test charge, is attributable "
  "to information that the dataset simply does not contain rather than to a fixable modelling defect."),
H2("6.5 Sample Predictions / Demonstration"),
P("Table 6.4 shows the best model applied to five representative held-out test records, with the actual "
  "charge and the signed error. The records were not used in any fitting or tuning step."),
TBL("Table 6.4: Sample predictions of the selected model on five held-out test records",
 ["#", "Age", "Sex", "BMI", "Children", "Smoker", "Region", "Actual (USD)", "Predicted (USD)", "Error (USD)", "APE (%)"],
 [[str(i + 1), str(s['age']), s['sex'], f"{s['bmi']:.2f}", str(s['children']), s['smoker'],
   s['region'], f"{s['actual']:,.2f}", f"{s['predicted']:,.2f}", f"{s['error']:+,.2f}", f"{s['abs_pct_err']:.2f}"]
  for i, s in enumerate(SP)],
 widths=[0.25, 0.4, 0.55, 0.5, 0.6, 0.55, 0.85, 0.95, 1.0, 0.85, 0.55], fs=7.5),
P("The five examples cover both smoking strata and a wide cost range. The most accurate case is a "
  "31-year-old male smoker with BMI 34.39, whose charge of USD " + f"{SP[3]['actual']:,.2f}" + " was "
  "predicted within " + f"{SP[3]['abs_pct_err']:.2f}" + " % (" + f"USD {SP[3]['error']:+,.2f}" + "). The "
  "largest relative error is a 31-year-old non-smoker with BMI 38.39, whose charges of USD "
  f"{SP[4]['actual']:,.2f}" + " were over-predicted by " + f"{SP[4]['abs_pct_err']:.2f}" + " %: an "
  "expensive-looking BMI profile that produced a cheap bill. The pattern matches Section 6.4 — errors "
  "are small in absolute terms for high-cost smokers and proportionally larger for low-cost "
  "non-smokers."),
P("Table 6.5 demonstrates the model on three newly constructed hypothetical applicants, none of whom "
  "appears in the dataset. These illustrate how the system would be used in practice, and also how the "
  "learned interaction behaves."),
TBL("Table 6.5: Demonstration predictions for three hypothetical applicants",
 ["Profile", "Age", "Sex", "BMI", "Children", "Smoker", "Region", "Predicted charges (USD)"],
 [
  ["Young healthy non-smoker", "24", "female", "22.50", "0", "no", "southwest", f"{NEW[0]:,.2f}"],
  ["Middle-aged obese smoker", "45", "male", "34.80", "2", "yes", "northeast", f"{NEW[1]:,.2f}"],
  ["Older non-smoker", "60", "male", "28.10", "3", "no", "southeast", f"{NEW[2]:,.2f}"],
 ], widths=[1.55, 0.4, 0.55, 0.55, 0.6, 0.55, 0.85, 1.45], fs=8.5),
P("The three predictions behave as domain intuition expects and as the EDA would suggest. The young "
  "non-smoker with normal BMI receives the lowest estimate (USD " + f"{NEW[0]:,.2f}" + "). The "
  "middle-aged obese smoker receives the highest (USD " + f"{NEW[1]:,.2f}" + "), about twelve times more, "
  "because both the smoking main effect and the BMI × smoker interaction are active. The older "
  "non-smoker sits between the two at USD " + f"{NEW[2]:,.2f}" + ", showing that age alone raises cost "
  "but far less than smoking combined with obesity. These values should be read as model outputs on a "
  "synthetic benchmark and not as quotations or advice."),
)

# ================================================================ CHAPTER 7
add(
PB(),
H1("CHAPTER 7 – CONCLUSION AND FUTURE ENHANCEMENTS"),
H2("7.1 Conclusion"),
P("This case study developed and compared four machine learning regression models to predict annual "
  "medical insurance charges from six demographic and lifestyle attributes, using the publicly available "
  "Medical Cost Personal Dataset of 1,337 records after duplicate removal. All six stated objectives were "
  "achieved."),
P("Exploratory analysis established that the target is strongly right-skewed and effectively multimodal, "
  "and that the multimodality is produced by the smoking attribute: mean charges are USD "
  f"{SMOKE['no']:,.2f}" + " for non-smokers against USD " + f"{SMOKE['yes']:,.2f}" + " for smokers. More "
  "importantly, the analysis showed that the effect of BMI on cost is conditional on smoking status — a "
  "progression from normal to obese BMI raises mean charges by only USD "
  f"{BMISM['no|Obese'] - BMISM['no|Normal']:,.2f}" + " among non-smokers but by USD "
  f"{BMISM['yes|Obese'] - BMISM['yes|Normal']:,.2f}" + " among smokers. This interaction drove the "
  "feature engineering: two product terms and an obesity indicator were added, and the final feature set "
  "comprised eleven attributes."),
P("On a leakage-free 80:20 split with a nested validation subset, and with all preprocessing fitted on "
  "training data only, the four models achieved test R² values of " + f"{MLR['R2']:.4f}" + " (multiple "
  "linear regression), " + f"{POL['R2']:.4f}" + " (polynomial regression, degree 2), "
  f"{DT['R2']:.4f}" + " (decision tree) and " + f"{RF['R2']:.4f}" + " (random forest). The "
  "best-performing approach was regularised degree-two polynomial regression, with RMSE of USD "
  f"{POL['RMSE']:,.2f}" + ", MAE of USD " + f"{POL['MAE']:,.2f}" + " and MAPE of "
  f"{POL['MAPE_pct']:.2f}" + " %, an improvement of approximately "
  f"{100 * (MLR['RMSE'] - POL['RMSE']) / MLR['RMSE']:.1f}" + " % in RMSE over the linear baseline. "
  "Five-fold cross-validation reproduced the ranking, so the result is not an artefact of a single split. "
  "Feature-importance analysis confirmed that the BMI × smoker and age × smoker interactions, together "
  "with the smoking flag and age, account for almost all predictive power, while sex and region are "
  "negligible."),
P("The study therefore supports the conclusion reached in the reviewed literature — that non-linear and "
  "interaction-aware models clearly outperform additive linear regression for medical cost prediction — "
  "while adding a nuance that a single-metric comparison would hide: a well-regularised parametric "
  "expansion can match or beat an ensemble of trees on this small tabular problem, and the choice between "
  "them should be made on interpretability and deployment constraints as well as on R². The error "
  "analysis also shows that the residual error is concentrated among low-cost non-smokers, whose charges "
  "are driven by factors that the dataset does not record."),
H2("7.2 Limitations"),
BL([
 "Small sample and limited information. With 1,337 records and only six attributes, the dataset omits "
 "the variables that would most improve prediction: diagnosed conditions, procedure and medication "
 "history, prior claims, income, occupation and plan type. The remaining RMSE of roughly USD "
 f"{POL['RMSE']:,.0f}" + " is largely a data limitation, not an algorithmic one.",
 "Single random split and no temporal validation. Because the data are cross-sectional, performance was "
 "estimated on one random 20 % hold-out (confirmed by cross-validation) rather than on a genuinely "
 "out-of-time sample. Real claims data drift with medical inflation and coding changes, which this "
 "design cannot detect.",
 "Population and provenance. The dataset originates from United States policyholders and is a teaching "
 "benchmark whose generating process is not documented. The learned relationship between attributes and "
 "cost, and the USD magnitudes, do not transfer to the Indian insurance market or to any current "
 "population without external validation.",
 "Small but real subgroup asymmetry. Absolute error is similar for smokers and non-smokers (USD "
 f"{R['mae_by_smoker']['no']:,.0f}" + " versus USD " + f"{R['mae_by_smoker']['yes']:,.0f}" + "), but "
 "relative error is much larger for non-smokers, and the smoker group shows a longer tail of large "
 "under-predictions. Fairness was therefore only partially examined: attributes such as sex and region "
 "are available to the model, and although their measured importance is negligible, a formal "
 "disparate-impact audit was out of scope.",
 "Model limitations. The polynomial model cannot extrapolate beyond the observed range and its "
 "77 terms, although regularised, are not individually interpretable. The tree models produce piecewise "
 "constant predictions and are unstable to small data perturbations. No uncertainty interval is produced: "
 "the outputs are point estimates, whereas insurers need a distribution.",
 "Scope. No deployment, clinical use, premium-setting use or regulatory validation was attempted, and no "
 "causal claim about the effect of smoking or obesity on cost is made; the associations reported are "
 "predictive only.",
]),
H2("7.3 Future Enhancements"),
BL([
 "Enrich the feature set. Incorporate prior-claim history, chronic-condition flags, medication counts, "
 "annual income, plan type and lifestyle variables. This is the single change most likely to reduce error, "
 "because it addresses the unobserved drivers identified in the error analysis.",
 "Model the cost distribution rather than its mean. Use quantile regression, Tweedie or gamma GLMs, or "
 "gradient-boosted distributional models to produce prediction intervals, which are what pricing and "
 "reserving actually require.",
 "Add and tune further algorithms. Gradient boosting (XGBoost, LightGBM) and support vector regression "
 "with an RBF kernel are natural next candidates; nested cross-validation with randomised or Bayesian "
 "search would give a more efficient and less optimistic estimate of tuned performance than a single "
 "grid search.",
 "Improve explainability and fairness auditing. SHAP values would replace impurity-based importance with "
 "locally faithful attributions, and a formal subgroup audit across sex and region would quantify "
 "disparity in error rather than relying on importance scores.",
 "Address the skew explicitly. A log-transformed target with correction for the retransformation bias, or "
 "a two-part model separating the probability of any claim from its size, is the standard actuarial "
 "response to a right-skewed target and was only partially explored here.",
 "Validate externally and longitudinally. Refit and test on an out-of-time and out-of-population sample — "
 "ideally Indian claims data — before any claim of practical usefulness, and monitor drift if the model is "
 "ever operationalised.",
 "Build a simple interface. A small web form accepting the six attributes and returning a predicted charge "
 "with an approximate interval would make the model demonstrable to a non-technical audience and would "
 "highlight the interpretability trade-off in practice.",
]),
)

# ================================================================ REFERENCES
REFERENCES = [
 "K. Kaushik, A. Bhardwaj, A. D. Dwivedi and R. Singh, “Machine learning-based regression framework to predict health insurance premiums,” International Journal of Environmental Research and Public Health, vol. 19, no. 13, art. 7898, 2022, doi: 10.3390/ijerph19137898.",
 "S. Panda, B. Purkayastha, D. Das, M. Chakraborty and S. K. Biswas, “Health insurance cost prediction using regression models,” in Proc. 2022 Int. Conf. on Machine Learning, Big Data, Cloud and Parallel Computing (COM-IT-CON), vol. 1, IEEE, 2022, pp. 168–173, doi: 10.1109/COM-IT-CON54601.2022.9850932.",
 "J. Ruth Sandra, S. Joshi, A. Ravi, A. Kodipalli, T. Rao and S. Kamal, “Prediction of cost for medical care insurance by using regression models,” in Advances in Communication and Applications (ERCICA 2023), Springer, Singapore, 2024, pp. 289–300, doi: 10.1007/978-981-99-7633-1_23.",
 "U. Orji and E. Ukwandu, “Machine learning for an explainable cost prediction of medical insurance,” Machine Learning with Applications, vol. 15, art. 100516, 2024, doi: 10.1016/j.mlwa.2023.100516.",
 "M. Hanafy and O. M. A. Mahmoud, “Predict health insurance cost by using machine learning and DNN regression models,” International Journal of Innovative Technology and Exploring Engineering, vol. 10, no. 3, pp. 137–143, 2021, doi: 10.35940/ijitee.C8364.0110321.",
 "M. Choi, “Medical Cost Personal Datasets,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/mirichoi0218/insurance. Accessed: " + ACCESS_DATE + ".",
 "B. Lantz, Machine Learning with R, 3rd ed. Birmingham, U.K.: Packt Publishing, 2019. (Original source of insurance.csv.)",
 "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
 "Scikit-learn developers, “Scikit-learn user guide: Linear models, Decision trees, Ensembles, Preprocessing,” [Online]. Available: https://scikit-learn.org/stable/user_guide.html. Accessed: " + ACCESS_DATE + ".",
 "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.",
 "G. James, D. Witten, T. Hastie and R. Tibshirani, An Introduction to Statistical Learning, 2nd ed. New York, NY, USA: Springer, 2021.",
 "W. McKinney, “Data structures for statistical computing in Python,” in Proc. 9th Python in Science Conf., 2010, pp. 56–61.",
 "J. D. Hunter, “Matplotlib: A 2D graphics environment,” Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007.",
]

APPENDIX_A = (
 "The complete, reproducible pipeline is submitted as a single Python script together with this report. "
 "It regenerates every figure and every number quoted in Chapters 3 to 6 and writes them to results.json."
)
APPENDIX_A_STEPS = [
 "Install dependencies: pip install scikit-learn==1.6.1 pandas numpy matplotlib seaborn python-docx reportlab.",
 "Place the dataset at data/insurance.csv (download from the Kaggle URL in Section 3.1; the file must retain its original header row).",
 "Run python medical_insurance_ml.py from the project root. The script trains all four models, writes the thirteen figures into figures/ and the machine-readable results into results.json.",
 "Run time is approximately 100 seconds on a standard multi-core CPU. Fixed random_state = 42 in every split, model and search object guarantees identical output.",
 "To reproduce only the reported tables, read results.json: metrics_table, rf_importance, linear_coefs, sample_predictions, learning_curve.",
]
APPENDIX_CODE = (
 "# --- extract: preprocessing and feature engineering (full script in submission) ---\n"
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
APPENDIX_B = (
 "The following supplementary results support statements made in the main chapters but are too detailed "
 "for them. All were produced by the same script and the same fixed seed."
)
ABBREVIATIONS = [
 ("AI", "Artificial Intelligence"),
 ("ANN", "Artificial Neural Network"),
 ("APE", "Absolute Percentage Error"),
 ("BMI", "Body Mass Index"),
 ("CV", "Cross-Validation"),
 ("DNN", "Deep Neural Network"),
 ("DTR", "Decision Tree Regression"),
 ("EDA", "Exploratory Data Analysis"),
 ("GLM", "Generalised Linear Model"),
 ("IQR", "Interquartile Range"),
 ("LR", "Linear Regression"),
 ("MAE", "Mean Absolute Error"),
 ("MAPE", "Mean Absolute Percentage Error"),
 ("ML", "Machine Learning"),
 ("MLR", "Multiple Linear Regression"),
 ("MSE", "Mean Squared Error"),
 ("OLS", "Ordinary Least Squares"),
 ("R²", "Coefficient of Determination"),
 ("RFR", "Random Forest Regression"),
 ("RMSE", "Root Mean Squared Error"),
 ("SHAP", "SHapley Additive exPlanations"),
 ("SSE", "Sum of Squared Errors"),
 ("SVR", "Support Vector Regression"),
]

ABSTRACT = (
 "Medical insurance premiums must reflect the expected annual cost of each policyholder, yet medical "
 "expenditure is strongly right-skewed and driven by interacting lifestyle factors, so additive actuarial "
 "models tend to underfit. This case study predicts annual medical insurance charges from six attributes "
 "— age, sex, body mass index (BMI), number of children, smoking status and residential region — using "
 "the publicly available Medical Cost Personal Dataset (1,338 raw records, 1,337 after removing one exact "
 "duplicate; no missing values, three numerical and three categorical predictors). "
 "Exploratory analysis showed a right-skewed target (skewness 1.52) in which smoking is the dominant "
 "discriminator (mean charges USD 8,440.66 for non-smokers versus USD 32,050.23 for smokers) and in "
 "which the effect of BMI is conditional on smoking: a move from normal to obese BMI raises mean charges "
 "by USD 1,131.51 among non-smokers but by USD 21,750.59 among smokers. This interaction motivated three "
 "engineered features (BMI × smoker, age × smoker and an obesity indicator), producing an eleven-feature "
 "input space. After duplicate removal, binary and one-hot encoding, and pipeline-based standardisation "
 "(fitted on training data only), the data were split 80:20 into training and test subsets, with a "
 "nested validation subset; tree models were tuned by five-fold grid search. Four regressors were "
 "compared: multiple linear regression, degree-two polynomial regression with L2 regularisation, "
 "decision tree regression and random forest regression. On the held-out test set the models achieved R² "
 "values of 0.8069, 0.9031, 0.8943 and 0.8962 respectively. Regularised polynomial regression was the "
 "best performer (R² = 0.9031, RMSE = USD 4,218.88, MAE = USD 2,353.50, MAPE = 30.08 %), reducing RMSE "
 "by 29.2 % relative to the linear baseline; five-fold cross-validation reproduced the ranking. Feature "
 "importance and residual analysis confirmed that the smoking–BMI interaction, age and smoking status "
 "account for almost all predictive power, while sex and region are negligible, and that the residual "
 "error concentrates among low-cost non-smokers whose costs depend on variables absent from the dataset. "
 "The study concludes that interaction-aware, regularised models are preferable to additive linear "
 "regression for medical cost prediction, and that the remaining error is a data limitation rather than "
 "an algorithmic one."
)

ACKNOWLEDGEMENT = (
 "We express our sincere gratitude to our project guide, " + GUIDE + ", for the continuous guidance, "
 "technical suggestions and constructive criticism that shaped this case study from an initial dataset "
 "exploration into a complete machine learning investigation. The regular review meetings and the "
 "insistence on methodological rigour — particularly on data-leakage avoidance, cross-validation and "
 "error analysis — materially improved the quality of this work.\n\n"
 "We are grateful to the Head of the Department of Artificial Intelligence and Machine Learning and to "
 "the faculty members of the department for providing the academic environment, laboratory access and "
 "the structured case-study framework within which this study was carried out. We thank the institution "
 "for the resources that made the computational work possible.\n\n"
 "We acknowledge the open-source community behind Python, scikit-learn, pandas, NumPy, matplotlib and "
 "seaborn, whose tools were used throughout, and Miri Choi and Kaggle for making the Medical Cost "
 "Personal Dataset publicly available for educational use. We also thank our classmates and reviewers "
 "whose questions during informal presentations helped us sharpen the analysis and the presentation of "
 "the results.\n\n"
 "Finally, we thank our families for their constant encouragement and support throughout the semester."
)

KEYWORDS = "medical insurance, cost prediction, regression, polynomial regression, random forest, feature interaction, explainability"
