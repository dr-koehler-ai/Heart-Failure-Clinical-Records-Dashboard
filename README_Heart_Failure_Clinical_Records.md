# Heart-Failure-Clinical-Records-Dashboard-V1
Machine learning project analyzing heart failure clinical records to predict patient mortality risk. Includes data preprocessing, exploratory data analysis, feature engineering, and model evaluation for clinical risk stratification.


## Project Objective

Heart failure remains one of the leading causes of hospitalization and mortality worldwide. Early identification of patients at increased risk of death may support clinical decision-making and improve patient management.

The objective of this project is to identify **demographic and clinical variables associated with mortality in patients with heart failure** using exploratory data analysis (EDA), statistical methods, and predictive machine learning models.

Ultimately, the project aims to demonstrate how clinical data can be transformed into actionable insights for healthcare professionals.

---

## Research Question

> **Which demographic and clinical variables are associated with mortality in patients with heart failure?**

---

## Target Variable

The outcome of interest is:

- **death_event**
  - **0** = Patient survived
  - **1** = Patient died during the follow-up period

This variable serves as the target for all subsequent analyses.

---

## Stakeholders

The findings of this project may be relevant for several stakeholder groups:

- **Cardiologists** – to identify high-risk patients and support treatment decisions.
- **Hospital physicians** – to improve early risk assessment during admission.
- **Nursing staff** – to increase awareness of patients requiring closer monitoring.
- **Hospital management** – to support resource allocation, including ICU capacity planning.
- **Clinical researchers** – to investigate prognostic factors in heart failure.
- **Healthcare AI and Data Science teams** – as a foundation for predictive risk models.

---

## Potential Clinical Impact

Identifying variables associated with increased mortality could support:

- Earlier identification of high-risk patients
- More intensive monitoring of unstable patients
- Earlier escalation of care (e.g., ICU admission)
- More frequent laboratory and clinical reassessment
- Risk stratification at hospital admission
- Improved allocation of healthcare resources

It is important to note that this project aims to identify statistical associations rather than establish causal relationships.

---

# Data Understanding

## Dataset Overview

## Dataset
@article{Chicco2020MachineLC,
  title={Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone},
  author={Davide Chicco and Giuseppe Jurman},
  journal={BMC Medical Informatics and Decision Making},
  year={2020},
  volume={20},
  url={https://api.semanticscholar.org/CorpusID:211018036}
}

This project uses the Heart Failure Clinical Records Dataset, which contains medical records of 299 patients collected at the Faisalabad Institute of Cardiology and the Allied Hospital in Faisalabad, Pakistan, between April and December 2015. The cohort includes 194 male and 105 female patients, aged between 40 and 95 years. All patients had left ventricular systolic dysfunction and had experienced previous heart failure classified as NYHA class III or IV, indicating advanced-stage heart failure.

The dataset consists of 13 clinical, physiological and lifestyle-related features, including both numerical and binary variables. Binary variables include anaemia, high blood pressure, diabetes, sex, and smoking status. Clinical measurements capture key biomarkers relevant to cardiovascular and organ function, such as:
  - Creatinine phosphokinase (CPK): an enzyme indicating potential muscle or cardiac injury when elevated
  - Ejection fraction: percentage of blood pumped out of the left ventricle per heartbeat, a key measure of cardiac function
  - Serum creatinine: an indicator of kidney function, with elevated levels suggesting renal impairment
  - Serum sodium: electrolyte levels associated with fluid balance and may be affected in heart failure patients
The target variable is death event, representing patient survival status during an average follow-up period of 130 days. 

This dataset is commonly used for binary classification tasks, particularly in predicting mortality risk in heart failure patients, and is well-suited for machine learning applications in clinical risk stratification.

The available variables include:

### Continuous Variables

- Age
- Creatinine phosphokinase (CPK)
- Ejection fraction
- Platelets
- Serum creatinine
- Serum sodium
- Follow-up time

### Binary Variables

- Anaemia
- Diabetes
- High blood pressure
- Sex
- Smoking
- Death event (target variable)

---

## Initial Data Quality Assessment

An initial inspection of the dataset included:

- Dataset dimensions
- Variable overview
- Descriptive statistics
- Missing value assessment

### Findings

- No missing values were identified.
- The dataset is complete and suitable for exploratory analysis without prior imputation.

## Outcome Distribution

The dataset contains **299 patients**, of whom:

- **203 patients (67.9%) survived**
- **96 patients (32.1%) died during follow-up**

Although both outcome classes are well represented, the dataset exhibits a **moderate class imbalance** favoring survivors.

This observation is particularly important for predictive modelling, as model performance cannot be adequately assessed using accuracy alone. Future models should therefore be evaluated using additional metrics such as precision, recall, F1-score, and ROC-AUC.

## Univariate Analysis

To explore the relationship between clinical characteristics and mortality, a univariate analysis was performed.

Continuous variables were analyzed using boxplots to compare their distributions between survivors and patients with a fatal outcome (`DEATH_EVENT`). The analysis included:

- Age
- Creatinine phosphokinase
- Ejection fraction
- Platelet count
- Serum creatinine

Binary clinical variables were analyzed using prevalence comparisons between outcome groups:

- Anaemia
- Diabetes
- Hypertension
- Smoking status

This exploratory analysis aimed to identify potential differences between survivors and non-survivors and to generate hypotheses for further statistical testing and machine learning modeling.

### Key Findings

The exploratory analysis revealed several differences in clinical characteristics between survivors and patients with a fatal outcome.

Patients who experienced a fatal event showed:
- Lower ejection fraction values, suggesting reduced cardiac function in this group.
- Higher serum creatinine levels, indicating a potential association between impaired renal function and mortality.
- Higher age compared to survivors.

Other variables, including diabetes, hypertension, smoking status, anaemia, platelet count, and creatinine phosphokinase, showed smaller differences between outcome groups in this cohort.

These findings represent associations within the dataset and were used to guide further statistical analysis and machine learning modeling.

## Statistical Analysis

To identify clinical features associated with mortality, statistical hypothesis testing was performed comparing patients who survived and patients who died during follow-up.

### Methods

Continuous variables were analyzed using appropriate two-sample tests:

- **Welch's t-test** was applied for approximately normally distributed variables.
- **Mann–Whitney U test** was applied for non-normally distributed variables.

Categorical variables were analyzed using:

- **Chi-square test of independence** to evaluate associations between categorical clinical characteristics and mortality outcome.

A significance level of **α = 0.05** was used.

> Note: Statistical testing was performed as an exploratory analysis. P-values indicate statistical significance but do not measure clinical relevance. Effect sizes should additionally be considered when interpreting differences.

---

## Results

### Continuous Variables

| Feature | Test | Statistic | p-value | Interpretation |
|---|---|---:|---:|---|
| Age | Welch t-test | 4.19 | <0.001 | Significant difference: deceased patients were older |
| Ejection fraction | Welch t-test | -4.57 | <0.001 | Significant difference: deceased patients had lower ejection fraction |
| Serum creatinine | Welch t-test | 4.15 | <0.001 | Significant difference: deceased patients had higher creatinine levels |
| Creatinine phosphokinase | Mann–Whitney U | 10028 | 0.684 | No significant difference detected |
| Platelets | Welch t-test | -0.84 | 0.399 | No significant difference detected |
| Serum creatinine | Mann–Whitney U | 14190 | <0.001 | Significant difference confirmed using non-parametric testing |

---

### Categorical Variables

| Feature | Test | Chi-square statistic | p-value | Interpretation |
|---|---|---:|---:|---|
| Anaemia | Chi-square test | 1.04 | 0.307 | No significant association with mortality |
| Diabetes | Chi-square test | 0.00 | 1.000 | No significant association with mortality |
| High blood pressure | Chi-square test | 1.54 | 0.214 | No significant association with mortality |
| Smoking | Chi-square test | 0.01 | 0.932 | No significant association with mortality |

---

## Key Findings

The statistical analysis identified several clinical variables significantly associated with mortality:

- **Higher age** was associated with increased mortality.
- **Lower ejection fraction** showed a significant association with mortality, reflecting reduced cardiac function.
- **Higher serum creatinine levels** were associated with mortality, suggesting a relationship between impaired renal function and outcome.

No statistically significant associations were observed for:

- Diabetes
- Smoking status
- Anaemia
- High blood pressure
- Platelet count
- Creatinine phosphokinase

Effect sizes were additionally considered for approximately normally distributed continuous variables. Age showed a **moderate positive effect** (Cohen's d = 0.56), while ejection fraction showed a **moderate negative effect** (Cohen's d = -0.60). Platelet count showed only a **small/negligible effect** (Cohen's d = -0.11).

Because creatinine phosphokinase and serum creatinine were analyzed using the non-parametric Mann–Whitney U test, Cohen's d was not used as the primary effect-size measure for these variables.

As multiple univariate tests were performed, p-values should be interpreted cautiously. No formal multiple-testing correction was applied because this analysis was exploratory.

These univariate findings were subsequently followed by multivariable logistic regression to evaluate the predictors jointly.


---

# Multivariable Baseline Model

## Correlation Analysis

A Pearson correlation matrix was used as an initial screen for pairwise relationships between the available numerical variables.

The selected baseline clinical predictors showed **no strong pairwise correlations**, suggesting a low risk of severe multicollinearity based on pairwise correlation alone.

Among the baseline variables, the strongest correlations with mortality were observed for:

- **Ejection fraction:** negative correlation with mortality
- **Serum creatinine:** positive correlation with mortality
- **Age:** positive correlation with mortality

These patterns were consistent with the preceding univariate analysis.

The variable **`time`** showed the strongest correlation with `death_event`, but it was **excluded from predictive modeling** because it represents follow-up duration rather than a baseline clinical characteristic available at the time of risk assessment. Including it would therefore not reflect the intended clinical prediction setting.

---

## Logistic Regression Baseline

A multivariable logistic regression model was developed as an **interpretable baseline model** for mortality prediction.

### Predictor Variables

The model included:

- Age
- Anaemia
- Creatinine phosphokinase
- Diabetes
- Ejection fraction
- High blood pressure
- Platelets
- Serum creatinine
- Sex
- Smoking status

The target variable was:

- **`death_event`**
  - `0` = No death
  - `1` = Death

### Train-Test Split

The dataset was split into training and testing sets using:

- **70% training data**
- **30% test data**
- `random_state = 42`
- **stratification by the target variable**

Stratification was used to preserve the mortality distribution across training and test sets.

No upsampling was applied before the split. This avoids introducing duplicated observations into both training and test data and keeps the held-out test set independent.

---

## Model Evaluation

The baseline logistic regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | **0.74** |
| ROC-AUC | **0.75** |
| Precision – Death | **0.65** |
| Recall – Death | **0.45** |
| F1-score – Death | **0.53** |
| Precision – No Death | **0.77** |
| Recall – No Death | **0.89** |
| F1-score – No Death | **0.82** |

### Confusion Matrix

On the held-out test set:

- **54 of 61 survivors** were correctly classified.
- **13 of 29 patients who died** were correctly identified.
- **7 survivors** were incorrectly classified as deaths.
- **16 patients who died** were incorrectly classified as survivors.

The model therefore performed considerably better for survivors than for patients with a fatal outcome.

---

## Clinical Interpretation

The overall accuracy of **74%** appears reasonable for a simple baseline model, but accuracy alone is insufficient in this moderately imbalanced clinical dataset.

The most important limitation is the **recall of 45% for mortality**. At the default classification threshold, the model detected fewer than half of the patients who subsequently died.

From a clinical risk-stratification perspective, these **false-negative predictions** are especially important because high-risk patients could remain unidentified.

The ROC-AUC of **0.75** indicates **moderate discriminatory ability**. The model ranks patients with and without the mortality outcome better than chance, but substantial overlap between the two groups remains.

The logistic regression should therefore be interpreted as an **interpretable reference model rather than a clinically deployable prediction tool**.

---

# Limitations

Several limitations should be considered when interpreting the results:

- The dataset contains only **299 patients**, limiting statistical power and model generalizability.
- The cohort originates from two hospitals in Faisalabad, Pakistan, and may not represent other healthcare systems or patient populations.
- The analysis is observational and identifies **associations rather than causal relationships**.
- Several univariate hypothesis tests were performed without formal multiple-testing correction because the analysis was exploratory.
- Pairwise correlation was used as an initial multicollinearity screen; more formal diagnostics such as the **variance inflation factor (VIF)** could be considered in future analyses.
- Model performance was evaluated on a single train-test split. Cross-validation would provide a more robust estimate of expected performance.
- The default classification threshold of **0.5** may not be optimal for a clinical mortality-screening setting.
- The current baseline model shows insufficient sensitivity for mortality and is **not suitable for clinical use**.

---

# Future Work

The next phase of the project will focus on comparing the logistic regression baseline with additional machine-learning approaches.

Potential extensions include:

- Cross-validation for more robust performance estimation
- Comparison with alternative classifiers
- Evaluation of class weighting or other imbalance-handling strategies
- Threshold optimization with explicit consideration of the sensitivity-specificity trade-off
- Precision-recall analysis
- Additional model calibration assessment
- Formal multicollinearity diagnostics
- Model interpretability and feature-importance analysis

The main objective of future modeling will not simply be to maximize accuracy, but to determine whether mortality detection can be improved while maintaining clinically acceptable false-positive rates.

---

# Current Conclusion

Exploratory and statistical analyses consistently identified **older age, lower ejection fraction, and higher serum creatinine** as the most prominent variables associated with mortality in this cohort.

A multivariable logistic regression baseline achieved **74% accuracy and a ROC-AUC of 0.75**, demonstrating moderate discriminatory performance. However, the model identified only **45% of patients who died**, highlighting the limitations of relying on overall accuracy in clinical prediction tasks.

The project therefore provides a transparent baseline for subsequent machine-learning experiments and illustrates the importance of evaluating predictive models using **clinically meaningful metrics in addition to overall accuracy**.
