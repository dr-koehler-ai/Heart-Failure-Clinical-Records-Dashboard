# Mortality Prediction in Patients with Heart Failure – V2

## Project Overview

This project evaluates several machine-learning models for predicting mortality among patients with heart failure using clinical variables available at baseline.

The project builds on the exploratory data analysis and statistical analysis performed in V1. This second part focuses on:

- Building and comparing classification models
- Prioritizing clinically relevant evaluation metrics
- Addressing moderate class imbalance
- Exploring classification thresholds
- Evaluating model stability using cross-validation
- Identifying potential target leakage
- Discussing clinical applicability and limitations

The project is intended as an educational portfolio project and not as a clinically validated decision-support system.

## Clinical Objective

The objective is to identify patients with heart failure who may be at increased risk of death during follow-up.

In a screening or early-warning context, missing a high-risk patient may be more harmful than producing an additional false-positive alert. Therefore, **recall (sensitivity) for the positive class** was selected as the primary evaluation metric.

Precision was considered alongside recall because excessive false-positive alerts could increase workload and contribute to alert fatigue in an already resource-constrained healthcare system.

## Dataset

The project uses the Heart Failure Clinical Records Dataset described by Chicco and Jurman:

> Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*. 2020;20:16.

The dataset contains records from **299 patients with heart failure**.

### Outcome distribution

- 203 patients survived: 67.9%
- 96 patients died during follow-up: 32.1%

This represents a moderate class imbalance. Accuracy alone would therefore be insufficient for evaluating model performance.

### Target variable

`death_event`

- `0`: Patient survived during follow-up
- `1`: Patient died during follow-up

### Predictor variables

The models use 11 baseline demographic, laboratory, clinical, and lifestyle variables:

- Age
- Anaemia
- Creatinine phosphokinase
- Diabetes
- Ejection fraction
- High blood pressure
- Platelets
- Serum creatinine
- Serum sodium
- Sex
- Smoking status

Both continuous and binary predictors were included.

## Feature Exclusions and Leakage Prevention

The following variables were excluded from the predictors:

### `Unnamed: 0`

This variable represents a technical row index and has no clinical meaning.

### `time`

`time` represents follow-up duration rather than information available at the intended baseline prediction point.

Including follow-up time could introduce target leakage because it contains information accumulated after the initial clinical assessment. It was therefore excluded from all predictive models.

## Data Preparation

The data were divided into:

- 75% training data
- 25% test data

A stratified split was used to preserve the outcome distribution:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

The same split was used for all models to ensure a fair comparison.

Feature scaling was not retained because the final model set consisted of Gaussian Naive Bayes and tree-based algorithms, which did not require scaling for this analysis.

## Models

The following models were evaluated:

1. Gaussian Naive Bayes
2. Decision Tree
3. Tuned Decision Tree
4. Random Forest
5. XGBoost

Gaussian Naive Bayes was included as a simple baseline. Because the predictors contain both continuous and binary variables, its Gaussian distribution assumption is not fully satisfied for every feature.

Hyperparameters for the tuned models were evaluated using `GridSearchCV` with five-fold cross-validation. Recall was used as the refitting metric because of the clinical priority placed on reducing false negatives.

## Test-Set Performance

All models were compared using predictions from the same held-out test set.

| Model | F1 | Recall | Accuracy | Precision |
|---|---:|---:|---:|---:|
| Gaussian Naive Bayes | 0.579 | 0.458 | 0.787 | 0.786 |
| Decision Tree | 0.696 | 0.667 | 0.813 | 0.727 |
| Tuned Decision Tree | 0.612 | 0.625 | 0.747 | 0.600 |
| Random Forest | 0.711 | 0.667 | 0.827 | 0.762 |
| XGBoost | 0.718 | 0.583 | 0.853 | 0.933 |

At the default threshold of 0.50, the Decision Tree and Random Forest achieved the highest recall.

The Random Forest produced the same recall as the Decision Tree while achieving higher precision and accuracy. It was therefore selected as the most suitable preliminary model for the stated clinical objective.

This selection should not be interpreted as evidence that the model is ready for clinical use.

## Confusion Matrix Interpretation

At the default threshold of 0.50, the unweighted Random Forest:

- Correctly identified 16 of 24 death events
- Missed 8 death events
- Generated 5 false-positive alerts
- Correctly classified 46 of 51 survivors

The corresponding recall was 66.7%.

Although this result was better than several alternative models, missing one-third of observed death events would remain clinically concerning.

## ROC-AUC and PR-AUC

Probabilities generated by each model were evaluated using ROC-AUC and precision–recall AUC.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| XGBoost | 0.855 | 0.818 |
| Random Forest | 0.877 | 0.812 |
| Gaussian Naive Bayes | 0.868 | 0.720 |
| Tuned Decision Tree | 0.803 | 0.655 |
| Decision Tree | 0.775 | 0.592 |

The Random Forest achieved the highest ROC-AUC, while XGBoost achieved the highest PR-AUC.

Because the positive class is clinically important and moderately underrepresented, PR-AUC was considered particularly informative. However, the difference between Random Forest and XGBoost was small and should not be interpreted as proof of superiority.

## Cross-Validation Results

Five-fold cross-validation was used within the training data for hyperparameter selection.

| Model | Mean CV Recall | SD CV Recall | Mean CV Precision | SD CV Precision |
|---|---:|---:|---:|---:|
| Tuned Decision Tree | 0.681 | 0.082 | 0.709 | 0.110 |
| Random Forest | 0.777 | 0.114 | 0.798 | 0.093 |
| XGBoost | 0.750 | 0.097 | 0.786 | 0.055 |

The Random Forest achieved the highest mean cross-validation recall and precision. Its recall varied noticeably between folds, indicating uncertainty caused partly by the small dataset.

The differences between models were not large enough to establish a definitive winner.

## Class Weight Experiment

A Random Forest using `class_weight="balanced"` was also evaluated.

The weighted model did not improve test-set recall compared with the unweighted model. Both achieved a recall of 66.7%, while the weighted model produced one additional false-positive prediction and slightly lower precision.

The unweighted Random Forest was therefore retained as the preliminary model.

## Exploratory Threshold Analysis

The default classification threshold was 0.50. Lower thresholds were explored to examine the trade-off between missed death events and false-positive alerts.

For the Random Forest:

| Threshold | Recall | Precision | False Negatives | False Positives |
|---:|---:|---:|---:|---:|
| 0.15 | 0.875 | 0.525 | 3 | 19 |
| 0.25 | 0.792 | 0.576 | 5 | 14 |
| 0.30 | 0.750 | 0.643 | 6 | 10 |
| 0.35 | 0.750 | 0.667 | 6 | 9 |
| 0.50 | 0.667 | 0.762 | 8 | 5 |

A threshold around 0.35 appeared to offer a potentially useful balance between sensitivity and alert burden. Compared with 0.30, it achieved the same recall with one fewer false-positive alert.

However, these thresholds were explored using the test set. Therefore, 0.35 must not be described as an independently validated or optimal clinical threshold. The default threshold of 0.50 remains the primary test-set evaluation.

A final threshold would need to be selected using separate validation data and predefined clinical workflow requirements.

## Calibration

Calibration was explored for Random Forest and XGBoost using calibration curves and Brier scores.

- Random Forest Brier score: 0.128
- XGBoost Brier score: 0.119

XGBoost showed a slightly lower Brier score on the test set. However, calibration estimates based on only 75 test observations are unstable. These findings were treated as exploratory and were not used to claim model superiority.

## Main Findings

- Accuracy alone was not sufficient because a model predicting survival for every patient would already achieve approximately 68% accuracy.
- Adding the available baseline variables improved performance compared with using only four continuous predictors.
- Random Forest provided the most suitable preliminary recall–precision balance at the default threshold.
- XGBoost achieved the highest PR-AUC and precision but lower recall at the default threshold.
- Lowering the Random Forest threshold increased recall but also increased the number of false-positive alerts.
- Class weighting did not improve Random Forest recall on the held-out test set.
- Model performance varied across cross-validation folds, demonstrating uncertainty caused by the small sample.
- No model achieved performance sufficient to support clinical deployment.

## Validation Limitations

Model development and hyperparameter selection were performed using cross-validation within the training set. Final performance was evaluated on a held-out internal test set.

No external validation was performed. The results therefore demonstrate internal performance only and should not be interpreted as evidence of generalizability to other hospitals or patient populations.

The test set contained only 75 patients, including 24 death events. Individual classification errors consequently have a substantial effect on the reported metrics.

## Outcome Limitation

The target indicates whether death occurred during follow-up, while follow-up duration varied between **4 and 285 days**.

The model therefore does not predict mortality within a fixed time horizon such as 30 or 90 days. Patients with longer follow-up also had more time in which the outcome could occur.

A time-to-event analysis would address varying follow-up and censoring more appropriately. Survival analysis was outside the scope of this classification project.

## Clinical Limitations

- The project is based on a small retrospective dataset.
- The cohort originates from a limited clinical setting.
- The prediction horizon is not fixed.
- No external or prospective validation was performed.
- The models were not compared with established clinical risk scores.
- No clinical utility or intervention study was conducted.
- Threshold selection was exploratory.
- Predictions represent statistical estimates and not treatment recommendations.
- Associations and feature importance must not be interpreted as causal effects.
- Performance may differ across demographic groups, hospitals, and healthcare systems.

## Conclusion

The project demonstrates a transparent machine-learning workflow for mortality classification in patients with heart failure.

Among the evaluated models, the Random Forest provided the most suitable preliminary balance between recall and precision for the defined screening objective. Nevertheless, the model missed a clinically relevant proportion of death events, and its performance showed uncertainty across validation folds.

The results should be regarded as an educational proof of concept. Larger datasets, a fixed prediction horizon, external validation, and prospective clinical evaluation would be required before considering real-world use.

## Technologies

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook / Google Colab

## Repository Structure

```text
├── Heart_Failure_Clinical_Records_V1.ipynb
├── Heart_Failure_Clinical_Records_V2.ipynb
├── README_Heart_Failure_Clinical_Records.md
└── data/
    └── heart_failure_clinical_records.csv
```

## Disclaimer

This project is intended solely for educational and portfolio purposes. It has not been clinically validated and must not be used for diagnosis, prognosis, triage, or treatment decisions.
