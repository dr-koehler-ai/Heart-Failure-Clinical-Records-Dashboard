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
| Gaussian Naive Bayes | 0.250 | 0.167 | 0.680 | 0.500 |
| Decision Tree | 0.538 | 0.583 | 0.680 | 0.500 |
| Tuned Decision Tree | 0.512 | 0.458 | 0.720 | 0.579 |
| Random Forest | 0.578 | 0.542 | 0.747 | 0.619 |
| XGBoost | 0.478 | 0.458 | 0.680 | 0.500 |

At the default threshold of 0.50, the Decision Tree achieved the highest recall. However, its precision was only 0.500 and its accuracy was equal to the 68% majority-class baseline.

The Random Forest achieved slightly lower recall but the highest F1 score, precision, accuracy, ROC-AUC, and PR-AUC among the evaluated models. It was therefore selected as the most suitable preliminary compromise for the stated clinical objective.

This selection should not be interpreted as evidence that the model is ready for clinical use.

## Confusion Matrix Interpretation

At the default threshold of 0.50, the unweighted Random Forest:

- Correctly identified 13 of 24 death events
- Missed 11 death events
- Generated 8 false-positive alerts
- Correctly classified 43 of 51 survivors

The corresponding recall was 54.2%.

Although this model offered the best overall metric balance, missing 11 of 24 observed death events would be clinically concerning and prevents any claim of clinical readiness.

## ROC-AUC and PR-AUC

Probabilities generated by each model were evaluated using ROC-AUC and precision–recall AUC.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest | 0.789 | 0.617 |
| Gaussian Naive Bayes | 0.781 | 0.538 |
| XGBoost | 0.739 | 0.528 |
| Tuned Decision Tree | 0.722 | 0.483 |
| Decision Tree | 0.654 | 0.425 |

The Random Forest achieved both the highest ROC-AUC and the highest PR-AUC.

Because the positive class is clinically important and moderately underrepresented, PR-AUC was considered particularly informative. These results come from a small internal test set and should not be interpreted as proof of general superiority.

## Cross-Validation

Five-fold cross-validation was used within the training data for hyperparameter selection. Mean scores and their standard deviations should be interpreted together because the small cohort can produce noticeable variation between folds.

The held-out test set was not used to fit model parameters. However, the later threshold analysis was exploratory and did use the test set, so it does not represent an independently validated threshold selection.

## Exploratory Threshold Analysis

The default classification threshold was 0.50. Lower thresholds were explored to examine the trade-off between missed death events and false-positive alerts.

For the Random Forest:

| Threshold | Recall | Precision | False Negatives | False Positives |
|---:|---:|---:|---:|---:|
| 0.10 | 0.958 | 0.411 | 1 | 33 |
| 0.15 | 0.917 | 0.449 | 2 | 27 |
| 0.20 | 0.833 | 0.465 | 4 | 23 |
| 0.25 | 0.833 | 0.500 | 4 | 20 |
| 0.30 | 0.833 | 0.541 | 4 | 17 |
| 0.35 | 0.708 | 0.548 | 7 | 14 |
| 0.40 | 0.583 | 0.560 | 10 | 11 |
| 0.45 | 0.542 | 0.591 | 11 | 9 |
| 0.50 | 0.542 | 0.619 | 11 | 8 |
| 0.55 | 0.458 | 0.611 | 13 | 7 |

A threshold of 0.30 increased recall to 83.3% but generated 17 false-positive alerts. A threshold of 0.35 reduced false positives to 14 but also reduced recall to 70.8%. This illustrates the operational trade-off between missed outcomes and alert burden.

No threshold was selected as optimal. These thresholds were explored using the test set and were included only to illustrate the clinical trade-off. The default threshold of 0.50 remains the primary test-set evaluation.

A final threshold would need to be selected using separate validation data and predefined clinical workflow requirements.

## Main Findings

- Accuracy alone was not sufficient because a model predicting survival for every patient would already achieve approximately 68% accuracy.
- Adding the available baseline variables improved performance compared with using only four continuous predictors.
- Random Forest provided the most suitable preliminary overall balance at the default threshold.
- Decision Tree achieved the highest default-threshold recall, but with lower precision and accuracy.
- Random Forest achieved the highest ROC-AUC and PR-AUC.
- Lowering the Random Forest threshold increased recall but also increased the number of false-positive alerts.
- Removing follow-up time substantially reduced model performance, demonstrating why leakage prevention is essential.
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

Among the evaluated models, the Random Forest provided the most suitable preliminary overall performance for the defined screening objective. Nevertheless, at the default threshold it identified only 13 of 24 death events and missed 11. Its performance is insufficient for clinical use.

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
