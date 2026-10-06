# Logistic Regression – Cost-Sensitive Churn Prediction

All numbers below come from running `train_logistic_regression.py` (`venv/bin/python train_logistic_regression.py`).

## Dataset

- Source: IBM Telco Customer Churn (`data/data.csv`), 7043 customers, 21 raw columns.
- Target: `Churn` (Yes/No), encoded as 1 = churn, 0 = no churn.
- Model input: `data/processed/X_{train,val,test}_std.csv` and `y_{train,val,test}.csv`, produced by `eda.py` and `features.py`; 34 features.
- Stratified 70/15/15 train/validation/test split, `random_state=42`: train 4929, validation 1057, test 1057 rows.
- The validation set is used for choosing the regularization strength and the decision threshold. The test set is used once, for final evaluation.

## Class Distribution

| Split | Rows | No Churn (0) | Churn (1) |
|---|---:|---:|---:|
| Train | 4929 | 3621 (73.5%) | 1308 (26.5%) |
| Validation | 1057 | 776 (73.4%) | 281 (26.6%) |
| Test | 1057 | 777 (73.5%) | 280 (26.5%) |
| Total | 7043 | 5174 (73.5%) | 1869 (26.5%) |

## Preprocessing

- **Missing values** (`eda.py`): the only missing values are 11 blank `TotalCharges` entries, all with `tenure = 0` (customers not yet billed), filled with 0.
- **Duplicates**: none in the data, so nothing was removed. `customerID` is dropped as an identifier.
- **Feature engineering** (`features.py`): "No internet service" / "No phone service" collapsed to "No"; engineered `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`, `IsAutoPayment`, `IsNewCustomer` (tenure <= 6), `LivesAlone`, `FiberNoSupport`, and `LogTotalCharges` (replaces `TotalCharges`).
- **Encoding**: binary Yes/No columns and gender mapped to 0/1; `InternetService`, `Contract` and `PaymentMethod` one-hot encoded. 34 features in total (7 continuous, 27 binary).
- **Scaling**: `StandardScaler` on the 7 continuous features (`tenure`, `MonthlyCharges`, `LogTotalCharges`, `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`), **fit on the training set only** and applied to validation/test. Binary flags are left as 0/1.
- No preprocessing step, hyperparameter or threshold uses test-set information.

## Class Imbalance Handling

- **Original class distribution (train)**: 3621 non-churn (73.5%) vs 1308 churn (26.5%), about 2.77 non-churners per churner.
- **`class_weight="balanced"`**: each class is weighted by `n_samples / (n_classes * n_class_samples)`. On the training set this gives:
  - No Churn (0): 4929 / (2 x 3621) = **0.6806**
  - Churn (1): 4929 / (2 x 1308) = **1.8842**
  - A churner therefore counts **2.77x** as much as a non-churner in the log-loss, so both classes contribute equally in total.
- **Why it is necessary**: with ~73.5% / 26.5% classes, an unweighted model minimizes loss mostly on the majority class and pushes churn probabilities down. The earlier unweighted baseline caught only about half of the churners (recall 0.521). A false negative (a churner we do not contact) costs far more than a false positive (an unneeded retention offer), so the loss should not be dominated by the majority class. Weighting shifts the probabilities upward for churn-like customers; the threshold search then trades precision for recall explicitly.

## Model

- `sklearn.linear_model.LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42)`
- **Regularization**: L2 (the scikit-learn default). `C` was chosen from [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] by **validation PR-AUC** (threshold-independent; test not used):

| C | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|
| 0.001 | 0.8303 | 0.6297 |
| 0.01 | 0.8346 | 0.6362 |
| 0.1 | 0.8362 | 0.6415 |
| 1.0 (selected) | 0.8369 | 0.6447 |
| 10.0 | 0.8370 | 0.6441 |
| 100.0 | 0.8371 | 0.6447 |

- Train ROC-AUC 0.8546 vs test ROC-AUC 0.8573: little overfitting.
- Largest positive coefficients (raise churn odds): InternetService_Fiber_optic (+0.87), Contract_Month_to_month (+0.78), NumServices (+0.44), PaperlessBilling (+0.36), StreamingMovies (+0.22).
- Largest negative coefficients (lower churn odds): InternetService_No (-1.09), Contract_Two_year (-1.03), LogTotalCharges (-0.73), PhoneService (-0.43), TechSupport (-0.41).
- Saved to `artifacts/logistic_regression.joblib` (model, chosen threshold, C, feature names). Validation and test probabilities are in `artifacts/logistic_regression_probs.csv` (`split,y_true,proba`).

## Baseline Evaluation

Threshold = 0.50, **test set**:

| Metric | Score |
|---|---:|
| Accuracy | 0.7455 |
| Precision | 0.5124 |
| Recall | 0.8107 |
| F1-score | 0.6279 |
| F2-score | 0.7262 |
| ROC-AUC | 0.8573 |
| PR-AUC | 0.6788 |
| False Positives | 216 |
| False Negatives | 53 |

Confusion matrix (test, threshold 0.50):

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 561 (TN) | 216 (FP) |
| **Actual Churn** | 53 (FN) | 227 (TP) |

Validation set at threshold 0.50: accuracy 0.7512, precision 0.5216, recall 0.7722, F1 0.6227, F2 0.7045, ROC-AUC 0.8369, PR-AUC 0.6447, FP 199, FN 64.

## Threshold Optimization

- Probabilities were computed on the **validation set only**. Thresholds from 0.10 to 0.90 in steps of 0.01 (81 values) were evaluated.
- **Criterion**: maximize **F2-score** (beta = 2, recall weighted 4x as much as precision in the harmonic mean). This is the same rule used for the MLP and XGBoost models, so the three are comparable. F2 favors recall but still penalizes a collapse in precision, unlike maximizing recall alone (which would go to the lowest threshold).
- **Selected threshold: 0.45** (validation F2 = 0.7359, vs 0.7045 at 0.50).
- The test set was not looked at during this selection.

Validation sweep (selected rows):

| Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.3886 | 0.9253 | 0.5474 | 0.7250 | 409 | 21 |
| 0.25 | 0.4123 | 0.9039 | 0.5663 | 0.7299 | 362 | 27 |
| 0.30 | 0.4269 | 0.8826 | 0.5754 | 0.7273 | 333 | 33 |
| 0.35 | 0.4449 | 0.8612 | 0.5867 | 0.7254 | 302 | 39 |
| 0.40 | 0.4742 | 0.8505 | 0.6089 | 0.7340 | 265 | 42 |
| 0.45 (selected) | 0.5076 | 0.8292 | 0.6297 | 0.7359 | 226 | 48 |
| 0.50 | 0.5216 | 0.7722 | 0.6227 | 0.7045 | 199 | 64 |
| 0.60 | 0.5736 | 0.6797 | 0.6221 | 0.6555 | 142 | 90 |

Validation set at threshold 0.45: accuracy 0.7408, precision 0.5076, recall 0.8292, F1 0.6297, F2 0.7359, ROC-AUC 0.8369, PR-AUC 0.6447, FP 226, FN 48.

## Final Evaluation

Threshold = 0.45, **test set** (single final evaluation):

| Metric | Score |
|---|---:|
| Accuracy | 0.7294 |
| Precision | 0.4938 |
| Recall | 0.8571 |
| F1-score | 0.6266 |
| F2-score | 0.7472 |
| ROC-AUC | 0.8573 |
| PR-AUC | 0.6788 |
| False Positives | 246 |
| False Negatives | 40 |

Confusion matrix (test, threshold 0.45):

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 531 (TN) | 246 (FP) |
| **Actual Churn** | 40 (FN) | 240 (TP) |

ROC-AUC and PR-AUC do not depend on the threshold, so they are the same as in the baseline.

## False Negative Reduction

Test set (280 actual churners):

| | Threshold 0.50 | Optimized Threshold (0.45) |
|---|---:|---:|
| Recall | 0.8107 | 0.8571 |
| Precision | 0.5124 | 0.4938 |
| F1 | 0.6279 | 0.6266 |
| F2 | 0.7262 | 0.7472 |
| False Negatives | 53 | 40 |
| False Positives | 216 | 246 |

False negatives before threshold optimization:
53

False negatives after threshold optimization:
40

Reduction:
53 - 40 = **13** (24.5% fewer missed churners), at the cost of +30 false positives.

## Observations

- **Churn detection**: class weighting alone already moves recall at 0.50 to 0.811 on test (the unweighted baseline in project memory had 0.521 on its earlier 80/20 split). With the F2 threshold, the model catches 240 of 280 test churners (recall 0.857).
- **False-negative reduction**: missed churners drop from 53 to 40 (13 fewer).
- **Precision/recall tradeoff**: precision moves from 0.512 to 0.494 and false positives from 216 to 246 (+30). Accuracy moves from 0.746 to 0.729; accuracy is not the target metric here. F1 goes from 0.628 to 0.627, while F2 goes from 0.726 to 0.747.
- **Ranking quality** is unchanged by the threshold: test ROC-AUC 0.857, PR-AUC 0.679 (churn base rate 0.265).
- **Is the optimized threshold useful?** Yes. When a missed churner costs much more than a retention offer, trading the extra false positives for fewer missed churners is worthwhile, and test F2 improves, which shows the validation-chosen threshold generalizes. The final operating point should come from the cost-sensitive stage (FN cost vs FP cost) using the saved validation probabilities.
