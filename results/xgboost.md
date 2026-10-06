# XGBoost – Cost-Sensitive Churn Prediction

## Dataset

IBM Telco Customer Churn (`data/data.csv`): 7,043 customers and 21 columns (customer ID, 19 attributes and the target `Churn`, Yes/No). `eda.py` cleans it into `data/data_clean.csv` and `features.py` builds the model inputs in `data/processed/`. This model uses the unscaled `X_*_raw.csv` feature files (34 features) with `y_*.csv` (1 = churn).

## Class Distribution

| Split | Rows | No Churn (0) | Churn (1) | Churn rate |
|---|---:|---:|---:|---:|
| Train | 4929 | 3621 | 1308 | 26.5% |
| Validation | 1057 | 776 | 281 | 26.6% |
| Test | 1057 | 777 | 280 | 26.5% |

Roughly 73.5% / 26.5%. The split is stratified 70/15/15 with `random_state=42`, so all three sets keep the same churn rate.

## Preprocessing

- **Missing values:** the only missing values are 11 blank `TotalCharges` entries, all for customers with `tenure == 0` (not yet billed). `eda.py` fills them with 0.
- **Duplicates:** there are no duplicate rows, so none are removed. `customerID` is dropped as an identifier.
- **Feature engineering** (`features.py`): "No internet service" / "No phone service" are collapsed to "No". Added `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`, `IsAutoPayment`, `IsNewCustomer` (tenure <= 6), `LivesAlone`, `FiberNoSupport` and `LogTotalCharges` (which replaces `TotalCharges`).
- **Encoding:** binary Yes/No and gender columns become 0/1; `InternetService`, `Contract` and `PaymentMethod` are one-hot encoded. That gives 34 numeric features.
- **No scaling:** gradient-boosted trees split on thresholds and are invariant to monotonic rescaling, so the raw (unscaled) features are used.
- **Leakage prevention:** the split is made before any fitting. Early stopping, configuration selection and threshold selection use only the validation set; the test set is scored once at the end.

## Class Imbalance Handling

Calculated from the training labels (3621 negatives, 1308 positives):

    scale_pos_weight = 3621 / 1308 = 2.7683

XGBoost multiplies the gradient/hessian of every positive (churn) example by this weight, so the total loss contribution of churners equals that of non-churners. Without it the loss is dominated by the 73.5% majority class and the model under-predicts churn, which shows up as missed churners (false negatives). Weighting shifts predicted probabilities upward for churners and is the built-in cost-sensitive mechanism; the decision threshold is then tuned separately on validation data.

## Model

`xgboost.XGBClassifier` (xgboost 3.4.1). The earlier baseline overfit (train AUC 0.919 vs test 0.842), so this version uses shallower trees, a lower learning rate, row/column subsampling, a minimum child weight, L2 regularization and early stopping on the validation set.

Final hyperparameters:

| Parameter | Value |
|---|---|
| objective | binary:logistic |
| max_depth | 3 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda (L2) | 5.0 |
| reg_alpha (L1) | 0.0 |
| gamma | 0.0 |
| scale_pos_weight | 2.7683 |
| n_estimators (max) | 1000 |
| early_stopping_rounds | 50 (on validation) |
| eval_metric | aucpr |
| tree_method | hist |
| random_state | 42 |
| **best_iteration** | **112** (113 trees used) |

Lightweight tuning: six configurations (`max_depth` in {2, 3, 4} x `min_child_weight` in {5, 10}), each early-stopped on validation PR-AUC. The configuration with the highest validation PR-AUC was kept.

| max_depth | min_child_weight | best_iteration | Train ROC-AUC | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 215 | 0.8664 | 0.8319 | 0.6258 |
| 2 | 10 | 113 | 0.8588 | 0.8309 | 0.6239 |
| 3 | 5 | 112 | 0.8710 | 0.8313 | 0.6267 |
| 3 | 10 | 123 | 0.8724 | 0.8315 | 0.6248 |
| 4 | 5 | 176 | 0.8988 | 0.8285 | 0.6224 |
| 4 | 10 | 42 | 0.8659 | 0.8284 | 0.6206 |

Generalization of the selected model (ROC-AUC): train **0.8710**, validation **0.8313**, test **0.8546** (train-test gap 0.0164, versus 0.077 for the baseline).

## Baseline Evaluation

Threshold = 0.50 (class-weighted model, default cut-off).

| Set | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.7483 | 0.5176 | 0.7865 | 0.6243 | 0.7124 | 0.8313 | 0.6267 | 206 | 60 |
| Test | 0.7455 | 0.5121 | 0.8286 | 0.6330 | 0.7374 | 0.8546 | 0.6665 | 221 | 48 |

Test confusion matrix (threshold 0.50):

| | Pred No Churn | Pred Churn |
|---|---:|---:|
| **Actual No Churn** | 556 (TN) | 221 (FP) |
| **Actual Churn** | 48 (FN) | 232 (TP) |

## Threshold Optimization

- **Validation-set threshold search:** predicted churn probabilities on the validation set (n = 1057) were thresholded at every value from 0.10 to 0.90 in steps of 0.01. The test set was not used.
- **Optimization criterion:** F2 score (F-beta with beta = 2), which weights recall four times as heavily as precision. This matches the business cost: a missed churner (FN) is a lost customer, while a false alarm (FP) only costs a retention offer. The same rule is used for the Logistic Regression and MLP models so the three are comparable.
- **Selected threshold:** **0.19** (validation F2 = 0.7337, versus 0.7124 at 0.50).

Validation metrics at both thresholds:

| Set | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Val @ 0.50 | 0.7483 | 0.5176 | 0.7865 | 0.6243 | 0.7124 | 0.8313 | 0.6267 | 206 | 60 |
| Val @ 0.19 | 0.5904 | 0.3886 | 0.9431 | 0.5504 | 0.7337 | 0.8313 | 0.6267 | 417 | 16 |

## Final Evaluation

The untouched test set (n = 1057, 280 churners), scored once with the selected threshold **0.19**:

| Set | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test @ 0.19 | 0.5866 | 0.3877 | 0.9679 | 0.5536 | 0.7449 | 0.8546 | 0.6665 | 428 | 9 |

Test confusion matrix (threshold 0.19):

| | Pred No Churn | Pred Churn |
|---|---:|---:|
| **Actual No Churn** | 349 (TN) | 428 (FP) |
| **Actual Churn** | 9 (FN) | 271 (TP) |

## False Negative Reduction

Test set, same model, threshold 0.50 vs the validation-selected threshold 0.19:

| | Threshold 0.50 | Optimized Threshold (0.19) |
|---|---:|---:|
| Recall | 0.8286 | 0.9679 |
| Precision | 0.5121 | 0.3877 |
| F1 | 0.6330 | 0.5536 |
| F2 | 0.7374 | 0.7449 |
| False Negatives | 48 | 9 |
| False Positives | 221 | 428 |

- False negatives: 48 -> 9 (**+39** churners caught, a 81.2% reduction in missed churners).
- False positives: 221 -> 428 (+207).
- Recall changes by +0.1393, precision by -0.1244, F2 by +0.0075.
- For reference, the unweighted baseline XGBoost at 0.50 had test recall 0.511 (on the earlier 80/20 split, so not directly comparable).

## Observations

- **Churn recall:** class weighting alone already lifts test recall to 0.829 at threshold 0.50, compared with about 0.51 for the unweighted baseline. With the F2-optimized threshold, test recall is 0.968: the model flags 271 of the 280 test churners.
- **False-negative reduction:** moving from 0.50 to 0.19 changes missed churners from 48 to 9 (81.2% fewer). The threshold was chosen on validation data only, so this is an honest estimate of how the rule transfers to unseen customers.
- **Precision/recall tradeoff:** the extra recall costs precision (0.512 -> 0.388) and +207 false positives. Accuracy moves from 0.746 to 0.587, which is expected and acceptable because accuracy is not the objective. Because `scale_pos_weight` already pushes probabilities upward, the F2-optimal threshold is low (0.19) and the trade is steep: about 5.3 extra false positives per extra churner caught. Whether that is worth it depends on the FN:FP cost ratio; F2 implicitly assumes recall matters roughly four times as much as precision. If retention offers are expensive, a threshold nearer 0.50 (already recall 0.829) may be the better operating point.
- **Generalization:** train/val/test ROC-AUC is 0.871 / 0.831 / 0.855. The train-test gap (0.016) is much smaller than the baseline's 0.077, so the shallower trees, subsampling, regularization and early stopping (stopping at iteration 112) removed most of the overfitting. Validation AUC is slightly below test AUC; with about 280 churners per split, a difference of this size is within normal sampling variation between the two held-out sets. Test PR-AUC is 0.666 (churn base rate 0.265).
- **Suitability for the cost-sensitive stage:** XGBoost is a good fit. It supports class weighting natively, its ranking quality (test ROC-AUC 0.855) is in line with the baseline models on this dataset (0.84-0.85), the threshold can be moved freely to match the business cost ratio, and TreeExplainer gives exact, fast SHAP values for the explainability deliverable. Since the baseline models all ranked customers about equally well, the final choice between models should rest on the cost-based comparison on this shared split and on interpretability, not on AUC alone.
