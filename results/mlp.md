# MLP – Cost-Sensitive Churn Prediction

All numbers below come from running `train_mlp.py` (scikit-learn 1.9.1).

## Dataset

- IBM Telco Customer Churn: `data/data.csv`, with 7043 customers and 21 raw columns. The target is `Churn` (1 = churned).
- The model reads the prepared features in `data/processed/X_{train,val,test}_std.csv` and `y_{train,val,test}.csv`, which `eda.py` and `features.py` produce.
- There are 34 model features after feature engineering and encoding.

## Class Distribution

| Split | Rows | Churn (1) | No churn (0) | Churn rate |
|---|---:|---:|---:|---:|
| Train | 4929 | 1308 | 3621 | 26.5% |
| Val | 1057 | 281 | 776 | 26.6% |
| Test | 1057 | 280 | 777 | 26.5% |
| **Total** | 7043 | 1869 | 5174 | 26.5% |

About 73.5% of customers stay and 26.5% churn. The split is stratified, so every split keeps the same ratio.

## Preprocessing

- **Missing values:** `eda.py` filled the only 11 missing values, which are blank `TotalCharges` entries. All of them have `tenure = 0`, so they were set to 0 because these customers have not been billed yet.
- **Duplicates:** The data has no duplicate rows, so nothing was removed.
- **Feature engineering** (`features.py`): "No internet service" and "No phone service" are collapsed to "No". The new features are AvgMonthlyCharge, ChargeIncrease, NumAddonServices, NumServices, IsAutoPayment, IsNewCustomer (tenure <= 6), LivesAlone, FiberNoSupport and LogTotalCharges, which replaces TotalCharges. `customerID` is dropped.
- **Encoding:** Binary Yes/No and gender columns become 0/1. InternetService, Contract and PaymentMethod are one-hot encoded. The result is 34 features.
- **Split:** A stratified 70/15/15 train/validation/test split with `random_state=42` gives 4929/1057/1057 rows.
- **Scaling:** A StandardScaler is applied to the 7 continuous features (tenure, MonthlyCharges, LogTotalCharges, AvgMonthlyCharge, ChargeIncrease, NumAddonServices, NumServices). It is **fit on the training set only** and then applied to the validation and test sets. Binary flags stay 0/1.

## Class Imbalance Handling

- **Method used: Option A, balanced sample weights.** `inspect.signature(MLPClassifier.fit)` in the installed scikit-learn 1.9.1 is `(self, X, y, sample_weight=None)`, so per-sample weights are supported.
- The weights come from `compute_sample_weight("balanced", y_train)`, which gives n_samples / (2 * n_class). That is **0.6806** for each non-churner and **1.8842** for each churner, so one churner counts about 2.77 times as much as one non-churner in the loss.
- The weights are passed only to `fit()` on the **training set**. **No oversampling** was done, and no rows were duplicated in any split. The validation and test sets were not changed in any way.
- `early_stopping=True` holds out its own internal 10% slice of the **training** data to decide when to stop. The shared validation set is never used for training or early stopping. It is used only for the threshold search.

## Model

`sklearn.neural_network.MLPClassifier`

| Hyperparameter | Value |
|---|---|
| hidden_layer_sizes | (32, 16) (two hidden layers: 34 -> 32 -> 16 -> 1) |
| activation | relu (output: logistic) |
| solver | adam |
| learning_rate / learning_rate_init | constant / 0.001 |
| alpha (L2) | 0.001 |
| batch_size | 64 |
| max_iter (epochs) | 500 |
| early_stopping | True (validation_fraction=0.1 of train, n_iter_no_change=20) |
| epochs actually run | 36 |
| random_state | 42 |
| sample_weight | balanced (class 0: 0.6806, class 1: 1.8842) |

Train ROC-AUC is 0.8666, compared with 0.8541 on test.

## Baseline Evaluation

Threshold = 0.50, with the model trained using balanced sample weights.

### Validation (threshold 0.50)

| Metric | Value |
|---|---:|
| Accuracy | 0.7616 |
| Precision | 0.5358 |
| Recall | 0.7722 |
| F1 | 0.6327 |
| F2 | 0.7096 |
| ROC-AUC | 0.8264 |
| PR-AUC (average precision) | 0.6293 |
| False Positives | 188 |
| False Negatives | 64 |

| | Predicted No churn | Predicted Churn |
|---|---:|---:|
| **Actual No churn** | TN = 588 | FP = 188 |
| **Actual Churn** | FN = 64 | TP = 217 |

### Test (threshold 0.50)

| Metric | Value |
|---|---:|
| Accuracy | 0.7597 |
| Precision | 0.5301 |
| Recall | 0.8179 |
| F1 | 0.6433 |
| F2 | 0.7378 |
| ROC-AUC | 0.8541 |
| PR-AUC (average precision) | 0.6564 |
| False Positives | 203 |
| False Negatives | 51 |

| | Predicted No churn | Predicted Churn |
|---|---:|---:|
| **Actual No churn** | TN = 574 | FP = 203 |
| **Actual Churn** | FN = 51 | TP = 229 |

## Threshold Optimization

- **Search:** Thresholds from 0.10 to 0.90 in steps of 0.01 (81 values) were tried on the **validation set only**. The test set was not used.
- **Criterion:** The threshold with the highest **F2 score** (beta = 2) was chosen. F2 weights recall 4 times as heavily as precision, which matches the goal that a missed churner costs much more than a false alarm. This is the same rule used for the Logistic Regression and XGBoost models.
- **Selected threshold: 0.48**, with validation F2 = 0.7331 (F2 at 0.50 is 0.7096).

Selected rows from the validation search:

| Threshold | Precision | Recall | F1 | F2 | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.3413 | 0.9609 | 0.5037 | 0.7050 | 11 | 521 |
| 0.20 | 0.3889 | 0.9217 | 0.5470 | 0.7235 | 22 | 407 |
| 0.30 | 0.4286 | 0.8754 | 0.5754 | 0.7244 | 35 | 328 |
| 0.40 | 0.4651 | 0.8292 | 0.5959 | 0.7169 | 48 | 268 |
| **0.48** | 0.5290 | 0.8114 | 0.6404 | 0.7331 | 53 | 203 |
| 0.50 | 0.5358 | 0.7722 | 0.6327 | 0.7096 | 64 | 188 |
| 0.60 | 0.5687 | 0.6477 | 0.6057 | 0.6302 | 99 | 138 |
| 0.70 | 0.6391 | 0.5231 | 0.5753 | 0.5428 | 134 | 83 |

### Validation (threshold 0.48)

| Metric | Value |
|---|---:|
| Accuracy | 0.7578 |
| Precision | 0.5290 |
| Recall | 0.8114 |
| F1 | 0.6404 |
| F2 | 0.7331 |
| ROC-AUC | 0.8264 |
| PR-AUC (average precision) | 0.6293 |
| False Positives | 203 |
| False Negatives | 53 |

| | Predicted No churn | Predicted Churn |
|---|---:|---:|
| **Actual No churn** | TN = 573 | FP = 203 |
| **Actual Churn** | FN = 53 | TP = 228 |

On validation, false negatives went from 64 to 53, a drop of 11.

## Final Evaluation

The test set was not used for any choice. It was scored once with the selected threshold of 0.48.

| Metric | Value |
|---|---:|
| Accuracy | 0.7540 |
| Precision | 0.5217 |
| Recall | 0.8571 |
| F1 | 0.6486 |
| F2 | 0.7595 |
| ROC-AUC | 0.8541 |
| PR-AUC (average precision) | 0.6564 |
| False Positives | 220 |
| False Negatives | 40 |

| | Predicted No churn | Predicted Churn |
|---|---:|---:|
| **Actual No churn** | TN = 557 | FP = 220 |
| **Actual Churn** | FN = 40 | TP = 240 |

## False Negative Reduction

Results on the test set (1057 customers, 280 churners):

| | Threshold 0.50 | Optimized Threshold (0.48) |
|---|---:|---:|
| Recall | 0.8179 | 0.8571 |
| Precision | 0.5301 | 0.5217 |
| F1 | 0.6433 | 0.6486 |
| F2 | 0.7378 | 0.7595 |
| False Negatives | 51 | 40 |
| False Positives | 203 | 220 |
| Accuracy | 0.7597 | 0.7540 |
| ROC-AUC | 0.8541 | 0.8541 |
| PR-AUC | 0.6564 | 0.6564 |

**False negatives fell from 51 to 40, which is 11 fewer missed churners (21.6% fewer).** The cost was +17 false positives (from 203 to 220). ROC-AUC and PR-AUC do not depend on the threshold, so they are the same in both columns.

## Observations

- **Churn recall:** With balanced sample weights, the MLP already catches 81.8% of test churners at the default 0.50 threshold (229 of 280). At the threshold of 0.48 chosen on validation, it catches 85.7% (240 of 280). The earlier unweighted MLP (32,16) reached only 0.529 recall at 0.50 on the older 80/20 split (see `.claude/memory.md`). That is a different split, so the comparison is only indicative, but the gap is large.
- **False-negative reduction:** Moving from 0.50 to 0.48 removed 11 missed churners on test (51 -> 40, 21.6% fewer). Validation showed the same direction (64 -> 53). The selected threshold is only slightly below 0.50. Balanced weighting already moves the predicted probabilities upward, so most of the gain in recall comes from the weighting and the threshold adds a smaller second step.
- **Precision/recall tradeoff:** The 11 fewer false negatives cost 17 more false positives (203 -> 220). Precision fell from 0.530 to 0.522, and accuracy fell from 0.760 to 0.754. F1 (0.643 -> 0.649) and F2 (0.738 -> 0.759) both went up. Roughly half of the customers flagged as churners do churn. That is acceptable when a retention offer costs much less than a lost customer.
- **Impact of imbalance handling:** Weighting churners about 2.77 times as heavily changes where the model sets its operating point, not how well it ranks customers. Test ROC-AUC is 0.854 and PR-AUC is 0.656, both well above the 0.265 PR-AUC of a random model. The lower accuracy compared with the unweighted baseline is expected and does not mean the model got worse.
- **Generalization:** Train ROC-AUC is 0.867, compared with 0.826 on validation and 0.854 on test. The small network, L2 penalty and early stopping (stopped after 36 epochs) keep overfitting low. The validation AUC is a little lower than the test AUC, which reflects normal variation between two samples of about 1,050 customers each.
- **Suitability for the cost-sensitive stage:** The MLP fits this stage. It supports sample weights directly, its probabilities give a usable threshold, and it reaches high churn recall with a PR-AUC in a competitive range. Its weaknesses are that it is harder to explain than Logistic Regression and that results vary somewhat with `random_state` because of the random weight initialization. For SHAP it needs KernelExplainer or DeepExplainer instead of fast exact explainers. It is a reasonable candidate, but it should be chosen over the Logistic Regression and XGBoost models only if it clearly beats them on test recall/F2 and PR-AUC under the same threshold rule.
