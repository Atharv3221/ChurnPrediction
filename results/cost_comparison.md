# Cost-Sensitive Model Comparison

Costs: C_FN = 500 (missed churner), C_FP = 100 (unneeded retention offer). These are placeholders; rerun
the per-model `cost_<model>.py --c-fn N --c-fp M` scripts and then this one with the same flags for real business costs.

Each model's threshold was chosen on the validation set only (minimum validation cost). The model is
**selected by validation cost**; the test set is used once for the final report.

## Results (sorted by validation cost)

| Model | Threshold | Val cost | Test cost | Test cost/customer | Test FN | Test FP | Test recall |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.45 | 46,600 | 44,600 | 42.19 | 40 | 246 | 0.857 |
| MLP | 0.48 | 46,800 | 42,000 | 39.74 | 40 | 220 | 0.857 |
| XGBoost | 0.44 | 48,700 | 45,000 | 42.57 | 39 | 255 | 0.861 |

Test reference baselines: target nobody = 140,000, target everyone = 77,700.

## Selected model

**Logistic Regression** at threshold 0.45 (lowest validation cost).
Test cost 44,600 (42.19 per customer), a
68.1% saving over targeting nobody.

## Are the differences real? (paired bootstrap on test, 5000 resamples)

| Comparison | Observed diff | 95% CI low | 95% CI high | Significant |
|---|---|---|---|---|
| Logistic Regression - MLP | 2,600 | -1,000 | 6,200 | False |
| Logistic Regression - XGBoost | -400 | -3,600 | 2,800 | False |
| MLP - XGBoost | -3,000 | -6,700 | 700 | False |

A difference is significant only if its 95% CI excludes 0.
