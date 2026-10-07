"""Cost-sensitive XGBoost for churn prediction.

Uses the unscaled (raw) features from features.py with its stratified
70/15/15 train/validation/test split (random_state=42).

- Class imbalance: scale_pos_weight = neg/pos computed from y_train.
- Regularized model, early stopping on the VALIDATION set only.
- Lightweight tuning: a small grid, selected by validation PR-AUC.
- Threshold: searched 0.10..0.90 (step 0.01) on VALIDATION predictions,
  maximizing F2 (beta=2). The test set is scored once at the end.

Writes artifacts/xgboost.json, artifacts/xgboost_metrics.json,
artifacts/xgboost_probs.csv and results/xgboost.md.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

DATA = PROCESSED_DIR
ART = ARTIFACTS_DIR
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.01), 2)


def load(split):
    X = pd.read_csv(DATA / f"X_{split}_raw.csv")
    y = pd.read_csv(DATA / f"y_{split}.csv")["Churn"].astype(int)
    return X, y


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "f2": fbeta_score(y, pred, beta=2),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    print(f"Train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"scale_pos_weight = {neg}/{pos} = {spw:.4f}")

    base = dict(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.0,
        gamma=0.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Lightweight grid; every config is early-stopped and compared on validation only.
    grid = [dict(max_depth=d, min_child_weight=m) for d, m in itertools.product([2, 3, 4], [5, 10])]

    results = []
    best = None
    for g in grid:
        model = xgb.XGBClassifier(**base, **g)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pv = model.predict_proba(X_val)[:, 1]
        pt = model.predict_proba(X_train)[:, 1]
        row = {**g, "best_iteration": int(model.best_iteration),
               "train_auc": roc_auc_score(y_train, pt),
               "val_auc": roc_auc_score(y_val, pv),
               "val_pr_auc": average_precision_score(y_val, pv)}
        results.append(row)
        print(row)
        if best is None or row["val_pr_auc"] > best[1]["val_pr_auc"]:
            best = (model, row)

    model, sel = best
    params = {k: v for k, v in model.get_params().items()
              if k in {**base, "max_depth": 0, "min_child_weight": 0}}
    print(f"\nSelected: {sel}")

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    # Threshold search on validation only (F2).
    f2s = [fbeta_score(y_val, (p_val >= t).astype(int), beta=2) for t in THRESHOLDS]
    thr = float(THRESHOLDS[int(np.argmax(f2s))])
    print(f"Chosen threshold (val F2 max) = {thr:.2f}, val F2 = {max(f2s):.4f}")

    val_050, val_opt = metrics(y_val, p_val, 0.5), metrics(y_val, p_val, thr)

    # Test set: scored once.
    p_test = model.predict_proba(X_test)[:, 1]
    test_050, test_opt = metrics(y_test, p_test, 0.5), metrics(y_test, p_test, thr)
    auc = {"train": roc_auc_score(y_train, p_train),
           "val": roc_auc_score(y_val, p_val),
           "test": roc_auc_score(y_test, p_test)}
    print("val@0.50", val_050, "\nval@opt", val_opt)
    print("test@0.50", test_050, "\ntest@opt", test_opt)
    print("AUC", auc)

    ART.mkdir(exist_ok=True)
    model.save_model(ART / "xgboost.json")
    pd.concat([
        pd.DataFrame({"split": "val", "y_true": y_val.values, "proba": p_val}),
        pd.DataFrame({"split": "test", "y_true": y_test.values, "proba": p_test}),
    ]).to_csv(ART / "xgboost_probs.csv", index=False)

    out = {
        "model": "XGBClassifier",
        "xgboost_version": xgb.__version__,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "class_counts_train": {"neg": neg, "pos": pos},
        "scale_pos_weight": spw,
        "params": params,
        "best_iteration": sel["best_iteration"],
        "n_trees_used": sel["best_iteration"] + 1,
        "tuning_grid": results,
        "threshold_rule": "maximize F2 (beta=2) on validation, thresholds 0.10-0.90 step 0.01",
        "threshold": thr,
        "val_threshold_curve": {f"{t:.2f}": float(f) for t, f in zip(THRESHOLDS, f2s)},
        "roc_auc": auc,
        "val": {"threshold_0.50": val_050, "threshold_opt": val_opt},
        "test": {"threshold_0.50": test_050, "threshold_opt": test_opt},
    }
    (ART / "xgboost_metrics.json").write_text(json.dumps(out, indent=2, default=float))

    write_report(out, y_train, y_val, y_test, results)
    print("\nSaved artifacts/xgboost.json, artifacts/xgboost_metrics.json, "
          "artifacts/xgboost_probs.csv and results/xgboost.md")


def mrow(name, m):
    return (f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['f2']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
            f"{m['fp']} | {m['fn']} |")


def cm(m):
    return ("| | Pred No Churn | Pred Churn |\n|---|---:|---:|\n"
            f"| **Actual No Churn** | {m['tn']} (TN) | {m['fp']} (FP) |\n"
            f"| **Actual Churn** | {m['fn']} (FN) | {m['tp']} (TP) |")


HDR = ("| Set | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def write_report(o, y_train, y_val, y_test, grid):
    t5, to = o["test"]["threshold_0.50"], o["test"]["threshold_opt"]
    v5, vo = o["val"]["threshold_0.50"], o["val"]["threshold_opt"]
    thr, p, auc = o["threshold"], o["params"], o["roc_auc"]
    n_churn_test = t5["tp"] + t5["fn"]
    fn_drop = t5["fn"] - to["fn"]
    fn_pct = 100 * fn_drop / t5["fn"] if t5["fn"] else 0.0
    fp_add = to["fp"] - t5["fp"]

    def dist(name, y):
        return f"| {name} | {len(y)} | {int((y == 0).sum())} | {int((y == 1).sum())} | {100 * y.mean():.1f}% |"

    grid_rows = "\n".join(
        f"| {g['max_depth']} | {g['min_child_weight']} | {g['best_iteration']} | "
        f"{g['train_auc']:.4f} | {g['val_auc']:.4f} | {g['val_pr_auc']:.4f} |" for g in grid)

    def diff(a, b, fmt="{:+.4f}"):
        return fmt.format(b - a)

    md = f"""# XGBoost – Cost-Sensitive Churn Prediction

## Dataset

IBM Telco Customer Churn (`data/data.csv`): 7,043 customers and 21 columns (customer ID, 19 attributes and the target `Churn`, Yes/No). `eda.py` cleans it into `data/data_clean.csv` and `features.py` builds the model inputs in `data/processed/`. This model uses the unscaled `X_*_raw.csv` feature files (34 features) with `y_*.csv` (1 = churn).

## Class Distribution

| Split | Rows | No Churn (0) | Churn (1) | Churn rate |
|---|---:|---:|---:|---:|
{dist("Train", y_train)}
{dist("Validation", y_val)}
{dist("Test", y_test)}

Roughly 73.5% / 26.5%. The split is stratified 70/15/15 with `random_state=42`, so all three sets keep the same churn rate.

## Preprocessing

- **Missing values:** the only missing values are 11 blank `TotalCharges` entries, all for customers with `tenure == 0` (not yet billed). `eda.py` fills them with 0.
- **Duplicates:** there are no duplicate rows, so none are removed. `customerID` is dropped as an identifier.
- **Feature engineering** (`features.py`): "No internet service" / "No phone service" are collapsed to "No". Added `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`, `IsAutoPayment`, `IsNewCustomer` (tenure <= 6), `LivesAlone`, `FiberNoSupport` and `LogTotalCharges` (which replaces `TotalCharges`).
- **Encoding:** binary Yes/No and gender columns become 0/1; `InternetService`, `Contract` and `PaymentMethod` are one-hot encoded. That gives 34 numeric features.
- **No scaling:** gradient-boosted trees split on thresholds and are invariant to monotonic rescaling, so the raw (unscaled) features are used.
- **Leakage prevention:** the split is made before any fitting. Early stopping, configuration selection and threshold selection use only the validation set; the test set is scored once at the end.

## Class Imbalance Handling

Calculated from the training labels ({o['class_counts_train']['neg']} negatives, {o['class_counts_train']['pos']} positives):

    scale_pos_weight = {o['class_counts_train']['neg']} / {o['class_counts_train']['pos']} = {o['scale_pos_weight']:.4f}

XGBoost multiplies the gradient/hessian of every positive (churn) example by this weight, so the total loss contribution of churners equals that of non-churners. Without it the loss is dominated by the 73.5% majority class and the model under-predicts churn, which shows up as missed churners (false negatives). Weighting shifts predicted probabilities upward for churners and is the built-in cost-sensitive mechanism; the decision threshold is then tuned separately on validation data.

## Model

`xgboost.XGBClassifier` (xgboost {o['xgboost_version']}). The earlier baseline overfit (train AUC 0.919 vs test 0.842), so this version uses shallower trees, a lower learning rate, row/column subsampling, a minimum child weight, L2 regularization and early stopping on the validation set.

Final hyperparameters:

| Parameter | Value |
|---|---|
| objective | {p['objective']} |
| max_depth | {p['max_depth']} |
| learning_rate | {p['learning_rate']} |
| min_child_weight | {p['min_child_weight']} |
| subsample | {p['subsample']} |
| colsample_bytree | {p['colsample_bytree']} |
| reg_lambda (L2) | {p['reg_lambda']} |
| reg_alpha (L1) | {p['reg_alpha']} |
| gamma | {p['gamma']} |
| scale_pos_weight | {p['scale_pos_weight']:.4f} |
| n_estimators (max) | {p['n_estimators']} |
| early_stopping_rounds | {p['early_stopping_rounds']} (on validation) |
| eval_metric | {p['eval_metric']} |
| tree_method | {p['tree_method']} |
| random_state | {p['random_state']} |
| **best_iteration** | **{o['best_iteration']}** ({o['n_trees_used']} trees used) |

Lightweight tuning: six configurations (`max_depth` in {{2, 3, 4}} x `min_child_weight` in {{5, 10}}), each early-stopped on validation PR-AUC. The configuration with the highest validation PR-AUC was kept.

| max_depth | min_child_weight | best_iteration | Train ROC-AUC | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|---:|---:|---:|
{grid_rows}

Generalization of the selected model (ROC-AUC): train **{auc['train']:.4f}**, validation **{auc['val']:.4f}**, test **{auc['test']:.4f}** (train-test gap {auc['train'] - auc['test']:.4f}, versus 0.077 for the baseline).

## Baseline Evaluation

Threshold = 0.50 (class-weighted model, default cut-off).

{HDR}
{mrow("Validation", v5)}
{mrow("Test", t5)}

Test confusion matrix (threshold 0.50):

{cm(t5)}

## Threshold Optimization

- **Validation-set threshold search:** predicted churn probabilities on the validation set (n = {len(y_val)}) were thresholded at every value from 0.10 to 0.90 in steps of 0.01. The test set was not used.
- **Optimization criterion:** F2 score (F-beta with beta = 2), which weights recall four times as heavily as precision. This matches the business cost: a missed churner (FN) is a lost customer, while a false alarm (FP) only costs a retention offer. The same rule is used for the Logistic Regression and MLP models so the three are comparable.
- **Selected threshold:** **{thr:.2f}** (validation F2 = {vo['f2']:.4f}, versus {v5['f2']:.4f} at 0.50).

Validation metrics at both thresholds:

{HDR}
{mrow("Val @ 0.50", v5)}
{mrow(f"Val @ {thr:.2f}", vo)}

## Final Evaluation

The untouched test set (n = {len(y_test)}, {n_churn_test} churners), scored once with the selected threshold **{thr:.2f}**:

{HDR}
{mrow(f"Test @ {thr:.2f}", to)}

Test confusion matrix (threshold {thr:.2f}):

{cm(to)}

## False Negative Reduction

Test set, same model, threshold 0.50 vs the validation-selected threshold {thr:.2f}:

| | Threshold 0.50 | Optimized Threshold ({thr:.2f}) |
|---|---:|---:|
| Recall | {t5['recall']:.4f} | {to['recall']:.4f} |
| Precision | {t5['precision']:.4f} | {to['precision']:.4f} |
| F1 | {t5['f1']:.4f} | {to['f1']:.4f} |
| F2 | {t5['f2']:.4f} | {to['f2']:.4f} |
| False Negatives | {t5['fn']} | {to['fn']} |
| False Positives | {t5['fp']} | {to['fp']} |

- False negatives: {t5['fn']} -> {to['fn']} (**{fn_drop:+d}** churners caught, a {fn_pct:.1f}% reduction in missed churners).
- False positives: {t5['fp']} -> {to['fp']} ({fp_add:+d}).
- Recall changes by {diff(t5['recall'], to['recall'])}, precision by {diff(t5['precision'], to['precision'])}, F2 by {diff(t5['f2'], to['f2'])}.
- For reference, the unweighted baseline XGBoost at 0.50 had test recall 0.511 (on the earlier 80/20 split, so not directly comparable).

## Observations

- **Churn recall:** class weighting alone already lifts test recall to {t5['recall']:.3f} at threshold 0.50, compared with about 0.51 for the unweighted baseline. With the F2-optimized threshold, test recall is {to['recall']:.3f}: the model flags {to['tp']} of the {n_churn_test} test churners.
- **False-negative reduction:** moving from 0.50 to {thr:.2f} changes missed churners from {t5['fn']} to {to['fn']} ({fn_pct:.1f}% fewer). The threshold was chosen on validation data only, so this is an honest estimate of how the rule transfers to unseen customers.
- **Precision/recall tradeoff:** the extra recall costs precision ({t5['precision']:.3f} -> {to['precision']:.3f}) and {fp_add:+d} false positives. Accuracy moves from {t5['accuracy']:.3f} to {to['accuracy']:.3f}, which is expected and acceptable because accuracy is not the objective. Because `scale_pos_weight` already pushes probabilities upward, the F2-optimal threshold is low ({thr:.2f}) and the trade is steep: about {fp_add / max(fn_drop, 1):.1f} extra false positives per extra churner caught. Whether that is worth it depends on the FN:FP cost ratio; F2 implicitly assumes recall matters roughly four times as much as precision. If retention offers are expensive, a threshold nearer 0.50 (already recall {t5['recall']:.3f}) may be the better operating point.
- **Generalization:** train/val/test ROC-AUC is {auc['train']:.3f} / {auc['val']:.3f} / {auc['test']:.3f}. The train-test gap ({auc['train'] - auc['test']:.3f}) is much smaller than the baseline's 0.077, so the shallower trees, subsampling, regularization and early stopping (stopping at iteration {o['best_iteration']}) removed most of the overfitting. Validation AUC is slightly below test AUC; with about 280 churners per split, a difference of this size is within normal sampling variation between the two held-out sets. Test PR-AUC is {to['pr_auc']:.3f} (churn base rate 0.265).
- **Suitability for the cost-sensitive stage:** XGBoost is a good fit. It supports class weighting natively, its ranking quality (test ROC-AUC {auc['test']:.3f}) is in line with the baseline models on this dataset (0.84-0.85), the threshold can be moved freely to match the business cost ratio, and TreeExplainer gives exact, fast SHAP values for the explainability deliverable. Since the baseline models all ranked customers about equally well, the final choice between models should rest on the cost-based comparison on this shared split and on interpretability, not on AUC alone.
"""
    (RESULTS_DIR / "xgboost.md").write_text(md)


if __name__ == "__main__":
    main()
