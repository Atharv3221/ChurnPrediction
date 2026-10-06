"""Cost-sensitive Logistic Regression for churn prediction.

Uses the stratified 70/15/15 split from features.py (data/processed/*_std.csv:
StandardScaler fit on train only, applied to the 7 continuous features; binary
flags kept as 0/1).

Steps:
  1. Train LogisticRegression(class_weight="balanced") on train. The L2
     strength C is chosen on VALIDATION PR-AUC (test is never used).
  2. Evaluate at threshold 0.50 on validation.
  3. Search thresholds 0.10..0.90 (step 0.01) on VALIDATION predictions and
     pick the one that maximizes F2 (beta=2), the rule shared with the MLP and
     XGBoost agents.
  4. Evaluate ONCE on the untouched test set at 0.50 and at the chosen threshold.

Writes artifacts/logistic_regression.joblib, artifacts/logistic_regression_probs.csv
and results/logistic_regression.md.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
RANDOM_STATE = 42
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.01), 2)


def load(split):
    X = pd.read_csv(DATA / f"X_{split}_std.csv")
    y = pd.read_csv(DATA / f"y_{split}.csv")["Churn"].to_numpy()
    return X, y


def make_model(C):
    return LogisticRegression(
        C=C, class_weight="balanced", solver="lbfgs",
        max_iter=5000, random_state=RANDOM_STATE,
    )


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1-score": f1_score(y, pred),
        "F2-score": fbeta_score(y, pred, beta=2),
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "TN": int(tn), "TP": int(tp),
    }


def fmt(v):
    return str(v) if isinstance(v, (int, np.integer)) else f"{v:.4f}"


def metric_table(m):
    keys = ["Accuracy", "Precision", "Recall", "F1-score", "F2-score",
            "ROC-AUC", "PR-AUC", "False Positives", "False Negatives"]
    rows = ["| Metric | Score |", "|---|---:|"] + [f"| {k} | {fmt(m[k])} |" for k in keys]
    return "\n".join(rows)


def cm_table(m):
    return (
        "| | Predicted No Churn | Predicted Churn |\n|---|---:|---:|\n"
        f"| **Actual No Churn** | {m['TN']} (TN) | {m['False Positives']} (FP) |\n"
        f"| **Actual Churn** | {m['False Negatives']} (FN) | {m['TP']} (TP) |"
    )


def main():
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")

    # Class distribution and the effective weights of class_weight="balanced"
    counts = {s: np.bincount(y, minlength=2) for s, y in
              [("train", y_train), ("val", y_val), ("test", y_test)]}
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    print("Class counts:", {k: v.tolist() for k, v in counts.items()})
    print(f"Balanced weights: 0 -> {weights[0]:.4f}, 1 -> {weights[1]:.4f}")

    # Choose regularization strength on validation PR-AUC (no test use)
    c_results = []
    for C in C_GRID:
        p = make_model(C).fit(X_train, y_train).predict_proba(X_val)[:, 1]
        c_results.append((C, roc_auc_score(y_val, p), average_precision_score(y_val, p)))
        print(f"C={C:<7} val ROC-AUC={c_results[-1][1]:.4f} PR-AUC={c_results[-1][2]:.4f}")
    best_C = max(c_results, key=lambda r: r[2])[0]
    print("Selected C:", best_C)

    model = make_model(best_C).fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    # Threshold search on VALIDATION only, maximize F2
    sweep = pd.DataFrame([
        {"threshold": t, **{k: metrics(y_val, p_val, t)[k] for k in
                            ["Precision", "Recall", "F1-score", "F2-score",
                             "False Positives", "False Negatives"]}}
        for t in THRESHOLDS
    ])
    best_idx = sweep["F2-score"].idxmax()  # first max -> highest-precision tie
    best_thr = float(sweep.loc[best_idx, "threshold"])
    print(f"Selected threshold (val F2 max): {best_thr:.2f}")

    val_050, val_opt = metrics(y_val, p_val, 0.5), metrics(y_val, p_val, best_thr)

    # Final, single evaluation on the untouched test set
    p_test = model.predict_proba(X_test)[:, 1]
    test_050, test_opt = metrics(y_test, p_test, 0.5), metrics(y_test, p_test, best_thr)
    train_auc = roc_auc_score(y_train, p_train)
    for name, m in [("VAL@0.50", val_050), (f"VAL@{best_thr:.2f}", val_opt),
                    ("TEST@0.50", test_050), (f"TEST@{best_thr:.2f}", test_opt)]:
        print(name, {k: fmt(v) for k, v in m.items()})

    ART.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": best_thr, "C": best_C,
                 "feature_names": list(X_train.columns), "input": "_std",
                 "threshold_rule": "max F2 on validation, 0.10-0.90 step 0.01"},
                ART / "logistic_regression.joblib")
    pd.concat([
        pd.DataFrame({"split": "val", "y_true": y_val, "proba": p_val}),
        pd.DataFrame({"split": "test", "y_true": y_test, "proba": p_test}),
    ]).to_csv(ART / "logistic_regression_probs.csv", index=False)

    write_report(counts, weights, c_results, best_C, sweep, best_thr,
                 val_050, val_opt, test_050, test_opt, train_auc, model, X_train.columns)
    print("Wrote results/logistic_regression.md")


def write_report(counts, weights, c_results, best_C, sweep, thr,
                 val_050, val_opt, test_050, test_opt, train_auc, model, cols):
    tr, va, te = counts["train"], counts["val"], counts["test"]
    total = tr + va + te
    n_tr = tr.sum()
    fn_red = test_050["False Negatives"] - test_opt["False Negatives"]
    fp_inc = test_opt["False Positives"] - test_050["False Positives"]
    n_pos_test = te[1]
    coefs = pd.Series(model.coef_[0], index=cols).sort_values()

    def dist_row(name, c):
        return f"| {name} | {c.sum()} | {c[0]} ({c[0]/c.sum():.1%}) | {c[1]} ({c[1]/c.sum():.1%}) |"

    c_rows = "\n".join(f"| {C}{' (selected)' if C == best_C else ''} | {a:.4f} | {p:.4f} |"
                       for C, a, p in c_results)
    sweep_show = sweep[sweep["threshold"].isin(
        sorted(set(np.round([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, thr], 2))))]
    sweep_rows = "\n".join(
        f"| {r.threshold:.2f}{' (selected)' if abs(r.threshold - thr) < 1e-9 else ''} | "
        f"{r.Precision:.4f} | {r.Recall:.4f} | {r['F1-score']:.4f} | {r['F2-score']:.4f} | "
        f"{int(r['False Positives'])} | {int(r['False Negatives'])} |"
        for _, r in sweep_show.iterrows())

    def cmp_row(label, key):
        return f"| {label} | {fmt(test_050[key])} | {fmt(test_opt[key])} |"

    def val_line(m):
        return (f"accuracy {m['Accuracy']:.4f}, precision {m['Precision']:.4f}, "
                f"recall {m['Recall']:.4f}, F1 {m['F1-score']:.4f}, F2 {m['F2-score']:.4f}, "
                f"ROC-AUC {m['ROC-AUC']:.4f}, PR-AUC {m['PR-AUC']:.4f}, "
                f"FP {m['False Positives']}, FN {m['False Negatives']}")

    top_pos = ", ".join(f"{k} ({v:+.2f})" for k, v in coefs[::-1].head(5).items())
    top_neg = ", ".join(f"{k} ({v:+.2f})" for k, v in coefs.head(5).items())

    md = f"""# Logistic Regression – Cost-Sensitive Churn Prediction

All numbers below come from running `train_logistic_regression.py` (`venv/bin/python train_logistic_regression.py`).

## Dataset

- Source: IBM Telco Customer Churn (`data/data.csv`), {total.sum()} customers, 21 raw columns.
- Target: `Churn` (Yes/No), encoded as 1 = churn, 0 = no churn.
- Model input: `data/processed/X_{{train,val,test}}_std.csv` and `y_{{train,val,test}}.csv`, produced by `eda.py` and `features.py`; 34 features.
- Stratified 70/15/15 train/validation/test split, `random_state=42`: train {tr.sum()}, validation {va.sum()}, test {te.sum()} rows.
- The validation set is used for choosing the regularization strength and the decision threshold. The test set is used once, for final evaluation.

## Class Distribution

| Split | Rows | No Churn (0) | Churn (1) |
|---|---:|---:|---:|
{dist_row("Train", tr)}
{dist_row("Validation", va)}
{dist_row("Test", te)}
{dist_row("Total", total)}

## Preprocessing

- **Missing values** (`eda.py`): the only missing values are 11 blank `TotalCharges` entries, all with `tenure = 0` (customers not yet billed), filled with 0.
- **Duplicates**: none in the data, so nothing was removed. `customerID` is dropped as an identifier.
- **Feature engineering** (`features.py`): "No internet service" / "No phone service" collapsed to "No"; engineered `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`, `IsAutoPayment`, `IsNewCustomer` (tenure <= 6), `LivesAlone`, `FiberNoSupport`, and `LogTotalCharges` (replaces `TotalCharges`).
- **Encoding**: binary Yes/No columns and gender mapped to 0/1; `InternetService`, `Contract` and `PaymentMethod` one-hot encoded. 34 features in total (7 continuous, 27 binary).
- **Scaling**: `StandardScaler` on the 7 continuous features (`tenure`, `MonthlyCharges`, `LogTotalCharges`, `AvgMonthlyCharge`, `ChargeIncrease`, `NumAddonServices`, `NumServices`), **fit on the training set only** and applied to validation/test. Binary flags are left as 0/1.
- No preprocessing step, hyperparameter or threshold uses test-set information.

## Class Imbalance Handling

- **Original class distribution (train)**: {tr[0]} non-churn ({tr[0]/n_tr:.1%}) vs {tr[1]} churn ({tr[1]/n_tr:.1%}), about {tr[0]/tr[1]:.2f} non-churners per churner.
- **`class_weight="balanced"`**: each class is weighted by `n_samples / (n_classes * n_class_samples)`. On the training set this gives:
  - No Churn (0): {n_tr} / (2 x {tr[0]}) = **{weights[0]:.4f}**
  - Churn (1): {n_tr} / (2 x {tr[1]}) = **{weights[1]:.4f}**
  - A churner therefore counts **{weights[1]/weights[0]:.2f}x** as much as a non-churner in the log-loss, so both classes contribute equally in total.
- **Why it is necessary**: with ~73.5% / 26.5% classes, an unweighted model minimizes loss mostly on the majority class and pushes churn probabilities down. The earlier unweighted baseline caught only about half of the churners (recall 0.521). A false negative (a churner we do not contact) costs far more than a false positive (an unneeded retention offer), so the loss should not be dominated by the majority class. Weighting shifts the probabilities upward for churn-like customers; the threshold search then trades precision for recall explicitly.

## Model

- `sklearn.linear_model.LogisticRegression(C={best_C}, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state={RANDOM_STATE})`
- **Regularization**: L2 (the scikit-learn default). `C` was chosen from {C_GRID} by **validation PR-AUC** (threshold-independent; test not used):

| C | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|
{c_rows}

- Train ROC-AUC {train_auc:.4f} vs test ROC-AUC {test_050['ROC-AUC']:.4f}: little overfitting.
- Largest positive coefficients (raise churn odds): {top_pos}.
- Largest negative coefficients (lower churn odds): {top_neg}.
- Saved to `artifacts/logistic_regression.joblib` (model, chosen threshold, C, feature names). Validation and test probabilities are in `artifacts/logistic_regression_probs.csv` (`split,y_true,proba`).

## Baseline Evaluation

Threshold = 0.50, **test set**:

{metric_table(test_050)}

Confusion matrix (test, threshold 0.50):

{cm_table(test_050)}

Validation set at threshold 0.50: {val_line(val_050)}.

## Threshold Optimization

- Probabilities were computed on the **validation set only**. Thresholds from 0.10 to 0.90 in steps of 0.01 ({len(sweep)} values) were evaluated.
- **Criterion**: maximize **F2-score** (beta = 2, recall weighted 4x as much as precision in the harmonic mean). This is the same rule used for the MLP and XGBoost models, so the three are comparable. F2 favors recall but still penalizes a collapse in precision, unlike maximizing recall alone (which would go to the lowest threshold).
- **Selected threshold: {thr:.2f}** (validation F2 = {val_opt['F2-score']:.4f}, vs {val_050['F2-score']:.4f} at 0.50).
- The test set was not looked at during this selection.

Validation sweep (selected rows):

| Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
{sweep_rows}

Validation set at threshold {thr:.2f}: {val_line(val_opt)}.

## Final Evaluation

Threshold = {thr:.2f}, **test set** (single final evaluation):

{metric_table(test_opt)}

Confusion matrix (test, threshold {thr:.2f}):

{cm_table(test_opt)}

ROC-AUC and PR-AUC do not depend on the threshold, so they are the same as in the baseline.

## False Negative Reduction

Test set ({n_pos_test} actual churners):

| | Threshold 0.50 | Optimized Threshold ({thr:.2f}) |
|---|---:|---:|
{cmp_row("Recall", "Recall")}
{cmp_row("Precision", "Precision")}
{cmp_row("F1", "F1-score")}
{cmp_row("F2", "F2-score")}
{cmp_row("False Negatives", "False Negatives")}
{cmp_row("False Positives", "False Positives")}

False negatives before threshold optimization:
{test_050['False Negatives']}

False negatives after threshold optimization:
{test_opt['False Negatives']}

Reduction:
{test_050['False Negatives']} - {test_opt['False Negatives']} = **{fn_red}** ({fn_red / max(test_050['False Negatives'], 1):.1%} fewer missed churners), at the cost of {fp_inc:+d} false positives.

## Observations

- **Churn detection**: class weighting alone already moves recall at 0.50 to {test_050['Recall']:.3f} on test (the unweighted baseline in project memory had 0.521 on its earlier 80/20 split). With the F2 threshold, the model catches {test_opt['TP']} of {n_pos_test} test churners (recall {test_opt['Recall']:.3f}).
- **False-negative reduction**: missed churners drop from {test_050['False Negatives']} to {test_opt['False Negatives']} ({fn_red} fewer).
- **Precision/recall tradeoff**: precision moves from {test_050['Precision']:.3f} to {test_opt['Precision']:.3f} and false positives from {test_050['False Positives']} to {test_opt['False Positives']} ({fp_inc:+d}). Accuracy moves from {test_050['Accuracy']:.3f} to {test_opt['Accuracy']:.3f}; accuracy is not the target metric here. F1 goes from {test_050['F1-score']:.3f} to {test_opt['F1-score']:.3f}, while F2 goes from {test_050['F2-score']:.3f} to {test_opt['F2-score']:.3f}.
- **Ranking quality** is unchanged by the threshold: test ROC-AUC {test_opt['ROC-AUC']:.3f}, PR-AUC {test_opt['PR-AUC']:.3f} (churn base rate {n_pos_test / te.sum():.3f}).
- **Is the optimized threshold useful?** {"Yes. When a missed churner costs much more than a retention offer, trading the extra false positives for fewer missed churners is worthwhile, and test F2 improves, which shows the validation-chosen threshold generalizes." if test_opt['F2-score'] > test_050['F2-score'] else "The F2 gain seen on validation did not carry over to test, so the threshold should be revisited with the explicit cost matrix."} The final operating point should come from the cost-sensitive stage (FN cost vs FP cost) using the saved validation probabilities.
"""
    (ROOT / "results" / "logistic_regression.md").write_text(md)


if __name__ == "__main__":
    main()
