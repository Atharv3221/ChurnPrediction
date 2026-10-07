"""Cost-sensitive MLP for churn prediction.

Uses the standardized features from features.py (stratified 70/15/15 split,
random_state=42, scalers fit on train only). Class imbalance is handled with
balanced sample weights passed to MLPClassifier.fit (supported in the installed
scikit-learn). The decision threshold is chosen on the VALIDATION set by
maximizing F2 over 0.10..0.90 (step 0.01); the test set is used only once for
the final report. Writes artifacts/mlp.joblib, artifacts/mlp_probs.csv, results/mlp.md.
"""
import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

DATA = PROCESSED_DIR
ART = ARTIFACTS_DIR
RANDOM_STATE = 42

PARAMS = dict(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-3,                 # L2 penalty
    batch_size=64,
    learning_rate="constant",
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,    # internal split of TRAIN only (not the shared val set)
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.01), 2)


def load(split):
    X = pd.read_csv(DATA / f"X_{split}_std.csv")
    y = pd.read_csv(DATA / f"y_{split}.csv")["Churn"].to_numpy()
    return X, y


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=thr,
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred),
        f1=f1_score(y, pred),
        f2=fbeta_score(y, pred, beta=2),
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def show(name, m):
    print(f"{name:22} thr={m['threshold']:.2f} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f} f2={m['f2']:.3f} auc={m['roc_auc']:.3f} "
          f"prauc={m['pr_auc']:.3f} TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")


def metric_table(m):
    return "\n".join([
        "| Metric | Value |", "|---|---:|",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        f"| F1 | {m['f1']:.4f} |",
        f"| F2 | {m['f2']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| PR-AUC (average precision) | {m['pr_auc']:.4f} |",
        f"| False Positives | {m['fp']} |",
        f"| False Negatives | {m['fn']} |",
    ])


def cm_table(m):
    return "\n".join([
        "| | Predicted No churn | Predicted Churn |", "|---|---:|---:|",
        f"| **Actual No churn** | TN = {m['tn']} | FP = {m['fp']} |",
        f"| **Actual Churn** | FN = {m['fn']} | TP = {m['tp']} |",
    ])


def main():
    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    X_te, y_te = load("test")

    # Imbalance handling: Option A (sample weights) if fit() supports it.
    supports_sw = "sample_weight" in inspect.signature(MLPClassifier.fit).parameters
    print(f"scikit-learn {sklearn.__version__}; MLPClassifier.fit accepts sample_weight: {supports_sw}")
    if not supports_sw:
        raise SystemExit("sample_weight not supported; switch to training-only oversampling")
    sw = compute_sample_weight("balanced", y_tr)
    w0, w1 = float(sw[y_tr == 0][0]), float(sw[y_tr == 1][0])
    print(f"balanced weights: class0={w0:.4f} class1={w1:.4f}")

    model = MLPClassifier(**PARAMS).fit(X_tr, y_tr, sample_weight=sw)
    print(f"stopped after {model.n_iter_} epochs; best internal val score {model.best_validation_score_:.4f}")

    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # Threshold search on VALIDATION only.
    search = [metrics(y_va, p_va, t) for t in THRESHOLDS]
    best = max(search, key=lambda m: m["f2"])  # first max on ties (lowest threshold)
    thr = float(best["threshold"])

    tr_auc = roc_auc_score(y_tr, p_tr)
    val05, valopt = metrics(y_va, p_va, 0.50), metrics(y_va, p_va, thr)
    test05, testopt = metrics(y_te, p_te, 0.50), metrics(y_te, p_te, thr)
    show("val @0.50", val05); show("val @opt", valopt)
    show("test @0.50", test05); show("test @opt", testopt)
    print(f"train ROC-AUC {tr_auc:.4f}")

    ART.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "params": PARAMS,
                 "imbalance": "balanced sample_weight", "features": list(X_tr.columns)},
                ART / "mlp.joblib")
    pd.concat([
        pd.DataFrame({"split": "val", "y_true": y_va, "proba": p_va}),
        pd.DataFrame({"split": "test", "y_true": y_te, "proba": p_te}),
    ]).to_csv(ART / "mlp_probs.csv", index=False)

    # Threshold-search excerpt for the report.
    picks = sorted({0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, round(thr, 2)})
    rows = [m for m in search if round(m["threshold"], 2) in picks]
    search_tbl = "\n".join(
        ["| Threshold | Precision | Recall | F1 | F2 | FN | FP |", "|---:|---:|---:|---:|---:|---:|---:|"]
        + [f"| {'**' if m['threshold'] == thr else ''}{m['threshold']:.2f}{'**' if m['threshold'] == thr else ''} "
           f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['f2']:.4f} | {m['fn']} | {m['fp']} |"
           for m in rows])

    fn_red = test05["fn"] - testopt["fn"]
    fn_pct = 100 * fn_red / test05["fn"] if test05["fn"] else 0.0
    fp_inc = testopt["fp"] - test05["fp"]
    val_fn_red = val05["fn"] - valopt["fn"]

    def row(label, key, fmt):
        return f"| {label} | {fmt.format(test05[key])} | {fmt.format(testopt[key])} |"

    n = {s: len(y) for s, y in [("train", y_tr), ("val", y_va), ("test", y_te)]}
    pos = {s: int(y.sum()) for s, y in [("train", y_tr), ("val", y_va), ("test", y_te)]}
    tot, totpos = sum(n.values()), sum(pos.values())

    md = f"""# MLP – Cost-Sensitive Churn Prediction

All numbers below come from running `train_mlp.py` (scikit-learn {sklearn.__version__}).

## Dataset

- IBM Telco Customer Churn: `data/data.csv`, with {tot} customers and 21 raw columns. The target is `Churn` (1 = churned).
- The model reads the prepared features in `data/processed/X_{{train,val,test}}_std.csv` and `y_{{train,val,test}}.csv`, which `eda.py` and `features.py` produce.
- There are {X_tr.shape[1]} model features after feature engineering and encoding.

## Class Distribution

| Split | Rows | Churn (1) | No churn (0) | Churn rate |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {s.capitalize()} | {n[s]} | {pos[s]} | {n[s] - pos[s]} | {pos[s] / n[s]:.1%} |" for s in n
    ) + f"""
| **Total** | {tot} | {totpos} | {tot - totpos} | {totpos / tot:.1%} |

About 73.5% of customers stay and 26.5% churn. The split is stratified, so every split keeps the same ratio.

## Preprocessing

- **Missing values:** `eda.py` filled the only 11 missing values, which are blank `TotalCharges` entries. All of them have `tenure = 0`, so they were set to 0 because these customers have not been billed yet.
- **Duplicates:** The data has no duplicate rows, so nothing was removed.
- **Feature engineering** (`features.py`): "No internet service" and "No phone service" are collapsed to "No". The new features are AvgMonthlyCharge, ChargeIncrease, NumAddonServices, NumServices, IsAutoPayment, IsNewCustomer (tenure <= 6), LivesAlone, FiberNoSupport and LogTotalCharges, which replaces TotalCharges. `customerID` is dropped.
- **Encoding:** Binary Yes/No and gender columns become 0/1. InternetService, Contract and PaymentMethod are one-hot encoded. The result is {X_tr.shape[1]} features.
- **Split:** A stratified 70/15/15 train/validation/test split with `random_state=42` gives {n['train']}/{n['val']}/{n['test']} rows.
- **Scaling:** A StandardScaler is applied to the 7 continuous features (tenure, MonthlyCharges, LogTotalCharges, AvgMonthlyCharge, ChargeIncrease, NumAddonServices, NumServices). It is **fit on the training set only** and then applied to the validation and test sets. Binary flags stay 0/1.

## Class Imbalance Handling

- **Method used: Option A, balanced sample weights.** `inspect.signature(MLPClassifier.fit)` in the installed scikit-learn {sklearn.__version__} is `{inspect.signature(MLPClassifier.fit)}`, so per-sample weights are supported.
- The weights come from `compute_sample_weight("balanced", y_train)`, which gives n_samples / (2 * n_class). That is **{w0:.4f}** for each non-churner and **{w1:.4f}** for each churner, so one churner counts about {w1 / w0:.2f} times as much as one non-churner in the loss.
- The weights are passed only to `fit()` on the **training set**. **No oversampling** was done, and no rows were duplicated in any split. The validation and test sets were not changed in any way.
- `early_stopping=True` holds out its own internal {PARAMS['validation_fraction']:.0%} slice of the **training** data to decide when to stop. The shared validation set is never used for training or early stopping. It is used only for the threshold search.

## Model

`sklearn.neural_network.MLPClassifier`

| Hyperparameter | Value |
|---|---|
| hidden_layer_sizes | {PARAMS['hidden_layer_sizes']} (two hidden layers: {X_tr.shape[1]} -> 32 -> 16 -> 1) |
| activation | {PARAMS['activation']} (output: logistic) |
| solver | {PARAMS['solver']} |
| learning_rate / learning_rate_init | {PARAMS['learning_rate']} / {PARAMS['learning_rate_init']} |
| alpha (L2) | {PARAMS['alpha']} |
| batch_size | {PARAMS['batch_size']} |
| max_iter (epochs) | {PARAMS['max_iter']} |
| early_stopping | {PARAMS['early_stopping']} (validation_fraction={PARAMS['validation_fraction']} of train, n_iter_no_change={PARAMS['n_iter_no_change']}) |
| epochs actually run | {model.n_iter_} |
| random_state | {PARAMS['random_state']} |
| sample_weight | balanced (class 0: {w0:.4f}, class 1: {w1:.4f}) |

Train ROC-AUC is {tr_auc:.4f}, compared with {test05['roc_auc']:.4f} on test.

## Baseline Evaluation

Threshold = 0.50, with the model trained using balanced sample weights.

### Validation (threshold 0.50)

{metric_table(val05)}

{cm_table(val05)}

### Test (threshold 0.50)

{metric_table(test05)}

{cm_table(test05)}

## Threshold Optimization

- **Search:** Thresholds from 0.10 to 0.90 in steps of 0.01 ({len(THRESHOLDS)} values) were tried on the **validation set only**. The test set was not used.
- **Criterion:** The threshold with the highest **F2 score** (beta = 2) was chosen. F2 weights recall 4 times as heavily as precision, which matches the goal that a missed churner costs much more than a false alarm. This is the same rule used for the Logistic Regression and XGBoost models.
- **Selected threshold: {thr:.2f}**, with validation F2 = {valopt['f2']:.4f} (F2 at 0.50 is {val05['f2']:.4f}).

Selected rows from the validation search:

{search_tbl}

### Validation (threshold {thr:.2f})

{metric_table(valopt)}

{cm_table(valopt)}

On validation, false negatives went from {val05['fn']} to {valopt['fn']}, a drop of {val_fn_red}.

## Final Evaluation

The test set was not used for any choice. It was scored once with the selected threshold of {thr:.2f}.

{metric_table(testopt)}

{cm_table(testopt)}

## False Negative Reduction

Results on the test set ({n['test']} customers, {pos['test']} churners):

| | Threshold 0.50 | Optimized Threshold ({thr:.2f}) |
|---|---:|---:|
{row('Recall', 'recall', '{:.4f}')}
{row('Precision', 'precision', '{:.4f}')}
{row('F1', 'f1', '{:.4f}')}
{row('F2', 'f2', '{:.4f}')}
{row('False Negatives', 'fn', '{}')}
{row('False Positives', 'fp', '{}')}
{row('Accuracy', 'accuracy', '{:.4f}')}
{row('ROC-AUC', 'roc_auc', '{:.4f}')}
{row('PR-AUC', 'pr_auc', '{:.4f}')}

**False negatives fell from {test05['fn']} to {testopt['fn']}, which is {fn_red} fewer missed churners ({fn_pct:.1f}% fewer).** The cost was {fp_inc:+d} false positives (from {test05['fp']} to {testopt['fp']}). ROC-AUC and PR-AUC do not depend on the threshold, so they are the same in both columns.

## Observations

- **Churn recall:** With balanced sample weights, the MLP already catches {test05['recall']:.1%} of test churners at the default 0.50 threshold ({test05['tp']} of {pos['test']}). At the threshold of {thr:.2f} chosen on validation, it catches {testopt['recall']:.1%} ({testopt['tp']} of {pos['test']}). The earlier unweighted MLP (32,16) reached only 0.529 recall at 0.50 on the older 80/20 split (see `.claude/memory.md`). That is a different split, so the comparison is only indicative, but the gap is large.
- **False-negative reduction:** Moving from 0.50 to {thr:.2f} removed {fn_red} missed churners on test ({test05['fn']} -> {testopt['fn']}, {fn_pct:.1f}% fewer). Validation showed the same direction ({val05['fn']} -> {valopt['fn']}). The selected threshold is only slightly below 0.50. Balanced weighting already moves the predicted probabilities upward, so most of the gain in recall comes from the weighting and the threshold adds a smaller second step.
- **Precision/recall tradeoff:** The {fn_red} fewer false negatives cost {fp_inc} more false positives ({test05['fp']} -> {testopt['fp']}). Precision fell from {test05['precision']:.3f} to {testopt['precision']:.3f}, and accuracy fell from {test05['accuracy']:.3f} to {testopt['accuracy']:.3f}. F1 ({test05['f1']:.3f} -> {testopt['f1']:.3f}) and F2 ({test05['f2']:.3f} -> {testopt['f2']:.3f}) both went up. Roughly half of the customers flagged as churners do churn. That is acceptable when a retention offer costs much less than a lost customer.
- **Impact of imbalance handling:** Weighting churners about {w1 / w0:.2f} times as heavily changes where the model sets its operating point, not how well it ranks customers. Test ROC-AUC is {test05['roc_auc']:.3f} and PR-AUC is {test05['pr_auc']:.3f}, both well above the {pos['test'] / n['test']:.3f} PR-AUC of a random model. The lower accuracy compared with the unweighted baseline is expected and does not mean the model got worse.
- **Generalization:** Train ROC-AUC is {tr_auc:.3f}, compared with {valopt['roc_auc']:.3f} on validation and {test05['roc_auc']:.3f} on test. The small network, L2 penalty and early stopping (stopped after {model.n_iter_} epochs) keep overfitting low. The validation AUC is a little lower than the test AUC, which reflects normal variation between two samples of about 1,050 customers each.
- **Suitability for the cost-sensitive stage:** The MLP fits this stage. It supports sample weights directly, its probabilities give a usable threshold, and it reaches high churn recall with a PR-AUC in a competitive range. Its weaknesses are that it is harder to explain than Logistic Regression and that results vary somewhat with `random_state` because of the random weight initialization. For SHAP it needs KernelExplainer or DeepExplainer instead of fast exact explainers. It is a reasonable candidate, but it should be chosen over the Logistic Regression and XGBoost models only if it clearly beats them on test recall/F2 and PR-AUC under the same threshold rule.
"""
    (RESULTS_DIR / "mlp.md").write_text(md)
    print(f"\nthreshold={thr:.2f}; test FN {test05['fn']} -> {testopt['fn']} ({fn_red} fewer, {fn_pct:.1f}%), "
          f"FP {test05['fp']} -> {testopt['fp']}")
    print("Wrote artifacts/mlp.joblib, artifacts/mlp_probs.csv, results/mlp.md")


if __name__ == "__main__":
    main()
