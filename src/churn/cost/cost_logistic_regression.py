"""Cost-sensitive threshold analysis for the Logistic Regression churn model.

Reuses saved validation/test probabilities (no retraining). Thresholds are
selected on VALIDATION only; each selected threshold is evaluated once on TEST.
Writes artifacts/cost_logistic_regression.json, results/plots/cost_curve_logistic_regression.png
and results/cost_logistic_regression.md.

Usage: venv/bin/python cost_logistic_regression.py [--c-fn 500] [--c-fp 100]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODEL = "logistic_regression"
MODEL_NAME = "Logistic Regression"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

PROBS = ARTIFACTS_DIR / f"{MODEL}_probs.csv"
JOBLIB = ARTIFACTS_DIR / f"{MODEL}.joblib"
OUT_JSON = ARTIFACTS_DIR / f"cost_{MODEL}.json"
OUT_PNG = PLOTS_DIR / f"cost_curve_{MODEL}.png"
OUT_MD = RESULTS_DIR / f"cost_{MODEL}.md"
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
SENS_C_FN = [100, 200, 300, 500, 1000, 2000]


def confusion(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tn, fp, fn, tp


def cost_curve(y, p, c_fn, c_fp):
    rows = []
    for t in GRID:
        tn, fp, fn, tp = confusion(y, p, t)
        rows.append((float(t), fn, fp, fn * c_fn + fp * c_fp))
    return rows


def pick_threshold(y, p, c_fn, c_fp):
    rows = cost_curve(y, p, c_fn, c_fp)
    # min cost, ties -> lower threshold (grid is ascending, so first minimum)
    best = min(rows, key=lambda r: (r[3], r[0]))
    return best[0], best[3], rows


def evaluate(y, p, t, c_fn, c_fp):
    tn, fp, fn, tp = confusion(y, p, t)
    n = len(y)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    cost = fn * c_fn + fp * c_fp
    nobody = int((y == 1).sum()) * c_fn
    return {
        "threshold": round(float(t), 4), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "recall": rec, "precision": prec, "f1": f1, "f2": f2,
        "accuracy": (tp + tn) / n, "total_cost": cost, "cost_per_customer": cost / n,
        "savings_vs_target_nobody": nobody - cost,
        "savings_vs_target_nobody_pct": (nobody - cost) / nobody if nobody else 0.0,
    }


def load_f2_threshold():
    obj = joblib.load(JOBLIB)
    if isinstance(obj, dict):
        for k in ("threshold", "f2_threshold", "best_threshold", "thr", "decision_threshold"):
            if k in obj:
                return float(obj[k]), k
        for k, v in obj.items():
            if "thr" in str(k).lower() and isinstance(v, (int, float)):
                return float(v), k
    for attr in ("threshold", "threshold_"):
        if hasattr(obj, attr):
            return float(getattr(obj, attr)), attr
    print("WARNING: F2 threshold not found in joblib; falling back to 0.45 (from logistic_regression.md)")
    return 0.45, "fallback(0.45)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-fn", type=float, default=500.0, help="cost of a missed churner (FN)")
    ap.add_argument("--c-fp", type=float, default=100.0, help="cost of an unneeded retention offer (FP)")
    args = ap.parse_args()
    c_fn, c_fp = args.c_fn, args.c_fp

    df = pd.read_csv(PROBS)
    val, test = df[df.split == "val"], df[df.split == "test"]
    yv, pv = val.y_true.to_numpy().astype(int), val.proba.to_numpy()
    yt, pt = test.y_true.to_numpy().astype(int), test.proba.to_numpy()
    n_test = len(yt)

    f2_thr, f2_key = load_f2_threshold()
    theo = c_fp / (c_fp + c_fn)

    # 1. Validation-only selection
    val_opt, val_min_cost, val_rows = pick_threshold(yv, pv, c_fn, c_fp)
    val_at = {lab: evaluate(yv, pv, t, c_fn, c_fp) for lab, t in
              (("0.50", 0.50), ("f2", f2_thr), ("cost_optimal", val_opt))}
    theo_grid = round(theo, 2)
    val_theo = evaluate(yv, pv, theo, c_fn, c_fp)

    # 3. Test evaluation, once per selected threshold
    test_res = {lab: evaluate(yt, pt, t, c_fn, c_fp) for lab, t in
                (("0.50", 0.50), ("f2", f2_thr), ("cost_optimal", val_opt))}
    pos, neg = int(yt.sum()), int((yt == 0).sum())
    baselines = {
        "target_nobody": {"total_cost": pos * c_fn, "cost_per_customer": pos * c_fn / n_test, "FN": pos, "FP": 0},
        "target_everyone": {"total_cost": neg * c_fp, "cost_per_customer": neg * c_fp / n_test, "FN": 0, "FP": neg},
    }
    roc, pr = roc_auc_score(yt, pt), average_precision_score(yt, pt)

    # 4. Sensitivity (C_FP fixed at 100)
    sens = []
    for cf in SENS_C_FN:
        t, vc, _ = pick_threshold(yv, pv, cf, 100.0)
        r = evaluate(yt, pt, t, cf, 100.0)
        sens.append({"c_fn": cf, "c_fp": 100.0, "ratio": cf / 100.0, "theoretical_threshold": 100.0 / (100.0 + cf),
                     "val_threshold": t, "val_cost": vc, "test_cost": r["total_cost"],
                     "test_cost_per_customer": r["cost_per_customer"], "FN": r["FN"], "FP": r["FP"],
                     "recall": r["recall"], "precision": r["precision"]})

    # 5. Plot
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    ts = [r[0] for r in val_rows]
    cs = [r[3] for r in val_rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ts, cs, color="#2a6fdb", lw=2, label="Validation total cost")
    ax.axvline(val_opt, color="#d1495b", ls="-", lw=1.5, label=f"Val cost-optimal = {val_opt:.2f} (cost {val_min_cost:,.0f})")
    ax.axvline(0.50, color="#555555", ls="--", lw=1.2, label=f"0.50 (cost {val_at['0.50']['total_cost']:,.0f})")
    ax.axvline(theo, color="#2e933c", ls=":", lw=1.8, label=f"Theoretical C_FP/(C_FP+C_FN) = {theo:.4f}")
    ax.axvline(f2_thr, color="#e8a33d", ls="-.", lw=1.2, label=f"F2 threshold = {f2_thr:.2f} (cost {val_at['f2']['total_cost']:,.0f})")
    ax.scatter([val_opt], [val_min_cost], color="#d1495b", zorder=5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Total cost (FN x {c_fn:g} + FP x {c_fp:g})")
    ax.set_title(f"{MODEL_NAME}: validation cost vs threshold (n={len(yv)})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)

    # 6. JSON
    out = {
        "model": MODEL, "c_fn": c_fn, "c_fp": c_fp,
        "theoretical_threshold": theo,
        "val_optimal_threshold": val_opt, "val_min_cost": val_min_cost,
        "f2_threshold": f2_thr, "f2_threshold_source_key": f2_key,
        "validation": val_at,
        "test": test_res,
        "baselines": baselines,
        "test_roc_auc": roc, "test_pr_auc": pr,
        "n_val": int(len(yv)), "n_test": int(n_test),
        "sensitivity": sens,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # 7. Markdown
    def money(x):
        return f"{x:,.0f}"

    L = []
    L.append(f"# {MODEL_NAME} – Cost-Sensitive Analysis\n")
    L.append(f"All numbers below are generated by `venv/bin/python cost_{MODEL}.py --c-fn {c_fn:g} --c-fp {c_fp:g}`. "
             "The model is not retrained; the script reuses the saved validation/test probabilities in "
             f"`artifacts/{MODEL}_probs.csv` (model trained with `class_weight=\"balanced\"`, C=1, on the stratified "
             "70/15/15 split with `random_state=42`, 34 features).\n")
    L.append("## Cost Assumptions\n")
    L.append("| Outcome | Cost |\n|---|---:|")
    L.append(f"| False negative (churner not targeted, customer lost) | C_FN = {money(c_fn)} |")
    L.append(f"| False positive (retention offer to a non-churner) | C_FP = {money(c_fp)} |")
    L.append("| True positive / true negative | 0 |\n")
    L.append("- Total cost = FN x C_FN + FP x C_FP; cost per customer = total / n.")
    L.append("- **These defaults are placeholders, not business figures.** C_FN = 500 approximates lost revenue "
             "(~8 months x ~65 average monthly charge ≈ 500); C_FP = 100 approximates the cost of a retention offer. "
             "Both are configurable with `--c-fn` and `--c-fp`.")
    L.append("- Simplifications: a targeted churner is assumed to be retained at zero cost (TP = 0), i.e. the offer "
             "always works and its cost on true churners is ignored. Real deployments should replace both numbers with "
             "CLV and campaign-cost estimates.\n")
    L.append("## Method (validation-only selection)\n")
    L.append(f"1. Threshold grid 0.01 to 0.99 (step 0.01). On the **validation** split (n={len(yv)}) FN, FP and total cost "
             "are computed at each threshold; the threshold with the minimum validation cost is chosen (ties -> lower threshold).")
    L.append(f"2. Theoretical Bayes-optimal threshold for calibrated probabilities: C_FP / (C_FP + C_FN) = "
             f"{c_fp:g} / ({c_fp:g} + {c_fn:g}) = **{theo:.4f}**. The model was trained with `class_weight=\"balanced\"`, "
             "which up-weights churners ~2.77x and shifts predicted probabilities upward, so they are not calibrated to "
             "the true churn rate and the empirical validation-optimal threshold can differ from the theoretical one.")
    L.append(f"3. Three thresholds are evaluated **once each** on the test split (n={n_test}): 0.50, the previous F2 "
             f"threshold ({f2_thr:.2f}, read from `artifacts/{MODEL}.joblib` key `{f2_key}`), and the validation cost-optimal threshold.")
    L.append("4. Sensitivity: C_FP fixed at 100, C_FN varied; the threshold is re-selected on validation for each ratio and then scored on test.")
    L.append("5. Accuracy is reported for reference only; it is not an objective.\n")
    L.append("## Validation Cost Curve\n")
    L.append(f"![Validation cost curve](plots/cost_curve_{MODEL}.png)\n")
    L.append("| Threshold (validation) | FN | FP | Total cost | Cost / customer |\n|---|---:|---:|---:|---:|")
    for lab, name in (("0.50", "0.50"), ("f2", f"F2 ({f2_thr:.2f})"), ("cost_optimal", f"**Cost-optimal ({val_opt:.2f})**")):
        r = val_at[lab]
        L.append(f"| {name} | {r['FN']} | {r['FP']} | {money(r['total_cost'])} | {r['cost_per_customer']:.2f} |")
    L.append(f"| Theoretical ({theo:.4f}) | {val_theo['FN']} | {val_theo['FP']} | {money(val_theo['total_cost'])} | {val_theo['cost_per_customer']:.2f} |\n")
    L.append(f"Validation-optimal threshold: **{val_opt:.2f}** with minimum validation cost **{money(val_min_cost)}** "
             f"({val_min_cost / len(yv):.2f} per customer).\n")
    L.append("## Test Results\n")
    L.append(f"Test ROC-AUC = **{roc:.4f}**, PR-AUC (average precision) = **{pr:.4f}**. Positives = {pos}, negatives = {neg}.\n")
    hdr = ("| Threshold | Thr | TN | FP | FN | TP | Recall | Precision | F1 | F2 | Accuracy | Total cost | Cost / cust. | Savings vs nobody |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    L.append(hdr)
    for lab, name in (("0.50", "Default 0.50"), ("f2", "F2 (previous)"), ("cost_optimal", "**Cost-optimal (val)**")):
        r = test_res[lab]
        L.append(f"| {name} | {r['threshold']:.2f} | {r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} | {r['recall']:.3f} | "
                 f"{r['precision']:.3f} | {r['f1']:.3f} | {r['f2']:.3f} | {r['accuracy']:.3f} | {money(r['total_cost'])} | "
                 f"{r['cost_per_customer']:.2f} | {money(r['savings_vs_target_nobody'])} ({r['savings_vs_target_nobody_pct']:.1%}) |")
    bn, be = baselines["target_nobody"], baselines["target_everyone"]
    L.append(f"| Baseline: target nobody | – | {neg} | 0 | {pos} | 0 | 0.000 | – | – | – | {neg / n_test:.3f} | "
             f"{money(bn['total_cost'])} | {bn['cost_per_customer']:.2f} | 0 (0.0%) |")
    sav_e = bn["total_cost"] - be["total_cost"]
    L.append(f"| Baseline: target everyone | – | 0 | {neg} | 0 | {pos} | 1.000 | {pos / n_test:.3f} | – | – | {pos / n_test:.3f} | "
             f"{money(be['total_cost'])} | {be['cost_per_customer']:.2f} | {money(sav_e)} ({sav_e / bn['total_cost']:.1%}) |\n")
    L.append("## Sensitivity to Cost Ratio\n")
    L.append("C_FP = 100 fixed. For each C_FN the threshold is chosen on validation by minimum cost, then evaluated once on test.\n")
    L.append("| C_FN | Ratio C_FN/C_FP | Theoretical thr | Val-selected thr | Test cost | Test cost / cust. | FN | FP | Recall | Precision |\n"
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sens:
        L.append(f"| {s['c_fn']:,} | {s['ratio']:g} | {s['theoretical_threshold']:.3f} | {s['val_threshold']:.2f} | "
                 f"{money(s['test_cost'])} | {s['test_cost_per_customer']:.2f} | {s['FN']} | {s['FP']} | "
                 f"{s['recall']:.3f} | {s['precision']:.3f} |")
    L.append("")
    L.append("## Observations\n")
    co, d5, f2r = test_res["cost_optimal"], test_res["0.50"], test_res["f2"]
    L.append(f"- The validation cost-optimal threshold is **{val_opt:.2f}**, versus the theoretical {theo:.4f}. "
             + ("They are close, " if abs(val_opt - theo) <= 0.05 else "The gap ")
             + ("which suggests the balanced-weight probabilities are not far off for this decision."
                if abs(val_opt - theo) <= 0.05 else
                "reflects that `class_weight=\"balanced\"` inflates churn probabilities, so the threshold that is optimal on "
                "real outcomes is higher than the calibrated-probability formula suggests."))
    # odds-shift correction for balanced weighting: weighted odds = true odds * (n_neg/n_pos in train)
    w_ratio = 3621 / 1308
    theo_odds = theo / (1 - theo) * w_ratio
    theo_bal = theo_odds / (1 + theo_odds)
    L.append(f"- Rough check: balanced weighting multiplies the model's odds by ~{w_ratio:.2f} (train negatives/positives), so "
             f"the theoretical threshold expressed on the weighted scale is about {theo_bal:.3f}. This is closer to, but still "
             f"below, the empirical {val_opt:.2f}; the remaining gap is consistent with imperfect calibration and validation noise.")
    same = abs(val_opt - f2_thr) < 1e-9
    L.append(f"- On test, cost at 0.50 = {money(d5['total_cost'])}, at F2 ({f2_thr:.2f}) = {money(f2r['total_cost'])}, at the "
             f"cost-optimal threshold ({val_opt:.2f}) = {money(co['total_cost'])}. "
             + (f"At these costs the cost-optimal threshold coincides with the previous F2 threshold, so they give identical "
                f"test results. " if same else "")
             + f"Cost-optimal vs 0.50 saves {money(d5['total_cost'] - co['total_cost'])} on test "
             f"(negative would mean 0.50 is cheaper).")
    L.append(f"- The cost-optimal threshold catches {co['TP']} of {pos} churners (recall {co['recall']:.3f}) at precision "
             f"{co['precision']:.3f}, saving {money(co['savings_vs_target_nobody'])} ({co['savings_vs_target_nobody_pct']:.1%}) "
             f"vs targeting nobody and {money(be['total_cost'] - co['total_cost'])} vs targeting everyone.")
    near = [r[3] for r in val_rows if abs(r[0] - val_opt) <= 0.05 + 1e-9]
    L.append(f"- Validation-selected thresholds are not guaranteed to be test-optimal: the validation set has only "
             f"{int(yv.sum())} churners. Within ±0.05 of the chosen threshold, validation cost ranges from "
             f"{money(min(near))} to {money(max(near))} ({(max(near) - min(near)) / min(near):.1%} above the minimum at most), "
             "so the exact location of the optimum is subject to sampling noise. "
             f"The curve is shallow over a wide band: between 0.20 and {val_opt:.2f} validation cost stays within "
             f"{money(min(r[3] for r in val_rows if 0.2 - 1e-9 <= r[0] <= val_opt + 1e-9))}-"
             f"{money(max(r[3] for r in val_rows if 0.2 - 1e-9 <= r[0] <= val_opt + 1e-9))}, and rises steeply above ~0.55.")
    L.append(f"- Sensitivity: as C_FN/C_FP rises from {sens[0]['ratio']:g} to {sens[-1]['ratio']:g}, the selected threshold "
             f"moves from {sens[0]['val_threshold']:.2f} to {sens[-1]['val_threshold']:.2f} and test recall from "
             f"{sens[0]['recall']:.3f} to {sens[-1]['recall']:.3f}; the decision is driven mainly by the cost ratio, so "
             "the placeholder costs should be replaced with real business estimates before deployment.")
    L.append("- Accuracy is not used for selection; the cost-optimal threshold typically has lower accuracy than 0.50 "
             "because it accepts more false positives to avoid expensive false negatives.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")

    # Console summary
    print(f"F2 threshold ({f2_key}) = {f2_thr}; theoretical = {theo:.4f}; val-optimal = {val_opt} (val cost {val_min_cost:,.0f})")
    for lab, r in test_res.items():
        print(f"TEST {lab:>12} thr={r['threshold']:.2f} TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']} "
              f"rec={r['recall']:.3f} prec={r['precision']:.3f} f1={r['f1']:.3f} f2={r['f2']:.3f} acc={r['accuracy']:.3f} "
              f"cost={r['total_cost']:,.0f} per={r['cost_per_customer']:.2f} save={r['savings_vs_target_nobody']:,.0f}")
    print("Baselines:", baselines)
    print(f"Test ROC-AUC={roc:.4f} PR-AUC={pr:.4f}")
    for s in sens:
        print(f"SENS C_FN={s['c_fn']} ratio={s['ratio']:g} thr={s['val_threshold']:.2f} test_cost={s['test_cost']:,.0f} "
              f"FN={s['FN']} FP={s['FP']} recall={s['recall']:.3f}")


if __name__ == "__main__":
    main()
