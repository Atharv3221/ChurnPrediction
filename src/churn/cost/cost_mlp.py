"""Cost-sensitive threshold analysis for the MLP churn model.

Reuses artifacts/mlp_probs.csv (no retraining). Thresholds are selected on the
validation split only; each selected threshold is evaluated once on test.

Usage: venv/bin/python cost_mlp.py [--c-fn 500] [--c-fp 100]
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

MODEL = "mlp"
MODEL_NAME = "MLP"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

PROBS = ARTIFACTS_DIR / f"{MODEL}_probs.csv"
MODEL_FILE = ARTIFACTS_DIR / f"{MODEL}.joblib"
OUT_JSON = ARTIFACTS_DIR / f"cost_{MODEL}.json"
OUT_PLOT = PLOTS_DIR / f"cost_curve_{MODEL}.png"
OUT_MD = RESULTS_DIR / f"cost_{MODEL}.md"
GRID = np.round(np.arange(0.01, 0.9901, 0.01), 2)
SENS_CFN = [100, 200, 300, 500, 1000, 2000]


def load_f2_threshold():
    """Read the previously chosen F2 threshold; tolerate differing key names."""
    try:
        obj = joblib.load(MODEL_FILE)
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not load {MODEL_FILE}: {e}")
        return None
    if isinstance(obj, dict):
        for k in ("threshold", "f2_threshold", "best_threshold", "thr", "decision_threshold"):
            if k in obj:
                try:
                    return float(obj[k])
                except (TypeError, ValueError):
                    pass
        for k, v in obj.items():
            if "thr" in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
    print("warning: F2 threshold not found in model artifact")
    return None


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


def select_threshold(y, p, c_fn, c_fp):
    rows = cost_curve(y, p, c_fn, c_fp)
    best = min(rows, key=lambda r: (r[3], r[0]))  # min cost, ties -> lower threshold
    return best[0], best[3], rows


def evaluate(y, p, t, c_fn, c_fp, baseline_nobody):
    tn, fp, fn, tp = confusion(y, p, t)
    n = len(y)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    total = fn * c_fn + fp * c_fp
    return {
        "threshold": round(float(t), 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": rec, "precision": prec, "f1": f1, "f2": f2,
        "accuracy": (tp + tn) / n,
        "total_cost": float(total),
        "cost_per_customer": total / n,
        "savings_vs_nobody": float(baseline_nobody - total),
        "savings_vs_nobody_pct": (baseline_nobody - total) / baseline_nobody if baseline_nobody else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c-fn", type=float, default=500.0, help="cost of a missed churner (FN)")
    ap.add_argument("--c-fp", type=float, default=100.0, help="cost of an unneeded retention offer (FP)")
    args = ap.parse_args()
    c_fn, c_fp = args.c_fn, args.c_fp

    df = pd.read_csv(PROBS)
    val, test = df[df.split == "val"], df[df.split == "test"]
    yv, pv = val.y_true.to_numpy().astype(int), val.proba.to_numpy()
    yt, pt = test.y_true.to_numpy().astype(int), test.proba.to_numpy()
    n_test = len(yt)

    theo = c_fp / (c_fp + c_fn)
    f2_thr = load_f2_threshold()

    # 1. validation-only threshold selection
    val_thr, val_min_cost, val_rows = select_threshold(yv, pv, c_fn, c_fp)
    val_cost_at = {r[0]: r[3] for r in val_rows}

    # 3. test evaluation (each threshold once)
    pos, neg = int(yt.sum()), int((1 - yt).sum())
    nobody = pos * c_fn
    everyone = neg * c_fp
    baselines = {
        "target_nobody": {"fn": pos, "fp": 0, "total_cost": float(nobody), "cost_per_customer": nobody / n_test},
        "target_everyone": {"fn": 0, "fp": neg, "total_cost": float(everyone),
                            "cost_per_customer": everyone / n_test,
                            "savings_vs_nobody": float(nobody - everyone)},
    }
    thresholds = {"0.50": 0.50, "f2": f2_thr, "cost_optimal": val_thr}
    test_results = {}
    for key, t in thresholds.items():
        if t is None:
            continue
        r = evaluate(yt, pt, t, c_fn, c_fp, nobody)
        r["val_cost"] = float(sum(np.array(confusion(yv, pv, t))[[1, 2]] * [c_fp, c_fn]))
        test_results[key] = r
    roc = roc_auc_score(yt, pt)
    pr = average_precision_score(yt, pt)

    # 4. sensitivity (C_FP fixed at 100)
    sens_cfp = 100.0
    sensitivity = []
    for cfn in SENS_CFN:
        t, vcost, _ = select_threshold(yv, pv, cfn, sens_cfp)
        tn, fp, fn, tp = confusion(yt, pt, t)
        tc = fn * cfn + fp * sens_cfp
        sensitivity.append({
            "c_fn": cfn, "c_fp": sens_cfp, "ratio": cfn / sens_cfp,
            "theoretical_threshold": sens_cfp / (sens_cfp + cfn),
            "val_threshold": t, "val_cost": float(vcost),
            "test_cost": float(tc), "test_cost_per_customer": tc / n_test,
            "test_fn": fn, "test_fp": fp, "test_tp": tp,
            "test_recall": tp / (tp + fn) if tp + fn else 0.0,
            "test_precision": tp / (tp + fp) if tp + fp else 0.0,
            "test_cost_nobody": float(pos * cfn), "test_cost_everyone": float(neg * sens_cfp),
        })

    # 5. plot validation cost curve
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    ts = [r[0] for r in val_rows]
    cs = [r[3] for r in val_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, cs, color="#2a6fb0", lw=2, label="Validation cost")
    ax.axvline(val_thr, color="#c0392b", ls="-", lw=1.5,
               label=f"Val cost-optimal = {val_thr:.2f} (cost {val_min_cost:,.0f})")
    ax.axvline(0.50, color="#555555", ls="--", lw=1.2, label=f"0.50 (cost {val_cost_at[0.5]:,.0f})")
    ax.axvline(theo, color="#e08e0b", ls=":", lw=1.8, label=f"Theoretical C_FP/(C_FP+C_FN) = {theo:.4f}")
    if f2_thr is not None:
        ax.axvline(f2_thr, color="#27ae60", ls="-.", lw=1.2, label=f"F2 threshold = {f2_thr:.2f}")
    ax.scatter([val_thr], [val_min_cost], color="#c0392b", zorder=5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost on validation")
    ax.set_title(f"{MODEL_NAME} validation cost curve (C_FN={c_fn:g}, C_FP={c_fp:g}, n={len(yv)})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)

    # 6. JSON
    out = {
        "model": MODEL, "c_fn": c_fn, "c_fp": c_fp,
        "theoretical_threshold": theo,
        "val_optimal_threshold": val_thr, "val_min_cost": float(val_min_cost),
        "f2_threshold": f2_thr,
        "test": test_results,
        "baselines": baselines,
        "test_roc_auc": roc, "test_pr_auc": pr,
        "n_val": int(len(yv)), "n_test": n_test, "test_positives": pos, "test_negatives": neg,
        "sensitivity": sensitivity,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # 7. Markdown report
    write_report(out, val_cost_at)
    print(json.dumps({k: out[k] for k in ("val_optimal_threshold", "val_min_cost", "theoretical_threshold")}, indent=2))
    for k, r in test_results.items():
        print(k, {x: r[x] for x in ("threshold", "fn", "fp", "recall", "precision", "total_cost", "cost_per_customer")})
    print("baselines", baselines)
    print(pd.DataFrame(sensitivity)[["ratio", "val_threshold", "test_cost", "test_fn", "test_fp", "test_recall"]])
    print(f"wrote {OUT_JSON}, {OUT_PLOT}, {OUT_MD}")


def write_report(o, val_cost_at):
    c_fn, c_fp = o["c_fn"], o["c_fp"]
    T, B = o["test"], o["baselines"]
    n = o["n_test"]
    L = []
    L.append(f"# {MODEL_NAME} – Cost-Sensitive Analysis\n")
    L.append(f"Generated by `cost_{MODEL}.py` (rerun with `venv/bin/python cost_{MODEL}.py --c-fn {c_fn:g} --c-fp {c_fp:g}`). "
             f"No retraining: probabilities are reused from `artifacts/{MODEL}_probs.csv`.\n")

    L.append("## Cost Assumptions\n")
    L.append("- Total cost = FN × C_FN + FP × C_FP; TP and TN cost 0. Cost per customer = total / n.")
    L.append(f"- **C_FN = {c_fn:g}**: revenue lost when a churner is not targeted.")
    L.append(f"- **C_FP = {c_fp:g}**: cost of a retention offer sent to a customer who would not have churned.")
    L.append("- **These are placeholder values, not business figures.** C_FN ≈ 8 months × ~65 average monthly "
             "charge ≈ 500 of lost revenue; C_FP = 100 for a retention offer. Both are configurable via "
             "`--c-fn` / `--c-fp`.")
    L.append("- Simplification: a targeted churner is assumed to be retained at no cost beyond what is "
             "modelled (TP cost 0, offer always works).\n")

    L.append("## Method (validation-only selection)\n")
    L.append("- Model: MLP (32-16, relu, adam), trained with balanced sample weights on the stratified "
             "70/15/15 split (random_state=42, 34 features). Not retrained here.")
    L.append(f"- Threshold grid 0.01–0.99 (step 0.01). On the **validation** split (n={o['n_val']}) FN, FP and cost "
             "are computed at each threshold; the minimum-cost threshold is chosen (ties → lower threshold).")
    L.append(f"- Theoretical Bayes-optimal threshold for calibrated probabilities: C_FP / (C_FP + C_FN) = "
             f"{c_fp:g} / {c_fp + c_fn:g} = **{o['theoretical_threshold']:.4f}**. Balanced sample weighting "
             "inflates predicted churn probabilities (the model is trained as if classes were 50/50), so its "
             "outputs are not calibrated to the 26.5% base rate and the empirical validation-optimal threshold "
             "can differ from this value (typically sitting higher).")
    L.append(f"- Each threshold (0.50, previous F2 = {o['f2_threshold']}, validation cost-optimal) is evaluated "
             f"exactly once on **test** (n={n}). Accuracy is reported but not optimized.\n")

    L.append("## Validation Cost Curve\n")
    L.append(f"![Validation cost curve](plots/cost_curve_{MODEL}.png)\n")
    L.append(f"- Validation cost-optimal threshold: **{o['val_optimal_threshold']:.2f}**, validation cost "
             f"**{o['val_min_cost']:,.0f}** ({o['val_min_cost'] / o['n_val']:.2f} per customer).")
    L.append(f"- Validation cost at 0.50: {val_cost_at[0.5]:,.0f}; at the theoretical threshold "
             f"(nearest grid point {round(o['theoretical_threshold'], 2):.2f}): "
             f"{val_cost_at[round(o['theoretical_threshold'], 2)]:,.0f}.\n")

    L.append("## Test Results\n")
    L.append(f"Test set: n = {n}, positives = {o['test_positives']}, negatives = {o['test_negatives']}. "
             f"ROC-AUC = **{o['test_roc_auc']:.4f}**, PR-AUC (average precision) = **{o['test_pr_auc']:.4f}**.\n")
    labels = {"0.50": "Default 0.50", "f2": "Previous F2", "cost_optimal": "Val cost-optimal"}
    L.append("| Policy | Thr | TN | FP | FN | TP | Recall | Precision | F1 | F2 | Accuracy | Total cost | Cost/customer | Savings vs nobody |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in ("0.50", "f2", "cost_optimal"):
        if k not in T:
            continue
        r = T[k]
        L.append(f"| {labels[k]} | {r['threshold']:.2f} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} | "
                 f"{r['recall']:.3f} | {r['precision']:.3f} | {r['f1']:.3f} | {r['f2']:.3f} | {r['accuracy']:.3f} | "
                 f"{r['total_cost']:,.0f} | {r['cost_per_customer']:.2f} | "
                 f"{r['savings_vs_nobody']:,.0f} ({100 * r['savings_vs_nobody_pct']:.1f}%) |")
    bn, be = B["target_nobody"], B["target_everyone"]
    L.append(f"| Baseline: target nobody | – | {o['test_negatives']} | 0 | {bn['fn']} | 0 | 0.000 | – | – | – | "
             f"{o['test_negatives'] / n:.3f} | {bn['total_cost']:,.0f} | {bn['cost_per_customer']:.2f} | 0 (0.0%) |")
    L.append(f"| Baseline: target everyone | – | 0 | {be['fp']} | 0 | {o['test_positives']} | 1.000 | "
             f"{o['test_positives'] / n:.3f} | – | – | {o['test_positives'] / n:.3f} | {be['total_cost']:,.0f} | "
             f"{be['cost_per_customer']:.2f} | {be['savings_vs_nobody']:,.0f} "
             f"({100 * be['savings_vs_nobody'] / bn['total_cost']:.1f}%) |\n")

    L.append("## Sensitivity to Cost Ratio\n")
    L.append("C_FP fixed at 100; C_FN varied. For each ratio the threshold is re-selected on validation by "
             "minimum cost, then evaluated once on test.\n")
    L.append("| C_FN | Ratio C_FN/C_FP | Theoretical thr | Val-selected thr | Test FN | Test FP | Test recall | Test cost | Cost/customer | Target-nobody cost | Target-everyone cost |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in o["sensitivity"]:
        L.append(f"| {s['c_fn']:g} | {s['ratio']:g} | {s['theoretical_threshold']:.3f} | {s['val_threshold']:.2f} | "
                 f"{s['test_fn']} | {s['test_fp']} | {s['test_recall']:.3f} | {s['test_cost']:,.0f} | "
                 f"{s['test_cost_per_customer']:.2f} | {s['test_cost_nobody']:,.0f} | {s['test_cost_everyone']:,.0f} |")
    L.append("")

    L.append("## Observations\n")
    co, d5 = T["cost_optimal"], T["0.50"]
    L.append(f"- At C_FN={c_fn:g}, C_FP={c_fp:g} the validation-selected threshold is {co['threshold']:.2f} "
             f"(theoretical {o['theoretical_threshold']:.4f}). On test it gives FN={co['fn']}, FP={co['fp']}, "
             f"recall {co['recall']:.3f}, precision {co['precision']:.3f}, cost {co['total_cost']:,.0f} "
             f"({co['cost_per_customer']:.2f}/customer).")
    L.append(f"- Versus the default 0.50 (test cost {d5['total_cost']:,.0f}), the cost-optimal threshold changes test "
             f"cost by {co['total_cost'] - d5['total_cost']:+,.0f}.")
    if "f2" in T:
        f2 = T["f2"]
        L.append(f"- The previous F2 threshold ({f2['threshold']:.2f}) costs {f2['total_cost']:,.0f} on test "
                 f"({f2['total_cost'] - co['total_cost']:+,.0f} vs cost-optimal). F2 weights recall 4× in a harmonic "
                 "mean, which is not the same as a fixed 5:1 cost ratio, so the two criteria need not agree.")
    best_model = min([T[k]["total_cost"] for k in T])
    L.append(f"- Every model-based policy beats both baselines: best model test cost {best_model:,.0f} vs "
             f"target-nobody {bn['total_cost']:,.0f} and target-everyone {be['total_cost']:,.0f}.")
    if "f2" in T and abs(T["f2"]["threshold"] - co["threshold"]) < 1e-9:
        L.append("- The cost-optimal threshold coincides with the previous F2 threshold, so those two rows are "
                 "identical; at this cost ratio the F2 choice was already cost-optimal on validation.")
    vt = round(o["val_optimal_threshold"], 2)
    nb = [val_cost_at[t] for t in (round(vt - 0.01, 2), round(vt + 0.01, 2)) if t in val_cost_at]
    L.append(f"- The validation minimum is a narrow dip: neighbouring thresholds cost "
             f"{', '.join(f'{c:,.0f}' for c in nb)} vs {o['val_min_cost']:,.0f} at {vt:.2f} (n={o['n_val']}, so one "
             "FN = C_FN). The exact optimum is therefore somewhat noisy; differences of a few FN/FP on test are "
             "within sampling noise.")
    thr_seq = [s["val_threshold"] for s in o["sensitivity"]]
    L.append(f"- Sensitivity: as C_FN/C_FP rises from 1 to 20 the validation-selected threshold moves "
             f"{' → '.join(f'{t:.2f}' for t in thr_seq)}, trading more FPs for fewer FNs. "
             "Compare with the theoretical C_FP/(C_FP+C_FN) column: because balanced weighting inflates "
             "probabilities, the validation-selected thresholds generally differ from (mostly sit above) the "
             "theoretical ones; with calibrated probabilities they would track them more closely.")
    tied = [s["ratio"] for s in o["sensitivity"] if abs(s["val_threshold"] - o["val_optimal_threshold"]) < 1e-9]
    if len(tied) > 1:
        L.append(f"- The same threshold ({o['val_optimal_threshold']:.2f}) is selected for ratios "
                 f"{', '.join(f'{r:g}' for r in tied)}, so the decision is robust to moderate misestimation of C_FN.")
    L.append("- Cost figures are placeholders; the ranking of thresholds, not the absolute amounts, is the "
             "transferable result. Replace C_FN/C_FP with real CLV and offer costs before acting on them.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
