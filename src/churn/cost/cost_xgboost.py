"""Cost-sensitive threshold analysis for the XGBoost churn model.

Reuses artifacts/xgboost_probs.csv (no retraining). Thresholds are selected on
VALIDATION only; each selected threshold is evaluated exactly once on TEST.

Outputs:
  artifacts/cost_xgboost.json
  results/plots/cost_curve_xgboost.png
  results/cost_xgboost.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

MODEL = "xgboost"
TITLE = "XGBoost"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

PROBS = ARTIFACTS_DIR / f"{MODEL}_probs.csv"
METRICS = ARTIFACTS_DIR / f"{MODEL}_metrics.json"
OUT_JSON = ARTIFACTS_DIR / f"cost_{MODEL}.json"
OUT_PLOT = PLOTS_DIR / f"cost_curve_{MODEL}.png"
OUT_MD = RESULTS_DIR / f"cost_{MODEL}.md"

GRID = np.round(np.arange(0.01, 0.991, 0.01), 2)
SENS_C_FN = [100, 200, 300, 500, 1000, 2000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-fn", type=float, default=500.0,
                   help="cost of a missed churner (FN); placeholder ~8 months x ~65 avg monthly charge")
    p.add_argument("--c-fp", type=float, default=100.0,
                   help="cost of a retention offer sent to a non-churner (FP); placeholder")
    return p.parse_args()


def load_f2_threshold(path):
    """Read the previously chosen F2 threshold, tolerating different key names."""
    with open(path) as f:
        m = json.load(f)
    for key in ("threshold", "best_threshold", "f2_threshold", "opt_threshold",
                "threshold_opt", "selected_threshold"):
        v = m.get(key)
        if isinstance(v, (int, float)):
            return float(v), key
        if isinstance(v, dict) and isinstance(v.get("threshold"), (int, float)):
            return float(v["threshold"]), key
    for split in ("val", "test"):
        v = m.get(split, {}).get("threshold_opt", {})
        if isinstance(v, dict) and isinstance(v.get("threshold"), (int, float)):
            return float(v["threshold"]), f"{split}.threshold_opt.threshold"
    # Last resort: any top-level numeric key containing "threshold"
    for k, v in m.items():
        if "threshold" in k.lower() and isinstance(v, (int, float)):
            return float(v), k
    raise KeyError(f"No F2 threshold found in {path}")


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


def pick_min(curve):
    """Minimum cost; ties broken by the lower threshold (grid is ascending)."""
    best = curve[0]
    for r in curve[1:]:
        if r[3] < best[3]:
            best = r
    return best


def evaluate(y, p, t, c_fn, c_fp, nobody_cost):
    tn, fp, fn, tp = confusion(y, p, t)
    n = len(y)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    total = fn * c_fn + fp * c_fp
    return {
        "threshold": round(float(t), 4), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": rec, "precision": prec, "f1": f1, "f2": f2,
        "accuracy": (tp + tn) / n, "total_cost": total, "cost_per_customer": total / n,
        "savings_vs_nobody": nobody_cost - total,
        "savings_vs_nobody_pct": (nobody_cost - total) / nobody_cost if nobody_cost else 0.0,
    }


def money(x):
    return f"{x:,.0f}"


def plot_curve(curve, chosen, theo, f2_thr, c_fn, c_fp):
    t = [r[0] for r in curve]
    c = [r[3] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(t, c, color="#2a78d6", lw=2, label="Validation total cost")
    ax.axvline(chosen[0], color="#1baf7a", lw=2, ls="-",
               label=f"Val cost-optimal t={chosen[0]:.2f} (cost {money(chosen[3])})")
    ax.axvline(theo, color="#4a3aa7", lw=1.5, ls="--",
               label=f"Theoretical Bayes t={theo:.4f}")
    ax.axvline(0.50, color="#eb6834", lw=1.5, ls=":", label="Default t=0.50")
    ax.axvline(f2_thr, color="#52514e", lw=1, ls="-.", label=f"Previous F2 t={f2_thr:.2f}")
    ax.scatter([chosen[0]], [chosen[3]], s=60, color="#1baf7a", zorder=5,
               edgecolor="#fcfcfb", linewidth=2)
    ax.set_xlabel("Decision threshold", color="#0b0b0b")
    ax.set_ylabel("Total cost on validation",
                  color="#0b0b0b")
    ax.set_title(f"{TITLE} - validation cost curve (C_FN={money(c_fn)}, C_FP={money(c_fp)})",
                 color="#0b0b0b", loc="left")
    ax.grid(True, color="#e5e4e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors="#52514e")
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=8.5, labelcolor="#0b0b0b", loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    a = parse_args()
    c_fn, c_fp = a.c_fn, a.c_fp

    df = pd.read_csv(PROBS)
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    yv, pv = val["y_true"].to_numpy().astype(int), val["proba"].to_numpy().astype(float)
    yt, pt = test["y_true"].to_numpy().astype(int), test["proba"].to_numpy().astype(float)

    f2_thr, f2_key = load_f2_threshold(METRICS)
    theo = c_fp / (c_fp + c_fn)

    # 1. Validation-only threshold selection
    curve = cost_curve(yv, pv, c_fn, c_fp)
    best = pick_min(curve)
    val_cost_at = {r[0]: r for r in curve}
    theo_on_grid = float(np.round(theo, 2))

    # 3. Test evaluation, each threshold once
    n_t = len(yt)
    pos_t, neg_t = int(yt.sum()), int((yt == 0).sum())
    nobody = pos_t * c_fn
    everyone = neg_t * c_fp
    baselines = {
        "target_nobody": {"total_cost": nobody, "cost_per_customer": nobody / n_t,
                          "fn": pos_t, "fp": 0, "recall": 0.0},
        "target_everyone": {"total_cost": everyone, "cost_per_customer": everyone / n_t,
                            "fn": 0, "fp": neg_t, "recall": 1.0,
                            "savings_vs_nobody": nobody - everyone},
    }
    test_results = {
        "0.50": evaluate(yt, pt, 0.50, c_fn, c_fp, nobody),
        "f2": evaluate(yt, pt, f2_thr, c_fn, c_fp, nobody),
        "cost_optimal": evaluate(yt, pt, best[0], c_fn, c_fp, nobody),
    }
    roc = roc_auc_score(yt, pt)
    pr = average_precision_score(yt, pt)

    # 4. Sensitivity (C_FP fixed at the configured value; spec default 100)
    sens = []
    for cfn in SENS_C_FN:
        b = pick_min(cost_curve(yv, pv, cfn, c_fp))
        tn, fp, fn, tp = confusion(yt, pt, b[0])
        sens.append({
            "c_fn": cfn, "c_fp": c_fp, "ratio": cfn / c_fp,
            "theoretical_threshold": c_fp / (c_fp + cfn),
            "val_threshold": b[0], "val_cost": b[3],
            "test_cost": fn * cfn + fp * c_fp, "test_cost_per_customer": (fn * cfn + fp * c_fp) / n_t,
            "test_fn": fn, "test_fp": fp, "test_recall": tp / (tp + fn) if tp + fn else 0.0,
            "test_nobody_cost": pos_t * cfn, "test_everyone_cost": neg_t * c_fp,
        })

    # 5. Plot
    plot_curve(curve, best, theo, f2_thr, c_fn, c_fp)

    # 6. JSON
    out = {
        "model": MODEL, "c_fn": c_fn, "c_fp": c_fp,
        "theoretical_threshold": theo,
        "val_optimal_threshold": best[0], "val_min_cost": best[3],
        "val_min_fn": best[1], "val_min_fp": best[2],
        "val_cost_at_0.50": val_cost_at[0.5][3],
        "f2_threshold": f2_thr, "f2_threshold_source_key": f2_key,
        "n_val": int(len(yv)), "n_test": n_t,
        "test": test_results, "baselines": baselines,
        "test_roc_auc": roc, "test_pr_auc": pr,
        "sensitivity": sens,
        "val_cost_curve": [{"threshold": r[0], "fn": r[1], "fp": r[2], "cost": r[3]} for r in curve],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # 7. Markdown report
    write_md(out, val_cost_at, theo_on_grid, pos_t, neg_t, yv)
    print(json.dumps({k: out[k] for k in ("val_optimal_threshold", "val_min_cost",
                                           "theoretical_threshold", "f2_threshold")}, indent=2))
    for k, r in test_results.items():
        print(k, {x: r[x] for x in ("threshold", "fn", "fp", "recall", "precision",
                                    "total_cost", "cost_per_customer")})
    print("baselines", baselines)
    for s in sens:
        print(s["ratio"], s["val_threshold"], s["test_cost"], s["test_fn"], s["test_fp"],
              round(s["test_recall"], 4))


def write_md(o, val_cost_at, theo_on_grid, pos_t, neg_t, yv):
    c_fn, c_fp = o["c_fn"], o["c_fp"]
    tr, bl = o["test"], o["baselines"]
    n_t, n_v = o["n_test"], o["n_val"]
    best_t = o["val_optimal_threshold"]
    co, d50, f2 = tr["cost_optimal"], tr["0.50"], tr["f2"]
    L = []
    L.append(f"# {TITLE} – Cost-Sensitive Analysis\n")
    L.append("_Generated by `cost_xgboost.py` from `artifacts/xgboost_probs.csv`; "
             "rerun the script to reproduce. No model was retrained._\n")

    L.append("## Cost Assumptions\n")
    L.append("| Outcome | Cost |\n|---|---:|")
    L.append(f"| False negative (churner not targeted, customer lost) | C_FN = {money(c_fn)} |")
    L.append(f"| False positive (retention offer to a non-churner) | C_FP = {money(c_fp)} |")
    L.append("| True positive / true negative | 0 |\n")
    L.append(f"Total cost = FN x C_FN + FP x C_FP; cost per customer = total / n. "
             f"Cost ratio C_FN / C_FP = {c_fn / c_fp:g}.\n")
    L.append("**These values are placeholders, not business-validated figures.** The default "
             "C_FN = 500 approximates lost revenue as ~8 months x ~65 average monthly charge; "
             "the default C_FP = 100 is an assumed retention-offer cost. TP is treated as cost 0, "
             "i.e. the offer is assumed to always retain a targeted churner and its cost is ignored "
             "for true churners. Both values are configurable: "
             "`venv/bin/python cost_xgboost.py --c-fn 500 --c-fp 100`.\n")

    L.append("## Method (validation-only selection)\n")
    L.append(f"- Reused the saved predicted probabilities of the trained XGBoost model "
             f"(scale_pos_weight ≈ 2.77, depth 3, 113 trees); no retraining. Split unchanged: "
             f"stratified 70/15/15, random_state=42 (validation n={n_v}, test n={n_t}).")
    L.append("- Threshold grid 0.01–0.99, step 0.01. On **validation only**, FN, FP and total cost were "
             "computed at each threshold; the minimum-cost threshold was selected (ties → lower threshold).")
    L.append(f"- Theoretical Bayes-optimal threshold for calibrated probabilities: "
             f"C_FP / (C_FP + C_FN) = {money(c_fp)} / ({money(c_fp)} + {money(c_fn)}) = "
             f"**{o['theoretical_threshold']:.4f}**. XGBoost was trained with scale_pos_weight ≈ 2.77, "
             "which inflates predicted churn probabilities relative to the true base rate, so its scores "
             "are not calibrated and the empirical validation-optimal threshold can differ from the "
             "theoretical one.")
    L.append(f"- Previous F2-selected threshold (read from `artifacts/xgboost_metrics.json`, key "
             f"`{o['f2_threshold_source_key']}`): {o['f2_threshold']:.2f}.")
    L.append("- Three thresholds (0.50, F2, validation cost-optimal) were each evaluated **once** on test. "
             "Accuracy was not used for selection.\n")

    L.append("## Validation Cost Curve\n")
    L.append(f"![Validation cost curve](plots/cost_curve_{MODEL}.png)\n")
    pos_v = int(yv.sum())
    L.append(f"Validation: {pos_v} churners / {n_v - pos_v} non-churners. "
             f"Target-nobody cost = {money(pos_v * c_fn)}.\n")
    L.append("| Threshold | Val FN | Val FP | Val total cost |\n|---|---:|---:|---:|")
    rows = sorted({best_t, 0.5, round(o['f2_threshold'], 2), theo_on_grid})
    for t in rows:
        r = val_cost_at.get(round(t, 2))
        if r is None:
            continue
        tag = []
        if t == best_t:
            tag.append("val cost-optimal")
        if t == 0.5:
            tag.append("default")
        if t == round(o["f2_threshold"], 2):
            tag.append("F2")
        if t == theo_on_grid:
            tag.append("≈ theoretical")
        L.append(f"| {t:.2f} ({', '.join(tag)}) | {r[1]} | {r[2]} | {money(r[3])} |")
    L.append(f"\nMinimum validation cost: **{money(o['val_min_cost'])}** at threshold "
             f"**{best_t:.2f}** (theoretical {o['theoretical_threshold']:.4f}).\n")

    L.append("## Test Results\n")
    L.append(f"Test set: n = {n_t} ({pos_t} churners, {neg_t} non-churners). "
             f"ROC-AUC = {o['test_roc_auc']:.4f}, PR-AUC (average precision) = {o['test_pr_auc']:.4f}.\n")
    L.append("| Metric | t = 0.50 | F2 threshold | Val cost-optimal |\n|---|---:|---:|---:|")
    cols = [d50, f2, co]
    L.append("| Threshold | " + " | ".join(f"{c['threshold']:.2f}" for c in cols) + " |")
    for key, lab in (("tn", "TN"), ("fp", "FP"), ("fn", "FN"), ("tp", "TP")):
        L.append(f"| {lab} | " + " | ".join(str(c[key]) for c in cols) + " |")
    for key, lab in (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"),
                     ("f2", "F2"), ("accuracy", "Accuracy")):
        L.append(f"| {lab} | " + " | ".join(f"{c[key]:.4f}" for c in cols) + " |")
    L.append("| **Total cost** | " + " | ".join(f"**{money(c['total_cost'])}**" for c in cols) + " |")
    L.append("| Cost per customer | " + " | ".join(f"{c['cost_per_customer']:.2f}" for c in cols) + " |")
    L.append("| Savings vs target nobody | " + " | ".join(
        f"{money(c['savings_vs_nobody'])} ({c['savings_vs_nobody_pct']:.1%})" for c in cols) + " |\n")
    L.append("Reference baselines on test:\n")
    L.append("| Baseline | FN | FP | Total cost | Cost per customer |\n|---|---:|---:|---:|---:|")
    b0, b1 = bl["target_nobody"], bl["target_everyone"]
    L.append(f"| Target nobody (all positives x C_FN) | {b0['fn']} | {b0['fp']} | "
             f"{money(b0['total_cost'])} | {b0['cost_per_customer']:.2f} |")
    L.append(f"| Target everyone (all negatives x C_FP) | {b1['fn']} | {b1['fp']} | "
             f"{money(b1['total_cost'])} | {b1['cost_per_customer']:.2f} |\n")

    L.append("## Sensitivity to Cost Ratio\n")
    L.append(f"C_FP fixed at {money(c_fp)}; for each C_FN the threshold is re-selected on validation by "
             "minimum cost, then evaluated once on test.\n")
    L.append("| C_FN | Ratio | Theoretical t | Val-optimal t | Test cost | Cost/customer | Test FN | "
             "Test FP | Test recall | Target-nobody cost | Target-everyone cost |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in o["sensitivity"]:
        L.append(f"| {money(s['c_fn'])} | {s['ratio']:g} | {s['theoretical_threshold']:.3f} | "
                 f"{s['val_threshold']:.2f} | {money(s['test_cost'])} | {s['test_cost_per_customer']:.2f} | "
                 f"{s['test_fn']} | {s['test_fp']} | {s['test_recall']:.4f} | "
                 f"{money(s['test_nobody_cost'])} | {money(s['test_everyone_cost'])} |")
    L.append("")

    L.append("## Observations\n")
    best_name = min(("0.50", "F2", "cost-optimal"),
                    key=lambda k: {"0.50": d50, "F2": f2, "cost-optimal": co}[k]["total_cost"])
    obs = []
    obs.append(f"At C_FN = {money(c_fn)} and C_FP = {money(c_fp)}, the validation cost-optimal threshold is "
               f"{best_t:.2f}. The theoretical Bayes threshold is {o['theoretical_threshold']:.4f}. "
               + ("They differ because scale_pos_weight pushes XGBoost's probabilities upward, so the "
                  "calibrated-probability rule does not apply directly."
                  if abs(best_t - o["theoretical_threshold"]) >= 0.02 else
                  "The two are close despite scale_pos_weight, partly because the cost curve is flat near "
                  "the minimum."))
    obs.append(f"On test, total cost is {money(d50['total_cost'])} at 0.50, {money(f2['total_cost'])} at the "
               f"F2 threshold ({f2['threshold']:.2f}) and {money(co['total_cost'])} at the cost-optimal "
               f"threshold ({co['threshold']:.2f}). The lowest test cost of the three is at the {best_name} "
               "threshold. That comparison uses a single test set of 1,057 customers, so small differences "
               "are within sampling noise.")
    obs.append(f"Every model threshold beats both baselines: target nobody costs {money(b0['total_cost'])} "
               f"and target everyone costs {money(b1['total_cost'])}. The cost-optimal threshold saves "
               f"{money(co['savings_vs_nobody'])} ({co['savings_vs_nobody_pct']:.1%}) vs target nobody.")
    obs.append(f"The cost-optimal threshold trades accuracy for cost: accuracy is {co['accuracy']:.3f} vs "
               f"{d50['accuracy']:.3f} at 0.50, because missing a churner is {c_fn / c_fp:g}x as costly as "
               "a wasted offer. Accuracy is not the objective here.")
    sens = o["sensitivity"]
    obs.append("As C_FN/C_FP rises from "
               f"{sens[0]['ratio']:g} to {sens[-1]['ratio']:g}, the validation-selected threshold moves from "
               f"{sens[0]['val_threshold']:.2f} to {sens[-1]['val_threshold']:.2f} and test recall moves from "
               f"{sens[0]['test_recall']:.3f} to {sens[-1]['test_recall']:.3f}. The threshold choice should "
               "therefore follow from the business's real cost estimates; the defaults used here are "
               "placeholders.")
    near = [r for r in o["val_cost_curve"] if r["cost"] <= o["val_min_cost"] * 1.05]
    ties = [r["threshold"] for r in o["val_cost_curve"] if r["cost"] == o["val_min_cost"]]
    obs.insert(1, f"The validation cost curve is flat near its minimum: {len(near)} of {len(GRID)} grid "
               f"thresholds (from {min(r['threshold'] for r in near):.2f} to "
               f"{max(r['threshold'] for r in near):.2f}) are within 5% of the minimum cost"
               + (f", and thresholds {', '.join(f'{t:.2f}' for t in ties)} tie exactly (lowest kept)"
                  if len(ties) > 1 else "")
               + ". The exact optimum is therefore not sharply identified; any threshold in that band "
               "gives similar expected cost.")
    L += [f"- {x}" for x in obs]
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
