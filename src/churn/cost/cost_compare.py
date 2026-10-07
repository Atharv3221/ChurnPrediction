"""Compare the cost-sensitive results of the three models.

Uses each model's validation-selected cost-optimal threshold (from
artifacts/cost_<model>.json) and its saved val/test probabilities. The model is
chosen by VALIDATION cost; test cost is reported once, with a paired bootstrap
over test customers to show whether the cost differences are meaningful.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402

ART = ARTIFACTS_DIR
MODELS = {"logistic_regression": "Logistic Regression", "mlp": "MLP", "xgboost": "XGBoost"}
N_BOOT = 5000
SEED = 42


def to_markdown(df):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def cost_per_row(y, proba, thr, c_fn, c_fp):
    pred = proba >= thr
    return np.where((y == 1) & ~pred, c_fn, 0) + np.where((y == 0) & pred, c_fp, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-fn", type=float, default=500)
    ap.add_argument("--c-fp", type=float, default=100)
    args = ap.parse_args()

    rows, test_costs = [], {}
    y_test_ref = None
    for key, name in MODELS.items():
        cfg = json.loads((ART / f"cost_{key}.json").read_text())
        if (cfg["c_fn"], cfg["c_fp"]) != (args.c_fn, args.c_fp):
            raise SystemExit(f"cost_{key}.json was built with different costs; rerun cost_{key}.py first")
        thr = cfg["val_optimal_threshold"]
        probs = pd.read_csv(ART / f"{key}_probs.csv")
        val, test = probs[probs.split == "val"], probs[probs.split == "test"]
        if y_test_ref is None:
            y_test_ref = test.y_true.to_numpy()
        assert (test.y_true.to_numpy() == y_test_ref).all(), "test rows are not aligned across models"

        val_cost = cost_per_row(val.y_true.to_numpy(), val.proba.to_numpy(), thr, args.c_fn, args.c_fp)
        tc = cost_per_row(test.y_true.to_numpy(), test.proba.to_numpy(), thr, args.c_fn, args.c_fp)
        test_costs[name] = tc
        pred = test.proba.to_numpy() >= thr
        y = test.y_true.to_numpy()
        rows.append({
            "Model": name, "Threshold": thr, "Val cost": val_cost.sum(),
            "Test cost": tc.sum(), "Test cost/customer": tc.mean(),
            "Test FN": int(((y == 1) & ~pred).sum()), "Test FP": int(((y == 0) & pred).sum()),
            "Test recall": ((y == 1) & pred).sum() / (y == 1).sum(),
        })

    table = pd.DataFrame(rows).sort_values("Val cost")
    chosen = table.iloc[0]["Model"]

    rng = np.random.default_rng(SEED)
    n = len(y_test_ref)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    pairs = []
    for a, b in combinations(test_costs, 2):
        diff = test_costs[a][idx].sum(axis=1) - test_costs[b][idx].sum(axis=1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pairs.append({
            "Comparison": f"{a} - {b}", "Observed diff": test_costs[a].sum() - test_costs[b].sum(),
            "95% CI low": lo, "95% CI high": hi, "Significant": not (lo <= 0 <= hi),
        })
    pairs = pd.DataFrame(pairs)

    nobody = (y_test_ref == 1).sum() * args.c_fn
    everyone = (y_test_ref == 0).sum() * args.c_fp

    fmt = {"Val cost": "{:,.0f}", "Test cost": "{:,.0f}", "Test cost/customer": "{:.2f}",
           "Test recall": "{:.3f}", "Threshold": "{:.2f}"}
    shown = table.copy()
    for c, f in fmt.items():
        shown[c] = shown[c].map(f.format)
    pshown = pairs.copy()
    for c in ["Observed diff", "95% CI low", "95% CI high"]:
        pshown[c] = pshown[c].map("{:,.0f}".format)

    md = f"""# Cost-Sensitive Model Comparison

Costs: C_FN = {args.c_fn:,.0f} (missed churner), C_FP = {args.c_fp:,.0f} (unneeded retention offer). These are placeholders; rerun
the per-model `cost_<model>.py --c-fn N --c-fp M` scripts and then this one with the same flags for real business costs.

Each model's threshold was chosen on the validation set only (minimum validation cost). The model is
**selected by validation cost**; the test set is used once for the final report.

## Results (sorted by validation cost)

{to_markdown(shown)}

Test reference baselines: target nobody = {nobody:,.0f}, target everyone = {everyone:,.0f}.

## Selected model

**{chosen}** at threshold {table.iloc[0]['Threshold']:.2f} (lowest validation cost).
Test cost {table.iloc[0]['Test cost']:,.0f} ({table.iloc[0]['Test cost/customer']:.2f} per customer), a
{1 - table.iloc[0]['Test cost'] / nobody:.1%} saving over targeting nobody.

## Are the differences real? (paired bootstrap on test, {N_BOOT} resamples)

{to_markdown(pshown)}

A difference is significant only if its 95% CI excludes 0.
"""
    (RESULTS_DIR / "cost_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
