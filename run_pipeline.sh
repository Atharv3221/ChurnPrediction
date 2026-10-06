#!/usr/bin/env bash
# Runs the full churn pipeline. All human-readable outputs go to results/.
# Optional cost overrides: ./run_pipeline.sh --c-fn 500 --c-fp 100
set -euo pipefail
cd "$(dirname "$0")"
PY=venv/bin/python
mkdir -p results

$PY eda.py > results/eda_report.txt
$PY features.py > results/features_report.txt
for m in logistic_regression mlp xgboost; do
    $PY train_$m.py
done
for m in logistic_regression mlp xgboost; do
    $PY cost_$m.py "$@"
done
$PY cost_compare.py "$@"
$PY build_all_results.py
