#!/usr/bin/env bash
# Runs the full churn pipeline. All human-readable outputs go to results/.
# Optional cost overrides: ./run_pipeline.sh --c-fn 500 --c-fp 100
set -euo pipefail
cd "$(dirname "$0")"
PY=venv/bin/python
SRC=src/churn
mkdir -p results

$PY $SRC/eda.py > results/eda_report.txt
$PY $SRC/features.py > results/features_report.txt
for m in logistic_regression mlp xgboost; do
    $PY $SRC/models/train_$m.py
done
for m in logistic_regression mlp xgboost; do
    $PY $SRC/cost/cost_$m.py "$@"
done
$PY $SRC/cost/cost_compare.py "$@"
$PY $SRC/reporting/build_all_results.py
