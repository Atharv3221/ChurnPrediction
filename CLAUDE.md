# Customer Churn Prediction: project guide

## Goal
Cost-sensitive, explainable churn prediction on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn).
Decision thresholds minimise expected business cost (missed churners vs unneeded retention offers), and SHAP
explanations are planned for the selected model.

## Folder map
```
data/raw/data.csv            original IBM Telco CSV
data/interim/data_clean.csv  cleaned data (written by eda.py)
data/processed/              X_{train,val,test}_{raw,std,norm}.csv, y_{train,val,test}.csv (written by features.py)
src/churn/paths.py           the ONLY place paths are defined (DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR, ...)
src/churn/eda.py             missing values + EDA plots
src/churn/features.py        feature engineering, stratified split, scaling
src/churn/models/            train_logistic_regression.py, train_mlp.py, train_xgboost.py
src/churn/cost/              cost_<model>.py (per-model cost analysis), cost_compare.py (selection + bootstrap)
src/churn/reporting/         build_all_results.py (writes results/all-result.md)
artifacts/                   model binaries, *_probs.csv (split,y_true,proba), cost_*.json, xgboost_metrics.json
results/                     every human-readable output: *.md, *.txt, plots/, README.md, all-result.md
app/streamlit_app.py         read-only dashboard over artifacts/ and results/
run_pipeline.sh              runs every stage in order (run_pipeline.bat / run_app.bat on Windows)
```

## How to run
```bash
./run_pipeline.sh                          # full pipeline with placeholder costs
./run_pipeline.sh --c-fn 800 --c-fp 50     # with other costs
venv/bin/streamlit run app/streamlit_app.py
```
Each script also runs on its own from the project root, e.g. `venv/bin/python src/churn/cost/cost_mlp.py --c-fn 500 --c-fp 100`.
Use `venv/bin/python`. `requirements.txt` lists direct dependencies only (pinned); add a new direct import there by hand,
never `pip freeze` (it pins platform-specific packages such as nvidia-nccl that break Windows installs).
Keep `run_pipeline.sh` and `run_pipeline.bat` in sync when stages change.

## Rules
- Split is stratified 70/15/15, `random_state=42`. Validation is used for ALL tuning and threshold choice;
  the test set is scored once and never used for selection.
- C_FN=500 and C_FP=100 are PLACEHOLDERS, always passed as flags (`--c-fn`, `--c-fp`), never hard-coded.
- Selected model: Logistic Regression, threshold 0.45 (lowest validation cost). The three models are
  statistically tied (paired bootstrap).
- Stages 5 (real costs) and 6 (SHAP) are still to do.
- New paths go in `src/churn/paths.py`; scripts import from there, never hard-code or use relative paths.
- `results/all-result.md` is generated: don't edit it by hand. New reports must be added to `build_all_results.py`.
- The app is read-only: it never retrains, it recomputes costs from the saved probabilities.
