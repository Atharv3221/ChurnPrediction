# Cost-Sensitive Churn Prediction

Churn prediction on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn). Three imbalance-aware
models (Logistic Regression, MLP, XGBoost) are trained on a stratified 70/15/15 split. The decision threshold
is then chosen to minimise expected cost on the validation set:

**cost = missed churners × C_FN + unneeded offers × C_FP**

With the placeholder costs C_FN = 500 and C_FP = 100, Logistic Regression at threshold 0.45 has the lowest validation
cost. Its test cost is 44,600, against 140,000 for contacting nobody. The three models are statistically tied.
Full results and progress are in [`results/README.md`](results/README.md), and every report is collected in
[`results/all-result.md`](results/all-result.md).

## Project structure

```
data/
  raw/data.csv                 original dataset
  interim/data_clean.csv       cleaned data
  processed/                   model-ready features and labels per split
src/churn/
  paths.py                     all project paths (pathlib), imported by every script
  eda.py                       stage 1: missing values + EDA
  features.py                  stage 2: features, split, scaling
  models/                      stage 3: train_logistic_regression.py, train_mlp.py, train_xgboost.py
  cost/                        stage 4: cost_<model>.py, cost_compare.py
  reporting/build_all_results.py
artifacts/                     models, saved probabilities, cost/metric JSON
results/                       reports (*.md, *.txt), plots/, all-result.md
app/streamlit_app.py           dashboard
run_pipeline.sh                full pipeline (Linux/macOS)
run_pipeline.bat               full pipeline (Windows)
run_app.bat                    start the dashboard (Windows)
```

## Setup

Requires Python 3.12. `requirements.txt` lists only the direct dependencies; pip installs the rest for your platform.

Linux / macOS:
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Windows (Command Prompt or PowerShell):
```bat
py -3.12 -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Run the pipeline

```bash
./run_pipeline.sh                          # placeholder costs (C_FN=500, C_FP=100)
./run_pipeline.sh --c-fn 800 --c-fp 50     # real business costs
```

On Windows, use the batch file with the same flags (in PowerShell, prefix it with `.\`):

```bat
run_pipeline.bat
run_pipeline.bat --c-fn 800 --c-fp 50
```

Individual stages run from the project root, with the same flags:

```bash
venv/bin/python src/churn/cost/cost_logistic_regression.py --c-fn 500 --c-fp 100
venv/bin/python src/churn/cost/cost_compare.py --c-fn 500 --c-fp 100
```

On Windows, use `venv\Scripts\python` instead, and set `PYTHONUTF8=1` first (`set PYTHONUTF8=1` in Command
Prompt, `$env:PYTHONUTF8=1` in PowerShell) so the reports, which contain non-ASCII characters, are written as UTF-8.
The batch files already do this.

## Start the app

```bash
venv/bin/streamlit run app/streamlit_app.py      # Linux / macOS
run_app.bat                                      # Windows
```

Then open http://localhost:8501. The app is read-only. It loads the saved artifacts and results, and recomputes
costs live from `artifacts/*_probs.csv`, so it needs no retraining. Its pages are:
- Overview
- Model comparison
- Cost explorer (sliders for C_FN, C_FP and the threshold)
- Sensitivity
- EDA
- Reports
- Logistic Regression coefficients

## Status

Stages 1–4 (EDA, features, models, cost-sensitive thresholds) are done. Still to do: stage 5, real business
costs, and stage 6, SHAP explanations.
