@echo off
rem Runs the full churn pipeline on Windows. All human-readable outputs go to results\.
rem Optional cost overrides: run_pipeline.bat --c-fn 500 --c-fp 100
setlocal
cd /d "%~dp0"
rem Reports contain non-ASCII characters; force UTF-8 instead of the Windows code page.
set PYTHONUTF8=1
set PY=venv\Scripts\python.exe
set SRC=src\churn
if not exist results mkdir results

%PY% %SRC%\eda.py > results\eda_report.txt || exit /b 1
%PY% %SRC%\features.py > results\features_report.txt || exit /b 1
for %%m in (logistic_regression mlp xgboost) do (
    %PY% %SRC%\models\train_%%m.py || exit /b 1
)
for %%m in (logistic_regression mlp xgboost) do (
    %PY% %SRC%\cost\cost_%%m.py %* || exit /b 1
)
%PY% %SRC%\cost\cost_compare.py %* || exit /b 1
%PY% %SRC%\reporting\build_all_results.py || exit /b 1
