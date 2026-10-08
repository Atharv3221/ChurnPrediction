@echo off
rem Starts the Streamlit dashboard on Windows (http://localhost:8501).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
venv\Scripts\streamlit.exe run app\streamlit_app.py
