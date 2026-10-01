@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Create the Python 3.11 environment first. See README.md.
  pause
  exit /b 1
)
start "Vantara API" cmd /k ".venv\Scripts\python.exe -m uvicorn api.main:app --port 8000"
start "Vantara Dashboard" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend/dashboard.py"
