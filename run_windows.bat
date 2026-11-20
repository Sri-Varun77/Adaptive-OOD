@echo off
cd /d %~dp0
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_two_models.py
if errorlevel 1 exit /b 1
python app.py
