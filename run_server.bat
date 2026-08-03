@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_server.py
pause
