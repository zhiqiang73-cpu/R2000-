@echo off
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"
if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PY_EXE=%ROOT%.venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)
"%PY_EXE%" "%ROOT%main.py"
endlocal
