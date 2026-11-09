@echo off
setlocal
chcp 65001 >nul
if "%~1"=="" (
  echo Usage: run_case.bat QUERY_FILE [IDENTITY_LIBRARY] [SOURCE_LIBRARY]
  exit /b 2
)
set "PRODUCT_ROOT=%~dp0"
set "PYTHON_CMD=%PRODUCT_ROOT%python_runtime\python.exe"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
if not defined FSTS_DATA_ROOT set "FSTS_DATA_ROOT=%PRODUCT_ROOT%..\用户数据"
if "%~2"=="" (
  "%PYTHON_CMD%" -B -X utf8 "%PRODUCT_ROOT%launcher.py" --query "%~1"
) else if "%~3"=="" (
  "%PYTHON_CMD%" -B -X utf8 "%PRODUCT_ROOT%launcher.py" --query "%~1" --identity-library "%~2"
) else (
  "%PYTHON_CMD%" -B -X utf8 "%PRODUCT_ROOT%launcher.py" --query "%~1" --identity-library "%~2" --source-library "%~3"
)
exit /b %ERRORLEVEL%
