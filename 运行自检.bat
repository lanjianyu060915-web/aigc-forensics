@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
if not exist "%~dp0系统\python_runtime\python.exe" (
  echo 运行环境还没有搭好，请先双击“搭建环境.bat”。
  pause
  exit /b 1
)
"%~dp0系统\python_runtime\python.exe" -B -X utf8 "%~dp0系统\tools\self_check.py"
echo.
pause
