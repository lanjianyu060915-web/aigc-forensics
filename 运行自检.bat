@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AIGC 鉴伪溯源系统 - 一键自检
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
"%~dp0系统\python_runtime\python.exe" -B -X utf8 "%~dp0系统\tools\self_check.py"
echo.
pause
