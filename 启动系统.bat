@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AIGC 鉴伪溯源系统 3.0
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0系统\start.ps1"
if errorlevel 1 (
  echo.
  echo 启动失败，请把上面的错误信息发给维护人员。
  pause
)
exit /b %ERRORLEVEL%
