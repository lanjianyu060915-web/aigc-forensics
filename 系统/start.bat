@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PRODUCT_ROOT=%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PRODUCT_ROOT%start.ps1"
exit /b %ERRORLEVEL%
