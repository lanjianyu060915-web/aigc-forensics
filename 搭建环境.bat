@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AIGC 鉴伪溯源系统 - 搭建运行环境
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0系统\tools\setup_runtime.ps1"
if errorlevel 1 (
  echo.
  echo 搭建失败，请看上面的错误信息。网络不好时可以重新运行，已下载的文件不会重复下载。
) else (
  echo.
  echo 搭建完成。下一步双击“运行自检.bat”确认系统可以正常使用。
)
pause
