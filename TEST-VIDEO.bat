@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\windows\video_acceptance.py" %*
  set "code=%ERRORLEVEL%"
  goto :finished
)

if exist "%LOCALAPPDATA%\Strata\terminal\current\python\python.exe" (
  "%LOCALAPPDATA%\Strata\terminal\current\python\python.exe" "tools\windows\video_acceptance.py" %*
  set "code=%ERRORLEVEL%"
  goto :finished
)

py -3 "tools\windows\video_acceptance.py" %*
set "code=%ERRORLEVEL%"

:finished
if "%~1"=="" pause
exit /b %code%
