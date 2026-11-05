@echo off
setlocal DisableDelayedExpansion
rem Keep the caller's working directory so relative video and output paths resolve as typed.
if not exist "%~dp0runtime\python.exe" (
  echo RGB MoCap is not installed. Run install.cmd first. 1>&2
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -u -B "%~dp0cli.py" %*
exit /b %errorlevel%
