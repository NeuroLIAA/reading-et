@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "R_HOME=C:\Program Files\R\R-4.5.3"
set "R_LIBS_USER=%LOCALAPPDATA%\R\win-library\4.5"
set "MPLBACKEND=Agg"

if not exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
  echo Python environment not found: "%REPO_ROOT%\.venv\Scripts\python.exe"
  exit /b 1
)

if not exist "%R_HOME%\bin\x64\Rscript.exe" (
  echo R not found in "%R_HOME%"
  exit /b 1
)

set "PATH=%R_HOME%\bin\x64;%PATH%"
"%REPO_ROOT%\.venv\Scripts\python.exe" "%REPO_ROOT%\em_analysis.py" %*
exit /b %ERRORLEVEL%
