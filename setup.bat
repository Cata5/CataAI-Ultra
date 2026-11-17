@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo =========================================================================
echo   LLM Maker — Setup & Installation Assistant
echo =========================================================================

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ and re-run.
    exit /b 1
)

REM Check Node.js
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm was not found in PATH. Please install Node.js 18+ and re-run.
    exit /b 1
)

REM Setup Python Virtual Environment
if not exist ".venv" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [1/3] Existing virtual environment detected (.venv).
)

set PYTHON_EXE=.\.venv\Scripts\python.exe
set PIP_EXE=.\.venv\Scripts\pip.exe

echo [2/3] Installing Python dependencies...
"%PIP_EXE%" install --upgrade pip
"%PIP_EXE%" install -r requirements.txt

echo [3/3] Installing Frontend dependencies...
cd frontend
call npm install
cd ..

echo =========================================================================
echo   Setup Complete!
echo   Run "run_studio_v2.bat" to start the workstation.
echo =========================================================================
pause
