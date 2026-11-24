@echo off
setlocal
cd /d "%~dp0"

set "CHECK_MODE="
if not "%~1"=="" for %%A in (%*) do if /I "%%A"=="-Check" set "CHECK_MODE=1"

echo ===========================================================================
echo   LLM Maker - Setup ^& Installation Assistant
echo ===========================================================================

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js was not found in PATH. Install Node.js 20+ and re-run.
    pause
    exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found in PATH. Reinstall Node.js 20+ and re-run.
    pause
    exit /b 1
)
where cargo >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Rust/cargo was not found in PATH. Install Rust from https://rustup.rs and re-run.
    pause
    exit /b 1
)

REM Hardware detection and PyTorch selection live in scripts\bootstrap.ps1.
REM Forward -Check, -CpuOnly, -CudaTag, -SkipFrontend and -SkipRust.
set "POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL%" set "POWERSHELL=powershell"
"%POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -File "scripts\bootstrap.ps1" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. Review the message above, then re-run.
    pause
    exit /b 1
)

echo.
if defined CHECK_MODE (
    echo Prerequisite check complete.
) else (
    echo Setup complete. Launch the workstation with run_desktop.bat.
)
pause
